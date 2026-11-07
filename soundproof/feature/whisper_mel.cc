#include "feature/whisper_mel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "dsp/pffft_wrapper.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kMelFloor = 1e-10f;
constexpr float kLogRange = 8.0f;
constexpr int kHalfWindow = WhisperMel::kNfft / 2;  // 200

// Slaney mel scale, as used by transformers.audio_utils.hertz_to_mel and
// mel_to_hertz (mel_scale="slaney").
double HertzToMelSlaney(double freq) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  constexpr double kLogStep = 27.0 / std::log(6.4);
  if (freq >= kMinLogHz) {
    return kMinLogMel + std::log(freq / kMinLogHz) * kLogStep;
  }
  return 3.0 * freq / 200.0;
}

double MelToHertzSlaney(double mel) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  constexpr double kLogStep = std::log(6.4) / 27.0;
  if (mel >= kMinLogMel) {
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
  }
  return 200.0 * mel / 3.0;
}

using FilterTable =
    std::array<float, WhisperMel::kNumMels * WhisperMel::kNumFreqBins>;

// 80 x 201 Slaney-normalized mel filters, computed once on first use.
const FilterTable& MelFilters() {
  static const FilterTable table = [] {
    FilterTable filters{};
    constexpr int kFreqBins = WhisperMel::kNumFreqBins;
    constexpr int kMels = WhisperMel::kNumMels;

    // Center frequencies of the triangular filters (kMels + 2 points).
    const double mel_min = HertzToMelSlaney(0.0);
    const double mel_max = HertzToMelSlaney(8000.0);
    std::array<double, kMels + 2> filter_freqs{};
    for (int i = 0; i < kMels + 2; ++i) {
      const double mel =
          mel_min + (mel_max - mel_min) * i / static_cast<double>(kMels + 1);
      filter_freqs[i] = MelToHertzSlaney(mel);
    }

    // FFT bin frequencies: linspace(0, sampling_rate / 2, 201).
    const double bin_width =
        (WhisperMel::kSampleRate / 2.0) / static_cast<double>(kFreqBins - 1);
    for (int m = 0; m < kMels; ++m) {
      const double left = filter_freqs[m];
      const double center = filter_freqs[m + 1];
      const double right = filter_freqs[m + 2];
      const double enorm = 2.0 / (right - left);
      for (int k = 0; k < kFreqBins; ++k) {
        const double freq = bin_width * k;
        const double down = (freq - left) / (center - left);
        const double up = (right - freq) / (right - center);
        filters[m * kFreqBins + k] =
            static_cast<float>(std::max(0.0, std::min(down, up)) * enorm);
      }
    }
    return filters;
  }();
  return table;
}

}  // namespace

struct WhisperMel::Impl {
  std::unique_ptr<webrtc::Pffft> fft;
  std::unique_ptr<webrtc::Pffft::FloatBuffer> fft_input;
  std::unique_ptr<webrtc::Pffft::FloatBuffer> fft_output;
  std::array<float, kNfft> window{};
  std::vector<float> padded;  // kContextSamples + kNfft (reflect padding)
  bool valid = false;

  Impl() {
    if (!webrtc::Pffft::IsValidFftSize(kNfft,
                                       webrtc::Pffft::FftType::kComplex)) {
      RTC_LOG(LS_ERROR) << "WhisperMel: unsupported FFT size " << kNfft;
      return;
    }
    fft = std::make_unique<webrtc::Pffft>(kNfft,
                                          webrtc::Pffft::FftType::kComplex);
    fft_input = fft->CreateBuffer();
    fft_output = fft->CreateBuffer();
    if (!fft_input || !fft_output) {
      RTC_LOG(LS_ERROR) << "WhisperMel: failed to allocate FFT buffers";
      return;
    }

    // Period window: 0.5 - 0.5*cos(2*pi*n/N), n = 0..N-1. This matches
    // torch.hann_window(N) and transformers' window_function(..., "hann").
    for (int i = 0; i < kNfft; ++i) {
      window[i] = static_cast<float>(
          0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / kNfft));
    }
    padded.resize(kContextSamples + kNfft);
    valid = true;
  }
};

WhisperMel::WhisperMel() : impl_(new Impl()) {}
WhisperMel::~WhisperMel() = default;

bool WhisperMel::ok() const {
  return impl_ != nullptr && impl_->valid;
}

bool WhisperMel::Compute(const float* waveform,
                         std::vector<float>* features) {
  if (!ok() || waveform == nullptr || features == nullptr) {
    RTC_LOG(LS_ERROR) << "WhisperMel: not ready or null argument";
    return false;
  }

  // Zero-mean / unit-variance normalization over the whole 8 s window
  // (population variance, eps = 1e-7), matching do_normalize=True.
  constexpr int kSamples = kContextSamples;
  double sum = 0.0;
  for (int i = 0; i < kSamples; ++i) {
    sum += waveform[i];
  }
  const double mean = sum / kSamples;
  double variance_sum = 0.0;
  for (int i = 0; i < kSamples; ++i) {
    const double d = static_cast<double>(waveform[i]) - mean;
    variance_sum += d * d;
  }
  const double variance = variance_sum / kSamples;
  const float inv_std = static_cast<float>(1.0 / std::sqrt(variance + 1e-7));

  float* padded = impl_->padded.data();
  float* center = padded + kHalfWindow;
  for (int i = 0; i < kSamples; ++i) {
    center[i] = static_cast<float>(
        (static_cast<double>(waveform[i]) - mean) * inv_std);
  }
  // np.pad(..., mode="reflect"): left[j] = x[200 - j], right[k] = x[N - 2 - k].
  for (int j = 0; j < kHalfWindow; ++j) {
    padded[j] = center[kHalfWindow - j];
  }
  for (int k = 0; k < kHalfWindow; ++k) {
    padded[kHalfWindow + kSamples + k] = center[kSamples - 2 - k];
  }

  features->assign(kNumMels * kNumFrames, 0.0f);
  float* mel_out = features->data();
  const FilterTable& filters = MelFilters();

  auto in = impl_->fft_input->GetView();
  auto out = impl_->fft_output->GetView();
  for (int frame = 0; frame < kNumFrames; ++frame) {
    const int start = frame * kHopLength;
    for (int i = 0; i < kNfft; ++i) {
      in[2 * i] = padded[start + i] * impl_->window[i];
      in[2 * i + 1] = 0.0f;
    }
    impl_->fft->ForwardTransform(*impl_->fft_input, impl_->fft_output.get(),
                                 /*ordered=*/true);

    for (int m = 0; m < kNumMels; ++m) {
      const float* filter = filters.data() + m * kNumFreqBins;
      float energy = 0.0f;
      for (int k = 0; k < kNumFreqBins; ++k) {
        const float re = out[2 * k];
        const float im = out[2 * k + 1];
        energy += filter[k] * (re * re + im * im);
      }
      mel_out[m * kNumFrames + frame] = energy;
    }
  }

  // log10(clamp(mel, 1e-10)), then dynamic-range clamp and scaling.
  float max_log = -std::numeric_limits<float>::infinity();
  for (float& value : *features) {
    value = std::log10(std::max(value, kMelFloor));
    max_log = std::max(max_log, value);
  }
  const float floor_log = max_log - kLogRange;
  for (float& value : *features) {
    value = (std::max(value, floor_log) + 4.0f) / 4.0f;
  }
  return true;
}

}  // namespace soundproof
