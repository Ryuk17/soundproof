#include "feature/feature_extractor.h"

#include <algorithm>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

knf::FbankOptions MakeFbankOptions(const FeatureExtractorConfig& config) {
  knf::FbankOptions opts;
  opts.frame_opts.dither = config.dither;
  opts.frame_opts.snip_edges = config.snip_edges;
  opts.frame_opts.samp_freq = static_cast<float>(config.sample_rate);
  opts.frame_opts.frame_shift_ms = config.frame_shift_ms;
  opts.frame_opts.frame_length_ms = config.frame_length_ms;
  opts.frame_opts.remove_dc_offset = config.remove_dc_offset;
  opts.frame_opts.preemph_coeff = config.preemph_coeff;
  opts.frame_opts.window_type = config.window_type;

  opts.mel_opts.num_bins = config.feature_dim;
  opts.mel_opts.low_freq = config.low_freq;
  opts.mel_opts.high_freq = config.high_freq;
  opts.mel_opts.is_librosa = config.is_librosa;
  return opts;
}

}  // namespace

struct FeatureExtractor::Impl {
  explicit Impl(const FeatureExtractorConfig& config)
      : fbank(new knf::OnlineFbank(MakeFbankOptions(config))) {}

  std::unique_ptr<knf::OnlineFbank> fbank;
  int32_t last_frame_index = 0;
};

FeatureExtractor::FeatureExtractor(const FeatureExtractorConfig& config)
    : impl_(new Impl(config)), config_(config) {}
FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::AcceptWaveform(const float* samples, int32_t n) {
  if (samples == nullptr || n <= 0) {
    return;
  }
  impl_->fbank->AcceptWaveform(static_cast<float>(config_.sample_rate), samples,
                               n);
}

void FeatureExtractor::InputFinished() {
  impl_->fbank->InputFinished();
}

int32_t FeatureExtractor::NumFramesReady() const {
  return impl_->fbank->NumFramesReady();
}

bool FeatureExtractor::GetFrames(int32_t frame_index, int32_t n,
                                 std::vector<float>* frames) {
  if (frames == nullptr || frame_index < 0 || n < 0) {
    return false;
  }
  const int32_t discard = frame_index - impl_->last_frame_index;
  if (discard < 0) {
    RTC_LOG(LS_ERROR) << "FeatureExtractor: frames already discarded ("
                      << impl_->last_frame_index << " > " << frame_index
                      << ")";
    return false;
  }
  if (frame_index + n > impl_->fbank->NumFramesReady()) {
    RTC_LOG(LS_ERROR) << "FeatureExtractor: requested " << n
                      << " frames from " << frame_index << ", only "
                      << impl_->fbank->NumFramesReady() << " ready";
    return false;
  }

  impl_->fbank->Pop(discard);
  const int32_t dim = impl_->fbank->Dim();
  frames->resize(static_cast<size_t>(dim) * static_cast<size_t>(n));
  float* dst = frames->data();
  for (int32_t i = 0; i < n; ++i) {
    const float* src = impl_->fbank->GetFrame(frame_index + i);
    std::copy(src, src + dim, dst);
    dst += dim;
  }
  impl_->last_frame_index = frame_index;
  return true;
}

int32_t FeatureExtractor::FeatureDim() const {
  return impl_->fbank->Dim();
}

void FeatureExtractor::Reset() {
  impl_->fbank.reset(new knf::OnlineFbank(MakeFbankOptions(config_)));
  impl_->last_frame_index = 0;
}

}  // namespace soundproof
