#ifndef SOUNDPROOF_CORE_TURN_DETECTION_WHISPER_MEL_H_
#define SOUNDPROOF_CORE_TURN_DETECTION_WHISPER_MEL_H_

#include <memory>
#include <vector>

namespace soundproof {

// Whisper log-mel front-end used by Smart Turn v3.2.
//
// 16 kHz mono, fixed 8 s window (128000 samples) -> 80 x 800 features. The
// implementation matches transformers' WhisperFeatureExtractor with
// chunk_length=8 and do_normalize=True:
//   1. zero-mean / unit-variance normalization of the whole window;
//   2. STFT with n_fft=400, hop=160, periodic Hann, reflect padding, then the
//      last of the 801 frames is dropped;
//   3. power spectrum -> Slaney mel filterbank (80 x 201, 0-8 kHz);
//   4. log10(clamp(mel, 1e-10)) -> max(x, max - 8) -> (x + 4) / 4.
class WhisperMel {
 public:
  static constexpr int kSampleRate = 16000;
  static constexpr int kContextSamples = 8 * kSampleRate;  // 128000
  static constexpr int kNfft = 400;
  static constexpr int kHopLength = 160;
  static constexpr int kNumMels = 80;
  static constexpr int kNumFrames = 800;
  static constexpr int kNumFreqBins = kNfft / 2 + 1;  // 201

  WhisperMel();
  ~WhisperMel();

  WhisperMel(const WhisperMel&) = delete;
  WhisperMel& operator=(const WhisperMel&) = delete;

  bool ok() const;

  // |waveform| must hold kContextSamples samples in [-1, 1]; the caller pads
  // short audio with zeros on the left. Writes kNumMels * kNumFrames values
  // in [mel][frame] order.
  bool Compute(const float* waveform, std::vector<float>* features);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TURN_DETECTION_WHISPER_MEL_H_
