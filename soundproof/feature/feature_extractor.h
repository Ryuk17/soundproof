#ifndef SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_
#define SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {

// Configuration of the kaldi-style fbank feature extractor.
// The defaults match sherpa-mnn's FeatureExtractor and are shared by the
// KWS / ASR / speaker-recognition algorithms.
struct FeatureExtractorConfig {
  int32_t sample_rate = 16000;
  int32_t feature_dim = 80;
  float frame_length_ms = 25.0f;
  float frame_shift_ms = 10.0f;
  bool snip_edges = false;
  float low_freq = 20.0f;
  float high_freq = -400.0f;  // <= 0: offset from Nyquist
  float dither = 0.0f;
  bool remove_dc_offset = true;
  float preemph_coeff = 0.97f;
  std::string window_type = "povey";
  bool is_librosa = false;
};

// Streaming log-mel filter bank extractor.
// Input samples must be normalized to [-1, 1] (the caller decides the int16
// scaling based on the model metadata).
class FeatureExtractor {
 public:
  static constexpr int32_t kSampleRate = 16000;
  static constexpr int32_t kFeatureDim = 80;

  explicit FeatureExtractor(const FeatureExtractorConfig& config = {});
  ~FeatureExtractor();

  FeatureExtractor(const FeatureExtractor&) = delete;
  FeatureExtractor& operator=(const FeatureExtractor&) = delete;

  void AcceptWaveform(const float* samples, int32_t n);

  // Flushes the last (incomplete) frame. Used for offline/one-shot feature
  // extraction. Streaming callers do not need to call it.
  void InputFinished();

  int32_t NumFramesReady() const;

  // Copies |n| frames starting at |frame_index| into |frames| (row-major,
  // |n| * FeatureDim() values). Frames before |frame_index| are discarded.
  // Returns false if the requested frames are not available yet.
  bool GetFrames(int32_t frame_index, int32_t n, std::vector<float>* frames);

  int32_t FeatureDim() const;

  const FeatureExtractorConfig& config() const { return config_; }

  // Drops all buffered audio and features.
  void Reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  FeatureExtractorConfig config_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_
