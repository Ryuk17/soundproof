#ifndef SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_
#define SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace soundproof {

// 80-dim log-mel filter bank features for the zipformer2 KWS model.
//
// Configuration matches sherpa-mnn's FeatureExtractor defaults:
//   sampling rate 16 kHz, 25 ms window / 10 ms shift, snip_edges=false,
//   povey window, pre-emphasis 0.97, remove DC, mel bins 20..7600 Hz,
//   dither disabled. Input samples must be normalized to [-1, 1].
class FeatureExtractor {
 public:
  static constexpr int32_t kSampleRate = 16000;
  static constexpr int32_t kFeatureDim = 80;

  FeatureExtractor();
  ~FeatureExtractor();

  FeatureExtractor(const FeatureExtractor&) = delete;
  FeatureExtractor& operator=(const FeatureExtractor&) = delete;

  void AcceptWaveform(const float* samples, int32_t n);

  int32_t NumFramesReady() const;

  // Copies |n| frames starting at |frame_index| into |frames| (row-major,
  // |n| * FeatureDim() values). Frames before |frame_index| are discarded.
  // Returns false if the requested frames are not available yet.
  bool GetFrames(int32_t frame_index, int32_t n, std::vector<float>* frames);

  int32_t FeatureDim() const;

  // Drops all buffered audio and features.
  void Reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_FEATURE_FEATURE_EXTRACTOR_H_
