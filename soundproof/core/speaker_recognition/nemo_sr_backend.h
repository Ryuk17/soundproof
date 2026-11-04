#ifndef SOUNDPROOF_CORE_SPEAKER_RECOGNITION_NEMO_SR_BACKEND_H_
#define SOUNDPROOF_CORE_SPEAKER_RECOGNITION_NEMO_SR_BACKEND_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/speaker_recognition/sr_backend.h"
#include "mnn_engine/mnn_engine.h"

namespace soundproof {

// NeMo speaker-embedding backend: configurable fbank (hann window, librosa
// mel banks, snip_edges) -> per-feature normalization -> transposed input
// [1, feat_dim, T] plus x_lens.
class NemoSrBackend : public SrBackend {
 public:
  NemoSrBackend() = default;
  ~NemoSrBackend() override = default;

  SrStatus Init(const SrModelConfig& config) override;

  const char* ModelType() const override { return "nemo"; }
  int32_t EmbeddingDim() const override { return output_dim_; }
  int32_t SampleRate() const override { return feature_config_.sample_rate; }
  int32_t FeatureDim() const override { return feature_config_.feature_dim; }
  bool NormalizeSamples() const override { return true; }
  FeatureExtractorConfig FeatureConfig() const override {
    return feature_config_;
  }
  int32_t MinFrames() const override { return min_frames_; }

  SrStatus ComputeEmbedding(const float* features, int32_t num_frames,
                            int32_t dim,
                            std::vector<float>* embedding) override;

 private:
  mnn_engine::Engine engine_;
  mnn_engine::Tensor input_;
  mnn_engine::Tensor length_;
  FeatureExtractorConfig feature_config_;
  int32_t output_dim_ = 0;
  int32_t min_frames_ = 20;
  bool per_feature_normalize_ = false;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEAKER_RECOGNITION_NEMO_SR_BACKEND_H_
