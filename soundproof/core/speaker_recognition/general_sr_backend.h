#ifndef SOUNDPROOF_CORE_SPEAKER_RECOGNITION_GENERAL_SR_BACKEND_H_
#define SOUNDPROOF_CORE_SPEAKER_RECOGNITION_GENERAL_SR_BACKEND_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/speaker_recognition/sr_backend.h"
#include "mnn_engine/mnn_engine.h"

namespace soundproof {

// WeSpeaker / 3D-Speaker backend: shared fbank -> optional global-mean
// normalization -> model -> embedding.
class GeneralSrBackend : public SrBackend {
 public:
  GeneralSrBackend() = default;
  ~GeneralSrBackend() override = default;

  SrStatus Init(const SrModelConfig& config) override;

  const char* ModelType() const override { return model_type_.c_str(); }
  int32_t EmbeddingDim() const override { return output_dim_; }
  int32_t SampleRate() const override { return feature_config_.sample_rate; }
  int32_t FeatureDim() const override { return feature_config_.feature_dim; }
  bool NormalizeSamples() const override { return normalize_samples_; }
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
  FeatureExtractorConfig feature_config_;
  std::string model_type_;
  int32_t output_dim_ = 0;
  int32_t min_frames_ = 50;
  bool normalize_samples_ = true;
  bool global_mean_ = false;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEAKER_RECOGNITION_GENERAL_SR_BACKEND_H_
