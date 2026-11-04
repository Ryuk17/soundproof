#ifndef SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SR_BACKEND_H_
#define SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SR_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "feature/feature_extractor.h"

namespace soundproof {

enum class SrStatus {
  kOk = 0,
  kNotInitialized,
  kModelLoadFailed,
  kUnsupportedModel,
  kAudioTooShort,
  kFeatureFailed,
  kInferenceFailed,
  kSpeakerExists,
  kSpeakerNotFound,
  kEmbeddingDimMismatch,
};

struct SrModelConfig {
  // Empty means "detect from the model's framework metadata".
  std::string model_type;
  std::string model_path;
  int num_threads = 1;
};

// A speaker-embedding model backend. The recognizer owns feature extraction
// (using FeatureConfig()) and passes fbank features to ComputeEmbedding().
class SrBackend {
 public:
  virtual ~SrBackend() = default;

  virtual SrStatus Init(const SrModelConfig& config) = 0;
  virtual const char* ModelType() const = 0;

  virtual int32_t EmbeddingDim() const = 0;
  virtual int32_t SampleRate() const = 0;
  virtual int32_t FeatureDim() const = 0;
  // True if int16 samples must be scaled to [-1, 1] before feature extraction.
  virtual bool NormalizeSamples() const = 0;
  // fbank configuration expected by this model.
  virtual FeatureExtractorConfig FeatureConfig() const = 0;
  // Minimum number of feature frames needed to produce a valid embedding.
  virtual int32_t MinFrames() const = 0;

  // |features| is row-major, num_frames * dim floats, dim == FeatureDim().
  virtual SrStatus ComputeEmbedding(const float* features, int32_t num_frames,
                                    int32_t dim,
                                    std::vector<float>* embedding) = 0;
};

// Creates a backend from config.model_type when set; otherwise the model's
// "framework" metadata is probed (this loads the model once more).
std::unique_ptr<SrBackend> CreateSrBackend(const SrModelConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SR_BACKEND_H_
