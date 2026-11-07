#ifndef SOUNDPROOF_CORE_TURN_DETECTION_TURN_BACKEND_H_
#define SOUNDPROOF_CORE_TURN_DETECTION_TURN_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>

namespace soundproof {

enum class TurnStatus {
  kOk = 0,
  kNotInitialized,
  kModelLoadFailed,
  kUnsupportedModel,
  kInvalidInput,
  kFeatureFailed,
  kInferenceFailed,
};

struct TurnModelConfig {
  // Empty means "detect from model path" (v1: smart-turn).
  std::string model_type;
  std::string model_path;
  int num_threads = 1;
};

// A turn-detection model backend. Implementations own the model and its
// audio front-end; the detector owns the rolling audio buffer.
class TurnBackend {
 public:
  virtual ~TurnBackend() = default;

  virtual TurnStatus Init(const TurnModelConfig& config) = 0;

  virtual const char* ModelType() const = 0;

  // Number of samples the model consumes (e.g. 8 s at 16 kHz).
  virtual int32_t ContextSamples() const = 0;

  // |audio| holds ContextSamples() samples in [-1, 1], already left-padded
  // by the caller. Writes a probability in [0, 1].
  virtual TurnStatus Predict(const float* audio, int samples,
                             float* probability) = 0;
};

// Creates a backend from config.model_type (empty -> smart-turn).
std::unique_ptr<TurnBackend> CreateTurnBackend(
    const TurnModelConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TURN_DETECTION_TURN_BACKEND_H_
