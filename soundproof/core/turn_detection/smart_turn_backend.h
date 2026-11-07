#ifndef SOUNDPROOF_CORE_TURN_DETECTION_SMART_TURN_BACKEND_H_
#define SOUNDPROOF_CORE_TURN_DETECTION_SMART_TURN_BACKEND_H_

#include <cstdint>
#include <memory>

#include "core/turn_detection/turn_backend.h"

namespace soundproof {

// Smart Turn v3.2 end-of-turn classifier.
//
// The model consumes 8 s of 16 kHz mono audio as 80 x 800 Whisper log-mel
// features and returns a sigmoid probability. The Whisper front-end lives in
// whisper_mel.h; this class loads the MNN model and runs the forward pass.
class SmartTurnBackend : public TurnBackend {
 public:
  SmartTurnBackend();
  ~SmartTurnBackend() override;

  TurnStatus Init(const TurnModelConfig& config) override;

  const char* ModelType() const override { return "smart-turn"; }
  int32_t ContextSamples() const override { return context_samples_; }

  TurnStatus Predict(const float* audio, int samples,
                     float* probability) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int32_t context_samples_ = 0;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TURN_DETECTION_SMART_TURN_BACKEND_H_
