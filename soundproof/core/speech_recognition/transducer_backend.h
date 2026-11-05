#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BACKEND_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BACKEND_H_

#include <memory>
#include <string>

#include "model/online_transducer_model.h"
#include "core/speech_recognition/asr_backend.h"

namespace soundproof {

class OnlineRnnLm;
class TransducerBeamSearchDecoder;
class TransducerDecoder;

// Transducer backend (zipformer / zipformer2 / conformer / lstm /
// ebranchformer): encoder + stateless decoder + joiner, greedy search.
class TransducerBackend : public AsrBackend {
 public:
  TransducerBackend();
  ~TransducerBackend() override;

  bool Init(const AsrBackendConfig& config) override;

  const char* ModelType() const override { return model_type_.c_str(); }
  int32_t FeatureDim() const override { return model_.FeatureDim(); }
  bool NormalizeSamples() const override { return true; }
  int32_t ChunkSize() const override { return model_.ChunkSize(); }
  int32_t ChunkShift() const override { return model_.ChunkShift(); }

  bool DecodeChunk(const float* features, int32_t processed_frames,
                   AsrResult* result) override;
  std::string TextFromTokens(const std::vector<int32_t>& tokens,
                             const SymbolTable& symbols) const override;

  void ResetSegment() override;
  void Reset() override;
  bool SetBlankPenalty(float blank_penalty) override;

 private:
  OnlineTransducerModel model_;
  std::unique_ptr<TransducerDecoder> decoder_;
  std::unique_ptr<OnlineRnnLm> lm_;
  std::unique_ptr<TransducerBeamSearchDecoder> beam_decoder_;
  bool beam_mode_ = false;
  std::string model_type_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BACKEND_H_
