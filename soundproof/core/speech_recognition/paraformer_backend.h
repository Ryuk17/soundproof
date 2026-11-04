#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_BACKEND_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_BACKEND_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/speech_recognition/asr_backend.h"
#include "core/speech_recognition/paraformer_model.h"

namespace soundproof {

// Streaming paraformer backend: fbank -> LFR/CMVN/positional encoding ->
// encoder -> CIF -> decoder. 61 feature frames per step (shift 60), with
// 5/3 frames of left/right context cached between steps.
class ParaformerBackend : public AsrBackend {
 public:
  static constexpr int32_t kRawFeatureDim = 80;
  static constexpr int32_t kChunkSize = 61;
  static constexpr int32_t kLeftChunkSize = 5;
  static constexpr int32_t kRightChunkSize = 3;

  ParaformerBackend() = default;
  ~ParaformerBackend() override = default;

  bool Init(const AsrBackendConfig& config) override;

  const char* ModelType() const override { return "paraformer"; }
  int32_t FeatureDim() const override { return kRawFeatureDim; }
  bool NormalizeSamples() const override { return false; }
  int32_t ChunkSize() const override { return kChunkSize; }
  int32_t ChunkShift() const override { return kChunkSize - 1; }

  bool DecodeChunk(const float* features, int32_t processed_frames,
                   AsrResult* result) override;
  std::string TextFromTokens(const std::vector<int32_t>& tokens,
                             const SymbolTable& symbols) const override;

  void ResetSegment() override;
  void Reset() override;
  bool SetBlankPenalty(float blank_penalty) override;

 private:
  ParaformerModel model_;
  std::vector<float> feature_cache_;
  std::vector<float> encoder_out_cache_;
  std::vector<float> alpha_cache_;
  std::vector<int32_t> tokens_;
  int32_t last_non_blank_frame_index_ = 0;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_BACKEND_H_
