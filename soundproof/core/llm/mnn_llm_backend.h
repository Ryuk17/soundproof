#ifndef SOUNDPROOF_CORE_LLM_MNN_LLM_BACKEND_H_
#define SOUNDPROOF_CORE_LLM_MNN_LLM_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>

#include "core/llm/llm_backend.h"

namespace soundproof {

// Qwen3.5 chat backend on top of MNN-LLM.
//
// Supports text and single-image turns. The backend renders the chat template
// itself (the shared LlmEngine disables MNN's automatic templating) and feeds
// images either as a file path (MNN::CV::imread) or as raw RGB8.
class MnnLlmBackend : public LlmBackend {
 public:
  MnnLlmBackend();
  ~MnnLlmBackend() override;

  LlmStatus Init(const LlmModelConfig& config) override;

  const char* ModelType() const override { return model_type_.c_str(); }

  LlmStatus Generate(const std::vector<LlmMessage>& messages,
                     const LlmImage& image,
                     const LlmSamplingParams& params,
                     LlmReply* reply) override;

  LlmStatus Reset() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_type_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_LLM_MNN_LLM_BACKEND_H_
