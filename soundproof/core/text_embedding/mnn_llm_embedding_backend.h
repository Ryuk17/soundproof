#ifndef SOUNDPROOF_CORE_TEXT_EMBEDDING_MNN_LLM_EMBEDDING_BACKEND_H_
#define SOUNDPROOF_CORE_TEXT_EMBEDDING_MNN_LLM_EMBEDDING_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/text_embedding/embedding_backend.h"

namespace soundproof {

// Text embedding backed by MNN's LLM engine
// (MNN::Transformer::Embedding). It handles external weights, the int4
// embedding table, the HuggingFace tokenizer and the sentence_embeddings
// output; see transformers/llm/engine/src/embedding.cpp in MNN.
//
// The model is created from an MNN-LLM config.json. All other files
// (llm_config.json, embedding.mnn/.weight, embeddings_int4.bin,
// tokenizer.txt) are resolved relative to that config file.
class MnnLlmEmbeddingBackend : public EmbeddingBackend {
 public:
  MnnLlmEmbeddingBackend();
  ~MnnLlmEmbeddingBackend() override;

  EmbeddingStatus Init(const EmbeddingModelConfig& config) override;

  const char* ModelType() const override { return model_type_.c_str(); }
  int32_t Dim() const override { return dim_; }

  EmbeddingStatus Compute(const std::string& text,
                          std::vector<float>* embedding) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_type_;
  int32_t dim_ = 0;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_EMBEDDING_MNN_LLM_EMBEDDING_BACKEND_H_
