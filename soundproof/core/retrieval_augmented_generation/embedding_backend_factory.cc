#include "core/retrieval_augmented_generation/embedding_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/retrieval_augmented_generation/mnn_llm_embedding_backend.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

std::unique_ptr<EmbeddingBackend> CreateEmbeddingBackend(
    const EmbeddingModelConfig& config) {
  const std::string type = ToLower(config.model_type);
  if (type.empty() || type == "mnn-llm" || type == "qwen3-embedding" ||
      type == "qwen3") {
    return std::make_unique<MnnLlmEmbeddingBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateEmbeddingBackend: unsupported model type '"
                    << type << "'";
  return nullptr;
}

}  // namespace soundproof
