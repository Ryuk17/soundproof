#ifndef SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_BACKEND_H_
#define SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {

enum class EmbeddingStatus {
  kOk = 0,
  kNotInitialized,
  kModelLoadFailed,
  kUnsupportedModel,
  kInvalidInput,
  kEmbeddingDimMismatch,
  kInferenceFailed,
};

struct EmbeddingModelConfig {
  // Empty means "detect from paths" (v1: MNN-LLM embedding).
  std::string model_type;
  // Path to the MNN-LLM config.json (model/weight/tokenizer are resolved
  // relative to it).
  std::string config_path;
  // 0 = keep the thread_num from config.json; > 0 overrides it.
  int num_threads = 0;
};

// Generic text-embedding backend. Implementations own their model and
// tokenizer; the engine handles normalization and result caching.
class EmbeddingBackend {
 public:
  virtual ~EmbeddingBackend() = default;

  virtual EmbeddingStatus Init(const EmbeddingModelConfig& config) = 0;

  virtual const char* ModelType() const = 0;
  virtual int32_t Dim() const = 0;

  // Returns the raw sentence embedding (no L2 normalization).
  virtual EmbeddingStatus Compute(const std::string& text,
                                  std::vector<float>* embedding) = 0;
};

// Creates a backend from config.model_type (empty -> mnn-llm).
std::unique_ptr<EmbeddingBackend> CreateEmbeddingBackend(
    const EmbeddingModelConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_BACKEND_H_
