#ifndef SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_ENGINE_H_
#define SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/retrieval_augmented_generation/embedding_backend.h"

namespace soundproof {

// Text-embedding front-end: owns the backend, applies optional L2
// normalization and caches the raw vector of the last input so that repeated
// queries do not re-run inference.
class EmbeddingEngine {
 public:
  EmbeddingEngine();
  ~EmbeddingEngine();

  EmbeddingEngine(const EmbeddingEngine&) = delete;
  EmbeddingEngine& operator=(const EmbeddingEngine&) = delete;

  EmbeddingStatus Init(const EmbeddingModelConfig& config);
  bool ok() const { return ok_; }

  // Runs (or reuses) the embedding for |text|. When |normalize| is true the
  // result is L2-normalized.
  EmbeddingStatus Compute(const std::string& text, bool normalize);

  int32_t Dim() const;

  // Engine-owned result, valid until the next Compute/Reset/Init.
  const float* embedding() const {
    return last_embedding_.empty() ? nullptr : last_embedding_.data();
  }
  int32_t output_dim() const {
    return static_cast<int32_t>(last_embedding_.size());
  }

  // Drops the cached text/vectors and the last result; the model stays loaded.
  void Reset();

 private:
  void ClearResults();

  std::unique_ptr<EmbeddingBackend> backend_;
  std::vector<float> last_embedding_;
  std::string last_text_;
  std::vector<float> last_raw_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_RETRIEVAL_AUGMENTED_GENERATION_EMBEDDING_ENGINE_H_
