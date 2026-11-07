#include "core/text_embedding/embedding_engine.h"

#include <cmath>
#include <utility>

#include "utils/logging.h"

namespace soundproof {

EmbeddingEngine::EmbeddingEngine() = default;
EmbeddingEngine::~EmbeddingEngine() = default;

EmbeddingStatus EmbeddingEngine::Init(const EmbeddingModelConfig& config) {
  backend_.reset();
  ok_ = false;
  ClearResults();

  backend_ = CreateEmbeddingBackend(config);
  if (backend_ == nullptr) {
    RTC_LOG(LS_ERROR) << "EmbeddingEngine: cannot create backend";
    return EmbeddingStatus::kUnsupportedModel;
  }

  const EmbeddingStatus status = backend_->Init(config);
  if (status != EmbeddingStatus::kOk) {
    RTC_LOG(LS_ERROR) << "EmbeddingEngine: backend init failed";
    backend_.reset();
    return status;
  }

  ok_ = true;
  return EmbeddingStatus::kOk;
}

EmbeddingStatus EmbeddingEngine::Compute(const std::string& text,
                                         bool normalize) {
  if (!ok_ || backend_ == nullptr) {
    return EmbeddingStatus::kNotInitialized;
  }
  if (text.empty()) {
    RTC_LOG(LS_WARNING) << "EmbeddingEngine: empty text";
    return EmbeddingStatus::kInvalidInput;
  }

  if (text != last_text_ || last_raw_.empty()) {
    std::vector<float> raw;
    const EmbeddingStatus status = backend_->Compute(text, &raw);
    if (status != EmbeddingStatus::kOk) {
      return status;
    }
    if (raw.empty()) {
      RTC_LOG(LS_ERROR) << "EmbeddingEngine: backend returned no data";
      return EmbeddingStatus::kInferenceFailed;
    }
    if (static_cast<int32_t>(raw.size()) != backend_->Dim()) {
      RTC_LOG(LS_ERROR) << "EmbeddingEngine: dim mismatch, got " << raw.size()
                        << " expected " << backend_->Dim();
      return EmbeddingStatus::kEmbeddingDimMismatch;
    }
    last_text_ = text;
    last_raw_ = std::move(raw);
  }

  last_embedding_ = last_raw_;
  if (normalize) {
    double sum = 0.0;
    for (float value : last_embedding_) {
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const float norm = static_cast<float>(std::sqrt(sum));
    if (!std::isfinite(norm) || norm <= 1e-12f) {
      RTC_LOG(LS_ERROR) << "EmbeddingEngine: degenerate embedding (norm="
                        << norm << ")";
      return EmbeddingStatus::kInferenceFailed;
    }
    const float inv_norm = 1.0f / norm;
    for (float& value : last_embedding_) {
      value *= inv_norm;
    }
  }

  return EmbeddingStatus::kOk;
}

int32_t EmbeddingEngine::Dim() const {
  return backend_ == nullptr ? 0 : backend_->Dim();
}

void EmbeddingEngine::Reset() {
  ClearResults();
}

void EmbeddingEngine::ClearResults() {
  last_embedding_.clear();
  last_text_.clear();
  last_raw_.clear();
}

}  // namespace soundproof
