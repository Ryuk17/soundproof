#include "core/turn_detection/turn_detector.h"

#include <algorithm>

#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr float kPcm16Scale = 1.0f / 32768.0f;

}  // namespace

TurnDetector::TurnDetector() = default;
TurnDetector::~TurnDetector() = default;

TurnStatus TurnDetector::Init(const TurnModelConfig& config) {
  backend_.reset();
  ok_ = false;
  ring_.clear();
  linear_.clear();
  write_pos_ = 0;
  size_ = 0;
  has_last_ = false;
  dirty_ = true;

  backend_ = CreateTurnBackend(config);
  if (backend_ == nullptr) {
    RTC_LOG(LS_ERROR) << "TurnDetector: cannot create backend";
    return TurnStatus::kUnsupportedModel;
  }

  const TurnStatus status = backend_->Init(config);
  if (status != TurnStatus::kOk) {
    RTC_LOG(LS_ERROR) << "TurnDetector: backend init failed";
    backend_.reset();
    return status;
  }

  const int32_t context = backend_->ContextSamples();
  if (context <= 0) {
    RTC_LOG(LS_ERROR) << "TurnDetector: invalid context size " << context;
    backend_.reset();
    return TurnStatus::kUnsupportedModel;
  }
  ring_.assign(static_cast<size_t>(context), 0.0f);
  linear_.assign(static_cast<size_t>(context), 0.0f);
  ok_ = true;
  return TurnStatus::kOk;
}

TurnStatus TurnDetector::Append(const short* audio, int samples) {
  if (!ok_ || backend_ == nullptr) {
    return TurnStatus::kNotInitialized;
  }
  if (samples <= 0) {
    return TurnStatus::kOk;
  }
  if (audio == nullptr) {
    return TurnStatus::kInvalidInput;
  }
  const size_t capacity = ring_.size();
  for (int i = 0; i < samples; ++i) {
    ring_[write_pos_] = static_cast<float>(audio[i]) * kPcm16Scale;
    write_pos_ = (write_pos_ + 1) % capacity;
    if (size_ < capacity) {
      ++size_;
    }
  }
  dirty_ = true;
  return TurnStatus::kOk;
}

TurnStatus TurnDetector::Predict(float* probability) {
  if (!ok_ || backend_ == nullptr) {
    return TurnStatus::kNotInitialized;
  }
  if (probability == nullptr) {
    return TurnStatus::kInvalidInput;
  }
  if (size_ == 0) {
    RTC_LOG(LS_WARNING) << "TurnDetector: no audio buffered";
    return TurnStatus::kInvalidInput;
  }
  if (!dirty_ && has_last_) {
    *probability = last_probability_;
    return TurnStatus::kOk;
  }

  const size_t capacity = ring_.size();
  const size_t padding = capacity - size_;
  std::fill(linear_.begin(), linear_.end(), 0.0f);
  const size_t start = (write_pos_ + capacity - size_) % capacity;
  for (size_t i = 0; i < size_; ++i) {
    linear_[padding + i] = ring_[(start + i) % capacity];
  }

  float value = 0.0f;
  const TurnStatus status =
      backend_->Predict(linear_.data(), static_cast<int>(capacity), &value);
  if (status != TurnStatus::kOk) {
    return status;
  }
  last_probability_ = value;
  has_last_ = true;
  dirty_ = false;
  *probability = value;
  return TurnStatus::kOk;
}

int32_t TurnDetector::ContextSamples() const {
  return backend_ == nullptr ? 0 : backend_->ContextSamples();
}

int32_t TurnDetector::BufferedSamples() const {
  return static_cast<int32_t>(size_);
}

void TurnDetector::Reset() {
  write_pos_ = 0;
  size_ = 0;
  has_last_ = false;
  dirty_ = true;
}

}  // namespace soundproof
