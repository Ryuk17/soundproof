#include "core/speaker_recognition/speaker_embedding_manager.h"

#include <algorithm>
#include <cmath>

namespace soundproof {
namespace {

bool Normalize(const float* src, int32_t dim, std::vector<float>* dst) {
  if (src == nullptr || dst == nullptr || dim <= 0) {
    return false;
  }
  double norm_sq = 0.0;
  for (int32_t i = 0; i < dim; ++i) {
    norm_sq += static_cast<double>(src[i]) * src[i];
  }
  const double norm = std::sqrt(norm_sq);
  if (norm <= 1e-12) {
    return false;
  }
  dst->resize(static_cast<size_t>(dim));
  for (int32_t i = 0; i < dim; ++i) {
    (*dst)[static_cast<size_t>(i)] =
        static_cast<float>(src[i] / norm);
  }
  return true;
}

float Dot(const float* a, const std::vector<float>& b) {
  double sum = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return static_cast<float>(sum);
}

}  // namespace

SpeakerEmbeddingManager::SpeakerEmbeddingManager(int32_t dim) : dim_(dim) {}

bool SpeakerEmbeddingManager::Add(const std::string& name,
                                  const float* embedding) {
  if (Contains(name)) {
    return false;
  }
  std::vector<float> normalized;
  if (!Normalize(embedding, dim_, &normalized)) {
    return false;
  }
  entries_.emplace_back(name, std::move(normalized));
  return true;
}

bool SpeakerEmbeddingManager::Add(
    const std::string& name,
    const std::vector<std::vector<float>>& embedding_list) {
  if (Contains(name) || embedding_list.empty()) {
    return false;
  }
  std::vector<double> sum(static_cast<size_t>(dim_), 0.0);
  for (const auto& embedding : embedding_list) {
    if (static_cast<int32_t>(embedding.size()) != dim_) {
      return false;
    }
    for (int32_t i = 0; i < dim_; ++i) {
      sum[static_cast<size_t>(i)] += embedding[static_cast<size_t>(i)];
    }
  }
  std::vector<float> average(static_cast<size_t>(dim_));
  for (int32_t i = 0; i < dim_; ++i) {
    average[static_cast<size_t>(i)] =
        static_cast<float>(sum[static_cast<size_t>(i)] / embedding_list.size());
  }
  return Add(name, average.data());
}

bool SpeakerEmbeddingManager::Remove(const std::string& name) {
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->first == name) {
      entries_.erase(it);
      return true;
    }
  }
  return false;
}

bool SpeakerEmbeddingManager::Contains(const std::string& name) const {
  for (const auto& entry : entries_) {
    if (entry.first == name) {
      return true;
    }
  }
  return false;
}

void SpeakerEmbeddingManager::Clear() {
  entries_.clear();
}

std::string SpeakerEmbeddingManager::Search(const float* embedding,
                                            float threshold,
                                            float* score) const {
  if (score != nullptr) {
    *score = 0.0f;
  }
  std::vector<float> query;
  if (!Normalize(embedding, dim_, &query)) {
    return {};
  }

  std::string best_name;
  float best_score = -2.0f;
  for (const auto& entry : entries_) {
    const float s = Dot(query.data(), entry.second);
    if (s > best_score) {
      best_score = s;
      best_name = entry.first;
    }
  }
  if (score != nullptr) {
    *score = best_score < -1.0f ? 0.0f : best_score;
  }
  if (best_score < threshold) {
    return {};
  }
  return best_name;
}

float SpeakerEmbeddingManager::Score(const std::string& name,
                                     const float* embedding) const {
  std::vector<float> query;
  if (!Normalize(embedding, dim_, &query)) {
    return -2.0f;
  }
  for (const auto& entry : entries_) {
    if (entry.first == name) {
      return Dot(query.data(), entry.second);
    }
  }
  return -2.0f;
}

bool SpeakerEmbeddingManager::Verify(const std::string& name,
                                     const float* embedding, float threshold,
                                     float* score) const {
  const float s = Score(name, embedding);
  if (score != nullptr) {
    *score = s < -1.0f ? 0.0f : s;
  }
  return s >= threshold;
}

std::vector<std::string> SpeakerEmbeddingManager::GetAllSpeakers() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const auto& entry : entries_) {
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace soundproof
