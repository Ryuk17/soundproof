#ifndef SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_EMBEDDING_MANAGER_H_
#define SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_EMBEDDING_MANAGER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace soundproof {

// Registry of speaker embeddings. Embeddings are L2-normalized on insertion
// and queries, so matching is a plain dot product (cosine similarity).
class SpeakerEmbeddingManager {
 public:
  explicit SpeakerEmbeddingManager(int32_t dim);

  // Returns false if |name| already exists or the dimension is wrong.
  bool Add(const std::string& name, const float* embedding);
  // Averages |embedding_list| (after validation) and stores the normalized
  // average.
  bool Add(const std::string& name,
           const std::vector<std::vector<float>>& embedding_list);

  bool Remove(const std::string& name);
  bool Contains(const std::string& name) const;
  void Clear();

  // Finds the best match. Returns an empty name when nothing reaches
  // |threshold|. |score| always receives the best cosine score (0 if empty).
  std::string Search(const float* embedding, float threshold,
                     float* score) const;
  // Score of |embedding| against |name|. Returns -2 if the name is unknown.
  float Score(const std::string& name, const float* embedding) const;
  bool Verify(const std::string& name, const float* embedding,
              float threshold, float* score) const;

  int32_t NumSpeakers() const {
    return static_cast<int32_t>(entries_.size());
  }
  int32_t Dim() const { return dim_; }
  // Speaker names sorted lexicographically.
  std::vector<std::string> GetAllSpeakers() const;

 private:
  int32_t dim_;
  std::vector<std::pair<std::string, std::vector<float>>> entries_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_EMBEDDING_MANAGER_H_
