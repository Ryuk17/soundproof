#ifndef SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_RECOGNIZER_H_
#define SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_RECOGNIZER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/speaker_recognition/speaker_embedding_manager.h"
#include "core/speaker_recognition/sr_backend.h"

namespace soundproof {

class FeatureExtractor;

struct SrRecognizerConfig {
  SrModelConfig backend;
  float threshold = 0.5f;
};

struct SrRecognizerResult {
  std::string speaker_name;
  float score = 0.0f;
  bool has_result = false;
};

// Streaming speaker recognition front-end.
//
// Features are cached as they arrive. A trigger (typically "KWS fired") runs
// embedding extraction over the cached features and matches them against the
// registered speakers; the cache is then cleared for the next segment.
// Enrollment reuses the same backend on complete utterances.
class SpeakerRecognizer {
 public:
  SpeakerRecognizer();
  ~SpeakerRecognizer();

  SpeakerRecognizer(const SpeakerRecognizer&) = delete;
  SpeakerRecognizer& operator=(const SpeakerRecognizer&) = delete;

  bool Init(const SrRecognizerConfig& config);
  bool ok() const { return initialized_; }

  // Feeds audio. trigger=false only caches features; trigger=true computes an
  // embedding from all cached features, identifies the speaker and clears the
  // cache.
  SrStatus Process(const int16_t* audio_in, size_t samples, bool trigger,
                   SrRecognizerResult* result);

  // One-shot identification over a complete segment. Does not touch the
  // streaming feature cache.
  SrStatus IdentifySegment(const int16_t* audio_in, size_t samples,
                           SrRecognizerResult* result);

  // Offline enrollment helpers.
  SrStatus Extract(const int16_t* audio_in, size_t samples,
                   std::vector<float>* embedding);
  SrStatus RegisterPcm(const std::string& name, const int16_t* audio_in,
                       size_t samples);
  SrStatus AddEmbedding(const std::string& name,
                        const std::vector<float>& embedding);
  SrStatus AddEmbeddings(
      const std::string& name,
      const std::vector<std::vector<float>>& embedding_list);
  SrStatus Remove(const std::string& name);
  void Clear();
  int32_t NumSpeakers() const;
  std::vector<std::string> GetAllSpeakers() const;

  bool SetThreshold(float threshold);
  float threshold() const { return threshold_; }
  int32_t EmbeddingDim() const;
  int32_t SampleRate() const;

  // Clears the streaming feature cache only (registrations are kept).
  void ResetStream();

 private:
  SrStatus ComputeFromPcm(FeatureExtractor* extractor, const int16_t* audio_in,
                          size_t samples, std::vector<float>* embedding);
  SrStatus IdentifyEmbedding(const std::vector<float>& embedding,
                             SrRecognizerResult* result) const;

  std::unique_ptr<SrBackend> backend_;
  std::unique_ptr<FeatureExtractor> stream_features_;
  std::unique_ptr<FeatureExtractor> enroll_features_;
  std::unique_ptr<SpeakerEmbeddingManager> manager_;
  float threshold_ = 0.5f;
  bool initialized_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEAKER_RECOGNITION_SPEAKER_RECOGNIZER_H_
