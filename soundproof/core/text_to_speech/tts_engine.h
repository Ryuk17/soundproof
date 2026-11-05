#ifndef SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_ENGINE_H_
#define SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/text_to_speech/tts_backend.h"

namespace soundproof {

enum class TtsInputType {
  kText = 0,
  kPhoneme = 1,
  kTokenIds = 2,
};

struct TtsRequest {
  TtsInputType input_type = TtsInputType::kText;
  std::string text;
  std::vector<int32_t> token_ids;
  int32_t speaker_id = 0;
  float speed = 1.0f;
};

struct TtsAudio {
  std::vector<int16_t> samples;
  int32_t sample_rate = 0;
};

// TTS front-end: input conversion (text/phoneme/ids), synthesis and
// float-to-int16 conversion. One instance handles one request at a time.
class TtsEngine {
 public:
  TtsEngine();
  ~TtsEngine();

  TtsEngine(const TtsEngine&) = delete;
  TtsEngine& operator=(const TtsEngine&) = delete;

  bool Init(const TtsModelConfig& config);
  bool ok() const { return ok_; }

  TtsStatus Synthesize(const TtsRequest& request, TtsAudio* audio);

  int32_t SampleRate() const;
  int32_t NumSpeakers() const;

  // Clears the last generated audio; configurations are kept.
  void Reset();

  const TtsAudio& last_audio() const { return last_audio_; }

 private:
  std::unique_ptr<TtsBackend> backend_;
  TtsAudio last_audio_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_ENGINE_H_
