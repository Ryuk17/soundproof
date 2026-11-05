#ifndef SOUNDPROOF_CORE_TEXT_TO_SPEECH_KOKORO_BACKEND_H_
#define SOUNDPROOF_CORE_TEXT_TO_SPEECH_KOKORO_BACKEND_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/text_to_speech/tts_backend.h"
#include "core/text_to_speech/tts_tokenizer.h"
#include "mnn_engine/mnn_engine.h"

namespace soundproof {

// Kokoro TTS backend.
//
// Supports the MNN-exported Kokoro layout (tts.mnn + voices.mnn +
// tokenizer.txt) as well as sherpa-style metadata when present.
class KokoroBackend : public TtsBackend {
 public:
  KokoroBackend();
  ~KokoroBackend() override;

  TtsStatus Init(const TtsModelConfig& config) override;

  const char* ModelType() const override { return "kokoro"; }
  int32_t SampleRate() const override { return sample_rate_; }
  int32_t NumSpeakers() const override { return num_speakers_; }
  int32_t StyleDim() const override { return style_dim_; }
  int32_t MaxTokenLen() const override { return max_token_len_; }

  TtsStatus EncodePhonemes(const std::string& text,
                           std::vector<int32_t>* token_ids) override;

  TtsStatus Synthesize(const std::vector<int32_t>& token_ids,
                       int32_t speaker_id, float speed,
                       std::vector<float>* samples) override;

 private:
  TtsStatus LoadVoices(const std::string& path);
  TtsStatus LoadMnnVoices(const std::string& path);
  TtsStatus LoadRawVoices(const std::string& path);

  mnn_engine::Engine engine_;
  std::vector<mnn_engine::Tensor> inputs_;
  size_t token_index_ = 0;
  size_t style_index_ = 1;
  size_t speed_index_ = 2;

  TtsTokenizer tokenizer_;

  // styles_[sid][len][style_dim]
  std::vector<float> styles_;
  int32_t sample_rate_ = 24000;
  int32_t num_speakers_ = 1;
  int32_t style_dim_ = 256;
  int32_t max_token_len_ = 0;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_TO_SPEECH_KOKORO_BACKEND_H_
