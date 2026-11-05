#ifndef SOUNDPROOF_CORE_TEXT_TO_SPEECH_VITS_BACKEND_H_
#define SOUNDPROOF_CORE_TEXT_TO_SPEECH_VITS_BACKEND_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/text_to_speech/tts_backend.h"
#include "core/text_to_speech/tts_tokenizer.h"
#include "mnn_engine/mnn_engine.h"

namespace soundproof {

// VITS / Piper-style TTS backend. Inputs are mapped by name
// (x, x_lengths, noise_scale, length_scale, noise_scale_w, sid, tones,
// scales) so standard converted models load without model-specific code.
// Text input needs an external G2P frontend; phoneme/token ids are supported.
class VitsBackend : public TtsBackend {
 public:
  VitsBackend();
  ~VitsBackend() override;

  TtsStatus Init(const TtsModelConfig& config) override;

  const char* ModelType() const override { return "vits"; }
  int32_t SampleRate() const override { return sample_rate_; }
  int32_t NumSpeakers() const override { return num_speakers_; }
  int32_t StyleDim() const override { return 0; }
  int32_t MaxTokenLen() const override { return 0; }

  TtsStatus EncodePhonemes(const std::string& text,
                           std::vector<int32_t>* token_ids) override;

  TtsStatus Synthesize(const std::vector<int32_t>& token_ids,
                       int32_t speaker_id, float speed,
                       std::vector<float>* samples) override;

 private:
  enum class Role {
    kToken,
    kTones,
    kLength,
    kNoiseScale,
    kLengthScale,
    kNoiseScaleW,
    kSid,
    kScales,
    kOther,
  };

  mnn_engine::Engine engine_;
  std::vector<mnn_engine::Tensor> inputs_;
  std::map<size_t, Role> roles_;

  TtsTokenizer tokenizer_;
  int32_t sample_rate_ = 22050;
  int32_t num_speakers_ = 1;
  float noise_scale_ = 0.667f;
  float length_scale_ = 1.0f;
  float noise_scale_w_ = 0.8f;
  int32_t add_blank_ = 0;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_TO_SPEECH_VITS_BACKEND_H_
