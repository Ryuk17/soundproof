#ifndef SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_BACKEND_H_
#define SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {

enum class TtsStatus {
  kOk = 0,
  kNotInitialized,
  kModelLoadFailed,
  kUnsupportedModel,
  kUnsupportedInput,
  kInvalidInput,
  kInferenceFailed,
};

struct TtsModelConfig {
  // Empty means "detect from paths" (voices present -> kokoro).
  std::string model_type;
  std::string model_path;
  std::string voices_path;   // Kokoro style embeddings (.mnn)
  std::string tokens_path;   // tokens.txt or MNN tokenizer.txt
  std::string lexicon_path;  // reserved for text frontends
  std::string data_dir;      // reserved for espeak-ng data
  std::string dict_dir;      // reserved for jieba dictionaries
  int num_threads = 1;
  // 0 -> backend default (Kokoro: 24000).
  int output_sample_rate = 0;
};

// A TTS model backend. Text/phoneme tokenization is exposed by the backend so
// that model families with different frontends stay independent.
class TtsBackend {
 public:
  virtual ~TtsBackend() = default;

  virtual TtsStatus Init(const TtsModelConfig& config) = 0;

  virtual const char* ModelType() const = 0;
  virtual int32_t SampleRate() const = 0;
  virtual int32_t NumSpeakers() const = 0;
  virtual int32_t StyleDim() const = 0;
  virtual int32_t MaxTokenLen() const = 0;

  // Encodes a phoneme/symbol string with the model's tokenizer. Returns
  // kUnsupportedInput if no tokenizer is configured.
  virtual TtsStatus EncodePhonemes(const std::string& text,
                                   std::vector<int32_t>* token_ids) = 0;

  // Pure text input requires a G2P frontend (espeak-ng/lexicon), which is not
  // part of this version.
  virtual bool SupportsText() const { return false; }
  virtual TtsStatus EncodeText(const std::string& /*text*/,
                               std::vector<int32_t>* /*token_ids*/) {
    return TtsStatus::kUnsupportedInput;
  }

  // Generates float samples in [-1, 1].
  virtual TtsStatus Synthesize(const std::vector<int32_t>& token_ids,
                               int32_t speaker_id, float speed,
                               std::vector<float>* samples) = 0;
};

// Creates a backend from config.model_type or available paths.
std::unique_ptr<TtsBackend> CreateTtsBackend(const TtsModelConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_BACKEND_H_
