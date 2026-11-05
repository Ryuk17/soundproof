#ifndef SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_TOKENIZER_H_
#define SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soundproof {

// Tokenizer for TTS symbols/phones.
//
// Supports two formats:
//  - MNN LLM tokenizer.txt (magic 430, Tiktoken type): base64-encoded tokens.
//  - sherpa/Kaldi tokens.txt ("<symbol> <id>" per line).
// Encoding uses greedy longest match over the token strings.
class TtsTokenizer {
 public:
  bool Load(const std::string& path);

  bool ok() const { return ok_; }
  int32_t VocabSize() const {
    return static_cast<int32_t>(token_to_id_.size());
  }

  bool Encode(const std::string& text,
              std::vector<int32_t>* token_ids) const;

 private:
  bool LoadMnnTokenizer(const std::vector<std::string>& lines);
  bool LoadSymbolTable(const std::vector<std::string>& lines);

  std::unordered_map<std::string, int32_t> token_to_id_;
  size_t max_token_bytes_ = 0;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TEXT_TO_SPEECH_TTS_TOKENIZER_H_
