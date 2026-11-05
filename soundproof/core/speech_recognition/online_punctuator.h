#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_PUNCTUATOR_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_PUNCTUATOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/speech_recognition/punctuation_model.h"

namespace ssentencepiece {
class Ssentencepiece;
}

namespace soundproof {

struct OnlinePunctuatorConfig {
  std::string model_path;
  std::string bpe_vocab_path;
  int num_threads = 1;
};

// Adds punctuation and casing to ASR text using the CNN-BiLSTM model and a
// sentencepiece BPE vocabulary (port of sherpa-mnn's OnlinePunctuation).
class OnlinePunctuator {
 public:
  OnlinePunctuator();
  ~OnlinePunctuator();

  OnlinePunctuator(const OnlinePunctuator&) = delete;
  OnlinePunctuator& operator=(const OnlinePunctuator&) = delete;

  bool Init(const OnlinePunctuatorConfig& config);
  bool ok() const { return ok_; }

  std::string AddPunctuationWithCase(const std::string& text);

 private:
  void EncodeSentences(const std::string& text,
                       std::vector<int32_t>* tokens_list,
                       std::vector<int32_t>* valids_list,
                       std::vector<int32_t>* label_lens) const;
  std::string DecodeSentences(const std::string& raw_text,
                              const std::vector<int32_t>& case_pred,
                              const std::vector<int32_t>& punct_pred) const;

  PunctuationModel model_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_PUNCTUATOR_H_
