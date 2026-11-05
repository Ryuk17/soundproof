#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_PUNCTUATION_MODEL_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_PUNCTUATION_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mnn_engine/mnn_engine.h"

namespace soundproof {

struct PunctuationMeta {
  int32_t comma_id = 1;
  int32_t period_id = 2;
  int32_t question_id = 3;
  int32_t upper_id = 1;
  int32_t cap_id = 2;
  int32_t mix_case_id = 3;
};

struct PunctuationModelConfig {
  std::string model_path;
  int num_threads = 1;
};

// CNN-BiLSTM punctuation/casing model (sherpa-mnn's online punctuation model).
// Input: token ids and valid masks of shape [1, 200] plus a label length.
// Output: case logits and punctuation logits.
class PunctuationModel {
 public:
  PunctuationModel();
  ~PunctuationModel();

  PunctuationModel(const PunctuationModel&) = delete;
  PunctuationModel& operator=(const PunctuationModel&) = delete;

  bool Init(const PunctuationModelConfig& config);
  bool ok() const { return ok_; }

  static constexpr int32_t kMaxSeqLen = 200;

  const PunctuationMeta& meta() const { return meta_; }

  bool Forward(const std::vector<int32_t>& token_ids,
               const std::vector<int32_t>& valid_ids, int32_t label_len,
               std::vector<float>* case_logits, int32_t* num_cases,
               std::vector<float>* punct_logits, int32_t* num_punctuations);

 private:
  mnn_engine::Engine engine_;
  std::vector<mnn_engine::Tensor> inputs_;
  PunctuationMeta meta_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_PUNCTUATION_MODEL_H_
