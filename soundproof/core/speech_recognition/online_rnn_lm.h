#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RNN_LM_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RNN_LM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/speech_recognition/hypothesis.h"
#include "mnn_engine/mnn_engine.h"

namespace soundproof {

struct OnlineRnnLmConfig {
  std::string model_path;
  float scale = 0.5f;
  bool shallow_fusion = true;
  int num_threads = 1;
};

// Streaming RNN (LSTM) language model used by transducer beam search.
// Port of sherpa-mnn's OnlineRnnLM: model inputs are the current token(s) and
// the LSTM states, outputs are next-token scores and updated states.
class OnlineRnnLm {
 public:
  OnlineRnnLm();
  ~OnlineRnnLm();

  OnlineRnnLm(const OnlineRnnLm&) = delete;
  OnlineRnnLm& operator=(const OnlineRnnLm&) = delete;

  bool Init(const OnlineRnnLmConfig& config);
  bool ok() const { return ok_; }

  // Shallow fusion: updates hyp->lm_log_prob for the last emitted token and
  // prepares the LM scores/states for the next token.
  bool ComputeScoreSF(Hypothesis* hyp);

  // Classic rescore: scores the not-yet-scored tokens of every hypothesis.
  bool ComputeScore(int32_t context_size, Hypotheses* hyps);

 private:
  bool ScoreToken(const std::vector<int32_t>& tokens,
                  const std::vector<mnn_engine::Tensor>& states,
                  std::vector<float>* scores,
                  std::vector<mnn_engine::Tensor>* next_states);

  mnn_engine::Engine engine_;
  std::vector<mnn_engine::Tensor> inputs_;
  size_t token_index_ = 0;
  size_t state_h_index_ = 1;
  size_t state_c_index_ = 2;

  int32_t num_layers_ = 2;
  int32_t hidden_size_ = 512;
  int32_t sos_id_ = 1;
  float scale_ = 0.5f;
  bool shallow_fusion_ = true;
  bool ok_ = false;

  std::vector<float> init_scores_;
  std::vector<mnn_engine::Tensor> init_states_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RNN_LM_H_
