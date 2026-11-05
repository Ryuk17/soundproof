#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_HYPOTHESIS_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_HYPOTHESIS_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mnn_engine/mnn_engine.h"

namespace soundproof {

// A decoding hypothesis for transducer beam search (port of sherpa-mnn's
// Hypothesis, without hotword/context-graph fields).
struct Hypothesis {
  std::vector<int32_t> ys;
  std::vector<int32_t> timestamps;
  std::vector<float> ys_probs;
  std::vector<float> lm_probs;

  // Acoustic score of ys (log domain).
  double log_prob = 0.0;
  // LM score of ys (log domain).
  double lm_log_prob = 0.0;

  // Scores of the next token produced by the LM for the current ys, and the
  // corresponding LM states.
  std::vector<float> nn_lm_scores;
  std::vector<mnn_engine::Tensor> nn_lm_states;

  // Number of tokens already scored by the classic LM rescoring path.
  int32_t cur_scored_pos = 0;
  int32_t num_trailing_blanks = 0;

  double TotalLogProb() const { return log_prob + lm_log_prob; }

  std::string Key() const;
};

class Hypotheses {
 public:
  void Add(Hypothesis hyp);
  Hypothesis GetMostProbable(bool length_norm) const;
  std::vector<Hypothesis> Vec() const;

  size_t Size() const { return hyps_.size(); }
  bool Empty() const { return hyps_.empty(); }
  void Clear() { hyps_.clear(); }

  // Applies |fn| to every hypothesis in place (used by LM rescoring).
  template <typename F>
  void ForEach(F&& fn) {
    for (auto& kv : hyps_) {
      fn(&kv.second);
    }
  }

 private:
  std::unordered_map<std::string, Hypothesis> hyps_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_HYPOTHESIS_H_
