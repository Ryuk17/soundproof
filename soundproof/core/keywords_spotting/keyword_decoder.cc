#include "core/keywords_spotting/keyword_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "core/keywords_spotting/context_graph.h"
#include "model/online_transducer_model.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

void LogSoftmax(std::vector<float>* logits) {
  if (logits->empty()) {
    return;
  }
  const float max_value = *std::max_element(logits->begin(), logits->end());
  double sum = 0.0;
  for (float v : *logits) {
    sum += std::exp(static_cast<double>(v) - max_value);
  }
  const double log_sum = std::log(sum) + max_value;
  for (float& v : *logits) {
    v = static_cast<float>(static_cast<double>(v) - log_sum);
  }
}

double LogAdd(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity()) {
    return b;
  }
  if (b == -std::numeric_limits<double>::infinity()) {
    return a;
  }
  const double hi = std::max(a, b);
  const double lo = std::min(a, b);
  return hi + std::log1p(std::exp(lo - hi));
}

}  // namespace

std::string Hypothesis::Key() const {
  std::string key;
  key.reserve(ys.size() * 4);
  for (size_t i = 0; i < ys.size(); ++i) {
    if (i != 0) {
      key.push_back('-');
    }
    key += std::to_string(ys[i]);
  }
  return key;
}

void Hypotheses::Add(Hypothesis hyp) {
  const std::string key = hyp.Key();
  auto it = hyps_.find(key);
  if (it == hyps_.end()) {
    hyps_.emplace(key, std::move(hyp));
  } else {
    it->second.log_prob = LogAdd(it->second.log_prob, hyp.log_prob);
  }
}

Hypothesis Hypotheses::GetMostProbable() const {
  if (hyps_.empty()) {
    return Hypothesis();
  }
  return std::max_element(
             hyps_.begin(), hyps_.end(),
             [](const auto& left, const auto& right) {
               return left.second.log_prob < right.second.log_prob;
             })
      ->second;
}

std::vector<Hypothesis> Hypotheses::Vec() const {
  std::vector<Hypothesis> result;
  result.reserve(hyps_.size());
  for (const auto& kv : hyps_) {
    result.push_back(kv.second);
  }
  return result;
}

KeywordDecoder::KeywordDecoder(OnlineTransducerModel* model,
                               int32_t max_active_paths,
                               int32_t num_trailing_blanks, int32_t unk_id)
    : model_(model),
      max_active_paths_(max_active_paths),
      num_trailing_blanks_(num_trailing_blanks),
      unk_id_(unk_id) {}

void KeywordDecoder::Reset(const ContextGraph& graph) {
  const int32_t context_size = model_->ContextSize();
  blanks_.assign(static_cast<size_t>(context_size), -1);
  blanks_.back() = 0;

  Hypothesis hyp;
  hyp.ys = blanks_;
  hyp.log_prob = 0.0;
  hyp.context_state = graph.Root();

  hyps_.Clear();
  hyps_.Add(std::move(hyp));
  frame_offset_ = 0;
  initialized_ = true;
}

bool KeywordDecoder::Decode(const ContextGraph& graph, const float* encoder_out,
                            int32_t num_frames, int32_t frame_dim,
                            KeywordHit* hit) {
  if (!initialized_ || encoder_out == nullptr || hit == nullptr ||
      num_frames < 0 || frame_dim <= 0) {
    return false;
  }
  *hit = KeywordHit();

  const int32_t context_size = model_->ContextSize();
  const int32_t vocab_size = model_->VocabSize();
  bool found = false;

  std::vector<Hypothesis> prev = hyps_.Vec();

  for (int32_t t = 0; t < num_frames; ++t) {
    if (prev.empty()) {
      break;
    }

    // Pure log-softmax scores for each hypothesis row.
    std::vector<std::vector<float>> row_logprobs(prev.size());
    for (size_t h = 0; h < prev.size(); ++h) {
      std::vector<int32_t> tokens(prev[h].ys.end() - context_size,
                                  prev[h].ys.end());
      std::vector<float> decoder_out;
      if (!model_->RunDecoder(tokens, &decoder_out)) {
        RTC_LOG(LS_ERROR) << "KeywordDecoder: decoder forward failed";
        return found;
      }
      std::vector<float> logits;
      if (!model_->RunJoiner(encoder_out + static_cast<size_t>(t) * frame_dim,
                             decoder_out, &logits) ||
          static_cast<int32_t>(logits.size()) != vocab_size) {
        RTC_LOG(LS_ERROR) << "KeywordDecoder: joiner forward failed";
        return found;
      }
      LogSoftmax(&logits);
      row_logprobs[h] = std::move(logits);
    }

    struct Candidate {
      float score;
      size_t hyp;
      int32_t token;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(prev.size() * static_cast<size_t>(vocab_size));
    for (size_t h = 0; h < prev.size(); ++h) {
      for (int32_t token = 0; token < vocab_size; ++token) {
        candidates.push_back(
            {row_logprobs[h][static_cast<size_t>(token)] +
                 static_cast<float>(prev[h].log_prob),
             h, token});
      }
    }

    const size_t k = std::min(
        static_cast<size_t>(std::max(max_active_paths_, 1)),
        candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(k),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    Hypotheses next;
    for (size_t i = 0; i < k; ++i) {
      const Candidate& candidate = candidates[i];
      Hypothesis new_hyp = prev[candidate.hyp];
      const float log_prob = row_logprobs[candidate.hyp]
                                          [static_cast<size_t>(candidate.token)];
      if (candidate.token != 0 && candidate.token != unk_id_) {
        new_hyp.ys.push_back(candidate.token);
        new_hyp.timestamps.push_back(t + frame_offset_);
        new_hyp.ys_probs.push_back(std::exp(log_prob));
        new_hyp.num_trailing_blanks = 0;

        const auto context_result =
            graph.ForwardOneStep(new_hyp.context_state, candidate.token);
        new_hyp.log_prob =
            static_cast<double>(candidate.score) + std::get<0>(context_result);
        new_hyp.context_state = std::get<1>(context_result);
        if (new_hyp.context_state != nullptr &&
            new_hyp.context_state->token == -1) {
          // Back to the root: forget the matched prefix.
          new_hyp.ys = blanks_;
          new_hyp.timestamps.clear();
          new_hyp.ys_probs.clear();
        }
      } else {
        ++new_hyp.num_trailing_blanks;
        new_hyp.log_prob = candidate.score;
      }
      next.Add(std::move(new_hyp));
    }

    const Hypothesis best = next.GetMostProbable();
    const auto matched = graph.IsMatched(best.context_state);
    if (matched.first && matched.second != nullptr) {
      const ContextState* state = matched.second;
      const int32_t level = state->level;
      if (best.num_trailing_blanks > num_trailing_blanks_ &&
          level > 0 &&
          static_cast<size_t>(level) <= best.ys_probs.size()) {
        float ys_prob = 0.0f;
        for (int32_t i = 0; i < level; ++i) {
          ys_prob += best.ys_probs[static_cast<size_t>(i)];
        }
        ys_prob /= static_cast<float>(level);
        if (ys_prob >= state->ac_threshold) {
          hit->phrase = state->phrase;
          hit->tokens.assign(best.ys.end() - level, best.ys.end());
          hit->timestamps.assign(best.timestamps.end() - level,
                                 best.timestamps.end());
          hit->score = ys_prob;
          found = true;

          // Reset the search to the initial blank hypothesis (sherpa).
          Hypothesis reset;
          reset.ys = blanks_;
          reset.context_state = graph.Root();
          next.Clear();
          next.Add(std::move(reset));
        }
      }
    }

    hyps_ = std::move(next);
    prev = hyps_.Vec();
  }

  frame_offset_ += num_frames;
  return found;
}

}  // namespace soundproof
