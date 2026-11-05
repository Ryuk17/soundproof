#include "core/speech_recognition/transducer_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "model/online_transducer_model.h"
#include "core/speech_recognition/online_rnn_lm.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int32_t kBlankId = 0;

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

std::vector<int32_t> MakeBlankContext(int32_t context_size) {
  std::vector<int32_t> tokens(static_cast<size_t>(context_size), -1);
  tokens.back() = kBlankId;
  return tokens;
}

}  // namespace

TransducerBeamSearchDecoder::TransducerBeamSearchDecoder(
    OnlineTransducerModel* model, OnlineRnnLm* lm, int32_t max_active_paths,
    float lm_scale, bool shallow_fusion, int32_t unk_id, float blank_penalty,
    float temperature_scale)
    : model_(model),
      lm_(lm),
      max_active_paths_(std::max(max_active_paths, 1)),
      unk_id_(unk_id),
      lm_scale_(lm_scale),
      shallow_fusion_(shallow_fusion),
      blank_penalty_(blank_penalty),
      temperature_scale_(temperature_scale > 0.0f ? temperature_scale : 1.0f) {
  Reset();
}

void TransducerBeamSearchDecoder::Reset() {
  const std::vector<int32_t> blanks = MakeBlankContext(model_->ContextSize());
  Hypothesis hyp;
  hyp.ys = blanks;
  hyps_.Clear();
  hyps_.Add(std::move(hyp));

  tokens_ = blanks;
  timestamps_.clear();
  num_trailing_blanks_ = 0;
  frame_offset_ = 0;
}

void TransducerBeamSearchDecoder::ResetSegment() {
  const int32_t context_size = model_->ContextSize();
  const Hypothesis best = hyps_.GetMostProbable(true);

  Hypothesis hyp;
  if (static_cast<int32_t>(best.ys.size()) > context_size) {
    hyp.ys.assign(best.ys.end() - context_size, best.ys.end());
  } else {
    hyp.ys = MakeBlankContext(context_size);
  }
  hyps_.Clear();
  hyps_.Add(std::move(hyp));

  tokens_ = hyps_.GetMostProbable(true).ys;
  timestamps_.clear();
  num_trailing_blanks_ = 0;
  frame_offset_ = 0;
}

std::vector<int32_t> TransducerBeamSearchDecoder::ContextTokens(
    const std::vector<int32_t>& ys) const {
  const int32_t context_size = model_->ContextSize();
  if (static_cast<int32_t>(ys.size()) < context_size) {
    return MakeBlankContext(context_size);
  }
  return std::vector<int32_t>(ys.end() - context_size, ys.end());
}

bool TransducerBeamSearchDecoder::Decode(const float* encoder_out,
                                         int32_t num_frames,
                                         int32_t frame_dim) {
  if (encoder_out == nullptr || num_frames < 0 || frame_dim <= 0) {
    return false;
  }
  const int32_t vocab_size = model_->VocabSize();

  std::vector<Hypothesis> prev = hyps_.Vec();
  if (prev.empty()) {
    Reset();
    prev = hyps_.Vec();
  }

  struct Candidate {
    float score;
    size_t hyp;
    int32_t token;
  };

  for (int32_t t = 0; t < num_frames; ++t) {
    std::vector<std::vector<float>> row_logprobs(prev.size());
    std::vector<std::vector<float>> row_temp_logprobs(prev.size());

    for (size_t h = 0; h < prev.size(); ++h) {
      std::vector<float> decoder_out;
      if (!model_->RunDecoder(ContextTokens(prev[h].ys), &decoder_out)) {
        RTC_LOG(LS_ERROR) << "BeamSearch: decoder forward failed";
        return false;
      }
      std::vector<float> logits;
      if (!model_->RunJoiner(encoder_out + static_cast<size_t>(t) * frame_dim,
                             decoder_out, &logits) ||
          static_cast<int32_t>(logits.size()) != vocab_size) {
        RTC_LOG(LS_ERROR) << "BeamSearch: joiner forward failed";
        return false;
      }

      std::vector<float> temp_logits = logits;
      for (float& v : temp_logits) {
        v /= temperature_scale_;
      }
      LogSoftmax(&temp_logits);

      if (blank_penalty_ > 0.0f) {
        logits[kBlankId] -= blank_penalty_;
      }
      LogSoftmax(&logits);
      row_logprobs[h] = std::move(logits);
      row_temp_logprobs[h] = std::move(temp_logits);
    }

    std::vector<Candidate> candidates;
    candidates.reserve(prev.size() * static_cast<size_t>(vocab_size));
    for (size_t h = 0; h < prev.size(); ++h) {
      const float base = static_cast<float>(prev[h].log_prob);
      const float lm_base =
          (lm_ != nullptr && shallow_fusion_)
              ? static_cast<float>(prev[h].lm_log_prob)
              : 0.0f;
      for (int32_t token = 0; token < vocab_size; ++token) {
        candidates.push_back(
            {base + lm_base + row_logprobs[h][static_cast<size_t>(token)], h,
             token});
      }
    }

    const size_t k = std::min(
        static_cast<size_t>(max_active_paths_), candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(k),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    Hypotheses next;
    for (size_t i = 0; i < k; ++i) {
      const Candidate& candidate = candidates[i];
      Hypothesis new_hyp = prev[candidate.hyp];
      const double prev_lm_log_prob = new_hyp.lm_log_prob;

      if (candidate.token != kBlankId && candidate.token != unk_id_) {
        new_hyp.ys.push_back(candidate.token);
        new_hyp.timestamps.push_back(t + frame_offset_);
        new_hyp.num_trailing_blanks = 0;

        if (lm_ != nullptr && shallow_fusion_ &&
            !lm_->ComputeScoreSF(&new_hyp)) {
          RTC_LOG(LS_ERROR) << "BeamSearch: LM scoring failed";
          return false;
        }
        new_hyp.log_prob =
            static_cast<double>(candidate.score) -
            ((lm_ != nullptr && shallow_fusion_) ? prev_lm_log_prob : 0.0);
        new_hyp.ys_probs.push_back(
            row_temp_logprobs[candidate.hyp]
                             [static_cast<size_t>(candidate.token)]);
        if (lm_ != nullptr && shallow_fusion_) {
          double lm_prob = new_hyp.lm_log_prob - prev_lm_log_prob;
          if (lm_scale_ != 0.0f) {
            lm_prob /= lm_scale_;
          }
          new_hyp.lm_probs.push_back(static_cast<float>(lm_prob));
        }
      } else {
        ++new_hyp.num_trailing_blanks;
        new_hyp.log_prob =
            static_cast<double>(candidate.score) -
            ((lm_ != nullptr && shallow_fusion_) ? prev_lm_log_prob : 0.0);
      }
      next.Add(std::move(new_hyp));
    }

    hyps_ = std::move(next);
    prev = hyps_.Vec();
  }

  if (lm_ != nullptr && !shallow_fusion_) {
    if (!lm_->ComputeScore(model_->ContextSize(), &hyps_)) {
      RTC_LOG(LS_ERROR) << "BeamSearch: LM rescoring failed";
      return false;
    }
  }

  const Hypothesis best = hyps_.GetMostProbable(true);
  tokens_ = best.ys;
  timestamps_ = best.timestamps;
  num_trailing_blanks_ = best.num_trailing_blanks;
  frame_offset_ += num_frames;
  return true;
}

}  // namespace soundproof
