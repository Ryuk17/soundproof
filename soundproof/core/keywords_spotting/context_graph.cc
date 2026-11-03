#include "core/keywords_spotting/context_graph.h"

#include <algorithm>
#include <queue>

#include "utils/logging.h"

namespace soundproof {

ContextGraph::ContextGraph(
    const std::vector<std::vector<int32_t>>& token_ids, float context_score,
    float ac_threshold, const std::vector<float>& scores,
    const std::vector<std::string>& phrases,
    const std::vector<float>& ac_thresholds)
    : context_score_(context_score), ac_threshold_(ac_threshold) {
  root_ = std::make_unique<ContextState>();
  root_->fail = root_.get();
  Build(token_ids, scores, phrases, ac_thresholds);
}

void ContextGraph::Build(
    const std::vector<std::vector<int32_t>>& token_ids,
    const std::vector<float>& scores,
    const std::vector<std::string>& phrases,
    const std::vector<float>& ac_thresholds) const {
  if (!scores.empty() && scores.size() != token_ids.size()) {
    RTC_LOG(LS_ERROR) << "ContextGraph: scores size mismatch";
    return;
  }
  if (!phrases.empty() && phrases.size() != token_ids.size()) {
    RTC_LOG(LS_ERROR) << "ContextGraph: phrases size mismatch";
    return;
  }
  if (!ac_thresholds.empty() && ac_thresholds.size() != token_ids.size()) {
    RTC_LOG(LS_ERROR) << "ContextGraph: thresholds size mismatch";
    return;
  }

  for (size_t i = 0; i < token_ids.size(); ++i) {
    auto* node = root_.get();
    float score = scores.empty() ? 0.0f : scores[i];
    score = score == 0.0f ? context_score_ : score;
    float ac_threshold = ac_thresholds.empty() ? 0.0f : ac_thresholds[i];
    ac_threshold = ac_threshold == 0.0f ? ac_threshold_ : ac_threshold;
    const std::string phrase = phrases.empty() ? std::string() : phrases[i];

    for (size_t j = 0; j < token_ids[i].size(); ++j) {
      const int32_t token = token_ids[i][j];
      const bool is_last = j + 1 == token_ids[i].size();
      auto it = node->next.find(token);
      if (it == node->next.end()) {
        node->next[token] = std::make_unique<ContextState>(
            token, score, node->node_score + score,
            is_last ? node->node_score + score : 0.0f,
            static_cast<int32_t>(j + 1), is_last ? ac_threshold : 0.0f,
            is_last, is_last ? phrase : std::string());
      } else {
        ContextState& child = *it->second;
        const float token_score = std::max(score, child.token_score);
        child.token_score = token_score;
        const float node_score = node->node_score + token_score;
        child.node_score = node_score;
        const bool is_end = is_last || child.is_end;
        child.output_score = is_end ? node_score : 0.0f;
        child.is_end = is_end;
        if (is_last) {
          child.phrase = phrase;
          child.ac_threshold = ac_threshold;
        }
      }
      node = node->next[token].get();
    }
  }
  FillFailOutput();
}

std::tuple<float, const ContextState*, const ContextState*>
ContextGraph::ForwardOneStep(const ContextState* state,
                             int32_t token) const {
  const ContextState* node = nullptr;
  float score = 0.0f;
  auto it = state->next.find(token);
  if (it != state->next.end()) {
    node = it->second.get();
    score = node->token_score;
  } else {
    node = state->fail;
    while (node->next.find(token) == node->next.end()) {
      node = node->fail;
      if (node->token == -1) {
        break;  // root
      }
    }
    auto next_it = node->next.find(token);
    if (next_it != node->next.end()) {
      node = next_it->second.get();
    }
    score = node->node_score - state->node_score;
  }

  if (node == nullptr) {
    RTC_LOG(LS_ERROR) << "ContextGraph: failed to advance context state";
    return std::make_tuple(0.0f, root_.get(), nullptr);
  }

  const ContextState* matched_node =
      node->is_end ? node : (node->output != nullptr ? node->output : nullptr);
  return std::make_tuple(score + node->output_score, node, matched_node);
}

std::pair<float, const ContextState*> ContextGraph::Finalize(
    const ContextState* state) const {
  return std::make_pair(-state->node_score, root_.get());
}

std::pair<bool, const ContextState*> ContextGraph::IsMatched(
    const ContextState* state) const {
  if (state == nullptr) {
    return std::make_pair(false, nullptr);
  }
  if (state->is_end) {
    return std::make_pair(true, state);
  }
  if (state->output != nullptr) {
    return std::make_pair(true, state->output);
  }
  return std::make_pair(false, nullptr);
}

void ContextGraph::FillFailOutput() const {
  std::queue<const ContextState*> node_queue;
  for (auto& kv : root_->next) {
    kv.second->fail = root_.get();
    node_queue.push(kv.second.get());
  }
  while (!node_queue.empty()) {
    const ContextState* current_node = node_queue.front();
    node_queue.pop();
    for (auto& kv : current_node->next) {
      const ContextState* fail = current_node->fail;
      auto fail_it = fail->next.find(kv.first);
      if (fail_it != fail->next.end()) {
        fail = fail_it->second.get();
      } else {
        fail = fail->fail;
        while (fail->next.find(kv.first) == fail->next.end()) {
          fail = fail->fail;
          if (fail->token == -1) {
            break;
          }
        }
        fail_it = fail->next.find(kv.first);
        if (fail_it != fail->next.end()) {
          fail = fail_it->second.get();
        }
      }
      kv.second->fail = fail;

      // Fill the output arc.
      const ContextState* output = fail;
      while (!output->is_end) {
        output = output->fail;
        if (output->token == -1) {
          output = nullptr;
          break;
        }
      }
      kv.second->output = output;
      kv.second->output_score +=
          output == nullptr ? 0.0f : output->output_score;
      node_queue.push(kv.second.get());
    }
  }
}

}  // namespace soundproof
