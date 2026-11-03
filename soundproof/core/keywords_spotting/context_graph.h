#ifndef SOUNDPROOF_CORE_KEYWORDS_SPOTTING_CONTEXT_GRAPH_H_
#define SOUNDPROOF_CORE_KEYWORDS_SPOTTING_CONTEXT_GRAPH_H_

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace soundproof {

// A node of the keyword context trie (port of sherpa-mnn's ContextState).
struct ContextState {
  int32_t token = -1;
  float token_score = 0.0f;
  float node_score = 0.0f;
  float output_score = 0.0f;
  int32_t level = 0;
  float ac_threshold = 0.0f;
  bool is_end = false;
  std::string phrase;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState* fail = nullptr;
  const ContextState* output = nullptr;

  ContextState() = default;
  ContextState(int32_t token, float token_score, float node_score,
               float output_score, int32_t level, float ac_threshold,
               bool is_end, std::string phrase)
      : token(token),
        token_score(token_score),
        node_score(node_score),
        output_score(output_score),
        level(level),
        ac_threshold(ac_threshold),
        is_end(is_end),
        phrase(std::move(phrase)) {}
};

// Keyword trie with Aho-Corasick style fail/output links, ported from
// sherpa-mnn's ContextGraph. Token sequences are matched exactly; scoring
// follows the sherpa implementation.
class ContextGraph {
 public:
  ContextGraph() = default;
  ContextGraph(const std::vector<std::vector<int32_t>>& token_ids,
               float context_score, float ac_threshold,
               const std::vector<float>& scores = {},
               const std::vector<std::string>& phrases = {},
               const std::vector<float>& ac_thresholds = {});

  std::tuple<float, const ContextState*, const ContextState*> ForwardOneStep(
      const ContextState* state, int32_t token) const;

  std::pair<bool, const ContextState*> IsMatched(
      const ContextState* state) const;

  std::pair<float, const ContextState*> Finalize(
      const ContextState* state) const;

  const ContextState* Root() const { return root_.get(); }

 private:
  void Build(const std::vector<std::vector<int32_t>>& token_ids,
             const std::vector<float>& scores,
             const std::vector<std::string>& phrases,
             const std::vector<float>& ac_thresholds) const;
  void FillFailOutput() const;

  float context_score_ = 0.0f;
  float ac_threshold_ = 0.0f;
  std::unique_ptr<ContextState> root_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_KEYWORDS_SPOTTING_CONTEXT_GRAPH_H_
