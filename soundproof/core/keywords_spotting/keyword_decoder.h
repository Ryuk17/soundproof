#ifndef SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_DECODER_H_
#define SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_DECODER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soundproof {

class ContextGraph;
struct ContextState;
class OnlineTransducerModel;

struct KeywordHit {
  std::string phrase;
  std::vector<int32_t> tokens;
  std::vector<int32_t> timestamps;
  // Average posterior probability of the matched keyword tokens.
  float score = 0.0f;
};

struct Hypothesis {
  std::vector<int32_t> ys;
  std::vector<int32_t> timestamps;
  std::vector<float> ys_probs;
  double log_prob = 0.0;
  const ContextState* context_state = nullptr;
  int32_t num_trailing_blanks = 0;

  std::string Key() const;
};

class Hypotheses {
 public:
  void Add(Hypothesis hyp);
  Hypothesis GetMostProbable() const;
  std::vector<Hypothesis> Vec() const;
  size_t Size() const { return hyps_.size(); }
  bool Empty() const { return hyps_.empty(); }
  void Clear() { hyps_.clear(); }

 private:
  std::unordered_map<std::string, Hypothesis> hyps_;
};

// Port of sherpa-mnn's TransducerKeywordDecoder for a single stream.
//
// Differences from upstream: decoder/joiner are invoked once per hypothesis
// (the converted models have a fixed batch dimension of 1), which is
// mathematically equivalent to the batched search.
class KeywordDecoder {
 public:
  KeywordDecoder(OnlineTransducerModel* model, int32_t max_active_paths,
                 int32_t num_trailing_blanks, int32_t unk_id);

  // Resets the search to the initial blank hypothesis at the graph root.
  void Reset(const ContextGraph& graph);

  // Decodes |num_frames| frames of encoder output (row-major,
  // |num_frames| * |frame_dim| floats). Returns true if a keyword triggered;
  // |hit| is filled in that case.
  bool Decode(const ContextGraph& graph, const float* encoder_out,
              int32_t num_frames, int32_t frame_dim, KeywordHit* hit);

 private:
  OnlineTransducerModel* model_;
  int32_t max_active_paths_;
  int32_t num_trailing_blanks_;
  int32_t unk_id_;
  int32_t frame_offset_ = 0;
  std::vector<int32_t> blanks_;
  Hypotheses hyps_;
  bool initialized_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_DECODER_H_
