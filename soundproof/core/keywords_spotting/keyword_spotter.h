#ifndef SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_SPOTTER_H_
#define SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_SPOTTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "core/keywords_spotting/context_graph.h"
#include "feature/feature_extractor.h"
#include "core/keywords_spotting/keyword_encoder.h"
#include "model/online_transducer_model.h"
#include "utils/symbol_table.h"

namespace soundproof {

class KeywordDecoder;

struct KeywordSpotterConfig {
  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  std::string tokens_path;
  std::string keywords_file_path;
  int num_threads = 1;
  float keywords_threshold = 0.25f;

  // Fixed to sherpa-mnn's defaults for zipformer2 KWS.
  int32_t max_active_paths = 4;
  int32_t num_trailing_blanks = 1;
  float keywords_score = 1.0f;
};

// Streaming keyword spotter built on the zipformer2 transducer model.
// Input: 16 kHz mono int16 PCM, any chunk size (the unified interface sends
// 10 ms frames). One instance belongs to one audio stream.
class KeywordSpotter {
 public:
  KeywordSpotter();
  ~KeywordSpotter();

  KeywordSpotter(const KeywordSpotter&) = delete;
  KeywordSpotter& operator=(const KeywordSpotter&) = delete;

  bool Init(const KeywordSpotterConfig& config);
  bool ok() const { return initialized_; }

  // Feeds audio. On success |keyword_index| is the 0-based index of the
  // triggered keyword in keywords.txt (-1 if nothing triggered), and
  // |keyword_score| is the matched keyword's average posterior (0 when no
  // trigger).
  bool Process(const int16_t* audio_in, size_t samples, int* keyword_index,
               float* keyword_score);

  // Rebuilds the context graph with a new trigger threshold and resets the
  // decoder search state. Valid range is (0, 1).
  bool SetThreshold(float threshold);

  // Clears buffered audio, encoder state and decoder hypotheses.
  void Reset();

  size_t num_keywords() const { return keywords_.size(); }
  const std::string& keyword(size_t index) const;

 private:
  int FindKeywordIndex(const struct KeywordHit& hit) const;

  KeywordSpotterConfig config_;
  FeatureExtractor features_;
  SymbolTable symbols_;
  KeywordList keywords_;
  ContextGraph graph_;
  OnlineTransducerModel model_;
  std::unique_ptr<KeywordDecoder> decoder_;
  int32_t processed_frames_ = 0;
  bool initialized_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_SPOTTER_H_
