#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BEAM_SEARCH_DECODER_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BEAM_SEARCH_DECODER_H_

#include <cstdint>
#include <vector>

#include "core/speech_recognition/hypothesis.h"

namespace soundproof {

class OnlineRnnLm;
class OnlineTransducerModel;

// Single-stream modified beam search for transducer models, with optional
// RNN LM shallow fusion or classic rescoring (port of sherpa-mnn's
// OnlineTransducerModifiedBeamSearchDecoder).
class TransducerBeamSearchDecoder {
 public:
  TransducerBeamSearchDecoder(OnlineTransducerModel* model, OnlineRnnLm* lm,
                              int32_t max_active_paths, float lm_scale,
                              bool shallow_fusion, int32_t unk_id,
                              float blank_penalty,
                              float temperature_scale = 2.0f);

  void SetBlankPenalty(float blank_penalty) { blank_penalty_ = blank_penalty; }

  void Reset();
  void ResetSegment();

  bool Decode(const float* encoder_out, int32_t num_frames, int32_t frame_dim);

  // Full token ids (including the leading context tokens).
  const std::vector<int32_t>& tokens() const { return tokens_; }
  const std::vector<int32_t>& timestamps() const { return timestamps_; }
  int32_t num_trailing_blanks() const { return num_trailing_blanks_; }

 private:
  std::vector<int32_t> ContextTokens(const std::vector<int32_t>& ys) const;

  OnlineTransducerModel* model_;
  OnlineRnnLm* lm_;  // may be nullptr
  int32_t max_active_paths_;
  int32_t unk_id_;
  float lm_scale_;
  bool shallow_fusion_;
  float blank_penalty_;
  float temperature_scale_;

  Hypotheses hyps_;
  std::vector<int32_t> tokens_;
  std::vector<int32_t> timestamps_;
  int32_t num_trailing_blanks_ = 0;
  int32_t frame_offset_ = 0;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_BEAM_SEARCH_DECODER_H_
