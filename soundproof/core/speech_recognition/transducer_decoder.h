#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_DECODER_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_DECODER_H_

#include <cstdint>
#include <vector>

namespace soundproof {

class OnlineTransducerModel;

// Greedy-search transducer decoder for a single stream (port of sherpa-mnn's
// OnlineTransducerGreedySearchDecoder).
class TransducerDecoder {
 public:
  TransducerDecoder(OnlineTransducerModel* model, int32_t unk_id,
                    float blank_penalty);

  void SetBlankPenalty(float blank_penalty) { blank_penalty_ = blank_penalty; }

  // Full reset: tokens become the leading blanks, counters clear.
  void Reset();

  // Endpoint reset: keep the last ContextSize() tokens as decoder context.
  void ResetSegment();

  // Decodes |num_frames| encoder output frames (row-major,
  // |num_frames| * frame_dim floats) and appends emitted tokens.
  bool Decode(const float* encoder_out, int32_t num_frames, int32_t frame_dim);

  // Token ids including the leading context tokens.
  const std::vector<int32_t>& tokens() const { return tokens_; }
  const std::vector<int32_t>& timestamps() const { return timestamps_; }
  int32_t num_trailing_blanks() const { return num_trailing_blanks_; }

 private:
  std::vector<int32_t> ContextTokens() const;

  OnlineTransducerModel* model_;
  int32_t unk_id_;
  float blank_penalty_;
  std::vector<int32_t> tokens_;
  std::vector<int32_t> timestamps_;
  std::vector<float> decoder_out_;
  int32_t frame_offset_ = 0;
  int32_t num_trailing_blanks_ = 0;
  bool decoder_out_valid_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_TRANSDUCER_DECODER_H_
