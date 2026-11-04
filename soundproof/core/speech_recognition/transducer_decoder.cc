#include "core/speech_recognition/transducer_decoder.h"

#include <algorithm>
#include <limits>

#include "model/online_transducer_model.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int32_t kBlankId = 0;

std::vector<int32_t> MakeBlankContext(int32_t context_size) {
  std::vector<int32_t> tokens(static_cast<size_t>(context_size), -1);
  tokens.back() = kBlankId;
  return tokens;
}

}  // namespace

TransducerDecoder::TransducerDecoder(OnlineTransducerModel* model,
                                     int32_t unk_id, float blank_penalty)
    : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {
  Reset();
}

void TransducerDecoder::Reset() {
  tokens_ = MakeBlankContext(model_->ContextSize());
  timestamps_.clear();
  decoder_out_.clear();
  decoder_out_valid_ = false;
  frame_offset_ = 0;
  num_trailing_blanks_ = 0;
}

void TransducerDecoder::ResetSegment() {
  const int32_t context_size = model_->ContextSize();
  if (static_cast<int32_t>(tokens_.size()) > context_size) {
    std::vector<int32_t> context(tokens_.end() - context_size, tokens_.end());
    tokens_ = std::move(context);
  } else {
    tokens_ = MakeBlankContext(context_size);
  }
  timestamps_.clear();
  decoder_out_.clear();
  decoder_out_valid_ = false;
  frame_offset_ = 0;
  num_trailing_blanks_ = 0;
}

std::vector<int32_t> TransducerDecoder::ContextTokens() const {
  const int32_t context_size = model_->ContextSize();
  if (static_cast<int32_t>(tokens_.size()) < context_size) {
    return MakeBlankContext(context_size);
  }
  return std::vector<int32_t>(tokens_.end() - context_size, tokens_.end());
}

bool TransducerDecoder::Decode(const float* encoder_out, int32_t num_frames,
                               int32_t frame_dim) {
  if (encoder_out == nullptr || num_frames < 0 || frame_dim <= 0) {
    return false;
  }

  if (!decoder_out_valid_) {
    if (!model_->RunDecoder(ContextTokens(), &decoder_out_)) {
      RTC_LOG(LS_ERROR) << "TransducerDecoder: decoder forward failed";
      return false;
    }
    decoder_out_valid_ = true;
  }

  const int32_t vocab_size = model_->VocabSize();
  std::vector<float> logits;

  for (int32_t t = 0; t < num_frames; ++t) {
    if (!model_->RunJoiner(encoder_out + static_cast<size_t>(t) * frame_dim,
                           decoder_out_, &logits) ||
        static_cast<int32_t>(logits.size()) != vocab_size) {
      RTC_LOG(LS_ERROR) << "TransducerDecoder: joiner forward failed";
      return false;
    }
    if (blank_penalty_ > 0.0f) {
      logits[kBlankId] -= blank_penalty_;
    }

    const int32_t token = static_cast<int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (token != kBlankId && token != unk_id_) {
      tokens_.push_back(token);
      timestamps_.push_back(t + frame_offset_);
      num_trailing_blanks_ = 0;

      // The decoder is stateless: recompute its output for the next frame.
      if (!model_->RunDecoder(ContextTokens(), &decoder_out_)) {
        RTC_LOG(LS_ERROR) << "TransducerDecoder: decoder forward failed";
        return false;
      }
      decoder_out_valid_ = true;
    } else {
      ++num_trailing_blanks_;
    }
  }

  frame_offset_ += num_frames;
  return true;
}

}  // namespace soundproof
