#include "core/speech_recognition/endpoint.h"

namespace soundproof {
namespace {

bool RuleActivated(const EndpointRule& rule, float trailing_silence,
                   float utterance_length) {
  const bool contain_nonsilence = utterance_length > trailing_silence;
  return (contain_nonsilence || !rule.must_contain_nonsilence) &&
         trailing_silence >= rule.min_trailing_silence &&
         utterance_length >= rule.min_utterance_length;
}

}  // namespace

bool Endpoint::IsEndpoint(int32_t num_frames_decoded,
                          int32_t trailing_silence_frames,
                          float frame_shift_in_seconds) const {
  const float utterance_length =
      static_cast<float>(num_frames_decoded) * frame_shift_in_seconds;
  const float trailing_silence =
      static_cast<float>(trailing_silence_frames) * frame_shift_in_seconds;

  return RuleActivated(config_.rule1, trailing_silence, utterance_length) ||
         RuleActivated(config_.rule2, trailing_silence, utterance_length) ||
         RuleActivated(config_.rule3, trailing_silence, utterance_length);
}

}  // namespace soundproof
