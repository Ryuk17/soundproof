#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_ENDPOINT_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_ENDPOINT_H_

#include <cstdint>

namespace soundproof {

struct EndpointRule {
  // If true, the rule requires non-silence in the decoded traceback.
  bool must_contain_nonsilence = true;
  // Minimum trailing silence, in seconds.
  float min_trailing_silence = 2.0f;
  // Minimum utterance length, in seconds.
  float min_utterance_length = 0.0f;
};

struct EndpointConfig {
  // sherpa-mnn defaults:
  //   rule1: 2.4 s of silence even if nothing was decoded
  //   rule2: 1.2 s of silence after decoding something
  //   rule3: utterance reaches 20 s
  EndpointRule rule1{false, 2.4f, 0.0f};
  EndpointRule rule2{true, 1.2f, 0.0f};
  EndpointRule rule3{false, 0.0f, 20.0f};
};

class Endpoint {
 public:
  explicit Endpoint(const EndpointConfig& config = EndpointConfig())
      : config_(config) {}

  // |num_frames_decoded|: processed 10 ms frames; |trailing_silence_frames|:
  // trailing silence in the same unit; |frame_shift_in_seconds| usually 0.01.
  bool IsEndpoint(int32_t num_frames_decoded, int32_t trailing_silence_frames,
                  float frame_shift_in_seconds) const;

 private:
  EndpointConfig config_;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_ENDPOINT_H_
