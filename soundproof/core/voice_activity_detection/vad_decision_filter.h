#ifndef SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_VAD_DECISION_FILTER_H_
#define SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_VAD_DECISION_FILTER_H_

#include <cstdint>

namespace soundproof {

// Decision smoothing shared by all VAD backends (WebRTC DSP and Silero).
//
// This is a port of the speech state machine in sherpa-mnn's
// SileroVadModel::IsSpeech:
//   - speech onset is debounced by |min_speech_duration|;
//   - once triggered, a probability above threshold - |hysteresis| keeps the
//     speech state (boundary jitter suppression);
//   - after the probability drops, the speech state is held for
//     |min_silence_duration| (hangover) before it is released.
//
// The caller feeds a fresh probability together with the number of new audio
// samples it covers:
//   - WebRTC DSP produces one probability per 10 ms chunk  -> elapsed = 10 ms;
//   - Silero produces one probability per 512-sample window -> elapsed = 512.
class VadDecisionFilter {
 public:
  struct Config {
    float threshold = 0.5f;
    float hysteresis = 0.15f;
    float min_speech_duration = 0.25f;   // seconds
    float min_silence_duration = 0.5f;   // seconds (hangover)
    int sample_rate = 16000;
  };

  VadDecisionFilter();
  explicit VadDecisionFilter(const Config& config);

  // Recomputes the sample-count thresholds. The current state is kept, so it
  // is safe to call while streaming.
  void SetConfig(const Config& config);
  const Config& config() const { return config_; }

  // Advances the state machine with a fresh probability covering
  // |elapsed_samples| new audio samples. Returns the smoothed decision.
  bool Update(float probability, int64_t elapsed_samples);

  bool is_speech() const { return is_speech_; }

  void Reset();

 private:
  Config config_;
  int64_t min_speech_samples_ = 0;
  int64_t min_silence_samples_ = 0;
  int64_t current_sample_ = 0;
  int64_t temp_start_ = 0;
  int64_t temp_end_ = 0;
  bool triggered_ = false;
  bool is_speech_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_VAD_DECISION_FILTER_H_
