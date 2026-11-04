#ifndef SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_SILERO_VAD_H_
#define SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_SILERO_VAD_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mnn_engine/mnn_engine.h"

namespace soundproof {

class VadDecisionFilter;

// Streaming Silero VAD backed by the MNN inference engine.
//
// Input: mono 16 kHz int16 PCM, any chunk size (the unified VAD interface
// feeds 10 ms chunks). Internally the samples are converted to float in
// [-1, 1], accumulated into the model window (v5: 576 samples, shift 512;
// v4: 512 samples, shift 512) and the LSTM state is carried between windows.
//
// One instance belongs to one audio stream.
class SileroVad {
 public:
  explicit SileroVad(const std::string& model_path, int num_threads = 1);
  ~SileroVad();

  SileroVad(const SileroVad&) = delete;
  SileroVad& operator=(const SileroVad&) = delete;

  bool ok() const { return ok_; }

  // Feeds |samples| int16 samples at 16 kHz. Returns false on failure.
  // Until a full model window has been accumulated, the probability stays
  // at its last value (0 after construction/reset).
  // If |decision_filter| is non-null it is advanced once per completed model
  // window (window_shift samples), so the shared post-processing sees Silero
  // decisions at the model's real cadence rather than at call cadence.
  bool ProcessChunk(const int16_t* audio_in, size_t samples, int sample_rate_hz,
                    VadDecisionFilter* decision_filter = nullptr);

  // Probability of the most recent model window, in [0, 1].
  float last_voice_probability() const { return last_probability_; }

  // Clears the accumulated audio and the LSTM state. Keeps the loaded model.
  void Reset();

 private:
  bool Init(const std::string& model_path, int num_threads);
  bool RunWindow(const float* window);

  mnn_engine::Engine engine_;

  // Inference inputs in the model's input order. "input" and the state
  // tensors are updated in place; "sr" is constant.
  std::vector<mnn_engine::Tensor> forward_inputs_;
  std::vector<mnn_engine::Tensor> state_templates_;
  std::vector<size_t> state_indices_;
  size_t input_index_ = 0;
  size_t sample_rate_index_ = 0;

  std::vector<float> pending_;
  size_t window_size_ = 0;
  size_t window_shift_ = 0;
  float last_probability_ = 0.0f;
  bool is_v5_ = false;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_VOICE_ACTIVITY_DETECTION_SILERO_VAD_H_
