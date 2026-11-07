#ifndef SOUNDPROOF_CORE_TURN_DETECTION_TURN_DETECTOR_H_
#define SOUNDPROOF_CORE_TURN_DETECTION_TURN_DETECTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "core/turn_detection/turn_backend.h"

namespace soundproof {

// Streaming turn detector: keeps the most recent ContextSamples() of 16 kHz
// mono PCM in a ring buffer and runs the backend on demand. Short buffers are
// left-padded with zeros, matching the Smart Turn reference preprocessing.
class TurnDetector {
 public:
  TurnDetector();
  ~TurnDetector();

  TurnDetector(const TurnDetector&) = delete;
  TurnDetector& operator=(const TurnDetector&) = delete;

  TurnStatus Init(const TurnModelConfig& config);
  bool ok() const { return ok_; }

  // Appends PCM16 samples; audio may be null only when samples == 0.
  TurnStatus Append(const short* audio, int samples);

  // Runs the backend on the buffered audio (cached until new audio arrives).
  // Returns kInvalidInput when the buffer is empty.
  TurnStatus Predict(float* probability);

  int32_t ContextSamples() const;
  int32_t BufferedSamples() const;

  // Clears the buffer and the cached prediction; the model stays loaded.
  void Reset();

 private:
  std::unique_ptr<TurnBackend> backend_;
  std::vector<float> ring_;
  size_t write_pos_ = 0;
  size_t size_ = 0;
  std::vector<float> linear_;
  float last_probability_ = 0.0f;
  bool has_last_ = false;
  bool dirty_ = true;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_TURN_DETECTION_TURN_DETECTOR_H_
