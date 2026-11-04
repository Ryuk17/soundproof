#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RECOGNIZER_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RECOGNIZER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "feature/feature_extractor.h"
#include "core/speech_recognition/asr_backend.h"
#include "core/speech_recognition/endpoint.h"
#include "utils/symbol_table.h"

namespace soundproof {

struct AsrRecognizerConfig {
  AsrBackendConfig backend;
  std::string tokens_path;
  bool enable_endpoint = true;
};

struct AsrRecognizerResult {
  // UTF-8 text decoded so far (or the finalized segment when is_final).
  std::string text;
  // True when this call crossed an endpoint; text is the final segment text.
  bool is_final = false;
  // Segment index (incremented for each finalized non-empty segment).
  int32_t segment = 0;
};

// Streaming speech recognizer front-end shared by all model backends.
// Input: mono 16 kHz int16 PCM, 10 ms chunks recommended (any size works).
class OnlineRecognizer {
 public:
  OnlineRecognizer();
  ~OnlineRecognizer();

  OnlineRecognizer(const OnlineRecognizer&) = delete;
  OnlineRecognizer& operator=(const OnlineRecognizer&) = delete;

  bool Init(const AsrRecognizerConfig& config);
  bool ok() const { return initialized_; }

  // Feeds audio and returns the current partial text, or the finalized text
  // when an endpoint was reached in this call.
  bool Process(const int16_t* audio_in, size_t samples,
               AsrRecognizerResult* result);

  // Clears the stream (features, model state, segment counter).
  void Reset();

  // Runtime tuning; only supported by transducer backends.
  bool SetBlankPenalty(float blank_penalty);

 private:
  std::unique_ptr<AsrBackend> backend_;
  FeatureExtractor features_;
  SymbolTable symbols_;
  Endpoint endpoint_;
  AsrResult current_;
  int32_t start_frame_index_ = 0;
  int32_t num_processed_frames_ = 0;
  int32_t segment_ = 0;
  bool enable_endpoint_ = true;
  bool initialized_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_ONLINE_RECOGNIZER_H_
