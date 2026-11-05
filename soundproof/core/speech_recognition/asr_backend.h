#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_ASR_BACKEND_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_ASR_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {

class SymbolTable;

// Result produced by a backend for the audio decoded so far.
struct AsrResult {
  // Token ids. For transducers the first ContextSize tokens are the leading
  // blanks used as decoder context and are stripped when building text.
  std::vector<int32_t> tokens;
  // Encoder frame index at which each token was emitted (transducer only).
  std::vector<int32_t> timestamps;
  // Trailing silence used by endpointing, in 10 ms frames.
  int32_t trailing_silence_frames = 0;
};

struct AsrLmConfig {
  // 0 = greedy_search, 1 = modified_beam_search.
  int decoding_method = 0;
  int max_active_paths = 4;
  std::string model_path;
  float scale = 0.5f;
  bool shallow_fusion = true;
  int num_threads = 1;
};

struct AsrBackendConfig {
  // Empty means "detect from encoder metadata / paths".
  std::string model_type;
  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  int num_threads = 1;
  float blank_penalty = 0.0f;
  // Id of "<unk>" from tokens.txt, -1 if absent.
  int32_t unk_id = -1;
  AsrLmConfig lm;
};

// A streaming ASR model backend. The recognizer front-end owns feature
// extraction, chunking, endpointing and text rendering; the backend owns the
// model-specific acoustic/decoding state.
class AsrBackend {
 public:
  virtual ~AsrBackend() = default;

  virtual bool Init(const AsrBackendConfig& config) = 0;

  virtual const char* ModelType() const = 0;

  virtual int32_t FeatureDim() const = 0;

  // True if int16 samples should be scaled to [-1, 1] before feature
  // extraction (transducer). False means raw int16 magnitude is expected
  // (paraformer).
  virtual bool NormalizeSamples() const = 0;

  // Number of feature frames consumed per decode step and how far the
  // processed-frame counter advances after it.
  virtual int32_t ChunkSize() const = 0;
  virtual int32_t ChunkShift() const = 0;

  // Decodes one chunk of ChunkSize() feature frames (row-major,
  // ChunkSize() * FeatureDim() floats). |processed_frames| is the number of
  // frames processed before this chunk (used by paraformer for endpointing).
  virtual bool DecodeChunk(const float* features, int32_t processed_frames,
                           AsrResult* result) = 0;

  // Renders |tokens| as UTF-8 text using model-specific rules.
  virtual std::string TextFromTokens(const std::vector<int32_t>& tokens,
                                     const SymbolTable& symbols) const = 0;

  // Called after an endpoint: reset decoding state for a new segment while
  // keeping whatever context the model wants to carry over.
  virtual void ResetSegment() = 0;

  // Full reset (new stream).
  virtual void Reset() = 0;

  // Returns false if the backend does not support runtime blank penalty.
  virtual bool SetBlankPenalty(float /*blank_penalty*/) { return false; }
};

// Creates a backend based on config.model_type when set, otherwise based on
// the available model paths (joiner present -> transducer, else paraformer).
// The chosen backend validates the encoder metadata during Init().
std::unique_ptr<AsrBackend> CreateAsrBackend(const AsrBackendConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_ASR_BACKEND_H_
