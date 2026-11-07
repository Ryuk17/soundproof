#ifndef SOUNDPROOF_CORE_LLM_LLM_BACKEND_H_
#define SOUNDPROOF_CORE_LLM_LLM_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {

enum class LlmStatus {
  kOk = 0,
  kNotInitialized,
  kModelLoadFailed,
  kUnsupportedModel,
  kInvalidInput,
  kFeatureFailed,
  kInferenceFailed,
};

struct LlmModelConfig {
  // Empty means "detect from config" (v1: qwen3.5).
  std::string model_type;
  // Path to the MNN-LLM config.json.
  std::string config_path;
  // 0 = keep the thread_num from config.json; > 0 overrides it.
  int num_threads = 0;
  // Optional system prompt, kept at the front of every conversation.
  std::string system_prompt;
};

struct LlmSamplingParams {
  // 0 = use max_new_tokens from config.json.
  int max_new_tokens = 512;
  // < 0 = config default, 0 = greedy, > 0 = temperature sampling.
  float temperature = -1.0f;
  // < 0 = config default.
  float top_p = -1.0f;
  // < 0 = config default.
  int top_k = -1;
  // -1 = config default, 0/1 = override jinja enable_thinking.
  int enable_thinking = -1;
};

struct LlmMessage {
  std::string role;
  std::string content;
};

// Image for the current turn. Exactly one of |path| or |rgb| is expected.
struct LlmImage {
  std::string path;
  std::vector<uint8_t> rgb;  // raw RGB8, width * height * 3
  int width = 0;
  int height = 0;

  bool empty() const { return path.empty() && rgb.empty(); }
};

struct LlmReply {
  std::string text;
  int prompt_tokens = 0;
  int output_tokens = 0;
};

// A chat LLM backend. Implementations own the model; the engine owns the
// conversation history and thinking-strip policy.
class LlmBackend {
 public:
  virtual ~LlmBackend() = default;

  virtual LlmStatus Init(const LlmModelConfig& config) = 0;

  virtual const char* ModelType() const = 0;

  // |messages| is the full conversation including the current user turn. When
  // |image| is set, the current user content must contain the placeholder
  // "<img>mnn_image</img>".
  virtual LlmStatus Generate(const std::vector<LlmMessage>& messages,
                             const LlmImage& image,
                             const LlmSamplingParams& params,
                             LlmReply* reply) = 0;

  // Clears the model-side KV/history state.
  virtual LlmStatus Reset() = 0;
};

// Creates a backend from config.model_type (empty -> qwen3.5).
std::unique_ptr<LlmBackend> CreateLlmBackend(const LlmModelConfig& config);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_LLM_LLM_BACKEND_H_
