#ifndef SOUNDPROOF_CORE_LLM_LLM_ENGINE_H_
#define SOUNDPROOF_CORE_LLM_LLM_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "core/llm/llm_backend.h"

namespace soundproof {

// Chat front-end: owns the backend, keeps the conversation history, renders
// the current user turn (text + optional image) and strips thinking blocks.
class LlmEngine {
 public:
  LlmEngine();
  ~LlmEngine();

  LlmEngine(const LlmEngine&) = delete;
  LlmEngine& operator=(const LlmEngine&) = delete;

  LlmStatus Init(const LlmModelConfig& config);
  bool ok() const { return ok_; }

  // Generates one assistant reply. When |reset_history| is true the previous
  // conversation is cleared first. The image applies to this turn only; the
  // stored history keeps the user text without the image placeholder.
  LlmStatus Generate(const std::string& text, const LlmImage& image,
                     bool reset_history, const LlmSamplingParams& params,
                     bool strip_thinking);

  // Engine-owned result, valid until the next Generate/Reset/Init.
  const std::string& last_text() const { return last_reply_.text; }
  int prompt_tokens() const { return last_reply_.prompt_tokens; }
  int output_tokens() const { return last_reply_.output_tokens; }

  // Clears the conversation and the last reply; the model stays loaded.
  void Reset();

 private:
  std::unique_ptr<LlmBackend> backend_;
  std::vector<LlmMessage> history_;
  std::string system_prompt_;
  LlmReply last_reply_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_LLM_LLM_ENGINE_H_
