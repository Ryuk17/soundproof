#ifndef SOUNDPROOF_MNN_ENGINE_MNN_LLM_ENGINE_H_
#define SOUNDPROOF_MNN_ENGINE_MNN_LLM_ENGINE_H_

#include <memory>
#include <string>

#include "llm/llm.hpp"

namespace soundproof {
namespace mnn_engine {

enum class LlmTask {
  kChat = 0,
  kEmbedding = 1,
};

struct LlmEngineConfig {
  // Path to the MNN-LLM config.json (model files are resolved relative to it).
  std::string config_path;
  LlmTask task = LlmTask::kChat;
  // 0 = keep the thread_num from config.json; > 0 overrides text (and vision
  // for kChat) threads.
  int num_threads = 0;
  // kChat only: render the chat template manually in the caller. Used when the
  // caller needs full ChatMessages control (history, images).
  bool disable_template = false;
};

// Shared MNN-LLM lifecycle wrapper used by RAG (Embedding) and LLM (chat).
//
// It owns one MNN::Transformer::Llm / Embedding instance and centralizes the
// config path validation, thread overrides and load() call. Inference stays in
// the algorithm modules, so this file has no dependency on MNN vision/OpenCV.
class LlmEngine {
 public:
  LlmEngine();
  ~LlmEngine();

  LlmEngine(const LlmEngine&) = delete;
  LlmEngine& operator=(const LlmEngine&) = delete;

  // Creates and loads the model. Any previously loaded model is released.
  // Returns false on failure; the reason is logged with MNN_ERROR.
  bool Init(const LlmEngineConfig& config);

  // Releases the model. Idempotent; the engine may be Init-ed again.
  void Deinit();

  bool initialized() const;

  // Non-owning pointers, valid until the next Init/Deinit.
  MNN::Transformer::Llm* model();
  // Non-null only for LlmTask::kEmbedding.
  MNN::Transformer::Embedding* embedding();

 private:
  std::unique_ptr<MNN::Transformer::Llm> model_;
  MNN::Transformer::Embedding* embedding_ = nullptr;
  LlmTask task_ = LlmTask::kChat;
};

}  // namespace mnn_engine
}  // namespace soundproof

#endif  // SOUNDPROOF_MNN_ENGINE_MNN_LLM_ENGINE_H_
