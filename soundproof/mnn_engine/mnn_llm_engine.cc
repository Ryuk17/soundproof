#include "mnn_engine/mnn_llm_engine.h"

#include <fstream>

namespace soundproof {
namespace mnn_engine {
namespace {

bool FileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

bool ApplyConfig(MNN::Transformer::Llm* model, LlmTask task,
                 const LlmEngineConfig& config) {
  const bool need_threads = config.num_threads > 0;
  const bool need_template =
      task == LlmTask::kChat && config.disable_template;
  if (!need_threads && !need_template) {
    return true;
  }

  std::string json = "{";
  bool first = true;
  auto append = [&json, &first](const std::string& field) {
    if (!first) {
      json += ",";
    }
    json += field;
    first = false;
  };
  if (need_threads) {
    const std::string threads = std::to_string(config.num_threads);
    append("\"thread_num\":" + threads);
    if (task == LlmTask::kChat) {
      append("\"mllm\":{\"thread_num\":" + threads + "}");
    }
  }
  if (need_template) {
    append("\"use_template\":false");
  }
  json += "}";

  if (!model->set_config(json)) {
    MNN_ERROR("[mnn_llm_engine] failed to apply config: %s\n", json.c_str());
    return false;
  }
  return true;
}

}  // namespace

LlmEngine::LlmEngine() = default;

LlmEngine::~LlmEngine() {
  Deinit();
}

bool LlmEngine::Init(const LlmEngineConfig& config) {
  Deinit();

  if (config.config_path.empty()) {
    MNN_ERROR("[mnn_llm_engine] config path is empty\n");
    return false;
  }
  if (!FileExists(config.config_path)) {
    MNN_ERROR("[mnn_llm_engine] config file not found: %s\n",
              config.config_path.c_str());
    return false;
  }

  task_ = config.task;
  switch (task_) {
    case LlmTask::kEmbedding:
      model_.reset(
          MNN::Transformer::Embedding::createEmbedding(config.config_path,
                                                       /*load=*/false));
      embedding_ = static_cast<MNN::Transformer::Embedding*>(model_.get());
      break;
    case LlmTask::kChat:
      model_.reset(MNN::Transformer::Llm::createLLM(config.config_path));
      break;
    default:
      MNN_ERROR("[mnn_llm_engine] unsupported task\n");
      return false;
  }
  if (!model_) {
    MNN_ERROR("[mnn_llm_engine] cannot create model from %s\n",
              config.config_path.c_str());
    return false;
  }

  if (!ApplyConfig(model_.get(), task_, config)) {
    Deinit();
    return false;
  }
  if (!model_->load()) {
    MNN_ERROR("[mnn_llm_engine] failed to load %s\n",
              config.config_path.c_str());
    Deinit();
    return false;
  }
  return true;
}

void LlmEngine::Deinit() {
  embedding_ = nullptr;
  model_.reset();
}

bool LlmEngine::initialized() const {
  return model_ != nullptr;
}

MNN::Transformer::Llm* LlmEngine::model() {
  return model_.get();
}

MNN::Transformer::Embedding* LlmEngine::embedding() {
  return embedding_;
}

}  // namespace mnn_engine
}  // namespace soundproof
