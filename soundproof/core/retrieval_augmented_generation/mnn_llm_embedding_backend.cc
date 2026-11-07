#include "core/retrieval_augmented_generation/mnn_llm_embedding_backend.h"

#include <algorithm>
#include <cctype>

#include "mnn_engine/mnn_llm_engine.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

struct MnnLlmEmbeddingBackend::Impl {
  mnn_engine::LlmEngine engine;
};

MnnLlmEmbeddingBackend::MnnLlmEmbeddingBackend() : impl_(new Impl()) {}
MnnLlmEmbeddingBackend::~MnnLlmEmbeddingBackend() = default;

EmbeddingStatus MnnLlmEmbeddingBackend::Init(
    const EmbeddingModelConfig& config) {
  std::string type = ToLower(config.model_type);
  if (!type.empty() && type != "mnn-llm" && type != "qwen3-embedding" &&
      type != "qwen3") {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: unsupported model type '"
                      << type << "'";
    return EmbeddingStatus::kUnsupportedModel;
  }
  model_type_ = "mnn-llm";

  mnn_engine::LlmEngineConfig engine_config;
  engine_config.config_path = config.config_path;
  engine_config.task = mnn_engine::LlmTask::kEmbedding;
  engine_config.num_threads = config.num_threads;
  if (!impl_->engine.Init(engine_config)) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: failed to load "
                      << config.config_path;
    return EmbeddingStatus::kModelLoadFailed;
  }

  MNN::Transformer::Embedding* embedding = impl_->engine.embedding();
  if (embedding == nullptr) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: no embedding model";
    impl_->engine.Deinit();
    return EmbeddingStatus::kUnsupportedModel;
  }
  dim_ = embedding->dim();
  if (dim_ <= 0) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: invalid embedding dim "
                      << dim_;
    impl_->engine.Deinit();
    return EmbeddingStatus::kUnsupportedModel;
  }

  RTC_LOG(LS_INFO) << "MnnLlmEmbeddingBackend: loaded " << config.config_path
                   << ", dim=" << dim_;
  return EmbeddingStatus::kOk;
}

EmbeddingStatus MnnLlmEmbeddingBackend::Compute(
    const std::string& text, std::vector<float>* embedding) {
  if (embedding == nullptr) {
    return EmbeddingStatus::kInvalidInput;
  }
  if (!impl_->engine.initialized() || impl_->engine.embedding() == nullptr) {
    return EmbeddingStatus::kNotInitialized;
  }
  if (text.empty()) {
    RTC_LOG(LS_WARNING) << "MnnLlmEmbeddingBackend: empty text";
    return EmbeddingStatus::kInvalidInput;
  }

  MNN::Express::VARP output = impl_->engine.embedding()->txt_embedding(text);
  if (output == nullptr) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: txt_embedding failed";
    return EmbeddingStatus::kInferenceFailed;
  }

  // readMap first: it triggers the graph evaluation (same order as MNN's
  // embedding_demo). getInfo() alone does not guarantee the data is ready.
  const float* data = output->readMap<float>();
  if (data == nullptr) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: cannot read embedding output";
    return EmbeddingStatus::kInferenceFailed;
  }

  const MNN::Express::Variable::Info* info = output->getInfo();
  if (info == nullptr || info->size == 0) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: invalid embedding output";
    return EmbeddingStatus::kInferenceFailed;
  }
  if (info->type.code != halide_type_float || info->type.bits != 32) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: unexpected output type";
    return EmbeddingStatus::kUnsupportedModel;
  }
  if (dim_ <= 0 || info->size != static_cast<size_t>(dim_)) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: output size " << info->size
                      << " does not match dim " << dim_;
    return EmbeddingStatus::kEmbeddingDimMismatch;
  }

  embedding->assign(data, data + dim_);
  return EmbeddingStatus::kOk;
}

}  // namespace soundproof
