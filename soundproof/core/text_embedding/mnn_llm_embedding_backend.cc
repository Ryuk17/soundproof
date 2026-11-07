#include "core/text_embedding/mnn_llm_embedding_backend.h"

#include <algorithm>
#include <cctype>
#include <fstream>

#include "llm/llm.hpp"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool FileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

}  // namespace

struct MnnLlmEmbeddingBackend::Impl {
  std::unique_ptr<MNN::Transformer::Embedding> embedding;
};

MnnLlmEmbeddingBackend::MnnLlmEmbeddingBackend() : impl_(new Impl()) {}
MnnLlmEmbeddingBackend::~MnnLlmEmbeddingBackend() = default;

EmbeddingStatus MnnLlmEmbeddingBackend::Init(
    const EmbeddingModelConfig& config) {
  if (config.config_path.empty()) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: config path is empty";
    return EmbeddingStatus::kInvalidInput;
  }
  if (!FileExists(config.config_path)) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: config file not found: "
                      << config.config_path;
    return EmbeddingStatus::kModelLoadFailed;
  }

  std::string type = ToLower(config.model_type);
  if (!type.empty() && type != "mnn-llm" && type != "qwen3-embedding" &&
      type != "qwen3") {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: unsupported model type '"
                      << type << "'";
    return EmbeddingStatus::kUnsupportedModel;
  }
  model_type_ = "mnn-llm";

  std::unique_ptr<MNN::Transformer::Embedding> embedding(
      MNN::Transformer::Embedding::createEmbedding(config.config_path, false));
  if (!embedding) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: cannot create embedding from "
                      << config.config_path;
    return EmbeddingStatus::kModelLoadFailed;
  }

  // thread_num is read when load() creates the runtime, so it must be applied
  // before load(). 0 keeps the value from config.json (MNN default is 4).
  if (config.num_threads > 0) {
    const std::string overrides =
        "{\"thread_num\":" + std::to_string(config.num_threads) + "}";
    if (!embedding->set_config(overrides)) {
      RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: cannot apply thread_num="
                        << config.num_threads;
      return EmbeddingStatus::kModelLoadFailed;
    }
  }

  if (!embedding->load()) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: failed to load "
                      << config.config_path;
    return EmbeddingStatus::kModelLoadFailed;
  }

  dim_ = embedding->dim();
  if (dim_ <= 0) {
    RTC_LOG(LS_ERROR) << "MnnLlmEmbeddingBackend: invalid embedding dim "
                      << dim_;
    return EmbeddingStatus::kUnsupportedModel;
  }

  impl_->embedding = std::move(embedding);
  RTC_LOG(LS_INFO) << "MnnLlmEmbeddingBackend: loaded " << config.config_path
                   << ", dim=" << dim_;
  return EmbeddingStatus::kOk;
}

EmbeddingStatus MnnLlmEmbeddingBackend::Compute(
    const std::string& text, std::vector<float>* embedding) {
  if (embedding == nullptr) {
    return EmbeddingStatus::kInvalidInput;
  }
  if (!impl_->embedding) {
    return EmbeddingStatus::kNotInitialized;
  }
  if (text.empty()) {
    RTC_LOG(LS_WARNING) << "MnnLlmEmbeddingBackend: empty text";
    return EmbeddingStatus::kInvalidInput;
  }

  MNN::Express::VARP output = impl_->embedding->txt_embedding(text);
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
