#include "core/llm/mnn_llm_backend.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include "cv/imgcodecs.hpp"
#include "mnn_engine/mnn_llm_engine.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string FloatToJson(float value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

std::string BuildSamplingJson(const LlmSamplingParams& params) {
  std::string json = "{";
  bool first = true;
  auto append = [&json, &first](const std::string& field) {
    if (!first) {
      json += ",";
    }
    json += field;
    first = false;
  };

  if (params.temperature == 0.0f) {
    append("\"sampler_type\":\"greedy\"");
  } else {
    // Reset greedy state left by a previous call.
    append("\"sampler_type\":\"mixed\"");
    if (params.temperature > 0.0f) {
      append("\"temperature\":" + FloatToJson(params.temperature));
    }
    if (params.top_p >= 0.0f) {
      append("\"top_p\":" + FloatToJson(params.top_p));
    }
    if (params.top_k >= 0) {
      append("\"top_k\":" + std::to_string(params.top_k));
    }
  }
  if (params.enable_thinking >= 0) {
    append(std::string("\"jinja\":{\"context\":{\"enable_thinking\":") +
           (params.enable_thinking != 0 ? "true" : "false") + "}}");
  }
  json += "}";
  return json;
}

}  // namespace

struct MnnLlmBackend::Impl {
  mnn_engine::LlmEngine engine;
};

MnnLlmBackend::MnnLlmBackend() : impl_(new Impl()) {}
MnnLlmBackend::~MnnLlmBackend() = default;

LlmStatus MnnLlmBackend::Init(const LlmModelConfig& config) {
  std::string type = ToLower(config.model_type);
  if (!type.empty() && type != "qwen3.5" && type != "qwen3.5-vl" &&
      type != "qwen3_5" && type != "qwen3.5-0.8b") {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: unsupported model type '" << type
                      << "'";
    return LlmStatus::kUnsupportedModel;
  }
  model_type_ = "qwen3.5";

  mnn_engine::LlmEngineConfig engine_config;
  engine_config.config_path = config.config_path;
  engine_config.task = mnn_engine::LlmTask::kChat;
  engine_config.num_threads = config.num_threads;
  // The engine renders ChatMessages itself for history/image support.
  engine_config.disable_template = true;
  if (!impl_->engine.Init(engine_config)) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: failed to load " << config.config_path;
    return LlmStatus::kModelLoadFailed;
  }
  if (impl_->engine.model() == nullptr) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: no chat model";
    impl_->engine.Deinit();
    return LlmStatus::kUnsupportedModel;
  }
  return LlmStatus::kOk;
}

LlmStatus MnnLlmBackend::Generate(const std::vector<LlmMessage>& messages,
                                  const LlmImage& image,
                                  const LlmSamplingParams& params,
                                  LlmReply* reply) {
  MNN::Transformer::Llm* model = impl_->engine.model();
  if (model == nullptr) {
    return LlmStatus::kNotInitialized;
  }
  if (reply == nullptr) {
    return LlmStatus::kInvalidInput;
  }
  if (messages.empty()) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: empty conversation";
    return LlmStatus::kInvalidInput;
  }

  if (!model->set_config(BuildSamplingJson(params))) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: failed to apply sampling config";
    return LlmStatus::kInferenceFailed;
  }

  MNN::Transformer::ChatMessages chat;
  chat.reserve(messages.size());
  for (const LlmMessage& message : messages) {
    chat.emplace_back(message.role, message.content);
  }

  MNN::Transformer::MultimodalPrompt multimodal;
  multimodal.prompt_template = model->apply_chat_template(chat);
  if (multimodal.prompt_template.empty()) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: empty chat prompt";
    return LlmStatus::kInferenceFailed;
  }

  // Keeps raw RGB converted to BGR alive until response() returns.
  std::vector<uint8_t> bgr;
  if (!image.empty()) {
    MNN::Express::VARP image_var;
    int width = 0;
    int height = 0;
    if (!image.path.empty()) {
      image_var = MNN::CV::imread(image.path);
      // width/height stay 0: let the model's configured image size apply.
    } else {
      const size_t expected = static_cast<size_t>(image.width) *
                              static_cast<size_t>(image.height) * 3;
      if (image.width <= 0 || image.height <= 0 ||
          image.rgb.size() != expected) {
        RTC_LOG(LS_ERROR) << "MnnLlmBackend: invalid raw image "
                          << image.width << "x" << image.height;
        return LlmStatus::kInvalidInput;
      }
      width = image.width;
      height = image.height;
      // MNN's vision front-end expects BGR (as produced by imread).
      bgr.resize(expected);
      for (size_t i = 0; i < expected; i += 3) {
        bgr[i] = image.rgb[i + 2];
        bgr[i + 1] = image.rgb[i + 1];
        bgr[i + 2] = image.rgb[i];
      }
      image_var = MNN::Express::_Const(
          bgr.data(), {height, width, 3}, MNN::Express::NHWC,
          halide_type_of<uint8_t>());
    }
    if (image_var == nullptr) {
      RTC_LOG(LS_ERROR) << "MnnLlmBackend: cannot decode image";
      return LlmStatus::kFeatureFailed;
    }
    MNN::Transformer::PromptImagePart part;
    part.image_data = image_var;
    part.width = width;
    part.height = height;
    multimodal.images["mnn_image"] = part;
  }

  std::ostringstream output;
  const int max_new_tokens =
      params.max_new_tokens > 0 ? params.max_new_tokens : -1;
  model->response(multimodal, &output, "", max_new_tokens);

  const MNN::Transformer::LlmContext* context = model->getContext();
  if (context == nullptr) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: missing model context";
    return LlmStatus::kInferenceFailed;
  }
  if (context->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR ||
      context->status == MNN::Transformer::LlmStatus::TIMEOUT ||
      context->status == MNN::Transformer::LlmStatus::USER_CANCEL) {
    RTC_LOG(LS_ERROR) << "MnnLlmBackend: generation failed (status "
                      << static_cast<int>(context->status) << ")";
    return LlmStatus::kInferenceFailed;
  }

  reply->text = output.str();
  reply->prompt_tokens = context->prompt_len;
  reply->output_tokens = context->gen_seq_len;
  return LlmStatus::kOk;
}

LlmStatus MnnLlmBackend::Reset() {
  MNN::Transformer::Llm* model = impl_->engine.model();
  if (model == nullptr) {
    return LlmStatus::kNotInitialized;
  }
  model->reset();
  return LlmStatus::kOk;
}

}  // namespace soundproof
