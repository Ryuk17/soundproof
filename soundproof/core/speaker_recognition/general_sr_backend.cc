#include "core/speaker_recognition/general_sr_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

#include "utils/logging.h"

namespace soundproof {
namespace {

bool ParseIntMetadata(const std::map<std::string, std::string>& metadata,
                      const char* key, int32_t* value) {
  const auto it = metadata.find(key);
  if (it == metadata.end()) {
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0') {
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

std::string GetStringMetadata(
    const std::map<std::string, std::string>& metadata, const char* key) {
  const auto it = metadata.find(key);
  return it == metadata.end() ? std::string() : it->second;
}

bool AllocateData(mnn_engine::Tensor* tensor) {
  size_t count = 1;
  for (int d : tensor->dim) {
    if (d <= 0) {
      return false;
    }
    count *= static_cast<size_t>(d);
  }
  const int32_t bytes = mnn_engine::BytesPerElement(tensor->type);
  if (bytes <= 0) {
    return false;
  }
  tensor->data.assign(count * static_cast<size_t>(bytes), 0);
  return true;
}

}  // namespace

SrStatus GeneralSrBackend::Init(const SrModelConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: empty model path";
    return SrStatus::kModelLoadFailed;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: cannot load model "
                      << config.model_path;
    return SrStatus::kModelLoadFailed;
  }

  const auto metadata = engine_.metadata();
  model_type_ = GetStringMetadata(metadata, "framework");
  if (model_type_ != "wespeaker" && model_type_ != "3d-speaker") {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: unsupported framework '"
                      << model_type_ << "'";
    return SrStatus::kUnsupportedModel;
  }
  if (!config.model_type.empty() && config.model_type != model_type_) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: requested model_type '"
                      << config.model_type << "' but model is '" << model_type_
                      << "'";
    return SrStatus::kUnsupportedModel;
  }

  if (!ParseIntMetadata(metadata, "output_dim", &output_dim_) ||
      output_dim_ <= 0) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: missing/invalid output_dim";
    return SrStatus::kUnsupportedModel;
  }

  int32_t sample_rate = 16000;
  if (!ParseIntMetadata(metadata, "sample_rate", &sample_rate) ||
      sample_rate != 16000) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: only 16 kHz models are supported";
    return SrStatus::kUnsupportedModel;
  }

  int32_t normalize_samples = 1;
  ParseIntMetadata(metadata, "normalize_samples", &normalize_samples);
  normalize_samples_ = normalize_samples != 0;

  const std::string normalize_type =
      GetStringMetadata(metadata, "feature_normalize_type");
  if (!normalize_type.empty()) {
    if (normalize_type != "global-mean") {
      RTC_LOG(LS_ERROR) << "GeneralSrBackend: unsupported feature "
                           "normalize type '"
                        << normalize_type << "'";
      return SrStatus::kUnsupportedModel;
    }
    global_mean_ = true;
  }

  const auto inputs = engine_.inputs();
  if (inputs.size() != 1 ||
      inputs[0].type != mnn_engine::DataType::kFloat32 ||
      inputs[0].dim.size() != 3) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: expected one 3-D float input";
    return SrStatus::kUnsupportedModel;
  }
  input_ = inputs[0];
  input_.dim[0] = 1;
  if (input_.dim[1] <= 0) {
    input_.dim[1] = 1;
  }
  int32_t feature_dim = input_.dim[2] > 0 ? input_.dim[2]
                                          : FeatureExtractor::kFeatureDim;
  if (feature_dim <= 0) {
    return SrStatus::kUnsupportedModel;
  }
  input_.dim[2] = feature_dim;
  if (!AllocateData(&input_)) {
    return SrStatus::kUnsupportedModel;
  }

  feature_config_ = FeatureExtractorConfig();
  feature_config_.sample_rate = sample_rate;
  feature_config_.feature_dim = feature_dim;
  min_frames_ = static_cast<int32_t>(
      std::ceil(0.5f * 1000.0f / feature_config_.frame_shift_ms));
  ok_ = true;
  return SrStatus::kOk;
}

SrStatus GeneralSrBackend::ComputeEmbedding(
    const float* features, int32_t num_frames, int32_t dim,
    std::vector<float>* embedding) {
  if (!ok_ || features == nullptr || embedding == nullptr) {
    return SrStatus::kInferenceFailed;
  }
  if (dim != FeatureDim()) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: feature dim " << dim
                      << " != model dim " << FeatureDim();
    return SrStatus::kFeatureFailed;
  }
  if (num_frames < min_frames_) {
    RTC_LOG(LS_WARNING) << "GeneralSrBackend: audio too short (" << num_frames
                        << " frames < " << min_frames_ << ")";
    return SrStatus::kAudioTooShort;
  }

  std::vector<float> buffer(features,
                            features + static_cast<size_t>(num_frames) * dim);
  if (global_mean_) {
    std::vector<double> mean(static_cast<size_t>(dim), 0.0);
    for (int32_t t = 0; t < num_frames; ++t) {
      for (int32_t d = 0; d < dim; ++d) {
        mean[static_cast<size_t>(d)] +=
            buffer[static_cast<size_t>(t) * dim + d];
      }
    }
    for (int32_t d = 0; d < dim; ++d) {
      mean[static_cast<size_t>(d)] /= num_frames;
    }
    for (int32_t t = 0; t < num_frames; ++t) {
      for (int32_t d = 0; d < dim; ++d) {
        buffer[static_cast<size_t>(t) * dim + d] -=
            static_cast<float>(mean[static_cast<size_t>(d)]);
      }
    }
  }

  input_.dim = {1, num_frames, dim};
  if (!AllocateData(&input_)) {
    return SrStatus::kFeatureFailed;
  }
  std::memcpy(input_.data.data(), buffer.data(), input_.data.size());

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward({input_}, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: model forward failed";
    return SrStatus::kInferenceFailed;
  }
  const mnn_engine::Tensor& out = outputs[0];
  if (out.type != mnn_engine::DataType::kFloat32 ||
      out.data.size() < static_cast<size_t>(output_dim_) * sizeof(float)) {
    RTC_LOG(LS_ERROR) << "GeneralSrBackend: invalid embedding output";
    return SrStatus::kInferenceFailed;
  }
  const float* data = reinterpret_cast<const float*>(out.data.data());
  embedding->assign(data, data + output_dim_);
  return SrStatus::kOk;
}

}  // namespace soundproof
