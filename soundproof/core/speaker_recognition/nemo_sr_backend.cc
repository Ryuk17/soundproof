#include "core/speaker_recognition/nemo_sr_backend.h"

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

SrStatus NemoSrBackend::Init(const SrModelConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: empty model path";
    return SrStatus::kModelLoadFailed;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: cannot load model "
                      << config.model_path;
    return SrStatus::kModelLoadFailed;
  }

  const auto metadata = engine_.metadata();
  if (GetStringMetadata(metadata, "framework") != "nemo") {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: model framework is not nemo";
    return SrStatus::kUnsupportedModel;
  }
  if (!config.model_type.empty() && config.model_type != "nemo") {
    return SrStatus::kUnsupportedModel;
  }

  int32_t sample_rate = 16000;
  int32_t feat_dim = 80;
  int32_t window_size_ms = 25;
  int32_t window_stride_ms = 25;
  if (!ParseIntMetadata(metadata, "output_dim", &output_dim_) ||
      output_dim_ <= 0 || !ParseIntMetadata(metadata, "sample_rate",
                                            &sample_rate) ||
      sample_rate != 16000 || !ParseIntMetadata(metadata, "feat_dim",
                                                &feat_dim) ||
      feat_dim <= 0 ||
      !ParseIntMetadata(metadata, "window_size_ms", &window_size_ms) ||
      !ParseIntMetadata(metadata, "window_stride_ms", &window_stride_ms)) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: missing/invalid model metadata";
    return SrStatus::kUnsupportedModel;
  }

  const std::string normalize_type =
      GetStringMetadata(metadata, "feature_normalize_type");
  if (!normalize_type.empty()) {
    if (normalize_type != "per_feature") {
      RTC_LOG(LS_ERROR) << "NemoSrBackend: unsupported feature normalize "
                           "type '"
                        << normalize_type << "'";
      return SrStatus::kUnsupportedModel;
    }
    per_feature_normalize_ = true;
  }

  feature_config_ = FeatureExtractorConfig();
  feature_config_.sample_rate = sample_rate;
  feature_config_.feature_dim = feat_dim;
  feature_config_.frame_length_ms = static_cast<float>(window_size_ms);
  feature_config_.frame_shift_ms = static_cast<float>(window_stride_ms);
  feature_config_.snip_edges = true;
  feature_config_.low_freq = 0.0f;
  feature_config_.remove_dc_offset = false;
  feature_config_.is_librosa = true;
  const std::string window_type =
      GetStringMetadata(metadata, "window_type");
  if (!window_type.empty()) {
    feature_config_.window_type = window_type;
  }
  min_frames_ = static_cast<int32_t>(
      std::ceil(0.5f * 1000.0f / feature_config_.frame_shift_ms));
  if (min_frames_ < 1) {
    min_frames_ = 1;
  }

  const auto inputs = engine_.inputs();
  if (inputs.size() != 2) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: expected 2 model inputs, got "
                      << inputs.size();
    return SrStatus::kUnsupportedModel;
  }
  bool has_x = false;
  bool has_len = false;
  for (const auto& tensor : inputs) {
    if (tensor.type == mnn_engine::DataType::kFloat32 &&
        tensor.dim.size() == 3 && !has_x) {
      input_ = tensor;
      has_x = true;
    } else if (tensor.type == mnn_engine::DataType::kInt32 &&
               tensor.dim.size() == 1 && !has_len) {
      length_ = tensor;
      has_len = true;
    }
  }
  if (!has_x || !has_len) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: unexpected model input layout";
    return SrStatus::kUnsupportedModel;
  }
  input_.dim = {1, feat_dim, 1};
  if (!AllocateData(&input_) || !AllocateData(&length_)) {
    return SrStatus::kUnsupportedModel;
  }
  ok_ = true;
  return SrStatus::kOk;
}

SrStatus NemoSrBackend::ComputeEmbedding(const float* features,
                                         int32_t num_frames, int32_t dim,
                                         std::vector<float>* embedding) {
  if (!ok_ || features == nullptr || embedding == nullptr) {
    return SrStatus::kInferenceFailed;
  }
  if (dim != FeatureDim()) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: feature dim " << dim
                      << " != model dim " << FeatureDim();
    return SrStatus::kFeatureFailed;
  }
  if (num_frames < min_frames_) {
    RTC_LOG(LS_WARNING) << "NemoSrBackend: audio too short (" << num_frames
                        << " frames < " << min_frames_ << ")";
    return SrStatus::kAudioTooShort;
  }

  std::vector<float> buffer(features,
                            features + static_cast<size_t>(num_frames) * dim);
  if (per_feature_normalize_) {
    std::vector<double> mean(static_cast<size_t>(dim), 0.0);
    std::vector<double> sq_mean(static_cast<size_t>(dim), 0.0);
    for (int32_t t = 0; t < num_frames; ++t) {
      for (int32_t d = 0; d < dim; ++d) {
        const double v = buffer[static_cast<size_t>(t) * dim + d];
        mean[static_cast<size_t>(d)] += v;
        sq_mean[static_cast<size_t>(d)] += v * v;
      }
    }
    for (int32_t d = 0; d < dim; ++d) {
      mean[static_cast<size_t>(d)] /= num_frames;
      sq_mean[static_cast<size_t>(d)] /= num_frames;
      const double variance = std::max(
          0.0, sq_mean[static_cast<size_t>(d)] -
                   mean[static_cast<size_t>(d)] * mean[static_cast<size_t>(d)]);
      const double stddev = std::sqrt(variance);
      for (int32_t t = 0; t < num_frames; ++t) {
        float& v = buffer[static_cast<size_t>(t) * dim + d];
        v = static_cast<float>(
            (v - mean[static_cast<size_t>(d)]) / (stddev + 1e-5));
      }
    }
  }

  // Transpose (T, C) -> (C, T) for the NeMo model.
  input_.dim = {1, dim, num_frames};
  if (!AllocateData(&input_)) {
    return SrStatus::kFeatureFailed;
  }
  float* dst = reinterpret_cast<float*>(input_.data.data());
  for (int32_t d = 0; d < dim; ++d) {
    for (int32_t t = 0; t < num_frames; ++t) {
      dst[static_cast<size_t>(d) * num_frames + t] =
          buffer[static_cast<size_t>(t) * dim + d];
    }
  }

  if (length_.data.size() < sizeof(int32_t)) {
    return SrStatus::kFeatureFailed;
  }
  *reinterpret_cast<int32_t*>(length_.data.data()) = num_frames;

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward({input_, length_}, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: model forward failed";
    return SrStatus::kInferenceFailed;
  }
  const mnn_engine::Tensor& out = outputs[0];
  if (out.type != mnn_engine::DataType::kFloat32 ||
      out.data.size() < static_cast<size_t>(output_dim_) * sizeof(float)) {
    RTC_LOG(LS_ERROR) << "NemoSrBackend: invalid embedding output";
    return SrStatus::kInferenceFailed;
  }
  const float* data = reinterpret_cast<const float*>(out.data.data());
  embedding->assign(data, data + output_dim_);
  return SrStatus::kOk;
}

}  // namespace soundproof
