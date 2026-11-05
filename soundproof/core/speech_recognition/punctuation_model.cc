#include "core/speech_recognition/punctuation_model.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

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

PunctuationModel::PunctuationModel() = default;
PunctuationModel::~PunctuationModel() = default;

bool PunctuationModel::Init(const PunctuationModelConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "PunctuationModel: empty model path";
    return false;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "PunctuationModel: cannot load model "
                      << config.model_path;
    return false;
  }

  const auto metadata = engine_.metadata();
  ParseIntMetadata(metadata, "COMMA", &meta_.comma_id);
  ParseIntMetadata(metadata, "PERIOD", &meta_.period_id);
  ParseIntMetadata(metadata, "QUESTION", &meta_.question_id);
  ParseIntMetadata(metadata, "UPPER", &meta_.upper_id);
  ParseIntMetadata(metadata, "CAP", &meta_.cap_id);
  ParseIntMetadata(metadata, "MIX_CASE", &meta_.mix_case_id);

  const auto inputs = engine_.inputs();
  if (inputs.size() != 3) {
    RTC_LOG(LS_ERROR) << "PunctuationModel: expected 3 inputs, got "
                      << inputs.size();
    return false;
  }
  inputs_.clear();
  int32_t rank2_count = 0;
  int32_t rank1_count = 0;
  for (const auto& input : inputs) {
    mnn_engine::Tensor tensor = input;
    if (tensor.type != mnn_engine::DataType::kInt32) {
      RTC_LOG(LS_ERROR) << "PunctuationModel: inputs must be int32";
      return false;
    }
    if (tensor.dim.size() == 2) {
      tensor.dim = {1, kMaxSeqLen};
      ++rank2_count;
    } else if (tensor.dim.size() == 1) {
      tensor.dim = {1};
      ++rank1_count;
    } else {
      RTC_LOG(LS_ERROR) << "PunctuationModel: unsupported input rank";
      return false;
    }
    if (!AllocateData(&tensor)) {
      return false;
    }
    inputs_.push_back(std::move(tensor));
  }
  if (rank2_count != 2 || rank1_count != 1) {
    RTC_LOG(LS_ERROR) << "PunctuationModel: unexpected input layout";
    return false;
  }
  ok_ = true;
  return true;
}

bool PunctuationModel::Forward(const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& valid_ids,
                               int32_t label_len,
                               std::vector<float>* case_logits,
                               int32_t* num_cases,
                               std::vector<float>* punct_logits,
                               int32_t* num_punctuations) {
  if (!ok_ || case_logits == nullptr || punct_logits == nullptr ||
      num_cases == nullptr || num_punctuations == nullptr ||
      token_ids.size() != static_cast<size_t>(kMaxSeqLen) ||
      valid_ids.size() != static_cast<size_t>(kMaxSeqLen)) {
    return false;
  }

  int32_t rank2_index = 0;
  bool label_set = false;
  for (auto& tensor : inputs_) {
    if (tensor.dim.size() == 2) {
      if (rank2_index == 0) {
        std::memcpy(tensor.data.data(), token_ids.data(), tensor.data.size());
      } else {
        std::memcpy(tensor.data.data(), valid_ids.data(), tensor.data.size());
      }
      ++rank2_index;
    } else if (!label_set) {
      *reinterpret_cast<int32_t*>(tensor.data.data()) = label_len;
      label_set = true;
    }
  }

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward(inputs_, &outputs) || outputs.size() < 2) {
    RTC_LOG(LS_ERROR) << "PunctuationModel: forward failed";
    return false;
  }
  const mnn_engine::Tensor& case_tensor = outputs[0];
  const mnn_engine::Tensor& punct_tensor = outputs[1];
  if (case_tensor.type != mnn_engine::DataType::kFloat32 ||
      punct_tensor.type != mnn_engine::DataType::kFloat32 ||
      case_tensor.dim.empty() || punct_tensor.dim.empty()) {
    return false;
  }
  *num_cases = case_tensor.dim.back();
  *num_punctuations = punct_tensor.dim.back();
  if (*num_cases <= 0 || *num_punctuations <= 0) {
    return false;
  }

  const float* case_data =
      reinterpret_cast<const float*>(case_tensor.data.data());
  const float* punct_data =
      reinterpret_cast<const float*>(punct_tensor.data.data());
  case_logits->assign(case_data,
                      case_data + case_tensor.data.size() / sizeof(float));
  punct_logits->assign(punct_data,
                       punct_data + punct_tensor.data.size() / sizeof(float));
  return true;
}

}  // namespace soundproof
