#include "core/text_to_speech/vits_backend.h"

#include <algorithm>
#include <cctype>
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

std::string Lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::vector<int32_t> AddBlank(const std::vector<int32_t>& x) {
  std::vector<int32_t> buffer(x.size() * 2 + 1, 0);
  size_t i = 1;
  for (int32_t id : x) {
    buffer[i] = id;
    i += 2;
  }
  return buffer;
}

}  // namespace

VitsBackend::VitsBackend() = default;
VitsBackend::~VitsBackend() = default;

TtsStatus VitsBackend::Init(const TtsModelConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "VitsBackend: empty model path";
    return TtsStatus::kModelLoadFailed;
  }
  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "VitsBackend: cannot load model " << config.model_path;
    return TtsStatus::kModelLoadFailed;
  }

  const auto metadata = engine_.metadata();
  int32_t metadata_sample_rate = 0;
  ParseIntMetadata(metadata, "sample_rate", &metadata_sample_rate);
  ParseIntMetadata(metadata, "n_speakers", &num_speakers_);
  ParseIntMetadata(metadata, "add_blank", &add_blank_);
  sample_rate_ = config.output_sample_rate > 0
                     ? config.output_sample_rate
                     : (metadata_sample_rate > 0 ? metadata_sample_rate : 22050);
  if (num_speakers_ <= 0) {
    num_speakers_ = 1;
  }

  const auto model_inputs = engine_.inputs();
  if (model_inputs.empty()) {
    return TtsStatus::kUnsupportedModel;
  }
  inputs_.clear();
  roles_.clear();
  bool has_tokens = false;
  bool has_length = false;
  for (size_t i = 0; i < model_inputs.size(); ++i) {
    mnn_engine::Tensor tensor = model_inputs[i];
    const std::string name = Lower(tensor.name);
    Role role = Role::kOther;
    if (name == "x" || name == "input_ids" || name == "tokens" ||
        name == "text") {
      role = Role::kToken;
      has_tokens = true;
      tensor.dim = {1, 1};
    } else if (name.find("tone") != std::string::npos) {
      role = Role::kTones;
      tensor.dim = {1, 1};
    } else if (name.find("length") != std::string::npos &&
               name.find("scale") == std::string::npos) {
      role = Role::kLength;
      has_length = true;
      tensor.dim = {1};
    } else if (name == "noise_scale_w") {
      role = Role::kNoiseScaleW;
    } else if (name == "noise_scale") {
      role = Role::kNoiseScale;
    } else if (name == "length_scale") {
      role = Role::kLengthScale;
    } else if (name == "sid" || name == "speaker" || name == "speaker_id") {
      role = Role::kSid;
    } else if (name == "scales") {
      role = Role::kScales;
    } else {
      RTC_LOG(LS_ERROR) << "VitsBackend: unsupported input '" << tensor.name
                        << "'";
      return TtsStatus::kUnsupportedModel;
    }
    for (int& d : tensor.dim) {
      if (d <= 0) {
        d = 1;
      }
    }
    if (!AllocateData(&tensor)) {
      return TtsStatus::kUnsupportedModel;
    }
    roles_[i] = role;
    inputs_.push_back(std::move(tensor));
  }
  if (!has_tokens || !has_length) {
    RTC_LOG(LS_ERROR) << "VitsBackend: model needs token and length inputs";
    return TtsStatus::kUnsupportedModel;
  }

  if (!config.tokens_path.empty()) {
    tokenizer_.Load(config.tokens_path);
  }
  ok_ = true;
  return TtsStatus::kOk;
}

TtsStatus VitsBackend::EncodePhonemes(
    const std::string& text, std::vector<int32_t>* token_ids) {
  if (!tokenizer_.ok()) {
    RTC_LOG(LS_ERROR) << "VitsBackend: no tokenizer configured";
    return TtsStatus::kUnsupportedInput;
  }
  if (!tokenizer_.Encode(text, token_ids)) {
    return TtsStatus::kUnsupportedInput;
  }
  return TtsStatus::kOk;
}

TtsStatus VitsBackend::Synthesize(const std::vector<int32_t>& token_ids,
                                  int32_t speaker_id, float speed,
                                  std::vector<float>* samples) {
  if (!ok_ || samples == nullptr) {
    return TtsStatus::kNotInitialized;
  }
  if (token_ids.empty() || speed <= 0.0f) {
    return TtsStatus::kInvalidInput;
  }
  if (speaker_id < 0 || speaker_id >= num_speakers_) {
    RTC_LOG(LS_ERROR) << "VitsBackend: invalid speaker_id " << speaker_id;
    return TtsStatus::kInvalidInput;
  }

  std::vector<int32_t> tokens = token_ids;
  if (add_blank_) {
    tokens = AddBlank(tokens);
  }
  const float length_scale =
      speed > 0.0f ? (length_scale_ / speed) : length_scale_;

  bool token_filled = false;
  bool length_filled = false;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    mnn_engine::Tensor& tensor = inputs_[i];
    const Role role = roles_[i];
    switch (role) {
      case Role::kToken: {
        tensor.dim = {1, static_cast<int>(tokens.size())};
        if (!AllocateData(&tensor)) {
          return TtsStatus::kInferenceFailed;
        }
        if (tensor.type == mnn_engine::DataType::kInt64) {
          std::vector<int64_t> values(tokens.begin(), tokens.end());
          std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
        } else {
          std::memcpy(tensor.data.data(), tokens.data(), tensor.data.size());
        }
        token_filled = true;
        break;
      }
      case Role::kLength:
        if (!AllocateData(&tensor)) {
          return TtsStatus::kInferenceFailed;
        }
        *reinterpret_cast<int32_t*>(tensor.data.data()) =
            static_cast<int32_t>(tokens.size());
        length_filled = true;
        break;
      case Role::kTones:
        tensor.dim = {1, static_cast<int>(tokens.size())};
        if (!AllocateData(&tensor)) {
          return TtsStatus::kInferenceFailed;
        }
        break;
      case Role::kNoiseScale:
        *reinterpret_cast<float*>(tensor.data.data()) = noise_scale_;
        break;
      case Role::kLengthScale:
        *reinterpret_cast<float*>(tensor.data.data()) = length_scale;
        break;
      case Role::kNoiseScaleW:
        *reinterpret_cast<float*>(tensor.data.data()) = noise_scale_w_;
        break;
      case Role::kSid:
        *reinterpret_cast<int32_t*>(tensor.data.data()) = speaker_id;
        break;
      case Role::kScales: {
        if (tensor.data.size() < 3 * sizeof(float)) {
          return TtsStatus::kUnsupportedModel;
        }
        float* scales = reinterpret_cast<float*>(tensor.data.data());
        scales[0] = noise_scale_;
        scales[1] = length_scale;
        scales[2] = noise_scale_w_;
        break;
      }
      case Role::kOther:
        break;
    }
  }
  if (!token_filled || !length_filled) {
    return TtsStatus::kInferenceFailed;
  }

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward(inputs_, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "VitsBackend: synthesis failed";
    return TtsStatus::kInferenceFailed;
  }
  const mnn_engine::Tensor& audio = outputs[0];
  if (audio.type != mnn_engine::DataType::kFloat32 || audio.data.empty()) {
    return TtsStatus::kInferenceFailed;
  }
  const float* data = reinterpret_cast<const float*>(audio.data.data());
  samples->assign(data, data + audio.data.size() / sizeof(float));
  return TtsStatus::kOk;
}

}  // namespace soundproof
