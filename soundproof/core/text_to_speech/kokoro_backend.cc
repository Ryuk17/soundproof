#include "core/text_to_speech/kokoro_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

bool EndsWithMnn(const std::string& path) {
  if (path.size() < 4) {
    return false;
  }
  std::string ext = path.substr(path.size() - 4);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".mnn";
}

}  // namespace

KokoroBackend::KokoroBackend() = default;
KokoroBackend::~KokoroBackend() = default;

TtsStatus KokoroBackend::Init(const TtsModelConfig& config) {
  if (config.model_path.empty() || config.voices_path.empty()) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: model and voices paths are required";
    return TtsStatus::kModelLoadFailed;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: cannot load model "
                      << config.model_path;
    return TtsStatus::kModelLoadFailed;
  }

  const auto metadata = engine_.metadata();
  int32_t metadata_sample_rate = 0;
  ParseIntMetadata(metadata, "sample_rate", &metadata_sample_rate);
  int32_t metadata_speakers = 0;
  ParseIntMetadata(metadata, "n_speakers", &metadata_speakers);
  int32_t metadata_style_dim = 0;
  ParseIntMetadata(metadata, "style_dim", &metadata_style_dim);
  if (metadata_style_dim > 0) {
    style_dim_ = metadata_style_dim;
  }

  const auto model_inputs = engine_.inputs();
  if (model_inputs.size() != 3) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: expected 3 inputs, got "
                      << model_inputs.size();
    return TtsStatus::kUnsupportedModel;
  }
  inputs_.clear();
  bool has_tokens = false;
  bool has_style = false;
  bool has_speed = false;
  for (size_t i = 0; i < model_inputs.size(); ++i) {
    mnn_engine::Tensor tensor = model_inputs[i];
    if (tensor.type == mnn_engine::DataType::kInt32 &&
        tensor.dim.size() == 2) {
      if (has_tokens) {
        return TtsStatus::kUnsupportedModel;
      }
      has_tokens = true;
      token_index_ = i;
      tensor.dim = {1, 1};
    } else if (tensor.type == mnn_engine::DataType::kFloat32 &&
               tensor.dim.size() == 2) {
      if (has_style) {
        return TtsStatus::kUnsupportedModel;
      }
      has_style = true;
      style_index_ = i;
      if (tensor.dim.back() > 0) {
        style_dim_ = tensor.dim.back();
      }
      tensor.dim = {1, style_dim_};
    } else if (tensor.type == mnn_engine::DataType::kFloat32 &&
               tensor.dim.size() == 1) {
      if (has_speed) {
        return TtsStatus::kUnsupportedModel;
      }
      has_speed = true;
      speed_index_ = i;
      tensor.dim = {1};
    } else {
      RTC_LOG(LS_ERROR) << "KokoroBackend: unsupported input '"
                        << tensor.name << "'";
      return TtsStatus::kUnsupportedModel;
    }
    if (!AllocateData(&tensor)) {
      return TtsStatus::kUnsupportedModel;
    }
    inputs_.push_back(std::move(tensor));
  }
  if (!has_tokens || !has_style || !has_speed) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: incomplete model inputs";
    return TtsStatus::kUnsupportedModel;
  }

  const TtsStatus voices_status = LoadVoices(config.voices_path);
  if (voices_status != TtsStatus::kOk) {
    return voices_status;
  }
  if (metadata_speakers > 0 && metadata_speakers != num_speakers_) {
    num_speakers_ = std::min(metadata_speakers, num_speakers_);
  }

  if (!config.tokens_path.empty()) {
    tokenizer_.Load(config.tokens_path);
  }

  sample_rate_ = config.output_sample_rate > 0
                     ? config.output_sample_rate
                     : (metadata_sample_rate > 0 ? metadata_sample_rate
                                                 : 24000);
  ok_ = true;
  return TtsStatus::kOk;
}

TtsStatus KokoroBackend::LoadVoices(const std::string& path) {
  if (EndsWithMnn(path)) {
    return LoadMnnVoices(path);
  }
  return LoadRawVoices(path);
}

TtsStatus KokoroBackend::LoadMnnVoices(const std::string& path) {
  mnn_engine::Engine engine;
  mnn_engine::EngineConfig config;
  config.num_threads = 1;
  if (!engine.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: cannot load voices " << path;
    return TtsStatus::kModelLoadFailed;
  }
  std::vector<mnn_engine::Tensor> outputs;
  const std::vector<mnn_engine::Tensor> no_inputs;
  if (!engine.Forward(no_inputs, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: voices forward failed";
    return TtsStatus::kInferenceFailed;
  }
  const mnn_engine::Tensor& voices = outputs[0];
  if (voices.type != mnn_engine::DataType::kFloat32 || voices.dim.empty()) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: invalid voices output";
    return TtsStatus::kUnsupportedModel;
  }

  if (voices.dim.size() == 3) {
    num_speakers_ = voices.dim[0];
    max_token_len_ = voices.dim[1];
    style_dim_ = voices.dim[2];
  } else if (voices.dim.size() == 2) {
    num_speakers_ = 1;
    max_token_len_ = voices.dim[0];
    style_dim_ = voices.dim[1];
  } else if (voices.dim.size() == 1 && style_dim_ > 0) {
    num_speakers_ = 1;
    max_token_len_ = voices.dim[0] / style_dim_;
  } else {
    return TtsStatus::kUnsupportedModel;
  }
  if (num_speakers_ <= 0 || max_token_len_ <= 0 || style_dim_ <= 0) {
    return TtsStatus::kUnsupportedModel;
  }

  const size_t count = voices.data.size() / sizeof(float);
  if (count != static_cast<size_t>(num_speakers_) *
                    static_cast<size_t>(max_token_len_) *
                    static_cast<size_t>(style_dim_)) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: voices size mismatch";
    return TtsStatus::kUnsupportedModel;
  }
  const float* data = reinterpret_cast<const float*>(voices.data.data());
  styles_.assign(data, data + count);
  return TtsStatus::kOk;
}

TtsStatus KokoroBackend::LoadRawVoices(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: cannot open voices " << path;
    return TtsStatus::kModelLoadFailed;
  }
  const std::streamsize size = file.tellg();
  if (size <= 0 || style_dim_ <= 0) {
    return TtsStatus::kUnsupportedModel;
  }
  file.seekg(0, std::ios::beg);
  const size_t count = static_cast<size_t>(size) / sizeof(float);
  styles_.resize(count);
  if (!file.read(reinterpret_cast<char*>(styles_.data()), size)) {
    return TtsStatus::kModelLoadFailed;
  }

  if (num_speakers_ <= 0) {
    num_speakers_ = 1;
  }
  max_token_len_ =
      static_cast<int32_t>(count / (num_speakers_ * style_dim_));
  if (max_token_len_ <= 0 ||
      static_cast<size_t>(num_speakers_) * max_token_len_ * style_dim_ !=
          count) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: raw voices size mismatch";
    return TtsStatus::kUnsupportedModel;
  }
  return TtsStatus::kOk;
}

TtsStatus KokoroBackend::EncodePhonemes(
    const std::string& text, std::vector<int32_t>* token_ids) {
  if (!tokenizer_.ok()) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: no tokenizer configured";
    return TtsStatus::kUnsupportedInput;
  }
  if (!tokenizer_.Encode(text, token_ids)) {
    return TtsStatus::kUnsupportedInput;
  }
  return TtsStatus::kOk;
}

TtsStatus KokoroBackend::Synthesize(const std::vector<int32_t>& token_ids,
                                    int32_t speaker_id, float speed,
                                    std::vector<float>* samples) {
  if (!ok_ || samples == nullptr) {
    return TtsStatus::kNotInitialized;
  }
  if (token_ids.empty() || speed <= 0.0f) {
    return TtsStatus::kInvalidInput;
  }
  if (speaker_id < 0 || speaker_id >= num_speakers_) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: invalid speaker_id " << speaker_id
                      << " (num_speakers=" << num_speakers_ << ")";
    return TtsStatus::kInvalidInput;
  }
  const int32_t len = static_cast<int32_t>(token_ids.size());
  if (max_token_len_ > 1 && len >= max_token_len_) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: too many tokens " << len
                      << " (max " << max_token_len_ - 1 << ")";
    return TtsStatus::kInvalidInput;
  }

  std::vector<int32_t> x;
  x.reserve(static_cast<size_t>(len) + 2);
  x.push_back(0);
  x.insert(x.end(), token_ids.begin(), token_ids.end());
  x.push_back(0);

  mnn_engine::Tensor& token_tensor = inputs_[token_index_];
  token_tensor.dim = {1, static_cast<int>(x.size())};
  if (!AllocateData(&token_tensor)) {
    return TtsStatus::kInferenceFailed;
  }
  std::memcpy(token_tensor.data.data(), x.data(), token_tensor.data.size());

  mnn_engine::Tensor& style_tensor = inputs_[style_index_];
  style_tensor.dim = {1, style_dim_};
  if (!AllocateData(&style_tensor)) {
    return TtsStatus::kInferenceFailed;
  }
  // Models converted by MNN may have one style vector per speaker
  // ([num_speakers, 1, style_dim]); sherpa-style models index the style by
  // the number of tokens.
  const size_t style_index =
      max_token_len_ > 1
          ? static_cast<size_t>(speaker_id) * max_token_len_ + len
          : static_cast<size_t>(speaker_id);
  const size_t style_offset = style_index * style_dim_;
  if (style_offset + style_dim_ > styles_.size()) {
    return TtsStatus::kInvalidInput;
  }
  std::memcpy(style_tensor.data.data(), styles_.data() + style_offset,
              static_cast<size_t>(style_dim_) * sizeof(float));

  mnn_engine::Tensor& speed_tensor = inputs_[speed_index_];
  if (speed_tensor.data.size() < sizeof(float)) {
    return TtsStatus::kInferenceFailed;
  }
  *reinterpret_cast<float*>(speed_tensor.data.data()) = speed;

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward(inputs_, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "KokoroBackend: synthesis failed";
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
