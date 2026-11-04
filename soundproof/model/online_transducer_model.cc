#include "model/online_transducer_model.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

#include "utils/logging.h"

namespace soundproof {
namespace {

bool ResolveDynamicDims(mnn_engine::Tensor* tensor) {
  for (int& d : tensor->dim) {
    if (d <= 0) {
      d = 1;
    }
  }
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

bool ParseIntMetadata(const std::map<std::string, std::string>& metadata,
                      const char* key, int32_t* value) {
  const auto it = metadata.find(key);
  if (it == metadata.end()) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: missing metadata '" << key
                      << "'";
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0' || parsed <= 0) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: invalid metadata '" << key
                      << "'=" << it->second;
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

size_t FindInputByName(const std::vector<mnn_engine::Tensor>& inputs,
                       const std::string& name) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].name == name) {
      return i;
    }
  }
  return inputs.size();
}

}  // namespace

OnlineTransducerModel::OnlineTransducerModel() = default;
OnlineTransducerModel::~OnlineTransducerModel() = default;

bool OnlineTransducerModel::Init(const TransducerModelConfig& config) {
  if (!InitEncoder(config.encoder_path, config.num_threads)) {
    return false;
  }
  if (!InitDecoder(config.decoder_path, config.num_threads)) {
    return false;
  }
  if (!InitJoiner(config.joiner_path, config.num_threads)) {
    return false;
  }
  ok_ = true;
  return true;
}

bool OnlineTransducerModel::InitEncoder(const std::string& path,
                                        int num_threads) {
  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!encoder_.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: cannot load encoder " << path;
    return false;
  }

  const std::map<std::string, std::string> metadata = encoder_.metadata();
  if (!ParseIntMetadata(metadata, "T", &chunk_size_) ||
      !ParseIntMetadata(metadata, "decode_chunk_len", &chunk_shift_)) {
    return false;
  }
  const auto model_type_it = metadata.find("model_type");
  if (model_type_it != metadata.end()) {
    model_type_ = model_type_it->second;
  }

  const std::vector<mnn_engine::Tensor> inputs = encoder_.inputs();
  encoder_inputs_.clear();
  encoder_state_indices_.clear();
  bool has_feature = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    mnn_engine::Tensor tensor = inputs[i];
    if (!ResolveDynamicDims(&tensor) || !AllocateData(&tensor)) {
      RTC_LOG(LS_ERROR) << "OnlineTransducerModel: bad encoder input '"
                        << tensor.name << "'";
      return false;
    }
    if (tensor.name == "x") {
      if (has_feature || tensor.type != mnn_engine::DataType::kFloat32 ||
          tensor.dim.size() != 3) {
        RTC_LOG(LS_ERROR) << "OnlineTransducerModel: invalid encoder feature "
                             "input";
        return false;
      }
      has_feature = true;
      encoder_feature_index_ = i;
      if (tensor.dim[1] != chunk_size_) {
        RTC_LOG(LS_ERROR) << "OnlineTransducerModel: feature chunk size "
                          << tensor.dim[1] << " != metadata T " << chunk_size_;
        return false;
      }
      feature_dim_ = tensor.dim[2];
    } else {
      encoder_state_indices_.push_back(i);
    }
    encoder_inputs_.push_back(std::move(tensor));
  }
  if (!has_feature) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: encoder has no 'x' input";
    return false;
  }

  encoder_state_templates_.clear();
  encoder_state_templates_.reserve(encoder_state_indices_.size());
  for (size_t index : encoder_state_indices_) {
    encoder_state_templates_.push_back(encoder_inputs_[index]);
  }
  return true;
}

bool OnlineTransducerModel::InitDecoder(const std::string& path,
                                        int num_threads) {
  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!decoder_.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: cannot load decoder " << path;
    return false;
  }

  const std::map<std::string, std::string> metadata = decoder_.metadata();
  if (!ParseIntMetadata(metadata, "vocab_size", &vocab_size_) ||
      !ParseIntMetadata(metadata, "context_size", &context_size_)) {
    return false;
  }

  const std::vector<mnn_engine::Tensor> inputs = decoder_.inputs();
  if (inputs.size() != 1) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: decoder expects 1 input, got "
                      << inputs.size();
    return false;
  }
  mnn_engine::Tensor tensor = inputs[0];
  if (!ResolveDynamicDims(&tensor) || !AllocateData(&tensor) ||
      tensor.type != mnn_engine::DataType::kInt32 || tensor.dim.size() != 2 ||
      tensor.dim[1] != context_size_) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: unsupported decoder input";
    return false;
  }
  decoder_inputs_.clear();
  decoder_inputs_.push_back(std::move(tensor));
  return true;
}

bool OnlineTransducerModel::InitJoiner(const std::string& path,
                                       int num_threads) {
  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!joiner_.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: cannot load joiner " << path;
    return false;
  }

  const std::vector<mnn_engine::Tensor> inputs = joiner_.inputs();
  if (inputs.size() != 2) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: joiner expects 2 inputs, got "
                      << inputs.size();
    return false;
  }
  joiner_inputs_.clear();
  for (const mnn_engine::Tensor& input : inputs) {
    mnn_engine::Tensor tensor = input;
    if (!ResolveDynamicDims(&tensor) || !AllocateData(&tensor) ||
        tensor.type != mnn_engine::DataType::kFloat32) {
      RTC_LOG(LS_ERROR) << "OnlineTransducerModel: unsupported joiner input '"
                        << tensor.name << "'";
      return false;
    }
    joiner_inputs_.push_back(std::move(tensor));
  }

  joiner_encoder_index_ = FindInputByName(joiner_inputs_, "encoder_out");
  joiner_decoder_index_ = FindInputByName(joiner_inputs_, "decoder_out");
  if (joiner_encoder_index_ == joiner_inputs_.size() ||
      joiner_decoder_index_ == joiner_inputs_.size()) {
    joiner_encoder_index_ = 0;
    joiner_decoder_index_ = 1;
  }
  joiner_dim_ = joiner_inputs_[joiner_encoder_index_].dim.back();
  if (joiner_inputs_[joiner_decoder_index_].dim.back() != joiner_dim_) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: joiner dim mismatch";
    return false;
  }
  return true;
}

bool OnlineTransducerModel::RunEncoder(const std::vector<float>& features,
                                       std::vector<float>* encoder_out,
                                       int32_t* num_frames, int32_t* out_dim) {
  if (!ok_ || encoder_out == nullptr || num_frames == nullptr ||
      out_dim == nullptr) {
    return false;
  }
  const size_t expected =
      static_cast<size_t>(chunk_size_) * static_cast<size_t>(feature_dim_);
  if (features.size() != expected) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: expected " << expected
                      << " feature values, got " << features.size();
    return false;
  }

  mnn_engine::Tensor& feature = encoder_inputs_[encoder_feature_index_];
  if (feature.data.size() != expected * sizeof(float)) {
    return false;
  }
  std::memcpy(feature.data.data(), features.data(), feature.data.size());

  std::vector<mnn_engine::Tensor> outputs;
  if (!encoder_.Forward(encoder_inputs_, &outputs)) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: encoder forward failed";
    return false;
  }
  if (outputs.size() != encoder_state_indices_.size() + 1) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: encoder returned "
                      << outputs.size() << " outputs, expected "
                      << encoder_state_indices_.size() + 1;
    return false;
  }

  const mnn_engine::Tensor& enc = outputs[0];
  if (enc.type != mnn_engine::DataType::kFloat32 || enc.dim.size() != 3) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: invalid encoder output";
    return false;
  }
  *num_frames = enc.dim[1];
  *out_dim = enc.dim[2];
  const float* enc_data = reinterpret_cast<const float*>(enc.data.data());
  encoder_out->assign(enc_data,
                      enc_data + enc.data.size() / sizeof(float));

  for (size_t i = 0; i < encoder_state_indices_.size(); ++i) {
    mnn_engine::Tensor& state = encoder_inputs_[encoder_state_indices_[i]];
    const mnn_engine::Tensor& next = outputs[i + 1];
    if (next.type != state.type || next.dim != state.dim) {
      RTC_LOG(LS_ERROR) << "OnlineTransducerModel: encoder state shape "
                           "mismatch for '"
                        << state.name << "'";
      return false;
    }
    state.data = next.data;
  }
  return true;
}

bool OnlineTransducerModel::RunDecoder(const std::vector<int32_t>& tokens,
                                       std::vector<float>* decoder_out) {
  if (!ok_ || decoder_out == nullptr ||
      tokens.size() != static_cast<size_t>(context_size_)) {
    return false;
  }
  mnn_engine::Tensor& input = decoder_inputs_[0];
  if (input.data.size() != tokens.size() * sizeof(int32_t)) {
    return false;
  }
  std::memcpy(input.data.data(), tokens.data(), input.data.size());

  std::vector<mnn_engine::Tensor> outputs;
  if (!decoder_.Forward(decoder_inputs_, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: decoder forward failed";
    return false;
  }
  const mnn_engine::Tensor& out = outputs[0];
  if (out.type != mnn_engine::DataType::kFloat32) {
    return false;
  }
  decoder_out->assign(
      reinterpret_cast<const float*>(out.data.data()),
      reinterpret_cast<const float*>(out.data.data()) + out.data.size() / 4);
  return true;
}

bool OnlineTransducerModel::RunJoiner(const float* encoder_frame,
                                      const std::vector<float>& decoder_out,
                                      std::vector<float>* logits) {
  if (!ok_ || encoder_frame == nullptr || logits == nullptr ||
      decoder_out.size() != static_cast<size_t>(joiner_dim_)) {
    return false;
  }
  mnn_engine::Tensor& enc_input = joiner_inputs_[joiner_encoder_index_];
  mnn_engine::Tensor& dec_input = joiner_inputs_[joiner_decoder_index_];
  if (enc_input.data.size() != static_cast<size_t>(joiner_dim_) * sizeof(float) ||
      dec_input.data.size() != static_cast<size_t>(joiner_dim_) * sizeof(float)) {
    return false;
  }
  std::memcpy(enc_input.data.data(), encoder_frame, enc_input.data.size());
  std::memcpy(dec_input.data.data(), decoder_out.data(), dec_input.data.size());

  std::vector<mnn_engine::Tensor> outputs;
  if (!joiner_.Forward(joiner_inputs_, &outputs) || outputs.empty()) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: joiner forward failed";
    return false;
  }
  const mnn_engine::Tensor& out = outputs[0];
  if (out.type != mnn_engine::DataType::kFloat32 ||
      out.data.size() != static_cast<size_t>(vocab_size_) * sizeof(float)) {
    RTC_LOG(LS_ERROR) << "OnlineTransducerModel: invalid joiner output";
    return false;
  }
  logits->assign(reinterpret_cast<const float*>(out.data.data()),
                 reinterpret_cast<const float*>(out.data.data()) +
                     vocab_size_);
  return true;
}

void OnlineTransducerModel::ResetStates() {
  if (!ok_) {
    return;
  }
  for (size_t i = 0; i < encoder_state_indices_.size(); ++i) {
    encoder_inputs_[encoder_state_indices_[i]] = encoder_state_templates_[i];
  }
}

}  // namespace soundproof
