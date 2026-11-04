#include "core/speech_recognition/paraformer_model.h"

#include <cmath>
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
    RTC_LOG(LS_ERROR) << "ParaformerModel: missing metadata '" << key << "'";
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0') {
    RTC_LOG(LS_ERROR) << "ParaformerModel: invalid metadata '" << key
                      << "'=" << it->second;
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

bool ParseFloatVectorMetadata(
    const std::map<std::string, std::string>& metadata, const char* key,
    std::vector<float>* values) {
  const auto it = metadata.find(key);
  if (it == metadata.end()) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: missing metadata '" << key << "'";
    return false;
  }
  values->clear();
  const std::string& text = it->second;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t comma = text.find(',', begin);
    const std::string item = text.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin);
    if (!item.empty()) {
      values->push_back(std::strtof(item.c_str(), nullptr));
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  return !values->empty();
}

bool ResolveDynamicDims(mnn_engine::Tensor* tensor, int value = 1) {
  for (int& d : tensor->dim) {
    if (d <= 0) {
      d = value;
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

bool PrepareTensor(mnn_engine::Tensor* tensor, const std::vector<int>& dims) {
  tensor->dim = dims;
  return AllocateData(tensor);
}

int32_t LastDim(const mnn_engine::Tensor& tensor) {
  return tensor.dim.empty() ? 0 : tensor.dim.back();
}

}  // namespace

ParaformerModel::ParaformerModel() = default;
ParaformerModel::~ParaformerModel() = default;

bool ParaformerModel::Init(const ParaformerModelConfig& config) {
  if (config.encoder_path.empty() || config.decoder_path.empty()) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: encoder/decoder paths are required";
    return false;
  }
  if (!InitEncoder(config.encoder_path, config.num_threads)) {
    return false;
  }
  if (!InitDecoder(config.decoder_path, config.num_threads)) {
    return false;
  }
  ok_ = true;
  return true;
}

bool ParaformerModel::InitEncoder(const std::string& path, int num_threads) {
  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!encoder_.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: cannot load encoder " << path;
    return false;
  }

  const auto metadata = encoder_.metadata();
  if (!ParseIntMetadata(metadata, "vocab_size", &vocab_size_) ||
      !ParseIntMetadata(metadata, "lfr_window_size", &lfr_window_size_) ||
      !ParseIntMetadata(metadata, "lfr_window_shift", &lfr_window_shift_) ||
      !ParseIntMetadata(metadata, "encoder_output_size",
                        &encoder_output_size_) ||
      !ParseIntMetadata(metadata, "decoder_num_blocks", &decoder_num_blocks_) ||
      !ParseIntMetadata(metadata, "decoder_kernel_size",
                        &decoder_kernel_size_) ||
      !ParseFloatVectorMetadata(metadata, "neg_mean", &negative_mean_) ||
      !ParseFloatVectorMetadata(metadata, "inv_stddev", &inverse_stddev_)) {
    return false;
  }
  if (negative_mean_.size() != inverse_stddev_.size()) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: CMVN size mismatch";
    return false;
  }
  const float scale = std::sqrt(static_cast<float>(encoder_output_size_));
  for (float& v : inverse_stddev_) {
    v *= scale;
  }

  const auto inputs = encoder_.inputs();
  if (inputs.size() != 2) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: encoder expects 2 inputs, got "
                      << inputs.size();
    return false;
  }
  encoder_inputs_.clear();
  bool has_features = false;
  bool has_length = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    mnn_engine::Tensor tensor = inputs[i];
    if (tensor.type == mnn_engine::DataType::kFloat32 &&
        tensor.dim.size() == 3) {
      if (has_features) {
        return false;
      }
      has_features = true;
      encoder_feature_index_ = i;
      tensor.dim[0] = 1;
      if (tensor.dim.size() > 1 && tensor.dim[1] <= 0) {
        tensor.dim[1] = 1;  // placeholder, set per call
      }
      if (LastDim(tensor) <= 0) {
        tensor.dim[2] = FeatureDim();
      } else if (LastDim(tensor) != FeatureDim()) {
        RTC_LOG(LS_ERROR) << "ParaformerModel: feature dim "
                          << LastDim(tensor) << " != CMVN dim " << FeatureDim();
        return false;
      }
      if (!AllocateData(&tensor)) {
        return false;
      }
    } else if (tensor.type == mnn_engine::DataType::kInt32 &&
               tensor.dim.size() == 1) {
      if (has_length) {
        return false;
      }
      has_length = true;
      encoder_length_index_ = i;
      if (!ResolveDynamicDims(&tensor) || !AllocateData(&tensor)) {
        return false;
      }
    } else {
      RTC_LOG(LS_ERROR) << "ParaformerModel: unexpected encoder input '"
                        << tensor.name << "'";
      return false;
    }
    encoder_inputs_.push_back(std::move(tensor));
  }
  if (!has_features || !has_length) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: incomplete encoder inputs";
    return false;
  }
  return true;
}

bool ParaformerModel::InitDecoder(const std::string& path, int num_threads) {
  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!decoder_.Init(path, config)) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: cannot load decoder " << path;
    return false;
  }

  const auto inputs = decoder_.inputs();
  const size_t expected = 4 + static_cast<size_t>(decoder_num_blocks_);
  if (inputs.size() != expected) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: decoder expects " << expected
                      << " inputs, got " << inputs.size();
    return false;
  }

  decoder_inputs_.clear();
  decoder_state_indices_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    mnn_engine::Tensor tensor = inputs[i];
    if (i < 4) {
      if (!ResolveDynamicDims(&tensor) || !AllocateData(&tensor)) {
        RTC_LOG(LS_ERROR) << "ParaformerModel: bad decoder input '"
                          << tensor.name << "'";
        return false;
      }
      if ((i == 0 || i == 2) &&
          (tensor.type != mnn_engine::DataType::kFloat32 ||
           tensor.dim.size() != 3)) {
        RTC_LOG(LS_ERROR) << "ParaformerModel: decoder input " << i
                          << " must be a 3-D float tensor";
        return false;
      }
      if ((i == 1 || i == 3) &&
          (tensor.type != mnn_engine::DataType::kInt32 ||
           tensor.dim.size() != 1)) {
        RTC_LOG(LS_ERROR) << "ParaformerModel: decoder input " << i
                          << " must be a 1-D int32 tensor";
        return false;
      }
    } else {
      if (tensor.type != mnn_engine::DataType::kFloat32 ||
          tensor.dim.size() != 3) {
        RTC_LOG(LS_ERROR) << "ParaformerModel: unsupported decoder state '"
                          << tensor.name << "'";
        return false;
      }
      if (tensor.dim[1] <= 0) {
        tensor.dim[1] = encoder_output_size_;
      }
      if (tensor.dim[2] <= 0) {
        tensor.dim[2] = decoder_kernel_size_ - 1;
      }
      ResolveDynamicDims(&tensor);
      if (!AllocateData(&tensor)) {
        return false;
      }
      decoder_state_indices_.push_back(i);
    }
    decoder_inputs_.push_back(std::move(tensor));
  }

  decoder_state_templates_.clear();
  for (size_t index : decoder_state_indices_) {
    decoder_state_templates_.push_back(decoder_inputs_[index]);
  }
  return true;
}

bool ParaformerModel::ForwardEncoder(const std::vector<float>& features,
                                     int32_t num_frames,
                                     std::vector<float>* encoder_out,
                                     int32_t* out_frames, int32_t* out_dim,
                                     std::vector<float>* alpha) {
  if (!ok_ || encoder_out == nullptr || out_frames == nullptr ||
      out_dim == nullptr || alpha == nullptr || num_frames <= 0) {
    return false;
  }
  const int32_t feat_dim = FeatureDim();
  if (features.size() !=
      static_cast<size_t>(num_frames) * static_cast<size_t>(feat_dim)) {
    return false;
  }

  mnn_engine::Tensor& feature_tensor = encoder_inputs_[encoder_feature_index_];
  if (!PrepareTensor(&feature_tensor, {1, num_frames, feat_dim})) {
    return false;
  }
  std::memcpy(feature_tensor.data.data(), features.data(),
              feature_tensor.data.size());

  mnn_engine::Tensor& length_tensor = encoder_inputs_[encoder_length_index_];
  if (length_tensor.data.size() < sizeof(int32_t)) {
    return false;
  }
  *reinterpret_cast<int32_t*>(length_tensor.data.data()) = num_frames;

  std::vector<mnn_engine::Tensor> outputs;
  if (!encoder_.Forward(encoder_inputs_, &outputs) || outputs.size() < 3) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: encoder forward failed";
    return false;
  }
  const mnn_engine::Tensor& enc = outputs[0];
  const mnn_engine::Tensor& alpha_tensor = outputs[2];
  if (enc.type != mnn_engine::DataType::kFloat32 || enc.dim.size() != 3 ||
      alpha_tensor.type != mnn_engine::DataType::kFloat32) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: invalid encoder outputs";
    return false;
  }

  *out_frames = enc.dim[1];
  *out_dim = enc.dim[2];
  const float* enc_data = reinterpret_cast<const float*>(enc.data.data());
  encoder_out->assign(enc_data, enc_data + enc.data.size() / sizeof(float));

  const size_t alpha_count = alpha_tensor.data.size() / sizeof(float);
  const float* alpha_data =
      reinterpret_cast<const float*>(alpha_tensor.data.data());
  if (alpha_count < static_cast<size_t>(*out_frames)) {
    return false;
  }
  alpha->assign(alpha_data, alpha_data + *out_frames);
  return true;
}

bool ParaformerModel::ForwardDecoder(
    const std::vector<float>& encoder_out, int32_t encoder_frames,
    int32_t encoder_dim, const std::vector<float>& acoustic_embedding,
    int32_t num_tokens, std::vector<int32_t>* sample_ids) {
  if (!ok_ || sample_ids == nullptr || num_tokens < 0 ||
      encoder_dim != encoder_output_size_ ||
      encoder_out.size() != static_cast<size_t>(encoder_frames) *
                               static_cast<size_t>(encoder_dim) ||
      acoustic_embedding.size() != static_cast<size_t>(num_tokens) *
                                      static_cast<size_t>(encoder_dim)) {
    return false;
  }

  mnn_engine::Tensor& enc_tensor = decoder_inputs_[decoder_encoder_out_index_];
  if (!PrepareTensor(&enc_tensor, {1, encoder_frames, encoder_dim})) {
    return false;
  }
  std::memcpy(enc_tensor.data.data(), encoder_out.data(),
              enc_tensor.data.size());

  mnn_engine::Tensor& enc_len = decoder_inputs_[decoder_encoder_len_index_];
  *reinterpret_cast<int32_t*>(enc_len.data.data()) = encoder_frames;

  mnn_engine::Tensor& ac_tensor = decoder_inputs_[decoder_acoustic_index_];
  if (!PrepareTensor(&ac_tensor, {1, num_tokens, encoder_dim})) {
    return false;
  }
  std::memcpy(ac_tensor.data.data(), acoustic_embedding.data(),
              ac_tensor.data.size());

  mnn_engine::Tensor& ac_len = decoder_inputs_[decoder_acoustic_len_index_];
  *reinterpret_cast<int32_t*>(ac_len.data.data()) = num_tokens;

  std::vector<mnn_engine::Tensor> outputs;
  if (!decoder_.Forward(decoder_inputs_, &outputs) ||
      outputs.size() < 2 + decoder_state_indices_.size()) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: decoder forward failed";
    return false;
  }

  const mnn_engine::Tensor& ids = outputs[1];
  if (ids.type != mnn_engine::DataType::kInt32) {
    RTC_LOG(LS_ERROR) << "ParaformerModel: sample_ids is not int32";
    return false;
  }
  const int32_t* ids_data = reinterpret_cast<const int32_t*>(ids.data.data());
  sample_ids->assign(ids_data, ids_data + ids.data.size() / sizeof(int32_t));

  for (size_t i = 0; i < decoder_state_indices_.size(); ++i) {
    mnn_engine::Tensor& state = decoder_inputs_[decoder_state_indices_[i]];
    const mnn_engine::Tensor& next = outputs[2 + i];
    if (next.type != state.type || next.dim != state.dim) {
      RTC_LOG(LS_ERROR) << "ParaformerModel: decoder state shape mismatch";
      return false;
    }
    state.data = next.data;
  }
  return true;
}

void ParaformerModel::ResetStates() {
  if (!ok_) {
    return;
  }
  for (size_t i = 0; i < decoder_state_indices_.size(); ++i) {
    decoder_inputs_[decoder_state_indices_[i]] = decoder_state_templates_[i];
  }
}

}  // namespace soundproof
