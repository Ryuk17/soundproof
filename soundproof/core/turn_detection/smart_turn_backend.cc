#include "core/turn_detection/smart_turn_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "feature/whisper_mel.h"
#include "mnn_engine/mnn_engine.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

bool FileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
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

struct SmartTurnBackend::Impl {
  mnn_engine::Engine engine;
  mnn_engine::Tensor input;
  std::vector<mnn_engine::Tensor> inputs;
  int output_index = 0;
  std::unique_ptr<WhisperMel> mel;
  std::vector<float> features;
};

SmartTurnBackend::SmartTurnBackend() : impl_(new Impl()) {}
SmartTurnBackend::~SmartTurnBackend() = default;

TurnStatus SmartTurnBackend::Init(const TurnModelConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: model_path is empty";
    return TurnStatus::kInvalidInput;
  }
  if (!FileExists(config.model_path)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: model not found: "
                      << config.model_path;
    return TurnStatus::kModelLoadFailed;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  engine_config.low_memory = true;
  if (!impl_->engine.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: failed to load "
                      << config.model_path;
    return TurnStatus::kModelLoadFailed;
  }

  const std::vector<mnn_engine::Tensor> model_inputs = impl_->engine.inputs();
  if (model_inputs.size() != 1) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: expected 1 input, got "
                      << model_inputs.size();
    return TurnStatus::kUnsupportedModel;
  }
  const mnn_engine::Tensor& descriptor = model_inputs[0];
  if (descriptor.type != mnn_engine::DataType::kFloat32 ||
      descriptor.dim.size() != 3 ||
      (descriptor.dim[1] > 0 &&
       descriptor.dim[1] != WhisperMel::kNumMels) ||
      (descriptor.dim[2] > 0 &&
       descriptor.dim[2] != WhisperMel::kNumFrames) ||
      (descriptor.dim[0] > 0 && descriptor.dim[0] != 1)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: unsupported input shape";
    return TurnStatus::kUnsupportedModel;
  }
  impl_->input = descriptor;
  impl_->input.dim = {1, WhisperMel::kNumMels, WhisperMel::kNumFrames};

  const std::vector<std::string> output_names = impl_->engine.output_names();
  impl_->output_index = -1;
  for (size_t i = 0; i < output_names.size(); ++i) {
    if (output_names[i] == "logits") {
      impl_->output_index = static_cast<int>(i);
      break;
    }
  }
  if (impl_->output_index < 0 && output_names.size() == 1) {
    impl_->output_index = 0;
  }
  if (impl_->output_index < 0) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: no 'logits' output";
    return TurnStatus::kUnsupportedModel;
  }

  impl_->mel = std::make_unique<WhisperMel>();
  if (!impl_->mel->ok()) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: cannot initialize mel front-end";
    return TurnStatus::kFeatureFailed;
  }
  impl_->features.resize(static_cast<size_t>(WhisperMel::kNumMels) *
                         WhisperMel::kNumFrames);
  impl_->inputs.clear();
  impl_->inputs.push_back(impl_->input);
  context_samples_ = WhisperMel::kContextSamples;

  RTC_LOG(LS_INFO) << "SmartTurnBackend: loaded " << config.model_path
                   << ", context=" << context_samples_;
  return TurnStatus::kOk;
}

TurnStatus SmartTurnBackend::Predict(const float* audio, int samples,
                                     float* probability) {
  if (impl_->mel == nullptr) {
    return TurnStatus::kNotInitialized;
  }
  if (audio == nullptr || probability == nullptr ||
      samples != context_samples_) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: invalid audio (" << samples
                      << " samples, expected " << context_samples_ << ")";
    return TurnStatus::kInvalidInput;
  }

  if (!impl_->mel->Compute(audio, &impl_->features)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: feature extraction failed";
    return TurnStatus::kFeatureFailed;
  }

  mnn_engine::Tensor& input = impl_->inputs[0];
  input.dim = {1, WhisperMel::kNumMels, WhisperMel::kNumFrames};
  if (!AllocateData(&input)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: cannot allocate model input";
    return TurnStatus::kInferenceFailed;
  }
  std::memcpy(input.data.data(), impl_->features.data(),
              impl_->features.size() * sizeof(float));

  std::vector<mnn_engine::Tensor> outputs;
  if (!impl_->engine.Forward(impl_->inputs, &outputs)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: forward failed";
    return TurnStatus::kInferenceFailed;
  }
  if (impl_->output_index < 0 ||
      static_cast<size_t>(impl_->output_index) >= outputs.size()) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: missing output";
    return TurnStatus::kInferenceFailed;
  }
  const mnn_engine::Tensor& output = outputs[impl_->output_index];
  if (output.type != mnn_engine::DataType::kFloat32 ||
      output.data.size() < sizeof(float)) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: unexpected output tensor";
    return TurnStatus::kInferenceFailed;
  }

  float value = 0.0f;
  std::memcpy(&value, output.data.data(), sizeof(float));
  if (!std::isfinite(value) || value < -1e-3f || value > 1.0f + 1e-3f) {
    RTC_LOG(LS_ERROR) << "SmartTurnBackend: unexpected probability " << value;
    return TurnStatus::kInferenceFailed;
  }
  *probability = std::min(1.0f, std::max(0.0f, value));
  return TurnStatus::kOk;
}

}  // namespace soundproof
