#include "core/voice_activity_detection/silero_vad.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "core/voice_activity_detection/vad_decision_filter.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int kSampleRate = 16000;
constexpr float kInt16ToFloat = 1.0f / 32768.0f;

// v5 consumes 512 new samples plus 64 samples of overlap from the previous
// window; v4 consumes a plain 512-sample window.
constexpr size_t kWindowSizeV5 = 576;
constexpr size_t kWindowSizeV4 = 512;
constexpr size_t kWindowShift = 512;

// Resolves dynamic dimensions (<= 0) so a tensor can be submitted to the
// engine. The last dynamic axis takes |last_value| (the window length for
// "input", the batch size 1 for state tensors); earlier dynamic axes become 1.
void ResolveDynamicDims(mnn_engine::Tensor* tensor, int last_value) {
  int last_dynamic = -1;
  for (size_t i = 0; i < tensor->dim.size(); ++i) {
    if (tensor->dim[i] <= 0) {
      last_dynamic = static_cast<int>(i);
    }
  }
  for (size_t i = 0; i < tensor->dim.size(); ++i) {
    if (tensor->dim[i] <= 0) {
      tensor->dim[i] =
          (static_cast<int>(i) == last_dynamic) ? last_value : 1;
    }
  }
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

bool HasNames(const std::vector<mnn_engine::Tensor>& inputs,
              const std::vector<std::string>& expected) {
  if (inputs.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (inputs[i].name != expected[i]) {
      return false;
    }
  }
  return true;
}

bool HasNames(const std::vector<std::string>& names,
              const std::vector<std::string>& expected) {
  return names == expected;
}

}  // namespace

SileroVad::SileroVad(const std::string& model_path, int num_threads) {
  ok_ = Init(model_path, num_threads);
}

SileroVad::~SileroVad() = default;

bool SileroVad::Init(const std::string& model_path, int num_threads) {
  if (model_path.empty()) {
    RTC_LOG(LS_WARNING) << "SileroVad: empty model path";
    return false;
  }

  mnn_engine::EngineConfig config;
  config.num_threads = num_threads > 0 ? num_threads : 1;
  if (!engine_.Init(model_path, config)) {
    RTC_LOG(LS_ERROR) << "SileroVad: failed to load model: " << model_path;
    return false;
  }

  const std::vector<mnn_engine::Tensor> model_inputs = engine_.inputs();
  const std::vector<std::string> output_names = engine_.output_names();

  std::vector<std::string> state_names;
  if (model_inputs.size() == 3 && output_names.size() == 2) {
    // v5: input, state, sr -> output, stateN
    static const std::vector<std::string> kInputs = {"input", "state", "sr"};
    static const std::vector<std::string> kOutputs = {"output", "stateN"};
    if (!HasNames(model_inputs, kInputs) || !HasNames(output_names, kOutputs)) {
      RTC_LOG(LS_ERROR) << "SileroVad: unsupported v5 model I/O names";
      return false;
    }
    is_v5_ = true;
    window_size_ = kWindowSizeV5;
    state_names = {"state"};
  } else if (model_inputs.size() == 4 && output_names.size() == 3) {
    // v4: input, sr, h, c -> output, hn, cn
    static const std::vector<std::string> kInputs = {"input", "sr", "h", "c"};
    static const std::vector<std::string> kOutputs = {"output", "hn", "cn"};
    if (!HasNames(model_inputs, kInputs) || !HasNames(output_names, kOutputs)) {
      RTC_LOG(LS_ERROR) << "SileroVad: unsupported v4 model I/O names";
      return false;
    }
    is_v5_ = false;
    window_size_ = kWindowSizeV4;
    state_names = {"h", "c"};
  } else {
    RTC_LOG(LS_ERROR) << "SileroVad: unsupported model (inputs="
                      << model_inputs.size()
                      << ", outputs=" << output_names.size() << ")";
    return false;
  }
  window_shift_ = kWindowShift;

  forward_inputs_.clear();
  forward_inputs_.reserve(model_inputs.size());
  state_indices_.clear();
  bool saw_input = false;
  bool saw_sample_rate = false;

  for (size_t i = 0; i < model_inputs.size(); ++i) {
    mnn_engine::Tensor tensor = model_inputs[i];
    if (tensor.name == "input") {
      if (saw_input || tensor.type != mnn_engine::DataType::kFloat32) {
        RTC_LOG(LS_ERROR) << "SileroVad: invalid 'input' tensor";
        return false;
      }
      saw_input = true;
      input_index_ = i;
      ResolveDynamicDims(&tensor, static_cast<int>(window_size_));
      if (tensor.dim.empty() ||
          tensor.dim.back() != static_cast<int>(window_size_)) {
        RTC_LOG(LS_ERROR) << "SileroVad: model window size mismatch";
        return false;
      }
      if (!AllocateData(&tensor)) {
        RTC_LOG(LS_ERROR) << "SileroVad: invalid 'input' shape";
        return false;
      }
    } else if (tensor.name == "sr") {
      if (saw_sample_rate || tensor.type != mnn_engine::DataType::kInt32) {
        RTC_LOG(LS_ERROR) << "SileroVad: invalid 'sr' tensor";
        return false;
      }
      saw_sample_rate = true;
      sample_rate_index_ = i;
      ResolveDynamicDims(&tensor, 1);
      if (!AllocateData(&tensor) ||
          tensor.data.size() != sizeof(int32_t)) {
        RTC_LOG(LS_ERROR) << "SileroVad: 'sr' must be a scalar int32";
        return false;
      }
      *reinterpret_cast<int32_t*>(tensor.data.data()) = kSampleRate;
    } else {
      if (tensor.type != mnn_engine::DataType::kFloat32 ||
          std::find(state_names.begin(), state_names.end(), tensor.name) ==
              state_names.end()) {
        RTC_LOG(LS_ERROR) << "SileroVad: unexpected input '" << tensor.name
                          << "'";
        return false;
      }
      ResolveDynamicDims(&tensor, 1);
      if (!AllocateData(&tensor)) {
        RTC_LOG(LS_ERROR) << "SileroVad: invalid state shape";
        return false;
      }
      state_indices_.push_back(i);
    }
    forward_inputs_.push_back(std::move(tensor));
  }

  if (!saw_input || !saw_sample_rate ||
      state_indices_.size() != state_names.size()) {
    RTC_LOG(LS_ERROR) << "SileroVad: incomplete model inputs";
    return false;
  }

  state_templates_.clear();
  state_templates_.reserve(state_indices_.size());
  for (size_t index : state_indices_) {
    state_templates_.push_back(forward_inputs_[index]);
  }

  pending_.clear();
  last_probability_ = 0.0f;
  ok_ = true;
  RTC_LOG(LS_INFO) << "SileroVad: ready (v5=" << (is_v5_ ? 1 : 0)
                   << ", window=" << window_size_ << ")";
  return true;
}

bool SileroVad::ProcessChunk(const int16_t* audio_in, size_t samples,
                             int sample_rate_hz,
                             VadDecisionFilter* decision_filter) {
  if (!ok_ || audio_in == nullptr) {
    return false;
  }
  if (sample_rate_hz != kSampleRate) {
    RTC_LOG(LS_ERROR) << "SileroVad: unsupported sample rate "
                      << sample_rate_hz << " (expected " << kSampleRate
                      << ")";
    return false;
  }
  if (samples == 0) {
    return true;
  }

  pending_.reserve(pending_.size() + samples);
  for (size_t i = 0; i < samples; ++i) {
    pending_.push_back(static_cast<float>(audio_in[i]) * kInt16ToFloat);
  }

  size_t offset = 0;
  while (pending_.size() - offset >= window_size_) {
    if (!RunWindow(pending_.data() + offset)) {
      ok_ = false;
      return false;
    }
    if (decision_filter != nullptr) {
      decision_filter->Update(last_probability_,
                              static_cast<int64_t>(window_shift_));
    }
    offset += window_shift_;
  }
  if (offset > 0) {
    pending_.erase(pending_.begin(),
                   pending_.begin() + static_cast<std::ptrdiff_t>(offset));
  }
  return true;
}

bool SileroVad::RunWindow(const float* window) {
  mnn_engine::Tensor& input = forward_inputs_[input_index_];
  if (input.data.size() < window_size_ * sizeof(float)) {
    return false;
  }
  std::memcpy(input.data.data(), window, window_size_ * sizeof(float));

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward(forward_inputs_, &outputs)) {
    RTC_LOG(LS_ERROR) << "SileroVad: inference failed";
    return false;
  }
  if (outputs.size() != state_indices_.size() + 1) {
    RTC_LOG(LS_ERROR) << "SileroVad: unexpected output count "
                      << outputs.size();
    return false;
  }
  if (outputs[0].type != mnn_engine::DataType::kFloat32 ||
      outputs[0].data.size() < sizeof(float)) {
    RTC_LOG(LS_ERROR) << "SileroVad: invalid probability output";
    return false;
  }
  last_probability_ =
      *reinterpret_cast<const float*>(outputs[0].data.data());

  for (size_t i = 0; i < state_indices_.size(); ++i) {
    mnn_engine::Tensor& state = forward_inputs_[state_indices_[i]];
    const mnn_engine::Tensor& next = outputs[i + 1];
    if (next.type != state.type || next.dim != state.dim) {
      RTC_LOG(LS_ERROR) << "SileroVad: state shape mismatch";
      return false;
    }
    state.data = next.data;
  }
  return true;
}

void SileroVad::Reset() {
  if (!ok_) {
    return;
  }
  for (size_t i = 0; i < state_indices_.size(); ++i) {
    forward_inputs_[state_indices_[i]] = state_templates_[i];
  }
  pending_.clear();
  last_probability_ = 0.0f;
}

}  // namespace soundproof
