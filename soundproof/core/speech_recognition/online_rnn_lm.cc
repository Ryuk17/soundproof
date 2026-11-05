#include "core/speech_recognition/online_rnn_lm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int32_t kLmRescoreMinChunk = 2;

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

OnlineRnnLm::OnlineRnnLm() = default;
OnlineRnnLm::~OnlineRnnLm() = default;

bool OnlineRnnLm::Init(const OnlineRnnLmConfig& config) {
  if (config.model_path.empty()) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: empty model path";
    return false;
  }

  mnn_engine::EngineConfig engine_config;
  engine_config.num_threads = config.num_threads > 0 ? config.num_threads : 1;
  if (!engine_.Init(config.model_path, engine_config)) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: cannot load model "
                      << config.model_path;
    return false;
  }

  const auto metadata = engine_.metadata();
  if (!ParseIntMetadata(metadata, "num_layers", &num_layers_) ||
      !ParseIntMetadata(metadata, "hidden_size", &hidden_size_)) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: missing num_layers/hidden_size";
    return false;
  }
  ParseIntMetadata(metadata, "sos_id", &sos_id_);
  scale_ = config.scale;
  shallow_fusion_ = config.shallow_fusion;

  const auto inputs = engine_.inputs();
  if (inputs.size() != 3) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: expected 3 inputs, got "
                      << inputs.size();
    return false;
  }
  inputs_.clear();
  token_index_ = inputs.size();
  state_h_index_ = inputs.size();
  state_c_index_ = inputs.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    mnn_engine::Tensor tensor = inputs[i];
    if (tensor.type == mnn_engine::DataType::kInt32) {
      if (token_index_ != inputs.size()) {
        return false;
      }
      token_index_ = i;
      tensor.dim = {1, 1};
      if (!AllocateData(&tensor)) {
        return false;
      }
    } else if (tensor.type == mnn_engine::DataType::kFloat32 &&
               tensor.dim.size() == 3) {
      if (tensor.dim[0] <= 0) {
        tensor.dim[0] = num_layers_;
      }
      if (tensor.dim[1] <= 0) {
        tensor.dim[1] = 1;
      }
      if (tensor.dim[2] <= 0) {
        tensor.dim[2] = hidden_size_;
      }
      if (!AllocateData(&tensor)) {
        return false;
      }
      if (state_h_index_ == inputs.size()) {
        state_h_index_ = i;
      } else if (state_c_index_ == inputs.size()) {
        state_c_index_ = i;
      } else {
        return false;
      }
    } else {
      RTC_LOG(LS_ERROR) << "OnlineRnnLm: unsupported input '" << tensor.name
                        << "'";
      return false;
    }
    inputs_.push_back(std::move(tensor));
  }
  if (token_index_ == inputs.size() || state_h_index_ == inputs.size() ||
      state_c_index_ == inputs.size()) {
    return false;
  }

  const std::vector<mnn_engine::Tensor> zero_states = {
      inputs_[state_h_index_], inputs_[state_c_index_]};
  if (!ScoreToken({sos_id_}, zero_states, &init_scores_, &init_states_)) {
    return false;
  }
  ok_ = true;
  return true;
}

bool OnlineRnnLm::ScoreToken(const std::vector<int32_t>& tokens,
                             const std::vector<mnn_engine::Tensor>& states,
                             std::vector<float>* scores,
                             std::vector<mnn_engine::Tensor>* next_states) {
  if (scores == nullptr || next_states == nullptr || tokens.empty()) {
    return false;
  }

  mnn_engine::Tensor& token_tensor = inputs_[token_index_];
  if (!token_tensor.dim.empty()) {
    token_tensor.dim = {1, static_cast<int>(tokens.size())};
  }
  if (!AllocateData(&token_tensor)) {
    return false;
  }
  std::memcpy(token_tensor.data.data(), tokens.data(), token_tensor.data.size());

  const size_t state_indices[2] = {state_h_index_, state_c_index_};
  for (size_t s = 0; s < 2; ++s) {
    mnn_engine::Tensor& dst = inputs_[state_indices[s]];
    if (s < states.size() &&
        states[s].data.size() == dst.data.size()) {
      std::memcpy(dst.data.data(), states[s].data.data(), dst.data.size());
    } else {
      std::fill(dst.data.begin(), dst.data.end(), 0);
    }
  }

  std::vector<mnn_engine::Tensor> outputs;
  if (!engine_.Forward(inputs_, &outputs) || outputs.size() < 3) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: forward failed";
    return false;
  }
  const mnn_engine::Tensor& score_tensor = outputs[0];
  if (score_tensor.type != mnn_engine::DataType::kFloat32 ||
      score_tensor.data.empty()) {
    return false;
  }
  const float* score_data =
      reinterpret_cast<const float*>(score_tensor.data.data());
  scores->assign(score_data,
                 score_data + score_tensor.data.size() / sizeof(float));

  next_states->clear();
  next_states->reserve(2);
  for (size_t s = 1; s <= 2; ++s) {
    const mnn_engine::Tensor& state = outputs[s];
    if (state.type != mnn_engine::DataType::kFloat32) {
      return false;
    }
    next_states->push_back(state);
  }
  return true;
}

bool OnlineRnnLm::ComputeScoreSF(Hypothesis* hyp) {
  if (!ok_ || hyp == nullptr || hyp->ys.empty()) {
    return false;
  }
  if (hyp->nn_lm_states.size() != 2) {
    hyp->nn_lm_scores = init_scores_;
    hyp->nn_lm_states = init_states_;
  }
  if (hyp->nn_lm_scores.empty()) {
    hyp->nn_lm_scores = init_scores_;
  }

  const int32_t token = hyp->ys.back();
  if (token < 0 ||
      static_cast<size_t>(token) >= hyp->nn_lm_scores.size()) {
    RTC_LOG(LS_ERROR) << "OnlineRnnLm: token " << token
                      << " out of LM score range";
    return false;
  }
  hyp->lm_log_prob += hyp->nn_lm_scores[static_cast<size_t>(token)] * scale_;

  std::vector<float> scores;
  std::vector<mnn_engine::Tensor> states;
  if (!ScoreToken({token}, hyp->nn_lm_states, &scores, &states)) {
    return false;
  }
  hyp->nn_lm_scores = std::move(scores);
  hyp->nn_lm_states = std::move(states);
  return true;
}

bool OnlineRnnLm::ComputeScore(int32_t context_size, Hypotheses* hyps) {
  if (!ok_ || hyps == nullptr) {
    return false;
  }

  bool ok = true;
  hyps->ForEach([&](Hypothesis* hyp) {
    if (!ok || hyp == nullptr) {
      return;
    }
    if (static_cast<int32_t>(hyp->ys.size()) <= context_size) {
      return;
    }
    const int32_t token_num = static_cast<int32_t>(hyp->ys.size()) -
                              context_size - hyp->cur_scored_pos - 1;
    if (token_num < 1) {
      return;
    }
    if (hyp->nn_lm_states.size() != 2) {
      hyp->nn_lm_states = init_states_;
    }
    if (token_num < kLmRescoreMinChunk) {
      return;
    }

    const std::vector<int32_t> tokens(
        hyp->ys.begin() + context_size + hyp->cur_scored_pos,
        hyp->ys.end() - 1);
    std::vector<float> scores;
    std::vector<mnn_engine::Tensor> states;
    if (!ScoreToken(tokens, hyp->nn_lm_states, &scores, &states) ||
        scores.empty()) {
      ok = false;
      return;
    }
    hyp->lm_log_prob = -scale_ * scores[0];
    hyp->nn_lm_states = std::move(states);
    hyp->cur_scored_pos += token_num;
  });
  return ok;
}

}  // namespace soundproof
