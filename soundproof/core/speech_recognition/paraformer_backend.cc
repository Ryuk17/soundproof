#include "core/speech_recognition/paraformer_backend.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "core/speech_recognition/paraformer_utils.h"
#include "utils/symbol_table.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr float kCifThreshold = 1.0f;

void ScaleAddInPlace(const float* src, int32_t n, float scale, float* dst) {
  for (int32_t i = 0; i < n; ++i) {
    dst[i] += src[i] * scale;
  }
}

void Scale(const float* src, int32_t n, float scale, float* dst) {
  for (int32_t i = 0; i < n; ++i) {
    dst[i] = src[i] * scale;
  }
}

}  // namespace

bool ParaformerBackend::Init(const AsrBackendConfig& config) {
  if (config.encoder_path.empty() || config.decoder_path.empty()) {
    RTC_LOG(LS_ERROR)
        << "ParaformerBackend: encoder and decoder paths are required";
    return false;
  }
  if (config.lm.decoding_method != 0 || !config.lm.model_path.empty()) {
    RTC_LOG(LS_ERROR)
        << "ParaformerBackend: beam search and LM are not supported";
    return false;
  }
  ParaformerModelConfig model_config;
  model_config.encoder_path = config.encoder_path;
  model_config.decoder_path = config.decoder_path;
  model_config.num_threads = config.num_threads;
  if (!model_.Init(model_config)) {
    return false;
  }
  if (model_.LfrWindowSize() <= 0 || model_.LfrWindowShift() <= 0) {
    return false;
  }
  ok_ = true;
  return true;
}

bool ParaformerBackend::DecodeChunk(const float* features,
                                    int32_t processed_frames,
                                    AsrResult* result) {
  if (!ok_ || features == nullptr || result == nullptr) {
    return false;
  }

  // 1. LFR, CMVN, positional encoding.
  std::vector<float> frames =
      ApplyLfr(features, kChunkSize, kRawFeatureDim, model_.LfrWindowSize(),
               model_.LfrWindowShift());
  if (frames.empty()) {
    RTC_LOG(LS_ERROR) << "ParaformerBackend: LFR produced no frames";
    return false;
  }
  const int32_t post_dim = model_.FeatureDim();
  ApplyCMVN(&frames, model_.NegativeMean(), model_.InverseStdDev());
  ApplyPositionalEncoding(&frames, post_dim,
                          processed_frames / model_.LfrWindowShift());

  // 2. Left/right context cache around the current chunk.
  const int32_t cache_frames = kLeftChunkSize + kRightChunkSize;
  const size_t cache_size = static_cast<size_t>(cache_frames) * post_dim;
  if (feature_cache_.empty()) {
    feature_cache_.assign(cache_size, 0.0f);
  }
  if (feature_cache_.size() != cache_size) {
    RTC_LOG(LS_ERROR) << "ParaformerBackend: feature cache size mismatch";
    return false;
  }
  std::vector<float> input = feature_cache_;
  input.insert(input.end(), frames.begin(), frames.end());
  std::copy(input.end() - cache_size, input.end(), feature_cache_.begin());
  const int32_t num_frames = static_cast<int32_t>(input.size() / post_dim);

  // 3. Encoder.
  std::vector<float> encoder_out;
  std::vector<float> alpha;
  int32_t encoder_frames = 0;
  int32_t encoder_dim = 0;
  if (!model_.ForwardEncoder(input, num_frames, &encoder_out, &encoder_frames,
                             &encoder_dim, &alpha)) {
    return false;
  }
  if (static_cast<int32_t>(alpha.size()) < encoder_frames ||
      encoder_dim != model_.EncoderOutputSize()) {
    return false;
  }
  std::fill(alpha.begin(), alpha.begin() + kLeftChunkSize, 0.0f);
  std::fill(alpha.begin() + encoder_frames - kRightChunkSize,
            alpha.begin() + encoder_frames, 0.0f);

  // 4. CIF: integrate alphas and emit acoustic embeddings.
  if (encoder_out_cache_.empty()) {
    encoder_out_cache_.resize(static_cast<size_t>(encoder_dim), 0.0f);
  }
  if (alpha_cache_.empty()) {
    alpha_cache_.push_back(0.0f);
  }

  std::vector<float> acoustic_embedding;
  acoustic_embedding.reserve(
      static_cast<size_t>(encoder_frames) * encoder_dim);
  float integrate = alpha_cache_[0];
  const float* enc = encoder_out.data();

  for (int32_t i = 0; i < encoder_frames; ++i) {
    const float this_alpha = alpha[i];
    if (integrate + this_alpha < kCifThreshold) {
      integrate += this_alpha;
      ScaleAddInPlace(enc + static_cast<size_t>(i) * encoder_dim, encoder_dim,
                      this_alpha, encoder_out_cache_.data());
      continue;
    }
    ScaleAddInPlace(enc + static_cast<size_t>(i) * encoder_dim, encoder_dim,
                    kCifThreshold - integrate, encoder_out_cache_.data());
    acoustic_embedding.insert(acoustic_embedding.end(),
                              encoder_out_cache_.begin(),
                              encoder_out_cache_.end());
    integrate += this_alpha - kCifThreshold;
    Scale(enc + static_cast<size_t>(i) * encoder_dim, encoder_dim, integrate,
          encoder_out_cache_.data());
  }
  alpha_cache_[0] = integrate;

  const int32_t trailing_silence =
      processed_frames + (kChunkSize - 1) - last_non_blank_frame_index_;

  if (acoustic_embedding.empty()) {
    result->tokens = tokens_;
    result->timestamps.clear();
    result->trailing_silence_frames = trailing_silence;
    return true;
  }

  // 5. Decoder.
  const int32_t num_tokens =
      static_cast<int32_t>(acoustic_embedding.size() / encoder_dim);
  std::vector<int32_t> sample_ids;
  if (!model_.ForwardDecoder(encoder_out, encoder_frames, encoder_dim,
                             acoustic_embedding, num_tokens, &sample_ids)) {
    return false;
  }

  bool non_blank_detected = false;
  const int32_t count =
      std::min(num_tokens, static_cast<int32_t>(sample_ids.size()));
  for (int32_t i = 0; i < count; ++i) {
    if (sample_ids[i] == 0) {
      continue;
    }
    non_blank_detected = true;
    tokens_.push_back(sample_ids[i]);
  }
  if (non_blank_detected) {
    last_non_blank_frame_index_ = processed_frames;
  }

  result->tokens = tokens_;
  result->timestamps.clear();
  result->trailing_silence_frames =
      processed_frames + (kChunkSize - 1) - last_non_blank_frame_index_;
  return true;
}

std::string ParaformerBackend::TextFromTokens(
    const std::vector<int32_t>& tokens, const SymbolTable& symbols) const {
  return ParaformerTextFromTokens(tokens, symbols);
}

void ParaformerBackend::ResetSegment() {
  tokens_.clear();
  last_non_blank_frame_index_ = 0;
  encoder_out_cache_.clear();
  alpha_cache_.clear();
  model_.ResetStates();
  // The feature cache is kept across segments (sherpa behavior).
}

void ParaformerBackend::Reset() {
  tokens_.clear();
  last_non_blank_frame_index_ = 0;
  feature_cache_.clear();
  encoder_out_cache_.clear();
  alpha_cache_.clear();
  model_.ResetStates();
}

bool ParaformerBackend::SetBlankPenalty(float blank_penalty) {
  // Paraformer decoding does not use a blank penalty.
  return blank_penalty == 0.0f;
}

}  // namespace soundproof
