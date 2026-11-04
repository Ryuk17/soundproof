#include "core/speech_recognition/transducer_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/speech_recognition/transducer_decoder.h"
#include "utils/symbol_table.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int32_t kSubsamplingFactor = 4;

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool IsTransducerModelType(const std::string& type) {
  return type.empty() || type == "zipformer" || type == "zipformer2" ||
         type == "conformer" || type == "lstm" || type == "ebranchformer";
}

}  // namespace

TransducerBackend::TransducerBackend() = default;
TransducerBackend::~TransducerBackend() = default;

bool TransducerBackend::Init(const AsrBackendConfig& config) {
  if (config.encoder_path.empty() || config.decoder_path.empty() ||
      config.joiner_path.empty()) {
    RTC_LOG(LS_ERROR)
        << "TransducerBackend: encoder, decoder and joiner paths are required";
    return false;
  }

  TransducerModelConfig model_config;
  model_config.encoder_path = config.encoder_path;
  model_config.decoder_path = config.decoder_path;
  model_config.joiner_path = config.joiner_path;
  model_config.num_threads = config.num_threads;
  if (!model_.Init(model_config)) {
    return false;
  }

  model_type_ = ToLower(model_.ModelType());
  if (!IsTransducerModelType(model_type_)) {
    RTC_LOG(LS_ERROR) << "TransducerBackend: unsupported model type '"
                      << model_type_ << "'";
    return false;
  }

  decoder_ = std::make_unique<TransducerDecoder>(&model_, config.unk_id,
                                                 config.blank_penalty);
  return true;
}

bool TransducerBackend::DecodeChunk(const float* features,
                                    int32_t /*processed_frames*/,
                                    AsrResult* result) {
  if (decoder_ == nullptr || features == nullptr || result == nullptr) {
    return false;
  }
  const size_t expected = static_cast<size_t>(model_.ChunkSize()) *
                          static_cast<size_t>(model_.FeatureDim());
  std::vector<float> input(features, features + expected);

  std::vector<float> encoder_out;
  int32_t num_frames = 0;
  int32_t frame_dim = 0;
  if (!model_.RunEncoder(input, &encoder_out, &num_frames, &frame_dim)) {
    return false;
  }
  if (!decoder_->Decode(encoder_out.data(), num_frames, frame_dim)) {
    return false;
  }

  result->tokens = decoder_->tokens();
  result->timestamps = decoder_->timestamps();
  result->trailing_silence_frames =
      decoder_->num_trailing_blanks() * kSubsamplingFactor;
  return true;
}

std::string TransducerBackend::TextFromTokens(
    const std::vector<int32_t>& tokens, const SymbolTable& symbols) const {
  const int32_t context_size = model_.ContextSize();
  const size_t begin =
      tokens.size() > static_cast<size_t>(context_size) ? context_size : 0;

  std::string text;
  for (size_t i = begin; i < tokens.size(); ++i) {
    text += symbols.GetDisplaySymbol(tokens[i]);
  }
  return text;
}

void TransducerBackend::ResetSegment() {
  if (decoder_ != nullptr) {
    decoder_->ResetSegment();
  }
}

void TransducerBackend::Reset() {
  model_.ResetStates();
  if (decoder_ != nullptr) {
    decoder_->Reset();
  }
}

bool TransducerBackend::SetBlankPenalty(float blank_penalty) {
  if (blank_penalty < 0.0f || decoder_ == nullptr) {
    return false;
  }
  decoder_->SetBlankPenalty(blank_penalty);
  return true;
}

}  // namespace soundproof
