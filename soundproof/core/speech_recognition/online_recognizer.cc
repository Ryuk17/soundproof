#include "core/speech_recognition/online_recognizer.h"

#include <vector>

#include "core/speech_recognition/online_punctuator.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr float kInt16ToFloat = 1.0f / 32768.0f;
constexpr float kFrameShiftSeconds = 0.01f;

bool HasNonSpace(const std::string& text) {
  for (char c : text) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      return true;
    }
  }
  return false;
}

}  // namespace

OnlineRecognizer::OnlineRecognizer() = default;
OnlineRecognizer::~OnlineRecognizer() = default;

bool OnlineRecognizer::Init(const AsrRecognizerConfig& config) {
  if (config.tokens_path.empty()) {
    RTC_LOG(LS_ERROR) << "OnlineRecognizer: tokens_path is required";
    return false;
  }
  if (!symbols_.LoadFromFile(config.tokens_path)) {
    return false;
  }

  AsrBackendConfig backend_config = config.backend;
  backend_config.unk_id = symbols_.GetId("<unk>");
  backend_ = CreateAsrBackend(backend_config);
  if (backend_ == nullptr || !backend_->Init(backend_config)) {
    RTC_LOG(LS_ERROR) << "OnlineRecognizer: failed to init ASR backend";
    backend_.reset();
    return false;
  }
  if (backend_->FeatureDim() != FeatureExtractor::kFeatureDim) {
    RTC_LOG(LS_ERROR) << "OnlineRecognizer: backend feature dim "
                      << backend_->FeatureDim() << " != extractor dim "
                      << FeatureExtractor::kFeatureDim;
    backend_.reset();
    return false;
  }

  if (config.enable_punctuation) {
    if (config.punctuation_model_path.empty() ||
        config.punctuation_bpe_vocab_path.empty()) {
      RTC_LOG(LS_ERROR) << "OnlineRecognizer: punctuation requires model and "
                           "BPE vocab paths";
      backend_.reset();
      return false;
    }
    OnlinePunctuatorConfig punctuator_config;
    punctuator_config.model_path = config.punctuation_model_path;
    punctuator_config.bpe_vocab_path = config.punctuation_bpe_vocab_path;
    punctuator_config.num_threads = config.punctuation_num_threads;
    punctuator_ = std::make_unique<OnlinePunctuator>();
    if (!punctuator_->Init(punctuator_config)) {
      RTC_LOG(LS_ERROR) << "OnlineRecognizer: failed to init punctuator";
      punctuator_.reset();
      backend_.reset();
      return false;
    }
  }
  punctuate_partial_ = config.punctuate_partial;

  enable_endpoint_ = config.enable_endpoint;
  start_frame_index_ = 0;
  num_processed_frames_ = 0;
  segment_ = 0;
  current_ = AsrResult();
  initialized_ = true;
  return true;
}

bool OnlineRecognizer::Process(const int16_t* audio_in, size_t samples,
                               AsrRecognizerResult* result) {
  if (!initialized_ || backend_ == nullptr || audio_in == nullptr ||
      result == nullptr) {
    return false;
  }
  result->text.clear();
  result->is_final = false;
  result->segment = segment_;

  if (samples > 0) {
    const float scale =
        backend_->NormalizeSamples() ? kInt16ToFloat : 1.0f;
    std::vector<float> waveform(samples);
    for (size_t i = 0; i < samples; ++i) {
      waveform[i] = static_cast<float>(audio_in[i]) * scale;
    }
    features_.AcceptWaveform(waveform.data(), static_cast<int32_t>(samples));
  }

  const int32_t chunk_size = backend_->ChunkSize();
  const int32_t chunk_shift = backend_->ChunkShift();
  if (chunk_size <= 0 || chunk_shift <= 0) {
    return false;
  }

  std::vector<float> chunk;
  while (num_processed_frames_ + chunk_size < features_.NumFramesReady()) {
    if (!features_.GetFrames(start_frame_index_ + num_processed_frames_,
                             chunk_size, &chunk)) {
      return false;
    }
    const int32_t processed_before = num_processed_frames_;
    num_processed_frames_ += chunk_shift;

    if (!backend_->DecodeChunk(chunk.data(), processed_before, &current_)) {
      RTC_LOG(LS_ERROR) << "OnlineRecognizer: backend decode failed";
      return false;
    }

    if (enable_endpoint_ &&
        endpoint_.IsEndpoint(num_processed_frames_,
                             current_.trailing_silence_frames,
                             kFrameShiftSeconds)) {
      const std::string raw_text =
          backend_->TextFromTokens(current_.tokens, symbols_);
      if (HasNonSpace(raw_text)) {
        ++segment_;
      }
      const std::string text =
          punctuator_ ? punctuator_->AddPunctuationWithCase(raw_text)
                      : raw_text;

      backend_->ResetSegment();
      start_frame_index_ += num_processed_frames_;
      num_processed_frames_ = 0;
      current_ = AsrResult();

      result->text = text;
      result->is_final = true;
      result->segment = segment_;
      return true;
    }
  }

  const std::string text =
      backend_->TextFromTokens(current_.tokens, symbols_);
  result->text = (punctuator_ && punctuate_partial_)
                     ? punctuator_->AddPunctuationWithCase(text)
                     : text;
  result->segment = segment_;
  return true;
}

void OnlineRecognizer::Reset() {
  features_.Reset();
  if (backend_ != nullptr) {
    backend_->Reset();
  }
  start_frame_index_ = 0;
  num_processed_frames_ = 0;
  segment_ = 0;
  current_ = AsrResult();
}

bool OnlineRecognizer::SetBlankPenalty(float blank_penalty) {
  return backend_ != nullptr && backend_->SetBlankPenalty(blank_penalty);
}

}  // namespace soundproof
