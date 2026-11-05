#include "core/text_to_speech/tts_engine.h"

#include <algorithm>
#include <utility>

#include "utils/logging.h"

namespace soundproof {
namespace {

int16_t FloatToInt16(float v) {
  const float scaled = v * 32768.0f;
  if (scaled >= 32767.0f) {
    return 32767;
  }
  if (scaled <= -32768.0f) {
    return -32768;
  }
  return static_cast<int16_t>(scaled);
}

}  // namespace

TtsEngine::TtsEngine() = default;
TtsEngine::~TtsEngine() = default;

bool TtsEngine::Init(const TtsModelConfig& config) {
  backend_ = CreateTtsBackend(config);
  if (backend_ == nullptr) {
    return false;
  }
  const TtsStatus status = backend_->Init(config);
  if (status != TtsStatus::kOk) {
    RTC_LOG(LS_ERROR) << "TtsEngine: backend init failed";
    backend_.reset();
    return false;
  }
  last_audio_ = TtsAudio();
  ok_ = true;
  return true;
}

TtsStatus TtsEngine::Synthesize(const TtsRequest& request, TtsAudio* audio) {
  if (!ok_ || backend_ == nullptr) {
    return TtsStatus::kNotInitialized;
  }
  if (audio == nullptr || request.speed <= 0.0f) {
    return TtsStatus::kInvalidInput;
  }

  std::vector<int32_t> token_ids;
  if (request.input_type == TtsInputType::kTokenIds) {
    token_ids = request.token_ids;
  } else if (request.input_type == TtsInputType::kPhoneme) {
    const TtsStatus status = backend_->EncodePhonemes(request.text, &token_ids);
    if (status != TtsStatus::kOk) {
      return status;
    }
  } else {
    if (!backend_->SupportsText()) {
      RTC_LOG(LS_WARNING) << "TtsEngine: text input needs a G2P frontend; "
                             "use phoneme or token ids for this model";
      return TtsStatus::kUnsupportedInput;
    }
    const TtsStatus status = backend_->EncodeText(request.text, &token_ids);
    if (status != TtsStatus::kOk) {
      return status;
    }
  }
  if (token_ids.empty()) {
    RTC_LOG(LS_ERROR) << "TtsEngine: empty token sequence";
    return TtsStatus::kInvalidInput;
  }

  std::vector<float> float_samples;
  const TtsStatus status = backend_->Synthesize(
      token_ids, request.speaker_id, request.speed, &float_samples);
  if (status != TtsStatus::kOk || float_samples.empty()) {
    return status == TtsStatus::kOk ? TtsStatus::kInferenceFailed : status;
  }

  TtsAudio generated;
  generated.sample_rate = backend_->SampleRate();
  generated.samples.resize(float_samples.size());
  for (size_t i = 0; i < float_samples.size(); ++i) {
    generated.samples[i] = FloatToInt16(float_samples[i]);
  }
  last_audio_ = std::move(generated);
  *audio = last_audio_;
  return TtsStatus::kOk;
}

int32_t TtsEngine::SampleRate() const {
  return backend_ == nullptr ? 0 : backend_->SampleRate();
}

int32_t TtsEngine::NumSpeakers() const {
  return backend_ == nullptr ? 0 : backend_->NumSpeakers();
}

void TtsEngine::Reset() {
  last_audio_ = TtsAudio();
}

}  // namespace soundproof
