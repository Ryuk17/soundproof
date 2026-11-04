#include "core/speaker_recognition/speaker_recognizer.h"

#include <utility>

#include "feature/feature_extractor.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr float kInt16ToFloat = 1.0f / 32768.0f;

std::vector<float> ConvertSamples(const int16_t* audio_in, size_t samples,
                                  bool normalize) {
  std::vector<float> waveform(samples);
  const float scale = normalize ? kInt16ToFloat : 1.0f;
  for (size_t i = 0; i < samples; ++i) {
    waveform[i] = static_cast<float>(audio_in[i]) * scale;
  }
  return waveform;
}

}  // namespace

SpeakerRecognizer::SpeakerRecognizer() = default;
SpeakerRecognizer::~SpeakerRecognizer() = default;

bool SpeakerRecognizer::Init(const SrRecognizerConfig& config) {
  if (config.threshold <= 0.0f || config.threshold > 1.0f) {
    RTC_LOG(LS_ERROR) << "SpeakerRecognizer: invalid threshold "
                      << config.threshold;
    return false;
  }

  backend_ = CreateSrBackend(config.backend);
  if (backend_ == nullptr) {
    return false;
  }
  const SrStatus status = backend_->Init(config.backend);
  if (status != SrStatus::kOk) {
    RTC_LOG(LS_ERROR) << "SpeakerRecognizer: backend init failed";
    backend_.reset();
    return false;
  }

  const FeatureExtractorConfig feature_config = backend_->FeatureConfig();
  stream_features_ = std::make_unique<FeatureExtractor>(feature_config);
  enroll_features_ = std::make_unique<FeatureExtractor>(feature_config);
  manager_ = std::make_unique<SpeakerEmbeddingManager>(
      backend_->EmbeddingDim());
  threshold_ = config.threshold;
  initialized_ = true;
  return true;
}

SrStatus SpeakerRecognizer::Process(const int16_t* audio_in, size_t samples,
                                    bool trigger,
                                    SrRecognizerResult* result) {
  if (!initialized_) {
    return SrStatus::kNotInitialized;
  }
  if (result == nullptr || (audio_in == nullptr && samples > 0)) {
    return SrStatus::kFeatureFailed;
  }
  *result = SrRecognizerResult();

  if (samples > 0) {
    const std::vector<float> waveform =
        ConvertSamples(audio_in, samples, backend_->NormalizeSamples());
    stream_features_->AcceptWaveform(waveform.data(),
                                     static_cast<int32_t>(waveform.size()));
  }
  if (!trigger) {
    return SrStatus::kOk;
  }

  const int32_t frames_ready = stream_features_->NumFramesReady();
  if (frames_ready < backend_->MinFrames()) {
    RTC_LOG(LS_WARNING) << "SpeakerRecognizer: audio too short (" << frames_ready
                        << " frames < " << backend_->MinFrames()
                        << "), dropping the cached segment";
    stream_features_->Reset();
    return SrStatus::kAudioTooShort;
  }

  std::vector<float> features;
  if (!stream_features_->GetFrames(0, frames_ready, &features)) {
    stream_features_->Reset();
    return SrStatus::kFeatureFailed;
  }
  stream_features_->Reset();

  std::vector<float> embedding;
  const SrStatus status = backend_->ComputeEmbedding(
      features.data(), frames_ready, backend_->FeatureDim(), &embedding);
  if (status != SrStatus::kOk) {
    return status;
  }
  return IdentifyEmbedding(embedding, result);
}

SrStatus SpeakerRecognizer::IdentifySegment(const int16_t* audio_in,
                                            size_t samples,
                                            SrRecognizerResult* result) {
  if (!initialized_) {
    return SrStatus::kNotInitialized;
  }
  if (result == nullptr) {
    return SrStatus::kFeatureFailed;
  }
  *result = SrRecognizerResult();

  std::vector<float> embedding;
  const SrStatus status =
      ComputeFromPcm(enroll_features_.get(), audio_in, samples, &embedding);
  if (status != SrStatus::kOk) {
    return status;
  }
  return IdentifyEmbedding(embedding, result);
}

SrStatus SpeakerRecognizer::Extract(const int16_t* audio_in, size_t samples,
                                    std::vector<float>* embedding) {
  if (!initialized_) {
    return SrStatus::kNotInitialized;
  }
  return ComputeFromPcm(enroll_features_.get(), audio_in, samples, embedding);
}

SrStatus SpeakerRecognizer::RegisterPcm(const std::string& name,
                                        const int16_t* audio_in,
                                        size_t samples) {
  std::vector<float> embedding;
  const SrStatus status = Extract(audio_in, samples, &embedding);
  if (status != SrStatus::kOk) {
    return status;
  }
  return AddEmbedding(name, embedding);
}

SrStatus SpeakerRecognizer::AddEmbedding(
    const std::string& name, const std::vector<float>& embedding) {
  if (!initialized_ || manager_ == nullptr) {
    return SrStatus::kNotInitialized;
  }
  if (static_cast<int32_t>(embedding.size()) != manager_->Dim()) {
    RTC_LOG(LS_ERROR) << "SpeakerRecognizer: embedding dim "
                      << embedding.size() << " != " << manager_->Dim();
    return SrStatus::kEmbeddingDimMismatch;
  }
  if (manager_->Contains(name)) {
    RTC_LOG(LS_WARNING) << "SpeakerRecognizer: speaker '" << name
                        << "' already exists";
    return SrStatus::kSpeakerExists;
  }
  if (!manager_->Add(name, embedding.data())) {
    RTC_LOG(LS_ERROR) << "SpeakerRecognizer: failed to add embedding for '"
                      << name << "'";
    return SrStatus::kFeatureFailed;
  }
  return SrStatus::kOk;
}

SrStatus SpeakerRecognizer::AddEmbeddings(
    const std::string& name,
    const std::vector<std::vector<float>>& embedding_list) {
  if (!initialized_ || manager_ == nullptr) {
    return SrStatus::kNotInitialized;
  }
  if (manager_->Contains(name)) {
    RTC_LOG(LS_WARNING) << "SpeakerRecognizer: speaker '" << name
                        << "' already exists";
    return SrStatus::kSpeakerExists;
  }
  for (const auto& embedding : embedding_list) {
    if (static_cast<int32_t>(embedding.size()) != manager_->Dim()) {
      RTC_LOG(LS_ERROR) << "SpeakerRecognizer: embedding dim "
                        << embedding.size() << " != " << manager_->Dim();
      return SrStatus::kEmbeddingDimMismatch;
    }
  }
  if (!manager_->Add(name, embedding_list)) {
    RTC_LOG(LS_ERROR) << "SpeakerRecognizer: failed to add embeddings for '"
                      << name << "'";
    return SrStatus::kFeatureFailed;
  }
  return SrStatus::kOk;
}

SrStatus SpeakerRecognizer::Remove(const std::string& name) {
  if (!initialized_ || manager_ == nullptr) {
    return SrStatus::kNotInitialized;
  }
  if (!manager_->Remove(name)) {
    RTC_LOG(LS_WARNING) << "SpeakerRecognizer: speaker '" << name
                        << "' not found";
    return SrStatus::kSpeakerNotFound;
  }
  return SrStatus::kOk;
}

void SpeakerRecognizer::Clear() {
  if (manager_ != nullptr) {
    manager_->Clear();
  }
}

int32_t SpeakerRecognizer::NumSpeakers() const {
  return manager_ == nullptr ? 0 : manager_->NumSpeakers();
}

std::vector<std::string> SpeakerRecognizer::GetAllSpeakers() const {
  return manager_ == nullptr ? std::vector<std::string>()
                             : manager_->GetAllSpeakers();
}

bool SpeakerRecognizer::SetThreshold(float threshold) {
  if (threshold <= 0.0f || threshold > 1.0f) {
    return false;
  }
  threshold_ = threshold;
  return true;
}

int32_t SpeakerRecognizer::EmbeddingDim() const {
  return manager_ == nullptr ? 0 : manager_->Dim();
}

int32_t SpeakerRecognizer::SampleRate() const {
  return backend_ == nullptr ? 0 : backend_->SampleRate();
}

void SpeakerRecognizer::ResetStream() {
  if (stream_features_ != nullptr) {
    stream_features_->Reset();
  }
}

SrStatus SpeakerRecognizer::ComputeFromPcm(FeatureExtractor* extractor,
                                           const int16_t* audio_in,
                                           size_t samples,
                                           std::vector<float>* embedding) {
  if (!initialized_ || extractor == nullptr || embedding == nullptr ||
      audio_in == nullptr || samples == 0) {
    return SrStatus::kFeatureFailed;
  }
  const std::vector<float> waveform =
      ConvertSamples(audio_in, samples, backend_->NormalizeSamples());

  extractor->Reset();
  extractor->AcceptWaveform(waveform.data(),
                            static_cast<int32_t>(waveform.size()));
  extractor->InputFinished();

  const int32_t frames = extractor->NumFramesReady();
  if (frames < backend_->MinFrames()) {
    RTC_LOG(LS_WARNING) << "SpeakerRecognizer: audio too short (" << frames
                        << " frames < " << backend_->MinFrames() << ")";
    return SrStatus::kAudioTooShort;
  }

  std::vector<float> features;
  if (!extractor->GetFrames(0, frames, &features)) {
    return SrStatus::kFeatureFailed;
  }
  return backend_->ComputeEmbedding(features.data(), frames,
                                    backend_->FeatureDim(), embedding);
}

SrStatus SpeakerRecognizer::IdentifyEmbedding(
    const std::vector<float>& embedding, SrRecognizerResult* result) const {
  if (manager_ == nullptr || result == nullptr) {
    return SrStatus::kNotInitialized;
  }
  float score = 0.0f;
  result->speaker_name =
      manager_->Search(embedding.data(), threshold_, &score);
  result->score = score;
  result->has_result = true;
  return SrStatus::kOk;
}

}  // namespace soundproof
