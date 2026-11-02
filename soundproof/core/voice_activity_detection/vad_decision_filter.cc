#include "core/voice_activity_detection/vad_decision_filter.h"

#include <cmath>

namespace soundproof {

VadDecisionFilter::VadDecisionFilter() {
  SetConfig(Config{});
}

VadDecisionFilter::VadDecisionFilter(const Config& config) {
  SetConfig(config);
}

void VadDecisionFilter::SetConfig(const Config& config) {
  config_ = config;
  if (config_.sample_rate <= 0) {
    config_.sample_rate = 16000;
  }
  min_speech_samples_ = static_cast<int64_t>(
      std::llround(static_cast<double>(config_.min_speech_duration) *
                   config_.sample_rate));
  min_silence_samples_ = static_cast<int64_t>(
      std::llround(static_cast<double>(config_.min_silence_duration) *
                   config_.sample_rate));
}

bool VadDecisionFilter::Update(float probability, int64_t elapsed_samples) {
  if (elapsed_samples <= 0) {
    return is_speech_;
  }

  // Port of sherpa-mnn SileroVadModel::IsSpeech.
  const float threshold = config_.threshold;
  current_sample_ += elapsed_samples;

  if (probability > threshold && temp_end_ != 0) {
    temp_end_ = 0;
  }

  if (probability > threshold && temp_start_ == 0) {
    // Speech onset: wait until it lasts |min_speech_duration|.
    temp_start_ = current_sample_;
    is_speech_ = false;
    return false;
  }

  if (probability > threshold && temp_start_ != 0 && !triggered_) {
    if (current_sample_ - temp_start_ < min_speech_samples_) {
      is_speech_ = false;
      return false;
    }
    triggered_ = true;
    is_speech_ = true;
    return true;
  }

  if (probability < threshold && !triggered_) {
    temp_start_ = 0;
    temp_end_ = 0;
    is_speech_ = false;
    return false;
  }

  if (probability > threshold - config_.hysteresis && triggered_) {
    // Above the release threshold: keep speaking (hysteresis).
    is_speech_ = true;
    return true;
  }

  if (probability > threshold && !triggered_) {
    triggered_ = true;
    is_speech_ = true;
    return true;
  }

  if (probability < threshold && triggered_) {
    if (temp_end_ == 0) {
      temp_end_ = current_sample_;
    }
    if (current_sample_ - temp_end_ < min_silence_samples_) {
      // Hangover: keep reporting speech for a while.
      is_speech_ = true;
      return true;
    }
    temp_start_ = 0;
    temp_end_ = 0;
    triggered_ = false;
    is_speech_ = false;
    return false;
  }

  is_speech_ = false;
  return false;
}

void VadDecisionFilter::Reset() {
  current_sample_ = 0;
  temp_start_ = 0;
  temp_end_ = 0;
  triggered_ = false;
  is_speech_ = false;
}

}  // namespace soundproof
