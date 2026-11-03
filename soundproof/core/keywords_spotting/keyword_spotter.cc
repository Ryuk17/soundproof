#include "core/keywords_spotting/keyword_spotter.h"

#include <utility>
#include <vector>

#include "core/keywords_spotting/keyword_decoder.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr float kInt16ToFloat = 1.0f / 32768.0f;

ContextGraph BuildGraph(const KeywordList& keywords, float score,
                        float threshold) {
  return ContextGraph(keywords.token_ids, score, threshold, keywords.scores,
                      keywords.phrases, keywords.thresholds);
}

}  // namespace

KeywordSpotter::KeywordSpotter() = default;
KeywordSpotter::~KeywordSpotter() = default;

bool KeywordSpotter::Init(const KeywordSpotterConfig& config) {
  if (config.encoder_path.empty() || config.decoder_path.empty() ||
      config.joiner_path.empty() || config.tokens_path.empty() ||
      config.keywords_file_path.empty()) {
    RTC_LOG(LS_ERROR) << "KeywordSpotter: missing model/keyword paths";
    return false;
  }
  if (config.keywords_threshold <= 0.0f || config.keywords_threshold >= 1.0f) {
    RTC_LOG(LS_ERROR) << "KeywordSpotter: invalid threshold "
                      << config.keywords_threshold;
    return false;
  }

  config_ = config;

  if (!symbols_.LoadFromFile(config.tokens_path)) {
    return false;
  }
  if (!LoadKeywordsFromFile(config.keywords_file_path, symbols_, &keywords_)) {
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
  if (model_.FeatureDim() != FeatureExtractor::kFeatureDim) {
    RTC_LOG(LS_ERROR) << "KeywordSpotter: model expects feature dim "
                      << model_.FeatureDim() << ", extractor produces "
                      << FeatureExtractor::kFeatureDim;
    return false;
  }

  graph_ =
      BuildGraph(keywords_, config_.keywords_score, config_.keywords_threshold);

  const int32_t unk_id = symbols_.GetId("<unk>");
  decoder_ = std::make_unique<KeywordDecoder>(&model_, config_.max_active_paths,
                                              config_.num_trailing_blanks,
                                              unk_id);
  decoder_->Reset(graph_);
  processed_frames_ = 0;
  initialized_ = true;
  return true;
}

bool KeywordSpotter::Process(const int16_t* audio_in, size_t samples,
                             int* keyword_index, float* keyword_score) {
  if (!initialized_ || audio_in == nullptr || keyword_index == nullptr ||
      keyword_score == nullptr) {
    return false;
  }
  *keyword_index = -1;
  *keyword_score = 0.0f;

  if (samples > 0) {
    std::vector<float> waveform(samples);
    for (size_t i = 0; i < samples; ++i) {
      waveform[i] = static_cast<float>(audio_in[i]) * kInt16ToFloat;
    }
    features_.AcceptWaveform(waveform.data(), static_cast<int32_t>(samples));
  }

  const int32_t chunk_size = model_.ChunkSize();
  const int32_t chunk_shift = model_.ChunkShift();
  const int32_t feature_dim = model_.FeatureDim();
  if (chunk_size <= 0 || chunk_shift <= 0 || feature_dim <= 0) {
    return false;
  }

  while (processed_frames_ + chunk_size < features_.NumFramesReady()) {
    std::vector<float> frames;
    if (!features_.GetFrames(processed_frames_, chunk_size, &frames)) {
      return false;
    }
    processed_frames_ += chunk_shift;

    std::vector<float> encoder_out;
    int32_t num_frames = 0;
    int32_t out_dim = 0;
    if (!model_.RunEncoder(frames, &encoder_out, &num_frames, &out_dim)) {
      return false;
    }

    KeywordHit hit;
    if (decoder_->Decode(graph_, encoder_out.data(), num_frames, out_dim,
                         &hit)) {
      const int index = FindKeywordIndex(hit);
      if (index >= 0) {
        *keyword_index = index;
        *keyword_score = hit.score;
        return true;
      }
      RTC_LOG(LS_WARNING) << "KeywordSpotter: triggered unknown keyword '"
                          << hit.phrase << "'";
    }
  }
  return true;
}

bool KeywordSpotter::SetThreshold(float threshold) {
  if (!initialized_ || threshold <= 0.0f || threshold >= 1.0f) {
    return false;
  }
  config_.keywords_threshold = threshold;
  graph_ =
      BuildGraph(keywords_, config_.keywords_score, config_.keywords_threshold);
  decoder_->Reset(graph_);
  return true;
}

void KeywordSpotter::Reset() {
  features_.Reset();
  model_.ResetStates();
  if (decoder_) {
    decoder_->Reset(graph_);
  }
  processed_frames_ = 0;
}

const std::string& KeywordSpotter::keyword(size_t index) const {
  static const std::string kEmpty;
  if (index >= keywords_.phrases.size()) {
    return kEmpty;
  }
  return keywords_.phrases[index];
}

int KeywordSpotter::FindKeywordIndex(const KeywordHit& hit) const {
  if (!hit.phrase.empty()) {
    for (size_t i = 0; i < keywords_.phrases.size(); ++i) {
      if (keywords_.phrases[i] == hit.phrase) {
        return static_cast<int>(i);
      }
    }
  }
  for (size_t i = 0; i < keywords_.token_ids.size(); ++i) {
    if (keywords_.token_ids[i] == hit.tokens) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace soundproof
