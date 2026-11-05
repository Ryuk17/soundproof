#include "core/speech_recognition/online_punctuator.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

#include "ssentencepiece/csrc/ssentencepiece.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::vector<std::string> SplitWords(const std::string& text) {
  std::vector<std::string> words;
  std::istringstream iss(text);
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }
  return words;
}

}  // namespace

OnlinePunctuator::OnlinePunctuator() = default;
OnlinePunctuator::~OnlinePunctuator() = default;

bool OnlinePunctuator::Init(const OnlinePunctuatorConfig& config) {
  if (config.model_path.empty() || config.bpe_vocab_path.empty()) {
    RTC_LOG(LS_ERROR)
        << "OnlinePunctuator: model and BPE vocab paths are required";
    return false;
  }

  PunctuationModelConfig model_config;
  model_config.model_path = config.model_path;
  model_config.num_threads = config.num_threads;
  if (!model_.Init(model_config)) {
    return false;
  }

  try {
    bpe_ = std::make_unique<ssentencepiece::Ssentencepiece>(
        config.bpe_vocab_path);
  } catch (...) {
    RTC_LOG(LS_ERROR) << "OnlinePunctuator: failed to load BPE vocab "
                      << config.bpe_vocab_path;
    return false;
  }
  ok_ = true;
  return true;
}

std::string OnlinePunctuator::AddPunctuationWithCase(const std::string& text) {
  if (!ok_ || text.empty()) {
    return text;
  }

  std::vector<int32_t> tokens_list;
  std::vector<int32_t> valids_list;
  std::vector<int32_t> label_lens;
  EncodeSentences(text, &tokens_list, &valids_list, &label_lens);
  if (label_lens.empty()) {
    return text;
  }

  std::vector<int32_t> case_pred;
  std::vector<int32_t> punct_pred;

  for (size_t chunk = 0; chunk < label_lens.size(); ++chunk) {
    const size_t offset =
        chunk * static_cast<size_t>(PunctuationModel::kMaxSeqLen);
    std::vector<int32_t> token_ids(
        tokens_list.begin() + offset,
        tokens_list.begin() + offset + PunctuationModel::kMaxSeqLen);
    std::vector<int32_t> valid_ids(
        valids_list.begin() + offset,
        valids_list.begin() + offset + PunctuationModel::kMaxSeqLen);

    std::vector<float> case_logits;
    std::vector<float> punct_logits;
    int32_t num_cases = 0;
    int32_t num_puncts = 0;
    if (!model_.Forward(token_ids, valid_ids, label_lens[chunk], &case_logits,
                        &num_cases, &punct_logits, &num_puncts)) {
      RTC_LOG(LS_ERROR) << "OnlinePunctuator: model forward failed";
      return text;
    }

    const int32_t case_rows =
        static_cast<int32_t>(case_logits.size()) / num_cases;
    for (int32_t i = 0; i < case_rows; ++i) {
      const float* row = case_logits.data() + i * num_cases;
      case_pred.push_back(static_cast<int32_t>(
          std::max_element(row, row + num_cases) - row));
    }

    const int32_t punct_rows =
        static_cast<int32_t>(punct_logits.size()) / num_puncts;
    for (int32_t i = 0; i < punct_rows; ++i) {
      const float* row = punct_logits.data() + i * num_puncts;
      punct_pred.push_back(static_cast<int32_t>(
          std::max_element(row, row + num_puncts) - row));
    }
  }

  const std::vector<std::string> words = SplitWords(text);
  if (words.size() != case_pred.size() || words.size() != punct_pred.size()) {
    RTC_LOG(LS_WARNING) << "OnlinePunctuator: prediction/word count mismatch ("
                        << words.size() << " vs " << case_pred.size() << ", "
                        << punct_pred.size() << "), returning raw text";
    return text;
  }
  return DecodeSentences(text, case_pred, punct_pred);
}

void OnlinePunctuator::EncodeSentences(const std::string& text,
                                       std::vector<int32_t>* tokens_list,
                                       std::vector<int32_t>* valids_list,
                                       std::vector<int32_t>* label_lens) const {
  tokens_list->clear();
  valids_list->clear();
  label_lens->clear();

  std::vector<int32_t> tokens;
  std::vector<int32_t> valids;
  tokens.push_back(1);  // <s>
  valids.push_back(1);

  auto flush_chunk = [&]() {
    tokens.push_back(2);  // </s>
    valids.push_back(1);
    const int32_t label_len =
        static_cast<int32_t>(std::count(valids.begin(), valids.end(), 1));
    if (tokens.size() < static_cast<size_t>(PunctuationModel::kMaxSeqLen)) {
      tokens.resize(PunctuationModel::kMaxSeqLen, 0);
      valids.resize(PunctuationModel::kMaxSeqLen, 0);
    }
    tokens_list->insert(tokens_list->end(), tokens.begin(), tokens.end());
    valids_list->insert(valids_list->end(), valids.begin(), valids.end());
    label_lens->push_back(label_len);
    tokens.clear();
    valids.clear();
    tokens.push_back(1);
    valids.push_back(1);
  };

  std::istringstream iss(text);
  std::string word;
  while (iss >> word) {
    std::vector<int32_t> word_tokens;
    bpe_->Encode(word, &word_tokens);

    const size_t seq_len = tokens.size() + word_tokens.size();
    if (seq_len > static_cast<size_t>(PunctuationModel::kMaxSeqLen - 1)) {
      flush_chunk();
    }
    tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    valids.push_back(1);  // only the first sub-word is a valid label
    const int32_t remaining =
        static_cast<int32_t>(word_tokens.size()) - 1;
    if (remaining > 0) {
      valids.resize(valids.size() + remaining, 0);
    }
  }

  if (tokens.size() > 1) {
    flush_chunk();
  }
}

std::string OnlinePunctuator::DecodeSentences(
    const std::string& raw_text, const std::vector<int32_t>& case_pred,
    const std::vector<int32_t>& punct_pred) const {
  const std::vector<std::string> words = SplitWords(raw_text);
  const PunctuationMeta& meta = model_.meta();

  std::string result;
  for (size_t i = 0; i < words.size(); ++i) {
    std::string word = words[i];
    if (i != 0) {
      result.push_back(' ');
    }

    const int32_t case_id = case_pred[i];
    if (case_id == meta.upper_id) {
      std::transform(word.begin(), word.end(), word.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::toupper(c));
                     });
    } else if (case_id == meta.cap_id && !word.empty()) {
      word[0] = static_cast<char>(
          std::toupper(static_cast<unsigned char>(word[0])));
    }
    result += word;

    const int32_t punct_id = punct_pred[i];
    if (punct_id == meta.comma_id) {
      result.push_back(',');
    } else if (punct_id == meta.period_id) {
      result.push_back('.');
    } else if (punct_id == meta.question_id) {
      result.push_back('?');
    }
  }
  return result;
}

}  // namespace soundproof
