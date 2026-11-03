#include "core/keywords_spotting/keyword_encoder.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "utils/logging.h"

namespace soundproof {

bool LoadKeywordsFromFile(const std::string& filename,
                          const SymbolTable& table, KeywordList* keywords) {
  std::ifstream is(filename);
  if (!is) {
    RTC_LOG(LS_ERROR) << "LoadKeywords: cannot open " << filename;
    return false;
  }
  std::stringstream buffer;
  buffer << is.rdbuf();
  return LoadKeywordsFromString(buffer.str(), table, keywords);
}

bool LoadKeywordsFromString(const std::string& content,
                            const SymbolTable& table, KeywordList* keywords) {
  if (keywords == nullptr) {
    return false;
  }
  keywords->token_ids.clear();
  keywords->phrases.clear();
  keywords->scores.clear();
  keywords->thresholds.clear();

  std::vector<std::vector<int32_t>> token_ids;
  std::vector<std::string> phrases;
  std::vector<float> scores;
  std::vector<float> thresholds;
  bool has_scores = false;
  bool has_thresholds = false;

  std::istringstream is(content);
  std::string line;
  while (std::getline(is, line)) {
    std::vector<int32_t> ids;
    std::string phrase;
    float score = 0.0f;
    float threshold = 0.0f;

    std::istringstream line_stream(line);
    std::string word;
    while (line_stream >> word) {
      if (table.Contains(word)) {
        ids.push_back(table.GetId(word));
        continue;
      }
      switch (word[0]) {
        case ':':
          score = std::strtof(word.c_str() + 1, nullptr);
          has_scores = true;
          break;
        case '#':
          threshold = std::strtof(word.c_str() + 1, nullptr);
          has_thresholds = true;
          break;
        case '@':
          phrase = word.substr(1);
          break;
        default:
          RTC_LOG(LS_ERROR) << "LoadKeywords: token '" << word
                            << "' (line: " << line << ") is not in tokens.txt";
          return false;
      }
    }

    if (ids.empty()) {
      continue;  // skip blank/comment lines
    }
    token_ids.push_back(std::move(ids));
    phrases.push_back(std::move(phrase));
    scores.push_back(score);
    thresholds.push_back(threshold);
  }

  if (token_ids.empty()) {
    RTC_LOG(LS_ERROR) << "LoadKeywords: no keywords found";
    return false;
  }

  keywords->token_ids = std::move(token_ids);
  keywords->phrases = std::move(phrases);
  if (has_scores) {
    keywords->scores = std::move(scores);
  }
  if (has_thresholds) {
    keywords->thresholds = std::move(thresholds);
  }
  return true;
}

}  // namespace soundproof
