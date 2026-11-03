#ifndef SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_ENCODER_H_
#define SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_ENCODER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/keywords_spotting/symbol_table.h"

namespace soundproof {

// One keyword line from keywords.txt:
//   n ǐ h ǎo @你好
// Optional ":score" and "#threshold" fields may be appended (sherpa format).
struct KeywordList {
  std::vector<std::vector<int32_t>> token_ids;
  std::vector<std::string> phrases;
  std::vector<float> scores;      // empty if no line specifies ":score"
  std::vector<float> thresholds;  // empty if no line specifies "#threshold"

  size_t size() const { return token_ids.size(); }
};

// Returns false if the file cannot be opened or a token is not in |table|.
bool LoadKeywordsFromFile(const std::string& filename,
                          const SymbolTable& table, KeywordList* keywords);

bool LoadKeywordsFromString(const std::string& content,
                            const SymbolTable& table, KeywordList* keywords);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_KEYWORDS_SPOTTING_KEYWORD_ENCODER_H_
