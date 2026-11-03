#ifndef SOUNDPROOF_CORE_KEYWORDS_SPOTTING_SYMBOL_TABLE_H_
#define SOUNDPROOF_CORE_KEYWORDS_SPOTTING_SYMBOL_TABLE_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soundproof {

// Maps between tokens and their integer ids. Each line of tokens.txt has the
// form "<symbol> <id>".
class SymbolTable {
 public:
  bool LoadFromFile(const std::string& filename);
  bool LoadFromString(const std::string& content);

  bool Contains(const std::string& symbol) const;
  bool Contains(int32_t id) const;

  // Returns -1 if the symbol is unknown.
  int32_t GetId(const std::string& symbol) const;
  // Returns an empty string if the id is unknown.
  const std::string& GetSymbol(int32_t id) const;

  size_t Size() const { return id_to_symbol_.size(); }

 private:
  std::unordered_map<std::string, int32_t> symbol_to_id_;
  std::vector<std::string> id_to_symbol_;
  static const std::string kEmptySymbol;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_KEYWORDS_SPOTTING_SYMBOL_TABLE_H_
