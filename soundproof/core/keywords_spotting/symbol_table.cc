#include "core/keywords_spotting/symbol_table.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "utils/logging.h"

namespace soundproof {

const std::string SymbolTable::kEmptySymbol;

bool SymbolTable::LoadFromFile(const std::string& filename) {
  std::ifstream is(filename);
  if (!is) {
    RTC_LOG(LS_ERROR) << "SymbolTable: cannot open " << filename;
    return false;
  }
  std::stringstream buffer;
  buffer << is.rdbuf();
  return LoadFromString(buffer.str());
}

bool SymbolTable::LoadFromString(const std::string& content) {
  symbol_to_id_.clear();
  id_to_symbol_.clear();

  std::istringstream is(content);
  std::string line;
  int32_t max_id = -1;
  std::vector<std::pair<int32_t, std::string>> entries;

  while (std::getline(is, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t pos = line.rfind(' ');
    if (pos == std::string::npos) {
      RTC_LOG(LS_ERROR) << "SymbolTable: malformed line: " << line;
      return false;
    }
    const std::string symbol = line.substr(0, pos);
    const int32_t id = std::atoi(line.substr(pos + 1).c_str());
    if (id < 0) {
      RTC_LOG(LS_ERROR) << "SymbolTable: invalid id in line: " << line;
      return false;
    }
    entries.emplace_back(id, symbol);
    max_id = std::max(max_id, id);
  }

  id_to_symbol_.resize(static_cast<size_t>(max_id + 1));
  for (auto& entry : entries) {
    const int32_t id = entry.first;
    if (!id_to_symbol_[static_cast<size_t>(id)].empty()) {
      RTC_LOG(LS_WARNING) << "SymbolTable: duplicate id " << id;
    }
    id_to_symbol_[static_cast<size_t>(id)] = entry.second;
    symbol_to_id_[entry.second] = id;
  }
  return true;
}

bool SymbolTable::Contains(const std::string& symbol) const {
  return symbol_to_id_.find(symbol) != symbol_to_id_.end();
}

bool SymbolTable::Contains(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < id_to_symbol_.size() &&
         !id_to_symbol_[static_cast<size_t>(id)].empty();
}

int32_t SymbolTable::GetId(const std::string& symbol) const {
  const auto it = symbol_to_id_.find(symbol);
  return it == symbol_to_id_.end() ? -1 : it->second;
}

const std::string& SymbolTable::GetSymbol(int32_t id) const {
  if (!Contains(id)) {
    return kEmptySymbol;
  }
  return id_to_symbol_[static_cast<size_t>(id)];
}

}  // namespace soundproof
