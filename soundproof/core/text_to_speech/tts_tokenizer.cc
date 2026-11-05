#include "core/text_to_speech/tts_tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr int kMnnTokenizerMagic = 430;
constexpr int kTiktokenType = 1;

std::string Base64Decode(const std::string& input) {
  static const char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int decode[256];
  for (int& v : decode) {
    v = -1;
  }
  for (int i = 0; i < 64; ++i) {
    decode[static_cast<uint8_t>(kTable[i])] = i;
  }

  std::string output;
  output.reserve(input.size() * 3 / 4 + 2);
  int value = 0;
  int bits = -8;
  for (unsigned char c : input) {
    const int d = decode[c];
    if (d < 0) {
      continue;
    }
    value = (value << 6) + d;
    bits += 6;
    if (bits >= 0) {
      output.push_back(static_cast<char>((value >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return output;
}

}  // namespace

bool TtsTokenizer::Load(const std::string& path) {
  token_to_id_.clear();
  max_token_bytes_ = 0;
  ok_ = false;

  std::ifstream file(path);
  if (!file) {
    RTC_LOG(LS_ERROR) << "TtsTokenizer: cannot open " << path;
    return false;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(line);
  }
  if (lines.empty()) {
    return false;
  }

  int magic = 0;
  int type = 0;
  {
    std::istringstream iss(lines[0]);
    iss >> magic >> type;
  }
  if (magic == kMnnTokenizerMagic && type == kTiktokenType) {
    ok_ = LoadMnnTokenizer(lines);
  } else {
    ok_ = LoadSymbolTable(lines);
  }
  return ok_;
}

bool TtsTokenizer::LoadMnnTokenizer(
    const std::vector<std::string>& lines) {
  size_t index = 1;
  int special_num = 0;
  int stop_num = 0;
  int prefix_num = 0;
  if (index < lines.size()) {
    std::istringstream iss(lines[index]);
    iss >> special_num >> stop_num >> prefix_num;
    ++index;
  }
  // The token list of special tokens occupies the next line, even when the
  // counts are zero.
  if (index < lines.size()) {
    ++index;
  }
  if (index >= lines.size()) {
    RTC_LOG(LS_ERROR) << "TtsTokenizer: truncated MNN tokenizer file";
    return false;
  }
  int vocab_size = std::atoi(lines[index].c_str());
  ++index;
  if (vocab_size <= 0) {
    RTC_LOG(LS_ERROR) << "TtsTokenizer: invalid vocab size " << vocab_size;
    return false;
  }

  for (int i = 0; i < vocab_size; ++i, ++index) {
    if (index >= lines.size()) {
      RTC_LOG(LS_ERROR) << "TtsTokenizer: expected " << vocab_size
                        << " tokens, file ended early";
      return false;
    }
    const std::string token = Base64Decode(lines[index]);
    token_to_id_[token] = i;
    max_token_bytes_ = std::max(max_token_bytes_, token.size());
  }
  return !token_to_id_.empty();
}

bool TtsTokenizer::LoadSymbolTable(const std::vector<std::string>& lines) {
  for (const std::string& line : lines) {
    if (line.empty()) {
      continue;
    }
    const size_t pos = line.rfind(' ');
    if (pos == std::string::npos) {
      continue;
    }
    const std::string symbol = line.substr(0, pos);
    const int id = std::atoi(line.substr(pos + 1).c_str());
    if (symbol.empty() || id < 0) {
      continue;
    }
    token_to_id_[symbol] = id;
    max_token_bytes_ = std::max(max_token_bytes_, symbol.size());
  }
  if (token_to_id_.empty()) {
    RTC_LOG(LS_ERROR) << "TtsTokenizer: no valid symbols found";
    return false;
  }
  return true;
}

bool TtsTokenizer::Encode(const std::string& text,
                          std::vector<int32_t>* token_ids) const {
  if (!ok_ || token_ids == nullptr) {
    return false;
  }
  token_ids->clear();
  if (text.empty()) {
    return true;
  }

  size_t pos = 0;
  while (pos < text.size()) {
    const size_t max_len =
        std::min(max_token_bytes_, text.size() - pos);
    bool matched = false;
    for (size_t len = max_len; len > 0; --len) {
      const std::string candidate = text.substr(pos, len);
      const auto it = token_to_id_.find(candidate);
      if (it != token_to_id_.end()) {
        token_ids->push_back(it->second);
        pos += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      RTC_LOG(LS_ERROR) << "TtsTokenizer: no token for byte at offset " << pos
                        << " (0x" << std::hex
                        << static_cast<int>(static_cast<uint8_t>(text[pos]))
                        << std::dec << ")";
      return false;
    }
  }
  return true;
}

}  // namespace soundproof
