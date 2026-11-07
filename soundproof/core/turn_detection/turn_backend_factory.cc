#include "core/turn_detection/turn_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/turn_detection/smart_turn_backend.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

std::unique_ptr<TurnBackend> CreateTurnBackend(
    const TurnModelConfig& config) {
  const std::string type = ToLower(config.model_type);
  if (type.empty() || type == "smart-turn" || type == "smart-turn-v3" ||
      type == "smart-turn-v3.2") {
    return std::make_unique<SmartTurnBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateTurnBackend: unsupported model type '" << type
                    << "'";
  return nullptr;
}

}  // namespace soundproof
