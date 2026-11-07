#include "core/llm/llm_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/llm/mnn_llm_backend.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

std::unique_ptr<LlmBackend> CreateLlmBackend(const LlmModelConfig& config) {
  const std::string type = ToLower(config.model_type);
  if (type.empty() || type == "qwen3.5" || type == "qwen3.5-vl" ||
      type == "qwen3_5" || type == "qwen3.5-0.8b") {
    return std::make_unique<MnnLlmBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateLlmBackend: unsupported model type '" << type
                    << "'";
  return nullptr;
}

}  // namespace soundproof
