#include "core/speech_recognition/asr_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/speech_recognition/paraformer_backend.h"
#include "core/speech_recognition/transducer_backend.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool IsTransducerType(const std::string& type) {
  return type == "transducer" || type == "zipformer" || type == "zipformer2" ||
         type == "conformer" || type == "lstm" || type == "ebranchformer";
}

}  // namespace

std::unique_ptr<AsrBackend> CreateAsrBackend(const AsrBackendConfig& config) {
  std::string type = ToLower(config.model_type);
  if (type.empty()) {
    // No explicit type: a joiner means transducer, otherwise paraformer.
    type = config.joiner_path.empty() ? "paraformer" : "transducer";
  }

  if (type == "paraformer") {
    return std::make_unique<ParaformerBackend>();
  }
  if (IsTransducerType(type)) {
    return std::make_unique<TransducerBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateAsrBackend: unsupported model type '" << type
                    << "'";
  return nullptr;
}

}  // namespace soundproof
