#include "core/text_to_speech/tts_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/text_to_speech/kokoro_backend.h"
#include "core/text_to_speech/vits_backend.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

std::unique_ptr<TtsBackend> CreateTtsBackend(const TtsModelConfig& config) {
  std::string type = ToLower(config.model_type);
  if (type.empty()) {
    // Kokoro is identified by its style/voices model; everything else is
    // attempted as a VITS-style model.
    type = config.voices_path.empty() ? "vits" : "kokoro";
  }

  if (type == "kokoro") {
    return std::make_unique<KokoroBackend>();
  }
  if (type == "vits" || type == "piper" || type == "coqui") {
    return std::make_unique<VitsBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateTtsBackend: unsupported model type '" << type
                    << "'";
  return nullptr;
}

}  // namespace soundproof
