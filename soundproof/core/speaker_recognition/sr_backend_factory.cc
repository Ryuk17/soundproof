#include "core/speaker_recognition/sr_backend.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "core/speaker_recognition/general_sr_backend.h"
#include "core/speaker_recognition/nemo_sr_backend.h"
#include "mnn_engine/mnn_engine.h"
#include "utils/logging.h"

namespace soundproof {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Reads the "framework" metadata. This loads the model once; it happens only
// when the caller does not specify model_type explicitly.
std::string ProbeFramework(const std::string& model_path) {
  mnn_engine::Engine engine;
  mnn_engine::EngineConfig config;
  config.num_threads = 1;
  if (!engine.Init(model_path, config)) {
    return {};
  }
  const auto metadata = engine.metadata();
  const auto it = metadata.find("framework");
  return it == metadata.end() ? std::string() : it->second;
}

}  // namespace

std::unique_ptr<SrBackend> CreateSrBackend(const SrModelConfig& config) {
  std::string type = ToLower(config.model_type);
  if (type.empty()) {
    type = ToLower(ProbeFramework(config.model_path));
    if (type.empty()) {
      RTC_LOG(LS_ERROR) << "CreateSrBackend: cannot detect model framework "
                           "from '"
                        << config.model_path << "'";
      return nullptr;
    }
  }

  if (type == "wespeaker" || type == "3d-speaker") {
    return std::make_unique<GeneralSrBackend>();
  }
  if (type == "nemo") {
    return std::make_unique<NemoSrBackend>();
  }

  RTC_LOG(LS_ERROR) << "CreateSrBackend: unsupported model type '" << type
                    << "'";
  return nullptr;
}

}  // namespace soundproof
