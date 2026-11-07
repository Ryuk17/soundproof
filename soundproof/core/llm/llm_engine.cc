#include "core/llm/llm_engine.h"

#include <cctype>
#include <utility>

#include "utils/logging.h"

namespace soundproof {
namespace {

constexpr const char* kImagePlaceholder = "<img>mnn_image</img>";

// Removes "<think>...</think>" blocks. An unterminated block drops the rest.
std::string StripThinking(const std::string& text) {
  std::string output;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t open = text.find("<think>", pos);
    if (open == std::string::npos) {
      output.append(text, pos, std::string::npos);
      break;
    }
    output.append(text, pos, open - pos);
    const size_t close = text.find("</think>", open + 7);
    if (close == std::string::npos) {
      break;
    }
    pos = close + 8;
  }
  return output;
}

void TrimWhitespace(std::string* text) {
  size_t begin = 0;
  while (begin < text->size() &&
         std::isspace(static_cast<unsigned char>((*text)[begin])) != 0) {
    ++begin;
  }
  size_t end = text->size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>((*text)[end - 1])) != 0) {
    --end;
  }
  text->erase(end);
  text->erase(0, begin);
}

}  // namespace

LlmEngine::LlmEngine() = default;
LlmEngine::~LlmEngine() = default;

LlmStatus LlmEngine::Init(const LlmModelConfig& config) {
  backend_.reset();
  history_.clear();
  last_reply_ = LlmReply();
  system_prompt_.clear();
  ok_ = false;

  backend_ = CreateLlmBackend(config);
  if (backend_ == nullptr) {
    RTC_LOG(LS_ERROR) << "LlmEngine: cannot create backend";
    return LlmStatus::kUnsupportedModel;
  }

  const LlmStatus status = backend_->Init(config);
  if (status != LlmStatus::kOk) {
    RTC_LOG(LS_ERROR) << "LlmEngine: backend init failed";
    backend_.reset();
    return status;
  }

  system_prompt_ = config.system_prompt;
  ok_ = true;
  return LlmStatus::kOk;
}

LlmStatus LlmEngine::Generate(const std::string& text, const LlmImage& image,
                              bool reset_history,
                              const LlmSamplingParams& params,
                              bool strip_thinking) {
  if (!ok_ || backend_ == nullptr) {
    return LlmStatus::kNotInitialized;
  }
  if (text.empty() && image.empty()) {
    RTC_LOG(LS_WARNING) << "LlmEngine: empty text and image";
    return LlmStatus::kInvalidInput;
  }

  if (reset_history) {
    history_.clear();
    backend_->Reset();
  }

  std::vector<LlmMessage> messages;
  if (!system_prompt_.empty()) {
    messages.push_back({"system", system_prompt_});
  }
  messages.insert(messages.end(), history_.begin(), history_.end());

  std::string content = text;
  if (!image.empty()) {
    content = std::string(kImagePlaceholder) + content;
  }
  messages.push_back({"user", content});

  LlmReply reply;
  const LlmStatus status =
      backend_->Generate(messages, image, params, &reply);
  if (status != LlmStatus::kOk) {
    return status;
  }

  std::string output =
      strip_thinking ? StripThinking(reply.text) : reply.text;
  TrimWhitespace(&output);

  last_reply_.text = output;
  last_reply_.prompt_tokens = reply.prompt_tokens;
  last_reply_.output_tokens = reply.output_tokens;

  history_.push_back({"user", text});
  history_.push_back({"assistant", output});
  return LlmStatus::kOk;
}

void LlmEngine::Reset() {
  history_.clear();
  last_reply_ = LlmReply();
  if (backend_ != nullptr) {
    backend_->Reset();
  }
}

}  // namespace soundproof
