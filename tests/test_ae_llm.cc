// Tests for libAE_LLM — Qwen3.5 VLM chat via MNN-LLM.
//
// Uses models/llm/Qwen3.5-0.8B-MNN and assets/test_image_red.png. The model
// (~540 MB) is not tracked, so the test prints SKIP and succeeds when the
// assets are missing.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "interface/audio_engine_llm.h"

namespace {

const char* kConfigPath = "models/llm/Qwen3.5-0.8B-MNN/config.json";
const char* kImagePath = "assets/test_image_red.png";

int g_failures = 0;

void Check(bool condition, const std::string& what)
{
    std::cout << "  [" << (condition ? " OK " : "FAIL") << "] " << what
              << std::endl;
    if (!condition) {
        ++g_failures;
    }
}

bool FileExists(const char* path)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        return false;
    }
    std::fclose(fp);
    return true;
}

bool AssetsAvailable()
{
    const char* files[] = {
        "models/llm/Qwen3.5-0.8B-MNN/config.json",
        "models/llm/Qwen3.5-0.8B-MNN/llm_config.json",
        "models/llm/Qwen3.5-0.8B-MNN/llm.mnn",
        "models/llm/Qwen3.5-0.8B-MNN/llm.mnn.weight",
        "models/llm/Qwen3.5-0.8B-MNN/visual.mnn",
        "models/llm/Qwen3.5-0.8B-MNN/visual.mnn.weight",
        "models/llm/Qwen3.5-0.8B-MNN/tokenizer.txt",
        "assets/test_image_red.png",
    };
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

void FillInitConfig(LlmInitConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->num_threads = 2;
    std::snprintf(config->model_type, sizeof(config->model_type), "%s",
                  "qwen3.5");
    std::snprintf(config->config_path, sizeof(config->config_path), "%s",
                  kConfigPath);
    config->system_prompt = nullptr;
}

void FillRtConfig(LlmRtConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->max_new_tokens = 24;
    config->temperature = 0.0f;  // greedy
    config->top_p = -1.0f;
    config->top_k = -1;
    config->enable_thinking = 0;
    config->strip_thinking = 1;
}

bool InitHandle(LlmHandle handle)
{
    LlmInitConfig config;
    FillInitConfig(&config);
    if (AudioEngine_Llm_Init(handle, &config) != AUDIO_ENGINE_SUCCESS) {
        return false;
    }
    LlmRtConfig rt_config;
    FillRtConfig(&rt_config);
    return AudioEngine_Llm_SetParam(handle, &rt_config) == AUDIO_ENGINE_SUCCESS;
}

std::string ToLower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return text;
}

int Generate(LlmHandle handle, const char* text, const char* image_path,
             const unsigned char* image_data, int width, int height,
             int reset_history, LlmResult_t* result)
{
    LlmInput_t input;
    std::memset(&input, 0, sizeof(input));
    input.text = text;
    input.image_path = image_path;
    input.image_data = image_data;
    input.image_width = width;
    input.image_height = height;
    input.reset_history = reset_history;
    return AudioEngine_Llm_Process(handle, &input, result);
}

std::vector<unsigned char> MakeRedSquare(int width, int height, int margin)
{
    std::vector<unsigned char> rgb(
        static_cast<size_t>(width) * height * 3, 255);
    for (int y = margin; y < height - margin; ++y) {
        for (int x = margin; x < width - margin; ++x) {
            unsigned char* pixel =
                rgb.data() + (static_cast<size_t>(y) * width + x) * 3;
            pixel[0] = 220;
            pixel[1] = 30;
            pixel[2] = 30;
        }
    }
    return rgb;
}

void TestUninitializedErrors()
{
    std::cout << "=== test_uninitialized_errors ===" << std::endl;

    LlmHandle handle = AudioEngine_Llm_Create();
    Check(handle != NULL, "Create");

    LlmInput_t input;
    std::memset(&input, 0, sizeof(input));
    input.text = "hello";
    LlmResult_t result;

    Check(AudioEngine_Llm_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init -> NOT_INITIALIZED");
    Check(AudioEngine_Llm_Reset(handle) == AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Reset before Init -> NOT_INITIALIZED");

    LlmRtConfig rt_config;
    FillRtConfig(&rt_config);
    Check(AudioEngine_Llm_SetParam(handle, &rt_config) == AUDIO_ENGINE_SUCCESS,
          "SetParam before Init is allowed");
    rt_config.top_p = 1.5f;
    Check(AudioEngine_Llm_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "SetParam invalid top_p -> INVALID_PARAM");
    rt_config.top_p = 0.9f;
    rt_config.max_new_tokens = -1;
    Check(AudioEngine_Llm_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "SetParam negative max_new_tokens -> INVALID_PARAM");
    Check(AudioEngine_Llm_SetParam(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "SetParam NULL -> NULL_POINTER");

    Check(AudioEngine_Llm_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit without Init");
    Check(AudioEngine_Llm_Destroy(handle) == AUDIO_ENGINE_SUCCESS,
          "Destroy");
    Check(AudioEngine_Llm_Process(NULL, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Process NULL handle -> INVALID_HANDLE");
}

void TestInvalidInit()
{
    std::cout << "=== test_invalid_init ===" << std::endl;

    LlmHandle handle = AudioEngine_Llm_Create();
    LlmInitConfig config;
    FillInitConfig(&config);

    config.config_path[0] = '\0';
    Check(AudioEngine_Llm_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty config_path -> INVALID_PARAM");

    FillInitConfig(&config);
    std::snprintf(config.config_path, sizeof(config.config_path), "%s",
                  "models/llm/does-not-exist/config.json");
    Check(AudioEngine_Llm_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED,
          "missing config -> MODEL_LOAD_FAILED");

    FillInitConfig(&config);
    std::snprintf(config.model_type, sizeof(config.model_type), "%s",
                  "unknown-llm");
    Check(AudioEngine_Llm_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL,
          "unknown model_type -> UNSUPPORTED_MODEL");

    FillInitConfig(&config);
    config.num_threads = -1;
    Check(AudioEngine_Llm_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "negative num_threads -> INVALID_PARAM");

    Check(AudioEngine_Llm_Init(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL init_config -> NULL_POINTER");

    AudioEngine_Llm_Destroy(handle);
}

void TestModel()
{
    std::cout << "=== test_model ===" << std::endl;

    LlmHandle handle = AudioEngine_Llm_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle), "Init Qwen3.5-0.8B");

    LlmResult_t result;
    Check(Generate(handle, "", nullptr, nullptr, 0, 0, 1, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty text and image -> INVALID_PARAM");
    Check(Generate(handle, "hello", "models/llm/no-such.png", nullptr, 0, 0, 1,
                   &result) == AUDIO_ENGINE_ERR_INVALID_PARAM,
          "missing image -> INVALID_PARAM");
    Check(Generate(handle, "hello", nullptr, nullptr, 0, 0, 2, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "invalid reset_history -> INVALID_PARAM");

    // Text-only.
    Check(Generate(handle, "What is 2+2? Answer with a single digit.", nullptr,
                   nullptr, 0, 0, 1, &result) == AUDIO_ENGINE_SUCCESS,
          "text prompt succeeds");
    const std::string text_answer =
        result.text ? std::string(result.text, result.text_len) : "";
    std::cout << "  text answer: " << text_answer << std::endl;
    const std::string lower_text = ToLower(text_answer);
    Check(lower_text.find('4') != std::string::npos ||
              lower_text.find("four") != std::string::npos,
          "text answer contains 4/four");
    Check(result.prompt_tokens > 0 && result.output_tokens > 0,
          "token statistics are set");

    // Image by path.
    Check(Generate(handle,
                   "What color is the square? Answer with one word.",
                   kImagePath, nullptr, 0, 0, 1, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "image-path prompt succeeds");
    const std::string image_answer =
        result.text ? std::string(result.text, result.text_len) : "";
    std::cout << "  image answer: " << image_answer << std::endl;
    Check(ToLower(image_answer).find("red") != std::string::npos,
          "image answer contains red");

    // Raw RGB image.
    const std::vector<unsigned char> red =
        MakeRedSquare(96, 96, 16);
    Check(Generate(handle,
                   "What color is the square? Answer with one word.",
                   nullptr, red.data(), 96, 96, 1, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "raw-image prompt succeeds");
    const std::string raw_answer =
        result.text ? std::string(result.text, result.text_len) : "";
    std::cout << "  raw-image answer: " << raw_answer << std::endl;
    Check(ToLower(raw_answer).find("red") != std::string::npos,
          "raw-image answer contains red");

    // Multi-turn history.
    Check(Generate(handle, "My name is Alice.", nullptr, nullptr, 0, 0, 1,
                   &result) == AUDIO_ENGINE_SUCCESS,
          "multi-turn first prompt succeeds");
    Check(Generate(handle, "What is my name? Answer with one word.", nullptr,
                   nullptr, 0, 0, 0, &result) == AUDIO_ENGINE_SUCCESS,
          "multi-turn second prompt succeeds");
    const std::string history_answer =
        result.text ? std::string(result.text, result.text_len) : "";
    std::cout << "  history answer: " << history_answer << std::endl;
    Check(ToLower(history_answer).find("alice") != std::string::npos,
          "history answer contains Alice");

    // Reset clears the conversation.
    Check(AudioEngine_Llm_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    Check(Generate(handle, "Say OK.", nullptr, nullptr, 0, 0, 0, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "prompt after Reset succeeds");

    Check(AudioEngine_Llm_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(InitHandle(handle), "Init again after Deinit");
    Check(Generate(handle, "What is 2+2? Answer with a single digit.", nullptr,
                   nullptr, 0, 0, 1, &result) == AUDIO_ENGINE_SUCCESS,
          "generation after re-Init succeeds");
    Check(AudioEngine_Llm_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit again");
    Check(AudioEngine_Llm_Destroy(handle) == AUDIO_ENGINE_SUCCESS, "Destroy");
}

}  // namespace

int main()
{
    if (!AssetsAvailable()) {
        std::cout << "SKIP: LLM model/assets are missing" << std::endl;
        return 0;
    }

    TestUninitializedErrors();
    TestInvalidInit();
    TestModel();

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << g_failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
