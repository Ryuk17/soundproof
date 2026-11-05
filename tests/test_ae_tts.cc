// Tests for libAE_TTS — Kokoro text-to-speech.
//
// Uses models/tts/kokoro-v1_0.mnn (tts.mnn + voices.mnn + tokenizer.txt).
// Plain text requires a G2P frontend, so the tests synthesize from phoneme
// strings and explicit token ids.

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "interface/audio_engine_tts.h"

namespace {

const char* kModelDir = "models/tts/kokoro-v1_0.mnn";
const char* kModelPath = "models/tts/kokoro-v1_0.mnn/tts.mnn";
const char* kVoicesPath = "models/tts/kokoro-v1_0.mnn/voices.mnn";
const char* kTokenizerPath = "models/tts/kokoro-v1_0.mnn/tokenizer.txt";

int g_failures = 0;

void Check(bool condition, const std::string& what)
{
    std::cout << "  [" << (condition ? " OK " : "FAIL") << "] " << what << std::endl;
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
    const char* files[] = {kModelPath, kVoicesPath, kTokenizerPath};
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

void FillConfig(TtsInitConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->num_threads = 1;
    config->output_sample_rate = 0;
    std::snprintf(config->model_type, sizeof(config->model_type), "kokoro");
    std::snprintf(config->model_path, sizeof(config->model_path), "%s",
                  kModelPath);
    std::snprintf(config->voices_path, sizeof(config->voices_path), "%s",
                  kVoicesPath);
    std::snprintf(config->tokens_path, sizeof(config->tokens_path), "%s",
                  kTokenizerPath);
}

bool InitHandle(TtsHandle handle)
{
    TtsInitConfig config;
    FillConfig(&config);
    const int ret = AudioEngine_Tts_Init(handle, &config);
    return ret == AUDIO_ENGINE_SUCCESS;
}

void TestInit()
{
    std::cout << "=== test_init ===" << std::endl;

    TtsHandle handle = AudioEngine_Tts_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle), "Init Kokoro");

    int sample_rate = 0;
    int speakers = 0;
    Check(AudioEngine_Tts_GetSampleRate(handle, &sample_rate) ==
              AUDIO_ENGINE_SUCCESS &&
              sample_rate == 24000,
          "sample rate is 24000");
    Check(AudioEngine_Tts_GetNumSpeakers(handle, &speakers) ==
              AUDIO_ENGINE_SUCCESS &&
              speakers > 0,
          "at least one speaker");

    AudioEngine_Tts_Deinit(handle);
    AudioEngine_Tts_Destroy(handle);
}

void TestTokenIdsAndPhonemes()
{
    std::cout << "=== test_synthesis ===" << std::endl;

    TtsHandle handle = AudioEngine_Tts_Create();
    if (!handle || !InitHandle(handle)) {
        Check(false, "Init for synthesis");
        if (handle) {
            AudioEngine_Tts_Destroy(handle);
        }
        return;
    }

    const int32_t ids[] = {10, 11, 12, 13, 14};
    TtsInput_t input{};
    input.input_type = 2;
    input.token_ids = ids;
    input.num_tokens = 5;

    TtsResult_t result{};
    const int token_ret = AudioEngine_Tts_Process(handle, &input, &result);
    Check(token_ret == AUDIO_ENGINE_SUCCESS && result.audio_out != nullptr &&
              result.out_samples > 0 && result.sample_rate == 24000,
          "synthesis from token ids");

    input.input_type = 1;
    input.text = "hello";
    input.token_ids = nullptr;
    input.num_tokens = 0;
    result = TtsResult_t{};
    const int phoneme_ret = AudioEngine_Tts_Process(handle, &input, &result);
    Check(phoneme_ret == AUDIO_ENGINE_SUCCESS && result.audio_out != nullptr &&
              result.out_samples > 0,
          "synthesis from phoneme string");

    input.input_type = 0;
    result = TtsResult_t{};
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_UNSUPPORTED_INPUT,
          "plain text is rejected without a G2P frontend");

    TtsRtConfig rt{};
    rt.speed = 0.9f;
    rt.speaker_id = 0;
    Check(AudioEngine_Tts_SetParam(handle, &rt) == AUDIO_ENGINE_SUCCESS,
          "SetParam(speed=0.9)");
    input.input_type = 2;
    input.token_ids = ids;
    input.num_tokens = 5;
    result = TtsResult_t{};
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_SUCCESS &&
              result.out_samples > 0,
          "synthesis with runtime speed");

    rt.speaker_id = 9999;
    Check(AudioEngine_Tts_SetParam(handle, &rt) == AUDIO_ENGINE_SUCCESS,
          "SetParam accepts non-negative speaker id");
    result = TtsResult_t{};
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "out-of-range speaker id is rejected at Process");

    Check(AudioEngine_Tts_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    rt.speaker_id = 0;
    AudioEngine_Tts_SetParam(handle, &rt);
    result = TtsResult_t{};
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "synthesis after Reset");

    AudioEngine_Tts_Deinit(handle);
    AudioEngine_Tts_Destroy(handle);
}

void TestErrors()
{
    std::cout << "=== test_errors ===" << std::endl;

    Check(AudioEngine_Tts_Destroy(NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Destroy(NULL)");
    Check(AudioEngine_Tts_Init(NULL, NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Init(NULL, NULL)");

    TtsHandle handle = AudioEngine_Tts_Create();
    TtsInput_t input{};
    TtsResult_t result{};
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init");

    TtsInitConfig config;
    FillConfig(&config);
    config.model_path[0] = '\0';
    Check(AudioEngine_Tts_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "missing model path is rejected");

    FillConfig(&config);
    std::snprintf(config.model_path, sizeof(config.model_path), "%s",
                  "models/tts/does_not_exist.mnn");
    Check(AudioEngine_Tts_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INIT_FAILED,
          "missing model file returns INIT_FAILED");

    Check(InitHandle(handle), "Init with valid config after failures");

    Check(AudioEngine_Tts_Process(handle, NULL, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL input is rejected");
    input.input_type = 3;
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "invalid input_type is rejected");
    input.input_type = 0;
    input.text = NULL;
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL text is rejected");
    input.input_type = 2;
    input.token_ids = NULL;
    input.num_tokens = 0;
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty token ids are rejected");

    TtsRtConfig rt{};
    rt.speed = 0.0f;
    Check(AudioEngine_Tts_SetParam(handle, &rt) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "invalid speed is rejected");

    Check(AudioEngine_Tts_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(AudioEngine_Tts_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process after Deinit");
    AudioEngine_Tts_Destroy(handle);
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "TTS Interface Unit Test" << std::endl;
    std::cout << "=======================" << std::endl;

    if (!AssetsAvailable()) {
        std::cout << "All tests skipped (model assets missing)." << std::endl;
        return 0;
    }

    TestInit();
    TestTokenIdsAndPhonemes();
    TestErrors();

    std::cout << "All tests done." << std::endl;
    if (g_failures != 0) {
        std::cout << g_failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    return 0;
}
