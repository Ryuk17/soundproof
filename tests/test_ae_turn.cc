// Tests for libAE_TURN — Smart Turn v3.2 end-of-turn detection.
//
// Uses models/turn/smart-turn-v3.2-gpu.mnn and the speech assets under
// assets/. The model (~9 MB) is not tracked, so the test prints SKIP and
// succeeds when it is missing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "interface/audio_engine_turn.h"
#include "utils/dr_wav.h"

namespace {

const char* kModelPath =
    "models/turn/smart-turn-v3.2-gpu.mnn/smart-turn-v3.2-gpu.mnn";
const char* kCompleteWav = "assets/audio_reverb16.wav";
const char* kShortWav = "assets/audio_short16.wav";
const char* kIncompleteWav = "assets/audio_tiny16.wav";

constexpr int kSampleRate = 16000;
constexpr int kFrameLen = 160;  // 10 ms
constexpr int kContextSamples = 8 * kSampleRate;

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
    const char* files[] = {kModelPath, kCompleteWav, kShortWav, kIncompleteWav};
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

bool LoadWav(const char* path, std::vector<short>* samples)
{
    DrWavReader reader(path);
    if (reader.sample_rate() != kSampleRate || reader.num_channels() != 1) {
        return false;
    }
    samples->resize(static_cast<size_t>(reader.num_samples()));
    return reader.ReadSamples(reader.num_samples(), samples->data()) ==
           reader.num_samples();
}

void FillConfig(TurnInitConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->sample_rate = kSampleRate;
    config->num_threads = 1;
    std::snprintf(config->model_type, sizeof(config->model_type), "%s",
                  "smart-turn");
    std::snprintf(config->model_path, sizeof(config->model_path), "%s",
                  kModelPath);
}

bool InitHandle(TurnHandle handle)
{
    TurnInitConfig config;
    FillConfig(&config);
    return AudioEngine_Turn_Init(handle, &config) == AUDIO_ENGINE_SUCCESS;
}

// Feeds |samples| in 10 ms frames and optionally triggers at the end.
int Feed(TurnHandle handle, const short* samples, int count,
         int trigger, TurnResult_t* result)
{
    int offset = 0;
    while (offset < count) {
        const int n = std::min(kFrameLen, count - offset);
        TurnInput_t input;
        input.audio_in = samples + offset;
        input.in_samples = n;
        input.trigger = 0;
        TurnResult_t step;
        const int ret = AudioEngine_Turn_Process(handle, &input, &step);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            return ret;
        }
        offset += n;
    }
    TurnInput_t input;
    input.audio_in = nullptr;
    input.in_samples = 0;
    input.trigger = trigger;
    return AudioEngine_Turn_Process(handle, &input, result);
}

void TestUninitializedErrors()
{
    std::cout << "=== test_uninitialized_errors ===" << std::endl;

    TurnHandle handle = AudioEngine_Turn_Create();
    Check(handle != NULL, "Create");

    short sample[16] = {0};
    TurnInput_t input;
    input.audio_in = sample;
    input.in_samples = 16;
    input.trigger = 0;
    TurnResult_t result;
    int context = 0;

    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init -> NOT_INITIALIZED");
    Check(AudioEngine_Turn_GetContextSamples(handle, &context) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "GetContextSamples before Init -> NOT_INITIALIZED");
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Reset before Init -> NOT_INITIALIZED");

    TurnRtConfig rt_config;
    rt_config.threshold = 0.5f;
    Check(AudioEngine_Turn_SetParam(handle, &rt_config) == AUDIO_ENGINE_SUCCESS,
          "SetParam before Init is allowed");
    rt_config.threshold = 1.5f;
    Check(AudioEngine_Turn_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "SetParam invalid threshold -> INVALID_PARAM");
    Check(AudioEngine_Turn_SetParam(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "SetParam NULL -> NULL_POINTER");

    Check(AudioEngine_Turn_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit without Init");
    Check(AudioEngine_Turn_Destroy(handle) == AUDIO_ENGINE_SUCCESS,
          "Destroy");
    Check(AudioEngine_Turn_Process(NULL, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Process NULL handle -> INVALID_HANDLE");
}

void TestInvalidInit()
{
    std::cout << "=== test_invalid_init ===" << std::endl;

    TurnHandle handle = AudioEngine_Turn_Create();
    TurnInitConfig config;
    FillConfig(&config);

    config.sample_rate = 8000;
    Check(AudioEngine_Turn_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "sample_rate 8000 -> INVALID_PARAM");

    FillConfig(&config);
    config.num_threads = -1;
    Check(AudioEngine_Turn_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "negative num_threads -> INVALID_PARAM");

    FillConfig(&config);
    config.model_path[0] = '\0';
    Check(AudioEngine_Turn_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty model_path -> INVALID_PARAM");

    FillConfig(&config);
    std::snprintf(config.model_path, sizeof(config.model_path), "%s",
                  "models/turn/does-not-exist.mnn");
    Check(AudioEngine_Turn_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED,
          "missing model -> MODEL_LOAD_FAILED");

    FillConfig(&config);
    std::snprintf(config.model_type, sizeof(config.model_type), "%s",
                  "unknown-turn");
    Check(AudioEngine_Turn_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL,
          "unknown model_type -> UNSUPPORTED_MODEL");

    Check(AudioEngine_Turn_Init(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL init_config -> NULL_POINTER");

    AudioEngine_Turn_Destroy(handle);
}

void TestModel()
{
    std::cout << "=== test_model ===" << std::endl;

    TurnHandle handle = AudioEngine_Turn_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle), "Init Smart Turn v3.2");

    int context = 0;
    Check(AudioEngine_Turn_GetContextSamples(handle, &context) ==
              AUDIO_ENGINE_SUCCESS &&
              context == kContextSamples,
          "context is 128000 samples");

    short sample[16] = {0};
    TurnInput_t input;
    input.audio_in = sample;
    input.in_samples = 16;
    input.trigger = 0;
    TurnResult_t result;

    input.audio_in = nullptr;
    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL audio with samples -> NULL_POINTER");
    input.audio_in = sample;
    input.in_samples = -1;
    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "negative in_samples -> INVALID_PARAM");
    input.in_samples = 16;
    input.trigger = 2;
    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "invalid trigger -> INVALID_PARAM");
    Check(AudioEngine_Turn_Process(handle, NULL, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL input -> NULL_POINTER");
    Check(AudioEngine_Turn_Process(handle, &input, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL result -> NULL_POINTER");

    input.audio_in = nullptr;
    input.in_samples = 0;
    input.trigger = 1;
    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "trigger without audio -> INVALID_PARAM");

    std::vector<short> complete;
    std::vector<short> incomplete;
    std::vector<short> long_wav;
    Check(LoadWav(kCompleteWav, &complete), "load complete-turn wav");
    Check(LoadWav(kIncompleteWav, &incomplete), "load incomplete-turn wav");
    Check(LoadWav(kShortWav, &long_wav), "load long wav");

    // Full utterance: expected complete (ONNX reference ~0.98).
    const int ret_complete = Feed(handle, complete.data(),
                                  static_cast<int>(complete.size()), 1,
                                  &result);
    Check(ret_complete == AUDIO_ENGINE_SUCCESS && result.has_result == 1,
          "detect complete utterance");
    std::cout << "  complete probability=" << result.probability << std::endl;
    Check(result.probability > 0.5f && result.is_complete == 1,
          "complete utterance -> is_complete=1");
    Check(result.buffered_samples == static_cast<int>(complete.size()),
          "buffered_samples matches fed audio");

    // First second of a short utterance: expected incomplete (ONNX ~0.01).
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    const int incomplete_samples = std::min(kSampleRate,
                                            static_cast<int>(incomplete.size()));
    const int ret_incomplete = Feed(handle, incomplete.data(),
                                    incomplete_samples, 1, &result);
    Check(ret_incomplete == AUDIO_ENGINE_SUCCESS && result.has_result == 1,
          "detect incomplete utterance");
    std::cout << "  incomplete probability=" << result.probability << std::endl;
    Check(result.probability < 0.5f && result.is_complete == 0,
          "incomplete utterance -> is_complete=0");

    const float first_probability = result.probability;

    // Same audio again: deterministic.
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    Check(Feed(handle, incomplete.data(), incomplete_samples, 1, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "detect incomplete utterance again");
    Check(std::fabs(result.probability - first_probability) < 1e-5f,
          "same audio is deterministic");

    // Left-pad consistency: one call vs two half-length calls.
    const int half = incomplete_samples / 2;
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    TurnInput_t part;
    part.audio_in = incomplete.data();
    part.in_samples = half;
    part.trigger = 0;
    Check(AudioEngine_Turn_Process(handle, &part, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "feed first half");
    part.audio_in = incomplete.data() + half;
    part.in_samples = incomplete_samples - half;
    part.trigger = 1;
    Check(AudioEngine_Turn_Process(handle, &part, &result) ==
              AUDIO_ENGINE_SUCCESS,
          "feed second half and trigger");
    Check(std::fabs(result.probability - first_probability) < 1e-5f,
          "left padding is independent of chunking");

    // Rolling window: feed more than 8 s, keep the last 8 s.
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    Check(Feed(handle, long_wav.data(), static_cast<int>(long_wav.size()), 1,
               &result) == AUDIO_ENGINE_SUCCESS,
          "feed long utterance");
    Check(result.buffered_samples == kContextSamples,
          "rolling window is capped at 128000");
    const float rolling_probability = result.probability;

    Check(static_cast<int>(long_wav.size()) >= kContextSamples,
          "long wav is at least 8 s");
    if (static_cast<int>(long_wav.size()) < kContextSamples) {
        AudioEngine_Turn_Deinit(handle);
        AudioEngine_Turn_Destroy(handle);
        return;
    }
    std::vector<short> tail(long_wav.end() - kContextSamples, long_wav.end());
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    Check(Feed(handle, tail.data(), static_cast<int>(tail.size()), 1,
               &result) == AUDIO_ENGINE_SUCCESS,
          "feed last 8 seconds directly");
    Check(std::fabs(result.probability - rolling_probability) < 1e-5f,
          "rolling window equals the last 8 seconds");

    // Threshold is applied to the cached probability.
    const float probability = result.probability;
    TurnRtConfig rt_config;
    rt_config.threshold = std::min(1.0f, probability + 0.1f);
    Check(AudioEngine_Turn_SetParam(handle, &rt_config) == AUDIO_ENGINE_SUCCESS,
          "SetParam high threshold");
    Check(Feed(handle, nullptr, 0, 1, &result) == AUDIO_ENGINE_SUCCESS &&
              result.probability == probability && result.is_complete == 0,
          "high threshold flips is_complete to 0");

    rt_config.threshold = std::max(0.01f, probability - 0.1f);
    Check(AudioEngine_Turn_ResetParam(handle, &rt_config) ==
              AUDIO_ENGINE_SUCCESS,
          "ResetParam low threshold");
    Check(Feed(handle, nullptr, 0, 1, &result) == AUDIO_ENGINE_SUCCESS &&
              result.probability == probability && result.is_complete == 1,
          "low threshold flips is_complete to 1");

    // Reset clears the buffer.
    Check(AudioEngine_Turn_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    input.audio_in = nullptr;
    input.in_samples = 0;
    input.trigger = 1;
    Check(AudioEngine_Turn_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "trigger after Reset -> INVALID_PARAM");

    Check(AudioEngine_Turn_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(InitHandle(handle), "Init again after Deinit");
    Check(AudioEngine_Turn_GetContextSamples(handle, &context) ==
              AUDIO_ENGINE_SUCCESS &&
              context == kContextSamples,
          "context after re-Init");
    Check(AudioEngine_Turn_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit again");
    Check(AudioEngine_Turn_Destroy(handle) == AUDIO_ENGINE_SUCCESS, "Destroy");
}

}  // namespace

int main()
{
    if (!AssetsAvailable()) {
        std::cout << "SKIP: turn detection model/assets are missing"
                  << std::endl;
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
