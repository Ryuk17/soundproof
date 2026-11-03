// Tests for libAE_KWS — streaming zipformer2 keyword spotting.
//
// Uses the model under models/kws/kws-zipformer-wenetspeech-3.3M-MNN and its
// bundled test wavs/keywords. The test is skipped when those assets are not
// present (model files are not tracked by this repository).

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "interface/audio_engine_kws.h"
#include "utils/dr_wav.h"

#define FRAME_LEN (160)  /* 10 ms @ 16 kHz */

namespace {

const char* kModelDir = "models/kws/kws-zipformer-wenetspeech-3.3M-MNN";
const char* kEncoder =
    "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/"
    "encoder-epoch-99-avg-1-chunk-16-left-64.mnn";
const char* kDecoder =
    "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/"
    "decoder-epoch-99-avg-1-chunk-16-left-64.mnn";
const char* kJoiner =
    "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/"
    "joiner-epoch-99-avg-1-chunk-16-left-64.mnn";
const char* kTokens =
    "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/tokens.txt";
const char* kKeywords =
    "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/test_keywords.txt";

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
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return false;
    }
    fclose(fp);
    return true;
}

bool AssetsAvailable()
{
    const char* files[] = {kEncoder, kDecoder, kJoiner, kTokens, kKeywords};
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

void FillInitConfig(KwsInitConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->sample_rate = 16000;
    config->frame_len = FRAME_LEN;
    config->num_threads = 1;
    std::snprintf(config->encoder_path, sizeof(config->encoder_path), "%s", kEncoder);
    std::snprintf(config->decoder_path, sizeof(config->decoder_path), "%s", kDecoder);
    std::snprintf(config->joiner_path, sizeof(config->joiner_path), "%s", kJoiner);
    std::snprintf(config->tokens_path, sizeof(config->tokens_path), "%s", kTokens);
    std::snprintf(config->keywords_file_path, sizeof(config->keywords_file_path),
                  "%s", kKeywords);
}

struct Detection {
    int index = -1;
    float score = 0.0f;
    int frame = -1;
};

// Feeds every 10 ms frame of |path|; returns the first detection (if any).
bool RunWav(KwsHandle handle, const char* path, Detection* first,
            int* voice_frames, std::vector<Detection>* all)
{
    DrWavReader reader(path);
    if (reader.sample_rate() != 16000 || reader.num_channels() != 1) {
        std::cerr << "  FAIL: expected 16 kHz mono WAV: " << path << std::endl;
        return false;
    }

    int16_t frame[FRAME_LEN];
    int frame_index = 0;
    int last_read = -1;
    while (true) {
        std::memset(frame, 0, sizeof(frame));
        const int read = reader.ReadSamples(FRAME_LEN, frame);
        if (read == 0) {
            break;
        }
        last_read = read;

        int index = -1;
        float score = 0.0f;
        const int ret =
            AudioEngine_Kws_Process(handle, frame, FRAME_LEN, &index, &score);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            std::cerr << "  FAIL: Process returned " << ret << " at frame "
                      << frame_index << std::endl;
            return false;
        }
        if (index >= 0) {
            Detection detection;
            detection.index = index;
            detection.score = score;
            detection.frame = frame_index;
            if (first->index < 0) {
                *first = detection;
            }
            if (voice_frames != nullptr) {
                ++(*voice_frames);
            }
            if (all != nullptr) {
                all->push_back(detection);
            }
        }
        ++frame_index;
    }
    (void)last_read;
    return true;
}

bool InitHandle(KwsHandle handle)
{
    KwsInitConfig config;
    FillInitConfig(&config);
    const int ret = AudioEngine_Kws_Init(handle, &config);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: Init returned " << ret << std::endl;
        return false;
    }
    return true;
}

void TestDetection()
{
    std::cout << "=== test_detection ===" << std::endl;

    // Expected keyword indices follow test_wavs/test_keywords.txt:
    //   3.wav -> 0 (文森特卡索), 4.wav -> 3 (蒋友伯),
    //   5.wav -> 1 (周望军),     6.wav -> 2 (朱丽楠)
    struct WavCase {
        const char* path;
        int expected_index;
    };
    const WavCase wavs[] = {
        {"models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/3.wav", 0},
        {"models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/4.wav", 3},
        {"models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/5.wav", 1},
        {"models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/6.wav", 2},
    };

    int detected = 0;
    for (const WavCase& wav : wavs) {
        if (!FileExists(wav.path)) {
            continue;
        }
        KwsHandle handle = AudioEngine_Kws_Create();
        if (!handle || !InitHandle(handle)) {
            Check(false, std::string("init for ") + wav.path);
            if (handle) {
                AudioEngine_Kws_Destroy(handle);
            }
            continue;
        }

        Detection first;
        const bool ok = RunWav(handle, wav.path, &first, nullptr, nullptr);
        std::cout << "  " << wav.path << ": index=" << first.index
                  << " (expected " << wav.expected_index
                  << ") score=" << first.score << " frame=" << first.frame
                  << std::endl;
        Check(ok && first.index == wav.expected_index,
              std::string("detects the expected keyword in ") + wav.path);
        Check(first.score > 0.0f && first.score <= 1.0f,
              "score is within (0, 1]");
        if (ok && first.index == wav.expected_index) {
            ++detected;
        }

        AudioEngine_Kws_Deinit(handle);
        AudioEngine_Kws_Destroy(handle);
    }
    Check(detected == 4, "all bundled test wavs trigger their keywords");
}

void TestResetDeterminism(const char* wav)
{
    std::cout << "=== test_reset ===" << std::endl;

    KwsHandle handle = AudioEngine_Kws_Create();
    if (!handle || !InitHandle(handle)) {
        Check(false, "init for reset test");
        if (handle) {
            AudioEngine_Kws_Destroy(handle);
        }
        return;
    }

    Detection first;
    Check(RunWav(handle, wav, &first, nullptr, nullptr) && first.index >= 0,
          "first pass detects a keyword");

    Check(AudioEngine_Kws_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");

    Detection second;
    Check(RunWav(handle, wav, &second, nullptr, nullptr) && second.index >= 0,
          "second pass detects a keyword after Reset");
    Check(first.index == second.index,
          "Reset reproduces the same keyword index");

    AudioEngine_Kws_Deinit(handle);
    AudioEngine_Kws_Destroy(handle);
}

void TestSilence()
{
    std::cout << "=== test_silence ===" << std::endl;

    KwsHandle handle = AudioEngine_Kws_Create();
    if (!handle || !InitHandle(handle)) {
        Check(false, "init for silence test");
        if (handle) {
            AudioEngine_Kws_Destroy(handle);
        }
        return;
    }

    int16_t silence[FRAME_LEN] = {0};
    int hits = 0;
    for (int i = 0; i < 300; ++i) {  /* 3 seconds */
        int index = -1;
        float score = 0.0f;
        if (AudioEngine_Kws_Process(handle, silence, FRAME_LEN, &index,
                                    &score) != AUDIO_ENGINE_SUCCESS) {
            Check(false, "silence processing failed");
            break;
        }
        if (index >= 0) {
            ++hits;
        }
    }
    Check(hits == 0, "silence does not trigger a keyword");

    AudioEngine_Kws_Deinit(handle);
    AudioEngine_Kws_Destroy(handle);
}

void TestErrorCases()
{
    std::cout << "=== test_error_cases ===" << std::endl;

    Check(AudioEngine_Kws_Destroy(NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Destroy(NULL)");
    Check(AudioEngine_Kws_Init(NULL, NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Init(NULL, NULL)");

    KwsHandle handle = AudioEngine_Kws_Create();
    int index = -1;
    float score = 0.0f;
    int16_t frame[FRAME_LEN] = {0};
    Check(AudioEngine_Kws_Process(handle, frame, FRAME_LEN, &index, &score) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init");

    KwsInitConfig config;
    FillInitConfig(&config);

    config.sample_rate = 8000;
    config.frame_len = 80;
    Check(AudioEngine_Kws_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "8 kHz is rejected");

    FillInitConfig(&config);
    config.frame_len = 100;
    Check(AudioEngine_Kws_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "non-10 ms frame is rejected");

    FillInitConfig(&config);
    config.encoder_path[0] = '\0';
    Check(AudioEngine_Kws_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "missing encoder path is rejected");

    FillInitConfig(&config);
    std::snprintf(config.encoder_path, sizeof(config.encoder_path), "%s",
                  "models/kws/does_not_exist.mnn");
    Check(AudioEngine_Kws_Init(handle, &config) == AUDIO_ENGINE_ERR_INIT_FAILED,
          "missing model file returns INIT_FAILED");

    Check(InitHandle(handle), "Init with valid config after failures");

    KwsRtConfig rt = {};
    rt.threshold = 1.5f;
    Check(AudioEngine_Kws_SetParam(handle, &rt) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "threshold out of range is rejected");

    rt.threshold = 0.5f;
    Check(AudioEngine_Kws_SetParam(handle, &rt) == AUDIO_ENGINE_SUCCESS,
          "SetParam(0.5)");

    Check(AudioEngine_Kws_Process(handle, frame, FRAME_LEN, &index, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL kws_score is rejected");

    Check(AudioEngine_Kws_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(AudioEngine_Kws_Process(handle, frame, FRAME_LEN, &index, &score) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process after Deinit");

    AudioEngine_Kws_Destroy(handle);
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "KWS Interface Unit Test" << std::endl;
    std::cout << "=======================" << std::endl;

    if (!AssetsAvailable()) {
        std::cout << "All tests skipped (model assets missing)." << std::endl;
        return 0;
    }

    TestDetection();
    TestResetDeterminism(
        "models/kws/kws-zipformer-wenetspeech-3.3M-MNN/test_wavs/3.wav");
    TestSilence();
    TestErrorCases();

    std::cout << "All tests done." << std::endl;
    if (g_failures != 0) {
        std::cout << g_failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    return 0;
}
