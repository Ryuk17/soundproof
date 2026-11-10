// Tests for libAE_SR — speaker recognition (embedding + registration +
// online identification triggered like KWS).
//
// Uses models/speaker_recognition/3dspeaker_...mnn and the 16 kHz assets.
// Tests are skipped when the model is missing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/speaker_recognition/speaker_embedding_manager.h"
#include "interface/audio_engine_sr.h"
#include "utils/dr_wav.h"

#define FRAME_LEN (160)  /* 10 ms @ 16 kHz */

namespace {

const char* kModelPath =
    "models/speaker_recognition/"
    "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.mnn";
const char* kWavA = "assets/audio/audio_nearend16k.wav";
const char* kWavB = "assets/audio/audio_farend16k.wav";

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
    const char* files[] = {kModelPath, kWavA, kWavB};
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

bool LoadWav(const char* path, std::vector<int16_t>* samples, int max_seconds)
{
    DrWavReader reader(path);
    if (reader.sample_rate() != 16000 || reader.num_channels() != 1) {
        return false;
    }
    int total = reader.num_samples();
    const int limit = 16000 * max_seconds;
    if (total > limit) {
        total = limit;
    }
    samples->resize(static_cast<size_t>(total));
    return reader.ReadSamples(total, samples->data()) == total;
}

void FillConfig(SrInitConfig* config, const char* model_type)
{
    std::memset(config, 0, sizeof(*config));
    config->sample_rate = 16000;
    config->frame_len = FRAME_LEN;
    config->num_threads = 1;
    std::snprintf(config->model_type, sizeof(config->model_type), "%s",
                  model_type == nullptr ? "" : model_type);
    std::snprintf(config->model_path, sizeof(config->model_path), "%s",
                  kModelPath);
}

bool InitHandle(SrHandle handle, const char* model_type)
{
    SrInitConfig config;
    FillConfig(&config, model_type);
    const int ret = AudioEngine_Sr_Init(handle, &config);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: Init returned " << ret << std::endl;
        return false;
    }
    return true;
}

struct SrOutput {
    std::string name;
    int has_result = 0;
    float score = 0.0f;
    int name_len = 0;
};

int CallProcess(SrHandle handle, const int16_t* frame, int trigger,
                SrOutput* out)
{
    char name[256] = {0};
    SrResult_t result;
    result.speaker_name = name;
    result.name_capacity = sizeof(name);
    result.name_len = 0;
    result.has_result = 0;
    result.score = 0.0f;
    SrInput_t input{frame, FRAME_LEN, trigger};
    const int ret = AudioEngine_Sr_Process(handle, &input, &result);
    if (out != nullptr) {
        out->name = name;
        out->has_result = result.has_result;
        out->score = result.score;
        out->name_len = result.name_len;
    }
    return ret;
}

// Feeds |samples| in 10 ms frames. When |trigger| is true the last frame
// carries trigger=1.
int FeedAudio(SrHandle handle, const std::vector<int16_t>& samples,
              bool trigger, SrOutput* out)
{
    if (out != nullptr) {
        *out = SrOutput();
    }
    if (samples.size() < FRAME_LEN) {
        return -1;
    }
    int ret = AUDIO_ENGINE_SUCCESS;
    for (size_t pos = 0; pos + FRAME_LEN <= samples.size(); pos += FRAME_LEN) {
        const bool last = (pos + FRAME_LEN == samples.size());
        SrOutput current;
        ret = CallProcess(handle, samples.data() + pos,
                          (trigger && last) ? 1 : 0, &current);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            return ret;
        }
        if (last && out != nullptr) {
            *out = current;
        }
    }
    return ret;
}

int CallIdentify(SrHandle handle, const std::vector<int16_t>& samples,
                 SrOutput* out)
{
    char name[256] = {0};
    SrResult_t result;
    result.speaker_name = name;
    result.name_capacity = sizeof(name);
    result.name_len = 0;
    result.has_result = 0;
    result.score = 0.0f;
    const int ret = AudioEngine_Sr_Identify(
        handle, samples.data(), static_cast<int>(samples.size()), &result);
    out->name = name;
    out->has_result = result.has_result;
    out->score = result.score;
    out->name_len = result.name_len;
    return ret;
}

void TestManager()
{
    std::cout << "=== test_manager ===" << std::endl;

    soundproof::SpeakerEmbeddingManager manager(4);
    const float a[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float b[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    const float mixed[4] = {0.8f, 0.2f, 0.0f, 0.0f};

    Check(manager.Add("alice", a), "add alice");
    Check(!manager.Add("alice", a), "duplicate add rejected");
    Check(manager.Add("bob", b), "add bob");
    Check(manager.NumSpeakers() == 2, "speaker count");
    Check(manager.GetAllSpeakers() ==
              std::vector<std::string>({"alice", "bob"}),
          "speakers sorted");

    float score = 0.0f;
    Check(manager.Search(mixed, 0.5f, &score) == "alice" && score > 0.9f,
          "search returns alice with high score");
    Check(manager.Verify("alice", mixed, 0.5f, &score) && score > 0.9f,
          "verify alice");
    Check(!manager.Verify("bob", mixed, 0.5f, &score),
          "verify bob fails");

    const std::vector<std::vector<float>> list = {
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}};
    Check(manager.Add("carol", list), "add averaged embeddings");
    Check(manager.Search(mixed, 0.0f, &score) == "alice" ||
              manager.Search(mixed, 0.0f, &score) == "carol",
          "averaged embedding stored");

    Check(manager.Remove("bob"), "remove bob");
    Check(!manager.Remove("bob"), "remove missing returns false");
    manager.Clear();
    Check(manager.NumSpeakers() == 0, "clear registry");
}

void TestModelExtraction()
{
    std::cout << "=== test_extraction ===" << std::endl;

    std::vector<int16_t> samples;
    if (!LoadWav(kWavA, &samples, 3)) {
        Check(false, "load WAV A");
        return;
    }
    const std::vector<int16_t> segment(samples.begin(),
                                       samples.begin() + 16000);

    SrHandle handle = AudioEngine_Sr_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle, ""), "Init (auto model type)");

    float emb1[512];
    float emb2[512];
    int dim1 = 0;
    int dim2 = 0;
    Check(AudioEngine_Sr_Extract(handle, segment.data(),
                                 static_cast<int>(segment.size()), emb1,
                                 sizeof(emb1) / sizeof(float), &dim1) ==
              AUDIO_ENGINE_SUCCESS,
          "Extract embedding");
    Check(AudioEngine_Sr_Extract(handle, segment.data(),
                                 static_cast<int>(segment.size()), emb2,
                                 sizeof(emb2) / sizeof(float), &dim2) ==
              AUDIO_ENGINE_SUCCESS,
          "Extract embedding again");
    Check(dim1 == 192 && dim2 == dim1, "embedding dim is 192");
    Check(std::memcmp(emb1, emb2, sizeof(float) * dim1) == 0,
          "embedding is deterministic");

    AudioEngine_Sr_Deinit(handle);
    AudioEngine_Sr_Destroy(handle);
}

void TestRegisterAndStreaming(const std::vector<int16_t>& wav_a,
                              const std::vector<int16_t>& wav_b)
{
    std::cout << "=== test_register_and_streaming ===" << std::endl;

    const std::vector<int16_t> enroll(wav_a.begin(), wav_a.begin() + 24000);
    const std::vector<int16_t> same(wav_a.begin(), wav_a.begin() + 24000);
    const std::vector<int16_t> other(wav_b.begin(), wav_b.begin() + 24000);

    SrHandle handle = AudioEngine_Sr_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle, "3d-speaker"), "Init (explicit model type)");

    Check(AudioEngine_Sr_Register(handle, "alice", enroll.data(),
                                  static_cast<int>(enroll.size())) ==
              AUDIO_ENGINE_SUCCESS,
          "register alice from PCM");
    int count = 0;
    Check(AudioEngine_Sr_GetNumSpeakers(handle, &count) ==
              AUDIO_ENGINE_SUCCESS &&
              count == 1,
          "one registered speaker");

    // Simulate KWS: feed frames without trigger, then trigger on the last one.
    SrOutput streamed;
    Check(FeedAudio(handle, same, true, &streamed) == AUDIO_ENGINE_SUCCESS,
          "streaming Process with KWS trigger");
    Check(streamed.has_result == 1, "trigger produced a result");
    std::cout << "  streamed: name=" << streamed.name
              << " score=" << streamed.score << std::endl;
    Check(streamed.name == "alice" && streamed.score > 0.5f,
          "streaming identifies alice");

    // After a trigger the feature cache is empty -> another trigger is short.
    int16_t silence[FRAME_LEN] = {0};
    Check(CallProcess(handle, silence, 1, nullptr) ==
              AUDIO_ENGINE_ERR_AUDIO_TOO_SHORT,
          "trigger without cached audio reports AUDIO_TOO_SHORT");

    // One-shot segment identification.
    SrOutput identified;
    Check(CallIdentify(handle, same, &identified) == AUDIO_ENGINE_SUCCESS,
          "Identify segment");
    Check(identified.name == "alice" && identified.score > 0.9f,
          "Identify returns alice with high score");

    SrOutput different;
    Check(CallIdentify(handle, other, &different) == AUDIO_ENGINE_SUCCESS,
          "Identify a different segment");
    std::cout << "  other: name=" << different.name
              << " score=" << different.score << std::endl;
    Check(different.score < identified.score,
          "different audio scores lower than the enrolled audio");

    // Duplicate registration and removal.
    Check(AudioEngine_Sr_Register(handle, "alice", enroll.data(),
                                  static_cast<int>(enroll.size())) ==
              AUDIO_ENGINE_ERR_SR_SPEAKER_EXISTS,
          "duplicate registration rejected");
    Check(AudioEngine_Sr_Remove(handle, "alice") == AUDIO_ENGINE_SUCCESS,
          "remove alice");
    Check(AudioEngine_Sr_Remove(handle, "alice") ==
              AUDIO_ENGINE_ERR_SR_SPEAKER_NOT_FOUND,
          "remove missing speaker rejected");

    // RegisterFiles (average of two files).
    const char* files[2] = {kWavB, kWavB};
    Check(AudioEngine_Sr_RegisterFiles(handle, "bob", files, 2) ==
              AUDIO_ENGINE_SUCCESS,
          "RegisterFiles registers bob");

    AudioEngine_Sr_Deinit(handle);
    AudioEngine_Sr_Destroy(handle);
}

void TestErrors(const std::vector<int16_t>& wav_a)
{
    std::cout << "=== test_errors ===" << std::endl;

    Check(AudioEngine_Sr_Destroy(NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Destroy(NULL)");
    Check(AudioEngine_Sr_Init(NULL, NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Init(NULL, NULL)");

    SrHandle handle = AudioEngine_Sr_Create();
    int16_t frame[FRAME_LEN] = {0};
    SrOutput out;
    Check(CallProcess(handle, frame, 0, &out) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init");

    SrInitConfig config;
    FillConfig(&config, "");
    config.sample_rate = 8000;
    config.frame_len = 80;
    Check(AudioEngine_Sr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "8 kHz is rejected");

    FillConfig(&config, "");
    config.frame_len = 100;
    Check(AudioEngine_Sr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "non-10 ms frame is rejected");

    FillConfig(&config, "");
    std::snprintf(config.model_path, sizeof(config.model_path), "%s",
                  "models/speaker_recognition/does_not_exist.mnn");
    Check(AudioEngine_Sr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INIT_FAILED,
          "missing model returns INIT_FAILED");

    Check(InitHandle(handle, ""), "Init with valid config after failures");

    // Audio too short (0.1 s).
    std::vector<int16_t> tiny(wav_a.begin(), wav_a.begin() + 1600);
    Check(AudioEngine_Sr_Register(handle, "tiny", tiny.data(),
                                  static_cast<int>(tiny.size())) ==
              AUDIO_ENGINE_ERR_AUDIO_TOO_SHORT,
          "short registration reports AUDIO_TOO_SHORT");

    const std::vector<int16_t> segment(wav_a.begin(), wav_a.begin() + 24000);
    Check(AudioEngine_Sr_Register(handle, "alice", segment.data(),
                                  static_cast<int>(segment.size())) ==
              AUDIO_ENGINE_SUCCESS,
          "register alice for buffer test");

    // Buffer too small.
    char name[2] = {0};
    SrResult_t result;
    result.speaker_name = name;
    result.name_capacity = sizeof(name);
    Check(AudioEngine_Sr_Identify(handle, segment.data(),
                                  static_cast<int>(segment.size()), &result) ==
              AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL,
          "small name buffer reports BUFFER_TOO_SMALL");

    float embedding[4];
    int dim = 0;
    Check(AudioEngine_Sr_Extract(handle, segment.data(),
                                 static_cast<int>(segment.size()), embedding,
                                 4, &dim) ==
              AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL &&
              dim == 192,
          "small embedding buffer reports required dim");

    // Wrong embedding dimension.
    float wrong[8] = {0};
    Check(AudioEngine_Sr_AddEmbedding(handle, "carol", wrong, 8) ==
              AUDIO_ENGINE_ERR_EMBEDDING_DIM_MISMATCH,
          "wrong embedding dim is rejected");

    SrRtConfig rt = {};
    rt.threshold = 1.5f;
    Check(AudioEngine_Sr_SetParam(handle, &rt) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "threshold out of range is rejected");

    Check(AudioEngine_Sr_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(CallProcess(handle, frame, 0, &out) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process after Deinit");

    AudioEngine_Sr_Destroy(handle);
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "SR Interface Unit Test" << std::endl;
    std::cout << "======================" << std::endl;

    TestManager();

    if (!AssetsAvailable()) {
        std::cout << "Model tests skipped (assets missing)." << std::endl;
    } else {
        std::vector<int16_t> wav_a;
        std::vector<int16_t> wav_b;
        if (LoadWav(kWavA, &wav_a, 10) && LoadWav(kWavB, &wav_b, 10)) {
            TestModelExtraction();
            TestRegisterAndStreaming(wav_a, wav_b);
            TestErrors(wav_a);
        } else {
            Check(false, "load 16 kHz test WAVs");
        }
    }

    std::cout << "All tests done." << std::endl;
    if (g_failures != 0) {
        std::cout << g_failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    return 0;
}
