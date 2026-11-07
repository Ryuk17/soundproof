// Tests for libAE_RAG — RAG text embeddings via MNN-LLM.
//
// Uses models/embedding/Qwen3-Embedding-0.6B-MNN/config.json. The model is
// ~360 MB and is not tracked, so the test prints SKIP and succeeds when the
// assets are missing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "interface/audio_engine_rag.h"

namespace {

const char* kModelDir = "models/embedding/Qwen3-Embedding-0.6B-MNN";
const char* kConfigPath = "models/embedding/Qwen3-Embedding-0.6B-MNN/config.json";

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
        "models/embedding/Qwen3-Embedding-0.6B-MNN/config.json",
        "models/embedding/Qwen3-Embedding-0.6B-MNN/llm_config.json",
        "models/embedding/Qwen3-Embedding-0.6B-MNN/embedding.mnn",
        "models/embedding/Qwen3-Embedding-0.6B-MNN/embedding.mnn.weight",
        "models/embedding/Qwen3-Embedding-0.6B-MNN/embeddings_int4.bin",
        "models/embedding/Qwen3-Embedding-0.6B-MNN/tokenizer.txt",
    };
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

void FillConfig(RagInitConfig* config)
{
    std::memset(config, 0, sizeof(*config));
    config->num_threads = 0;
    std::snprintf(config->model_type, sizeof(config->model_type), "%s", "");
    std::snprintf(config->config_path, sizeof(config->config_path), "%s",
                  kConfigPath);
}

bool InitHandle(RagHandle handle)
{
    RagInitConfig config;
    FillConfig(&config);
    return AudioEngine_Rag_Init(handle, &config) == AUDIO_ENGINE_SUCCESS;
}

bool Embed(RagHandle handle, const std::string& text,
           std::vector<float>* out, int* ret)
{
    RagInput_t input;
    input.text = text.c_str();
    RagResult_t result;
    std::memset(&result, 0, sizeof(result));
    *ret = AudioEngine_Rag_Process(handle, &input, &result);
    if (*ret != AUDIO_ENGINE_SUCCESS || result.embedding == nullptr ||
        result.dim <= 0) {
        return false;
    }
    out->assign(result.embedding, result.embedding + result.dim);
    return true;
}

float Norm(const std::vector<float>& values)
{
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    return static_cast<float>(std::sqrt(sum));
}

bool AllFinite(const std::vector<float>& values)
{
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

float CosSim(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    const float norm = Norm(a) * Norm(b);
    return norm <= 0.0f ? 0.0f : static_cast<float>(dot / norm);
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        return 1e9f;
    }
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

void TestUninitializedErrors()
{
    std::cout << "=== test_uninitialized_errors ===" << std::endl;

    RagHandle handle = AudioEngine_Rag_Create();
    Check(handle != NULL, "Create");

    RagInput_t input;
    input.text = "hello";
    RagResult_t result;
    int dim = 0;

    Check(AudioEngine_Rag_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init -> NOT_INITIALIZED");
    Check(AudioEngine_Rag_GetDim(handle, &dim) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "GetDim before Init -> NOT_INITIALIZED");
    Check(AudioEngine_Rag_Reset(handle) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Reset before Init -> NOT_INITIALIZED");

    RagRtConfig rt_config;
    rt_config.normalize = 1;
    Check(AudioEngine_Rag_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_SUCCESS,
          "SetParam before Init is allowed");
    rt_config.normalize = 2;
    Check(AudioEngine_Rag_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "SetParam invalid normalize -> INVALID_PARAM");
    Check(AudioEngine_Rag_SetParam(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "SetParam NULL -> NULL_POINTER");

    Check(AudioEngine_Rag_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit without Init");
    Check(AudioEngine_Rag_Destroy(handle) == AUDIO_ENGINE_SUCCESS,
          "Destroy");

    Check(AudioEngine_Rag_Process(NULL, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Process NULL handle -> INVALID_HANDLE");

    RagHandle second = AudioEngine_Rag_Create();
    Check(second != NULL, "Create again");
    AudioEngine_Rag_Destroy(second);
}

void TestInvalidInit()
{
    std::cout << "=== test_invalid_init ===" << std::endl;

    RagHandle handle = AudioEngine_Rag_Create();
    RagInitConfig config;
    FillConfig(&config);

    config.config_path[0] = '\0';
    Check(AudioEngine_Rag_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty config_path -> INVALID_PARAM");

    std::snprintf(config.config_path, sizeof(config.config_path), "%s",
                  "models/embedding/does-not-exist/config.json");
    Check(AudioEngine_Rag_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED,
          "missing config -> MODEL_LOAD_FAILED");

    Check(AudioEngine_Rag_Init(handle, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL init_config -> NULL_POINTER");

    AudioEngine_Rag_Destroy(handle);
}

void TestModel()
{
    std::cout << "=== test_model ===" << std::endl;

    RagHandle handle = AudioEngine_Rag_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle), "Init Qwen3-Embedding");

    int dim = 0;
    Check(AudioEngine_Rag_GetDim(handle, &dim) == AUDIO_ENGINE_SUCCESS &&
              dim == 1024,
          "embedding dim is 1024");

    RagInput_t input;
    RagResult_t result;
    input.text = NULL;
    Check(AudioEngine_Rag_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL text -> NULL_POINTER");
    Check(AudioEngine_Rag_Process(handle, NULL, &result) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL input -> NULL_POINTER");
    Check(AudioEngine_Rag_Process(handle, &input, NULL) ==
              AUDIO_ENGINE_ERR_NULL_POINTER,
          "NULL result -> NULL_POINTER");

    input.text = "";
    Check(AudioEngine_Rag_Process(handle, &input, &result) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "empty text -> INVALID_PARAM");

    std::vector<float> first;
    std::vector<float> second;
    int ret = AUDIO_ENGINE_SUCCESS;
    Check(Embed(handle, "hello world", &first, &ret) && first.size() == 1024,
          "embed 'hello world'");
    Check(AllFinite(first), "values are finite");
    Check(std::fabs(Norm(first) - 1.0f) < 1e-3f,
          "default output is L2-normalized");

    Check(Embed(handle, "hello world", &second, &ret), "embed same text again");
    Check(MaxAbsDiff(first, second) < 1e-6f, "same text is deterministic");

    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    Check(Embed(handle, "The weather is nice today.", &a, &ret) &&
              Embed(handle, "It is a beautiful sunny day.", &b, &ret) &&
              Embed(handle, "The stock market crashed yesterday.", &c, &ret),
          "embed semantic test sentences");
    const float sim_related = CosSim(a, b);
    const float sim_unrelated = CosSim(a, c);
    std::cout << "  related cos=" << sim_related
              << ", unrelated cos=" << sim_unrelated << std::endl;
    Check(sim_related > sim_unrelated + 0.05f,
          "related sentences are closer than unrelated ones");

    RagRtConfig rt_config;
    rt_config.normalize = 0;
    Check(AudioEngine_Rag_SetParam(handle, &rt_config) ==
              AUDIO_ENGINE_SUCCESS,
          "SetParam normalize=0");
    std::vector<float> raw;
    Check(Embed(handle, "hello world", &raw, &ret), "embed raw vector");
    const float raw_norm = Norm(raw);
    Check(AllFinite(raw) && raw_norm > 0.0f, "raw vector is valid");

    rt_config.normalize = 1;
    Check(AudioEngine_Rag_ResetParam(handle, &rt_config) ==
              AUDIO_ENGINE_SUCCESS,
          "ResetParam restores normalize=1");
    std::vector<float> normalized;
    Check(Embed(handle, "hello world", &normalized, &ret),
          "embed normalized vector again");
    std::vector<float> expected(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        expected[i] = raw[i] / raw_norm;
    }
    Check(MaxAbsDiff(normalized, expected) < 1e-5f,
          "normalized vector equals raw / ||raw||");

    Check(AudioEngine_Rag_Reset(handle) == AUDIO_ENGINE_SUCCESS,
          "Reset");
    std::vector<float> after_reset;
    Check(Embed(handle, "hello world", &after_reset, &ret),
          "embed after Reset");
    Check(MaxAbsDiff(after_reset, normalized) < 1e-6f,
          "Reset does not change results");

    Check(AudioEngine_Rag_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit");
    Check(InitHandle(handle), "Init again after Deinit");
    Check(AudioEngine_Rag_GetDim(handle, &dim) == AUDIO_ENGINE_SUCCESS &&
              dim == 1024,
          "dim after re-Init");
    Check(AudioEngine_Rag_Deinit(handle) == AUDIO_ENGINE_SUCCESS,
          "Deinit again");
    Check(AudioEngine_Rag_Destroy(handle) == AUDIO_ENGINE_SUCCESS,
          "Destroy");
}

}  // namespace

int main()
{
    if (!AssetsAvailable()) {
        std::cout << "SKIP: RAG model assets are missing in " << kModelDir
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
