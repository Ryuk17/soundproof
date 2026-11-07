/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: RAG interface implementation (text embedding)
 */

#include <exception>
#include <new>
#include <string>

#include "audio_engine_log.h"
#include "audio_engine_rag.h"

#include "core/text_embedding/embedding_engine.h"


namespace {

template <size_t N>
std::string PathFrom(const char (&field)[N])
{
    size_t len = 0;
    while (len < N && field[len] != '\0') {
        ++len;
    }
    return std::string(field, len);
}

int MapStatus(soundproof::EmbeddingStatus status, const char* api)
{
    const char* reason = "";
    int code = AUDIO_ENGINE_SUCCESS;
    switch (status) {
        case soundproof::EmbeddingStatus::kOk:
            return AUDIO_ENGINE_SUCCESS;
        case soundproof::EmbeddingStatus::kNotInitialized:
            reason = "not initialized";
            code = AUDIO_ENGINE_ERR_NOT_INITIALIZED;
            break;
        case soundproof::EmbeddingStatus::kModelLoadFailed:
            reason = "model load failed";
            code = AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED;
            break;
        case soundproof::EmbeddingStatus::kUnsupportedModel:
            reason = "unsupported model";
            code = AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL;
            break;
        case soundproof::EmbeddingStatus::kInvalidInput:
            reason = "invalid input";
            code = AUDIO_ENGINE_ERR_INVALID_PARAM;
            break;
        case soundproof::EmbeddingStatus::kEmbeddingDimMismatch:
            reason = "embedding dimension mismatch";
            code = AUDIO_ENGINE_ERR_EMBEDDING_DIM_MISMATCH;
            break;
        case soundproof::EmbeddingStatus::kInferenceFailed:
            reason = "inference failed";
            code = AUDIO_ENGINE_ERR_INFERENCE_FAILED;
            break;
    }
    LOG_WARN("%s: %s", api, reason);
    return code;
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineRag {
public:
    AudioEngineRag()
        : initialized_(false)
        , engine_(nullptr) {
        rt_config_.normalize = 1;
    }

    ~AudioEngineRag() {
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
    }

    bool initialized_;
    RagInitConfig init_config_;
    RagRtConfig rt_config_;
    soundproof::EmbeddingEngine* engine_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API RagHandle AudioEngine_Rag_Create(void)
{
    LOG_DEBUG("AudioEngine_Rag_Create");

    AudioEngineRag* ptr = new (std::nothrow) AudioEngineRag();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Rag_Create: memory allocation failed");
    }
    return static_cast<RagHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Rag_Destroy(RagHandle handle)
{
    LOG_DEBUG("AudioEngine_Rag_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineRag*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Rag_Init(
    RagHandle handle,
    const RagInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Rag_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Rag_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->num_threads < 0) {
        LOG_WARN("AudioEngine_Rag_Init: invalid num_threads=%d",
                 init_config->num_threads);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string config_path = PathFrom(init_config->config_path);
    if (config_path.empty()) {
        LOG_WARN("AudioEngine_Rag_Init: config_path is required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    if (ptr->engine_) {
        LOG_DEBUG("AudioEngine_Rag_Init: already initialized, deinit first");
        delete ptr->engine_;
        ptr->engine_ = nullptr;
    }
    ptr->initialized_ = false;

    soundproof::EmbeddingEngine* engine = nullptr;
    try {
        engine = new (std::nothrow) soundproof::EmbeddingEngine();
        if (!engine) {
            LOG_ERROR("AudioEngine_Rag_Init: memory allocation failed");
            return AUDIO_ENGINE_ERR_INIT_FAILED;
        }

        soundproof::EmbeddingModelConfig model_config;
        model_config.model_type = PathFrom(init_config->model_type);
        model_config.config_path = config_path;
        model_config.num_threads = init_config->num_threads;

        const soundproof::EmbeddingStatus status = engine->Init(model_config);
        if (status != soundproof::EmbeddingStatus::kOk) {
            delete engine;
            return MapStatus(status, "AudioEngine_Rag_Init");
        }
    } catch (const std::exception& e) {
        if (engine) {
            delete engine;
        }
        LOG_ERROR("AudioEngine_Rag_Init: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    } catch (...) {
        if (engine) {
            delete engine;
        }
        LOG_ERROR("AudioEngine_Rag_Init: unknown exception");
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->engine_ = engine;
    ptr->init_config_ = *init_config;
    // Init must restore the run-time defaults (CLAUDE.md 3.2).
    ptr->rt_config_.normalize = 1;
    ptr->initialized_ = true;

    LOG_INFO("AudioEngine_Rag_Init OK, dim=%d, threads=%d",
             engine->Dim(), ptr->init_config_.num_threads);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Rag_Deinit(RagHandle handle)
{
    LOG_DEBUG("AudioEngine_Rag_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    if (ptr->engine_) {
        delete ptr->engine_;
        ptr->engine_ = nullptr;
    }
    ptr->initialized_ = false;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * SetParam / ResetParam
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Rag_SetParam(
    RagHandle handle,
    const RagRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Rag_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Rag_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->normalize != 0 && rt_config->normalize != 1) {
        LOG_WARN("AudioEngine_Rag_SetParam: invalid normalize=%d",
                 rt_config->normalize);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Rag_SetParam OK, normalize=%d",
             ptr->rt_config_.normalize);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Rag_ResetParam(
    RagHandle handle,
    const RagRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Rag_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Rag_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->normalize != 0 && rt_config->normalize != 1) {
        LOG_WARN("AudioEngine_Rag_ResetParam: invalid normalize=%d",
                 rt_config->normalize);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    ptr->rt_config_.normalize = 1;
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Rag_ResetParam OK, normalize=%d",
             ptr->rt_config_.normalize);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Rag_Process(
    RagHandle handle,
    const RagInput_t* input,
    RagResult_t* result)
{
    LOG_DEBUG("AudioEngine_Rag_Process, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!input || !result) {
        LOG_WARN("AudioEngine_Rag_Process: input/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!input->text) {
        LOG_WARN("AudioEngine_Rag_Process: text is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Rag_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    if (input->text[0] == '\0') {
        LOG_WARN("AudioEngine_Rag_Process: empty text");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    try {
        const soundproof::EmbeddingStatus status = ptr->engine_->Compute(
            input->text, ptr->rt_config_.normalize != 0);
        if (status != soundproof::EmbeddingStatus::kOk) {
            result->embedding = nullptr;
            result->dim = 0;
            return MapStatus(status, "AudioEngine_Rag_Process");
        }
    } catch (const std::exception& e) {
        result->embedding = nullptr;
        result->dim = 0;
        LOG_ERROR("AudioEngine_Rag_Process: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    } catch (...) {
        result->embedding = nullptr;
        result->dim = 0;
        LOG_ERROR("AudioEngine_Rag_Process: unknown exception");
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    }

    result->embedding = ptr->engine_->embedding();
    result->dim = ptr->engine_->output_dim();
    if (result->embedding == nullptr || result->dim <= 0) {
        result->dim = 0;
        LOG_ERROR("AudioEngine_Rag_Process: engine returned no embedding");
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    }
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Queries
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Rag_GetDim(
    RagHandle handle,
    int* dim)
{
    LOG_DEBUG("AudioEngine_Rag_GetDim, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_GetDim: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!dim) {
        LOG_WARN("AudioEngine_Rag_GetDim: dim is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Rag_GetDim: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    const int value = ptr->engine_->Dim();
    if (value <= 0) {
        LOG_ERROR("AudioEngine_Rag_GetDim: invalid dimension %d", value);
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }
    *dim = value;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Rag_Reset(RagHandle handle)
{
    LOG_DEBUG("AudioEngine_Rag_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Rag_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineRag* ptr = static_cast<AudioEngineRag*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Rag_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->engine_->Reset();
    return AUDIO_ENGINE_SUCCESS;
}
