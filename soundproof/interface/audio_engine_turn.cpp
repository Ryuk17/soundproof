/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: Turn detection interface implementation
 */

#include <cstring>
#include <exception>
#include <new>
#include <string>

#include "audio_engine_log.h"
#include "audio_engine_turn.h"

#include "core/turn_detection/turn_detector.h"


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

int MapStatus(soundproof::TurnStatus status, const char* api)
{
    const char* reason = "";
    int code = AUDIO_ENGINE_SUCCESS;
    switch (status) {
        case soundproof::TurnStatus::kOk:
            return AUDIO_ENGINE_SUCCESS;
        case soundproof::TurnStatus::kNotInitialized:
            reason = "not initialized";
            code = AUDIO_ENGINE_ERR_NOT_INITIALIZED;
            break;
        case soundproof::TurnStatus::kModelLoadFailed:
            reason = "model load failed";
            code = AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED;
            break;
        case soundproof::TurnStatus::kUnsupportedModel:
            reason = "unsupported model";
            code = AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL;
            break;
        case soundproof::TurnStatus::kInvalidInput:
            reason = "invalid input";
            code = AUDIO_ENGINE_ERR_INVALID_PARAM;
            break;
        case soundproof::TurnStatus::kFeatureFailed:
            reason = "feature extraction failed";
            code = AUDIO_ENGINE_ERR_FEATURE_FAILED;
            break;
        case soundproof::TurnStatus::kInferenceFailed:
            reason = "inference failed";
            code = AUDIO_ENGINE_ERR_INFERENCE_FAILED;
            break;
    }
    LOG_WARN("%s: %s", api, reason);
    return code;
}

void ClearResult(TurnResult_t* result)
{
    std::memset(result, 0, sizeof(*result));
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineTurn {
public:
    AudioEngineTurn()
        : initialized_(false)
        , detector_(nullptr) {
        rt_config_.threshold = 0.5f;
    }

    ~AudioEngineTurn() {
        if (detector_) {
            delete detector_;
            detector_ = nullptr;
        }
    }

    bool initialized_;
    TurnInitConfig init_config_;
    TurnRtConfig rt_config_;
    soundproof::TurnDetector* detector_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API TurnHandle AudioEngine_Turn_Create(void)
{
    LOG_DEBUG("AudioEngine_Turn_Create");

    AudioEngineTurn* ptr = new (std::nothrow) AudioEngineTurn();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Turn_Create: memory allocation failed");
    }
    return static_cast<TurnHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Turn_Destroy(TurnHandle handle)
{
    LOG_DEBUG("AudioEngine_Turn_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineTurn*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Turn_Init(
    TurnHandle handle,
    const TurnInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Turn_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Turn_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->sample_rate != 16000) {
        LOG_WARN("AudioEngine_Turn_Init: unsupported sample_rate=%d",
                 init_config->sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->num_threads < 0) {
        LOG_WARN("AudioEngine_Turn_Init: invalid num_threads=%d",
                 init_config->num_threads);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string model_path = PathFrom(init_config->model_path);
    if (model_path.empty()) {
        LOG_WARN("AudioEngine_Turn_Init: model_path is required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    if (ptr->detector_) {
        LOG_DEBUG("AudioEngine_Turn_Init: already initialized, deinit first");
        delete ptr->detector_;
        ptr->detector_ = nullptr;
    }
    ptr->initialized_ = false;

    soundproof::TurnDetector* detector = nullptr;
    try {
        detector = new (std::nothrow) soundproof::TurnDetector();
        if (!detector) {
            LOG_ERROR("AudioEngine_Turn_Init: memory allocation failed");
            return AUDIO_ENGINE_ERR_INIT_FAILED;
        }

        soundproof::TurnModelConfig model_config;
        model_config.model_type = PathFrom(init_config->model_type);
        model_config.model_path = model_path;
        model_config.num_threads = init_config->num_threads;

        const soundproof::TurnStatus status = detector->Init(model_config);
        if (status != soundproof::TurnStatus::kOk) {
            delete detector;
            return MapStatus(status, "AudioEngine_Turn_Init");
        }
    } catch (const std::exception& e) {
        if (detector) {
            delete detector;
        }
        LOG_ERROR("AudioEngine_Turn_Init: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    } catch (...) {
        if (detector) {
            delete detector;
        }
        LOG_ERROR("AudioEngine_Turn_Init: unknown exception");
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->detector_ = detector;
    ptr->init_config_ = *init_config;
    // Init must restore the run-time defaults (CLAUDE.md 3.2).
    ptr->rt_config_.threshold = 0.5f;
    ptr->initialized_ = true;

    LOG_INFO("AudioEngine_Turn_Init OK, context=%d samples, threads=%d",
             detector->ContextSamples(), ptr->init_config_.num_threads);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Turn_Deinit(TurnHandle handle)
{
    LOG_DEBUG("AudioEngine_Turn_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    if (ptr->detector_) {
        delete ptr->detector_;
        ptr->detector_ = nullptr;
    }
    ptr->initialized_ = false;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * SetParam / ResetParam
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Turn_SetParam(
    TurnHandle handle,
    const TurnRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Turn_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Turn_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!(rt_config->threshold > 0.0f && rt_config->threshold <= 1.0f)) {
        LOG_WARN("AudioEngine_Turn_SetParam: invalid threshold=%.3f",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Turn_SetParam OK, threshold=%.3f",
             ptr->rt_config_.threshold);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Turn_ResetParam(
    TurnHandle handle,
    const TurnRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Turn_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Turn_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!(rt_config->threshold > 0.0f && rt_config->threshold <= 1.0f)) {
        LOG_WARN("AudioEngine_Turn_ResetParam: invalid threshold=%.3f",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    ptr->rt_config_.threshold = 0.5f;
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Turn_ResetParam OK, threshold=%.3f",
             ptr->rt_config_.threshold);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Turn_Process(
    TurnHandle handle,
    const TurnInput_t* input,
    TurnResult_t* result)
{
    LOG_DEBUG("AudioEngine_Turn_Process, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!input || !result) {
        LOG_WARN("AudioEngine_Turn_Process: input/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (input->in_samples > 0 && !input->audio_in) {
        LOG_WARN("AudioEngine_Turn_Process: audio_in is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    if (!ptr->initialized_ || !ptr->detector_) {
        LOG_WARN("AudioEngine_Turn_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    if (input->in_samples < 0) {
        LOG_WARN("AudioEngine_Turn_Process: invalid in_samples=%d",
                 input->in_samples);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (input->trigger != 0 && input->trigger != 1) {
        LOG_WARN("AudioEngine_Turn_Process: invalid trigger=%d",
                 input->trigger);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    ClearResult(result);
    try {
        if (input->in_samples > 0) {
            const soundproof::TurnStatus status =
                ptr->detector_->Append(input->audio_in, input->in_samples);
            if (status != soundproof::TurnStatus::kOk) {
                return MapStatus(status, "AudioEngine_Turn_Process");
            }
        }
        result->buffered_samples = ptr->detector_->BufferedSamples();

        if (input->trigger == 0) {
            return AUDIO_ENGINE_SUCCESS;
        }
        if (result->buffered_samples == 0) {
            LOG_WARN("AudioEngine_Turn_Process: trigger without buffered audio");
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }

        float probability = 0.0f;
        const soundproof::TurnStatus status =
            ptr->detector_->Predict(&probability);
        if (status != soundproof::TurnStatus::kOk) {
            return MapStatus(status, "AudioEngine_Turn_Process");
        }

        result->probability = probability;
        result->is_complete =
            probability >= ptr->rt_config_.threshold ? 1 : 0;
        result->has_result = 1;
        return AUDIO_ENGINE_SUCCESS;
    } catch (const std::exception& e) {
        ClearResult(result);
        LOG_ERROR("AudioEngine_Turn_Process: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    } catch (...) {
        ClearResult(result);
        LOG_ERROR("AudioEngine_Turn_Process: unknown exception");
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    }
}


/* ============================================================================
 * Queries
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Turn_GetContextSamples(
    TurnHandle handle,
    int* samples)
{
    LOG_DEBUG("AudioEngine_Turn_GetContextSamples, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_GetContextSamples: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!samples) {
        LOG_WARN("AudioEngine_Turn_GetContextSamples: samples is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    if (!ptr->initialized_ || !ptr->detector_) {
        LOG_WARN("AudioEngine_Turn_GetContextSamples: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    const int value = ptr->detector_->ContextSamples();
    if (value <= 0) {
        LOG_ERROR("AudioEngine_Turn_GetContextSamples: invalid context %d",
                  value);
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }
    *samples = value;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Turn_Reset(TurnHandle handle)
{
    LOG_DEBUG("AudioEngine_Turn_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Turn_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineTurn* ptr = static_cast<AudioEngineTurn*>(handle);
    if (!ptr->initialized_ || !ptr->detector_) {
        LOG_WARN("AudioEngine_Turn_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->detector_->Reset();
    return AUDIO_ENGINE_SUCCESS;
}
