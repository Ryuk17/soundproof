/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: Chat LLM/VLM interface implementation
 */

#include <cstring>
#include <exception>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include "audio_engine_log.h"
#include "audio_engine_llm.h"

#include "core/llm/llm_engine.h"


namespace {

constexpr int kMaxImageDimension = 8192;

template <size_t N>
std::string PathFrom(const char (&field)[N])
{
    size_t len = 0;
    while (len < N && field[len] != '\0') {
        ++len;
    }
    return std::string(field, len);
}

int MapStatus(soundproof::LlmStatus status, const char* api)
{
    const char* reason = "";
    int code = AUDIO_ENGINE_SUCCESS;
    switch (status) {
        case soundproof::LlmStatus::kOk:
            return AUDIO_ENGINE_SUCCESS;
        case soundproof::LlmStatus::kNotInitialized:
            reason = "not initialized";
            code = AUDIO_ENGINE_ERR_NOT_INITIALIZED;
            break;
        case soundproof::LlmStatus::kModelLoadFailed:
            reason = "model load failed";
            code = AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED;
            break;
        case soundproof::LlmStatus::kUnsupportedModel:
            reason = "unsupported model";
            code = AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL;
            break;
        case soundproof::LlmStatus::kInvalidInput:
            reason = "invalid input";
            code = AUDIO_ENGINE_ERR_INVALID_PARAM;
            break;
        case soundproof::LlmStatus::kFeatureFailed:
            reason = "feature extraction failed";
            code = AUDIO_ENGINE_ERR_FEATURE_FAILED;
            break;
        case soundproof::LlmStatus::kInferenceFailed:
            reason = "inference failed";
            code = AUDIO_ENGINE_ERR_INFERENCE_FAILED;
            break;
    }
    LOG_WARN("%s: %s", api, reason);
    return code;
}

bool FileExists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

void ClearResult(LlmResult_t* result)
{
    std::memset(result, 0, sizeof(*result));
}

soundproof::LlmSamplingParams ToSamplingParams(const LlmRtConfig& config)
{
    soundproof::LlmSamplingParams params;
    params.max_new_tokens = config.max_new_tokens;
    params.temperature = config.temperature;
    params.top_p = config.top_p;
    params.top_k = config.top_k;
    params.enable_thinking = config.enable_thinking;
    return params;
}

bool ValidRtConfig(const LlmRtConfig* config)
{
    if (!config) {
        return false;
    }
    if (config->max_new_tokens < 0) {
        return false;
    }
    if (!(config->temperature >= -1.0f)) {
        return false;
    }
    if (!(config->top_p == -1.0f ||
          (config->top_p > 0.0f && config->top_p <= 1.0f))) {
        return false;
    }
    if (!(config->top_k == -1 || config->top_k > 0)) {
        return false;
    }
    if (config->enable_thinking < -1 || config->enable_thinking > 1) {
        return false;
    }
    if (config->strip_thinking != 0 && config->strip_thinking != 1) {
        return false;
    }
    return true;
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineLlm {
public:
    AudioEngineLlm()
        : initialized_(false)
        , engine_(nullptr) {
        SetDefaults();
    }

    ~AudioEngineLlm() {
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
    }

    void SetDefaults() {
        rt_config_.max_new_tokens = 512;
        rt_config_.temperature = -1.0f;
        rt_config_.top_p = -1.0f;
        rt_config_.top_k = -1;
        rt_config_.enable_thinking = -1;
        rt_config_.strip_thinking = 1;
    }

    bool initialized_;
    LlmInitConfig init_config_;
    LlmRtConfig rt_config_;
    soundproof::LlmEngine* engine_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API LlmHandle AudioEngine_Llm_Create(void)
{
    LOG_DEBUG("AudioEngine_Llm_Create");

    AudioEngineLlm* ptr = new (std::nothrow) AudioEngineLlm();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Llm_Create: memory allocation failed");
    }
    return static_cast<LlmHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Llm_Destroy(LlmHandle handle)
{
    LOG_DEBUG("AudioEngine_Llm_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineLlm*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Llm_Init(
    LlmHandle handle,
    const LlmInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Llm_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Llm_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->num_threads < 0) {
        LOG_WARN("AudioEngine_Llm_Init: invalid num_threads=%d",
                 init_config->num_threads);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string config_path = PathFrom(init_config->config_path);
    if (config_path.empty()) {
        LOG_WARN("AudioEngine_Llm_Init: config_path is required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
    if (ptr->engine_) {
        LOG_DEBUG("AudioEngine_Llm_Init: already initialized, deinit first");
        delete ptr->engine_;
        ptr->engine_ = nullptr;
    }
    ptr->initialized_ = false;

    soundproof::LlmEngine* engine = nullptr;
    try {
        engine = new (std::nothrow) soundproof::LlmEngine();
        if (!engine) {
            LOG_ERROR("AudioEngine_Llm_Init: memory allocation failed");
            return AUDIO_ENGINE_ERR_INIT_FAILED;
        }

        soundproof::LlmModelConfig model_config;
        model_config.model_type = PathFrom(init_config->model_type);
        model_config.config_path = config_path;
        model_config.num_threads = init_config->num_threads;
        if (init_config->system_prompt) {
            model_config.system_prompt = init_config->system_prompt;
        }

        const soundproof::LlmStatus status = engine->Init(model_config);
        if (status != soundproof::LlmStatus::kOk) {
            delete engine;
            return MapStatus(status, "AudioEngine_Llm_Init");
        }
    } catch (const std::exception& e) {
        if (engine) {
            delete engine;
        }
        LOG_ERROR("AudioEngine_Llm_Init: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    } catch (...) {
        if (engine) {
            delete engine;
        }
        LOG_ERROR("AudioEngine_Llm_Init: unknown exception");
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->engine_ = engine;
    ptr->init_config_ = *init_config;
    // Do not keep a pointer into caller-owned memory.
    ptr->init_config_.system_prompt = nullptr;
    // Init must restore the run-time defaults (CLAUDE.md 3.2).
    ptr->SetDefaults();
    ptr->initialized_ = true;

    LOG_INFO("AudioEngine_Llm_Init OK, threads=%d",
             ptr->init_config_.num_threads);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Llm_Deinit(LlmHandle handle)
{
    LOG_DEBUG("AudioEngine_Llm_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
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
AUDIO_ENGINE_API int AudioEngine_Llm_SetParam(
    LlmHandle handle,
    const LlmRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Llm_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Llm_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!ValidRtConfig(rt_config)) {
        LOG_WARN("AudioEngine_Llm_SetParam: invalid parameters");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
    ptr->rt_config_ = *rt_config;
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Llm_ResetParam(
    LlmHandle handle,
    const LlmRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Llm_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Llm_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!ValidRtConfig(rt_config)) {
        LOG_WARN("AudioEngine_Llm_ResetParam: invalid parameters");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
    ptr->SetDefaults();
    ptr->rt_config_ = *rt_config;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Llm_Process(
    LlmHandle handle,
    const LlmInput_t* input,
    LlmResult_t* result)
{
    LOG_DEBUG("AudioEngine_Llm_Process, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!input || !result) {
        LOG_WARN("AudioEngine_Llm_Process: input/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Llm_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    if (input->reset_history != 0 && input->reset_history != 1) {
        LOG_WARN("AudioEngine_Llm_Process: invalid reset_history=%d",
                 input->reset_history);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    ClearResult(result);
    try {
        std::string text;
        if (input->text) {
            text = input->text;
        }

        soundproof::LlmImage image;
        if (input->image_data) {
            if (input->image_width <= 0 || input->image_height <= 0 ||
                input->image_width > kMaxImageDimension ||
                input->image_height > kMaxImageDimension) {
                LOG_WARN("AudioEngine_Llm_Process: invalid image size %dx%d",
                         input->image_width, input->image_height);
                return AUDIO_ENGINE_ERR_INVALID_PARAM;
            }
            const size_t bytes =
                static_cast<size_t>(input->image_width) *
                static_cast<size_t>(input->image_height) * 3;
            image.rgb.assign(input->image_data, input->image_data + bytes);
            image.width = input->image_width;
            image.height = input->image_height;
        } else if (input->image_path && input->image_path[0] != '\0') {
            image.path = input->image_path;
            if (!FileExists(image.path)) {
                LOG_WARN("AudioEngine_Llm_Process: image not found: %s",
                         image.path.c_str());
                return AUDIO_ENGINE_ERR_INVALID_PARAM;
            }
        }

        if (text.empty() && image.empty()) {
            LOG_WARN("AudioEngine_Llm_Process: empty text and image");
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }

        const soundproof::LlmStatus status = ptr->engine_->Generate(
            text, image, input->reset_history != 0,
            ToSamplingParams(ptr->rt_config_),
            ptr->rt_config_.strip_thinking != 0);
        if (status != soundproof::LlmStatus::kOk) {
            return MapStatus(status, "AudioEngine_Llm_Process");
        }

        const std::string& output = ptr->engine_->last_text();
        result->text = output.c_str();
        result->text_len = static_cast<int>(output.size());
        result->prompt_tokens = ptr->engine_->prompt_tokens();
        result->output_tokens = ptr->engine_->output_tokens();
        return AUDIO_ENGINE_SUCCESS;
    } catch (const std::exception& e) {
        ClearResult(result);
        LOG_ERROR("AudioEngine_Llm_Process: exception: %s", e.what());
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    } catch (...) {
        ClearResult(result);
        LOG_ERROR("AudioEngine_Llm_Process: unknown exception");
        return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
    }
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Llm_Reset(LlmHandle handle)
{
    LOG_DEBUG("AudioEngine_Llm_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Llm_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineLlm* ptr = static_cast<AudioEngineLlm*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Llm_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->engine_->Reset();
    return AUDIO_ENGINE_SUCCESS;
}
