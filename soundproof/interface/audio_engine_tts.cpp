/*
 * @Author: Ryuk
 * @Date: 2026-11-05
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-05 00:00:00
 * @Description: TTS interface implementation
 */

#include <new>
#include <string>
#include <vector>

#include "audio_engine_log.h"
#include "audio_engine_tts.h"

#include "core/text_to_speech/tts_engine.h"


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

int MapStatus(soundproof::TtsStatus status, const char* api)
{
    const char* reason = "";
    int code = AUDIO_ENGINE_SUCCESS;
    switch (status) {
        case soundproof::TtsStatus::kOk:
            return AUDIO_ENGINE_SUCCESS;
        case soundproof::TtsStatus::kNotInitialized:
            reason = "not initialized";
            code = AUDIO_ENGINE_ERR_NOT_INITIALIZED;
            break;
        case soundproof::TtsStatus::kModelLoadFailed:
            reason = "model load failed";
            code = AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED;
            break;
        case soundproof::TtsStatus::kUnsupportedModel:
            reason = "unsupported model";
            code = AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL;
            break;
        case soundproof::TtsStatus::kUnsupportedInput:
            reason = "unsupported input type";
            code = AUDIO_ENGINE_ERR_UNSUPPORTED_INPUT;
            break;
        case soundproof::TtsStatus::kInvalidInput:
            reason = "invalid input";
            code = AUDIO_ENGINE_ERR_INVALID_PARAM;
            break;
        case soundproof::TtsStatus::kInferenceFailed:
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
class AudioEngineTts {
public:
    AudioEngineTts()
        : initialized_(false)
        , engine_(nullptr) {
        rt_config_.speed = 1.0f;
        rt_config_.speaker_id = 0;
    }

    ~AudioEngineTts() {
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
    }

    bool initialized_;
    TtsInitConfig init_config_;
    TtsRtConfig rt_config_;
    soundproof::TtsEngine* engine_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API TtsHandle AudioEngine_Tts_Create(void)
{
    LOG_DEBUG("AudioEngine_Tts_Create");

    AudioEngineTts* ptr = new (std::nothrow) AudioEngineTts();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Tts_Create: memory allocation failed");
    }
    return static_cast<TtsHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Tts_Destroy(TtsHandle handle)
{
    LOG_DEBUG("AudioEngine_Tts_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Tts_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineTts*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Tts_Init(TtsHandle handle,
                                          const TtsInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Tts_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Tts_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Tts_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->output_sample_rate < 0) {
        LOG_WARN("AudioEngine_Tts_Init: invalid output_sample_rate=%d",
                 init_config->output_sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string model_path = PathFrom(init_config->model_path);
    if (model_path.empty()) {
        LOG_WARN("AudioEngine_Tts_Init: model_path is required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    if (ptr->engine_) {
        LOG_DEBUG("AudioEngine_Tts_Init: already initialized, deinit first");
        delete ptr->engine_;
        ptr->engine_ = nullptr;
    }
    ptr->initialized_ = false;
    ptr->init_config_ = *init_config;
    ptr->rt_config_.speed = 1.0f;
    ptr->rt_config_.speaker_id = 0;

    soundproof::TtsModelConfig config;
    config.model_type = PathFrom(init_config->model_type);
    config.model_path = model_path;
    config.voices_path = PathFrom(init_config->voices_path);
    config.tokens_path = PathFrom(init_config->tokens_path);
    config.lexicon_path = PathFrom(init_config->lexicon_path);
    config.data_dir = PathFrom(init_config->data_dir);
    config.dict_dir = PathFrom(init_config->dict_dir);
    config.num_threads =
        init_config->num_threads > 0 ? init_config->num_threads : 1;
    config.output_sample_rate = init_config->output_sample_rate;

    ptr->engine_ = new (std::nothrow) soundproof::TtsEngine();
    if (!ptr->engine_ || !ptr->engine_->Init(config)) {
        LOG_ERROR("AudioEngine_Tts_Init: TTS engine init failed");
        delete ptr->engine_;
        ptr->engine_ = nullptr;
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->initialized_ = true;
    LOG_INFO("AudioEngine_Tts_Init OK, model_type='%s', sample_rate=%d, "
             "speakers=%d",
             config.model_type.c_str(), ptr->engine_->SampleRate(),
             ptr->engine_->NumSpeakers());
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Tts_Deinit(TtsHandle handle)
{
    LOG_DEBUG("AudioEngine_Tts_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Tts_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
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
AUDIO_ENGINE_API int AudioEngine_Tts_SetParam(TtsHandle handle,
                                              const TtsRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Tts_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Tts_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Tts_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->speed <= 0.0f || rt_config->speaker_id < 0) {
        LOG_WARN("AudioEngine_Tts_SetParam: invalid speed=%.2f speaker_id=%d",
                 rt_config->speed, rt_config->speaker_id);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Tts_SetParam OK, speed=%.2f, speaker_id=%d",
             ptr->rt_config_.speed, ptr->rt_config_.speaker_id);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Tts_ResetParam(TtsHandle handle,
                                                const TtsRtConfig* rt_config)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Tts_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Tts_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->speed <= 0.0f || rt_config->speaker_id < 0) {
        LOG_WARN("AudioEngine_Tts_ResetParam: invalid speed=%.2f speaker_id=%d",
                 rt_config->speed, rt_config->speaker_id);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    ptr->rt_config_.speed = 1.0f;
    ptr->rt_config_.speaker_id = 0;
    ptr->rt_config_ = *rt_config;
    LOG_INFO("AudioEngine_Tts_ResetParam OK, speed=%.2f, speaker_id=%d",
             ptr->rt_config_.speed, ptr->rt_config_.speaker_id);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Tts_Process(TtsHandle handle,
                                             const TtsInput_t* input,
                                             TtsResult_t* result)
{
    LOG_DEBUG("AudioEngine_Tts_Process, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Tts_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!input || !result) {
        LOG_WARN("AudioEngine_Tts_Process: input/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Tts_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    if (input->input_type < 0 || input->input_type > 2) {
        LOG_WARN("AudioEngine_Tts_Process: invalid input_type=%d",
                 input->input_type);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (input->input_type == 0 || input->input_type == 1) {
        if (!input->text) {
            LOG_WARN("AudioEngine_Tts_Process: text is NULL");
            return AUDIO_ENGINE_ERR_NULL_POINTER;
        }
    } else if (!input->token_ids || input->num_tokens <= 0) {
        LOG_WARN("AudioEngine_Tts_Process: invalid token_ids/num_tokens");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    soundproof::TtsRequest request;
    request.input_type =
        static_cast<soundproof::TtsInputType>(input->input_type);
    if (input->text) {
        request.text = input->text;
    }
    if (input->token_ids && input->num_tokens > 0) {
        request.token_ids.assign(input->token_ids,
                                 input->token_ids + input->num_tokens);
    }
    request.speaker_id = ptr->rt_config_.speaker_id;
    request.speed = ptr->rt_config_.speed;

    soundproof::TtsAudio audio;
    const soundproof::TtsStatus status = ptr->engine_->Synthesize(request, &audio);
    const int ret = MapStatus(status, "AudioEngine_Tts_Process");
    if (ret != AUDIO_ENGINE_SUCCESS) {
        result->audio_out = nullptr;
        result->out_samples = 0;
        result->sample_rate = 0;
        return ret;
    }

    // Point at the engine-owned buffer (valid until the next Process/Reset).
    const soundproof::TtsAudio& generated = ptr->engine_->last_audio();
    result->audio_out = generated.samples.data();
    result->out_samples = static_cast<int>(generated.samples.size());
    result->sample_rate = generated.sample_rate;
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Tts_GetSampleRate(TtsHandle handle,
                                                   int* sample_rate)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Tts_GetSampleRate: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!sample_rate) {
        LOG_WARN("AudioEngine_Tts_GetSampleRate: sample_rate is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Tts_GetSampleRate: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    *sample_rate = ptr->engine_->SampleRate();
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Tts_GetNumSpeakers(TtsHandle handle,
                                                    int* num_speakers)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Tts_GetNumSpeakers: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!num_speakers) {
        LOG_WARN("AudioEngine_Tts_GetNumSpeakers: num_speakers is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Tts_GetNumSpeakers: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    *num_speakers = ptr->engine_->NumSpeakers();
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Tts_Reset(TtsHandle handle)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Tts_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineTts* ptr = static_cast<AudioEngineTts*>(handle);
    if (!ptr->initialized_ || !ptr->engine_) {
        LOG_WARN("AudioEngine_Tts_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->engine_->Reset();
    return AUDIO_ENGINE_SUCCESS;
}
