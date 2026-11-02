/*
 * @Author: Ryuk
 * @Date: 2026-08-09 18:44:06
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-02 00:00:00
 * @Description: VAD interface implementation (WebRTC DSP + Silero)
 */

#include <new>
#include <string>

#include "audio_engine_log.h"
#include "audio_engine_vad.h"

#include "core/voice_activity_detection/vad_decision_filter.h"
#include "core/voice_activity_detection/voice_activity_detector.h"
#ifdef AE_WITH_MNN
#include "core/voice_activity_detection/silero_vad.h"
#endif


namespace {

constexpr int kVadModeDsp = 0;
constexpr int kVadModeSilero = 1;

/* model_path is a fixed-size array that the caller may not NUL-terminate. */
std::string ModelPathFromConfig(const VadInitConfig& config)
{
    size_t len = 0;
    while (len < sizeof(config.model_path) && config.model_path[len] != '\0') {
        ++len;
    }
    return std::string(config.model_path, len);
}

void SetDefaultRtConfig(VadRtConfig* rt_config)
{
    rt_config->threshold = 0.5f;
    rt_config->min_speech_duration = 0.25f;
    rt_config->min_silence_duration = 0.5f;
    rt_config->hysteresis = 0.15f;
}

soundproof::VadDecisionFilter::Config ToFilterConfig(const VadRtConfig& rt_config,
                                                     int sample_rate)
{
    soundproof::VadDecisionFilter::Config config;
    config.threshold = rt_config.threshold;
    config.hysteresis = rt_config.hysteresis;
    config.min_speech_duration = rt_config.min_speech_duration;
    config.min_silence_duration = rt_config.min_silence_duration;
    config.sample_rate = sample_rate;
    return config;
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineVad {
public:
    AudioEngineVad()
        : initialized_(false)
        , dsp_vad_(nullptr)
#ifdef AE_WITH_MNN
        , silero_vad_(nullptr)
#endif
    {
        SetDefaultRtConfig(&rt_config_);
        decision_filter_.SetConfig(ToFilterConfig(rt_config_, 16000));
    }

    ~AudioEngineVad() {
        ReleaseInstances();
    }

    void ReleaseInstances() {
        if (dsp_vad_) {
            delete dsp_vad_;
            dsp_vad_ = nullptr;
        }
#ifdef AE_WITH_MNN
        if (silero_vad_) {
            delete silero_vad_;
            silero_vad_ = nullptr;
        }
#endif
    }

    bool initialized_;
    VadInitConfig init_config_;
    VadRtConfig rt_config_;
    soundproof::VadDecisionFilter decision_filter_;
    webrtc::VoiceActivityDetector* dsp_vad_;
#ifdef AE_WITH_MNN
    soundproof::SileroVad* silero_vad_;
#endif
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API VadHandle AudioEngine_Vad_Create(void)
{
    LOG_DEBUG("AudioEngine_Vad_Create");

    AudioEngineVad* ptr = new (std::nothrow) AudioEngineVad();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Vad_Create: memory allocation failed");
    }
    return static_cast<VadHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Vad_Destroy(VadHandle handle)
{
    LOG_DEBUG("AudioEngine_Vad_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);
    delete ptr;

    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Vad_Init(VadHandle handle, const VadInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Vad_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Vad_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->vad_mode != kVadModeDsp &&
        init_config->vad_mode != kVadModeSilero) {
        LOG_WARN("AudioEngine_Vad_Init: invalid vad_mode=%d (0=dsp, 1=silero)",
                 init_config->vad_mode);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->sample_rate <= 0) {
        LOG_WARN("AudioEngine_Vad_Init: invalid sample_rate=%d", init_config->sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->frame_len != init_config->sample_rate / 100) {
        LOG_WARN("AudioEngine_Vad_Init: frame_len=%d must equal sample_rate/100=%d (10ms constraint)",
                 init_config->frame_len, init_config->sample_rate / 100);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const int mode = init_config->vad_mode;
    const std::string model_path = ModelPathFromConfig(*init_config);

    if (mode == kVadModeSilero) {
        /* The bundled silero_vad.mnn model is 16 kHz only. */
        if (init_config->sample_rate != 16000) {
            LOG_WARN("AudioEngine_Vad_Init: silero mode supports only 16000 Hz, got %d",
                     init_config->sample_rate);
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }
        if (model_path.empty()) {
            LOG_WARN("AudioEngine_Vad_Init: silero mode requires model_path");
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }
#ifndef AE_WITH_MNN
        LOG_ERROR("AudioEngine_Vad_Init: silero mode requires a build with -DWITH_MNN=ON");
        return AUDIO_ENGINE_ERR_INIT_FAILED;
#endif
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);

    /* If already initialized, clean up first */
    if (ptr->initialized_ || ptr->dsp_vad_
#ifdef AE_WITH_MNN
        || ptr->silero_vad_
#endif
    ) {
        LOG_DEBUG("AudioEngine_Vad_Init: already initialized, deinit first");
        ptr->ReleaseInstances();
    }
    ptr->initialized_ = false;

    ptr->init_config_ = *init_config;

    /* Set runtime config defaults — prevent uninitialized values if user
       skips SetParam before Process */
    SetDefaultRtConfig(&ptr->rt_config_);
    ptr->decision_filter_.Reset();
    ptr->decision_filter_.SetConfig(
        ToFilterConfig(ptr->rt_config_, init_config->sample_rate));

    if (mode == kVadModeDsp) {
        ptr->dsp_vad_ = new (std::nothrow) webrtc::VoiceActivityDetector();
        if (!ptr->dsp_vad_) {
            LOG_ERROR("AudioEngine_Vad_Init: VoiceActivityDetector allocation failed");
            return AUDIO_ENGINE_ERR_INIT_FAILED;
        }
    } else {
#ifdef AE_WITH_MNN
        ptr->silero_vad_ = new (std::nothrow) soundproof::SileroVad(model_path);
        if (!ptr->silero_vad_ || !ptr->silero_vad_->ok()) {
            LOG_ERROR("AudioEngine_Vad_Init: silero model init failed: %s",
                      model_path.c_str());
            delete ptr->silero_vad_;
            ptr->silero_vad_ = nullptr;
            return AUDIO_ENGINE_ERR_INIT_FAILED;
        }
#endif
    }

    ptr->initialized_ = true;

    LOG_INFO("AudioEngine_Vad_Init OK, mode=%d, sample_rate=%d, frame_len=%d, threshold=%.2f",
             mode, init_config->sample_rate, init_config->frame_len,
             ptr->rt_config_.threshold);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Vad_Deinit(VadHandle handle)
{
    LOG_DEBUG("AudioEngine_Vad_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);
    ptr->ReleaseInstances();
    ptr->decision_filter_.Reset();
    ptr->initialized_ = false;

    LOG_DEBUG("AudioEngine_Vad_Deinit OK");
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * SetParam / ResetParam
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Vad_SetParam(VadHandle handle, const VadRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Vad_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Vad_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold > 1.0f) {
        LOG_WARN("AudioEngine_Vad_SetParam: threshold=%.2f out of range (0.0, 1.0]",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);

    /* Incremental update: only overwrite fields with meaningful values.
       Duration/hysteresis <= 0 keeps the current/default value, which also
       keeps legacy callers that only set `threshold` working. */
    ptr->rt_config_.threshold = rt_config->threshold;
    if (rt_config->min_speech_duration > 0.0f) {
        ptr->rt_config_.min_speech_duration = rt_config->min_speech_duration;
    }
    if (rt_config->min_silence_duration > 0.0f) {
        ptr->rt_config_.min_silence_duration = rt_config->min_silence_duration;
    }
    if (rt_config->hysteresis > 0.0f) {
        ptr->rt_config_.hysteresis = rt_config->hysteresis;
    }
    if (ptr->initialized_) {
        ptr->decision_filter_.SetConfig(
            ToFilterConfig(ptr->rt_config_, ptr->init_config_.sample_rate));
    }

    LOG_INFO("AudioEngine_Vad_SetParam OK, threshold=%.2f, min_speech=%.2fs, "
             "min_silence=%.2fs, hysteresis=%.2f",
             ptr->rt_config_.threshold, ptr->rt_config_.min_speech_duration,
             ptr->rt_config_.min_silence_duration, ptr->rt_config_.hysteresis);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Vad_ResetParam(VadHandle handle, const VadRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Vad_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Vad_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold > 1.0f) {
        LOG_WARN("AudioEngine_Vad_ResetParam: threshold=%.2f out of range (0.0, 1.0]",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);

    /* Full reset: restore factory defaults first, then apply new config */
    SetDefaultRtConfig(&ptr->rt_config_);

    ptr->rt_config_.threshold = rt_config->threshold;
    if (rt_config->min_speech_duration > 0.0f) {
        ptr->rt_config_.min_speech_duration = rt_config->min_speech_duration;
    }
    if (rt_config->min_silence_duration > 0.0f) {
        ptr->rt_config_.min_silence_duration = rt_config->min_silence_duration;
    }
    if (rt_config->hysteresis > 0.0f) {
        ptr->rt_config_.hysteresis = rt_config->hysteresis;
    }
    if (ptr->initialized_) {
        ptr->decision_filter_.SetConfig(
            ToFilterConfig(ptr->rt_config_, ptr->init_config_.sample_rate));
    }

    LOG_INFO("AudioEngine_Vad_ResetParam OK, threshold=%.2f, min_speech=%.2fs, "
             "min_silence=%.2fs, hysteresis=%.2f",
             ptr->rt_config_.threshold, ptr->rt_config_.min_speech_duration,
             ptr->rt_config_.min_silence_duration, ptr->rt_config_.hysteresis);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Vad_Process(
    VadHandle handle,
    const short* audio_in,
    int samples,
    int *vad_flag)
{
    LOG_DEBUG("AudioEngine_Vad_Process, handle=%p, samples=%d", (void*)handle, samples);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!audio_in) {
        LOG_WARN("AudioEngine_Vad_Process: audio_in is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!vad_flag) {
        LOG_WARN("AudioEngine_Vad_Process: vad_flag is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);

    if (!ptr->initialized_) {
        LOG_WARN("AudioEngine_Vad_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    const int expected_samples = ptr->init_config_.sample_rate / 100;
    if (samples != expected_samples) {
        LOG_WARN("AudioEngine_Vad_Process: samples=%d, expected %d (10ms constraint)",
                 samples, expected_samples);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    if (ptr->init_config_.vad_mode == kVadModeDsp) {
        ptr->dsp_vad_->ProcessChunk(audio_in, static_cast<size_t>(samples),
                                    ptr->init_config_.sample_rate);
        /* DSP emits a fresh probability for every 10 ms chunk. */
        ptr->decision_filter_.Update(ptr->dsp_vad_->last_voice_probability(),
                                     static_cast<int64_t>(samples));
    } else {
#ifdef AE_WITH_MNN
        if (!ptr->silero_vad_) {
            LOG_ERROR("AudioEngine_Vad_Process: silero instance is NULL");
            return AUDIO_ENGINE_ERR_PROCESS_FAILED;
        }
        /* Silero advances the filter itself, once per completed model window. */
        if (!ptr->silero_vad_->ProcessChunk(audio_in,
                                            static_cast<size_t>(samples),
                                            ptr->init_config_.sample_rate,
                                            &ptr->decision_filter_)) {
            LOG_ERROR("AudioEngine_Vad_Process: silero inference failed");
            return AUDIO_ENGINE_ERR_PROCESS_FAILED;
        }
#else
        LOG_ERROR("AudioEngine_Vad_Process: silero mode is not available "
                  "(build with -DWITH_MNN=ON)");
        return AUDIO_ENGINE_ERR_PROCESS_FAILED;
#endif
    }

    /* Shared decision smoothing: onset debounce, hysteresis and hangover. */
    *vad_flag = ptr->decision_filter_.is_speech() ? 1 : 0;

    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Vad_Reset(VadHandle handle)
{
    LOG_DEBUG("AudioEngine_Vad_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Vad_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineVad* ptr = static_cast<AudioEngineVad*>(handle);

    if (!ptr->initialized_) {
        LOG_WARN("AudioEngine_Vad_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    if (ptr->init_config_.vad_mode == kVadModeDsp) {
        /* Destroy and recreate the DSP VAD; keep init_config_ and rt_config_ */
        if (ptr->dsp_vad_) {
            delete ptr->dsp_vad_;
            ptr->dsp_vad_ = nullptr;
        }

        ptr->dsp_vad_ = new (std::nothrow) webrtc::VoiceActivityDetector();
        if (!ptr->dsp_vad_) {
            LOG_ERROR("AudioEngine_Vad_Reset: VoiceActivityDetector allocation failed");
            ptr->initialized_ = false;
            return AUDIO_ENGINE_ERR_PROCESS_FAILED;
        }
    } else {
#ifdef AE_WITH_MNN
        if (!ptr->silero_vad_) {
            LOG_ERROR("AudioEngine_Vad_Reset: silero instance is NULL");
            ptr->initialized_ = false;
            return AUDIO_ENGINE_ERR_PROCESS_FAILED;
        }
        /* Keep the loaded model; only clear stream state. */
        ptr->silero_vad_->Reset();
#else
        LOG_ERROR("AudioEngine_Vad_Reset: silero mode is not available "
                  "(build with -DWITH_MNN=ON)");
        ptr->initialized_ = false;
        return AUDIO_ENGINE_ERR_PROCESS_FAILED;
#endif
    }

    ptr->decision_filter_.Reset();

    /* initialized_ stays true, configs preserved */
    LOG_INFO("AudioEngine_Vad_Reset OK, mode=%d, threshold=%.2f",
             ptr->init_config_.vad_mode, ptr->rt_config_.threshold);
    return AUDIO_ENGINE_SUCCESS;
}
