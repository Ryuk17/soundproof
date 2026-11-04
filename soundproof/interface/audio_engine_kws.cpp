/*
 * @Author: Ryuk
 * @Date: 2026-08-09 18:44:06
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-03 00:00:00
 * @Description: KWS interface implementation (zipformer2 transducer)
 */

#include <new>
#include <string>

#include "audio_engine_kws.h"
#include "audio_engine_log.h"

#include "core/keywords_spotting/keyword_spotter.h"


namespace {

constexpr int kSampleRate = 16000;
constexpr int kFrameLen = 160;  /* 10 ms */
constexpr float kDefaultThreshold = 0.25f;

/* Fixed-size path fields may not be NUL-terminated by the caller. */
template <size_t N>
std::string PathFrom(const char (&field)[N])
{
    size_t len = 0;
    while (len < N && field[len] != '\0') {
        ++len;
    }
    return std::string(field, len);
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineKws {
public:
    AudioEngineKws()
        : initialized_(false)
        , rt_threshold_(kDefaultThreshold)
        , spotter_(nullptr) {}

    ~AudioEngineKws() {
        if (spotter_) {
            delete spotter_;
            spotter_ = nullptr;
        }
    }

    bool initialized_;
    KwsInitConfig init_config_;
    float rt_threshold_;
    soundproof::KeywordSpotter* spotter_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API KwsHandle AudioEngine_Kws_Create(void)
{
    LOG_DEBUG("AudioEngine_Kws_Create");

    AudioEngineKws* ptr = new (std::nothrow) AudioEngineKws();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Kws_Create: memory allocation failed");
    }
    return static_cast<KwsHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Kws_Destroy(KwsHandle handle)
{
    LOG_DEBUG("AudioEngine_Kws_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    delete ptr;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Kws_Init(KwsHandle handle,
                                          const KwsInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Kws_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Kws_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->sample_rate != kSampleRate) {
        LOG_WARN("AudioEngine_Kws_Init: only 16000 Hz is supported, got %d",
                 init_config->sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->frame_len != init_config->sample_rate / 100) {
        LOG_WARN("AudioEngine_Kws_Init: frame_len=%d must equal sample_rate/100=%d",
                 init_config->frame_len, kFrameLen);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string encoder_path = PathFrom(init_config->encoder_path);
    const std::string decoder_path = PathFrom(init_config->decoder_path);
    const std::string joiner_path = PathFrom(init_config->joiner_path);
    const std::string tokens_path = PathFrom(init_config->tokens_path);
    const std::string keywords_path = PathFrom(init_config->keywords_file_path);
    if (encoder_path.empty() || decoder_path.empty() || joiner_path.empty() ||
        tokens_path.empty() || keywords_path.empty()) {
        LOG_WARN("AudioEngine_Kws_Init: encoder/decoder/joiner/tokens/keywords "
                 "paths are all required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    if (ptr->spotter_) {
        LOG_DEBUG("AudioEngine_Kws_Init: already initialized, deinit first");
        delete ptr->spotter_;
        ptr->spotter_ = nullptr;
    }
    ptr->initialized_ = false;

    ptr->init_config_ = *init_config;
    ptr->rt_threshold_ = kDefaultThreshold;

    soundproof::KeywordSpotterConfig config;
    config.encoder_path = encoder_path;
    config.decoder_path = decoder_path;
    config.joiner_path = joiner_path;
    config.tokens_path = tokens_path;
    config.keywords_file_path = keywords_path;
    config.num_threads = init_config->num_threads > 0 ? init_config->num_threads : 1;
    config.keywords_threshold = ptr->rt_threshold_;

    ptr->spotter_ = new (std::nothrow) soundproof::KeywordSpotter();
    if (!ptr->spotter_ || !ptr->spotter_->Init(config)) {
        LOG_ERROR("AudioEngine_Kws_Init: keyword spotter init failed");
        delete ptr->spotter_;
        ptr->spotter_ = nullptr;
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->initialized_ = true;
    LOG_INFO("AudioEngine_Kws_Init OK, sample_rate=%d, frame_len=%d, "
             "keywords=%d, threshold=%.2f",
             init_config->sample_rate, init_config->frame_len,
             (int)ptr->spotter_->num_keywords(), ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Kws_Deinit(KwsHandle handle)
{
    LOG_DEBUG("AudioEngine_Kws_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    if (ptr->spotter_) {
        delete ptr->spotter_;
        ptr->spotter_ = nullptr;
    }
    ptr->initialized_ = false;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * SetParam / ResetParam
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Kws_SetParam(KwsHandle handle,
                                              const KwsRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Kws_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Kws_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold >= 1.0f) {
        LOG_WARN("AudioEngine_Kws_SetParam: threshold=%.2f out of range (0.0, 1.0)",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    ptr->rt_threshold_ = rt_config->threshold;
    if (ptr->initialized_ && ptr->spotter_) {
        if (!ptr->spotter_->SetThreshold(ptr->rt_threshold_)) {
            LOG_ERROR("AudioEngine_Kws_SetParam: applying threshold failed");
            return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
        }
    }

    LOG_INFO("AudioEngine_Kws_SetParam OK, threshold=%.2f", ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Kws_ResetParam(KwsHandle handle,
                                                const KwsRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Kws_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Kws_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold >= 1.0f) {
        LOG_WARN("AudioEngine_Kws_ResetParam: threshold=%.2f out of range (0.0, 1.0)",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    ptr->rt_threshold_ = kDefaultThreshold;
    ptr->rt_threshold_ = rt_config->threshold;
    if (ptr->initialized_ && ptr->spotter_) {
        if (!ptr->spotter_->SetThreshold(ptr->rt_threshold_)) {
            LOG_ERROR("AudioEngine_Kws_ResetParam: applying threshold failed");
            return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
        }
    }

    LOG_INFO("AudioEngine_Kws_ResetParam OK, threshold=%.2f", ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Kws_Process(
    KwsHandle handle,
    const KwsInput_t* input,
    KwsResult_t* result)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Kws_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!input || !result) {
        LOG_WARN("AudioEngine_Kws_Process: input/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    const short* audio_in = input->audio_in;
    const int samples = input->samples;
    int* kws_index = &result->kws_index;
    float* kws_score = &result->kws_score;
    LOG_DEBUG("AudioEngine_Kws_Process, handle=%p, samples=%d", (void*)handle, samples);
    if (!audio_in) {
        LOG_WARN("AudioEngine_Kws_Process: audio_in is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    if (!ptr->initialized_ || !ptr->spotter_) {
        LOG_WARN("AudioEngine_Kws_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    const int expected_samples = ptr->init_config_.sample_rate / 100;
    if (samples != expected_samples) {
        LOG_WARN("AudioEngine_Kws_Process: samples=%d, expected %d (10ms constraint)",
                 samples, expected_samples);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    int index = -1;
    float score = 0.0f;
    if (!ptr->spotter_->Process(audio_in, static_cast<size_t>(samples), &index,
                                &score)) {
        LOG_ERROR("AudioEngine_Kws_Process: inference failed");
        return AUDIO_ENGINE_ERR_PROCESS_FAILED;
    }
    *kws_index = index;
    *kws_score = score;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Kws_Reset(KwsHandle handle)
{
    LOG_DEBUG("AudioEngine_Kws_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Kws_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineKws* ptr = static_cast<AudioEngineKws*>(handle);
    if (!ptr->initialized_ || !ptr->spotter_) {
        LOG_WARN("AudioEngine_Kws_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    ptr->spotter_->Reset();
    LOG_INFO("AudioEngine_Kws_Reset OK");
    return AUDIO_ENGINE_SUCCESS;
}
