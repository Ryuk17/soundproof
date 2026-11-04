/*
 * @Author: Ryuk
 * @Date: 2026-11-03
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-03 00:00:00
 * @Description: ASR interface implementation (transducer / paraformer)
 */

#include <cstring>
#include <new>
#include <string>

#include "audio_engine_asr.h"
#include "audio_engine_log.h"

#include "core/speech_recognition/online_recognizer.h"


namespace {

constexpr int kSampleRate = 16000;
constexpr int kFrameLen = 160;  /* 10 ms */

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
class AudioEngineAsr {
public:
    AudioEngineAsr()
        : initialized_(false)
        , rt_blank_penalty_(0.0f)
        , recognizer_(nullptr) {}

    ~AudioEngineAsr() {
        if (recognizer_) {
            delete recognizer_;
            recognizer_ = nullptr;
        }
    }

    bool initialized_;
    AsrInitConfig init_config_;
    float rt_blank_penalty_;
    soundproof::OnlineRecognizer* recognizer_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API AsrHandle AudioEngine_Asr_Create(void)
{
    LOG_DEBUG("AudioEngine_Asr_Create");

    AudioEngineAsr* ptr = new (std::nothrow) AudioEngineAsr();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Asr_Create: memory allocation failed");
    }
    return static_cast<AsrHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Asr_Destroy(AsrHandle handle)
{
    LOG_DEBUG("AudioEngine_Asr_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineAsr*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Asr_Init(AsrHandle handle,
                                          const AsrInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Asr_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Asr_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->sample_rate != kSampleRate) {
        LOG_WARN("AudioEngine_Asr_Init: only 16000 Hz is supported, got %d",
                 init_config->sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->frame_len != init_config->sample_rate / 100) {
        LOG_WARN("AudioEngine_Asr_Init: frame_len=%d must equal sample_rate/100=%d",
                 init_config->frame_len, kFrameLen);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    const std::string encoder_path = PathFrom(init_config->encoder_path);
    const std::string decoder_path = PathFrom(init_config->decoder_path);
    const std::string joiner_path = PathFrom(init_config->joiner_path);
    const std::string tokens_path = PathFrom(init_config->tokens_path);
    if (encoder_path.empty() || decoder_path.empty() || tokens_path.empty()) {
        LOG_WARN("AudioEngine_Asr_Init: encoder/decoder/tokens paths are required");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    if (ptr->recognizer_) {
        LOG_DEBUG("AudioEngine_Asr_Init: already initialized, deinit first");
        delete ptr->recognizer_;
        ptr->recognizer_ = nullptr;
    }
    ptr->initialized_ = false;

    ptr->init_config_ = *init_config;
    ptr->rt_blank_penalty_ = 0.0f;

    soundproof::AsrRecognizerConfig config;
    config.backend.model_type = PathFrom(init_config->model_type);
    config.backend.encoder_path = encoder_path;
    config.backend.decoder_path = decoder_path;
    config.backend.joiner_path = joiner_path;
    config.backend.num_threads =
        init_config->num_threads > 0 ? init_config->num_threads : 1;
    config.backend.blank_penalty = ptr->rt_blank_penalty_;
    config.tokens_path = tokens_path;
    config.enable_endpoint = init_config->enable_endpoint != 0;

    ptr->recognizer_ = new (std::nothrow) soundproof::OnlineRecognizer();
    if (!ptr->recognizer_ || !ptr->recognizer_->Init(config)) {
        LOG_ERROR("AudioEngine_Asr_Init: recognizer init failed");
        delete ptr->recognizer_;
        ptr->recognizer_ = nullptr;
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->initialized_ = true;
    LOG_INFO("AudioEngine_Asr_Init OK, sample_rate=%d, frame_len=%d, "
             "model_type='%s', endpoint=%d",
             init_config->sample_rate, init_config->frame_len,
             config.backend.model_type.c_str(), (int)config.enable_endpoint);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Asr_Deinit(AsrHandle handle)
{
    LOG_DEBUG("AudioEngine_Asr_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    if (ptr->recognizer_) {
        delete ptr->recognizer_;
        ptr->recognizer_ = nullptr;
    }
    ptr->initialized_ = false;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * SetParam / ResetParam
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Asr_SetParam(AsrHandle handle,
                                              const AsrRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Asr_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Asr_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->blank_penalty < 0.0f) {
        LOG_WARN("AudioEngine_Asr_SetParam: negative blank_penalty=%.2f",
                 rt_config->blank_penalty);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    ptr->rt_blank_penalty_ = rt_config->blank_penalty;
    if (ptr->initialized_ && ptr->recognizer_) {
        if (!ptr->recognizer_->SetBlankPenalty(ptr->rt_blank_penalty_)) {
            LOG_ERROR("AudioEngine_Asr_SetParam: backend does not support "
                      "blank_penalty");
            return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
        }
    }

    LOG_INFO("AudioEngine_Asr_SetParam OK, blank_penalty=%.2f",
             ptr->rt_blank_penalty_);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Asr_ResetParam(AsrHandle handle,
                                                const AsrRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Asr_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Asr_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->blank_penalty < 0.0f) {
        LOG_WARN("AudioEngine_Asr_ResetParam: negative blank_penalty=%.2f",
                 rt_config->blank_penalty);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    ptr->rt_blank_penalty_ = rt_config->blank_penalty;
    if (ptr->initialized_ && ptr->recognizer_) {
        if (!ptr->recognizer_->SetBlankPenalty(ptr->rt_blank_penalty_)) {
            LOG_ERROR("AudioEngine_Asr_ResetParam: backend does not support "
                      "blank_penalty");
            return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
        }
    }

    LOG_INFO("AudioEngine_Asr_ResetParam OK, blank_penalty=%.2f",
             ptr->rt_blank_penalty_);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Process
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Asr_Process(
    AsrHandle handle,
    const short* audio_in,
    int samples,
    char* text,
    int text_capacity,
    int* text_len,
    int* is_final)
{
    LOG_DEBUG("AudioEngine_Asr_Process, handle=%p, samples=%d",
              (void*)handle, samples);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!audio_in || !text || !text_len || !is_final || text_capacity <= 0) {
        LOG_WARN("AudioEngine_Asr_Process: NULL pointer or empty text buffer");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Asr_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    const int expected_samples = ptr->init_config_.sample_rate / 100;
    if (samples != expected_samples) {
        LOG_WARN("AudioEngine_Asr_Process: samples=%d, expected %d (10ms constraint)",
                 samples, expected_samples);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    soundproof::AsrRecognizerResult result;
    if (!ptr->recognizer_->Process(audio_in, static_cast<size_t>(samples),
                                   &result)) {
        LOG_ERROR("AudioEngine_Asr_Process: recognizer failed");
        return AUDIO_ENGINE_ERR_PROCESS_FAILED;
    }

    const int required = static_cast<int>(result.text.size()) + 1;
    if (required > text_capacity) {
        *text_len = required;
        LOG_WARN("AudioEngine_Asr_Process: text buffer too small (%d < %d)",
                 text_capacity, required);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    std::memcpy(text, result.text.c_str(), result.text.size());
    text[result.text.size()] = '\0';
    *text_len = required;
    *is_final = result.is_final ? 1 : 0;
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Asr_Reset(AsrHandle handle)
{
    LOG_DEBUG("AudioEngine_Asr_Reset, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Asr_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineAsr* ptr = static_cast<AudioEngineAsr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Asr_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    ptr->recognizer_->Reset();
    LOG_INFO("AudioEngine_Asr_Reset OK");
    return AUDIO_ENGINE_SUCCESS;
}
