/*
 * @Author: Ryuk
 * @Date: 2026-11-04
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-04 00:00:00
 * @Description: SR interface implementation
 */

#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "audio_engine_log.h"
#include "audio_engine_sr.h"

#include "core/speaker_recognition/speaker_recognizer.h"
#include "utils/dr_wav.h"


namespace {

constexpr int kSampleRate = 16000;
constexpr int kFrameLen = 160;  /* 10 ms */
constexpr float kDefaultThreshold = 0.5f;

template <size_t N>
std::string PathFrom(const char (&field)[N])
{
    size_t len = 0;
    while (len < N && field[len] != '\0') {
        ++len;
    }
    return std::string(field, len);
}

const char* SrStatusReason(soundproof::SrStatus status)
{
    switch (status) {
        case soundproof::SrStatus::kOk: return "ok";
        case soundproof::SrStatus::kNotInitialized: return "not initialized";
        case soundproof::SrStatus::kModelLoadFailed: return "model load failed";
        case soundproof::SrStatus::kUnsupportedModel: return "unsupported model";
        case soundproof::SrStatus::kAudioTooShort: return "audio too short";
        case soundproof::SrStatus::kFeatureFailed: return "feature extraction failed";
        case soundproof::SrStatus::kInferenceFailed: return "model inference failed";
        case soundproof::SrStatus::kSpeakerExists: return "speaker already exists";
        case soundproof::SrStatus::kSpeakerNotFound: return "speaker not found";
        case soundproof::SrStatus::kEmbeddingDimMismatch:
            return "embedding dimension mismatch";
    }
    return "unknown error";
}

int MapStatus(soundproof::SrStatus status, const char* api)
{
    switch (status) {
        case soundproof::SrStatus::kOk:
            return AUDIO_ENGINE_SUCCESS;
        case soundproof::SrStatus::kNotInitialized:
            return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
        case soundproof::SrStatus::kModelLoadFailed:
            return AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED;
        case soundproof::SrStatus::kUnsupportedModel:
            return AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL;
        case soundproof::SrStatus::kAudioTooShort:
            return AUDIO_ENGINE_ERR_AUDIO_TOO_SHORT;
        case soundproof::SrStatus::kFeatureFailed:
            return AUDIO_ENGINE_ERR_FEATURE_FAILED;
        case soundproof::SrStatus::kInferenceFailed:
            return AUDIO_ENGINE_ERR_INFERENCE_FAILED;
        case soundproof::SrStatus::kSpeakerExists:
            return AUDIO_ENGINE_ERR_SR_SPEAKER_EXISTS;
        case soundproof::SrStatus::kSpeakerNotFound:
            return AUDIO_ENGINE_ERR_SR_SPEAKER_NOT_FOUND;
        case soundproof::SrStatus::kEmbeddingDimMismatch:
            return AUDIO_ENGINE_ERR_EMBEDDING_DIM_MISMATCH;
    }
    LOG_ERROR("%s: unknown SR status", api);
    return AUDIO_ENGINE_ERR_PROCESS_FAILED;
}

bool ReadWav16kMono(const std::string& path, std::vector<int16_t>* samples)
{
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        return false;
    }
    bool ok = wav.sampleRate == 16000 && wav.channels == 1 &&
              wav.totalPCMFrameCount > 0;
    if (ok) {
        const drwav_uint64 total = wav.totalPCMFrameCount;
        samples->resize(static_cast<size_t>(total));
        ok = drwav_read_pcm_frames_s16(&wav, total, samples->data()) == total;
    }
    drwav_uninit(&wav);
    return ok;
}

int FillResult(const soundproof::SrRecognizerResult& core, SrResult* out)
{
    if (!out) {
        LOG_WARN("SR: result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (!out->speaker_name || out->name_capacity <= 0) {
        LOG_WARN("SR: speaker_name buffer is NULL or empty");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    const int required = static_cast<int>(core.speaker_name.size()) + 1;
    out->name_len = required;
    out->has_result = core.has_result ? 1 : 0;
    out->score = core.score;
    if (required > out->name_capacity) {
        out->speaker_name[0] = '\0';
        LOG_WARN("SR: name buffer too small (%d < %d)",
                 out->name_capacity, required);
        return AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL;
    }
    std::memcpy(out->speaker_name, core.speaker_name.c_str(),
                core.speaker_name.size());
    out->speaker_name[core.speaker_name.size()] = '\0';
    return AUDIO_ENGINE_SUCCESS;
}

}  // namespace


/* ============================================================================
 * Internal data container
 * ==========================================================================*/
class AudioEngineSr {
public:
    AudioEngineSr()
        : initialized_(false)
        , rt_threshold_(kDefaultThreshold)
        , recognizer_(nullptr) {}

    ~AudioEngineSr() {
        if (recognizer_) {
            delete recognizer_;
            recognizer_ = nullptr;
        }
    }

    bool initialized_;
    SrInitConfig init_config_;
    float rt_threshold_;
    soundproof::SpeakerRecognizer* recognizer_;
};


/* ============================================================================
 * Create / Destroy
 * ==========================================================================*/
AUDIO_ENGINE_API SrHandle AudioEngine_Sr_Create(void)
{
    LOG_DEBUG("AudioEngine_Sr_Create");

    AudioEngineSr* ptr = new (std::nothrow) AudioEngineSr();
    if (!ptr) {
        LOG_ERROR("AudioEngine_Sr_Create: memory allocation failed");
    }
    return static_cast<SrHandle>(ptr);
}

AUDIO_ENGINE_API int AudioEngine_Sr_Destroy(SrHandle handle)
{
    LOG_DEBUG("AudioEngine_Sr_Destroy, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Destroy: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    delete static_cast<AudioEngineSr*>(handle);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Init / Deinit
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Sr_Init(SrHandle handle,
                                         const SrInitConfig* init_config)
{
    LOG_DEBUG("AudioEngine_Sr_Init, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Init: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!init_config) {
        LOG_WARN("AudioEngine_Sr_Init: init_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (init_config->sample_rate != kSampleRate) {
        LOG_WARN("AudioEngine_Sr_Init: only 16000 Hz is supported, got %d",
                 init_config->sample_rate);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    if (init_config->frame_len != init_config->sample_rate / 100) {
        LOG_WARN("AudioEngine_Sr_Init: frame_len=%d must equal sample_rate/100=%d",
                 init_config->frame_len, kFrameLen);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    const std::string model_path = PathFrom(init_config->model_path);
    if (model_path.empty()) {
        LOG_WARN("AudioEngine_Sr_Init: model_path is empty");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (ptr->recognizer_) {
        LOG_DEBUG("AudioEngine_Sr_Init: already initialized, deinit first");
        delete ptr->recognizer_;
        ptr->recognizer_ = nullptr;
    }
    ptr->initialized_ = false;
    ptr->init_config_ = *init_config;
    ptr->rt_threshold_ = kDefaultThreshold;

    soundproof::SrRecognizerConfig config;
    config.backend.model_type = PathFrom(init_config->model_type);
    config.backend.model_path = model_path;
    config.backend.num_threads =
        init_config->num_threads > 0 ? init_config->num_threads : 1;
    config.threshold = ptr->rt_threshold_;

    ptr->recognizer_ = new (std::nothrow) soundproof::SpeakerRecognizer();
    if (!ptr->recognizer_ || !ptr->recognizer_->Init(config)) {
        LOG_ERROR("AudioEngine_Sr_Init: recognizer init failed");
        delete ptr->recognizer_;
        ptr->recognizer_ = nullptr;
        return AUDIO_ENGINE_ERR_INIT_FAILED;
    }

    ptr->initialized_ = true;
    LOG_INFO("AudioEngine_Sr_Init OK, sample_rate=%d, frame_len=%d, "
             "model_type='%s', threshold=%.2f",
             init_config->sample_rate, init_config->frame_len,
             config.backend.model_type.c_str(), ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Sr_Deinit(SrHandle handle)
{
    LOG_DEBUG("AudioEngine_Sr_Deinit, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Deinit: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
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
AUDIO_ENGINE_API int AudioEngine_Sr_SetParam(SrHandle handle,
                                             const SrRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Sr_SetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_SetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Sr_SetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold > 1.0f) {
        LOG_WARN("AudioEngine_Sr_SetParam: threshold=%.2f out of range (0, 1]",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    ptr->rt_threshold_ = rt_config->threshold;
    if (ptr->initialized_ && ptr->recognizer_ &&
        !ptr->recognizer_->SetThreshold(ptr->rt_threshold_)) {
        LOG_ERROR("AudioEngine_Sr_SetParam: applying threshold failed");
        return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
    }

    LOG_INFO("AudioEngine_Sr_SetParam OK, threshold=%.2f", ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Sr_ResetParam(SrHandle handle,
                                               const SrRtConfig* rt_config)
{
    LOG_DEBUG("AudioEngine_Sr_ResetParam, handle=%p", (void*)handle);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_ResetParam: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!rt_config) {
        LOG_WARN("AudioEngine_Sr_ResetParam: rt_config is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    if (rt_config->threshold <= 0.0f || rt_config->threshold > 1.0f) {
        LOG_WARN("AudioEngine_Sr_ResetParam: threshold=%.2f out of range (0, 1]",
                 rt_config->threshold);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    ptr->rt_threshold_ = kDefaultThreshold;
    ptr->rt_threshold_ = rt_config->threshold;
    if (ptr->initialized_ && ptr->recognizer_ &&
        !ptr->recognizer_->SetThreshold(ptr->rt_threshold_)) {
        LOG_ERROR("AudioEngine_Sr_ResetParam: applying threshold failed");
        return AUDIO_ENGINE_ERR_SET_PARAM_FAILED;
    }

    LOG_INFO("AudioEngine_Sr_ResetParam OK, threshold=%.2f", ptr->rt_threshold_);
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Online recognition
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Sr_Process(SrHandle handle,
                                            const short* audio_in, int samples,
                                            int trigger, SrResult* result)
{
    LOG_DEBUG("AudioEngine_Sr_Process, handle=%p, samples=%d, trigger=%d",
              (void*)handle, samples, trigger);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Process: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!audio_in || !result) {
        LOG_WARN("AudioEngine_Sr_Process: audio_in/result is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Process: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    if (samples != ptr->init_config_.sample_rate / 100) {
        LOG_WARN("AudioEngine_Sr_Process: samples=%d, expected %d (10ms)",
                 samples, ptr->init_config_.sample_rate / 100);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    soundproof::SrRecognizerResult core_result;
    const soundproof::SrStatus status = ptr->recognizer_->Process(
        audio_in, static_cast<size_t>(samples), trigger != 0, &core_result);
    const int ret = MapStatus(status, "AudioEngine_Sr_Process");
    if (ret != AUDIO_ENGINE_SUCCESS) {
        return ret;
    }
    return FillResult(core_result, result);
}

AUDIO_ENGINE_API int AudioEngine_Sr_Identify(SrHandle handle,
                                             const short* audio_in,
                                             int samples, SrResult* result)
{
    LOG_DEBUG("AudioEngine_Sr_Identify, handle=%p, samples=%d",
              (void*)handle, samples);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Identify: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!audio_in || !result || samples <= 0) {
        LOG_WARN("AudioEngine_Sr_Identify: invalid audio/result");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Identify: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    soundproof::SrRecognizerResult core_result;
    const soundproof::SrStatus status = ptr->recognizer_->IdentifySegment(
        audio_in, static_cast<size_t>(samples), &core_result);
    const int ret = MapStatus(status, "AudioEngine_Sr_Identify");
    if (ret != AUDIO_ENGINE_SUCCESS) {
        return ret;
    }
    return FillResult(core_result, result);
}


/* ============================================================================
 * Offline enrollment
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Sr_Register(SrHandle handle,
                                             const char* speaker_name,
                                             const short* audio_in,
                                             int samples)
{
    LOG_DEBUG("AudioEngine_Sr_Register, handle=%p, samples=%d",
              (void*)handle, samples);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Register: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || speaker_name[0] == '\0' || !audio_in || samples <= 0) {
        LOG_WARN("AudioEngine_Sr_Register: invalid name/audio");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Register: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    const soundproof::SrStatus status = ptr->recognizer_->RegisterPcm(
        speaker_name, audio_in, static_cast<size_t>(samples));
    return MapStatus(status, "AudioEngine_Sr_Register");
}

AUDIO_ENGINE_API int AudioEngine_Sr_RegisterFile(SrHandle handle,
                                                 const char* speaker_name,
                                                 const char* wav_path)
{
    LOG_DEBUG("AudioEngine_Sr_RegisterFile, wav=%s", wav_path ? wav_path : "(null)");

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_RegisterFile: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || speaker_name[0] == '\0' || !wav_path) {
        LOG_WARN("AudioEngine_Sr_RegisterFile: invalid name/path");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    std::vector<int16_t> samples;
    if (!ReadWav16kMono(wav_path, &samples)) {
        LOG_ERROR("AudioEngine_Sr_RegisterFile: '%s' is not a readable "
                  "16 kHz mono WAV", wav_path);
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    return AudioEngine_Sr_Register(handle, speaker_name, samples.data(),
                                   static_cast<int>(samples.size()));
}

AUDIO_ENGINE_API int AudioEngine_Sr_RegisterFiles(
    SrHandle handle, const char* speaker_name, const char* const* wav_paths,
    int num_files)
{
    LOG_DEBUG("AudioEngine_Sr_RegisterFiles, files=%d", num_files);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_RegisterFiles: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || speaker_name[0] == '\0' || !wav_paths ||
        num_files <= 0) {
        LOG_WARN("AudioEngine_Sr_RegisterFiles: invalid name/paths");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_RegisterFiles: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(static_cast<size_t>(num_files));
    for (int i = 0; i < num_files; ++i) {
        const char* path = wav_paths[i];
        if (!path) {
            LOG_WARN("AudioEngine_Sr_RegisterFiles: NULL path at index %d", i);
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }
        std::vector<int16_t> samples;
        if (!ReadWav16kMono(path, &samples)) {
            LOG_ERROR("AudioEngine_Sr_RegisterFiles: '%s' is not a readable "
                      "16 kHz mono WAV", path);
            return AUDIO_ENGINE_ERR_INVALID_PARAM;
        }
        std::vector<float> embedding;
        const soundproof::SrStatus status = ptr->recognizer_->Extract(
            samples.data(), samples.size(), &embedding);
        if (status != soundproof::SrStatus::kOk) {
            return MapStatus(status, "AudioEngine_Sr_RegisterFiles");
        }
        embeddings.push_back(std::move(embedding));
    }

    const soundproof::SrStatus status =
        ptr->recognizer_->AddEmbeddings(speaker_name, embeddings);
    return MapStatus(status, "AudioEngine_Sr_RegisterFiles");
}

AUDIO_ENGINE_API int AudioEngine_Sr_Extract(SrHandle handle,
                                            const short* audio_in, int samples,
                                            float* embedding,
                                            int embedding_capacity,
                                            int* embedding_dim)
{
    LOG_DEBUG("AudioEngine_Sr_Extract, handle=%p, samples=%d",
              (void*)handle, samples);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Extract: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!audio_in || !embedding || !embedding_dim ||
        embedding_capacity <= 0 || samples <= 0) {
        LOG_WARN("AudioEngine_Sr_Extract: invalid audio/buffer");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Extract: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    std::vector<float> result;
    const soundproof::SrStatus status = ptr->recognizer_->Extract(
        audio_in, static_cast<size_t>(samples), &result);
    const int ret = MapStatus(status, "AudioEngine_Sr_Extract");
    if (ret != AUDIO_ENGINE_SUCCESS) {
        return ret;
    }
    const int dim = static_cast<int>(result.size());
    *embedding_dim = dim;
    if (embedding_capacity < dim) {
        LOG_WARN("AudioEngine_Sr_Extract: embedding buffer too small (%d < %d)",
                 embedding_capacity, dim);
        return AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL;
    }
    std::memcpy(embedding, result.data(), result.size() * sizeof(float));
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Sr_AddEmbedding(SrHandle handle,
                                                 const char* speaker_name,
                                                 const float* embedding,
                                                 int dim)
{
    LOG_DEBUG("AudioEngine_Sr_AddEmbedding, handle=%p, dim=%d",
              (void*)handle, dim);

    if (!handle) {
        LOG_WARN("AudioEngine_Sr_AddEmbedding: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || speaker_name[0] == '\0' || !embedding || dim <= 0) {
        LOG_WARN("AudioEngine_Sr_AddEmbedding: invalid name/embedding");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }

    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_AddEmbedding: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }

    const std::vector<float> vector(embedding, embedding + dim);
    const soundproof::SrStatus status =
        ptr->recognizer_->AddEmbedding(speaker_name, vector);
    return MapStatus(status, "AudioEngine_Sr_AddEmbedding");
}


/* ============================================================================
 * Registry management
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Sr_Remove(SrHandle handle,
                                           const char* speaker_name)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Remove: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || speaker_name[0] == '\0') {
        LOG_WARN("AudioEngine_Sr_Remove: invalid name");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Remove: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    return MapStatus(ptr->recognizer_->Remove(speaker_name),
                     "AudioEngine_Sr_Remove");
}

AUDIO_ENGINE_API int AudioEngine_Sr_Clear(SrHandle handle)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Clear: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Clear: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->recognizer_->Clear();
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Sr_GetNumSpeakers(SrHandle handle, int* count)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Sr_GetNumSpeakers: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!count) {
        LOG_WARN("AudioEngine_Sr_GetNumSpeakers: count is NULL");
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_GetNumSpeakers: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    *count = ptr->recognizer_->NumSpeakers();
    return AUDIO_ENGINE_SUCCESS;
}

AUDIO_ENGINE_API int AudioEngine_Sr_GetSpeakerName(
    SrHandle handle, int index, char* speaker_name, int name_capacity,
    int* name_len)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Sr_GetSpeakerName: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    if (!speaker_name || !name_len || name_capacity <= 0 || index < 0) {
        LOG_WARN("AudioEngine_Sr_GetSpeakerName: invalid argument");
        return AUDIO_ENGINE_ERR_INVALID_PARAM;
    }
    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_GetSpeakerName: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    const std::vector<std::string> names = ptr->recognizer_->GetAllSpeakers();
    if (index >= static_cast<int>(names.size())) {
        LOG_WARN("AudioEngine_Sr_GetSpeakerName: index %d out of range (%d)",
                 index, (int)names.size());
        return AUDIO_ENGINE_ERR_SR_SPEAKER_NOT_FOUND;
    }
    const std::string& name = names[static_cast<size_t>(index)];
    const int required = static_cast<int>(name.size()) + 1;
    *name_len = required;
    if (required > name_capacity) {
        speaker_name[0] = '\0';
        LOG_WARN("AudioEngine_Sr_GetSpeakerName: buffer too small (%d < %d)",
                 name_capacity, required);
        return AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL;
    }
    std::memcpy(speaker_name, name.c_str(), name.size());
    speaker_name[name.size()] = '\0';
    return AUDIO_ENGINE_SUCCESS;
}


/* ============================================================================
 * Reset
 * ==========================================================================*/
AUDIO_ENGINE_API int AudioEngine_Sr_Reset(SrHandle handle)
{
    if (!handle) {
        LOG_WARN("AudioEngine_Sr_Reset: invalid handle (NULL)");
        return AUDIO_ENGINE_ERR_INVALID_HANDLE;
    }
    AudioEngineSr* ptr = static_cast<AudioEngineSr*>(handle);
    if (!ptr->initialized_ || !ptr->recognizer_) {
        LOG_WARN("AudioEngine_Sr_Reset: not initialized");
        return AUDIO_ENGINE_ERR_NOT_INITIALIZED;
    }
    ptr->recognizer_->ResetStream();
    return AUDIO_ENGINE_SUCCESS;
}
