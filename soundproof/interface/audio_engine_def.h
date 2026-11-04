/*
 * @Author: Ryuk
 * @Date: 2026-08-09 19:02:18
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-02 23:34:58
 * @Description: Public API definitions — symbol visibility macro and status codes
 */
#ifndef AUDIO_ENGINE_DEF_H
#define AUDIO_ENGINE_DEF_H

#define AUDIO_ENGINE_API_VERSION "1.0.0"

#define AUDIO_ENGINE_MAX_FILE_PATH_LEN 256

/* ============================================================================
 * Symbol visibility: AUDIO_ENGINE_API
 *   - Static link:   empty (no attributes)
 *   - Build lib:     __declspec(dllexport) / visibility("default")
 *   - Consume DLL:   __declspec(dllimport)
 * ==========================================================================*/
#if defined(AUDIO_ENGINE_STATIC)
    #define AUDIO_ENGINE_API
#elif defined(_WIN32)
    #if defined(AUDIO_ENGINE_EXPORTS)
        #define AUDIO_ENGINE_API __declspec(dllexport)
    #else
        #define AUDIO_ENGINE_API __declspec(dllimport)
    #endif
#else
    #define AUDIO_ENGINE_API __attribute__((visibility("default")))
#endif



/* ============================================================================
 * Unified status codes returned by all libAE_xxx API functions
 * ==========================================================================*/
typedef enum {
    /* Common error codes: 0x00 - 0xFF, shared by all algorithms. */
    AUDIO_ENGINE_SUCCESS = 0x00,
    AUDIO_ENGINE_ERR_INIT_FAILED = 0x01,
    AUDIO_ENGINE_ERR_SET_PARAM_FAILED = 0x02,
    AUDIO_ENGINE_ERR_INVALID_HANDLE = 0x03,
    AUDIO_ENGINE_ERR_NULL_POINTER = 0x04,
    AUDIO_ENGINE_ERR_NOT_INITIALIZED = 0x05,
    AUDIO_ENGINE_ERR_PROCESS_FAILED = 0x06,
    AUDIO_ENGINE_ERR_INVALID_PARAM = 0x07,

    AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED = 0x08,
    AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL = 0x09,
    AUDIO_ENGINE_ERR_AUDIO_TOO_SHORT = 0x0A,
    AUDIO_ENGINE_ERR_FEATURE_FAILED = 0x0B,
    AUDIO_ENGINE_ERR_INFERENCE_FAILED = 0x0C,
    AUDIO_ENGINE_ERR_EMBEDDING_DIM_MISMATCH = 0x0D,

    /* Speaker recognition (SR) specific: 0x100 - 0x1FF. */
    AUDIO_ENGINE_ERR_SR_SPEAKER_EXISTS = 0x100,
    AUDIO_ENGINE_ERR_SR_SPEAKER_NOT_FOUND = 0x101,
    AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL = 0x102,
} AudioEngineStatus;

#endif /* AUDIO_ENGINE_DEF_H */
