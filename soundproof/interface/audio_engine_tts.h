/*
 * @Author: Ryuk
 * @Date: 2026-11-05
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-05 00:00:00
 * @Description: TTS (Text To Speech) unified C interface
 */
#ifndef AUDIO_ENGINE_TTS_H
#define AUDIO_ENGINE_TTS_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* TtsHandle;

typedef struct {
    int num_threads;          /* MNN inference threads, default 1              */
    int output_sample_rate;   /* 0 = model default (Kokoro: 24000)             */
    char model_type[32];      /* "", "kokoro", "vits"                          */
    char model_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
    char voices_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];  /* Kokoro only          */
    char tokens_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];  /* tokens.txt / tokenizer.txt */
    char lexicon_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN]; /* reserved for G2P     */
    char data_dir[AUDIO_ENGINE_MAX_FILE_PATH_LEN];     /* reserved for espeak  */
    char dict_dir[AUDIO_ENGINE_MAX_FILE_PATH_LEN];     /* reserved for jieba   */
} TtsInitConfig;

typedef struct {
    float speed;              /* 1.0 = normal speed, > 0                      */
    int speaker_id;           /* default 0                                    */
} TtsRtConfig;

typedef struct {
    const char* text;         /* UTF-8 text or phoneme/symbol string           */
    const int32_t* token_ids; /* used when input_type == 2                     */
    int num_tokens;
    int input_type;           /* 0 = text, 1 = phoneme string, 2 = token ids   */
} TtsInput_t;

typedef struct {
    /* Engine-owned PCM; valid until the next Process/Reset/Deinit call. */
    const short* audio_out;
    int out_samples;
    int sample_rate;
} TtsResult_t;


AUDIO_ENGINE_API TtsHandle AudioEngine_Tts_Create(void);

AUDIO_ENGINE_API int AudioEngine_Tts_Destroy(TtsHandle handle);

AUDIO_ENGINE_API int AudioEngine_Tts_Init(TtsHandle handle, const TtsInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Tts_SetParam(TtsHandle handle, const TtsRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Tts_ResetParam(TtsHandle handle, const TtsRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Tts_Process(
    TtsHandle handle,
    const TtsInput_t* input,
    TtsResult_t* result);

AUDIO_ENGINE_API int AudioEngine_Tts_GetSampleRate(TtsHandle handle, int* sample_rate);

AUDIO_ENGINE_API int AudioEngine_Tts_GetNumSpeakers(TtsHandle handle, int* num_speakers);

AUDIO_ENGINE_API int AudioEngine_Tts_Deinit(TtsHandle handle);

AUDIO_ENGINE_API int AudioEngine_Tts_Reset(TtsHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_TTS_H */
