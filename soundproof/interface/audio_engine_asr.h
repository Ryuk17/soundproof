/*
 * @Author: Ryuk
 * @Date: 2026-11-03
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-03 00:00:00
 * @Description: ASR (Automatic Speech Recognition) unified C interface
 */
#ifndef AUDIO_ENGINE_ASR_H
#define AUDIO_ENGINE_ASR_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* AsrHandle;

typedef struct {
    int sample_rate;      /* only 16000 is supported                          */
    int frame_len;        /* samples per Process call, must == sample_rate/100
                             (10ms); features are buffered internally           */
    int num_threads;      /* MNN inference threads, default 1                  */
    int enable_endpoint;  /* 0/1, default 1: split utterances on silence       */
    char model_type[32];  /* "", "zipformer", "zipformer2", "paraformer", ...
                             empty = auto-detect from paths / model metadata   */
    char encoder_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
    char decoder_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN]; /* transducer/paraformer */
    char joiner_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];  /* transducer only      */
    char tokens_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
} AsrInitConfig;


typedef struct {
    float blank_penalty;  /* transducer only, default 0; >= 0                 */
} AsrRtConfig;


AUDIO_ENGINE_API AsrHandle AudioEngine_Asr_Create(void);

AUDIO_ENGINE_API int AudioEngine_Asr_Destroy(AsrHandle handle);

AUDIO_ENGINE_API int AudioEngine_Asr_Init(AsrHandle handle, const AsrInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Asr_SetParam(AsrHandle handle, const AsrRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Asr_ResetParam(AsrHandle handle, const AsrRtConfig* rt_config);

/* Outputs the current partial text (UTF-8, NUL terminated).
   is_final: 1 if this call reached an endpoint; text is then the final
   segment text and the recognizer has been reset for the next segment.
   If text_capacity is too small, *text_len is set to the required size
   (including the NUL) and AUDIO_ENGINE_ERR_INVALID_PARAM is returned. */
AUDIO_ENGINE_API int AudioEngine_Asr_Process(
    AsrHandle handle,
    const short* audio_in,
    int samples,
    char* text,
    int text_capacity,
    int* text_len,
    int* is_final
);

AUDIO_ENGINE_API int AudioEngine_Asr_Deinit(AsrHandle handle);

AUDIO_ENGINE_API int AudioEngine_Asr_Reset(AsrHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_ASR_H */
