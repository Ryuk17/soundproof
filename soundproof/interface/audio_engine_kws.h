/*
 * @Author: Ryuk
 * @Date: 2026-08-09 18:44:06
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-03 00:00:00
 * @Description: KWS (Keyword Spotting) unified C interface
 */
#ifndef AUDIO_ENGINE_KWS_H
#define AUDIO_ENGINE_KWS_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* KwsHandle;

typedef struct {
    int sample_rate;     /* only 16000 is supported                          */
    int frame_len;       /* samples per Process call, must == sample_rate/100
                            (10ms); features are buffered internally          */
    int num_threads;     /* MNN inference threads, default 1                 */
    char encoder_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];  /* zipformer2 encoder */
    char decoder_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];  /* zipformer2 decoder */
    char joiner_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];   /* zipformer2 joiner  */
    char tokens_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];   /* tokens.txt         */
    char keywords_file_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN]; /* keywords.txt  */
} KwsInitConfig;


typedef struct {
    float threshold;     /* keyword trigger threshold, default 0.25,
                            range (0.0, 1.0)                                  */
} KwsRtConfig;


AUDIO_ENGINE_API KwsHandle AudioEngine_Kws_Create(void);

AUDIO_ENGINE_API int AudioEngine_Kws_Destroy(KwsHandle handle);

AUDIO_ENGINE_API int AudioEngine_Kws_Init(KwsHandle handle, const KwsInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Kws_SetParam(KwsHandle handle, const KwsRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Kws_ResetParam(KwsHandle handle, const KwsRtConfig* rt_config);

/* kws_index: 0-based line index in keywords.txt on a trigger (-1 otherwise)
   kws_score: average posterior of the matched keyword tokens, range [0, 1] */
typedef struct {
    const short* audio_in;
    int samples;
} KwsInput_t;

typedef struct {
    int kws_index;       /* 0-based keyword index, -1 if none */
    float kws_score;     /* trigger score */
} KwsResult_t;

AUDIO_ENGINE_API int AudioEngine_Kws_Process(
    KwsHandle handle,
    const KwsInput_t* input,
    KwsResult_t* result);

AUDIO_ENGINE_API int AudioEngine_Kws_Deinit(KwsHandle handle);

AUDIO_ENGINE_API int AudioEngine_Kws_Reset(KwsHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_KWS_H */
