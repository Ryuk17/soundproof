/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: Turn detection unified C interface (Smart Turn v3.2)
 */
#ifndef AUDIO_ENGINE_TURN_H
#define AUDIO_ENGINE_TURN_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* TurnHandle;

typedef struct {
    int sample_rate;          /* only 16000 is supported                      */
    int num_threads;          /* MNN inference threads, default 1             */
    char model_type[32];      /* "", "smart-turn", "smart-turn-v3.2"          */
    char model_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN]; /* .mnn model file     */
} TurnInitConfig;

typedef struct {
    float threshold;          /* complete-turn threshold, default 0.5,
                                 range (0, 1]                                 */
} TurnRtConfig;

/* Audio is 16 kHz mono PCM16. The engine keeps the most recent 8 seconds
   (128000 samples); shorter buffers are zero-padded on the left. */
typedef struct {
    const short* audio_in;    /* may be NULL when in_samples == 0             */
    int in_samples;
    int trigger;              /* 1 = run detection on the buffered audio      */
} TurnInput_t;

typedef struct {
    float probability;        /* sigmoid probability, valid when has_result   */
    int is_complete;          /* 1 when probability >= threshold              */
    int has_result;           /* 1 if this call produced a decision           */
    int buffered_samples;     /* samples currently in the rolling window      */
} TurnResult_t;


AUDIO_ENGINE_API TurnHandle AudioEngine_Turn_Create(void);

AUDIO_ENGINE_API int AudioEngine_Turn_Destroy(TurnHandle handle);

/* Loads the model. Init writes the default run-time parameters
   (threshold = 0.5), so SetParam may be skipped. */
AUDIO_ENGINE_API int AudioEngine_Turn_Init(
    TurnHandle handle,
    const TurnInitConfig* init_config);

/* Incrementally applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Turn_SetParam(
    TurnHandle handle,
    const TurnRtConfig* rt_config);

/* Restores the defaults and then applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Turn_ResetParam(
    TurnHandle handle,
    const TurnRtConfig* rt_config);

/* Feeds audio. With trigger=0 only the rolling buffer is updated; with
   trigger=1 the model runs on the buffered audio (cached until new audio
   arrives). Pass audio_in=NULL/in_samples=0 to trigger without new audio. */
AUDIO_ENGINE_API int AudioEngine_Turn_Process(
    TurnHandle handle,
    const TurnInput_t* input,
    TurnResult_t* result);

/* Returns the model context size in samples (128000 = 8 s at 16 kHz). */
AUDIO_ENGINE_API int AudioEngine_Turn_GetContextSamples(
    TurnHandle handle,
    int* samples);

/* Releases the model; the handle remains usable and can be Init-ed again. */
AUDIO_ENGINE_API int AudioEngine_Turn_Deinit(TurnHandle handle);

/* Clears the rolling buffer and the cached prediction; configuration and
   model are kept. */
AUDIO_ENGINE_API int AudioEngine_Turn_Reset(TurnHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_TURN_H */
