/*
 * @Author: Ryuk
 * @Date: 2026-08-09 22:00:00
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-08-09 22:00:00
 * @Description: Resample unified C interface
 */
#ifndef AUDIO_ENGINE_RESAMPLE_H
#define AUDIO_ENGINE_RESAMPLE_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* ResampleHandle;

typedef struct {
    int src_sample_rate;     /* source sample rate: 8000/16000/32000/48000 */
    int dst_sample_rate;     /* destination sample rate                    */
    int num_channels;        /* 1 = mono, 2 = stereo                       */
} ResampleInitConfig;

typedef struct {
    int reserved;            /* reserved, keep 0 */
} ResampleRtConfig;


AUDIO_ENGINE_API ResampleHandle AudioEngine_Resample_Create(void);

AUDIO_ENGINE_API int AudioEngine_Resample_Destroy(ResampleHandle handle);

AUDIO_ENGINE_API int AudioEngine_Resample_Init(ResampleHandle handle, const ResampleInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Resample_SetParam(ResampleHandle handle, const ResampleRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Resample_ResetParam(ResampleHandle handle, const ResampleRtConfig* rt_config);

typedef struct {
    const short* audio_in;
    int in_samples;
} ResampleInput_t;

typedef struct {
    short* audio_out;
    int max_out_samples;
    int out_samples;
} ResampleResult_t;

AUDIO_ENGINE_API int AudioEngine_Resample_Process(
    ResampleHandle handle,
    const ResampleInput_t* input,
    ResampleResult_t* result);

AUDIO_ENGINE_API int AudioEngine_Resample_Deinit(ResampleHandle handle);

AUDIO_ENGINE_API int AudioEngine_Resample_Reset(ResampleHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_RESAMPLE_H */