/*
 * @Author: Ryuk
 * @Date: 2026-08-09 18:44:06
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-02 23:34:50
 * @Description: VAD (Voice Activity Detection) unified C interface
 */
#ifndef AUDIO_ENGINE_VAD_H
#define AUDIO_ENGINE_VAD_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* VadHandle;

typedef struct {
    int sample_rate;     /* dsp: 8000/16000/32000/48000; silero: only 16000 */
    int frame_len;       /* samples per Process call, must == sample_rate/100 (10ms)
                            for both modes (silero buffers 10ms chunks internally) */
    int vad_mode;        /* 0: dsp (WebRTC) mode, 1: silero vad (requires -DWITH_MNN=ON) */
    char model_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN]; /* silero: path to the .mnn model;
                                                         ignored by dsp mode */
} VadInitConfig;

/* Zero-initialize this struct (VadRtConfig cfg = {0};) before setting fields:
   the fields below use "<= 0 keeps the current/default value" semantics, so
   uninitialized stack garbage would be interpreted as caller parameters. */
typedef struct {
    float threshold;            /* voice probability threshold, default 0.5,
                                   range (0.0, 1.0]                       */
    float min_speech_duration;  /* seconds, default 0.25: probability must stay
                                   above threshold this long before speech
                                   starts (onset debounce).
                                   <= 0 keeps the current/default value      */
    float min_silence_duration; /* seconds, default 0.5: hangover, keep
                                   reporting speech this long after the
                                   probability drops below threshold.
                                   <= 0 keeps the current/default value      */
    float hysteresis;           /* default 0.15: once speech is active,
                                   threshold - hysteresis keeps it active.
                                   <= 0 keeps the current/default value      */
} VadRtConfig;


AUDIO_ENGINE_API VadHandle AudioEngine_Vad_Create(void);

AUDIO_ENGINE_API int AudioEngine_Vad_Destroy(VadHandle handle);

AUDIO_ENGINE_API int AudioEngine_Vad_Init(VadHandle handle, const VadInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Vad_SetParam(VadHandle handle, const VadRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Vad_ResetParam(VadHandle handle, const VadRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Vad_Process(
    VadHandle handle,
    const short* audio_in,
    int samples,
    int *vad_flag
);

AUDIO_ENGINE_API int AudioEngine_Vad_Deinit(VadHandle handle);

AUDIO_ENGINE_API int AudioEngine_Vad_Reset(VadHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_VAD_H */