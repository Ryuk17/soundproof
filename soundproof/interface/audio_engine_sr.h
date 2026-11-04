/*
 * @Author: Ryuk
 * @Date: 2026-11-04
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-04 00:00:00
 * @Description: SR (Speaker Recognition) unified C interface
 */
#ifndef AUDIO_ENGINE_SR_H
#define AUDIO_ENGINE_SR_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* SrHandle;

typedef struct {
    int sample_rate;      /* only 16000 is supported                          */
    int frame_len;        /* samples per Process call, must == sample_rate/100
                             (10ms); features are cached internally            */
    int num_threads;      /* MNN inference threads, default 1                 */
    char model_type[32];  /* "", "3d-speaker", "wespeaker", "nemo";
                             empty = auto-detect from model metadata           */
    char model_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
} SrInitConfig;

typedef struct {
    float threshold;      /* identification threshold, default 0.5, (0, 1]    */
} SrRtConfig;

/* Input of the online interface. */
typedef struct {
    const short* audio_in;
    int samples;
    int trigger;          /* 1 = KWS fired: infer cached features now         */
} SrInput_t;

/* Output of the online interface. The caller owns |speaker_name|. */
typedef struct {
    char* speaker_name;   /* UTF-8 buffer, empty when unknown                 */
    int name_capacity;    /* input: buffer size in bytes                      */
    int name_len;         /* output: bytes including the trailing NUL         */
    int has_result;       /* output: 1 if this call produced a decision       */
    float score;          /* output: best cosine similarity                   */
} SrResult_t;


AUDIO_ENGINE_API SrHandle AudioEngine_Sr_Create(void);

AUDIO_ENGINE_API int AudioEngine_Sr_Destroy(SrHandle handle);

AUDIO_ENGINE_API int AudioEngine_Sr_Init(SrHandle handle, const SrInitConfig* init_config);

AUDIO_ENGINE_API int AudioEngine_Sr_SetParam(SrHandle handle, const SrRtConfig* rt_config);

AUDIO_ENGINE_API int AudioEngine_Sr_ResetParam(SrHandle handle, const SrRtConfig* rt_config);

/* Online interface, fed like KWS with 10 ms frames.
   trigger=0: only cache features.
   trigger=1: run inference on all cached features (typically right after a
              KWS trigger), fill |result| and clear the cache. */
AUDIO_ENGINE_API int AudioEngine_Sr_Process(
    SrHandle handle,
    const SrInput_t* input,
    SrResult_t* result
);

/* One-shot identification of a complete segment (e.g. the audio segment that
   triggered KWS). Does not touch the streaming feature cache. */
AUDIO_ENGINE_API int AudioEngine_Sr_Identify(
    SrHandle handle,
    const short* audio_in,
    int samples,
    SrResult_t* result
);

/* Offline enrollment from a complete PCM utterance. */
AUDIO_ENGINE_API int AudioEngine_Sr_Register(
    SrHandle handle,
    const char* speaker_name,
    const short* audio_in,
    int samples
);

/* Offline enrollment from a 16 kHz mono WAV file. */
AUDIO_ENGINE_API int AudioEngine_Sr_RegisterFile(
    SrHandle handle,
    const char* speaker_name,
    const char* wav_path
);

/* Offline enrollment from several WAV files; their embeddings are averaged
   (and L2-normalized) before being stored. */
AUDIO_ENGINE_API int AudioEngine_Sr_RegisterFiles(
    SrHandle handle,
    const char* speaker_name,
    const char* const* wav_paths,
    int num_files
);

/* Extracts an embedding from a complete utterance.
   embedding_capacity is in floats; *embedding_dim is set to the required
   dimension (also when the buffer is too small). */
AUDIO_ENGINE_API int AudioEngine_Sr_Extract(
    SrHandle handle,
    const short* audio_in,
    int samples,
    float* embedding,
    int embedding_capacity,
    int* embedding_dim
);

/* Adds an already extracted embedding (dim must match the model). */
AUDIO_ENGINE_API int AudioEngine_Sr_AddEmbedding(
    SrHandle handle,
    const char* speaker_name,
    const float* embedding,
    int dim
);

/* Registry management. */
AUDIO_ENGINE_API int AudioEngine_Sr_Remove(SrHandle handle, const char* speaker_name);

AUDIO_ENGINE_API int AudioEngine_Sr_Clear(SrHandle handle);

AUDIO_ENGINE_API int AudioEngine_Sr_GetNumSpeakers(SrHandle handle, int* count);

AUDIO_ENGINE_API int AudioEngine_Sr_GetSpeakerName(
    SrHandle handle,
    int index,
    char* speaker_name,
    int name_capacity,
    int* name_len
);

AUDIO_ENGINE_API int AudioEngine_Sr_Deinit(SrHandle handle);

/* Clears the streaming feature cache; registrations are kept. */
AUDIO_ENGINE_API int AudioEngine_Sr_Reset(SrHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_SR_H */
