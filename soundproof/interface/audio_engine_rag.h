/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: RAG unified C interface (text embedding)
 */
#ifndef AUDIO_ENGINE_RAG_H
#define AUDIO_ENGINE_RAG_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* RagHandle;

typedef struct {
    int num_threads;          /* 0 = use thread_num from config.json (4);
                                 > 0 overrides it                              */
    char model_type[32];      /* "", "mnn-llm", "qwen3-embedding"               */
    char config_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
                              /* MNN-LLM config.json. Model/weight/tokenizer
                                 files are resolved relative to it           */
} RagInitConfig;

typedef struct {
    int normalize;            /* 1 = L2-normalize the output (default),
                                 0 = return the raw model vector              */
} RagRtConfig;

typedef struct {
    const char* text;         /* UTF-8 text, must not be NULL or empty         */
} RagInput_t;

typedef struct {
    /* Engine-owned vector; valid until the next Process/Reset/Deinit call. */
    const float* embedding;
    int dim;
} RagResult_t;


AUDIO_ENGINE_API RagHandle AudioEngine_Rag_Create(void);

AUDIO_ENGINE_API int AudioEngine_Rag_Destroy(RagHandle handle);

/* Loads the model. Init writes the default run-time parameters
   (normalize = 1), so SetParam may be skipped. */
AUDIO_ENGINE_API int AudioEngine_Rag_Init(
    RagHandle handle,
    const RagInitConfig* init_config);

/* Incrementally applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Rag_SetParam(
    RagHandle handle,
    const RagRtConfig* rt_config);

/* Restores the defaults and then applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Rag_ResetParam(
    RagHandle handle,
    const RagRtConfig* rt_config);

/* Embeds one UTF-8 text. Fails with AUDIO_ENGINE_ERR_INVALID_PARAM when
   |text| is empty. */
AUDIO_ENGINE_API int AudioEngine_Rag_Process(
    RagHandle handle,
    const RagInput_t* input,
    RagResult_t* result);

/* Returns the model embedding dimension (e.g. 1024 for Qwen3-Embedding-0.6B). */
AUDIO_ENGINE_API int AudioEngine_Rag_GetDim(
    RagHandle handle,
    int* dim);

/* Releases the model; the handle remains usable and can be Init-ed again. */
AUDIO_ENGINE_API int AudioEngine_Rag_Deinit(RagHandle handle);

/* Drops cached text/vectors; the model and parameters are kept. */
AUDIO_ENGINE_API int AudioEngine_Rag_Reset(RagHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_RAG_H */
