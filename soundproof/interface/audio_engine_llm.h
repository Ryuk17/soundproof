/*
 * @Author: Ryuk
 * @Date: 2026-11-07
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-11-07 00:00:00
 * @Description: Chat LLM/VLM unified C interface (Qwen3.5)
 */
#ifndef AUDIO_ENGINE_LLM_H
#define AUDIO_ENGINE_LLM_H

#include <stdint.h>

#include "audio_engine_def.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* LlmHandle;

typedef struct {
    int num_threads;          /* 0 = config.json thread_num，>0 覆盖        */
    char model_type[32];      /* "", "qwen3.5", "qwen3.5-vl"               */
    char config_path[AUDIO_ENGINE_MAX_FILE_PATH_LEN];
                              /* MNN-LLM config.json                       */
    const char* system_prompt;/* 可选，Init 期间拷贝；NULL 表示无           */
} LlmInitConfig;

typedef struct {
    int max_new_tokens;       /* 默认 512；0 = config.json（8192）          */
    float temperature;        /* <0 = config，0 = greedy，>0 = temperature  */
    float top_p;              /* <0 = config                                */
    int top_k;                /* <0 = config                                */
    int enable_thinking;      /* -1 = 不修改，0/1 覆盖 jinja 思考开关        */
    int strip_thinking;       /* 1 = 输出去掉 <think>...</think>（默认）    */
} LlmRtConfig;

/* One conversation turn. |text| may be NULL/empty when an image is given.
   When both image_data and image_path are set, image_data wins.
   image_data is raw RGB8 (width * height * 3). */
typedef struct {
    const char* text;              /* UTF-8 user prompt                     */
    const char* image_path;        /* optional jpg/png/bmp                  */
    const unsigned char* image_data;/* optional raw RGB8                    */
    int image_width;
    int image_height;
    int reset_history;             /* 1 = clear the conversation first      */
} LlmInput_t;

typedef struct {
    /* Engine-owned UTF-8 text; valid until the next Process/Reset/Deinit. */
    const char* text;
    int text_len;                  /* bytes excluding the trailing NUL      */
    int prompt_tokens;
    int output_tokens;
} LlmResult_t;


AUDIO_ENGINE_API LlmHandle AudioEngine_Llm_Create(void);

AUDIO_ENGINE_API int AudioEngine_Llm_Destroy(LlmHandle handle);

/* Loads the model. Init writes the default run-time parameters
   (max_new_tokens=512, temperature/top_p/top_k=-1, thinking untouched,
   strip_thinking=1), so SetParam may be skipped. */
AUDIO_ENGINE_API int AudioEngine_Llm_Init(
    LlmHandle handle,
    const LlmInitConfig* init_config);

/* Incrementally applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Llm_SetParam(
    LlmHandle handle,
    const LlmRtConfig* rt_config);

/* Restores the defaults and then applies the given parameters. */
AUDIO_ENGINE_API int AudioEngine_Llm_ResetParam(
    LlmHandle handle,
    const LlmRtConfig* rt_config);

/* Generates one assistant reply synchronously. The conversation history is
   kept across calls unless reset_history=1 or Reset is called. Images apply
   to the current turn only; history keeps the user text. */
AUDIO_ENGINE_API int AudioEngine_Llm_Process(
    LlmHandle handle,
    const LlmInput_t* input,
    LlmResult_t* result);

/* Releases the model; the handle remains usable and can be Init-ed again. */
AUDIO_ENGINE_API int AudioEngine_Llm_Deinit(LlmHandle handle);

/* Clears the conversation history and the last reply; model and parameters
   are kept. */
AUDIO_ENGINE_API int AudioEngine_Llm_Reset(LlmHandle handle);


#ifdef __cplusplus
}
#endif

#endif /* AUDIO_ENGINE_LLM_H */
