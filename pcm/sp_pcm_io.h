/*
 * sp_pcm_io.h
 *
 *  Created on: 2023Äê3ÔÂ15ÈÕ
 *      Author: Ryuk
 */

#ifndef SP_PCM_IO_H_
#define SP_PCM_IO_H_

#include "sp_common.h"

typedef enum
{
	SP_LINEAR_PCM = 0,
	SP_ALAW_PCM,
	SP_ULAW_PCM
}SP_PCM_FORMAT;

typedef struct
{
	int sample_rate;
	int channels;
	int bit_width;

	SP_PCM_FORMAT pcm_format;
	SP_DATA_TYPE pcm_datatype;
}sp_pcm_info_t;

typedef void PCM_HANDLE;

PCM_HANDLE *sp_pcm_create(sp_pcm_info_t *pcm_info, const char *path);
int sp_pcm_read(PCM_HANDLE *pcm_handle, void* input);
int sp_pcm_write(PCM_HANDLE *pcm_handle, void *output, const char *path);
int sp_pcm_free(PCM_HANDLE *pcm_handle);

#endif /* SP_PCM_IO_H_ */
