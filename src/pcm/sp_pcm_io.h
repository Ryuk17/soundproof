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
	SP_PCM_RET_SUCCESS			= 0x0000,
	SP_PCM_INVALD_HANDLE 		= 0x1001,
	SP_PCM_INVALD_SAMPLE_RATE 	= 0x1002,
	SP_PCM_INVALD_CHANNEL 		= 0x1003,
	SP_PCM_INVALD_FORMAT 		= 0x1004,
	SP_PCM_INVALD_DATATYPE 		= 0x1005,
}SP_PCM_RET;

typedef enum
{
	SP_PCM_LINEAR = 0,
	SP_PCM_ALAW,
	SP_PCM_ULAW,
}SP_PCM_FORMAT;

typedef enum
{
	SP_PCM_DATA_TYPE_S16 = 0,
	SP_PCM_DATA_TYPE_S24,
	SP_PCM_DATA_TYPE_F32,
}SP_PCM_DATA_TYPE;

typedef struct
{
	int sample_rate;
	int channels;

	SP_PCM_FORMAT pcm_format;
	SP_PCM_DATA_TYPE pcm_datatype;
}sp_pcm_handle_t;


float *sp_pcm_read_api(sp_pcm_handle_t *pcm_handle, int *length, const char *pcm_file);
SP_PCM_RET sp_pcm_write_api(sp_pcm_handle_t *pcm_handle, int *length, const char *pcm_file, const float *output);


#endif /* SP_PCM_IO_H_ */
