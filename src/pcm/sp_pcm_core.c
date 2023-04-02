/*
 * sp_pcm_core.c
 *
 *  Created on: 2023Äê4ÔÂ1ÈÕ
 *      Author: Ryuk
 */


#include "sp_common.h"
#include "sp_log.h"

#include "sp_pcm_io.h"
#include "sp_pcm_core.h"

#define INT16_SCALER 32767

float *sp_pcm_read_linear(sp_pcm_handle_t *pcm_handle, void *raw_data, int bytes, FILE *fin)
{
	switch(pcm_handle->pcm_datatype)
	{
		case SP_PCM_DATA_TYPE_S16:
		{
			int samples = bytes / sizeof(short);
			short *s16_data = (short *)raw_data;
			float *pcm_data = (float *)calloc(samples, sizeof(float));

			for(int i=0; i<samples; i++)
			{
				pcm_data[i] = *(s16_data + i) * 1.0f / INT16_SCALER;
			}

			free(raw_data);
			fclose(fin);
			return pcm_data;
		}

		default:
		{
			SP_LOG(SP_LOG_ERROR, "%s\n", "Invalid PCM data type\n");
			return NULL;
		}
	}
}

int sp_pcm_write_linear(sp_pcm_handle_t *pcm_handle, float *data, int length, FILE *fout)
{
	switch(pcm_handle->pcm_datatype)
	{
		case SP_PCM_DATA_TYPE_S16:
		{
			short *raw_data = (short *)calloc(length, sizeof(short));
			for(int i=0; i<length; i++)
			{
				raw_data[i] = (short)(*(data+i) * INT16_SCALER);
				printf("%d, %f, %d\n", i, *(data+i), raw_data[i]);
			}
			fwrite(raw_data, sizeof(short), length, fout);

			free(raw_data);
			fclose(fout);
			return SP_PCM_RET_SUCCESS;
		}

		default:
		{
			SP_LOG(SP_LOG_ERROR, "%s\n", "Invalid PCM data type\n");
			return SP_PCM_INVALD_DATATYPE;
		}
	}
}

