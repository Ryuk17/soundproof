/*
 * sp_pcm_io.c
 *
 *  Created on: 2023Äê3ÔÂ15ÈÕ
 *      Author: Ryuk
 */


#include <stdio.h>
#include <string.h>

#include "sp_common.h"
#include "sp_log.h"

#include "sp_pcm_io.h"

typedef struct
{
	int sample_rate;
	int channels;
	int bit_width;

	SP_PCM_FORMAT pcm_format;
	SP_DATA_TYPE pcm_datatype;

	FILE *input;
	FILE *output;

	void *pcm_data;
}sp_pcm_inst_t;

PCM_HANDLE *sp_pcm_create(sp_pcm_info_t *pcm_info, const char *path)
{
	sp_pcm_inst_t *pcm_isnt_ptr = (sp_pcm_inst_t *)malloc(sizeof(sp_pcm_inst_t));
	if(pcm_info == NULL)
	{
		SP_LOG(SP_LOG_ERROR, "%s\n", "pcm_info is NULL\n");
		return SP_COM_NULL_INFO;
	}


	pcm_isnt_ptr->sample_rate = pcm_info->sample_rate;
	pcm_isnt_ptr->channels = pcm_info->channels;
	pcm_isnt_ptr->bit_width = pcm_info->bit_width;

	pcm_isnt_ptr->pcm_format = pcm_info->pcm_format;
	pcm_isnt_ptr->pcm_datatype = pcm_info->pcm_datatype;

	return SP_COM_SUCCESS;
}

