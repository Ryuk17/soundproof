/*
 * sp_pcm_io.c
 *
 *  Created on: 2023Äê3ÔÂ15ÈÕ
 *      Author: Ryuk
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sp_common.h"
#include "sp_log.h"

#include "sp_pcm_io.h"
#include "sp_pcm_core.h"

float *sp_pcm_read_api(sp_pcm_handle_t *pcm_handle, int *length, const char *pcm_file)
{
	if(pcm_handle == NULL)
	{
		SP_LOG(SP_LOG_ERROR, "%s\n", "pcm handle is NULL\n");
		return NULL;
	}

	FILE *fin;
	if(pcm_file != NULL)
	{
		fin = fopen(pcm_file, "rb");
		if(!fin)
		{
			perror(pcm_file);
			return NULL;
		}
		SP_LOG(SP_LOG_INFO, "Open %s successfully\n", pcm_file);
	}

	fseek(fin, 0, SEEK_END);
	*length = ftell(fin);
	rewind(fin);

	char *raw_data = (char *)malloc((*length) * sizeof(char));
	*length = fread(raw_data, 1, *length, fin);

	switch(pcm_handle->pcm_format)
	{
		case SP_PCM_LINEAR:
		{
			return sp_pcm_read_linear(pcm_handle, raw_data, *length, fin);
		}

		default:
		{
			SP_LOG(SP_LOG_ERROR, "%s\n", "Invalid PCM format\n");
			return NULL;
		}
	}
}

SP_PCM_RET sp_pcm_write_api(sp_pcm_handle_t *pcm_handle, int *length, const char *pcm_file, const float *pcm_data)
{
	if(pcm_handle == NULL)
	{
		SP_LOG(SP_LOG_ERROR, "%s\n", "pcm handle is NULL\n");
		return SP_PCM_INVALD_HANDLE;
	}

	FILE *fout;
	if(pcm_file != NULL)
	{
		fout = fopen(pcm_file, "wb+");
		if(!fout)
		{
			perror(pcm_file);
			return SP_PCM_INVALD_HANDLE;
		}
		SP_LOG(SP_LOG_INFO, "Open %s successfully\n", pcm_file);
	}

	switch(pcm_handle->pcm_format)
	{
		case SP_PCM_LINEAR:
		{
			return sp_pcm_write_linear(pcm_handle, pcm_data, *length, fout);
		}

		default:
		{
			SP_LOG(SP_LOG_ERROR, "%s\n", "Invalid PCM format\n");
			return SP_PCM_INVALD_FORMAT;
		}
	}
}

