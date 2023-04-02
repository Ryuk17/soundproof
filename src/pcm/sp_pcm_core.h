/*
 * sp_pcm_core.h
 *
 *  Created on: 2023Äê4ÔÂ1ÈÕ
 *      Author: Ryuk
 */

#ifndef SRC_SP_PCM_CORE_H_
#define SRC_SP_PCM_CORE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float *sp_pcm_read_linear(sp_pcm_handle_t *pcm_handle, void *raw_data, int length, FILE *fin);
int sp_pcm_write_linear(sp_pcm_handle_t *pcm_handle, float *pcm_data, int length, FILE *fout);


#endif /* SRC_SP_PCM_CORE_H_ */
