#include <stdio.h>
#include <stdlib.h>

#include "sp_pcm_io.h"

int main()
{
	char input_file[256];
	char output_file[256];

	sprintf(input_file, "%s", "assets/sp_linear_chn1_8k_16bit.pcm");
	sprintf(output_file, "%s", "assets/sp_linear_chn1_8k_16bit_out.pcm");

	sp_pcm_handle_t pcm_handle;
	pcm_handle.sample_rate = 8000;
	pcm_handle.channels = 1;
	pcm_handle.pcm_datatype = SP_PCM_DATA_TYPE_S16;
	pcm_handle.pcm_format = SP_PCM_LINEAR;

	int length = 0;
	float *pcm_data = sp_pcm_read_api(&pcm_handle, &length, input_file);
	printf("Read %d samples from %s\n", length, input_file);

	sp_pcm_write_api(&pcm_handle, &length, output_file, pcm_data);
	printf("Write %d samples to %s\n", length, output_file);

	printf("Finished\n");
	return 0;
}

