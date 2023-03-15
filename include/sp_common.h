/*
 * sp_common.h
 *
 *  Created on: 2023��3��15��
 *      Author: Ryuk
 */

#ifndef SP_COMMON_H_
#define SP_COMMON_H_

typedef enum
{
	SP_DATATYPE_INT8 = 0,
	SP_DATATYPE_INT16,
	SP_DATATYPE_INT32,
	SP_DATATYPE_FLOAT32
}SP_DATA_TYPE;

typedef enum
{
	SP_SAMPLE_RATE_8000 = 0,
	SP_SAMPLE_RATE_16000,
	SP_SAMPLE_RATE_24000,
	SP_SAMPLE_RATE_32000,
	SP_SAMPLE_RATE_44100,
	SP_SAMPLE_RATE_48000,
}SP_SAMPLE_RATE;


#endif /* SP_COMMON_H_ */
