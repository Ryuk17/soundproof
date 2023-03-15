/*
 * sp_log.h
 *
 *  Created on: 2023Äê3ÔÂ15ÈÕ
 *      Author: Ryuk
 */

#ifndef SP_LOG_H_
#define SP_LOG_H_


#define SP_COM_SUCCESS			(0X000)
#define SP_COM_NULL_INFO		(0x001)
#define SP_COM_INVALD_PATH		(0x002)
#define SP_PCM_INVALD_PATH		(0x002)

#define SP_LOG_DEBUG 			(0)
#define SP_LOG_INFO 			(1)
#define SP_LOG_WARNING 			(2)
#define SP_LOG_ERROR 			(3)
#define SP_LOG_LEVEL 			(SP_LOG_INFO)

#define SP_LOG(level,log_fmt,...) \
	do{ \
		if(level >= SP_LOG_LEVEL)\
		{\
			switch(level) \
			{ \
			case SP_LOG_DEBUG: \
				printf("SP_LOG_DEBUG: [%s:%d][%s] \t"log_fmt"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
				break;\
			case SP_LOG_INFO: \
				printf("SP_LOG_INFO: [%s:%d][%s] \t"log_fmt"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
				break;\
			case SP_LOG_WARNING: \
				printf("SP_LOG_WARNING: [%s:%d][%s] \t"log_fmt"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
				break;\
			case SP_LOG_ERROR: \
				printf("SP_LOG_ERROR: [%s:%d][%s] \t"log_fmt"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
				break;\
			default: \
				break;\
			} \
		}\
	}while (0)


#endif /* SP_LOG_H_ */
