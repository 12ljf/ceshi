#ifndef __18B20_H
#define __18B20_H

#include "init.h"

typedef struct
{
	float temp;
	u8 sta;
	
} DS18B20_t;

sbit DS18B20_PORT=P1^6;	//DS18B20数据口定义
void ds18b20_IO_init(void);
float ds18b20_read_temperture(void);

#endif

