#ifndef __INIT_H_
#define __INIT_H_

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "config.h"
#include "delay.h"
#include "uart.h"

// �ٷ���ͷ�ļ�����
#include "stc_gpio.h"
#include "stc_uart.h"
#include "stc_eeprom.h"

// �û���ͷ�ļ�����
#include "18b20.h"
#include "led.h"
#include "max31856.h"

void init(void);

#endif
