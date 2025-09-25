#ifndef __UART_H_
#define __UART_H_

#include "init.h"

void uart1_init(unsigned long bound);
void uart2_init(unsigned long bound);

void RxData_Handle(void);

#endif
