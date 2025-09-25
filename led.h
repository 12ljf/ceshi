#ifndef __LED_H
#define __LED_H

#include "init.h"

#define LED             P11

#define GPIO_LED        GPIO_P1

#define GPIO_PIN_LED    GPIO_Pin_1

void Led_Init(void);
void Led_Toggle(void);

#endif
