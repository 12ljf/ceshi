#include "led.h"

void Led_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    
    GPIO_InitStructure.Mode = GPIO_OUT_PP;
    GPIO_InitStructure.Pin = GPIO_PIN_LED;
    
    GPIO_Inilize(GPIO_LED, &GPIO_InitStructure);
    
    LED = 1;
}

void Led_Toggle(void)
{
    LED = ~LED;
}