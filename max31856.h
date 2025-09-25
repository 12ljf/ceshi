#ifndef __MAX31856_H
#define __MAX31856_H

#include "init.h"

#define CS_MAX31856                   P15
#define SCK_MAX31856                  P14
#define MISO_MAX31856                 P13
#define MOSI_MAX31856                 P12

#define GPIO_CS_MAX31856              GPIO_P1
#define GPIO_SCK_MAX31856             GPIO_P1
#define GPIO_MISO_MAX31856            GPIO_P1
#define GPIO_MOSI_MAX31856            GPIO_P1

#define GPIO_PIN_CS_MAX31856          GPIO_Pin_5
#define GPIO_PIN_SCK_MAX31856         GPIO_Pin_4
#define GPIO_PIN_MISO_MAX31856        GPIO_Pin_3
#define GPIO_PIN_MOSI_MAX31856        GPIO_Pin_2

void MAX31856_Init(void);
void MAX31856_TriggerOneShot(void);
float MAX31856_ReadColdJunctionTemp(void);
void MAX31856_WriteColdJunctionTemp(float temperature);
float MAX31856_ReadLinearizedTemp(void);
unsigned char MAX31856_AnalyzeFault(void);
void MAX31856_SetCJHighThreshold(char temp);
void MAX31856_SetCJLowThreshold(char temp);
void MAX31856_SetTCHighThreshold(float temp);
void MAX31856_SetTCLowThreshold(float temp);
void MAX31856_DisableColdJunctionSensor(void);
void MAX31856_WriteColdJunctionOffset(float offset);


void MAX31856_WriteCR0(unsigned char dat);
void MAX31856_WriteCR1(unsigned char dat);
unsigned char MAX31856_ReadCR0(void);
unsigned char MAX31856_ReadCR1(void);
unsigned char MAX31856_CheckFault(void);

void MAX31856_WriteAddress(unsigned char dat);
void MAX31856_WriteByte(unsigned char dat);
unsigned char MAX31856_ReadByte(void);
void MAX31856_End(void);

void MAX31856_delay(void);

#endif
