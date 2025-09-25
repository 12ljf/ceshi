#include	"delay.h"

//========================================================================
// ����: void delay_ms(unsigned char ms)
// ����: ��ʱ������
// ����: ms,Ҫ��ʱ��ms��, ����ֻ֧��1~255ms. �Զ���Ӧ��ʱ��.
// ����: none.
// �汾: VER1.0
// ����: 2021-3-9
// ��ע: 
//========================================================================
void delay_ms(unsigned int ms)
{
	unsigned int i;
	do
	{
		i = MAIN_Fosc / 10000;
		while(--i);
	}while(--ms);
}


void delay_us(unsigned int us)
{
	unsigned int i;
	do
	{
		i = MAIN_Fosc / 10000000;
		while(--i);
	}while(--us);
}

void delay_10us(u16 us)
{
unsigned int i;
	do
	{
		i = MAIN_Fosc / 1000000;
		while(--i);
	}while(--us);	
}

