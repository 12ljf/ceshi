#include "uart.h"


void uart1_init(unsigned long bound)
{
	GPIO_InitTypeDef	GPIO_InitStructure;		//�ṹ����
	COMx_InitDefine		COMx_InitStructure;					//�ṹ����
	
	
	GPIO_InitStructure.Pin  = GPIO_Pin_0 | GPIO_Pin_1;		//ָ��Ҫ��ʼ����IO, GPIO_Pin_0 ~ GPIO_Pin_7
	GPIO_InitStructure.Mode = GPIO_PullUp;	//ָ��IO������������ʽ,GPIO_PullUp,GPIO_HighZ,GPIO_OUT_OD,GPIO_OUT_PP
	GPIO_Inilize(GPIO_P3,&GPIO_InitStructure);	//��ʼ��
	
	
	COMx_InitStructure.UART_Mode      = UART_8bit_BRTx;		//ģʽ,       UART_ShiftRight,UART_8bit_BRTx,UART_9bit,UART_9bit_BRTx
	COMx_InitStructure.UART_BRT_Use   = BRT_Timer1;			//ѡ�����ʷ�����,   BRT_Timer1, BRT_Timer2 (ע��: ����2�̶�ʹ��BRT_Timer2)
	COMx_InitStructure.UART_BaudRate  = bound;			//������, һ�� 110 ~ 115200
	COMx_InitStructure.UART_RxEnable  = ENABLE;				//��������,   ENABLE��DISABLE
	COMx_InitStructure.BaudRateDouble = DISABLE;			//�����ʼӱ�, ENABLE��DISABLE
	COMx_InitStructure.UART_Interrupt = ENABLE;				//�ж�����,   ENABLE��DISABLE
	COMx_InitStructure.UART_Priority    = Priority_0;			//ָ���ж����ȼ�(�͵���) Priority_0,Priority_1,Priority_2,Priority_3
	COMx_InitStructure.UART_P_SW      = UART1_SW_P30_P31;	//�л��˿�,   UART1_SW_P30_P31,UART1_SW_P36_P37,UART1_SW_P16_P17,UART1_SW_P43_P44
	UART_Configuration(UART1, &COMx_InitStructure);		//��ʼ������1 UART1,UART2,UART3,UART4
}


void uart2_init(unsigned long bound)
{
	GPIO_InitTypeDef	GPIO_InitStructure;		//�ṹ����
	COMx_InitDefine		COMx_InitStructure;					//�ṹ����
	
	
	GPIO_InitStructure.Pin  = GPIO_Pin_0 | GPIO_Pin_1;		//ָ��Ҫ��ʼ����IO, GPIO_Pin_0 ~ GPIO_Pin_7
	GPIO_InitStructure.Mode = GPIO_PullUp;	//ָ��IO������������ʽ,GPIO_PullUp,GPIO_HighZ,GPIO_OUT_OD,GPIO_OUT_PP
	GPIO_Inilize(GPIO_P1,&GPIO_InitStructure);	//��ʼ��
	
	GPIO_InitStructure.Pin  = GPIO_Pin_2;		//ָ��Ҫ��ʼ����IO, GPIO_Pin_0 ~ GPIO_Pin_7
	GPIO_InitStructure.Mode = GPIO_OUT_PP;	//ָ��IO������������ʽ,GPIO_PullUp,GPIO_HighZ,GPIO_OUT_OD,GPIO_OUT_PP
	GPIO_Inilize(GPIO_P1,&GPIO_InitStructure);	//��ʼ��
	
	COMx_InitStructure.UART_Mode      = UART_8bit_BRTx;		//ģʽ,       UART_ShiftRight,UART_8bit_BRTx,UART_9bit,UART_9bit_BRTx
	COMx_InitStructure.UART_BRT_Use   = BRT_Timer2;			//ѡ�����ʷ�����, BRT_Timer2 (ע��: ����2�̶�ʹ��BRT_Timer2, ���Բ���ѡ��)
	COMx_InitStructure.UART_BaudRate  = bound;			//������,     110 ~ 115200
	COMx_InitStructure.UART_RxEnable  = ENABLE;				//��������,   ENABLE��DISABLE
	COMx_InitStructure.BaudRateDouble = DISABLE;			//�����ʼӱ�, ENABLE��DISABLE
	COMx_InitStructure.UART_Interrupt = ENABLE;				//�ж�����,   ENABLE��DISABLE
	COMx_InitStructure.UART_Priority    = Priority_0;			//ָ���ж����ȼ�(�͵���) Priority_0,Priority_1,Priority_2,Priority_3
	COMx_InitStructure.UART_P_SW      = UART2_SW_P10_P11;	//�л��˿�,   UART2_SW_P10_P11,UART2_SW_P46_P47
	UART_Configuration(UART2, &COMx_InitStructure);		//��ʼ������2 UART1,UART2,UART3,UART4
}


void RxData_Handle(void)
{
	//�жϳ�ʱ����
	if(COM1.RX_TimeOut > 0)
	{
		if(--COM1.RX_TimeOut == 0)
		{	
			if(COM1.RX_Cnt > 0)
			{
			}
			COM1.RX_Cnt = 0;
		}
	}
	
	if(COM2.RX_TimeOut > 0)
	{
		if(--COM2.RX_TimeOut == 0)
		{
			if(COM2.RX_Cnt > 0)
			{
			}
			COM2.RX_Cnt = 0;
		}
	}
}

