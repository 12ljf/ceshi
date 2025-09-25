#include "max31856.h"

void MAX31856_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    
    GPIO_InitStructure.Mode = GPIO_OUT_PP;
    GPIO_InitStructure.Pin = GPIO_PIN_CS_MAX31856 ;
    GPIO_Inilize(GPIO_CS_MAX31856 , &GPIO_InitStructure);
    
    GPIO_InitStructure.Mode = GPIO_OUT_PP;
    GPIO_InitStructure.Pin = GPIO_PIN_SCK_MAX31856 ;
    GPIO_Inilize(GPIO_SCK_MAX31856 , &GPIO_InitStructure);
    
    GPIO_InitStructure.Mode = GPIO_OUT_PP;
    GPIO_InitStructure.Pin = GPIO_PIN_MOSI_MAX31856 ;
    GPIO_Inilize(GPIO_MOSI_MAX31856 , &GPIO_InitStructure);
    
    GPIO_InitStructure.Mode = GPIO_HighZ;
    GPIO_InitStructure.Pin = GPIO_PIN_MISO_MAX31856 ;
    GPIO_Inilize(GPIO_MISO_MAX31856 , &GPIO_InitStructure);
    
    CS_MAX31856  = 1;
    SCK_MAX31856  = 1;                              // CPOL = 1���Զ��壩��CPHA = 1��MAX31856 ǿ�ƹ涨��
    
    delay_us(2);                                    // �ȴ���ʼ����ƽ�ȶ���CS��SCK �ߵ�ƽ�ȶ�     
    
    MAX31856_WriteCR0(0x11);                        // 00010001 ָ�����˲�Ƶ��50Hz��ʹ������¶ȼ�⡢����ģʽ��ʹ�ܿ�·���ϼ��
    MAX31856_WriteCR1(0x03);                        // 00000011 ָ�����ȵ�ż����ΪK���ȵ�ż���ȵ�ż��ѹת��ƽ��ģʽΪ1������
}

void MAX31856_TriggerOneShot(void)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR0();
    
    MAX31856_WriteAddress(0x80);                    // ָ�д CR0 �Ĵ���
    MAX31856_WriteByte(dat | 0x40);                 // �� CR0 �Ĵ����� 1SHOT λ��1��ִ��һ����˺��ȵ�żת��
    MAX31856_End();                                 // ����һ��ͨ��
}

void MAX31856_ChangeType(unsigned char thermocouple_type)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR1();                       // ָ��� CR1 �Ĵ���
    
    dat = dat & 0xF0;                               // ������������͵�����
    dat = dat | (thermocouple_type & 0x0F);         // ָ�����������͵�����
    
    MAX31856_WriteCR1(dat);                         // ָ�д CR1 �Ĵ���
}

float MAX31856_ReadColdJunctionTemp(void)
{
    unsigned char msb = 0, lsb = 0;
    int raw;
    float temp;
    
    MAX31856_WriteAddress(0x0A);                    // ָ��� CJTH �Ĵ���
    msb = MAX31856_ReadByte();                      // ��ȡ CJTH �Ĵ��������ݣ�����¶���ֵ�ĸ�8λ
    lsb = MAX31856_ReadByte();                      // ��ȡ CJTL �Ĵ��������ݣ�����¶����ݵĵ�8λ
    MAX31856_End();                                 // ����һ��ͨ��
    
    raw = (int)((unsigned int)msb << 8 | lsb);      // ƴ��16λ���ݣ�ʹ���߼����ƣ�����ԭʼ����
    
    raw >>= 2;                                      // ��ȡ14λ��Ч���ݣ�ͨ���������ƴ�������Ʋ�����ʽ
    
    temp = raw * 0.015625f;                         // ת��λ�¶�ֵ��14λ���룬�ֱ���0.015625��
    
    return temp;                                    // ����ʵ���¶�ֵ
}

float MAX31856_ReadLinearizedTemp(void)
{
    unsigned char msb, mid, lsb;
    long raw;
    float temp;
    
    MAX31856_WriteAddress(0x0C);                    // ָ��� LTCBH �Ĵ���
    msb = MAX31856_ReadByte();                      // ��ȡ LTCBH �Ĵ��������ݣ��ȶ����Ի��¶���ֵ�ĸ�8λ
    mid = MAX31856_ReadByte();                      // ��ȡ LTCBM �Ĵ��������ݣ��ȶ����Ի��¶���ֵ����8λ 
    lsb = MAX31856_ReadByte();                      // ��ȡ LTCBL �Ĵ��������ݣ��ȶ����Ի��¶���ֵ�ĵ�8λ
    MAX31856_End();                                 // ����һ��ͨ��
    
    raw = (long)(((unsigned long)msb << 16) | ((unsigned long)mid << 8) | lsb);         // ƴ��32λ���ݣ�ʹ���߼����ƣ�����ԭʼ����
    raw >>= 5;                                                                          // ��ȡ19λ��Ч���ݣ�ͨ���������ƴ�������Ʋ�����ʽ
    
    temp = raw * 0.0078125f;                        // ת��λ�¶�ֵ��19λ���룬�ֱ���0.0078125��
    
    return temp;                                    // ����ʵ���¶�ֵ
}

unsigned char MAX31856_AnalyzeFault(void)
{
    unsigned char fault = 0;
    
    fault = MAX31856_CheckFault();
    
    if (fault & 0x01) {
        PrintString1("Thermocouple Open-Circuit Fault\n");              // �ȵ�ż��·����
        return 0;
    }
    
    return 1;
}

void MAX31856_SetCJHighThreshold(char temp)
{
    // �������Ƿ񳬳���Χ
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    MAX31856_WriteAddress(0x83);                                        // ������¶����޼Ĵ���д������
    MAX31856_WriteByte((unsigned char)temp);                            // �洢��ʽΪ�����Ʋ��룬�ֱ���Ϊ1
    MAX31856_End();                                                     // ����һ��ͨ��
}

void MAX31856_SetCJLowThreshold(char temp)
{
    // �������Ƿ񳬳���Χ
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    MAX31856_WriteAddress(0x84);                                        // ������¶����޼Ĵ���д������
    MAX31856_WriteByte((unsigned char)temp);                            // �洢��ʽΪ�����Ʋ��룬�ֱ���Ϊ1
    MAX31856_End();                                                     // ����һ��ͨ��
}

void MAX31856_SetTCHighThreshold(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // �������Ƿ񳬳���Χ
    if (temp > 2047.9375) temp = 2047.9375;
    if (temp < -2048.0) temp = -2048.0;
    
    raw = (int)(temp / 0.0625);                                         // �ֱ���Ϊ0.0625
    msb = (unsigned char)(((unsigned int)raw >> 8) & 0xFF);             // ƴ�Ӹ�8λ���ݣ�ע��ʹ���߼�����
    lsb = (unsigned char)((unsigned int)raw & 0xFF);                    // ƴ�ӵ�8λ���ݣ�ע��ʹ���߼�����
    
    MAX31856_WriteAddress(0x85);                                        // �����Ի��¶����޼Ĵ���д������
    MAX31856_WriteByte(msb);                                            // д���8λ����
    MAX31856_WriteByte(lsb);                                            // д���8λ����
    MAX31856_End();                                                     // ����һ��ͨ��
}

void MAX31856_SetTCLowThreshold(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // �������Ƿ񳬳���Χ
    if (temp > 2047.9375) temp = 2047.9375;
    if (temp < -2048.0) temp = -2048.0;
    
    raw = (int)(temp / 0.0625);                                         // �ֱ���Ϊ0.0625
    msb = (unsigned char)(((unsigned int)raw >> 8) & 0xFF);             // ƴ�Ӹ�8λ���ݣ�ע��ʹ���߼�����
    lsb = (unsigned char)((unsigned int)raw & 0xFF);                    // ƴ�ӵ�8λ���ݣ�ע��ʹ���߼�����
    
    MAX31856_WriteAddress(0x87);                                        // �����Ի��¶����޼Ĵ���д������
    MAX31856_WriteByte(msb);                                            // д���8λ����
    MAX31856_WriteByte(lsb);                                            // д���8λ����
    MAX31856_End();                                                     // ����һ��ͨ��
}

void MAX31856_WriteColdJunctionTemp(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // �������Ƿ񳬳���Χ
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    raw = (int)(temp / 0.015625);                                       // �ֱ���Ϊ0.015625
    raw <<= 2;                                                          // ͨ���������ƣ�������λ��Чλ
    
    msb = ((unsigned int)raw >> 8) & 0xFF;                              // ��ȡ��8λ��ͨ���߼����ƣ�����ԭʼ����
    lsb = raw & 0xFF;                                                   // ��ȡ��8λ
    
    MAX31856_WriteAddress(0x8A);                                        // ������¶ȼĴ���д������
    MAX31856_WriteByte(msb);                                            // д���8λ����
    MAX31856_WriteByte(lsb);                                            // д���8λ����
    MAX31856_End();                                                     // ����һ��ͨ��
}

void MAX31856_DisableColdJunctionSensor(void)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR0();                                           // ��ȡCR0�Ĵ���������
    
    dat |= 0x08;                                                        // ƴ�����ݣ������ڲ�����¶ȼ��
    
    MAX31856_WriteCR0(dat);                                             // ��ƴ�ӵ�����д��CR0�Ĵ���
}

void MAX31856_WriteColdJunctionOffset(float offset)
{
    char dat;
    
    // �������Ƿ񳬳���Χ
    if (offset > 7.9375) offset = 7.9375;
    if (offset < -8) offset = -8;
    
    dat = (char)(offset / 0.0625);                  // �ֱ���Ϊ0.0625
    
    MAX31856_WriteAddress(0x89);                    // ������¶�ƫ�ƼĴ���д������
    MAX31856_WriteByte((unsigned char)dat);         // �洢��ʽλ�����Ʋ��룬�ֱ���Ϊ0.0625
    MAX31856_End();                                 // ����һ��ͨ��
}

unsigned char MAX31856_CheckFault(void)
{
    unsigned char fault;
    
    MAX31856_WriteAddress(0x0F);                    // ָ��� SR �Ĵ���
    fault = MAX31856_ReadByte();                    // ��ȡ SR �Ĵ��������ݣ�����״̬�Ĵ���
    MAX31856_End();                                 // ����һ��ͨ��
    
    return fault;
}

void MAX31856_WriteCR0(unsigned char dat)
{
    MAX31856_WriteAddress(0x80);                    // ָ�д CR0 �Ĵ���
    MAX31856_WriteByte(dat);                        // �� CR0 �Ĵ���д��ָ�����ݣ����üĴ���0
    MAX31856_End();                                 // ����һ��ͨ��
}

void MAX31856_WriteCR1(unsigned char dat)
{
    MAX31856_WriteAddress(0x81);                    // ָ�д CR1 �Ĵ���
    MAX31856_WriteByte(dat);                        // �� CR1 �Ĵ���д��ָ�����ݣ����üĴ���1
    MAX31856_End();                                 // ����һ��ͨ��
}

unsigned char MAX31856_ReadCR0(void)
{
    unsigned char dat;
    
    MAX31856_WriteAddress(0x00);                    // ָ��� CR0 �Ĵ���
    dat = MAX31856_ReadByte();                      // ��ȡ CR0 �Ĵ��������ݣ����üĴ���0
    MAX31856_End();                                 // ����һ��ͨ��
    
    return dat;
}

unsigned char MAX31856_ReadCR1(void)
{
    unsigned char dat;
    
    MAX31856_WriteAddress(0x01);                    // ָ��� CR1 �Ĵ���
    dat = MAX31856_ReadByte();                      // ��ȡ CR1 �Ĵ��������ݣ����üĴ���1
    MAX31856_End();                                 // ����һ��ͨ��
    
    return dat;
}

void MAX31856_WriteAddress(unsigned char dat)
{
    unsigned char i;
    
    CS_MAX31856  = 0;                               // ���� CS
    
    MAX31856_delay();                               // ʱ������tCC
    
    /* һ�� for ѭ��������Ҫ328ns������ʱ����������SCKʱ�����ڴ���3MHz������ MAX31856 �� fSCLҪ�� */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // ���� SCK
        
        MOSI_MAX31856 = (dat & 0x80) ? 1 : 0;       // �����ݷŵ� MOSI ��������
        
        MAX31856_delay();                           // ʱ������tCL��tDC
        
        SCK_MAX31856 = 1;                           // ���� SCK
        
        MAX31856_delay();                           // ʱ������tCH��tCDH
        
        dat <<= 1;                                  // �����´����������
    }
}

void MAX31856_WriteByte(unsigned char dat)
{
    unsigned char i;
    
    /* һ�� for ѭ��������Ҫ328ns������ʱ����������SCKʱ�����ڴ���3MHz������ MAX31856 �� fSCLҪ�� */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // ���� SCK
        
        MOSI_MAX31856 = (dat & 0x80) ? 1 : 0;       // �����ݷŵ� MOSI ��������
        
        MAX31856_delay();                           // ʱ������tCL��tDC
        
        SCK_MAX31856 = 1;                           // ���� SCK
        
        MAX31856_delay();                           // ʱ������tCH��tCDH
        
        dat <<= 1;                                  // �����´����������
    }
}

unsigned char MAX31856_ReadByte(void)
{
    unsigned char i, dat = 0;
    
    /* һ�� for ѭ��������Ҫ328ns������ʱ����������SCKʱ�����ڴ���3MHz������ MAX31856 �� fSCLҪ�� */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // ���� SCK
        
        MAX31856_delay();                           // ʱ������tCL��tDC��tCDD
        
        SCK_MAX31856 = 1;                           // ���� SCK
        
        MAX31856_delay();                           // ʱ������tCH��tCDH
        
        dat <<= 1;                                  // �����ϴζ��������
        if (MISO_MAX31856 ) dat |= 0x01;            // ��ȡ MISO �����ߵ�����
    }
    
    return dat;
}

void MAX31856_End(void)
{
    
    SCK_MAX31856 = 1;                               // ���� SCK
    
    MAX31856_delay();                               // ʱ������tCCH
    
    CS_MAX31856 = 1;                                // ���� CS
    
    /* ʱ������tCWH */
    MAX31856_delay();
    MAX31856_delay();
    MAX31856_delay();
    MAX31856_delay();
}

void MAX31856_delay(void)
{
    /* STC8Hϵ�е�Ƭ����ָ��Ƶ��λ24MHz
       ��Ƶ���£�ִ��һ������ָ�������Ҫ41ns���˺�����ȥ�������õ�������Ҫ�����������ӳ�4 * 41 = 164ns
    */
    _nop_();                                        // ִ��һ����ָ��
    _nop_();
    _nop_();
    _nop_();
}
