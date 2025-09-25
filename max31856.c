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
    SCK_MAX31856  = 1;                              // CPOL = 1（自定义），CPHA = 1（MAX31856 强制规定）
    
    delay_us(2);                                    // 等待初始化电平稳定，CS、SCK 高电平稳定     
    
    MAX31856_WriteCR0(0x11);                        // 00010001 指定：滤波频率50Hz、使能冷端温度检测、常闭模式、使能开路故障检测
    MAX31856_WriteCR1(0x03);                        // 00000011 指定：热电偶类型为K型热电偶、热电偶电压转换平均模式为1个采样
}

void MAX31856_TriggerOneShot(void)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR0();
    
    MAX31856_WriteAddress(0x80);                    // 指令：写 CR0 寄存器
    MAX31856_WriteByte(dat | 0x40);                 // 将 CR0 寄存器的 1SHOT 位置1，执行一次冷端和热电偶转换
    MAX31856_End();                                 // 结束一次通信
}

void MAX31856_ChangeType(unsigned char thermocouple_type)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR1();                       // 指令：读 CR1 寄存器
    
    dat = dat & 0xF0;                               // 清除传感器类型的数据
    dat = dat | (thermocouple_type & 0x0F);         // 指定传感器类型的数据
    
    MAX31856_WriteCR1(dat);                         // 指令：写 CR1 寄存器
}

float MAX31856_ReadColdJunctionTemp(void)
{
    unsigned char msb = 0, lsb = 0;
    int raw;
    float temp;
    
    MAX31856_WriteAddress(0x0A);                    // 指令：读 CJTH 寄存器
    msb = MAX31856_ReadByte();                      // 读取 CJTH 寄存器的数据，冷端温度数值的高8位
    lsb = MAX31856_ReadByte();                      // 读取 CJTL 寄存器的数据，冷端温度数据的低8位
    MAX31856_End();                                 // 结束一次通信
    
    raw = (int)((unsigned int)msb << 8 | lsb);      // 拼接16位数据，使用逻辑左移，保留原始内容
    
    raw >>= 2;                                      // 提取14位有效数据，通过算数右移处理二进制补码形式
    
    temp = raw * 0.015625f;                         // 转换位温度值：14位补码，分辨率0.015625℃
    
    return temp;                                    // 返回实际温度值
}

float MAX31856_ReadLinearizedTemp(void)
{
    unsigned char msb, mid, lsb;
    long raw;
    float temp;
    
    MAX31856_WriteAddress(0x0C);                    // 指令：读 LTCBH 寄存器
    msb = MAX31856_ReadByte();                      // 读取 LTCBH 寄存器的数据，热端线性化温度数值的高8位
    mid = MAX31856_ReadByte();                      // 读取 LTCBM 寄存器的数据，热端线性化温度数值的中8位 
    lsb = MAX31856_ReadByte();                      // 读取 LTCBL 寄存器的数据，热端线性化温度数值的低8位
    MAX31856_End();                                 // 结束一次通信
    
    raw = (long)(((unsigned long)msb << 16) | ((unsigned long)mid << 8) | lsb);         // 拼接32位数据，使用逻辑左移，保留原始内容
    raw >>= 5;                                                                          // 提取19位有效数据，通过算数右移处理二进制补码形式
    
    temp = raw * 0.0078125f;                        // 转换位温度值：19位补码，分辨率0.0078125℃
    
    return temp;                                    // 返回实际温度值
}

unsigned char MAX31856_AnalyzeFault(void)
{
    unsigned char fault = 0;
    
    fault = MAX31856_CheckFault();
    
    if (fault & 0x01) {
        PrintString1("Thermocouple Open-Circuit Fault\n");              // 热电偶开路故障
        return 0;
    }
    
    return 1;
}

void MAX31856_SetCJHighThreshold(char temp)
{
    // 检查参数是否超出范围
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    MAX31856_WriteAddress(0x83);                                        // 向冷端温度上限寄存器写入数据
    MAX31856_WriteByte((unsigned char)temp);                            // 存储格式为二进制补码，分辨率为1
    MAX31856_End();                                                     // 结束一次通信
}

void MAX31856_SetCJLowThreshold(char temp)
{
    // 检查参数是否超出范围
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    MAX31856_WriteAddress(0x84);                                        // 向冷端温度下限寄存器写入数据
    MAX31856_WriteByte((unsigned char)temp);                            // 存储格式为二进制补码，分辨率为1
    MAX31856_End();                                                     // 结束一次通信
}

void MAX31856_SetTCHighThreshold(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // 检查参数是否超出范围
    if (temp > 2047.9375) temp = 2047.9375;
    if (temp < -2048.0) temp = -2048.0;
    
    raw = (int)(temp / 0.0625);                                         // 分辨率为0.0625
    msb = (unsigned char)(((unsigned int)raw >> 8) & 0xFF);             // 拼接高8位数据，注意使用逻辑右移
    lsb = (unsigned char)((unsigned int)raw & 0xFF);                    // 拼接低8位数据，注意使用逻辑右移
    
    MAX31856_WriteAddress(0x85);                                        // 向线性化温度上限寄存器写入数据
    MAX31856_WriteByte(msb);                                            // 写入高8位数据
    MAX31856_WriteByte(lsb);                                            // 写入低8位数据
    MAX31856_End();                                                     // 结束一次通信
}

void MAX31856_SetTCLowThreshold(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // 检查参数是否超出范围
    if (temp > 2047.9375) temp = 2047.9375;
    if (temp < -2048.0) temp = -2048.0;
    
    raw = (int)(temp / 0.0625);                                         // 分辨率为0.0625
    msb = (unsigned char)(((unsigned int)raw >> 8) & 0xFF);             // 拼接高8位数据，注意使用逻辑右移
    lsb = (unsigned char)((unsigned int)raw & 0xFF);                    // 拼接低8位数据，注意使用逻辑右移
    
    MAX31856_WriteAddress(0x87);                                        // 向线性化温度下限寄存器写入数据
    MAX31856_WriteByte(msb);                                            // 写入高8位数据
    MAX31856_WriteByte(lsb);                                            // 写入低8位数据
    MAX31856_End();                                                     // 结束一次通信
}

void MAX31856_WriteColdJunctionTemp(float temp)
{
    int raw;
    unsigned char msb, lsb;
    
    // 检查参数是否超出范围
    if (temp > 125) temp = 125;
    if (temp < -55) temp = -55;
    
    raw = (int)(temp / 0.015625);                                       // 分辨率为0.015625
    raw <<= 2;                                                          // 通过算数左移，保留两位无效位
    
    msb = ((unsigned int)raw >> 8) & 0xFF;                              // 提取高8位，通过逻辑右移，保留原始数据
    lsb = raw & 0xFF;                                                   // 提取低8位
    
    MAX31856_WriteAddress(0x8A);                                        // 向冷端温度寄存器写入数据
    MAX31856_WriteByte(msb);                                            // 写入高8位数据
    MAX31856_WriteByte(lsb);                                            // 写入低8位数据
    MAX31856_End();                                                     // 结束一次通信
}

void MAX31856_DisableColdJunctionSensor(void)
{
    unsigned char dat;
    
    dat = MAX31856_ReadCR0();                                           // 读取CR0寄存器的内容
    
    dat |= 0x08;                                                        // 拼接数据，禁用内部冷端温度检测
    
    MAX31856_WriteCR0(dat);                                             // 将拼接的数据写入CR0寄存器
}

void MAX31856_WriteColdJunctionOffset(float offset)
{
    char dat;
    
    // 检查参数是否超出范围
    if (offset > 7.9375) offset = 7.9375;
    if (offset < -8) offset = -8;
    
    dat = (char)(offset / 0.0625);                  // 分辨率为0.0625
    
    MAX31856_WriteAddress(0x89);                    // 向冷端温度偏移寄存器写入数据
    MAX31856_WriteByte((unsigned char)dat);         // 存储格式位二进制补码，分辨率为0.0625
    MAX31856_End();                                 // 结束一次通信
}

unsigned char MAX31856_CheckFault(void)
{
    unsigned char fault;
    
    MAX31856_WriteAddress(0x0F);                    // 指令：读 SR 寄存器
    fault = MAX31856_ReadByte();                    // 读取 SR 寄存器的数据，故障状态寄存器
    MAX31856_End();                                 // 结束一次通信
    
    return fault;
}

void MAX31856_WriteCR0(unsigned char dat)
{
    MAX31856_WriteAddress(0x80);                    // 指令：写 CR0 寄存器
    MAX31856_WriteByte(dat);                        // 向 CR0 寄存器写入指定数据，配置寄存器0
    MAX31856_End();                                 // 结束一次通信
}

void MAX31856_WriteCR1(unsigned char dat)
{
    MAX31856_WriteAddress(0x81);                    // 指令：写 CR1 寄存器
    MAX31856_WriteByte(dat);                        // 向 CR1 寄存器写入指定数据，配置寄存器1
    MAX31856_End();                                 // 结束一次通信
}

unsigned char MAX31856_ReadCR0(void)
{
    unsigned char dat;
    
    MAX31856_WriteAddress(0x00);                    // 指令：读 CR0 寄存器
    dat = MAX31856_ReadByte();                      // 读取 CR0 寄存器的数据，配置寄存器0
    MAX31856_End();                                 // 结束一次通信
    
    return dat;
}

unsigned char MAX31856_ReadCR1(void)
{
    unsigned char dat;
    
    MAX31856_WriteAddress(0x01);                    // 指令：读 CR1 寄存器
    dat = MAX31856_ReadByte();                      // 读取 CR1 寄存器的数据，配置寄存器1
    MAX31856_End();                                 // 结束一次通信
    
    return dat;
}

void MAX31856_WriteAddress(unsigned char dat)
{
    unsigned char i;
    
    CS_MAX31856  = 0;                               // 拉低 CS
    
    MAX31856_delay();                               // 时间间隔：tCC
    
    /* 一次 for 循环至少需要328ns（两个时间间隔），即SCK时钟周期大致3MHz，符合 MAX31856 的 fSCL要求 */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // 拉低 SCK
        
        MOSI_MAX31856 = (dat & 0x80) ? 1 : 0;       // 将数据放到 MOSI 数据线上
        
        MAX31856_delay();                           // 时间间隔：tCL、tDC
        
        SCK_MAX31856 = 1;                           // 拉高 SCK
        
        MAX31856_delay();                           // 时间间隔：tCH、tCDH
        
        dat <<= 1;                                  // 更新下次输出的数据
    }
}

void MAX31856_WriteByte(unsigned char dat)
{
    unsigned char i;
    
    /* 一次 for 循环至少需要328ns（两个时间间隔），即SCK时钟周期大致3MHz，符合 MAX31856 的 fSCL要求 */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // 拉低 SCK
        
        MOSI_MAX31856 = (dat & 0x80) ? 1 : 0;       // 将数据放到 MOSI 数据线上
        
        MAX31856_delay();                           // 时间间隔：tCL、tDC
        
        SCK_MAX31856 = 1;                           // 拉高 SCK
        
        MAX31856_delay();                           // 时间间隔：tCH、tCDH
        
        dat <<= 1;                                  // 更新下次输出的数据
    }
}

unsigned char MAX31856_ReadByte(void)
{
    unsigned char i, dat = 0;
    
    /* 一次 for 循环至少需要328ns（两个时间间隔），即SCK时钟周期大致3MHz，符合 MAX31856 的 fSCL要求 */
    for (i = 0; i < 8; i++)
    {
        SCK_MAX31856 = 0;                           // 拉低 SCK
        
        MAX31856_delay();                           // 时间间隔：tCL、tDC、tCDD
        
        SCK_MAX31856 = 1;                           // 拉高 SCK
        
        MAX31856_delay();                           // 时间间隔：tCH、tCDH
        
        dat <<= 1;                                  // 更新上次读入的数据
        if (MISO_MAX31856 ) dat |= 0x01;            // 读取 MISO 数据线的数据
    }
    
    return dat;
}

void MAX31856_End(void)
{
    
    SCK_MAX31856 = 1;                               // 拉高 SCK
    
    MAX31856_delay();                               // 时间间隔：tCCH
    
    CS_MAX31856 = 1;                                // 拉高 CS
    
    /* 时间间隔：tCWH */
    MAX31856_delay();
    MAX31856_delay();
    MAX31856_delay();
    MAX31856_delay();
}

void MAX31856_delay(void)
{
    /* STC8H系列单片机，指定频率位24MHz
       此频率下，执行一个机器指令大致需要41ns，此函数除去函数调用等其他必要开销，最少延迟4 * 41 = 164ns
    */
    _nop_();                                        // 执行一个空指令
    _nop_();
    _nop_();
    _nop_();
}
