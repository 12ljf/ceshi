#include "18b20.h"


DS18B20_t DS18B20;

void ds18b20_IO_init(void)
{
	GPIO_InitTypeDef	GPIO_InitStructure;		//结构定义
	GPIO_InitStructure.Pin  = GPIO_Pin_6;		//指定要初始化的IO,
	GPIO_InitStructure.Mode = GPIO_PullUp;		//指定IO的输入或输出方式,GPIO_PullUp,GPIO_HighZ,GPIO_OUT_OD,GPIO_OUT_PP
	GPIO_Inilize(GPIO_P1,&GPIO_InitStructure);	//初始化
}

/*******************************************************************************
* 函 数 名         : ds18b201_reset
* 函数功能		   : 复位ds18b201  
* 输    入         : 无
* 输    出         : 无
*******************************************************************************/
void ds18b20_reset(void)
{
	DS18B20_PORT=0;	//拉低DQ
	delay_10us(75);	//拉低750us
	DS18B20_PORT=1;	//DQ=1
	delay_10us(2);	//20US
}
/*******************************************************************************
* 函 数 名         : ds18b201_check
* 函数功能		   : 检测ds18b201是否存在
* 输    入         : 无
* 输    出         : 1:未检测到ds18b201的存在，0:存在
*******************************************************************************/
u8 ds18b20_check(void)
{
	u8 time_temp=0;

	while(DS18B20_PORT&&time_temp<20)	//等待DQ为低电平
	{
		time_temp++;
		delay_10us(1);	
	}
	if(time_temp>=20)return 1;	//如果超时则强制返回1
	else time_temp=0;
	while((!DS18B20_PORT)&&time_temp<20)	//等待DQ为高电平
	{
		time_temp++;
		delay_10us(1);
	}
	if(time_temp>=20)return 1;	//如果超时则强制返回1
	return 0;
}
/*******************************************************************************
* 函 数 名         : ds18b201_read_bit
* 函数功能		   : 从ds18b201读取一个位
* 输    入         : 无
* 输    出         : 1/0
*******************************************************************************/
u8 ds18b20_read_bit(void)
{
	u8 dat=0;
	
	DS18B20_PORT=0;
	delay_us(6);
	DS18B20_PORT=1;	
	delay_10us(1); //该段时间不能过长，必须在15us内读取数据
	if(DS18B20_PORT)dat=1;	//如果总线上为1则数据dat为1，否则为0
	else dat=0;
	delay_10us(5);
	return dat;
}
/*******************************************************************************
* 函 数 名         : ds18b201_read_bit
* 函数功能		   : 从ds18b201读取两个位
* 输    入         : 无
* 输    出         : 1/0
*******************************************************************************/

// 从ds18b201读取2个位
u8 ds18b20_Read_2Bit(void)//读二位 子程序
{
	u8 i;
	u8 dat = 0;
	for (i = 2; i > 0; i--)
	{
		dat = dat << 1;
		DS18B20_PORT=0;
		delay_us(6);
		DS18B20_PORT=1;	
		delay_10us(1); //该段时间不能过长，必须在15us内读取数据
		if (DS18B20_PORT)	dat |= 0x01;
		delay_us(50);
	}
	return dat;
}
/*******************************************************************************
* 函 数 名         : ds18b201_read_byte
* 函数功能		   : 从ds18b201读取一个字节
* 输    入         : 无
* 输    出         : 一个字节数据
*******************************************************************************/
u8 ds18b20_read_byte(void)
{
	u8 i=0;
	u8 dat=0;
	u8 temp=0;

	for(i=0;i<8;i++)//循环8次，每次读取一位，且先读低位再读高位
	{
		temp=ds18b20_read_bit();
		dat=(temp<<7)|(dat>>1);
	}
	return dat;	
}
/*******************************************************************************
* 函 数 名         : ds18b201_write_byte
* 函数功能		   		 : 写一个字节到ds18b201
* 输    入         : dat：要写入的字节
* 输    出         : 无
*******************************************************************************/

void ds18b20_Write_Bit(u8 dat)
{
	if(dat)
	{
		DS18B20_PORT=0;
		delay_us(5);
		DS18B20_PORT=1;	
		delay_10us(6);
	}
	else
	{
		DS18B20_PORT=0;
		delay_10us(6);
		DS18B20_PORT=1;
		delay_us(5);	
	}	
}
/*******************************************************************************
* 函 数 名         : ds18b201_write_byte
* 函数功能		   : 写一个字节到ds18b201
* 输    入         : dat：要写入的字节
* 输    出         : 无
*******************************************************************************/
void ds18b20_write_byte(u8 dat)
{
	u8 i=0;
	u8 temp=0;

	for(i=0;i<8;i++)//循环8次，每次写一位，且先写低位再写高位
	{
		temp=dat&0x01;//选择低位准备写入
		dat>>=1;//将次高位移到低位
		if(temp)
		{
			DS18B20_PORT=0;
			delay_us(5);
			DS18B20_PORT=1;	
			delay_10us(6);
		}
		else
		{
			DS18B20_PORT=0;
			delay_10us(6);
			DS18B20_PORT=1;
			delay_us(5);	
		}	
	}	
}
/*******************************************************************************
* 函 数 名         : ds18b201_read_temperture
* 函数功能		   : 从ds18b201得到温度值
* 输    入         : 无
* 输    出         : 温度数据
*******************************************************************************/
float ds18b20_read_temperture(void)
{
	float temp;
	u8 dath=0;
	u8 datl=0;
	u16 value=0;


	ds18b20_reset();									//复位
	DS18B20.sta = ds18b20_check();	//检查ds18b201
	ds18b20_write_byte(0xCC);				//SKIP ROM
  ds18b20_write_byte(0x44);				//转换命令	

	ds18b20_reset();//复位
	DS18B20.sta = ds18b20_check();

	ds18b20_write_byte(0xCC);//查询ROM
  ds18b20_write_byte(0xBE);//读存储器

	datl=ds18b20_read_byte();//低字节
	dath=ds18b20_read_byte();//高字节

	value=(dath<<8)+datl;//合并为16位数据
	
	if((value&0xf800)==0xf800)//判断符号位，负温度
	{
		value=(~value)+1; //数据取反再加1
		temp=value*(-0.0625);	
	}
	else //正温度
		temp=value*(0.0625);
	return temp;
}