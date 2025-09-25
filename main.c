#include  "init.h"

unsigned char messages[200];

void main(void)
{
    float hot_temp = 0, cold_temp = 0, ds18b20_temp = 0;
    
	init();
    Led_Init();
    uart1_init(115200);

    MAX31856_Init();
    ds18b20_IO_init();
    
	while(1)
	{
        MAX31856_TriggerOneShot();
        delay_ms(500);
        hot_temp = MAX31856_ReadLinearizedTemp();
        cold_temp = MAX31856_ReadColdJunctionTemp();
        ds18b20_temp = ds18b20_read_temperture();
        
        if (MAX31856_AnalyzeFault())
        {
            sprintf(messages, "热端温度: %.2f ℃\n冷端温度: %.2f ℃\n", hot_temp, cold_temp);
            PrintString1(messages);
        }
        sprintf(messages, "DS18B20 温度: %.2f ℃\n", ds18b20_temp);
        PrintString1(messages);
        
        Led_Toggle();
	}
}
