#ifndef DELAY_H
#define DELAY_H

#include "main.h"

void delay_init(void);
void delay_us(uint32_t nus);
void delay_ms(uint32_t nms);

#endif