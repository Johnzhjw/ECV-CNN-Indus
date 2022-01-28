#ifndef _MOP_HDSN_
#define _MOP_HDSN_

//////////////////////////////////////////////////////////////////////////
//HDSN
#define ds 50//?????¡§¦Ì?¡ä??D?¡Â¨ºy¨¢?
#define UNIT 4
#define ds_delay 5
#define UNIT_delay 2

//#define DIM_HDSN (ds * UNIT + ds_delay * UNIT_delay)
#define DIM_HDSN (ds * UNIT)
#define HDSNOBJ 3

void Fitness_HDSN(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_HDSN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_HDSN(double* x, int nx);

#endif
