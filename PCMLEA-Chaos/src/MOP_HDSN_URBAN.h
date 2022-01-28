#ifndef _MOP_HDSN_URBAN_
#define _MOP_HDSN_URBAN_

//////////////////////////////////////////////////////////////////////////
//HDSN_URBAN

#define TII_SC_URBAN1

#ifdef TII_SC_URBAN1
#define n_sensor_URBAN 50//?????¡§¦Ì?¡ä??D?¡Â¨ºy¨¢?
#define n_relay_URBAN 10
#else
#define n_sensor_URBAN 50//?????¡§¦Ì?¡ä??D?¡Â¨ºy¨¢?
#define n_relay_URBAN 10
#endif
#define UNIT_URBAN 4
#define UNIT_relay_URBAN 2

#define DIM_HDSN_URBAN (n_sensor_URBAN * UNIT_URBAN + n_relay_URBAN * UNIT_relay_URBAN)
//#define DIM_HDSN_URBAN (n_sensor_URBAN * UNIT_URBAN)
#define HDSNOBJ_URBAN 3

void Fitness_HDSN_URBAN(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_HDSN_URBAN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_HDSN_URBAN(double* x, int nx);

#endif
