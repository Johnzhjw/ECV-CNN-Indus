#ifndef _MOP_IWSN_
#define _MOP_IWSN_

//////////////////////////////////////////////////////////////////////////
//IWSN
#define N_DIREC 17
#define N_RELAY 9
#define D_DIREC 5
#define D_RELAY 3
#define N_SENSOR (N_DIREC)//?????¡§¦Ì?¡ä??D?¡Â¨ºy¨¢?

#define DIM_IWSN (N_DIREC * D_DIREC + N_RELAY * D_RELAY)
#define IWSNOBJ 3

extern double avg_dist_SENSOR_IWSN;
extern double LT_std_IWSN;
extern double LT_min_IWSN;

void Fitness_IWSN(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_IWSN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_IWSN(double* x, int nx);

#endif
