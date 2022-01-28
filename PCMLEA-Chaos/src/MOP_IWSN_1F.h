#ifndef _MOP_IWSN_1F_
#define _MOP_IWSN_1F_

//////////////////////////////////////////////////////////////////////////
//IWSN_1F - One floor
#define N_DIREC_1F 180
#define N_RELAY_1F 60
#define D_DIREC_1F 4
#define D_RELAY_1F 2

#define DIM_IWSN_1F (N_DIREC_1F * D_DIREC_1F + N_RELAY_1F * D_RELAY_1F)
#define DIM_OBJ_IWSN_1F 2

void Fitness_IWSN_1F(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_IWSN_1F(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_IWSN_1F(double* x, int nx);

#endif
