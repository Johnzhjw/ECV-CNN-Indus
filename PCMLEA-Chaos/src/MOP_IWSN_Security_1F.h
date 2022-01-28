#ifndef _MOP_IWSN_SECURITY_1F_
#define _MOP_IWSN_SECURITY_1F_

//////////////////////////////////////////////////////////////////////////
//IWSN_Security_1F - One floor
//#define ADJUST_INDIV_S_1F

#define N_DIREC_S_1F 30
#define N_RELAY_S_1F 15
#define D_DIREC_S_1F 4
#define D_RELAY_S_1F 2
#define NUM_MAX_ROUTE 5
#define LEN_MAX_ROUTE N_RELAY_S_1F

#define DIM_IWSN_S_1F (N_DIREC_S_1F * D_DIREC_S_1F + N_RELAY_S_1F * D_RELAY_S_1F/* + N_DIREC_S_1F * NUM_MAX_ROUTE * LEN_MAX_ROUTE*/)
#define DIM_OBJ_IWSN_S_1F 3

void Fitness_IWSN_S_1F(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_IWSN_S_1F(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_IWSN_S_1F(double* x, int nx);
void AdjustIndiv_whole_IWSN_S_1F(double* individual);
void check_and_repair_IWSN_S_1F(double* individual);
void adjust_constraints_IWSN_S_1F(double cur_iter, double max_iter);

#endif
