#ifndef _MOP_RECSYS_SMARTCITY_
#define _MOP_RECSYS_SMARTCITY_

//////////////////////////////////////////////////////////////////////////
//RecSys_SmartCity
#define N_USR_RS_SC 164
#define N_LOC_RS_SC  168
#define N_ACT_RS_SC  5
#define N_FEA_RS_SC  14
#define N_IMP_RS_SC  N_ACT_RS_SC
#define N_ELEM_TENSER_RS_SC (N_USR_RS_SC*N_LOC_RS_SC*N_ACT_RS_SC)

#define DIM_RS_SC (1 + N_USR_RS_SC * N_IMP_RS_SC + N_LOC_RS_SC * N_IMP_RS_SC + N_ACT_RS_SC * N_IMP_RS_SC + N_FEA_RS_SC * N_IMP_RS_SC)
#define DIM_OBJ_RS_SC 10

void Initialize_data_RS_SC(int curN, int numN);
void Finalize_data_RS_SC();
void Fitness_RS_SC(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_RS_SC_testSet(double* individual, double* fitness);
void SetLimits_RS_SC(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_RS_SC(double* x, int nx);

#endif
