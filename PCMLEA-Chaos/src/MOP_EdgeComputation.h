#ifndef _MOP_EdgeComputation_
#define _MOP_EdgeComputation_

#define Latitude 1
#define Longitude 0
#define Mark_ES 2
#define Real_Distance 3
#define Core_Num 4
#define Load_Rate 5

#define RSU_SET_CARDINALITY_LESS 1500
#define RSU_SET_CARDINALITY_MORE 5995
#define RSU_SET_CARDINALITY_CUR  RSU_SET_CARDINALITY_LESS

#define NDIM_EdgeComputation RSU_SET_CARDINALITY_CUR
#define NOBJ_EdgeComputation 6

#define VAR_THRESHOLD_EdgeComputation 1.0

//////////////////////////////////////////////////////////////////////////
void Initialize_data_EdgeComputation(int curN, int numN);
void InitPara_EdgeComputation(char* instName, int numObj, int numVar, int posPara);
void Fitness_EdgeComputation(double *Per_Rsu_Mark, double *fitness, double *constrainV, int nx, int M);
void SetLimits_EdgeComputation(double* minLimit, double* maxLimit, int dim);
int  CheckLimits_EdgeComputation(double* Per_Rsu_Mark, int nx);
void Finalize_EdgeComputation();

#endif
