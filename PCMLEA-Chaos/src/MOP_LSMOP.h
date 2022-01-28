#ifndef _MOP_LSMOP_
#define _MOP_LSMOP_

//////////////////////////////////////////////////////////////////////////
// LSMOP
void InitPara_LSMOP(char* instName, int numObj, int numVar, int posPara);
void SetLimits_LSMOP(double* minLimit, double* maxLimit, int dim);
int CheckLimits_LSMOP(double* x, int dim);

void LSMOP_initialization(char* prob, int M);
void LSMOP_setLimits(double* lbound, double* ubound);
void LSMOP_fitness(char* prob, double* x, double* f);
void LSMOP_finalization();

void LSMOP1(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP2(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP3(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP4(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP5(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP6(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP7(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP8(double* x, double* fit, double* constrainV, int nx, int M);
void LSMOP9(double* x, double* fit, double* constrainV, int nx, int M);

#endif
