#ifndef _MOP_WFG_
#define _MOP_WFG_

//////////////////////////////////////////////////////////////////////////
//	WFG
void InitPara_WFG(char* instName, int numObj, int numVar, int posPara);
void SetLimits_WFG(double* minLimit, double* maxLimit, int dim);
int CheckLimits_WFG(double* x, int dim);

//void wfg_eval(double* x, int n, int M, char* problem, double* fit);
//void problem_calc_fitness(double* x,
//	const int k, const int M, char* fn, double* fit);

void WFG1(double* x, double* fit, double* constrainV, int nx, int M);
void WFG2(double* x, double* fit, double* constrainV, int nx, int M);
void WFG3(double* x, double* fit, double* constrainV, int nx, int M);
void WFG4(double* x, double* fit, double* constrainV, int nx, int M);
void WFG5(double* x, double* fit, double* constrainV, int nx, int M);
void WFG6(double* x, double* fit, double* constrainV, int nx, int M);
void WFG7(double* x, double* fit, double* constrainV, int nx, int M);
void WFG8(double* x, double* fit, double* constrainV, int nx, int M);
void WFG9(double* x, double* fit, double* constrainV, int nx, int M);

void I1(double* x, double* fit, double* constrainV, int nx, int M);
void I2(double* x, double* fit, double* constrainV, int nx, int M);
void I3(double* x, double* fit, double* constrainV, int nx, int M);
void I4(double* x, double* fit, double* constrainV, int nx, int M);
void I5(double* x, double* fit, double* constrainV, int nx, int M);

#endif
