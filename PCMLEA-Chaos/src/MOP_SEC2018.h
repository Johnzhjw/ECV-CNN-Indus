#ifndef _MOP_SEC2018_
#define _MOP_SEC2018_

// Evaluation.cpp : Defines the entry point for the console application.
//
//////////////////////////////////////////////////////////////////////////
// SEC2018 - MaOP
void InitPara_SEC18_MaOP(char* instName, int numObj, int numVar, int posPara);
void SetLimits_SEC18_MaOP(double* minLimit, double* maxLimit, int dim);
int CheckLimits_SEC18_MaOP(double* x, int dim);

void SEC18_MaOP1(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP2(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP3(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP4(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP5(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP6(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP7(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP8(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP9(double* x, double* fit, double* constrainV, int nx, int M);
void SEC18_MaOP10(double* x, double* fit, double* constrainV, int nx, int M);

#endif
