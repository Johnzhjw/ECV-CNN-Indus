#ifndef _MOP_UF_CF_
#define _MOP_UF_CF_

//////////////////////////////////////////////////////////////////////////
//	UF
void InitPara_UF(char* instName, int numObj, int numVar, int posPara);
void SetLimits_UF(double* minLimit, double* maxLimit, int dim);
int CheckLimits_UF(double* x, int dim);

void UF1(double* x, double* f, double* constrainV, int nx, int nobj);
void UF2(double* x, double* f, double* constrainV, int nx, int nobj);
void UF3(double* x, double* f, double* constrainV, int nx, int nobj);
void UF4(double* x, double* f, double* constrainV, int nx, int nobj);
void UF5(double* x, double* f, double* constrainV, int nx, int nobj);
void UF6(double* x, double* f, double* constrainV, int nx, int nobj);
void UF7(double* x, double* f, double* constrainV, int nx, int nobj);
void UF8(double* x, double* f, double* constrainV, int nx, int nobj);
void UF9(double* x, double* f, double* constrainV, int nx, int nobj);
void UF10(double* x, double* f, double* constrainV, int nx, int nobj);

//////////////////////////////////////////////////////////////////////////
//	CF
void InitPara_CF(char* instName, int numObj, int numVar, int posPara);
void SetLimits_CF(double* minLimit, double* maxLimit, int dim);
int CheckLimits_CF(double* x, int dim);

void CF1(double* x, double* f, double* c, int nx, int nobj);
void CF2(double* x, double* f, double* c, int nx, int nobj);
void CF3(double* x, double* f, double* c, int nx, int nobj);
void CF4(double* x, double* f, double* c, int nx, int nobj);
void CF5(double* x, double* f, double* c, int nx, int nobj);
void CF6(double* x, double* f, double* c, int nx, int nobj);
void CF7(double* x, double* f, double* c, int nx, int nobj);
void CF8(double* x, double* f, double* c, int nx, int nobj);
void CF9(double* x, double* f, double* c, int nx, int nobj);
void CF10(double* x, double* f, double* c, int nx, int nobj);

#endif
