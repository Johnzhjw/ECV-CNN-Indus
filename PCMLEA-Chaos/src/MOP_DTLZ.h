#ifndef _MOP_DTLZ_
#define _MOP_DTLZ_

//////////////////////////////////////////////////////////////////////////
//	DTLZ
void InitPara_DTLZ(char* instName, int numObj, int numVar, int posPara);
void SetLimits_DTLZ(double* minLimit, double* maxLimit, int dim);
int CheckLimits_DTLZ(double* x, int dim);

void dtlz1(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz2(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz3(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz4(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz5(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz6(double* xreal, double* obj, double* constrainV, int dim, int nobj);
void dtlz7(double* xreal, double* obj, double* constrainV, int dim, int nobj);

#endif
