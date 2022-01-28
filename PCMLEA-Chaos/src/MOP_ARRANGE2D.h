#ifndef _MOP_ARRANGE_2D_
#define _MOP_ARRANGE_2D_

//////////////////////////////////////////////////////////////////////////
//arrange_2D to maximize the correlation among neighborhood
//474 = 22 * 22 - 10
extern int N_ATTR_ARRANGE2D;// 474
#define N_ROW_ARRANGE2D 22
#define N_COL_ARRANGE2D 22
extern int NDIM_ARRANGE2D;// (N_ATTR_ARRANGE2D)
extern int NOBJ_ARRANGE2D;//  2

void Initialize_data_ARRANGE2D(int curN, int numN);
void Fitness_ARRANGE2D(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_ARRANGE2D(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_ARRANGE2D(double* x, int nx);

#endif
