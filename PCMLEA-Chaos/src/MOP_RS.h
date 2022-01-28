#ifndef _MOP_RS_
#define _MOP_RS_

//////////////////////////////////////////////////////////////////////////
//Recommender System - FGCS
#define NUM_RECOMMEND 10 //

extern int DIM_RS;
extern int n_M_PR;
extern int n_invalid;

void loadData();
void freeData();

void SetLimitsRS(double* minL, double* maxL, int nx);
int  CheckLimitsRS(double* x, int nx);
void getFitnessRS(double* x, double* y, double* constrainV, int nx, int M);

#endif
