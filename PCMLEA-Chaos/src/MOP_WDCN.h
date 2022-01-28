#ifndef _MOP_WDCN_
#define _MOP_WDCN_

//////////////////////////////////////////////////////////////////////////
//WDCN - Wireless Data Center Networks
#define N_RADIO_PER_ROW 20 //20 30 40
#define N_RADIO_PER_COL 20 //20 30 40
#define N_RADIO (N_RADIO_PER_ROW*N_RADIO_PER_COL) //400 900 1600

#define DIM_WDCN N_RADIO
#define WDCNOBJ 3

void Fitness_WDCN(double* individual, double* fitness, double* constrainV, int nx, int M);
void SetLimits_WDCN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_WDCN(double* x, int nx);

#endif
