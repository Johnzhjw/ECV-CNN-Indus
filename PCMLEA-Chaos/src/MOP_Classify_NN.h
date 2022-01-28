#ifndef _MOP_CLASSIFY_NN_
#define _MOP_CLASSIFY_NN_

#include "MOP_cnn.h"

//////////////////////////////////////////////////////////////////////////
//Classify_NN
extern int NUM_feature_Classify_NN_Indus;
extern int NDIM_Classify_NN_Indus;// DIM_LeNet
extern int NOBJ_Classify_NN_Indus;// 2
//
extern int NDIM_Classify_NN_Indus_BP;// NDIM_Classify_NN_Indus
extern int NOBJ_Classify_NN_Indus_BP;// 6
//
extern int DIM_ALL_PARA_NN;
extern int DIM_ALL_STRU_NN;
//
extern NN* NN_Classify;
//
#define OPTIMIZE_STRUCTURE_NN 0
//
#define FLAG_OFF_Classify_NN_Indus 0
#define FLAG_ON_Classify_NN_Indus 1
#define TAG_RAND_Classify_NN_Indus FLAG_OFF_Classify_NN_Indus
#define TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_NN_Indus FLAG_ON_Classify_NN_Indus
//
#define GENERALIZATION_NONE_Classify_NN_Indus 0
#define GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_NN_Indus 1
#define GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus 2
#define GENERALIZATION_ONE_INDEPENDENDT_Classify_NN_Indus 3
#define GENERALIZATION_EACH_INDEPENDENDT_Classify_NN_Indus 4
#define TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus

#define MAX_NOISE_LEVEL_MOP_CLASSIFY_NN 0.01
//
void Initialize_data_Classify_NN(int curN, int numN, int trainNo, int testNo, int endNo);
void Finalize_Classify_NN();
void Fitness_Classify_NN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_Classify_NN_validation(double* individual, double* fitness);
void Fitness_Classify_NN_test(double* individual, double* fitness);
void Fitness_raw_Classify_NN(double* individual, double* fitness_raw);
void SetLimits_Classify_NN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_Classify_NN(double* x, int nx);
//
void Fitness_Classify_NN_Indus_BP(double* individual, double* fitness, double* constrainV, int nx, int M);

#endif
