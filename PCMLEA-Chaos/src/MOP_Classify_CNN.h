#ifndef _MOP_CLASSIFY_CNN_
#define _MOP_CLASSIFY_CNN_

//////////////////////////////////////////////////////////////////////////
//Classify_CNN
extern int NUM_feature_Classify_CNN_Indus;
extern int NDIM_Classify_CNN_Indus;// DIM_LeNet
extern int NOBJ_Classify_CNN_Indus;// 2
//
extern int NDIM_Classify_CNN_Indus_BP;// NDIM_Classify_CNN_Indus
extern int NOBJ_Classify_CNN_Indus_BP;// 6
//
#define FLAG_OFF_Classify_CNN_Indus 0
#define FLAG_ON_Classify_CNN_Indus 1
#define TAG_RAND_Classify_CNN_Indus FLAG_OFF_Classify_CNN_Indus
#define TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus FLAG_ON_Classify_CNN_Indus
//
#define GENERALIZATION_NONE_Classify_CNN_Indus 0
#define GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus 1
#define GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus 2
#define GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus 3
#define GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus 4
#define TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus

#define MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN 0.01
//
void Initialize_data_Classify_CNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Finalize_Classify_CNN();
void Fitness_Classify_CNN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_Classify_CNN_validation(double* individual, double* fitness);
void Fitness_Classify_CNN_test(double* individual, double* fitness);
void Fitness_raw_Classify_CNN(double* individual, double* fitness_raw);
void SetLimits_Classify_CNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_Classify_CNN(double* x, int nx);
//
void Fitness_Classify_CNN_Indus_BP(double* individual, double* fitness, double* constrainV, int nx, int M);

#endif
