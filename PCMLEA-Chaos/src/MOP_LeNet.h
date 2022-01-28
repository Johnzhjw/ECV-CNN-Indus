#ifndef _MOP_LENET_H_
#define _MOP_LENET_H_

//////////////////////////////////////////////////////////////////////////
#define LeNet_CNN_Stock 1
#define LeNet_CNN_Indus 2

//#define LeNet_Type 0
//#define LeNet_Type LeNet_CNN_Stock
#define LeNet_Type LeNet_CNN_Indus
///////////////////////////////////
#define OPTIMIZE_STRUCTURE_CNN 0
//#define OPTIMIZE_STRUCTURE_CNN 1

//CNN LeNet
#if LeNet_Type == LeNet_CNN_Stock
//
#define IMG_H              32
#define IMG_W              32
#define MAP_SIZE_C         3
#define MAP_SIZE_S         2
#define NUM_CHANNEL_C1_IN  1
#define NUM_CHANNEL_C1_OUT 6
#define NUM_CHANNEL_C3_IN  NUM_CHANNEL_C1_OUT
#define NUM_CHANNEL_C3_OUT 12
#define IMG_H_END          (((IMG_H-MAP_SIZE_C+1)/2-MAP_SIZE_C+1)/2)
#define IMG_W_END          (((IMG_W-MAP_SIZE_C+1)/2-MAP_SIZE_C+1)/2)
#define NUM_CHANNEL_O5_IN  (NUM_CHANNEL_C3_OUT*IMG_H_END*IMG_W_END)
#define NUM_CHANNEL_O5_OUT 3
#elif LeNet_Type == LeNet_CNN_Indus
//
#define IMG_H              22
#define IMG_W              22
#define MAP_SIZE_C         3
#define MAP_SIZE_S         2
#define NUM_CHANNEL_C1_IN  1
#define NUM_CHANNEL_C1_OUT 6
#define NUM_CHANNEL_C3_IN  NUM_CHANNEL_C1_OUT
#define NUM_CHANNEL_C3_OUT 12
#define IMG_H_END          (((IMG_H-MAP_SIZE_C+1+MAP_SIZE_S-1)/MAP_SIZE_S-MAP_SIZE_C+1+MAP_SIZE_S-1)/MAP_SIZE_S)
#define IMG_W_END          (((IMG_W-MAP_SIZE_C+1+MAP_SIZE_S-1)/MAP_SIZE_S-MAP_SIZE_C+1+MAP_SIZE_S-1)/MAP_SIZE_S)
#define NUM_CHANNEL_O5_IN  (NUM_CHANNEL_C3_OUT*IMG_H_END*IMG_W_END)
#define NUM_CHANNEL_O5_OUT 2
#endif

//////////////////////////////////////////////////////////////////////////
#define NUM_PARA_C1_MAPS (NUM_CHANNEL_C1_IN*NUM_CHANNEL_C1_OUT)
#define NUM_PARA_C1_M (NUM_CHANNEL_C1_IN*NUM_CHANNEL_C1_OUT*(MAP_SIZE_C*MAP_SIZE_C))
#define NUM_PARA_C1_B (NUM_CHANNEL_C1_OUT)
#define NUM_PARA_C1_U (NUM_CHANNEL_C1_IN*(MAP_SIZE_C*MAP_SIZE_C)+1)
#define NUM_PARA_C1   (NUM_PARA_C1_M+NUM_PARA_C1_B)
#define NUM_PARA_C3_MAPS (NUM_CHANNEL_C3_IN*NUM_CHANNEL_C3_OUT)
#define NUM_PARA_C3_M (NUM_CHANNEL_C3_IN*NUM_CHANNEL_C3_OUT*(MAP_SIZE_C*MAP_SIZE_C))
#define NUM_PARA_C3_B (NUM_CHANNEL_C3_OUT)
#define NUM_PARA_C3_U (NUM_CHANNEL_C3_IN*(MAP_SIZE_C*MAP_SIZE_C)+1)
#define NUM_PARA_C3   (NUM_PARA_C3_M+NUM_PARA_C3_B)
#define NUM_PARA_O5_M ((NUM_CHANNEL_O5_IN)*NUM_CHANNEL_O5_OUT)
#define NUM_PARA_O5_B (NUM_CHANNEL_O5_OUT)
#define NUM_PARA_O5   (NUM_PARA_O5_M+NUM_PARA_O5_B)

#define DIM_ALL_PARA_CNN (NUM_PARA_C1+NUM_PARA_C3+NUM_PARA_O5)
#define DIM_ALL_STRU_CNN (NUM_PARA_C1_MAPS+NUM_PARA_C3_MAPS+NUM_PARA_O5_M)

#define MAX_WEIGHT_BIAS_CNN (6.0)

//#define DIM_LeNet (NUM_PARA_C1_MAPS+NUM_PARA_C3_MAPS+NUM_PARA_O5)
#if OPTIMIZE_STRUCTURE_CNN == 0
#define DIM_LeNet DIM_ALL_PARA_CNN
#else
#define DIM_LeNet (DIM_ALL_PARA_CNN + DIM_ALL_STRU_CNN)
#endif

#define OBJ_LeNet 3
extern int NDIM_LeNet_ENSEMBLE;
#define OBJ_LeNet_ENSEMBLE 3

void Initialize_data_LeNet(int curN, int numN, int trainNo, int testNo, int endNo);
void Finalize_LeNet();
void Fitness_LeNet(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_LeNet_test(double* individual, double* fitness);
void Fitness_LeNet_BP(double* fitness);
void SetLimits_LeNet(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_LeNet(double* x, int nx);
void InitiateIndiv_LeNet(double* individual);

//void Initialize_data_LeNet_ensemble(int curN, int numN, int trainNo, int testNo, int endNo);
void load_storedCNN(char* pro);
void SetLimits_LeNet_ensemble(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_LeNet_ensemble(double* x, int nx);
void Finalize_LeNet_ensemble();
void Fitness_LeNet_ensemble(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_LeNet_test_ensemble(double* indivWeight, double* fitness);

#endif
