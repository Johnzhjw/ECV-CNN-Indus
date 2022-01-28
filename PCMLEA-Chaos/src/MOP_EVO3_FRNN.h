#ifndef _MOP_EVO3_FRNN_
#define _MOP_EVO3_FRNN_

//////////////////////////////////////////////////////////////////////////
//EVO3_FRNN - fuzzy rough neural network
#define N_INPUT_EVO3_FRNN 3
#define N_IN_RSD_EVO3_FRNN (N_INPUT_EVO3_FRNN - 1)
#define N_OUTPUT_EVO3_FRNN 1
#define MAX_NUM_CLUSTER_EVO3_FRNN 3
#define MIN_NUM_CLUSTER_EVO3_FRNN 1
#define MAX_NUM_CLUSTER_MEM_PPARA_EVO3_FRNN 6
#define NUM_CLASS_EVO3_FRNN 3
#define MAX_NUM_FUZZY_RULE_EVO3_FRNN (2 * N_INPUT_EVO3_FRNN + 1)
#define DIM_CLUSTER_BIN_RULE_EVO3_FRNN (MAX_NUM_FUZZY_RULE_EVO3_FRNN * N_INPUT_EVO3_FRNN)
#define DIM_CLUSTER_MEM_EVO3_FRNN (N_INPUT_EVO3_FRNN * MAX_NUM_CLUSTER_EVO3_FRNN)
#define DIM_CLUSTER_MEM_TYPE_EVO3_FRNN DIM_CLUSTER_MEM_EVO3_FRNN
#define DIM_CLUSTER_MEM_PARA_EVO3_FRNN (MAX_NUM_CLUSTER_MEM_PPARA_EVO3_FRNN * DIM_CLUSTER_MEM_EVO3_FRNN)
#define DIM_FUZZY_BIN_ROUGH_EVO3_FRNN (NUM_CLASS_EVO3_FRNN * MAX_NUM_FUZZY_RULE_EVO3_FRNN)
#define DIM_FUZZY_ROUGH_MEM_PARA_EVO3_FRNN (NUM_CLASS_EVO3_FRNN * MAX_NUM_FUZZY_RULE_EVO3_FRNN)

#define DIM_EVO3_FRNN (DIM_CLUSTER_MEM_TYPE_EVO3_FRNN + DIM_CLUSTER_BIN_RULE_EVO3_FRNN + DIM_FUZZY_BIN_ROUGH_EVO3_FRNN + DIM_CLUSTER_MEM_PARA_EVO3_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO3_FRNN + NUM_CLASS_EVO3_FRNN)
#define DIM_OBJ_EVO3_FRNN 2

void Initialize_data_EVO3_FRNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Fitness_EVO3_FRNN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_EVO3_FRNN_testSet(double* individual, double* fitness);
void SetLimits_EVO3_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_EVO3_FRNN(double* x, int nx);

#endif
