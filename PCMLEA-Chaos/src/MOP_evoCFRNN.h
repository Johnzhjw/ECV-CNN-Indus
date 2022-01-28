#ifndef __MOP_evoCFRNN_H_
#define __MOP_evoCFRNN_H_

#include "MOP_FRNN_MODEL.h"

//////////////////////////////////////////////////////////////////////////
#define CFRNN_STRUCTURE_TYPE_0 0
#define CFRNN_STRUCTURE_TYPE_1 1
#define CFRNN_STRUCTURE_TYPE_2 2
#define CFRNN_STRUCTURE_TYPE_CUR CFRNN_STRUCTURE_TYPE_0

#define DATASET_MNIST_MOP_EVO_CFRNN 0
#define DATASET_MedicalInsuranceFraud_MOP_EVO_CFRNN 1
#define DATASET_MOP_EVO_CFRNN_CUR DATASET_MedicalInsuranceFraud_MOP_EVO_CFRNN

//////////////////////////////////////////////////////////////////////////
//
enum TAG_TRAIN_TEST_MOP_evoCFRNN_Classify {
    TRAIN_TAG_MOP_evoCFRNN_Classify,
    TEST_TAG_MOP_evoCFRNN_Classify
};

//////////////////////////////////////////////////////////////////////////
// FRNN network
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
typedef struct cnn_evoCFRNN_classification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;
    int typeCoding;

    int inputChannel;
    int inputHeightMax;
    int inputWidthMax;
    int numValIn;

    int layerNum;
    MemberLayer* M1;
    FuzzyLayer* F2;
    RoughLayer* R3;
    OutReduceLayer* OL;

    int numOutput;  //

    MY_FLT_TYPE* e; // —µ¡∑ŒÛ≤Ó
    MY_FLT_TYPE* L; // À≤ ±ŒÛ≤Óƒ‹¡ø

    MY_FLT_TYPE sum_all;
    MY_FLT_TYPE sum_wrong;

    MY_FLT_TYPE* N_sum;
    MY_FLT_TYPE* N_wrong;
    MY_FLT_TYPE* e_sum;

    MY_FLT_TYPE* N_TP;
    MY_FLT_TYPE* N_TN;
    MY_FLT_TYPE* N_FP;
    MY_FLT_TYPE* N_FN;

    int*** featureMapTagInitial;
    MY_FLT_TYPE*** dataflowInitial;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} CNN_evoCFRNN_C;
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
typedef struct cnn_evoCFRNN_classification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;
    int typeCoding;

    int inputChannel;
    int inputHeightMax;
    int inputWidthMax;
    int numValIn;

    int layerNum;
    ConvolutionLayer* C1;
    PoolLayer* P2;
    ConvolutionLayer* C3;
    PoolLayer* P4;
    Member2DLayer* M5;
    FuzzyLayer* F6;
    RoughLayer* R7;
    OutReduceLayer* OL;

    int numOutput;  //

    MY_FLT_TYPE* e; // —µ¡∑ŒÛ≤Ó
    MY_FLT_TYPE* L; // À≤ ±ŒÛ≤Óƒ‹¡ø

    MY_FLT_TYPE sum_all;
    MY_FLT_TYPE sum_wrong;

    MY_FLT_TYPE* N_sum;
    MY_FLT_TYPE* N_wrong;
    MY_FLT_TYPE* e_sum;

    MY_FLT_TYPE* N_TP;
    MY_FLT_TYPE* N_TN;
    MY_FLT_TYPE* N_FP;
    MY_FLT_TYPE* N_FN;

    int*** featureMapTagInitial;
    MY_FLT_TYPE*** dataflowInitial;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} CNN_evoCFRNN_C;
#else
typedef struct cnn_evoCFRNN_classification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;
    int typeCoding;

    int inputChannel;
    int inputHeightMax;
    int inputWidthMax;
    int numValIn;

    int layerNum;
    ConvolutionLayer* C1;
    PoolLayer* P2;
    ConvolutionLayer* C3;
    PoolLayer* P4;
    InterCPCLayer* I5;
    MemberLayer* M6;
    FuzzyLayer* F7;
    RoughLayer* R8;
    OutReduceLayer* OL;

    int numOutput;  //

    MY_FLT_TYPE* e; // —µ¡∑ŒÛ≤Ó
    MY_FLT_TYPE* L; // À≤ ±ŒÛ≤Óƒ‹¡ø

    MY_FLT_TYPE sum_all;
    MY_FLT_TYPE sum_wrong;

    MY_FLT_TYPE* N_sum;
    MY_FLT_TYPE* N_wrong;
    MY_FLT_TYPE* e_sum;

    MY_FLT_TYPE* N_TP;
    MY_FLT_TYPE* N_TN;
    MY_FLT_TYPE* N_FP;
    MY_FLT_TYPE* N_FN;

    int*** featureMapTagInitial;
    MY_FLT_TYPE*** dataflowInitial;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} CNN_evoCFRNN_C;
#endif
//////////////////////////////////////////////////////////////////////////
extern int NDIM_evoCFRNN_Classify;
extern int NOBJ_evoCFRNN_Classify;
extern CNN_evoCFRNN_C* cnn_evoCFRNN_c;

//////////////////////////////////////////////////////////////////////////
void Initialize_evoCFRNN_Classify(int curN, int numN);
void SetLimits_evoCFRNN_Classify(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_evoCFRNN_Classify(double* x, int nx);
void Fitness_evoCFRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M);
//void Fitness_IntrusionDetection_FRNN_Classify_valid(double* individual, double* fitness);
void Fitness_evoCFRNN_Classify_test(double* individual, double* fitness);
void Finalize_evoCFRNN_Classify();

//////////////////////////////////////////////////////////////////////////
void cnn_evoCFRNN_c_setup(CNN_evoCFRNN_C* cfrnn);
void cnn_evoCFRNN_c_free(CNN_evoCFRNN_C* cfrnn);
void cnn_evoCFRNN_c_init(CNN_evoCFRNN_C* cfrnn, double* x, int mode);
void ff_cnn_evoCFRNN_c(CNN_evoCFRNN_C* cfrnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode);

#endif
