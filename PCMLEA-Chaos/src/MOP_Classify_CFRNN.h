#ifndef __MOP_CLASSIFY_CFRNN_
#define __MOP_CLASSIFY_CFRNN_

#include "MOP_FRNN_MODEL.h"

//
#define DATASET_MNIST_MOP_CLASSIFY_CFRNN 0
#define DATASET_SECOM_MOP_CLASSIFY_CFRNN 1
#define DATASET_MOP_CLASSIFY_CFRNN_CUR DATASET_MNIST_MOP_CLASSIFY_CFRNN

//
#define CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I   0
#define CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II  1
#define CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III 2
#define CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV  3

#define CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III

//
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN       0
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED       1
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG       2
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG   3
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM     4
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM 5
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS       6
#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE                        7

#define CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE

//
#define MF_RULE_NUM_MOP_CLASSIFY_CFRNN_LESS 0
#define MF_RULE_NUM_MOP_CLASSIFY_CFRNN_MORE 1

#define MF_RULE_NUM_MOP_CLASSIFY_CFRNN_CUR MF_RULE_NUM_MOP_CLASSIFY_CFRNN_MORE

//
enum TAG_TRAIN_TEST_MOP_CLASSIFY_CFRNN {
    TRAIN_TAG_MOP_CLASSIFY_CFRNN,
    TEST_TAG_MOP_CLASSIFY_CFRNN
};

//////////////////////////////////////////////////////////////////////////
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
// CFRNN network
typedef struct cnn_Classify_CFRNN {
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

    int layerNum;
    ConvolutionLayer* C1;
    PoolLayer* P2;
    ConvolutionLayer* C3;
    PoolLayer* P4;
    MemberLayer** M5;
    FuzzyLayer** F6;
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

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} cfrnn_Classify_CFRNN;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
// CFRNN network
typedef struct cnn_Classify_CFRNN {
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

    int layerNum;
    ConvolutionLayer* C1;
    PoolLayer* P2;
    ConvolutionLayer* C3;
    PoolLayer* P4;
    FCLayer* OL;

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

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} cfrnn_Classify_CFRNN;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
// CFRNN network
typedef struct cnn_Classify_CFRNN {
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

    int layerNum;
    int regionNum;
    int regionNum_side;
    int* region_row_len;
    int* region_col_len;
    int* region_row_offset;
    int* region_col_offset;
    MemberLayer** M1;
    FuzzyLayer** F2;
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

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} cfrnn_Classify_CFRNN;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
// CFRNN network
typedef struct cnn_Classify_CFRNN {
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

    int layerNum;
    MemberLayer** M1;
    FuzzyLayer** F2;
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

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} cfrnn_Classify_CFRNN;
#endif
//////////////////////////////////////////////////////////////////////////
extern int NDIM_Classify_CFRNN;
extern int NOBJ_Classify_CFRNN;
extern cfrnn_Classify_CFRNN* cfrnn_Classify;

//////////////////////////////////////////////////////////////////////////
void Initialize_Classify_CFRNN(char* pro, int curN, int numN, int my_rank);
void SetLimits_Classify_CFRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_Classify_CFRNN(double* x, int nx);
void Fitness_Classify_CFRNN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_Classify_CFRNN_test(double* individual, double* fitness);
void Finalize_Classify_CFRNN();

//////////////////////////////////////////////////////////////////////////
void cfrnn_Classify_CFRNN_setup(cfrnn_Classify_CFRNN* cfrnn);
void cfrnn_Classify_CFRNN_free(cfrnn_Classify_CFRNN* cfrnn);
void cfrnn_Classify_CFRNN_init(cfrnn_Classify_CFRNN* cfrnn, double* x, int mode);
void ff_cfrnn_Classify_CFRNN(cfrnn_Classify_CFRNN* cfrnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut,
                             MY_FLT_TYPE** inputConsequenceNode);

#endif
