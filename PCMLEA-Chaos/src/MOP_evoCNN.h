#ifndef __MOP_evoCNN_
#define __MOP_evoCNN_

#include "MOP_FRNN_MODEL.h"

// FRNN network
typedef struct cnn_evoCNN_classification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectWeight;

    int inputHeightMax;
    int inputWidthMax;

    int layerNum;
    ConvolutionLayer* C1;
    PoolLayer* P2;
    ConvolutionLayer* C3;
    PoolLayer* P4;
    InterCPCLayer* O5;

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
} CNN_evoCNN_C;

//////////////////////////////////////////////////////////////////////////
extern int NDIM_evoCNN_Classify;
extern int NOBJ_evoCNN_Classify;
extern CNN_evoCNN_C* cnn_evoCNN_c;

extern int NDIM_evoCNN_Classify_BP;
extern int NOBJ_evoCNN_Classify_BP;

//////////////////////////////////////////////////////////////////////////
void Initialize_evoCNN_Classify(int curN, int numN);
void SetLimits_evoCNN_Classify(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_evoCNN_Classify(double* x, int nx);
void Fitness_evoCNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M);
//void Fitness_IntrusionDetection_FRNN_Classify_valid(double* individual, double* fitness);
void Fitness_evoCNN_Classify_test(double* individual, double* fitness);
void Finalize_evoCNN_Classify();

void Fitness_evoCNN_Classify_BP(double* individual, double* fitness, double* constrainV, int nx, int M);

//////////////////////////////////////////////////////////////////////////
void cnn_evoCNN_c_setup(CNN_evoCNN_C* cnn);
void cnn_evoCNN_c_free(CNN_evoCNN_C* cnn);
void cnn_evoCNN_c_init(CNN_evoCNN_C* cnn, double* x, int mode);
void ff_cnn_evoCNN_c(CNN_evoCNN_C* cnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode);
void bp_cnn_evoCNN_c(CNN_evoCNN_C* cnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* tarOut, MY_FLT_TYPE** inputConsequenceNode,
                     FRNNOpts opts);

#endif
