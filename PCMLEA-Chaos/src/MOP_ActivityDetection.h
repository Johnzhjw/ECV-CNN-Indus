#ifndef __MOP_ActivityClassification_
#define __MOP_ActivityClassification_

#include "MOP_FRNN_MODEL.h"

//
#define nCol_MHEALTH     24

//
enum TAG_TRAIN_TEST_MOP_ACTIVITY {
    TRAIN_TAG_MOP_ACTIVITY,
    TEST_TAG_MOP_ACTIVITY
};

// FRNN network
typedef struct frnn_ActivityDetectionClassification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;

    int typeCoding;

    int layerNum;
    MemberLayer* M1;
    FuzzyLayer* F2;
    RoughLayer* R3;
    OutReduceLayer* OL;

    int numInput;  //
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

    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} FRNN_ACT_C;

//////////////////////////////////////////////////////////////////////////
extern int NDIM_ActivityDetection_FRNN_Classify;
extern int NOBJ_ActivityDetection_FRNN_Classify;
extern FRNN_ACT_C* frnn_act_c;

//////////////////////////////////////////////////////////////////////////
void Initialize_ActivityDetection_FRNN_Classify(int curN, int numN, int my_rank);
void SetLimits_ActivityDetection_FRNN_Classify(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_ActivityDetection_FRNN_Classify(double* x, int nx);
void Fitness_ActivityDetection_FRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_ActivityDetection_FRNN_Classify_test(double* individual, double* fitness);
void Finalize_ActivityDetection_FRNN_Classify();

//////////////////////////////////////////////////////////////////////////
void frnn_act_c_setup(FRNN_ACT_C* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                      int* flagAdapMemship,
                      int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax, int typeFuzzySet,
                      int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
                      int consequenceNodeStatus, int centroid_num_tag,
                      int numInputConsequenceNode, MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq,
                      int flagConnectStatus, int flagConnectWeight);
void frnn_act_c_free(FRNN_ACT_C* frnn);
void frnn_act_c_init(FRNN_ACT_C* frnn, double* x, int mode);
void ff_frnn_act_c(FRNN_ACT_C* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE inputConsequenceNode[][nCol_MHEALTH]);

#endif
