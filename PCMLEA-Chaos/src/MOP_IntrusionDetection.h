#ifndef __MOP_INTRUSIONDETECTION_
#define __MOP_INTRUSIONDETECTION_

#include "MOP_FRNN_MODEL.h"

// FRNN network
typedef struct frnn_intrusion_detection_classification {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;

    int layerNum;
    MemberLayer* M1;
    FuzzyLayer* F2;
    RoughLayer* R3;
    OutReduceLayer* O4;

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
} FRNN_ID_C;

//////////////////////////////////////////////////////////////////////////
extern int NDIM_IntrusionDetection_FRNN_Classify;
extern int NOBJ_IntrusionDetection_FRNN_Classify;
extern FRNN_ID_C* frnn_id_c;

//////////////////////////////////////////////////////////////////////////
void Initialize_IntrusionDetection_FRNN_Classify(int curN, int numN);
void SetLimits_IntrusionDetection_FRNN_Classify(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_IntrusionDetection_FRNN_Classify(double* x, int nx);
void Fitness_IntrusionDetection_FRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M);
//void Fitness_IntrusionDetection_FRNN_Classify_valid(double* individual, double* fitness);
void Fitness_IntrusionDetection_FRNN_Classify_test(double* individual, double* fitness);
void Finalize_IntrusionDetection_FRNN_Classify();

//////////////////////////////////////////////////////////////////////////
void frnn_id_c_setup(FRNN_ID_C* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                     int* flagAdapMemship,
                     int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax, int typeFuzzySet,
                     int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
                     int consequenceNodeStatus, int centroid_num_tag, int numInputConsequenceNode,
                     int flagConnectStatus, int flagConnectWeight);
void frnn_id_c_free(FRNN_ID_C* frnn);
void frnn_id_c_init(FRNN_ID_C* frnn, double* x, int mode);
void ff_frnn_id_c(FRNN_ID_C* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode);

#endif
