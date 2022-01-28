#ifndef __MOP_PREDICT_FRNN_
#define __MOP_PREDICT_FRNN_

#include "MOP_FRNN_MODEL.h"

//
#define FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_FIXED       0
#define FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_ADAPT       1

#define FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_CUR FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_ADAPT

//
#define MF_RULE_NUM_MOP_PREDICT_FRNN_LESS 0
#define MF_RULE_NUM_MOP_PREDICT_FRNN_MORE 1

#define MF_RULE_NUM_MOP_PREDICT_FRNN_CUR MF_RULE_NUM_MOP_PREDICT_FRNN_LESS

//
enum TAG_TRAIN_TEST_MOP_PREDICT_FRNN {
    TRAIN_TAG_MOP_PREDICT_FRNN,
    TEST_TAG_MOP_PREDICT_FRNN
};

//
#define PREDICT_CLASSIFY_MOP_PREDICT_FRNN 0
#define STOCK_TRADING_MOP_PREDICT_FRNN 1
#define CURRENT_PROB_MOP_PREDICT_FRNN PREDICT_CLASSIFY_MOP_PREDICT_FRNN

//////////////////////////////////////////////////////////////////////////
// CFRNN network
typedef struct cnn_Predict_FRNN {
    int typeFuzzySet;
    int typeRules;
    int typeInRuleCorNum;
    int typeTypeReducer;
    int consequenceNodeStatus;
    int centroid_num_tag;
    int flagConnectStatus;
    int flagConnectWeight;
    int typeCoding;

    int tag_DIF;
    int tag_GEP;
    int num_GEP;
    int tag_GEPr;
    int GEP_head_len;

    int tag_multiKindInput;
    int num_multiKindInput;
    int num_FRNN;

    int numInput;
    int lenGap;

    int layerNum;
    codingGEP** GEP0;
    MemberLayer* M1;
    FuzzyLayer* F2;
    RoughLayer* R3;
    OutReduceLayer* OL;

    int numRules;
    int numRoughs;

    int tag_multiKindOutput;
    int num_multiKindOutput;

    int numOutput;  //

    MY_FLT_TYPE* e; // ÑµÁ·Îó²î

    MY_FLT_TYPE sum_all;
    MY_FLT_TYPE sum_wrong;

    MY_FLT_TYPE* N_sum;
    MY_FLT_TYPE* N_wrong;
    MY_FLT_TYPE* e_sum;

    MY_FLT_TYPE* N_TP;
    MY_FLT_TYPE* N_TN;
    MY_FLT_TYPE* N_FP;
    MY_FLT_TYPE* N_FN;

    MY_FLT_TYPE money_init;
    MY_FLT_TYPE money_in_hand;
    int* trading_actions;
    int num_stock_held;

    int* featureMapTagInitial;
    MY_FLT_TYPE* dataflowInitial;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} frnn_MOP_Predict_FRNN;
//////////////////////////////////////////////////////////////////////////
extern int NDIM_MOP_Predict_FRNN;
extern int NOBJ_MOP_Predict_FRNN;
extern frnn_MOP_Predict_FRNN* frnn_MOP_Predict;

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Predict_FRNN(char* pro, int curN, int numN, int trainNo, int testNo, int endNo, int my_rank);
void SetLimits_MOP_Predict_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_MOP_Predict_FRNN(double* x, int nx);
void Fitness_MOP_Predict_FRNN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_MOP_Predict_FRNN_test(double* individual, double* fitness);
void Finalize_MOP_Predict_FRNN();

//////////////////////////////////////////////////////////////////////////
void frnn_Predict_FRNN_setup(frnn_MOP_Predict_FRNN* frnn);
void frnn_Predict_FRNN_free(frnn_MOP_Predict_FRNN* frnn);
void frnn_Predict_FRNN_init(frnn_MOP_Predict_FRNN* frnn, double* x, int mode);
void ff_frnn_Predict_FRNN(frnn_MOP_Predict_FRNN* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                          MY_FLT_TYPE** inputConsequenceNode);
void statistics_MOP_Predict_FRNN(FILE* fpt);

#endif
