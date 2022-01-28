#ifndef __MOP_PREDICT_DFRNN_
#define __MOP_PREDICT_DFRNN_

#include "MOP_FRNN_MODEL.h"

//
#define FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_FIXED       0
#define FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_ADAPT       1

#define FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_CUR FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_ADAPT

//
#define MF_RULE_NUM_MOP_PREDICT_DFRNN_LESS 0
#define MF_RULE_NUM_MOP_PREDICT_DFRNN_MORE 1

#define MF_RULE_NUM_MOP_PREDICT_DFRNN_CUR MF_RULE_NUM_MOP_PREDICT_DFRNN_LESS

//
enum TAG_TRAIN_TEST_MOP_PREDICT_DFRNN {
    TRAIN_TAG_MOP_PREDICT_DFRNN,
    VAL_TAG_MOP_PREDICT_DFRNN,
    TEST_TAG_MOP_PREDICT_DFRNN
};
enum TYPE_IN_CNSQ_MOP_PREDICT_DFRNN {
    PRE_OUT_2_IN_CNSQ_MOP_PREDICT_DFRNN,
    ORIG_IN_2_IN_CNSQ_MOP_PREDICT_DFRNN,
    CONCAT_PRE_OUT_ORIG_IN_2_IN_CNSQ_MOP_PREDICT_DFRNN
};
enum TYPE_IN_MOP_PREDICT_DFRNN {
    PRE_OUT_2_IN_MOP_PREDICT_DFRNN,
    CONCAT_PRE_OUT_ORIG_IN_2_IN_MOP_PREDICT_DFRNN
};

//
#define PREDICT_CLASSIFY_MOP_PREDICT_DFRNN 0
#define STOCK_TRADING_MOP_PREDICT_DFRNN 1
#define CURRENT_PROB_MOP_PREDICT_DFRNN PREDICT_CLASSIFY_MOP_PREDICT_DFRNN

//////////////////////////////////////////////////////////////////////////
// CFRNN network
typedef struct strt_frnn_block_Pred_dfrnn {
    int num_multiKindInput;
    int numInput;
    int num_multiKindOutput;
    int numOutput;  //
    int num_GEP;
    codingGEP** GEP0 = NULL;
    MemberLayer* M1 = NULL;
    FuzzyLayer* F2 = NULL;
    RoughLayer* R3 = NULL;
    OutReduceLayer* OL = NULL;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} frnn_block_Pred_dfrnn;
typedef struct cnn_Predict_DFRNN {
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
    int tag_GEPr;
    int GEP_head_len;

    int tag_multiKindInput;
    int num_multiKindInput;
    int num_FRNN;

    int numInput;
    int lenGap;

    int layerNum;
    int layerSize[10];

    frnn_block_Pred_dfrnn** dfrnn = NULL;

    int numRules;
    int numRoughs;

    int tag_multiKindOutput;
    int num_multiKindOutput;

    int numOutput;  //

    MY_FLT_TYPE* e = NULL; // ÑµÁ·Îó²î

    MY_FLT_TYPE sum_all;
    MY_FLT_TYPE sum_wrong;

    MY_FLT_TYPE* N_sum = NULL;
    MY_FLT_TYPE* N_wrong = NULL;
    MY_FLT_TYPE* e_sum = NULL;

    MY_FLT_TYPE* N_TP = NULL;
    MY_FLT_TYPE* N_TN = NULL;
    MY_FLT_TYPE* N_FP = NULL;
    MY_FLT_TYPE* N_FN = NULL;

    MY_FLT_TYPE money_init;
    MY_FLT_TYPE money_in_hand;
    int* trading_actions = NULL;
    int num_stock_held;

    int num_in_all_max;
    int num_out_all_max;
    int num_out_max;
    int num_out_mk_max;
    int num_in_OL_max;

    int* featureMapTagInitial = NULL;
    MY_FLT_TYPE* dataflowInitial = NULL;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;

    int numParaLocal;
    int numParaLocal_disc;

    int* xType = NULL;
} frnn_MOP_Predict_DFRNN;
//////////////////////////////////////////////////////////////////////////
extern int NDIM_MOP_Predict_DFRNN;
extern int NOBJ_MOP_Predict_DFRNN;
extern frnn_MOP_Predict_DFRNN* frnn_MOP_Predict_dfrnn;

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Predict_DFRNN(char* pro, int curN, int numN, int trainNo, int testNo, int endNo, int my_rank);
void SetLimits_MOP_Predict_DFRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_MOP_Predict_DFRNN(double* x, int nx);
void Fitness_MOP_Predict_DFRNN(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_MOP_Predict_DFRNN_test(double* individual, double* fitness);
void Finalize_MOP_Predict_DFRNN();

//////////////////////////////////////////////////////////////////////////
void dfrnn_Predict_DFRNN_setup(frnn_MOP_Predict_DFRNN* dfrnn);
void dfrnn_Predict_DFRNN_free(frnn_MOP_Predict_DFRNN* dfrnn);
void dfrnn_Predict_DFRNN_init(frnn_MOP_Predict_DFRNN* dfrnn, double* x, int mode);
void ff_frnn_Predict_DFRNN(frnn_MOP_Predict_DFRNN* dfrnn, int iL, int iLS, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                           MY_FLT_TYPE* inputConsequenceNode);
void statistics_MOP_Predict_DFRNN(FILE* fpt);

#endif
