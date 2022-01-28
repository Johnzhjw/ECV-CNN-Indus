#ifndef __MOP_Mob_Sink_
#define __MOP_Mob_Sink_

#include "MOP_FRNN_MODEL.h"

//
#define FRNN_CONSEQUENCE_MOP_Mob_Sink_FIXED       0
#define FRNN_CONSEQUENCE_MOP_Mob_Sink_ADAPT       1

#define FRNN_CONSEQUENCE_MOP_Mob_Sink_CUR FRNN_CONSEQUENCE_MOP_Mob_Sink_ADAPT

//
#define MF_RULE_NUM_MOP_Mob_Sink_LESS 0
#define MF_RULE_NUM_MOP_Mob_Sink_MORE 1

#define MF_RULE_NUM_MOP_Mob_Sink_CUR MF_RULE_NUM_MOP_Mob_Sink_LESS

//
enum TAG_TRAIN_TEST_MOP_Mob_Sink {
    TRAIN_TAG_MOP_Mob_Sink,
    TEST_TAG_MOP_Mob_Sink
};

//
#define PREDICT_CLASSIFY_MOP_Mob_Sink 0
#define STOCK_TRADING_MOP_Mob_Sink 1
#define CURRENT_PROB_MOP_Mob_Sink PREDICT_CLASSIFY_MOP_Mob_Sink

// WSN
#define POS_TYPE_UNIF_MOP_Mob_Sink 0
#define POS_TYPE_GAUSSIAN_MOP_Mob_Sink 1
#define POS_TYPE_HYBRID_MOP_Mob_Sink 2
extern int POS_TYPE_CUR_MOP_Mob_Sink; //POS_TYPE_HYBRID_MOP_Mob_Sink
#define NUM_SENSOR_MOP_Mob_Sink 200
#define IND_X_MOP_Mob_Sink 0
#define IND_Y_MOP_Mob_Sink 1
#define IND_Z_MOP_Mob_Sink 2
#define REGION_W_MOP_Mob_Sink 100
#define REGION_L_MOP_Mob_Sink 100
#define GRID_LEN_MOP_Mob_Sink 6
#define T_MIN_MOP_Mob_Sink 500
#define E_0_MOP_Mob_Sink 50000
#define R_MOP_Mob_Sink 25
#define MAX_NUM_SINK_MOP_Mob_Sink 5
extern int    NUM_SINK_MOP_Mob_Sink;

//////////////////////////////////////////////////////////////////////////
// CFRNN network
typedef struct cnn_MOP_Mob_Sink {
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
    MemberLayer* M4;
    RoughLayer* R5;
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

    int* featureMapTagInitial;
    MY_FLT_TYPE* dataflowInitial;
    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;

    int numParaLocal;
    int numParaLocal_disc;

    int* xType;
} frnn_MOP_Mob_Sink;
//////////////////////////////////////////////////////////////////////////
extern int NDIM_MOP_Mob_Sink;
extern int NOBJ_MOP_Mob_Sink;
extern frnn_MOP_Mob_Sink* frnn_mop_mobile_sink;

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Mob_Sink(char* pro, int curN, int numN, int my_rank);
void SetLimits_MOP_Mob_Sink(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_MOP_Mob_Sink(double* x, int nx);
void Fitness_MOP_Mob_Sink(double* individual, double* fitness, double* constrainV, int nx, int M);
void Fitness_MOP_Mob_Sink_test(double* individual, double* fitness);
void Finalize_MOP_Mob_Sink();

//////////////////////////////////////////////////////////////////////////
void frnn_MOP_Mob_Sink_setup_GEP_only(frnn_MOP_Mob_Sink* frnn);
void frnn_MOP_Mob_Sink_free_GEP_only(frnn_MOP_Mob_Sink* frnn);
void frnn_MOP_Mob_Sink_init_GEP_only(frnn_MOP_Mob_Sink* frnn, double* x, int mode);
void ff_MOP_Mob_Sink_FRNN_GRP_only(frnn_MOP_Mob_Sink* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                                   MY_FLT_TYPE** inputConsequenceNode);
void frnn_MOP_Mob_Sink_setup(frnn_MOP_Mob_Sink* frnn);
void frnn_MOP_Mob_Sink_free(frnn_MOP_Mob_Sink* frnn);
void frnn_MOP_Mob_Sink_init(frnn_MOP_Mob_Sink* frnn, double* x, int mode);
void ff_MOP_Mob_Sink_FRNN(frnn_MOP_Mob_Sink* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                          MY_FLT_TYPE** inputConsequenceNode);
void statistics_MOP_Mob_Sink();

#endif
