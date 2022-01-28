#ifndef __FRNN_MODEL_
#define __FRNN_MODEL_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "MOP_NN_FLT_TYPE.h"

//////////////////////////////////////////////////////////////////////////
#define MAX_NUM_PARA_KERNEL_FUNC_CFRNN_MODEL 4
#define PARA_MAX_KERNEL_FUNC_CFRNN_MODEL 1.0
#define PARA_MIN_KERNEL_FUNC_CFRNN_MODEL (-1.0)
#define PARA_MAX_KERNEL_DATA_CFRNN_MODEL 1.0
#define PARA_MIN_KERNEL_DATA_CFRNN_MODEL (-1.0)
#define MIN_POOL_REGION_HEIGHT_CFRNN_MODEL (2)
#define MAX_POOL_REGION_HEIGHT_CFRNN_MODEL (5)
#define DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL (2)
#define MIN_POOL_REGION_WIDTH_CFRNN_MODEL (2)
#define MAX_POOL_REGION_WIDTH_CFRNN_MODEL (5)
#define DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL (2)
#define MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL (1)
#define MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL (5)
#define DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL (3)
#define MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL (1)
#define MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL (5)
#define DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL (3)
#define PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL (-1)
#define PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL (1)
#define MAX_NUM_LOW_RANK_CFRNN_MODEL (3)
#define PARA_MIN_VAL_CP_CFRNN_MODEL (-2)
#define PARA_MAX_VAL_CP_CFRNN_MODEL (2)
#define PARA_MIN_VAL_GEP_CFRNN_MODEL (-2)
#define PARA_MAX_VAL_GEP_CFRNN_MODEL (2)
//#define LAYER_SKIP_TAG_CFRNN_MODEL 1

//////////////////////////////////////////////////////////////////////////
#define NUM_PARA_MEM_FUNC_I_FRNN_MODEL 6
#define NUM_PARA_MEM_FUNC_II_FRNN_MODEL (NUM_PARA_MEM_FUNC_I_FRNN_MODEL+2)
#define MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL NUM_PARA_MEM_FUNC_II_FRNN_MODEL
#define GEP_HEAD_LENGTH_MAX_FRNN_MODEL 10
#define GEP_MAX_AUG_NUM_FRNN_MODEL 3 // max number of augments to each element function
#define GEP_TAIL_LENGTH_MAX_FRNN_MODEL (GEP_HEAD_LENGTH_MAX_FRNN_MODEL*(GEP_MAX_AUG_NUM_FRNN_MODEL-1)+1)
#define GEP_WEIGHT_NUM_MAX_FRNN_MODEL (2*GEP_HEAD_LENGTH_MAX_FRNN_MODEL)
#define MAX_NUM_PARA_CODING_GEP_FRNN_MODEL (GEP_HEAD_LENGTH_MAX_FRNN_MODEL+GEP_TAIL_LENGTH_MAX_FRNN_MODEL+GEP_WEIGHT_NUM_MAX_FRNN_MODEL)
#define MAX_NUM_GEP_ADF_FRNN_MODEL 10
#define MAX_NUM_GEP_C_ADF_FRNN_MODEL 10

#define PARA_MAX_W_MF_2D_FM_FRNN_MODEL 1.0

#define DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL 50
#define DEFAULT_MEMFUNC_NUM_FRNN_MODEL 3
#define DEFAULT_MEMFUNC_NUM_FRNN_MODEL_MIN 3
#define DEFAULT_MEMFUNC_NUM_FRNN_MODEL_MAX 7

//#define PARA_MAX_MEM_REAL_FRNN_MODEL 120.0
//#define PARA_MAX_MEM_REAL_DIFF_FRNN_MODEL 10.0
#define PARA_MAX_MEM_SIGMA_FRNN_MODEL 50.0
#define PARA_MAX_MEM_BELL_A_FRNN_MODEL 50.0
#define PARA_MAX_MEM_BELL_B_FRNN_MODEL 50.0
#define PARA_MAX_MEM_SIGMOID_FRNN_MODEL 50.0
#define PARA_MAX_MEM_RATIO_FRNN_MODEL 2.0
#define PARA_MAX_W_FRNN_MODEL 1.0
//#define PARA_MAX_BIAS_FRNN_MODEL 120.0

//////////////////////////////////////////////////////////////////////////
typedef struct coding_CANDECOMP_PARAFAC {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int flag_adap_rank;

    int numLowRankMax_CP;
    int numLowRankCur_CP;
    int num_dim_CP;
    int* size_dim_CP;
    MY_FLT_TYPE*** var_CP_4_w;

    MY_FLT_TYPE para_max;
    MY_FLT_TYPE para_min;
    MY_FLT_TYPE para_max_abs;
    MY_FLT_TYPE max_val;
    MY_FLT_TYPE min_val;

    int numParaLocal;
    int numParaLocal_disc;
} codingCP;

typedef struct coding_GeneExpressionProgramming {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int GEP_num_input;
    MY_FLT_TYPE* inputMin;
    MY_FLT_TYPE* inputMax;
    int dim_input;

    MY_FLT_TYPE op_ratio;

    int flag_logic;

    int GEP_head_length;
    int GEP_max_aug_num;
    int GEP_tail_length;
    int GEP_weight_num;
    int flag_GEP_weight;
    int GEP_ADF_num;
    int flag_GEP_ADF;
    int GEP_ADF_head_len;
    int GEP_ADF_tail_len;
    int GEP_C_ADF_num;
    int flag_GEP_C_ADF;
    int GEP_C_ADF_head_len;
    int GEP_C_ADF_tail_len;
    MY_FLT_TYPE* para_coding_GEP;
    int numPara_coding_GEP;

    int         check_level[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_parent_ind[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_children_num[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_vInd[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_op[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_valR[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL][2];
    MY_FLT_TYPE check_para[MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_weights[GEP_WEIGHT_NUM_MAX_FRNN_MODEL];
    //
    int         check_ADF_level[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_ADF_parent_ind[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_ADF_children_num[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_ADF_vInd[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_ADF_op[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_ADF_valR[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL][2];
    MY_FLT_TYPE check_ADF_para[MAX_NUM_GEP_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_ADF_weights[MAX_NUM_GEP_ADF_FRNN_MODEL][GEP_WEIGHT_NUM_MAX_FRNN_MODEL];
    //
    int         check_C_ADF_level[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_C_ADF_parent_ind[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_C_ADF_children_num[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_C_ADF_vInd[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    int         check_C_ADF_op[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_C_ADF_valR[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL][2];
    MY_FLT_TYPE check_C_ADF_para[MAX_NUM_GEP_C_ADF_FRNN_MODEL][MAX_NUM_PARA_CODING_GEP_FRNN_MODEL];
    MY_FLT_TYPE check_C_ADF_weights[MAX_NUM_GEP_C_ADF_FRNN_MODEL][GEP_WEIGHT_NUM_MAX_FRNN_MODEL];

    int check_head = 0;
    int check_tail = 1;

    int numParaLocal;
    int numParaLocal_disc;
} codingGEP;
//////////////////////////////////////////////////////////////////////////
// convolutional layer
typedef struct conv_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int* inputHeight;
    int* inputWidth;
    int inputHeightMax;
    int inputWidthMax;

    int channelsIn;
    int channelsOut;
    int channelsInMax;
    int channelsOutMax;

    int typeKernelCoding;

    int flag_kernelCoding_CP;
    codingCP* cdCP;

    int flag_kernelCoding_GEP;
    codingGEP* cdGEP;

    int flag_adapKernelSize;
    int kernelHeightDefault;
    int kernelWidthDefault;
    int kernelHeightMin;
    int kernelHeightMax;
    int kernelWidthMin;
    int kernelWidthMax;
    int** kernelHeight;
    int** kernelWidth;
    int flag_kernelFlagAdap;
    int kernelFlagDefault;
    int** kernelFlag;
    int** kernelFlagCountAll;
    int kernelFlagCount;
    int* kernelType;
    int flag_actFuncTypeAdap;
    int kernelTypeDefault;
    MY_FLT_TYPE**** kernelData;
    MY_FLT_TYPE**** kernelDelta;

    int flag_paddingTypeAdap;
    int paddingTypeDefault;
    int** paddingType;

    MY_FLT_TYPE* biasData;
    MY_FLT_TYPE* biasDelta;

    int* featureMapHeight;
    int* featureMapWidth;
    int featureMapHeightMax;
    int featureMapWidthMax;
    MY_FLT_TYPE*** featureMapData;
    MY_FLT_TYPE*** featureMapDelta;
    MY_FLT_TYPE*** featureMapDerivative;
    int*** featureMapTag;

    MY_FLT_TYPE*** dataflowStatus;

    int numParaLocal;
    int numParaLocal_disc;
} ConvolutionLayer;
// pooling layer
typedef struct pool_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int* inputHeight;
    int* inputWidth;
    int inputHeightMax;
    int inputWidthMax;

    int channelsInOut;
    int channelsInOutMax;

    int flag_poolSizeAdap;
    int poolHeightDefault;
    int poolWidthDefault;
    int poolHeightMin;
    int poolHeightMax;
    int poolWidthMin;
    int poolWidthMax;
    int poolHeightAll;
    int poolWidthAll;
    int* poolHeight;
    int* poolWidth;
    int* poolFlag;

    int flag_poolTypeAdap;
    int poolTypeDefault;
    int* poolType;

    int* featureMapHeight;
    int* featureMapWidth;
    int featureMapHeightMax;
    int featureMapWidthMax;
    MY_FLT_TYPE*** featureMapData;
    MY_FLT_TYPE*** featureMapDelta;
    MY_FLT_TYPE*** featureMapDerivative;
    int*** featureMapPos;
    int*** featureMapTag;

    MY_FLT_TYPE*** dataflowStatus;

    int numParaLocal;
    int numParaLocal_disc;
} PoolLayer;
//  full connection sets
typedef struct full_connection_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int numInput;
    int numOutput;
    int numInputMax;
    int numOutputMax;

    int flagActFunc;
    int flag_actFuncTypeAdap;
    int actFuncTypeDefault;
    int* actFuncType;

    int flag_connectAdap;
    int** connectStatus;
    MY_FLT_TYPE** connectWeight;
    MY_FLT_TYPE** connectWtDelta;
    int* connectCountAll;
    int connectCount;

    MY_FLT_TYPE* biasData;
    MY_FLT_TYPE* biasDelta;

    int numOutputCur;
    MY_FLT_TYPE* outputData;
    MY_FLT_TYPE* outputDelta;
    MY_FLT_TYPE* outputDerivative;

    MY_FLT_TYPE* dataflowStatus;

    int numParaLocal;
    int numParaLocal_disc;
} FCLayer;
// MLP network
typedef struct mlp_network {
    int layerNum;
    FCLayer** LayersPnt;

    int* numNodesAll;  //
    int numOutput;

    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;
} MLP_mine;
//
typedef struct intermidiate_C_FC_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int numOutput;

    int preFeatureMapChannels;
    int preFeatureMapHeightMax;
    int preFeatureMapWidthMax;
    int* preInputHeight;
    int* preInputWidth;

    int flagActFunc;
    int flag_actFuncTypeAdap;
    int actFuncTypeDefault;
    int* actFuncType;

    int flag_connectAdap;

    int typeConnectParaCoding;

    int flag_NN4Para;
    MLP_mine* NN4Para;

    int flag_connectCoding_CP; // tensor decomposition
    codingCP* cdCP_w;
    codingCP* cdCP_c;

    int flag_connectCoding_GEP; // tensor decomposition
    codingGEP* cdGEP_w;
    codingGEP* cdGEP_c;

    int flag_wt_positive;

    int flag_normalize_outData;

    int**** connectStatusAll;
    MY_FLT_TYPE**** connectWeightAll;
    MY_FLT_TYPE**** connectWtDeltaAll;
    int* connectCountAll;
    int connectCountSum;

    MY_FLT_TYPE* biasData;
    MY_FLT_TYPE* biasDelta;

    MY_FLT_TYPE* outputData;
    MY_FLT_TYPE* outputDelta;
    MY_FLT_TYPE* outputDerivative;

    MY_FLT_TYPE* dataflowStatus;

    int numParaLocal;
    int numParaLocal_disc;
} InterCPCLayer;

//////////////////////////////////////////////////////////////////////////
// fuzzy layer
typedef struct member_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int typeFuzzySet; // Type1 or Type2
    int dim_degree;

    int numInput;   //
    MY_FLT_TYPE* valInput;
    MY_FLT_TYPE* inputMin;
    MY_FLT_TYPE* inputMax;

    int typeMFCoding;

    int flag_MFCoding_CP;
    codingCP* cdCP;

    int flag_MFCoding_GEP;
    codingGEP* cdGEP;

    int numParaMembershipFun;
    int* numMembershipFunCur;
    int* numMembershipFun;
    int* flag_adapMembershipFun; // continuous or discrete
    int** typeMembershipFun;
    MY_FLT_TYPE*** paraMembershipFun;
    MY_FLT_TYPE*** degreeMembership;

    MY_FLT_TYPE** dataflowStatus;

    int outputSize;

    int numParaLocal;
    int numParaLocal_disc;
} MemberLayer;
typedef struct member_2D_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int typeFuzzySet; // Type1 or Type2
    int dim_degree;

    int numInput;   //

    int typeMFFeatMapCoding;

    int flag_MFFeatMapCoding_CP;
    codingCP* cdCP;

    int flag_MFFeatMapCoding_GEP;
    codingGEP* cdGEP;

    int preFeatureMapHeightMax;
    int preFeatureMapWidthMax;
    MY_FLT_TYPE**** mat_MFFeatMap;
    MY_FLT_TYPE** norm_mat_MFFeatMap;
    MY_FLT_TYPE* mean_featureMapDataIn;

    MY_FLT_TYPE*** para_MF_II_ratios;

    int flag_typeMembershipFunAdap;
    int typeMembershipFunDefault;
    int* numMembershipFunCur;
    int* numMembershipFun;
    int* flag_adapMembershipFun; // continuous or discrete
    int** typeMembershipFun;  // similarity type
    MY_FLT_TYPE*** degreeMembership;

    MY_FLT_TYPE** dataflowStatus;

    int outputSize;

    int numParaLocal;
    int numParaLocal_disc;
} Member2DLayer;
//  fuzzy rules
typedef struct fuzzy_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int typeFuzzySet;
    int dim_degree;
    int typeRules; // prod or min
    int typeInRuleCorNum;

    int numInput;
    int* numMembershipFunCur;
    int* numMembershipFun;
    int numRules;

    int tag_GEP_rule;
    int numMembershipAll;
    int ruleGEP_numInput;
    codingGEP** ruleGEP;

    int typeConnectCoding;

    int flag_connectCoding_CP;
    codingCP* cdCP;

    int flag_connectCoding_GEP;
    codingGEP* cdGEP;

    int*** connectStatusAll;
    MY_FLT_TYPE* dataflowStatus;

    MY_FLT_TYPE*  degreeMembs;
    MY_FLT_TYPE** degreeRules;

    int numParaLocal;
    int numParaLocal_disc;
} FuzzyLayer;
//  rough sets
typedef struct rough_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int typeFuzzySet;
    int dim_degree;

    int numInput;
    int numRoughSets;

    int typeConnectCoding;

    int flag_connectCoding_CP;
    codingCP* cdCP;

    int flag_connectCoding_GEP;
    codingGEP* cdGEP;

    int flagConnectStatusAdap;
    int** connectStatus;
    MY_FLT_TYPE** connectWeight;
    MY_FLT_TYPE* dataflowStatus;

    MY_FLT_TYPE** degreeRough;

    int numParaLocal;
    int numParaLocal_disc;
} RoughLayer;
// output
typedef struct out_reduce_layer {
    MY_FLT_TYPE* xMin;
    MY_FLT_TYPE* xMax;
    int* xType;

    int typeFuzzySet;
    int dim_degree;
    int typeTypeReducer; // Nie-Tan or Center of sets

    int numInput;
    int numOutput;
    MY_FLT_TYPE* outputMin;
    MY_FLT_TYPE* outputMax;

    int typeConnectCoding;

    int flag_connectCoding_CP;
    codingCP* cdCP_cw;
    int flag_consqCoding_CP;
    codingCP* cdCP_cq;

    int flag_connectCoding_GEP;
    codingGEP* cdGEP_cw;
    int flag_consqCoding_GEP;
    codingGEP* cdGEP_cq;

    int flagConnectStatusAdap;
    int flagConnectWeightAdap;
    int** connectStatus;
    MY_FLT_TYPE** connectWeight;
    MY_FLT_TYPE* dataflowStatus;

    int consequenceNodeStatus; // NO or FIXED or ADAPTIVE
    int centroid_num_tag; // all use one set, or, one set for each
    int numInputConsequenceNode;
    MY_FLT_TYPE** inputConsequenceNode;
    MY_FLT_TYPE**** paraConsequenceNode;
    MY_FLT_TYPE*** centroidsRough;
    MY_FLT_TYPE* inputMin_cnsq;
    MY_FLT_TYPE* inputMax_cnsq;

    MY_FLT_TYPE** valInputFinal;
    MY_FLT_TYPE* valOutputFinal;

    int numParaLocal;
    int numParaLocal_disc;
} OutReduceLayer;
// FRNN network
typedef struct frnn_network {
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

    MY_FLT_TYPE* e; // 训练误差
    MY_FLT_TYPE* L; // 瞬时误差能量

    MY_FLT_TYPE* N_sum;
    MY_FLT_TYPE* N_wrong;
    MY_FLT_TYPE* e_sum;

    MY_FLT_TYPE* N_TP;
    MY_FLT_TYPE* N_TN;
    MY_FLT_TYPE* N_FP;
    MY_FLT_TYPE* N_FN;

    MY_FLT_TYPE dataflowMax;
    MY_FLT_TYPE connectionMax;

    int numParaLocal;
    int numParaLocal_disc;
} FRNN;

//////////////////////////////////////////////////////////////////////////
typedef struct train_opts_frnn {
    int numepochs; // 训练的迭代次数
    int curepochs;
    MY_FLT_TYPE alpha; // 学习速率
    int batch_size;
    int all_sample_num;
    int cur_sample_num;
    int tag_init;
    int tag_update;
} FRNNOpts;

//////////////////////////////////////////////////////////////////////////
// ENUMs
enum LIST_LAYER_TYPES {
    NN_LAYER_CONV,
    NN_LAYER_POOL,
    NN_LAYER_NORMAL_FC,
    NN_LAYER_INTER_C_FC,
    NN_LAYER_FNN_MF,
    NN_LAYER_FNN_MF_2D,
    NN_LAYER_FNN_FUZZY_RULE,
    NN_LAYER_FNN_ROUGH,
    NN_LAYER_FNN_REDUCE_OUT,
    NN_LAYER_TYPE_NUM
};
enum CONV_ACT_FUNC_TYPE {
    ACT_FUNC_RELU,
    ACT_FUNC_LEAKYRELU,
    NUM_ACT_FUNC_TYPE,
    ACT_FUNC_SIGMA,
    ACT_FUNC_TANH,
    ACT_FUNC_ELU
};
enum CONV_KERNEL_FUNC_TYPE {
    KERNEL_FUNC_SIN,
    NUM_KERNEL_FUNC
};
enum CONV_PADDING_TYPE {
    PADDING_SAME,
    PADDING_VALID,
    NUM_PADDING_TYPE
};
enum POOL_KERNEL_TYPE {
    POOL_AVE,
    POOL_MAX,
    NUM_POOL_TYPE,
    POOL_MIN
};
enum FRNN_INIT_MODE {
    INIT_MODE_FRNN,
    INIT_BP_MODE_FRNN,
    ASSIGN_MODE_FRNN,
    OUTPUT_ALL_MODE_FRNN,
    OUTPUT_DISCRETE_MODE_FRNN,
    OUTPUT_CONTINUOUS_MODE_FRNN
};
enum FUZZY_SET_TYPE {
    FUZZY_SET_I,
    FUZZY_INTERVAL_TYPE_II
};
enum MEMBERSHIP_FUNC_TYPE {
    GAUSSIAN_MEM_FUNC,
    SIGMOID_MEM_FUNC,
    GAUSSIAN_COMB_MEM_FUNC,
    NUM_MEM_FUNC
};
enum GEP_COMPN_FUNC_TYPE {
    GEP_OP_F_SUBTRACT,
    GEP_OP_F_DIVIDE,
    GEP_OP_F_ADD,
    GEP_OP_F_MULTIPLY,
    GEP_OP_F_MAX,
    GEP_OP_F_MIN,
    GEP_OP_F_MEAN,
    GEP_OP_F_NUM,
    GEP_OP_F_SIN,
    GEP_OP_F_COS,
    GEP_OP_F_SQUARE,
    GEP_OP_F_SQUARE_ROOT,
    GEP_OP_F_LOG,
    GEP_OP_F_EXP
};
enum GEP_RULE_FUNC_TYPE {
    GEP_R_F_AND,
    GEP_R_F_OR,
    GEP_R_F_NOT,
    GEP_R_F_SQUARE,
    GEP_R_F_SQUARE_ROOT,
    GEP_R_F_NUM
};
enum MEMBERSHIP_FUNC_2D_COMPN_TERMINAL_TYPE {
    MF_2D_T_X = GEP_OP_F_NUM,
    MF_2D_T_Y = GEP_OP_F_NUM + 1,
    MF_2D_T_NUM = 2
};
enum PARA_MF_WEIGHT_CODING_TYPE {
    PARA_CODING_DIRECT,
    PARA_CODING_NN,
    PARA_CODING_CANDECOMP_PARAFAC, // tensor decomposition
    PARA_CODING_GEP,
    PARA_CODING_NUM
};
enum KERNEL_FLAG_TYPE {
    KERNEL_FLAG_SKIP,
    KERNEL_FLAG_OPERATE,
    KERNEL_FLAG_COPY
};
enum MEMBERSHIP_FUNC_2D_MAT_SIMILARITY_TYPE {
    MAT_SIMILARITY_T_COS,
    //MAT_SIMILARITY_T_ACOS,
    MAT_SIMILARITY_T_Norm2,
    MAT_SIMILARITY_T_NUM
};
enum FUZZY_RULE_TYPE {
    PRODUCT_INFERENCE_ENGINE,
    MINIMUM_INFERENCE_ENGINE
};
enum CONDITION_IN_RULE_NUM_CORRESPONDANCE {
    ONE_EACH_IN_TO_ONE_RULE,
    MUL_EACH_IN_TO_ONE_RULE
};
enum FUZZY_TYPE_REDUCER_TYPE {
    NIE_TAN_TYPE_REDUCER,
    CENTER_OF_SETS_TYPE_REDUCER,
};
enum FLAG_STATUS_TAG {
    FLAG_STATUS_OFF,
    FLAG_STATUS_ON
};
enum ROUGH_CENTROID_TYPE {
    NO_CONSEQUENCE_CENTROID,
    FIXED_CONSEQUENCE_CENTROID,
    ADAPTIVE_CONSEQUENCE_CENTROID
};
enum NUM_CENTROID_ALL_ONE_OR_ONE_EACH {
    CENTROID_ALL_ONESET,
    CENTROID_ONESET_EACH
};
enum VAR_TYPE_TAG {
    VAR_TYPE_CONTINUOUS,
    VAR_TYPE_DISCRETE,
    VAR_TYPE_BINARY
};

//////////////////////////////////////////////////////////////////////////
void frnnsetup(FRNN* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship, int* flagAdapMemship,
               int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
               int typeFuzzySet, int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
               int consequenceNodeStatus, int centroid_num_tag, int numInputConsequenceNode,
               MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq,
               int flagConnectStatus, int flagConnectWeight);
void frnnfree(FRNN* frnn);
void frnninit(FRNN* frnn, double* x, int mode);
void ff_frnn(FRNN* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
codingCP* setupCodingCP(int flag_adap_rank, int num_dim_CP, int* size_dim_CP, int numLowRankMax,
                        MY_FLT_TYPE para_min, MY_FLT_TYPE para_max);
void assignCodingCP(codingCP* cdCP, double* x, int mode);
void resetCodingCP(codingCP* cdCP);
void freeCodingCP(codingCP* cdCP);
codingGEP* setupCodingGEP(int GEP_num_input, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int dim_input, MY_FLT_TYPE op_ratio,
                          int flag_logic,
                          int GEP_head_length, int flag_GEP_weight, MY_FLT_TYPE para_min, MY_FLT_TYPE para_max);
void assignCodingGEP(codingGEP* cdGEP, double* x, int mode);
void resetCodingGEP(codingGEP* cdGEP);
void freeCodingGEP(codingGEP* cdGEP);

//////////////////////////////////////////////////////////////////////////
void print_para_codingCP(codingCP* cdCP);
void print_para_codingGEP(codingGEP* cdGEP);
//void print_para_codingGEPR(codingGEP* cdGEP);

//////////////////////////////////////////////////////////////////////////
//inline void decodingCP(codingCP* cdCP, int* vec_input, MY_FLT_TYPE* vec_output);
inline void decodingCP(codingCP* cdCP, int* vec_input, MY_FLT_TYPE* vec_output)
{
    MY_FLT_TYPE tmp_w = 0;
    for(int m = 0; m < cdCP->numLowRankCur_CP; m++) {
        MY_FLT_TYPE tmp_mul = 1;
        for(int n = 0; n < cdCP->num_dim_CP; n++) {
            tmp_mul *= cdCP->var_CP_4_w[n][vec_input[n]][m];
        }
        tmp_w += tmp_mul;
    }
    vec_output[0] = tmp_w;

    return;
}
inline void getConnectGEP(codingGEP* cdGEP)
{
    int len_F = 0;
    if(cdGEP->flag_logic == FLAG_STATUS_OFF)
        len_F = GEP_OP_F_NUM;
    else
        len_F = GEP_R_F_NUM;
    for(int m = 0; m < MAX_NUM_PARA_CODING_GEP_FRNN_MODEL; m++) cdGEP->check_vInd[m] = -1;
    for(int m = 0;
        m < cdGEP->GEP_head_length && m < cdGEP->check_tail;
        m++) {
        int tmp = (int)(len_F * cdGEP->check_para[m] / cdGEP->op_ratio);
        if(tmp >= len_F) {
            int tmp_ind = (int)(cdGEP->GEP_num_input * (cdGEP->check_para[m] - cdGEP->op_ratio) / (1 - cdGEP->op_ratio + 1e-6));
            if(tmp_ind >= 0 && tmp_ind < cdGEP->GEP_num_input) {
                cdGEP->check_vInd[m] = tmp_ind;
            } else {
                printf("%s(%d): para error - %d - %d, exiting...\n",
                       __FILE__, __LINE__, m, tmp_ind);
                exit(-9466981);
            }
        }
    }
    for(int m = cdGEP->GEP_head_length;
        m < cdGEP->GEP_head_length + cdGEP->GEP_tail_length &&
        m < cdGEP->check_tail;
        m++) {
        int tmp_ind = (int)cdGEP->check_para[m];
        if(tmp_ind >= 0 && tmp_ind < cdGEP->GEP_num_input) {
            cdGEP->check_vInd[m] = tmp_ind;
        } else {
            printf("%s(%d): para error - %d, exiting...\n",
                   __FILE__, __LINE__, tmp_ind);
            exit(-285612);
        }
    }
    //
    return;
}
inline void decodingGEP(codingGEP* cdGEP, MY_FLT_TYPE* vec_input, MY_FLT_TYPE* vec_output)
{
    int cur_op_num;
    if(cdGEP->flag_logic == FLAG_STATUS_OFF)
        cur_op_num = GEP_OP_F_NUM;
    else
        cur_op_num = GEP_R_F_NUM;
    int tmp_weight_ind = cdGEP->GEP_weight_num - 1;
    for(int m = 0;
        m < cdGEP->GEP_head_length && m < cdGEP->check_tail;
        m++) {
        int tmp = (int)(cur_op_num * cdGEP->check_para[m] / cdGEP->op_ratio);
        if(tmp >= cur_op_num) {
            int tmp_ind = (int)(cdGEP->GEP_num_input * (cdGEP->check_para[m] - cdGEP->op_ratio) / (1 - cdGEP->op_ratio + 1e-6));
            if(tmp_ind >= 0 && tmp_ind < cdGEP->GEP_num_input) {
                for(int i = 0; i < cdGEP->dim_input; i++)
                    cdGEP->check_valR[m][i] = (vec_input[tmp_ind * cdGEP->dim_input + i] - cdGEP->inputMin[tmp_ind]) /
                                              (cdGEP->inputMax[tmp_ind] - cdGEP->inputMin[tmp_ind]);
            } else {
                printf("%s(%d): para error - %d - %d, exiting...\n",
                       __FILE__, __LINE__, tmp, tmp_ind);
                exit(-1);
            }
        }
    }
    for(int m = cdGEP->GEP_head_length;
        m < cdGEP->GEP_head_length + cdGEP->GEP_tail_length &&
        m < cdGEP->check_tail;
        m++) {
        int tmp_ind = (int)cdGEP->check_para[m];
        if(tmp_ind >= 0 && tmp_ind < cdGEP->GEP_num_input) {
            for(int i = 0; i < cdGEP->dim_input; i++)
                cdGEP->check_valR[m][i] = (vec_input[tmp_ind * cdGEP->dim_input + i] - cdGEP->inputMin[tmp_ind]) /
                                          (cdGEP->inputMax[tmp_ind] - cdGEP->inputMin[tmp_ind]);
        } else {
            printf("%s(%d): para error - %d, exiting...\n",
                   __FILE__, __LINE__, tmp_ind);
            exit(-1);
        }
    }
    int cur_ind = cdGEP->check_tail - 1;
    int cur_level = cdGEP->check_level[cur_ind];
    if(cdGEP->flag_logic == FLAG_STATUS_OFF) {
        while(cur_level > 0) {
            int cur_parent = cdGEP->check_parent_ind[cur_ind];
            int cur_int = (int)(GEP_OP_F_NUM * cdGEP->check_para[cur_parent] / cdGEP->op_ratio);
            MY_FLT_TYPE aug1 = 0;
            MY_FLT_TYPE aug2 = 0;
            if(cur_int == GEP_OP_F_ADD) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 + aug2;
                }
            } else if(cur_int == GEP_OP_F_SUBTRACT) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 - aug2;
                }
            } else if(cur_int == GEP_OP_F_MULTIPLY) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 * aug2;
                }
            } else if(cur_int == GEP_OP_F_DIVIDE) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    if(aug2 > FLT_EPSILON || aug2 < -FLT_EPSILON)
                        cdGEP->check_valR[cur_parent][i] = aug1 / aug2;
                    else if(aug2 > 0)
                        cdGEP->check_valR[cur_parent][i] = aug1;
                    else
                        cdGEP->check_valR[cur_parent][i] = -aug1;
                }
            } else if(cur_int == GEP_OP_F_MAX) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 > aug2 ? aug1 : aug2;
                }
            } else if(cur_int == GEP_OP_F_MIN) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                }
            } else if(cur_int == GEP_OP_F_MEAN) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = (aug1 + aug2) / 2;
                }
            } else if(cur_int == GEP_OP_F_SIN) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = sin(aug2);
                }
            } else if(cur_int == GEP_OP_F_COS) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = cos(aug2);
                }
            } else if(cur_int == GEP_OP_F_EXP) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = exp(aug2);
                }
            } else if(cur_int == GEP_OP_F_SQUARE) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug2 * aug2;
                }
            } else if(cur_int == GEP_OP_F_SQUARE_ROOT) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = sqrt(fabs(aug2));
                }
            } else if(cur_int == GEP_OP_F_LOG) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    if(aug2 > FLT_EPSILON)
                        cdGEP->check_valR[cur_parent][i] = log(aug2);
                    else if(aug2 <= FLT_EPSILON && aug2 >= -FLT_EPSILON)
                        cdGEP->check_valR[cur_parent][i] = 0;// log(FLT_EPSILON);
                    else
                        cdGEP->check_valR[cur_parent][i] = log(-aug2);
                }
            } else {
                printf("%s(%d): Unknown GEP_RULE_FUNC_TYPE - %d, exiting...\n",
                       __FILE__, __LINE__, cur_int);
                exit(-1);
            }
            int cur_children_num = cdGEP->check_children_num[cur_parent];
            cur_ind -= cur_children_num;
            cur_level = cdGEP->check_level[cur_ind];
            tmp_weight_ind -= cur_children_num;
        }
    } else {
        while(cur_level > 0) {
            int cur_parent = cdGEP->check_parent_ind[cur_ind];
            int cur_int = (int)(GEP_R_F_NUM * cdGEP->check_para[cur_parent] / cdGEP->op_ratio);
            MY_FLT_TYPE aug1 = 0;
            MY_FLT_TYPE aug2 = 0;
            if(cur_int == GEP_R_F_AND) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 < aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = aug1 * aug2;
                }
            } else if(cur_int == GEP_R_F_OR) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug1 = cdGEP->check_valR[cur_ind - 1][i] * cdGEP->check_weights[tmp_weight_ind - 1];
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    //cdGEP->check_valR[cur_parent][i] = aug1 > aug2 ? aug1 : aug2;
                    cdGEP->check_valR[cur_parent][i] = 1 - ((1 - aug1) * (1 - aug2));
                }
            } else if(cur_int == GEP_R_F_NOT) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    cdGEP->check_valR[cur_parent][i] = 1 - aug2;
                }
            } else if(cur_int == GEP_R_F_SQUARE) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    cdGEP->check_valR[cur_parent][i] = aug2 * aug2;
                }
            } else if(cur_int == GEP_R_F_SQUARE_ROOT) {
                for(int i = 0; i < cdGEP->dim_input; i++) {
                    aug2 = cdGEP->check_valR[cur_ind][i] * cdGEP->check_weights[tmp_weight_ind];
                    cdGEP->check_valR[cur_parent][i] = sqrt(aug2);
                }
            } else {
                printf("%s(%d): Unknown GEP_RULE_FUNC_TYPE - %d, exiting...\n",
                       __FILE__, __LINE__, cur_int);
                exit(-1);
            }
            int cur_children_num = cdGEP->check_children_num[cur_parent];
            cur_ind -= cur_children_num;
            cur_level = cdGEP->check_level[cur_ind];
            tmp_weight_ind -= cur_children_num;
        }
    }
    for(int i = 0; i < cdGEP->dim_input; i++) {
        vec_output[i] = cdGEP->check_valR[0][i];
    }
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
ConvolutionLayer* setupConvLayer(int inputHeightMax, int inputWidthMax, int channelsIn, int channelsOut, int channelsInMax,
                                 int typeKernelCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight,
                                 int flag_adapKernelSize, int default_kernelHeight, int default_kernelWidth,
                                 int kernelHeightMin, int kernelHeightMax, int kernelWidthMin, int kernelWidthMax,
                                 int flag_kernelFlagAdap, int default_kernelFlag,
                                 int flag_actFuncTypeAdap, int default_actFuncType,
                                 int flag_paddingTypeAdap, int default_paddingType);
void assignConvLayer(ConvolutionLayer* cLayer, double* x, int mode);
void resetConvLayer(ConvolutionLayer* cLayer);
void freeConvLayer(ConvolutionLayer* cLayer);
PoolLayer* setupPoolLayer(int inputHeightMax, int inputWidthMax, int channelsInOut, int channelsInOutMax,
                          int flag_poolSizeAdap, int default_poolHeight, int default_poolWidth,
                          int poolHeightMin, int poolHeightMax, int poolWidthMin, int poolWidthMax,
                          int flag_poolTypeAdap, int default_poolType);
void assignPoolLayer(PoolLayer* pLayer, double* x, int mode);
void resetPoolLayer(PoolLayer* pLayer);
void freePoolLayer(PoolLayer* pLayer);
InterCPCLayer* setupInterCFCLayer(int preFeatureMapChannels, int preFeatureMapHeightMax, int preFeatureMapWidthMax,
                                  int numOutput,
                                  int flagActFunc, int flag_actFuncTypeAdap, int default_actFuncType,
                                  int flag_connectAdap, int typeConnectParaCoding,
                                  int layerNum, int* numNodesAll,
                                  int numLowRankMax, int GEP_head_length,
                                  int flag_GEP_weight, int flag_wt_positive, int flag_normalize_outData);
void assignInterCFCLayer(InterCPCLayer* icfcLayer, double* x, int mode);
void resetInterCFCLayer(InterCPCLayer* icfcLayer);
void freeInterCFCLayer(InterCPCLayer* icfcLayer);
FCLayer* setupFCLayer(int numInput, int numInputMax, int numOutput, int flagActFunc, int flag_actFuncTypeAdap,
                      int default_actFuncType, int flag_connectAdap);
void assignFCLayer(FCLayer* fcLayer, double* x, int mode);
void resetFCLayer(FCLayer* fcLayer);
void freeFCLayer(FCLayer* fcLayer);
MemberLayer* setupMemberLayer(int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                              int* flag_adapMemship, int typeFuzzySet,
                              int typeMFCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight);
void assignMemberLayer(MemberLayer* mLayer, double* x, int mode);
void resetMemberLayer(MemberLayer* mLayer);
void freeMemberLayer(MemberLayer* mLayer);
Member2DLayer* setupMember2DLayer(int numInput,
                                  int typeMFFeatMapCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight,
                                  int preFeatureMapHeightMax, int preFeatureMapWidthMax,
                                  int flag_typeMembershipFunAdap, int default_typeMembershipFun, int* numMemship, int* flag_adapMemship, int typeFuzzySet);
void assignMember2DLayer(Member2DLayer* m2DLayer, double* x, int mode);
void resetMember2DLayer(Member2DLayer* m2DLayer);
void freeMember2DLayer(Member2DLayer* m2DLayer);
FuzzyLayer* setupFuzzyLayer(int numInput, int* numMemship, int numRules, int typeFuzzySet, int typeRules, int typeInRuleCorNum,
                            int tag_GEP_rule, int typeConnectCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight);
void assignFuzzyLayer(FuzzyLayer* fLayer, double* x, int mode);
void resetFuzzyLayer(FuzzyLayer* fLayer);
void freeFuzzyLayer(FuzzyLayer* fLayer);
RoughLayer* setupRoughLayer(int numInput, int numRoughSets, int typeFuzzySet, int flagConnectStatusAdap,
                            int typeConnectCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight);
void assignRoughLayer(RoughLayer* rLayer, double* x, int mode);
void resetRoughLayer(RoughLayer* rLayer);
void freeRoughLayer(RoughLayer* rLayer);
OutReduceLayer* setupOutReduceLayer(int numInput, int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
                                    int typeFuzzySet, int typeTypeReducer,
                                    int consequenceNodeStatus, int centroid_num_tag,
                                    int numInputConsequenceNode, MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq, int flagConnectStatusAdap,
                                    int flagConnectWeightAdap,
                                    int typeConnectCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight);
void assignOutReduceLayer(OutReduceLayer* oLayer, double* x, int mode);
void resetOutReduceLayer(OutReduceLayer* oLayer);
void freeOutReduceLayer(OutReduceLayer* oLayer);

//////////////////////////////////////////////////////////////////////////
void print_para_convLayer(ConvolutionLayer* cLayer);
void print_para_poolLayer(PoolLayer* pLayer);
void print_para_icfcLayer(InterCPCLayer* icfcLayer);
void print_para_fcLayer(FCLayer* fcLayer);
void print_para_memberLayer(MemberLayer* mLayer);
void print_para_member2DLayer(Member2DLayer* m2DLayer);
void print_para_fuzzyLayer(FuzzyLayer* fLayer);
void print_para_roughLayer(RoughLayer* rLayer);
void print_para_outReduceLayer(OutReduceLayer* oLayer);

//////////////////////////////////////////////////////////////////////////
void ff_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus);
void ff_poolLayer(PoolLayer* pLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus);
void ff_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus);
void ff_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* theDataIn, int* theTagIn, int numDataIn, MY_FLT_TYPE* dataflowStatus);
void ff_memberLayer(MemberLayer* mLayer, MY_FLT_TYPE* valInput, MY_FLT_TYPE* dataflowStatus);
void ff_member2DLayer(Member2DLayer* m2DLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                      int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus);
void ff_fuzzyLayer(FuzzyLayer* fLayer, MY_FLT_TYPE*** degreesMemb, MY_FLT_TYPE** dataflowStatus);
void ff_roughLayer(RoughLayer* rLayer, MY_FLT_TYPE** degreesInput, MY_FLT_TYPE* dataflowStatus);
void ff_outReduceLayer(OutReduceLayer* oLayer, MY_FLT_TYPE** degreesInput, MY_FLT_TYPE* dataflowStatus);

//////////////////////////////////////////////////////////////////////////
void print_data_convLayer(ConvolutionLayer* cLayer);
void print_data_poolLayer(PoolLayer* pLayer);
void print_data_icfcLayer(InterCPCLayer* icfcLayer);
void print_data_fcLayer(FCLayer* fcLayer);
void print_data_memberLayer(MemberLayer* mLayer);
void print_data_member2DLayer(Member2DLayer* m2DLayer);
void print_data_fuzzyLayer(FuzzyLayer* fLayer);
void print_data_roughLayer(RoughLayer* rLayer);
void print_data_outReduceLayer(OutReduceLayer* oLayer);

//////////////////////////////////////////////////////////////////////////
void bp_derivative_convLayer(ConvolutionLayer* cLayer);
void bp_derivative_poolLayer(PoolLayer* pLayer);
void bp_derivative_icfcLayer(InterCPCLayer* icfcLayer);
void bp_derivative_fcLayer(FCLayer* fcLayer);

void bp_delta_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer);
void bp_delta_poolLayer(PoolLayer* pLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer);
void bp_delta_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer);
void bp_delta_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* deltaPriorLayer, MY_FLT_TYPE* derivativePriorLayer, int* tagPriorLayer);

void bp_update_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** dataInput, int*** tagPriorLayer, FRNNOpts opts);
void bp_update_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** dataInput, int*** tagPriorLayer, FRNNOpts opts);
void bp_update_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* dataInput, int* tagPriorLayer, FRNNOpts opts);

//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
double rnd_uni_FRNN_MODEL(long* idum);
extern int     seed_FRNN_MODEL;
extern long    rnd_uni_init_FRNN_MODEL;

MY_FLT_TYPE rndreal_FRNN_MODEL(MY_FLT_TYPE low, MY_FLT_TYPE high);
MY_FLT_TYPE gaussrand_FRNN_MODEL(MY_FLT_TYPE a = 0.0, MY_FLT_TYPE b = 1.0);
int rnd_FRNN_MODEL(int low, int high);
void shuffle_FRNN_MODEL(int* x, int size);

#endif
