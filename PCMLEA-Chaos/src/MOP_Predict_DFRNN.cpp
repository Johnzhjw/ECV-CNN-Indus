#include "MOP_Predict_DFRNN.h"
#include <float.h>
#include <math.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
#endif

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_Predict_DFRNN 0
#define FLAG_ON_MOP_Predict_DFRNN 1
#define STATUS_OUT_INDEICES_MOP_Predict_DFRNN FLAG_OFF_MOP_Predict_DFRNN
#define STATUS_OUT_INDEICES_MOP_PREDICT_DFRNN FLAG_OFF_MOP_Predict_DFRNN

//#define OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_DFRNN
#define PRINT_ERROR_PARA_MOP_Predict_DFRNN

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN_MOP_Predict_DFRNN 1024
#define MAX_OUT_NUM_MOP_Predict_DFRNN 1024
#define VIOLATION_PENALTY_Predict_DFRNN 1e6

//////////////////////////////////////////////////////////////////////////
#define TAG_VALI_MOP_PREDICT_DFRNN -2
#define TAG_NULL_MOP_PREDICT_DFRNN 0
#define TAG_INVA_MOP_PREDICT_DFRNN -1

//////////////////////////////////////////////////////////////////////////
#define MAX_DATA_LEN_MOP_PREDICT_DFRNN 10000
#define MAX_ATTR_NUM_Pred_DFRNN 64

//////////////////////////////////////////////////////////////////////////
#define THRESHOLD_NUM_ROUGH_NODES_Pred_DFRNN 1

//////////////////////////////////////////////////////////////////////////
enum ENUM_DATA_INTERVAL_TYPE {
    D_INTV_T_MONTHLY,
    D_INTV_T_QUARTERLY,
    D_INTV_T_HOURLY,
    D_INTV_T_YEARLY,
    D_INTV_T_WEEKLY,
    D_INTV_T_DAILY
};
int d_intv_t_MOP_PREDICT_DFRNN;

#define PARA_M_MONTHLY   12
#define PARA_M_QUARTERLY 4
#define PARA_M_HOURLY    24
#define PARA_M_YEARLY    1
#define PARA_M_WEEKLY    1
#define PARA_M_DAILY     1
int p_m_MOP_PREDICT_DFRNN;

//////////////////////////////////////////////////////////////////////////
#define NUM_LABEL_TRADING_MOP_PREDICT_DFRNN 3
#define CLASS_IND_BUY_MOP_PREDICT_DFRNN  0
#define CLASS_IND_HOLD_MOP_PREDICT_DFRNN 1
#define CLASS_IND_SELL_MOP_PREDICT_DFRNN 2
#if CURRENT_PROB_MOP_PREDICT_DFRNN == STOCK_TRADING_MOP_PREDICT_DFRNN
#define win_size_max_MOP_Predict_DFRNN 35
#define win_size_min_MOP_Predict_DFRNN 7
#define win_size_cases_MOP_Predict_DFRNN 15 // 7 9 11 13 15 17 ... 35
int train_trading_label_MOP_Predict_DFRNN[win_size_cases_MOP_Predict_DFRNN][MAX_DATA_LEN_MOP_PREDICT_DFRNN];
int test_trading_label_MOP_Predict_DFRNN[win_size_cases_MOP_Predict_DFRNN][MAX_DATA_LEN_MOP_PREDICT_DFRNN];
#endif

//////////////////////////////////////////////////////////////////////////
int NDIM_MOP_Predict_DFRNN = 0;
int NOBJ_MOP_Predict_DFRNN = 0;
char prob_name_MOP_Predict_DFRNN[128];
int tag_classification_MOP_Predict_DFRNN;
int num_class_MOP_Predict_DFRNN;

int num_in__predict_MOP_Predict_DFRNN;
int ind_in__predict_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN];
int num_out_predict_MOP_Predict_DFRNN;
int ind_out_predict_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN];

int flag_multi_in_cnsq_MOP_Predict_DFRNN;
int flag_fuse_in_cnsq_MOP_Predict_DFRNN;
int type_frnn_cnsq_MOP_Predict_DFRNN;
int type_in_cnsq_MOP_Predict_DFRNN;
int type_in_MOP_Predict_DFRNN;
int flag_sep_in_MOP_Predict_DFRNN;

//////////////////////////////////////////////////////////////////////////
#define DATA_MIN_MOP_Predict_DFRNN 0
#define DATA_MAX_MOP_Predict_DFRNN 1
#define DATA_MEAN_MOP_Predict_DFRNN 2
#define DATA_STD_MOP_Predict_DFRNN 3
int numAttr_MOP_Predict_DFRNN; // not including the label for classification
double allData_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN][MAX_DATA_LEN_MOP_PREDICT_DFRNN];
double trainStat_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN][4];
double trainFact_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN];
double valStat_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN][4];
double testStat_MOP_Predict_DFRNN[MAX_ATTR_NUM_Pred_DFRNN][4];
int allDataSize_MOP_Predict_DFRNN = 0;
int trainDataSize_MOP_Predict_DFRNN = 0;
int valDataSize_MOP_Predict_DFRNN = 0;
int testDataSize_MOP_Predict_DFRNN = 0;
int NORMALIZE_Z_SCORE_MOP_Predict_DFRNN = 0;
int NORMALIZE_MIN_MAX_MOP_Predict_DFRNN = 1;
int NORMALIZE_MOP_Predict_DFRNN = -1;
int flag_T2_MOP_Predict_DFRNN = -1;

int  repNum_MOP_Predict_DFRNN;
int  repNo_MOP_Predict_DFRNN;

MY_FLT_TYPE total_penalty_MOP_Predict_DFRNN = 0.0;
MY_FLT_TYPE penaltyVal_MOP_Predict_DFRNN = 1e6;

frnn_MOP_Predict_DFRNN* frnn_MOP_Predict_dfrnn = NULL;

int num_layers_Predict_DFRNN;
int size_layers_Predict_DFRNN[10];

//////////////////////////////////////////////////////////////////////////
static void ff_Predict_DFRNN_c(double* individual, int tag_train_test);
static double simplicity_MOP_Predict_DFRNN();
static double generality_MOP_Predict_DFRNN();
static double get_profit_MOP_Predict_DFRNN(int tag_train_test);
static void readData_stock_MOP_Predict_DFRNN(char* fname, int trainNo, int testNo, int endNo);
static void readData_general_MOP_Predict_DFRNN(char* fname, int tag_classification);
static void normalizeData_MOP_Predict_DFRNN();
static void get_Evaluation_Indicators_MOP_Predict_DFRNN(int num_class, MY_FLT_TYPE* N_TP, MY_FLT_TYPE* N_FP, MY_FLT_TYPE* N_TN,
        MY_FLT_TYPE* N_FN, MY_FLT_TYPE* N_wrong, MY_FLT_TYPE* N_sum,
        MY_FLT_TYPE* mean_prc, MY_FLT_TYPE* std_prc, MY_FLT_TYPE* mean_rec, MY_FLT_TYPE* std_rec, MY_FLT_TYPE* mean_ber,
        MY_FLT_TYPE* std_ber);
#if CURRENT_PROB_MOP_PREDICT_DFRNN == STOCK_TRADING_MOP_PREDICT_DFRNN
static void genTradingLabel_MOP_Predict_DFRNN();
#endif
//
int     seed_Predict_DFRNN = 237;
long    rnd_uni_init_Predict_DFRNN = -(long)seed_Predict_DFRNN;
static double rnd_uni_Predict_DFRNN(long* idum);
static int rnd_Predict_DFRNN(int low, int high);
static void shuffle_Predict_DFRNN(int* x, int size);
static void trimLine_MOP_Predict_DFRNN(char line[]);
static int get_setting_MOP_Predict_DFRNN(char* wholestr, const char* candidstr, int& val, int* vec);

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Predict_DFRNN(char* pro, int curN, int numN, int trainNo, int testNo, int endNo, int my_rank)
{
    //
    sprintf(prob_name_MOP_Predict_DFRNN, "%s", pro);
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    seed_Predict_DFRNN = 237 + my_rank;
    seed_Predict_DFRNN = seed_Predict_DFRNN % 1235;
    rnd_uni_init_Predict_DFRNN = -(long)seed_Predict_DFRNN;
    for(int i = 0; i < curN; i++) {
        seed_Predict_DFRNN = (seed_Predict_DFRNN + 111) % 1235;
        rnd_uni_init_Predict_DFRNN = -(long)seed_Predict_DFRNN;
    }
    //
    repNo_MOP_Predict_DFRNN = curN;
    repNum_MOP_Predict_DFRNN = numN;
    //
    flag_multi_in_cnsq_MOP_Predict_DFRNN = FLAG_STATUS_OFF;
    flag_fuse_in_cnsq_MOP_Predict_DFRNN = FLAG_STATUS_OFF;
    get_setting_MOP_Predict_DFRNN(pro, "FuseF", flag_fuse_in_cnsq_MOP_Predict_DFRNN, NULL);
    NORMALIZE_MOP_Predict_DFRNN = NORMALIZE_MIN_MAX_MOP_Predict_DFRNN;
    get_setting_MOP_Predict_DFRNN(pro, "Tnorm", NORMALIZE_MOP_Predict_DFRNN, NULL);
    flag_T2_MOP_Predict_DFRNN = 1;
    get_setting_MOP_Predict_DFRNN(pro, "flagT2", flag_T2_MOP_Predict_DFRNN, NULL);
    type_frnn_cnsq_MOP_Predict_DFRNN = FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_ADAPT;
    get_setting_MOP_Predict_DFRNN(pro, "Tcnsq", type_frnn_cnsq_MOP_Predict_DFRNN, NULL);
    num_layers_Predict_DFRNN = 1;
    size_layers_Predict_DFRNN[0] = 1;
    get_setting_MOP_Predict_DFRNN(pro, "Deep", num_layers_Predict_DFRNN, size_layers_Predict_DFRNN);
    type_in_cnsq_MOP_Predict_DFRNN = 0;
    get_setting_MOP_Predict_DFRNN(pro, "TpInCnsq", type_in_cnsq_MOP_Predict_DFRNN, NULL);
    type_in_MOP_Predict_DFRNN = 0;
    get_setting_MOP_Predict_DFRNN(pro, "TpIn", type_in_MOP_Predict_DFRNN, NULL);
    flag_sep_in_MOP_Predict_DFRNN = FLAG_STATUS_OFF;
    get_setting_MOP_Predict_DFRNN(pro, "SepIn", flag_sep_in_MOP_Predict_DFRNN, NULL);
    num_class_MOP_Predict_DFRNN = 0;
    //
    char filename[MAX_STR_LEN_MOP_Predict_DFRNN];
    if(strstr(pro, "Stock_")) {
        tag_classification_MOP_Predict_DFRNN = 0;
        numAttr_MOP_Predict_DFRNN = 6;
        sprintf(filename, "../Data_all/AllFileNames_FRNN");
        readData_stock_MOP_Predict_DFRNN(filename, trainNo, testNo, endNo);
    } else if(strstr(pro, "Classify_")) {
        tag_classification_MOP_Predict_DFRNN = 1;
        char* ret = strstr(pro, "Classify_");
        ret += strlen("Classify_");
        sprintf(filename, "../Data_all/UCI_Data/%s", ret);
        readData_general_MOP_Predict_DFRNN(filename, tag_classification_MOP_Predict_DFRNN);
    } else if(strstr(pro, "TimeSeries_")) {
        tag_classification_MOP_Predict_DFRNN = 0;
        char* ret = strstr(pro, "TimeSeries_");
        ret += strlen("TimeSeries_");
        char tmp_str[MAX_STR_LEN_MOP_Predict_DFRNN];
        sprintf(tmp_str, "%s", ret);
        for(int i = 0; i < MAX_STR_LEN_MOP_Predict_DFRNN; i++) {
            if(tmp_str[i] == '_') {
                tmp_str[i] = '\0';
                break;
            }
        }
        sprintf(filename, "../Data_all/UCI_Data/%s", tmp_str);
        readData_general_MOP_Predict_DFRNN(filename, tag_classification_MOP_Predict_DFRNN);
    } else {
        printf("\n%s(%d): Unknown problem name ~ %s, the dataset cannot be found, exiting...\n",
               __FILE__, __LINE__, pro);
        exit(-9124);
    }
    if(tag_classification_MOP_Predict_DFRNN && num_class_MOP_Predict_DFRNN == 0) {
        printf("\n%s(%d): The number of classes is 0 for classification, the data file should include the number of classes, exiting...\n",
               __FILE__, __LINE__);
        exit(-91224);
    }
    num_in__predict_MOP_Predict_DFRNN = numAttr_MOP_Predict_DFRNN;
    for(int i = 0; i < numAttr_MOP_Predict_DFRNN; i++) ind_in__predict_MOP_Predict_DFRNN[i] = i;
    if(tag_classification_MOP_Predict_DFRNN) {
        num_out_predict_MOP_Predict_DFRNN = 1;
        flag_multi_in_cnsq_MOP_Predict_DFRNN = FLAG_STATUS_ON;
    } else {
        if(strstr(pro, "Stock_")) {
            num_out_predict_MOP_Predict_DFRNN = 1;
            ind_out_predict_MOP_Predict_DFRNN[0] = 0;
        } else if(strstr(pro, "TimeSeries_")) {
            if(strstr(pro, "gnfuv")) {
                num_out_predict_MOP_Predict_DFRNN = 1;
                ind_out_predict_MOP_Predict_DFRNN[0] = 0;
                ind_out_predict_MOP_Predict_DFRNN[1] = 1;
            } else if(strstr(pro, "hungaryChickenpox")) {
                num_out_predict_MOP_Predict_DFRNN = 1;
                ind_out_predict_MOP_Predict_DFRNN[0] = 0;
            } else if(strstr(pro, "SML2010-DATA")) {
                num_out_predict_MOP_Predict_DFRNN = 1;
                ind_out_predict_MOP_Predict_DFRNN[0] = 0;
                ind_out_predict_MOP_Predict_DFRNN[1] = 1;
            } else if(strstr(pro, "traffic")) {
                num_out_predict_MOP_Predict_DFRNN = 1;
                ind_out_predict_MOP_Predict_DFRNN[0] = 0;
            } else if(strstr(pro, "Daily_Demand_Forecasting_Orders")) {
                num_out_predict_MOP_Predict_DFRNN = 1;
                ind_out_predict_MOP_Predict_DFRNN[0] = 0;
            } else {
                printf("\n%s(%d): Unknown problem name ~ %s, cannot set parameters, exiting...\n",
                       __FILE__, __LINE__, pro);
                exit(-9124);
            }
        } else {
            printf("\n%s(%d): Unknown problem name ~ %s, cannot set parameters, exiting...\n",
                   __FILE__, __LINE__, pro);
            exit(-9124);
        }
    }
    //
    d_intv_t_MOP_PREDICT_DFRNN = D_INTV_T_DAILY;
    p_m_MOP_PREDICT_DFRNN = 1;
    if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_MONTHLY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_MONTHLY;
    else if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_QUARTERLY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_QUARTERLY;
    else if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_HOURLY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_HOURLY;
    else if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_YEARLY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_YEARLY;
    else if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_WEEKLY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_WEEKLY;
    else if(d_intv_t_MOP_PREDICT_DFRNN == D_INTV_T_DAILY)
        p_m_MOP_PREDICT_DFRNN = PARA_M_DAILY;
    // Normalization
    for(int i = 0; i < MAX_ATTR_NUM_Pred_DFRNN; i++) trainFact_MOP_Predict_DFRNN[i] = 1;
    normalizeData_MOP_Predict_DFRNN();
    //////////////////////////////////////////////////////////////////////////
#if CURRENT_PROB_MOP_PREDICT_DFRNN == STOCK_TRADING_MOP_PREDICT_DFRNN
    genTradingLabel_MOP_Predict_DFRNN();
#endif
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    frnn_MOP_Predict_dfrnn = (frnn_MOP_Predict_DFRNN*)calloc(1, sizeof(frnn_MOP_Predict_DFRNN));
    if(strstr(pro, "evoDeepFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepGFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepDFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepDFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepGFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDeepBFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoDeepBGFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoDeepBDFRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoDeepBFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoDeepBDFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoDeepBGFGRNN")) {
        frnn_MOP_Predict_dfrnn->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict_dfrnn->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict_dfrnn->tag_multiKindInput = FLAG_STATUS_ON;
    } else {
        printf("\n%s(%d): Unknown problem name ~ %s, exiting...\n",
               __FILE__, __LINE__, pro);
        exit(-91284);
    }
    if(frnn_MOP_Predict_dfrnn->tag_multiKindInput == FLAG_STATUS_OFF) {
        num_in__predict_MOP_Predict_DFRNN = 1;
        ind_in__predict_MOP_Predict_DFRNN[0] = 0;
    }
    if(flag_fuse_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_ON)
        flag_multi_in_cnsq_MOP_Predict_DFRNN = FLAG_STATUS_ON;
    if(flag_fuse_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_ON &&
       tag_classification_MOP_Predict_DFRNN) {
        printf("%s(%d): For the classification problem, there is no need to set feature fusing, exiting...\n",
               __FILE__, __LINE__);
        exit(-972849);
    }
    if(flag_fuse_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_ON) {
        // remove the target ind in the input features
        int i = 0;
        while(i < num_in__predict_MOP_Predict_DFRNN) {
            for(int o = 0; o < num_out_predict_MOP_Predict_DFRNN; o++) {
                if(ind_in__predict_MOP_Predict_DFRNN[i] == ind_out_predict_MOP_Predict_DFRNN[o]) {
                    num_in__predict_MOP_Predict_DFRNN--;
                    if(num_in__predict_MOP_Predict_DFRNN <= 0) {
                        printf("%s(%d): There are no remaining input features, exiting...\n", __FILE__, __LINE__);
                        exit(-97281);
                    }
                    for(int ii = i; ii < num_in__predict_MOP_Predict_DFRNN; ii++) {
                        ind_in__predict_MOP_Predict_DFRNN[ii] = ind_in__predict_MOP_Predict_DFRNN[ii + 1];
                    }
                }
            }
            i++;
        }
    }
    dfrnn_Predict_DFRNN_setup(frnn_MOP_Predict_dfrnn);
    //
    NDIM_MOP_Predict_DFRNN = frnn_MOP_Predict_dfrnn->numParaLocal;
    NOBJ_MOP_Predict_DFRNN = 3;
    //
    return;
}
void SetLimits_MOP_Predict_DFRNN(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    for(int iL = 0; iL < frnn_MOP_Predict_dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < frnn_MOP_Predict_dfrnn->layerSize[iL]; iLS++) {
            if(frnn_MOP_Predict_dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal; i++) {
                        minLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMin[i];
                        maxLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMax[i];
                        count++;
                    }
                }
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numParaLocal; i++) {
                minLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMin[i];
                maxLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMax[i];
                count++;
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numParaLocal; i++) {
                minLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMin[i];
                maxLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMax[i];
                count++;
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numParaLocal; i++) {
                minLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMin[i];
                maxLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMax[i];
                count++;
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numParaLocal; i++) {
                minLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMin[i];
                maxLimit[count] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMax[i];
                count++;
            }
        }
    }
    //
    return;
}

int CheckLimits_MOP_Predict_DFRNN(double* x, int nx)
{
    int count = 0;
    //
    for(int iL = 0; iL < frnn_MOP_Predict_dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < frnn_MOP_Predict_dfrnn->layerSize[iL]; iLS++) {
            if(frnn_MOP_Predict_dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal; i++) {
                        if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMin[i] ||
                           x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMax[i]) {
                            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].GEP0[%d] %d, %.16e not in [%.16e, %.16e]\n",
                                   __FILE__, __LINE__, iL, iLS, n, i, x[count],
                                   frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMin[i],
                                   frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[n]->xMax[i]);
                            return 0;
                        }
                        count++;
                    }
                }
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMin[i] ||
                   x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].M1 %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, iL, iLS, i, x[count],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMin[i],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->xMax[i]);
                    return 0;
                }
                count++;
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMin[i] ||
                   x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].F2 %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, iL, iLS, i, x[count],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMin[i],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->xMax[i]);
                    return 0;
                }
                count++;
            }
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMin[i] ||
                   x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].R3 %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, iL, iLS, i, x[count],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMin[i],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->xMax[i]);
                    return 0;
                }
                count++;
            }
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMin[i] ||
                   x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].OL %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, iL, iLS, i, x[count],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMin[i],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMax[i]);
                    return 0;
                }
                count++;
            }
#else
            if(frnn_MOP_Predict_dfrnn->flagConnectStatus != FLAG_STATUS_OFF ||
               frnn_MOP_Predict_dfrnn->flagConnectWeight != FLAG_STATUS_ON ||
               frnn_MOP_Predict_dfrnn->typeCoding != PARA_CODING_DIRECT) {
                printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
                       __FILE__, __LINE__, frnn_MOP_Predict_dfrnn->flagConnectStatus, frnn_MOP_Predict_dfrnn->flagConnectWeight,
                       frnn_MOP_Predict_dfrnn->typeCoding);
                exit(-275082);
            }
            int tmp_offset = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numOutput * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput;
            count += tmp_offset;
            for(int i = tmp_offset; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMin[i] ||
                   x[count] > frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->dfrnn[%d][%d].OL %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, iL, iLS, i, x[count],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMin[i],
                           frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->xMax[i]);
                    return 0;
                }
                count++;
            }
#endif
        }
    }
    //
    return 1;
}

void Fitness_MOP_Predict_DFRNN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    ff_Predict_DFRNN_c(individual, TRAIN_TAG_MOP_PREDICT_DFRNN);
    ff_Predict_DFRNN_c(individual, VAL_TAG_MOP_PREDICT_DFRNN);
    //
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
    if(!tag_classification_MOP_Predict_DFRNN) {
        fitness[0] = sqrt(frnn_MOP_Predict_dfrnn->sum_wrong / frnn_MOP_Predict_dfrnn->sum_all);
    } else {
        MY_FLT_TYPE mean_prc = 0;
        get_Evaluation_Indicators_MOP_Predict_DFRNN(num_class_MOP_Predict_DFRNN,
                frnn_MOP_Predict_dfrnn->N_TP, frnn_MOP_Predict_dfrnn->N_FP, frnn_MOP_Predict_dfrnn->N_TN, frnn_MOP_Predict_dfrnn->N_FN,
                frnn_MOP_Predict_dfrnn->N_wrong, frnn_MOP_Predict_dfrnn->N_sum,
                &mean_prc, NULL, NULL, NULL, NULL, NULL);
        fitness[0] = 1 - mean_prc;
    }
#else
    fitness[0] = get_profit_MOP_Predict_DFRNN(VAL_TAG_MOP_PREDICT_DFRNN);
#endif
    fitness[1] = simplicity_MOP_Predict_DFRNN();
    //
    fitness[2] = generality_MOP_Predict_DFRNN();
    //
    for(int i = 0; i < NOBJ_MOP_Predict_DFRNN; i++)
        fitness[i] += total_penalty_MOP_Predict_DFRNN;
    //
    return;
}

void Fitness_MOP_Predict_DFRNN_test(double* individual, double* fitness)
{
    ff_Predict_DFRNN_c(individual, TEST_TAG_MOP_PREDICT_DFRNN);
    //
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
    if(!tag_classification_MOP_Predict_DFRNN) {
        fitness[0] = sqrt(frnn_MOP_Predict_dfrnn->sum_wrong / frnn_MOP_Predict_dfrnn->sum_all);
    } else {
        MY_FLT_TYPE mean_prc = 0;
        get_Evaluation_Indicators_MOP_Predict_DFRNN(num_class_MOP_Predict_DFRNN,
                frnn_MOP_Predict_dfrnn->N_TP, frnn_MOP_Predict_dfrnn->N_FP, frnn_MOP_Predict_dfrnn->N_TN, frnn_MOP_Predict_dfrnn->N_FN,
                frnn_MOP_Predict_dfrnn->N_wrong, frnn_MOP_Predict_dfrnn->N_sum,
                &mean_prc, NULL, NULL, NULL, NULL, NULL);
        fitness[0] = mean_prc;
    }
#else
    fitness[0] = get_profit_MOP_Predict_DFRNN(TEST_TAG_MOP_PREDICT_DFRNN);
#endif
    fitness[1] = simplicity_MOP_Predict_DFRNN();
    //
    fitness[2] = generality_MOP_Predict_DFRNN();
    //
    for(int i = 0; i < NOBJ_MOP_Predict_DFRNN; i++)
        fitness[i] += total_penalty_MOP_Predict_DFRNN;
    //
    return;
}

static void ff_Predict_DFRNN_c(double* individual, int tag_train_val_test)
{
    int num_in = 0;
    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
        num_in = trainDataSize_MOP_Predict_DFRNN;
    else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
        num_in = valDataSize_MOP_Predict_DFRNN;
    else
        num_in = testDataSize_MOP_Predict_DFRNN;
    //
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
    int num_sample = num_in - (frnn_MOP_Predict_dfrnn->numInput - 1) * frnn_MOP_Predict_dfrnn->lenGap - 1 -
                     frnn_MOP_Predict_dfrnn->numOutput * frnn_MOP_Predict_dfrnn->lenGap + 1;
    if(tag_train_val_test != TRAIN_TAG_MOP_PREDICT_DFRNN)
        num_sample = num_in - (frnn_MOP_Predict_dfrnn->numOutput - 1) * frnn_MOP_Predict_dfrnn->lenGap;
    if(tag_classification_MOP_Predict_DFRNN)
        num_sample = num_in;
#else
    int num_sample = num_in - frnn_MOP_Predict_dfrnn->numInput + 1;
    if(tag_train_val_test != TRAIN_TAG_MOP_PREDICT_DFRNN)
        num_sample = num_in;
#endif
    dfrnn_Predict_DFRNN_init(frnn_MOP_Predict_dfrnn, individual, ASSIGN_MODE_FRNN);
    //
    const int len_valIn = 100 * MAX_ATTR_NUM_Pred_DFRNN;
    MY_FLT_TYPE valIn[len_valIn];
    MY_FLT_TYPE valIn2[len_valIn];
    MY_FLT_TYPE valOut[MAX_OUT_NUM_MOP_Predict_DFRNN];
    //
    MY_FLT_TYPE***** matBp = NULL;
    int layerNum = frnn_MOP_Predict_dfrnn->layerNum;
    matBp = (MY_FLT_TYPE*****)malloc(layerNum * sizeof(MY_FLT_TYPE****));
    for(int iL = 0; iL < layerNum; iL++) {
        int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
        matBp[iL] = (MY_FLT_TYPE****)malloc(layerSize * sizeof(MY_FLT_TYPE***));
        for(int iLS = 0; iLS < layerSize; iLS++) {
            int nMultiOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
            int nOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput;
            matBp[iL][iLS] = (MY_FLT_TYPE***)malloc(nMultiOut * sizeof(MY_FLT_TYPE**));
            for(int n = 0; n < nMultiOut; n++) {
                matBp[iL][iLS][n] = (MY_FLT_TYPE**)malloc(nOut * sizeof(MY_FLT_TYPE*));
                for(int i = 0; i < nOut; i++) {
                    matBp[iL][iLS][n][i] = (MY_FLT_TYPE*)calloc(num_sample, sizeof(MY_FLT_TYPE));
                }
            }
        }
    }
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    MY_FLT_TYPE***** matA = NULL;
    MY_FLT_TYPE***** matB = NULL;
    MY_FLT_TYPE***** matLeft = NULL;
    MY_FLT_TYPE***** matRight = NULL;
    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
        matA = (MY_FLT_TYPE*****)malloc(layerNum * sizeof(MY_FLT_TYPE****));
        matB = (MY_FLT_TYPE*****)malloc(layerNum * sizeof(MY_FLT_TYPE****));
        matLeft = (MY_FLT_TYPE*****)malloc(layerNum * sizeof(MY_FLT_TYPE****));
        matRight = (MY_FLT_TYPE*****)malloc(layerNum * sizeof(MY_FLT_TYPE****));
        for(int iL = 0; iL < layerNum; iL++) {
            int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
            matA[iL] = (MY_FLT_TYPE****)malloc(layerSize * sizeof(MY_FLT_TYPE***));
            matB[iL] = (MY_FLT_TYPE****)malloc(layerSize * sizeof(MY_FLT_TYPE***));
            matLeft[iL] = (MY_FLT_TYPE****)malloc(layerSize * sizeof(MY_FLT_TYPE***));
            matRight[iL] = (MY_FLT_TYPE****)malloc(layerSize * sizeof(MY_FLT_TYPE***));
            for(int iLS = 0; iLS < layerSize; iLS++) {
                int nMultiOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
                int nOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput;
                matA[iL][iLS] = (MY_FLT_TYPE***)malloc(nMultiOut * sizeof(MY_FLT_TYPE**));
                matB[iL][iLS] = (MY_FLT_TYPE***)malloc(nMultiOut * sizeof(MY_FLT_TYPE**));
                matLeft[iL][iLS] = (MY_FLT_TYPE***)malloc(nMultiOut * sizeof(MY_FLT_TYPE**));
                matRight[iL][iLS] = (MY_FLT_TYPE***)malloc(nMultiOut * sizeof(MY_FLT_TYPE**));
                for(int n = 0; n < nMultiOut; n++) {
                    matA[iL][iLS][n] = (MY_FLT_TYPE**)malloc(nOut * sizeof(MY_FLT_TYPE*));
                    matB[iL][iLS][n] = (MY_FLT_TYPE**)malloc(nOut * sizeof(MY_FLT_TYPE*));
                    matLeft[iL][iLS][n] = (MY_FLT_TYPE**)malloc(nOut * sizeof(MY_FLT_TYPE*));
                    matRight[iL][iLS][n] = (MY_FLT_TYPE**)malloc(nOut * sizeof(MY_FLT_TYPE*));
                    int nIn = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput;
                    for(int i = 0; i < nOut; i++) {
                        matA[iL][iLS][n][i] = (MY_FLT_TYPE*)calloc(num_sample * nIn, sizeof(MY_FLT_TYPE));
                        matB[iL][iLS][n][i] = (MY_FLT_TYPE*)calloc(num_sample, sizeof(MY_FLT_TYPE));
                        matLeft[iL][iLS][n][i] = (MY_FLT_TYPE*)calloc(nIn * nIn, sizeof(MY_FLT_TYPE));
                        matRight[iL][iLS][n][i] = (MY_FLT_TYPE*)calloc(nIn, sizeof(MY_FLT_TYPE));
                    }
                }
            }
        }
    }
#endif
    //
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_DFRNN
    FILE *fpt = NULL;
    char tmp_fnm[128];
    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
        sprintf(tmp_fnm, "tmpFile/OUT_%s_train.csv", prob_name_MOP_Predict_DFRNN);
    else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
        sprintf(tmp_fnm, "tmpFile/OUT_%s_val.csv", prob_name_MOP_Predict_DFRNN);
    else
        sprintf(tmp_fnm, "tmpFile/OUT_%s_test.csv", prob_name_MOP_Predict_DFRNN);
    fpt = fopen(tmp_fnm, "w");
#endif
    //
    for(int iL = 0; iL < layerNum; iL++) {
        int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
        for(int iLS = 0; iLS < layerSize; iLS++) {
            frnn_MOP_Predict_dfrnn->sum_all = 0;
            frnn_MOP_Predict_dfrnn->sum_wrong = 0;
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->num_out_all_max; i++) {
                frnn_MOP_Predict_dfrnn->N_sum[i] = 0;
                frnn_MOP_Predict_dfrnn->N_wrong[i] = 0;
                frnn_MOP_Predict_dfrnn->e_sum[i] = 0;
                frnn_MOP_Predict_dfrnn->N_TP[i] = 0;
                frnn_MOP_Predict_dfrnn->N_TN[i] = 0;
                frnn_MOP_Predict_dfrnn->N_FP[i] = 0;
                frnn_MOP_Predict_dfrnn->N_FN[i] = 0;
            }
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
            int tmp_offset_samp = 0;
            if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                for(int iL = 0; iL < layerNum; iL++) {
                    int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
                    for(int iLS = 0; iLS < layerSize; iLS++) {
                        int nMultiOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
                        int nOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput;
                        for(int n = 0; n < nMultiOut; n++) {
                            int nIn = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput;
                            for(int i = 0; i < nOut; i++) {
                                for(int j = 0; j < num_sample; j++) {
                                    for(int k = 0; k < nIn; k++) {
                                        matA[iL][iLS][n][i][j * nIn + k] = 0;
                                    }
                                    matB[iL][iLS][n][i][j] = 0;
                                }
                                for(int j = 0; j < nIn; j++) {
                                    for(int k = 0; k < nIn; k++) {
                                        matLeft[iL][iLS][n][i][j * nIn + k] = 0;
                                    }
                                    matRight[iL][iLS][n][i][j] = 0;
                                }
                            }
                        }
                    }
                }
            }
#endif
            //
            for(int m = 0; m < num_sample; m++) {
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->num_multiKindInput; n++) {
                    int t_i = ind_in__predict_MOP_Predict_DFRNN[n];
                    int tmp_ind_os = n * frnn_MOP_Predict_dfrnn->numInput;
                    if(tag_classification_MOP_Predict_DFRNN) {
                        if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][m + i * frnn_MOP_Predict_dfrnn->lenGap];
                        else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][trainDataSize_MOP_Predict_DFRNN + m + i * frnn_MOP_Predict_dfrnn->lenGap];
                        else
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN + m + i *
                                                                   frnn_MOP_Predict_dfrnn->lenGap];
                    } else {
                        if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][m + i * frnn_MOP_Predict_dfrnn->lenGap];
                        else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][trainDataSize_MOP_Predict_DFRNN + m + i * frnn_MOP_Predict_dfrnn->lenGap -
                                                                   (frnn_MOP_Predict_dfrnn->numInput - 1) * frnn_MOP_Predict_dfrnn->lenGap - 1];
                        else
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->numInput; i++)
                                valIn[tmp_ind_os + i] =
                                    allData_MOP_Predict_DFRNN[t_i][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN + m + i *
                                                                   frnn_MOP_Predict_dfrnn->lenGap -
                                                                   (frnn_MOP_Predict_dfrnn->numInput - 1) * frnn_MOP_Predict_dfrnn->lenGap - 1];
                    }
                }
                if(iL == 0) {
                    memcpy(valIn2, valIn, frnn_MOP_Predict_dfrnn->num_multiKindInput * frnn_MOP_Predict_dfrnn->numInput * sizeof(MY_FLT_TYPE));
                } else {
                    for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindInput; n++) {
                        int tmp_ind_os = n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numInput;
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numInput; i++) {
                            int i_in_n = i / frnn_MOP_Predict_dfrnn->dfrnn[iL - 1][i].numOutput;
                            int i_in = i % frnn_MOP_Predict_dfrnn->dfrnn[iL - 1][i].numOutput;
                            valIn2[tmp_ind_os + i] = matBp[iL - 1][i_in_n][n][i_in][m];
                        }
                    }
                }
                //
                ff_frnn_Predict_DFRNN(frnn_MOP_Predict_dfrnn, iL, iLS, valIn2, valOut, valIn);
                //
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        matBp[iL][iLS][n][j][m] = valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j];
                        if(CHECK_INVALID(valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j])) {
                            printf("%d~%lf", j, valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j]);
                        }
                    }
                }
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
                if(tag_train_val_test != TRAIN_TAG_MOP_PREDICT_DFRNN)
#endif
                {
#ifdef PRINT_ERROR_PARA_MOP_Predict_DFRNN
                    for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numOutput; i++) {
                        if(CHECK_INVALID(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->valOutputFinal[i])) {
                            printf("%s(%d): Invalid output dfrnn[%d][%d] %d ~ %lf, exiting...\n",
                                   __FILE__, __LINE__, iL, iLS, i,
                                   frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->valOutputFinal[i]);
                            print_para_memberLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1);
                            print_data_memberLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1);
                            print_para_fuzzyLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2);
                            print_data_fuzzyLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2);
                            print_para_roughLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3);
                            print_data_roughLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3);
                            print_para_outReduceLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL);
                            print_data_outReduceLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL);
                            exit(-94628);
                        }
                    }
#endif
                }
                MY_FLT_TYPE* cur_out = valOut;
                double* true_out;
                int true_label;
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
                if(!tag_classification_MOP_Predict_DFRNN) {
                    MY_FLT_TYPE tmp_dif1 = 0.0;
                    for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                        int tmp_o = ind_out_predict_MOP_Predict_DFRNN[n];
                        if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][m + frnn_MOP_Predict_dfrnn->numInput * frnn_MOP_Predict_dfrnn->lenGap];
                        else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][trainDataSize_MOP_Predict_DFRNN + m];
                        else
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN + m];
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                            tmp_dif1 += (cur_out[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i] - true_out[i * frnn_MOP_Predict_dfrnn->lenGap]) *
                                        (cur_out[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i] - true_out[i * frnn_MOP_Predict_dfrnn->lenGap]) *
                                        trainFact_MOP_Predict_DFRNN[tmp_o];
                        }
                    }
                    frnn_MOP_Predict_dfrnn->sum_all++;
                    frnn_MOP_Predict_dfrnn->sum_wrong += tmp_dif1 / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput /
                                                         frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
                } else {
                    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
                        true_out = &allData_MOP_Predict_DFRNN[numAttr_MOP_Predict_DFRNN][m];
                    else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
                        true_out = &allData_MOP_Predict_DFRNN[numAttr_MOP_Predict_DFRNN][trainDataSize_MOP_Predict_DFRNN + m];
                    else
                        true_out = &allData_MOP_Predict_DFRNN[numAttr_MOP_Predict_DFRNN][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN
                                   + m];
                    int cur_label = 0;
                    MY_FLT_TYPE cur_val = valOut[0];
                    for(int j = 1; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        if(cur_val < valOut[j]) {
                            cur_val = valOut[j];
                            cur_label = j;
                        }
                    }
                    true_label = true_out[0];
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        if(j == cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_TP[j]++;
                        if(j == cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_FP[j]++;
                        if(j != cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_FN[j]++;
                        if(j != cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_TN[j]++;
                    }
                    frnn_MOP_Predict_dfrnn->sum_all++;
                    frnn_MOP_Predict_dfrnn->N_sum[true_label]++;
                    if(cur_label != true_label) {
                        frnn_MOP_Predict_dfrnn->sum_wrong++;
                        frnn_MOP_Predict_dfrnn->N_wrong[true_label]++;
                    }
                }
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_DFRNN
                fprintf(fpt, "%e,%e\n", true_out[0], cur_out[0]);
#endif
#else
                int cur_label = 0;
                MY_FLT_TYPE cur_val = valOut[0];
                for(int j = 1; j < frnn_MOP_Predict_dfrnn->numOutput; j++) {
                    if(cur_val < valOut[j]) {
                        cur_val = valOut[j];
                        cur_label = j;
                    }
                }
                true_label = 0;
                if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                    frnn_MOP_Predict_dfrnn->trading_actions[m + frnn_MOP_Predict_dfrnn->numInput - 1] = cur_label;
                    true_label = train_trading_label_MOP_Predict_DFRNN[2][m + frnn_MOP_Predict_dfrnn->numInput - 1];
                } else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN) {
                    frnn_MOP_Predict_dfrnn->trading_actions[m] = cur_label;
                    true_label = val_trading_label_MOP_Predict_DFRNN[2][m];
                } else {
                    frnn_MOP_Predict_dfrnn->trading_actions[m] = cur_label;
                    true_label = test_trading_label_MOP_Predict_DFRNN[2][m];
                }
                for(int j = 0; j < frnn_MOP_Predict_dfrnn->numOutput; j++) {
                    if(j == cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_TP[j]++;
                    if(j == cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_FP[j]++;
                    if(j != cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_FN[j]++;
                    if(j != cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_TN[j]++;
                }
                frnn_MOP_Predict_dfrnn->sum_all++;
                frnn_MOP_Predict_dfrnn->N_sum[true_label]++;
                if(cur_label != true_label) {
                    frnn_MOP_Predict_dfrnn->sum_wrong++;
                    frnn_MOP_Predict_dfrnn->N_wrong[true_label]++;
                }
#endif
                //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
                if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                    for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                        int tmp_o = ind_out_predict_MOP_Predict_DFRNN[n];
                        if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN)
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][m +
                                       frnn_MOP_Predict_dfrnn->numInput * frnn_MOP_Predict_dfrnn->lenGap];
                        else if(tag_train_val_test == VAL_TAG_MOP_PREDICT_DFRNN)
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][trainDataSize_MOP_Predict_DFRNN + m];
                        else
                            true_out = &allData_MOP_Predict_DFRNN[tmp_o][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN + m];
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                            for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; j++) {
                                int ind_cur = tmp_offset_samp * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + j;
                                matA[iL][iLS][n][i][ind_cur] = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->valInputFinal[n *
                                                               frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i][j];
                            }
                            if(!tag_classification_MOP_Predict_DFRNN)
                                matB[iL][iLS][n][i][tmp_offset_samp] = true_out[i * frnn_MOP_Predict_dfrnn->lenGap];
                            else if(i == true_label)
                                matB[iL][iLS][n][i][tmp_offset_samp] = 1;
                            else
                                matB[iL][iLS][n][i][tmp_offset_samp] = -1;
                        }
                    }
                }
#else
                if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                    for(int i = 0; i < frnn_MOP_Predict_dfrnn->OL->numOutput; i++) {
                        for(int j = 0; j < frnn_MOP_Predict_dfrnn->OL->numInput; j++) {
                            int ind_cur = tmp_offset_samp * frnn_MOP_Predict_dfrnn->OL->numInput + j;
                            matA[0][i][ind_cur] = frnn_MOP_Predict_dfrnn->OL->valInputFinal[i][j];
                        }
                        if(i == true_label)
                            matB[0][i][tmp_offset_samp] = 1;
                        else
                            matB[0][i][tmp_offset_samp] = -1;
                    }
                }
#endif
                tmp_offset_samp++;
#endif
            }
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_DFRNN
            fclose(fpt);
#endif
            //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
            if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                frnn_MOP_Predict_dfrnn->sum_all = 0;
                frnn_MOP_Predict_dfrnn->sum_wrong = 0;
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->num_out_all_max; i++) {
                    frnn_MOP_Predict_dfrnn->N_sum[i] = 0;
                    frnn_MOP_Predict_dfrnn->N_wrong[i] = 0;
                    frnn_MOP_Predict_dfrnn->e_sum[i] = 0;
                    frnn_MOP_Predict_dfrnn->N_TP[i] = 0;
                    frnn_MOP_Predict_dfrnn->N_TN[i] = 0;
                    frnn_MOP_Predict_dfrnn->N_FP[i] = 0;
                    frnn_MOP_Predict_dfrnn->N_FN[i] = 0;
                }
                //
                //printf("tmp_offset_samp = %d\n", tmp_offset_samp);
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                    for(int iOut = 0; iOut < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; iOut++) {
                        MY_FLT_TYPE lambda = 9.3132e-10;
                        MY_FLT_TYPE tmp_max = 0;
                        int tmp_max_flag = 0;
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; i++) {
                            for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; j++) {
                                int tmp_o0 = i * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + j;
                                for(int k = 0; k < tmp_offset_samp; k++) {
                                    int tmp_i1 = k * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + i;
                                    int tmp_i2 = k * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + j;
                                    matLeft[iL][iLS][n][iOut][tmp_o0] += matA[iL][iLS][n][iOut][tmp_i1] * matA[iL][iLS][n][iOut][tmp_i2];
                                }
                                //if(i == j)
                                //    matLeft[tmp_o0] += lambda * fabs(matLeft[tmp_o0]);
                                if(i == j) {
                                    if(!tmp_max_flag) {
                                        tmp_max = matLeft[iL][iLS][n][iOut][tmp_o0];
                                        tmp_max_flag = 1;
                                    } else {
                                        if(tmp_max < matLeft[iL][iLS][n][iOut][tmp_o0])
                                            tmp_max = matLeft[iL][iLS][n][iOut][tmp_o0];
                                    }
                                }
                            }
                        }
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; i++) {
                            int tmp_o0 = i * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + i;
                            matLeft[iL][iLS][n][iOut][tmp_o0] += lambda;// *tmp_max;
                        }
                        for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; i++) {
                            for(int k = 0; k < tmp_offset_samp; k++) {
                                int tmp_i1 = k * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + i;
                                matRight[iL][iLS][n][iOut][i] += matA[iL][iLS][n][iOut][tmp_i1] * matB[iL][iLS][n][iOut][k];
                            }
                        }
                        int N = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput;
                        int NRHS = 1;
                        int LDA = N;
                        int LDB = NRHS;
                        int nn = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
                        int* ipiv = (int*)calloc(N, sizeof(int));
                        info = LAPACKE_dgesv(matStoreType, nn, nrhs, matLeft[iL][iLS][n][iOut], lda, ipiv, matRight[iL][iLS][n][iOut], ldb);
                        if(info > 0) {
                            printf("The diagonal element of the triangular factor of A,\n");
                            printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
                            printf("the solution could not be computed.\n");
                            exit(1);
                        }
                        free(ipiv);
                    }
                }
                //
                for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                    for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                        for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; j++) {
                            if(CHECK_INVALID(matRight[iL][iLS][n][i][j])) {
                                printf("%s(%d): Error - invalid value of matRight[%d][%d][%d] = %lf",
                                       __FILE__, __LINE__, n, i, j, matRight[iL][iLS][n][i][j]);
                                exit(-112);
                            }
                            frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->connectWeight[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i][j] =
                                matRight[iL][iLS][n][i][j];
                        }
                    }
                }
                dfrnn_Predict_DFRNN_init(frnn_MOP_Predict_dfrnn, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
                //
                for(int m = 0; m < tmp_offset_samp; m++) {
                    //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
                    //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
                    for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                        for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                            valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j] = 0;
                            for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; k++) {
                                int ind_cur = m * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput + k;
                                valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j] +=
                                    matA[iL][iLS][n][j][ind_cur] *
                                    frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->connectWeight[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j][k];
                            }
                            matBp[iL][iLS][n][j][m] = valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j];
                            if(CHECK_INVALID(valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j])) {
                                printf("%d~%lf", j, valOut[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + j]);
                            }
                        }
                    }
                    MY_FLT_TYPE* cur_out = valOut;
                    MY_FLT_TYPE** true_out = matB[iL][iLS][0];
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
                    if(!tag_classification_MOP_Predict_DFRNN) {
                        MY_FLT_TYPE tmp_loss = 0.0;
                        for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                            int tmp_o = ind_out_predict_MOP_Predict_DFRNN[n];
                            true_out = matB[iL][iLS][n];
                            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; i++)
                                tmp_loss += (cur_out[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i] - true_out[i][m]) *
                                            (cur_out[n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput + i] - true_out[i][m]) *
                                            trainFact_MOP_Predict_DFRNN[tmp_o];
                        }
                        frnn_MOP_Predict_dfrnn->sum_all++;
                        frnn_MOP_Predict_dfrnn->sum_wrong += tmp_loss / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput /
                                                             frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
                    } else {
                        int cur_label = 0;
                        MY_FLT_TYPE cur_val = valOut[0];
                        for(int j = 1; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                            if(cur_val < valOut[j]) {
                                cur_val = valOut[j];
                                cur_label = j;
                            }
                        }
                        int true_label = 0;
                        MY_FLT_TYPE true_val = true_out[0][m];
                        for(int j = 1; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                            if(true_val < true_out[j][m]) {
                                true_val = true_out[j][m];
                                true_label = j;
                            }
                        }
                        for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                            if(j == cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_TP[j]++;
                            if(j == cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_FP[j]++;
                            if(j != cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_FN[j]++;
                            if(j != cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_TN[j]++;
                        }
                        frnn_MOP_Predict_dfrnn->sum_all++;
                        frnn_MOP_Predict_dfrnn->N_sum[true_label]++;
                        if(cur_label != true_label) {
                            frnn_MOP_Predict_dfrnn->sum_wrong++;
                            frnn_MOP_Predict_dfrnn->N_wrong[true_label]++;
                        }
                    }
#else
                    int cur_label = 0;
                    MY_FLT_TYPE cur_val = valOut[0];
                    for(int j = 1; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        if(cur_val < valOut[j]) {
                            cur_val = valOut[j];
                            cur_label = j;
                        }
                    }
                    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
                        frnn_MOP_Predict_dfrnn->trading_actions[m + frnn_MOP_Predict_dfrnn->numInput - 1] = cur_label;
                    } else {
                        frnn_MOP_Predict_dfrnn->trading_actions[m] = cur_label;
                    }
                    int true_label = 0;
                    MY_FLT_TYPE true_val = true_out[0][m];
                    for(int j = 1; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        if(true_val < true_out[j][m]) {
                            true_val = true_out[j][m];
                            true_label = j;
                        }
                    }
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput; j++) {
                        if(j == cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_TP[j]++;
                        if(j == cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_FP[j]++;
                        if(j != cur_label && j == true_label) frnn_MOP_Predict_dfrnn->N_FN[j]++;
                        if(j != cur_label && j != true_label) frnn_MOP_Predict_dfrnn->N_TN[j]++;
                    }
                    frnn_MOP_Predict_dfrnn->sum_all++;
                    frnn_MOP_Predict_dfrnn->N_sum[true_label]++;
                    if(cur_label != true_label) {
                        frnn_MOP_Predict_dfrnn->sum_wrong++;
                        frnn_MOP_Predict_dfrnn->N_wrong[true_label]++;
                    }
#endif
                }
                //
            }
#endif
        }
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_val_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
        for(int iL = 0; iL < layerNum; iL++) {
            int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
            for(int iLS = 0; iLS < layerSize; iLS++) {
                int nMultiOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
                int nOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput;
                for(int n = 0; n < nMultiOut; n++) {
                    for(int i = 0; i < nOut; i++) {
                        free(matA[iL][iLS][n][i]);
                        free(matB[iL][iLS][n][i]);
                        free(matLeft[iL][iLS][n][i]);
                        free(matRight[iL][iLS][n][i]);
                    }
                    free(matA[iL][iLS][n]);
                    free(matB[iL][iLS][n]);
                    free(matLeft[iL][iLS][n]);
                    free(matRight[iL][iLS][n]);
                }
                free(matA[iL][iLS]);
                free(matB[iL][iLS]);
                free(matLeft[iL][iLS]);
                free(matRight[iL][iLS]);
            }
            free(matA[iL]);
            free(matB[iL]);
            free(matLeft[iL]);
            free(matRight[iL]);
        }
        free(matA);
        free(matB);
        free(matLeft);
        free(matRight);
        matA = NULL;
        matB = NULL;
        matLeft = NULL;
        matRight = NULL;
    }
#endif
    for(int iL = 0; iL < layerNum; iL++) {
        int layerSize = frnn_MOP_Predict_dfrnn->layerSize[iL];
        for(int iLS = 0; iLS < layerSize; iLS++) {
            int nMultiOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
            int nOut = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numOutput;
            for(int n = 0; n < nMultiOut; n++) {
                for(int i = 0; i < nOut; i++) {
                    free(matBp[iL][iLS][n][i]);
                }
                free(matBp[iL][iLS][n]);
            }
            free(matBp[iL][iLS]);
        }
        free(matBp[iL]);
    }
    free(matBp);
    matBp = NULL;
    //
    return;
}

void Finalize_MOP_Predict_DFRNN()
{
    dfrnn_Predict_DFRNN_free(frnn_MOP_Predict_dfrnn);
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
void dfrnn_Predict_DFRNN_setup(frnn_MOP_Predict_DFRNN* dfrnn)
{
    dfrnn->layerNum = num_layers_Predict_DFRNN;
    dfrnn->dfrnn = (frnn_block_Pred_dfrnn**)malloc(dfrnn->layerNum * sizeof(frnn_block_Pred_dfrnn*));
    for(int n = 0; n < dfrnn->layerNum; n++) {
        dfrnn->layerSize[n] = size_layers_Predict_DFRNN[n];
        if(flag_sep_in_MOP_Predict_DFRNN == FLAG_STATUS_ON && n == 0) dfrnn->layerSize[n] *= num_in__predict_MOP_Predict_DFRNN;
        dfrnn->dfrnn[n] = (frnn_block_Pred_dfrnn*)calloc(dfrnn->layerSize[n], sizeof(frnn_block_Pred_dfrnn));
    }
    //
    dfrnn->num_multiKindInput = num_in__predict_MOP_Predict_DFRNN;
    if(dfrnn->tag_multiKindInput == FLAG_STATUS_ON) {
        if(NORMALIZE_MOP_Predict_DFRNN != NORMALIZE_MIN_MAX_MOP_Predict_DFRNN &&
           NORMALIZE_MOP_Predict_DFRNN != NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
            printf("%s(%d): If different kinds of inputs are utilized, all inputs should be normalized, exiting...\n",
                   __FILE__, __LINE__);
            exit(-1759);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //
    dfrnn->numInput = 3;
    dfrnn->lenGap = 1;
    if(tag_classification_MOP_Predict_DFRNN)
        dfrnn->numInput = 1;
#if CURRENT_PROB_MOP_PREDICT_DFRNN == PREDICT_CLASSIFY_MOP_PREDICT_DFRNN
    int numOutput = 1;
    if(tag_classification_MOP_Predict_DFRNN)
        numOutput = num_class_MOP_Predict_DFRNN;
#else
    int numOutput = NUM_LABEL_TRADING_MOP_PREDICT_DFRNN;
#endif
    dfrnn->num_multiKindOutput = num_out_predict_MOP_Predict_DFRNN;
    if(dfrnn->tag_multiKindOutput == FLAG_STATUS_OFF)
        dfrnn->num_multiKindOutput = 1;
    dfrnn->numOutput = numOutput;
    //
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    if(!flag_T2_MOP_Predict_DFRNN) typeFuzzySet = FUZZY_SET_I;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE; // MUL_EACH_IN_TO_ONE_RULE; //
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;// FIXED_ROUGH_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int flagConnectStatus = FLAG_STATUS_OFF;
    int flagConnectWeight = FLAG_STATUS_OFF;
    if(dfrnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    if(dfrnn->num_multiKindOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    if(tag_classification_MOP_Predict_DFRNN) {
        consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
        //centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    //
    dfrnn->typeFuzzySet = typeFuzzySet;
    dfrnn->typeRules = typeRules;
    dfrnn->typeInRuleCorNum = typeInRuleCorNum;
    dfrnn->typeTypeReducer = typeTypeReducer;
    dfrnn->consequenceNodeStatus = consequenceNodeStatus;
    dfrnn->centroid_num_tag = centroid_num_tag;
    dfrnn->flagConnectStatus = flagConnectStatus;
    dfrnn->flagConnectWeight = flagConnectWeight;
    //
#if MF_RULE_NUM_MOP_PREDICT_DFRNN_CUR == MF_RULE_NUM_MOP_PREDICT_DFRNN_LESS
    int numFuzzyRules = 7;
    int numRoughSets = 3;// (int)sqrt(numFuzzyRules);
#else
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#endif
    if(numFuzzyRules > DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL)
        numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    //numRoughSets = 2 * frnn->numOutput * frnn->num_multiKindOutput;
    //numRoughSets = numFuzzyRules / 2;
    if(numRoughSets < 3)
        numRoughSets = 3;
    //
    int GEP_head_len = 8;
    dfrnn->GEP_head_len = GEP_head_len;
    //
    dfrnn->layerNum = 8;
    dfrnn->numRules = numFuzzyRules;
    dfrnn->numRoughs = numRoughSets;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    dfrnn->typeCoding = tmp_typeCoding;
    //
    int numInputConsequenceNode = 0;
    if(type_frnn_cnsq_MOP_Predict_DFRNN == FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_FIXED) {
        numInputConsequenceNode = 0;
        consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
        dfrnn->consequenceNodeStatus = consequenceNodeStatus;
    } else if(type_frnn_cnsq_MOP_Predict_DFRNN == FRNN_CONSEQUENCE_MOP_PREDICT_DFRNN_ADAPT) {
        if(flag_multi_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_OFF)
            numInputConsequenceNode = dfrnn->numInput;
        else
            numInputConsequenceNode = dfrnn->numInput * dfrnn->num_multiKindInput;
        consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
        dfrnn->consequenceNodeStatus = consequenceNodeStatus;
    } else {
        printf("%s(%d): Invalid ``type_frnn_cnsq_MOP_Predict_DFRNN''-%d, exiting...\n",
               __FILE__, __LINE__, type_frnn_cnsq_MOP_Predict_DFRNN);
        exit(-12759);
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    flagConnectStatus = FLAG_STATUS_OFF;
    dfrnn->flagConnectStatus = flagConnectStatus;
    flagConnectWeight = FLAG_STATUS_ON;
    dfrnn->flagConnectWeight = flagConnectWeight;
    tmp_typeCoding = PARA_CODING_DIRECT;
    dfrnn->typeCoding = tmp_typeCoding;
    //numInputConsequenceNode = 0;
    //consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
    //frnn->consequenceNodeStatus = consequenceNodeStatus;
    centroid_num_tag = CENTROID_ALL_ONESET;
    if(dfrnn->num_multiKindOutput > 1 || dfrnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
    }
    dfrnn->centroid_num_tag = centroid_num_tag;
    if(numRoughSets < numOutput) {
        numRoughSets = numOutput;
        dfrnn->numRoughs = numRoughSets;
    }
#endif
    //
    dfrnn->num_in_all_max = 0;
    dfrnn->num_out_all_max = 0;
    dfrnn->num_out_max = 0;
    dfrnn->num_out_mk_max = 0;
    dfrnn->num_in_OL_max = 0;
    for(int iL = 0; iL < dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < dfrnn->layerSize[iL]; iLS++) {
            int num_out_pre_utl = 0;
            if(iL == 0) {
                dfrnn->dfrnn[iL][iLS].numInput = dfrnn->numInput;
                if(flag_sep_in_MOP_Predict_DFRNN == FLAG_STATUS_ON)
                    dfrnn->dfrnn[iL][iLS].num_multiKindInput = 1;
                else
                    dfrnn->dfrnn[iL][iLS].num_multiKindInput = dfrnn->num_multiKindInput;
            } else {
                dfrnn->dfrnn[iL][iLS].numInput = dfrnn->dfrnn[iL - 1][0].numOutput * dfrnn->layerSize[iL - 1];
                dfrnn->dfrnn[iL][iLS].num_multiKindInput = dfrnn->dfrnn[iL - 1][0].num_multiKindOutput;
            }
            if(flag_multi_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_OFF && iL == 0)
                num_out_pre_utl = dfrnn->dfrnn[iL][iLS].numInput;
            else
                num_out_pre_utl = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
            int num_in_all_bk = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
            int num_in_all = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
            if(type_in_MOP_Predict_DFRNN == CONCAT_PRE_OUT_ORIG_IN_2_IN_MOP_PREDICT_DFRNN && iL)
                num_in_all += dfrnn->numInput;
            if(num_in_all > dfrnn->num_in_all_max) dfrnn->num_in_all_max = num_in_all;
            MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(num_in_all, sizeof(MY_FLT_TYPE));
            MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(num_in_all, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < num_in_all; i++) {
                if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                    inputMin[i] = 0;
                    inputMax[i] = 1;
                } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                    inputMin[i] = -10;
                    inputMax[i] = 10;
                } else {
                    inputMin[i] = 0;
                    inputMax[i] = 120;
                }
            }
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                dfrnn->dfrnn[iL][iLS].num_GEP = num_in_all;
                dfrnn->dfrnn[iL][iLS].GEP0 = (codingGEP**)malloc(dfrnn->dfrnn[iL][iLS].num_GEP * sizeof(codingGEP*));
                int num_in_GEP = 0;
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    if(n >= num_in_all_bk)
                        num_in_GEP = dfrnn->numInput;
                    else
                        num_in_GEP = dfrnn->dfrnn[iL][iLS].numInput;
                    dfrnn->dfrnn[iL][iLS].GEP0[n] = setupCodingGEP(num_in_GEP, inputMin, inputMax, 1, 0.5,
                                                    FLAG_STATUS_OFF,
                                                    dfrnn->GEP_head_len,
                                                    FLAG_STATUS_OFF,
                                                    PARA_MIN_VAL_GEP_CFRNN_MODEL,
                                                    PARA_MAX_VAL_GEP_CFRNN_MODEL);
                }
                for(int i = 0; i < num_in_all; i++) {
                    inputMin[i] = -10;
                    inputMax[i] = 10;
                }
            }
            if(dfrnn->tag_DIF == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_multiKindInput; n++) {
                    int tmp_ind_os = n * dfrnn->dfrnn[iL][iLS].numInput;
                    if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                        inputMin[tmp_ind_os + 0] = 0;
                        inputMax[tmp_ind_os + 0] = 1;
                    } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                        inputMin[tmp_ind_os + 0] = -10;
                        inputMax[tmp_ind_os + 0] = 10;
                    } else {
                        inputMin[tmp_ind_os + 0] = 0;
                        inputMax[tmp_ind_os + 0] = 120;
                    }
                    for(int i = 1; i < dfrnn->dfrnn[iL][iLS].numInput; i++) {
                        if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                            inputMin[tmp_ind_os + i] = -1;
                            inputMax[tmp_ind_os + i] = 1;
                        } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                            inputMin[tmp_ind_os + i] = -10;
                            inputMax[tmp_ind_os + i] = 10;
                        } else {
                            inputMin[tmp_ind_os + i] = -10;
                            inputMax[tmp_ind_os + i] = 10;
                        }
                    }
                }
                if(type_in_MOP_Predict_DFRNN == CONCAT_PRE_OUT_ORIG_IN_2_IN_MOP_PREDICT_DFRNN && iL) {
                    int tmp_ind_os = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
                    if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                        inputMin[tmp_ind_os + 0] = 0;
                        inputMax[tmp_ind_os + 0] = 1;
                    } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                        inputMin[tmp_ind_os + 0] = -10;
                        inputMax[tmp_ind_os + 0] = 10;
                    } else {
                        inputMin[tmp_ind_os + 0] = 0;
                        inputMax[tmp_ind_os + 0] = 120;
                    }
                    for(int i = 1; i < dfrnn->numInput; i++) {
                        if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                            inputMin[tmp_ind_os + i] = -1;
                            inputMax[tmp_ind_os + i] = 1;
                        } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                            inputMin[tmp_ind_os + i] = -10;
                            inputMax[tmp_ind_os + i] = 10;
                        } else {
                            inputMin[tmp_ind_os + i] = -10;
                            inputMax[tmp_ind_os + i] = 10;
                        }
                    }
                }
            }
            int* numMemship = (int*)calloc(num_in_all, sizeof(int));
            for(int i = 0; i < num_in_all; i++) {
#if MF_RULE_NUM_MOP_PREDICT_DFRNN_CUR == MF_RULE_NUM_MOP_PREDICT_DFRNN_LESS
                numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#else
                numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#endif
            }
            int* flagAdapMemship = (int*)calloc(num_in_all, sizeof(int));
            for(int i = 0; i < num_in_all; i++) {
                flagAdapMemship[i] = FLAG_STATUS_ON;
            }
            dfrnn->dfrnn[iL][iLS].M1 = setupMemberLayer(num_in_all, inputMin, inputMax,
                                       numMemship, flagAdapMemship, dfrnn->typeFuzzySet,
                                       tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, dfrnn->GEP_head_len, 1);
            dfrnn->dfrnn[iL][iLS].F2 = setupFuzzyLayer(num_in_all, dfrnn->dfrnn[iL][iLS].M1->numMembershipFun, dfrnn->numRules,
                                       dfrnn->typeFuzzySet, dfrnn->typeRules,
                                       dfrnn->typeInRuleCorNum, dfrnn->tag_GEPr,
                                       tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, dfrnn->GEP_head_len, FLAG_STATUS_OFF);
            dfrnn->dfrnn[iL][iLS].R3 = setupRoughLayer(dfrnn->numRules, dfrnn->numRoughs, dfrnn->typeFuzzySet,
                                       FLAG_STATUS_ON,
                                       tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, dfrnn->GEP_head_len, 1);
            MY_FLT_TYPE outputMin[MAX_OUT_NUM_MOP_Predict_DFRNN];
            MY_FLT_TYPE outputMax[MAX_OUT_NUM_MOP_Predict_DFRNN];
            dfrnn->dfrnn[iL][iLS].numOutput = dfrnn->numOutput;
            dfrnn->dfrnn[iL][iLS].num_multiKindOutput = dfrnn->num_multiKindOutput;
            int num_out_all = dfrnn->dfrnn[iL][iLS].numOutput * dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
            if(num_out_all > dfrnn->num_out_all_max) dfrnn->num_out_all_max = num_out_all;
            if(dfrnn->dfrnn[iL][iLS].numOutput > dfrnn->num_out_max)
                dfrnn->num_out_max = dfrnn->dfrnn[iL][iLS].numOutput;
            if(dfrnn->dfrnn[iL][iLS].num_multiKindOutput > dfrnn->num_out_mk_max)
                dfrnn->num_out_mk_max = dfrnn->dfrnn[iL][iLS].num_multiKindOutput;
            for(int i = 0; i < dfrnn->numOutput * dfrnn->num_multiKindOutput; i++) {
                if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                    outputMin[i] = 0;
                    outputMax[i] = 1;
                } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                    outputMin[i] = -10;
                    outputMax[i] = 10;
                } else {
                    outputMin[i] = 0;
                    outputMax[i] = 120;
                }
            }
            int numInputConsequenceNode_cur = 0;
            if(type_in_cnsq_MOP_Predict_DFRNN == PRE_OUT_2_IN_CNSQ_MOP_PREDICT_DFRNN)
                numInputConsequenceNode_cur = num_out_pre_utl;
            else if(type_in_cnsq_MOP_Predict_DFRNN == ORIG_IN_2_IN_CNSQ_MOP_PREDICT_DFRNN)
                numInputConsequenceNode_cur = numInputConsequenceNode;
            else
                numInputConsequenceNode_cur = num_out_pre_utl + numInputConsequenceNode;
            if(iL == 0)
                numInputConsequenceNode_cur = numInputConsequenceNode;
            MY_FLT_TYPE* inputMin_cnsq = (MY_FLT_TYPE*)calloc(numInputConsequenceNode_cur + 1, sizeof(MY_FLT_TYPE));
            MY_FLT_TYPE* inputMax_cnsq = (MY_FLT_TYPE*)calloc(numInputConsequenceNode_cur + 1, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < numInputConsequenceNode_cur; i++) {
                if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
                    inputMin_cnsq[i] = 0;
                    inputMax_cnsq[i] = 1;
                } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
                    inputMin_cnsq[i] = -10;
                    inputMax_cnsq[i] = 10;
                } else {
                    inputMin_cnsq[i] = 0;
                    inputMax_cnsq[i] = 120;
                }
            }
            dfrnn->dfrnn[iL][iLS].OL = setupOutReduceLayer(dfrnn->dfrnn[iL][iLS].R3->numRoughSets,
                                       dfrnn->numOutput * dfrnn->num_multiKindOutput,
                                       outputMin, outputMax,
                                       dfrnn->typeFuzzySet, dfrnn->typeTypeReducer,
                                       dfrnn->consequenceNodeStatus, dfrnn->centroid_num_tag,
                                       numInputConsequenceNode_cur, inputMin_cnsq, inputMax_cnsq,
                                       dfrnn->flagConnectStatus, dfrnn->flagConnectWeight,
                                       tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, dfrnn->GEP_head_len, 1);
            //
            free(inputMin);
            free(inputMax);
            free(numMemship);
            free(flagAdapMemship);
            free(inputMin_cnsq);
            free(inputMax_cnsq);
            //
            dfrnn->dfrnn[iL][iLS].dataflowMax = 0;
            dfrnn->dfrnn[iL][iLS].connectionMax = 0;
            if(dfrnn->tag_GEP) {
                for(int i = 0; i < dfrnn->dfrnn[iL][iLS].num_GEP; i++)
                    dfrnn->dfrnn[iL][iLS].connectionMax += dfrnn->dfrnn[iL][iLS].GEP0[i]->GEP_head_length;
            }
            if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                dfrnn->dfrnn[iL][iLS].dataflowMax += (MY_FLT_TYPE)(dfrnn->dfrnn[iL][iLS].numInput *
                                                     dfrnn->dfrnn[iL][iLS].num_multiKindInput *
                                                     dfrnn->numRules * dfrnn->numRoughs *
                                                     dfrnn->dfrnn[iL][iLS].numOutput *
                                                     dfrnn->dfrnn[iL][iLS].num_multiKindOutput);
                dfrnn->dfrnn[iL][iLS].connectionMax += (MY_FLT_TYPE)(dfrnn->dfrnn[iL][iLS].numInput *
                                                       dfrnn->dfrnn[iL][iLS].num_multiKindInput *
                                                       dfrnn->numRules +
                                                       dfrnn->numRules * dfrnn->numRoughs);
            } else {
                dfrnn->dfrnn[iL][iLS].dataflowMax += (MY_FLT_TYPE)(dfrnn->dfrnn[iL][iLS].M1->outputSize *
                                                     dfrnn->numRules * dfrnn->numRoughs *
                                                     dfrnn->dfrnn[iL][iLS].numOutput *
                                                     dfrnn->dfrnn[iL][iLS].num_multiKindOutput);
                dfrnn->dfrnn[iL][iLS].connectionMax += (MY_FLT_TYPE)(dfrnn->dfrnn[iL][iLS].M1->outputSize * dfrnn->numRules +
                                                       dfrnn->numRules * dfrnn->numRoughs);
            }
        }
    }
    //
    dfrnn->numParaLocal = 0;
    dfrnn->numParaLocal_disc = 0;
    dfrnn->layerNum = 4;
    for(int iL = 0; iL < dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < dfrnn->layerSize[iL]; iLS++) {
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    dfrnn->numParaLocal += dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal;
                }
            }
            dfrnn->numParaLocal += dfrnn->dfrnn[iL][iLS].M1->numParaLocal;
            dfrnn->numParaLocal += dfrnn->dfrnn[iL][iLS].F2->numParaLocal;
            dfrnn->numParaLocal += dfrnn->dfrnn[iL][iLS].R3->numParaLocal;
            dfrnn->numParaLocal += dfrnn->dfrnn[iL][iLS].OL->numParaLocal;
            //
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    dfrnn->numParaLocal_disc += dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal_disc;
                }
            }
            dfrnn->numParaLocal_disc += dfrnn->dfrnn[iL][iLS].M1->numParaLocal_disc;
            dfrnn->numParaLocal_disc += dfrnn->dfrnn[iL][iLS].F2->numParaLocal_disc;
            dfrnn->numParaLocal_disc += dfrnn->dfrnn[iL][iLS].R3->numParaLocal_disc;
            dfrnn->numParaLocal_disc += dfrnn->dfrnn[iL][iLS].OL->numParaLocal_disc;
        }
    }
    //
    int tmp_cnt_p = 0;
    dfrnn->xType = (int*)malloc(dfrnn->numParaLocal * sizeof(int));
    for(int iL = 0; iL < dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < dfrnn->layerSize[iL]; iLS++) {
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    memcpy(&dfrnn->xType[tmp_cnt_p], dfrnn->dfrnn[iL][iLS].GEP0[n]->xType,
                           dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal * sizeof(int));
                    tmp_cnt_p += dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal;
                }
            }
            memcpy(&dfrnn->xType[tmp_cnt_p], dfrnn->dfrnn[iL][iLS].M1->xType, dfrnn->dfrnn[iL][iLS].M1->numParaLocal * sizeof(int));
            tmp_cnt_p += dfrnn->dfrnn[iL][iLS].M1->numParaLocal;
            memcpy(&dfrnn->xType[tmp_cnt_p], dfrnn->dfrnn[iL][iLS].F2->xType, dfrnn->dfrnn[iL][iLS].F2->numParaLocal * sizeof(int));
            tmp_cnt_p += dfrnn->dfrnn[iL][iLS].F2->numParaLocal;
            memcpy(&dfrnn->xType[tmp_cnt_p], dfrnn->dfrnn[iL][iLS].R3->xType, dfrnn->dfrnn[iL][iLS].R3->numParaLocal * sizeof(int));
            tmp_cnt_p += dfrnn->dfrnn[iL][iLS].R3->numParaLocal;
            memcpy(&dfrnn->xType[tmp_cnt_p], dfrnn->dfrnn[iL][iLS].OL->xType, dfrnn->dfrnn[iL][iLS].OL->numParaLocal * sizeof(int));
            tmp_cnt_p += dfrnn->dfrnn[iL][iLS].OL->numParaLocal;
        }
    }
    //tmp_cnt_p = 0;
    //for(int i = 0; i < cfrnn->numParaLocal; i++) {
    //    if(cfrnn->xType[i] != VAR_TYPE_CONTINUOUS)
    //        tmp_cnt_p++;
    //}
    //printf("%d ~ %d \n", tmp_cnt_p, cfrnn->numParaLocal_disc);

    dfrnn->e = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));

    dfrnn->N_sum = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));
    dfrnn->N_wrong = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));
    dfrnn->e_sum = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));

    dfrnn->N_TP = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));
    dfrnn->N_TN = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));
    dfrnn->N_FP = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));
    dfrnn->N_FN = (MY_FLT_TYPE*)calloc(dfrnn->num_out_all_max, sizeof(MY_FLT_TYPE));

    dfrnn->money_in_hand = 100000;
    dfrnn->trading_actions = (int*)calloc(MAX_DATA_LEN_MOP_PREDICT_DFRNN, sizeof(int));
    dfrnn->num_stock_held = 0;

    dfrnn->featureMapTagInitial = (int*)calloc(dfrnn->num_in_all_max, sizeof(int));
    dfrnn->dataflowInitial = (MY_FLT_TYPE*)calloc(dfrnn->num_in_all_max, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < dfrnn->num_in_all_max; i++) {
        dfrnn->featureMapTagInitial[i] = 1;
        dfrnn->dataflowInitial[i] = 1;
    }

    dfrnn->dataflowMax = 0;
    dfrnn->connectionMax = 0;
    for(int i = 0; i < dfrnn->layerNum; i++) {
        for(int j = 0; j < dfrnn->layerSize[i]; j++) {
            dfrnn->dataflowMax += dfrnn->dfrnn[i][j].dataflowMax;
            dfrnn->connectionMax += dfrnn->dfrnn[i][j].connectionMax;
        }
    }
    //
    return;
}

void dfrnn_Predict_DFRNN_free(frnn_MOP_Predict_DFRNN* dfrnn)
{
    for(int iL = 0; iL < dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < dfrnn->layerSize[iL]; iLS++) {
            freeOutReduceLayer(dfrnn->dfrnn[iL][iLS].OL);
            freeRoughLayer(dfrnn->dfrnn[iL][iLS].R3);
            freeFuzzyLayer(dfrnn->dfrnn[iL][iLS].F2);
            freeMemberLayer(dfrnn->dfrnn[iL][iLS].M1);
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int i = 0; i < dfrnn->dfrnn[iL][iLS].num_GEP; i++) {
                    freeCodingGEP(dfrnn->dfrnn[iL][iLS].GEP0[i]);
                }
                free(dfrnn->dfrnn[iL][iLS].GEP0);
            }
        }
    }

    free(dfrnn->xType);

    free(dfrnn->e);

    free(dfrnn->N_sum);
    free(dfrnn->N_wrong);
    free(dfrnn->e_sum);

    free(dfrnn->N_TP);
    free(dfrnn->N_TN);
    free(dfrnn->N_FP);
    free(dfrnn->N_FN);

    free(dfrnn->trading_actions);

    free(dfrnn->featureMapTagInitial);
    free(dfrnn->dataflowInitial);

    free(dfrnn);

    return;
}

void dfrnn_Predict_DFRNN_init(frnn_MOP_Predict_DFRNN* dfrnn, double* x, int mode)
{
    int count = 0;
    switch(mode) {
    case INIT_MODE_FRNN:
    case ASSIGN_MODE_FRNN:
    case OUTPUT_ALL_MODE_FRNN:
    case OUTPUT_CONTINUOUS_MODE_FRNN:
    case OUTPUT_DISCRETE_MODE_FRNN:
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    for(int iL = 0; iL < dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < dfrnn->layerSize[iL]; iLS++) {
            if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
                    assignCodingGEP(dfrnn->dfrnn[iL][iLS].GEP0[n], &x[count], mode);
                    count += dfrnn->dfrnn[iL][iLS].GEP0[n]->numParaLocal;
                }
            }
            assignMemberLayer(dfrnn->dfrnn[iL][iLS].M1, &x[count], mode);
            count += dfrnn->dfrnn[iL][iLS].M1->numParaLocal;
            assignFuzzyLayer(dfrnn->dfrnn[iL][iLS].F2, &x[count], mode);
            count += dfrnn->dfrnn[iL][iLS].F2->numParaLocal;
            assignRoughLayer(dfrnn->dfrnn[iL][iLS].R3, &x[count], mode);
            count += dfrnn->dfrnn[iL][iLS].R3->numParaLocal;
            assignOutReduceLayer(dfrnn->dfrnn[iL][iLS].OL, &x[count], mode);
            count += dfrnn->dfrnn[iL][iLS].OL->numParaLocal;
        }
    }
    //
    return;
}

void ff_frnn_Predict_DFRNN(frnn_MOP_Predict_DFRNN* dfrnn, int iL, int iLS, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                           MY_FLT_TYPE* inputConsequenceNode)
{
    for(int i = 0; i < dfrnn->num_in_all_max; i++) {
        dfrnn->dataflowInitial[i] = 1;
    }
    if(iL) {
        for(int n = 0; n < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_multiKindInput; n++) {
            int tmp_ind_os = n * frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numInput;
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].numInput; i++) {
                int i_in_n = i / frnn_MOP_Predict_dfrnn->dfrnn[iL - 1][i].numOutput;
                int i_in = i % frnn_MOP_Predict_dfrnn->dfrnn[iL - 1][i].numOutput;
                int tmp_ind = n * frnn_MOP_Predict_dfrnn->dfrnn[iL - 1][i].numOutput + i_in;
                dfrnn->dataflowInitial[tmp_ind_os + i] = dfrnn->dfrnn[iL - 1][i_in_n].OL->dataflowStatus[tmp_ind];
            }
        }
    }
    //
    int n_rep = dfrnn->layerSize[0] / dfrnn->num_multiKindInput;
    int i_att = ind_out_predict_MOP_Predict_DFRNN[0];
    if(flag_sep_in_MOP_Predict_DFRNN == FLAG_STATUS_ON && iL == 0)
        i_att = ind_out_predict_MOP_Predict_DFRNN[iLS / n_rep];
    //
    int len_valIn_bk = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
    int len_valIn = len_valIn_bk;
    if(type_in_MOP_Predict_DFRNN == CONCAT_PRE_OUT_ORIG_IN_2_IN_MOP_PREDICT_DFRNN && iL)
        len_valIn += dfrnn->numInput;
    MY_FLT_TYPE* tmpIn = (MY_FLT_TYPE*)malloc(len_valIn * sizeof(MY_FLT_TYPE));
    if(flag_sep_in_MOP_Predict_DFRNN == FLAG_STATUS_ON && iL == 0)
        memcpy(tmpIn, &valIn[i_att * dfrnn->numInput], len_valIn_bk * sizeof(MY_FLT_TYPE));
    else
        memcpy(tmpIn, valIn, len_valIn_bk * sizeof(MY_FLT_TYPE));
    if(len_valIn > len_valIn_bk)
        memcpy(&tmpIn[len_valIn_bk], &inputConsequenceNode[i_att * dfrnn->numInput], (len_valIn - len_valIn_bk) * sizeof(MY_FLT_TYPE));
    if(dfrnn->tag_GEP == FLAG_STATUS_ON) {
        MY_FLT_TYPE* tmpOut = (MY_FLT_TYPE*)malloc(dfrnn->dfrnn[iL][iLS].num_GEP * sizeof(MY_FLT_TYPE));
        for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_GEP; n++) {
            int tmp_ind = n / dfrnn->dfrnn[iL][iLS].numInput;
            if(n >= len_valIn_bk) {
                tmp_ind = (n - len_valIn_bk) / dfrnn->numInput;
                decodingGEP(dfrnn->dfrnn[iL][iLS].GEP0[n], &tmpIn[len_valIn_bk + tmp_ind * dfrnn->numInput], &tmpOut[n]);
            } else {
                decodingGEP(dfrnn->dfrnn[iL][iLS].GEP0[n], &tmpIn[tmp_ind * dfrnn->dfrnn[iL][iLS].numInput], &tmpOut[n]);
            }
            //printf("%lf ", tmpOut[n]);
        }
        ff_memberLayer(dfrnn->dfrnn[iL][iLS].M1, tmpOut, dfrnn->dataflowInitial);
        free(tmpOut);
    } else if(dfrnn->tag_DIF == FLAG_STATUS_ON) {
        for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_multiKindInput; n++) {
            int tmp_ind_os = n * dfrnn->dfrnn[iL][iLS].numInput;
            for(int i = 1; i < dfrnn->dfrnn[iL][iLS].numInput; i++) {
                tmpIn[tmp_ind_os + i] = valIn[tmp_ind_os + i - 1] - valIn[tmp_ind_os + i];
            }
        }
        for(int a = len_valIn_bk; a < len_valIn; a++) {
            int i = (a - len_valIn_bk) % dfrnn->numInput;
            if(i) tmpIn[a] = valIn[a - 1] - valIn[a];
        }
        ff_memberLayer(dfrnn->dfrnn[iL][iLS].M1, tmpIn, dfrnn->dataflowInitial);
    } else {
        ff_memberLayer(dfrnn->dfrnn[iL][iLS].M1, tmpIn, dfrnn->dataflowInitial);
    }
    free(tmpIn);
    //
    ff_fuzzyLayer(dfrnn->dfrnn[iL][iLS].F2, dfrnn->dfrnn[iL][iLS].M1->degreeMembership, dfrnn->dfrnn[iL][iLS].M1->dataflowStatus);
    ff_roughLayer(dfrnn->dfrnn[iL][iLS].R3, dfrnn->dfrnn[iL][iLS].F2->degreeRules, dfrnn->dfrnn[iL][iLS].F2->dataflowStatus);
    //
    if(dfrnn->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        if(iL == 0) {
            if(flag_multi_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_ON) {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                    for(int i = 0; i < dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                        memcpy(dfrnn->dfrnn[iL][iLS].OL->inputConsequenceNode[n * dfrnn->dfrnn[iL][iLS].numOutput + i],
                               inputConsequenceNode,
                               dfrnn->dfrnn[iL][iLS].OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
                    }
                }
            } else {
                for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                    int tmp_o = ind_out_predict_MOP_Predict_DFRNN[n];
                    for(int i = 0; i < dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                        memcpy(dfrnn->dfrnn[iL][iLS].OL->inputConsequenceNode[n * dfrnn->dfrnn[iL][iLS].numOutput + i],
                               &inputConsequenceNode[tmp_o * dfrnn->numInput],
                               dfrnn->dfrnn[iL][iLS].OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
                    }
                }
            }
        } else {
            for(int n = 0; n < dfrnn->dfrnn[iL][iLS].num_multiKindOutput; n++) {
                int tmp_o = ind_out_predict_MOP_Predict_DFRNN[n];
                for(int i = 0; i < dfrnn->dfrnn[iL][iLS].numOutput; i++) {
                    int ttt_len = 0;
                    if(type_in_cnsq_MOP_Predict_DFRNN != ORIG_IN_2_IN_CNSQ_MOP_PREDICT_DFRNN) {
                        ttt_len = dfrnn->dfrnn[iL][iLS].numInput * dfrnn->dfrnn[iL][iLS].num_multiKindInput;
                        memcpy(dfrnn->dfrnn[iL][iLS].OL->inputConsequenceNode[n * dfrnn->dfrnn[iL][iLS].numOutput + i],
                               valIn,
                               ttt_len * sizeof(MY_FLT_TYPE));
                    }
                    if(type_in_cnsq_MOP_Predict_DFRNN != PRE_OUT_2_IN_CNSQ_MOP_PREDICT_DFRNN) {
                        if(flag_multi_in_cnsq_MOP_Predict_DFRNN == FLAG_STATUS_ON)
                            memcpy(&dfrnn->dfrnn[iL][iLS].OL->inputConsequenceNode[n * dfrnn->dfrnn[iL][iLS].numOutput + i][ttt_len],
                                   inputConsequenceNode,
                                   (dfrnn->dfrnn[iL][iLS].OL->numInputConsequenceNode - ttt_len) * sizeof(MY_FLT_TYPE));
                        else
                            memcpy(&dfrnn->dfrnn[iL][iLS].OL->inputConsequenceNode[n * dfrnn->dfrnn[iL][iLS].numOutput + i][ttt_len],
                                   &inputConsequenceNode[tmp_o * dfrnn->numInput],
                                   (dfrnn->dfrnn[iL][iLS].OL->numInputConsequenceNode - ttt_len) * sizeof(MY_FLT_TYPE));
                    }
                }
            }
        }
    }
    ff_outReduceLayer(dfrnn->dfrnn[iL][iLS].OL, dfrnn->dfrnn[iL][iLS].R3->degreeRough, dfrnn->dfrnn[iL][iLS].R3->dataflowStatus);
    //for(int i = 0; i < frnn->OL->numOutput; i++) {
    //    if(CHECK_INVALID(frnn->OL->valOutputFinal[i])) {
    //        printf("%s(%d): Invalid output %d ~ %lf, exiting...\n",
    //               __FILE__, __LINE__, i, frnn->OL->valOutputFinal[i]);
    //        print_para_memberLayer(frnn->M1);
    //        print_data_memberLayer(frnn->M1);
    //        print_para_fuzzyLayer(frnn->F2);
    //        print_data_fuzzyLayer(frnn->F2);
    //        print_para_roughLayer(frnn->R3);
    //        print_data_roughLayer(frnn->R3);
    //        print_para_outReduceLayer(frnn->OL);
    //        print_data_outReduceLayer(frnn->OL);
    //        exit(-94628);
    //    }
    //}
    //
    memcpy(valOut, dfrnn->dfrnn[iL][iLS].OL->valOutputFinal, dfrnn->dfrnn[iL][iLS].OL->numOutput * sizeof(MY_FLT_TYPE));
    //
    return;
}

static double simplicity_MOP_Predict_DFRNN()
{
    //
    double f_simpl = 0.0;
    int n_frnn_block = 0;
    double f_simpl_gl = 0.0;
    double f_simpl_fl = 0.0;
    double f_simpl_rl = 0.0;
    total_penalty_MOP_Predict_DFRNN = 0.0;
    //
    for(int iL = 0; iL < frnn_MOP_Predict_dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < frnn_MOP_Predict_dfrnn->layerSize[iL]; iLS++) {
            n_frnn_block++;
            f_simpl_gl = 0;
            f_simpl_fl = 0;
            f_simpl_rl = 0;
            int *tmp_rule, *tmp_rough, **tmp_mem;
            tmp_rule = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules, sizeof(int));
            tmp_rough = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets, sizeof(int));
            tmp_mem = (int**)malloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput * sizeof(int*));
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; i++) {
                tmp_mem[i] = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[i], sizeof(int));
            }
            if(frnn_MOP_Predict_dfrnn->tag_GEP) {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_GEP; i++) {
                    int tmp_g = 0;
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[i]->check_head; j++) {
                        if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[i]->check_op[j] >= 0) {
                            tmp_g++;
                        }
                    }
                    f_simpl_gl += (double)tmp_g / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].GEP0[i]->GEP_head_length;
                }
            }
            if(frnn_MOP_Predict_dfrnn->tag_GEPr == FLAG_STATUS_OFF) {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                    tmp_rule[i] = 0;
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; j++) {
                        int tmp_count = 0;
                        for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[j]; k++) {
                            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->connectStatusAll[i][j][k]) {
                                tmp_count++;
                                tmp_mem[j][k]++;
                            }
                        }
                        if(tmp_count) {
                            tmp_rule[i]++;
                        }
                    }
                    f_simpl_fl += (double)tmp_rule[i] / 9; // frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput;
                }
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                    tmp_rough[i] = 0;
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; j++) {
                        if(tmp_rule[j] && frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->connectStatus[i][j]) {
                            tmp_rough[i]++;
                        }
                    }
                    f_simpl_rl += (double)tmp_rough[i] / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules;
                }
            } else {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; j++) {
                        for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[j]; k++) {
                            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->connectStatusAll[i][j][k]) {
                                tmp_mem[j][k]++;
                            }
                        }
                    }
                    //
                    tmp_rule[i] = 0;
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_head; j++) {
                        if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_op[j] >= 0) {
                            tmp_rule[i]++;
                        }
                    }
                    f_simpl_fl += (double)tmp_rule[i] / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->GEP_head_length;
                }
                //
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                    tmp_rough[i] = 0;
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; j++) {
                        if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->connectStatus[i][j]) {
                            tmp_rough[i]++;
                        }
                    }
                    f_simpl_rl += (double)tmp_rough[i] / frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules;
                }
            }
            if(frnn_MOP_Predict_dfrnn->tag_GEP)
                f_simpl += (f_simpl_gl + f_simpl_fl + f_simpl_rl) /
                           (frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].num_GEP + frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules +
                            frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets);
            else
                f_simpl += (f_simpl_fl + f_simpl_rl) /
                           (frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules + frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets);
            //
            //if (flag_no_fuzzy_rule) {
            //  f_prcsn += 1e6;
            //  f_simpl += 1e6;
            //  f_normp += 1e6;
            //}
            int tmp_sum = 0;
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                tmp_sum += tmp_rule[i];
            }
            if(tmp_sum == 0) {
                total_penalty_MOP_Predict_DFRNN += penaltyVal_MOP_Predict_DFRNN;
            }
            //tmp_sum = 0;
            //for(int i = 0; i < NUM_CLASS_MOP_Predict_DFRNN; i++) {
            //    tmp_sum += tmp2[i];
            //}
            //if(tmp_sum == 0.0) {
            //    f_prcsn += 1e6;
            //    f_simpl += 1e6;
            //}
            tmp_sum = 0;
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                if(tmp_rough[i])
                    tmp_sum++;
                //tmp_sum += tmp_rough[i];
                //if(tmp_rough[i] == 0)
                //    total_penalty_MOP_Predict_DFRNN += penaltyVal_MOP_Predict_DFRNN;
            }
            if(tmp_sum < THRESHOLD_NUM_ROUGH_NODES_Pred_DFRNN) {
                total_penalty_MOP_Predict_DFRNN += penaltyVal_MOP_Predict_DFRNN * (THRESHOLD_NUM_ROUGH_NODES_Pred_DFRNN - tmp_sum);
            }
            //
            free(tmp_rule);
            free(tmp_rough);
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; i++) {
                free(tmp_mem[i]);
            }
            free(tmp_mem);
        }
    }
    //
    return f_simpl / n_frnn_block;
}

static double generality_MOP_Predict_DFRNN()
{
    double tmp_sum = 0.0;
    int tmp_cnt = 0;
    //
    for(int iL = 0; iL < frnn_MOP_Predict_dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < frnn_MOP_Predict_dfrnn->layerSize[iL]; iLS++) {
            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numOutput; i++) {
                    if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                        continue;
                    }
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; j++) {
                        for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->dim_degree; k++) {
                            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                                continue;
                            }
                            for(int m = 0; m < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInputConsequenceNode; m++) {
                                tmp_sum += fabs(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->paraConsequenceNode[i][j][k][m]);
                                tmp_cnt++;
                            }
                        }
                    }
                }
                //printf("Tag 1\n");
            }
            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->flagConnectWeightAdap == FLAG_STATUS_ON) {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numOutput; i++) {
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->numInput; j++) {
                        tmp_sum += fabs(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL->connectWeight[i][j]);
                        tmp_cnt++;
                    }
                }
                //printf("Tag 2\n");
            }
        }
    }
    if(tmp_cnt)
        tmp_sum /= tmp_cnt;
    //
    return tmp_sum;
}

static double get_profit_MOP_Predict_DFRNN(int tag_train_test)
{
    //
    frnn_MOP_Predict_dfrnn->money_init = 100000;
    frnn_MOP_Predict_dfrnn->money_in_hand = 100000;
    frnn_MOP_Predict_dfrnn->num_stock_held = 0;
    //
    double total_profit = 0;
    double* all_close_prices = NULL;
    int num_close_prices = 0;
    int* trading_actions = frnn_MOP_Predict_dfrnn->trading_actions;
    //
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_DFRNN) {
        all_close_prices = allData_MOP_Predict_DFRNN[0];
        num_close_prices = trainDataSize_MOP_Predict_DFRNN - frnn_MOP_Predict_dfrnn->numInput + 1;
    } else if(tag_train_test == VAL_TAG_MOP_PREDICT_DFRNN) {
        all_close_prices = &allData_MOP_Predict_DFRNN[0][trainDataSize_MOP_Predict_DFRNN];
        num_close_prices = valDataSize_MOP_Predict_DFRNN;
    } else {
        all_close_prices = &allData_MOP_Predict_DFRNN[0][trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN];
        num_close_prices = testDataSize_MOP_Predict_DFRNN;
    }
    //
    for(int i = 0; i < num_close_prices; i++) {
        int cur_i = i + frnn_MOP_Predict_dfrnn->numInput - 1;
        if(tag_train_test != TRAIN_TAG_MOP_PREDICT_DFRNN)
            cur_i = i;
        if(trading_actions[cur_i] == CLASS_IND_BUY_MOP_PREDICT_DFRNN) {
            if(frnn_MOP_Predict_dfrnn->money_in_hand > all_close_prices[cur_i]) {
                frnn_MOP_Predict_dfrnn->money_in_hand -= all_close_prices[cur_i];
                frnn_MOP_Predict_dfrnn->num_stock_held++;
            }
        } else if(trading_actions[cur_i] == CLASS_IND_SELL_MOP_PREDICT_DFRNN) {
            if(frnn_MOP_Predict_dfrnn->num_stock_held > 0) {
                frnn_MOP_Predict_dfrnn->money_in_hand += all_close_prices[cur_i];
                frnn_MOP_Predict_dfrnn->num_stock_held--;
            }
        }
    }
    //
    total_profit = (frnn_MOP_Predict_dfrnn->money_in_hand - frnn_MOP_Predict_dfrnn->money_init) /
                   frnn_MOP_Predict_dfrnn->money_init;
    total_profit = 1 - total_profit;
    //
    return total_profit;
}

void statistics_MOP_Predict_DFRNN(FILE* fpt)
{
    for(int iL = 0; iL < frnn_MOP_Predict_dfrnn->layerNum; iL++) {
        for(int iLS = 0; iLS < frnn_MOP_Predict_dfrnn->layerSize[iL]; iLS++) {
            //
            print_para_memberLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1);
            print_data_memberLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1);
            print_para_fuzzyLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2);
            print_data_fuzzyLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2);
            print_para_roughLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3);
            print_data_roughLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3);
            print_para_outReduceLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL);
            print_data_outReduceLayer(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].OL);
            //
            int *tmp_rule, **tmp_mem;
            tmp_rule = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules, sizeof(int));
            tmp_mem = (int**)malloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput * sizeof(int*));
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; i++) {
                tmp_mem[i] = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[i], sizeof(int));
            }
            int *tmp_rough, *tmp_rough_op, *tmp_rough_in;
            tmp_rough = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets, sizeof(int));
            tmp_rough_op = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets, sizeof(int));
            tmp_rough_in = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets, sizeof(int));
            int *tmp_rule_op, *tmp_rule_in;
            tmp_rule_op = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules, sizeof(int));
            tmp_rule_in = (int*)calloc(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules, sizeof(int));
            //
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; j++) {
                    if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->connectStatus[i][j]) {
                        tmp_rule[j]++;
                    }
                }
            }
            //
            if(frnn_MOP_Predict_dfrnn->tag_GEPr == FLAG_STATUS_OFF) {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; j++) {
                        for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[j]; k++) {
                            if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->connectStatusAll[i][j][k]) {
                                tmp_mem[j][k] += tmp_rule[i];
                                tmp_rule_in[i]++;
                            }
                        }
                    }
                }
            } else {
                for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                    for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_tail; j++) {
                        if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_vInd[j] >= 0) {
                            tmp_rule_in[i]++;
                            int cur_in = frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_vInd[j];
                            for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[cur_in]; k++) {
                                if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->connectStatusAll[i][cur_in][k]) {
                                    tmp_mem[cur_in][k] += tmp_rule[i];
                                    break;
                                }
                            }
                        }
                        if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->ruleGEP[i]->check_op[j] >= 0) {
                            tmp_rule_op[i]++;
                        }
                    }
                }
            }
            //
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; j++) {
                    if(frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->connectStatus[i][j]) {
                        tmp_rough[i]++;
                        tmp_rough_op[i] += tmp_rule_op[j];
                        tmp_rough_in[i] += tmp_rule_in[j];
                    }
                }
            }
            //
            for(int j = 0; j < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; j++) {
                for(int k = 0; k < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numMembershipFun[j]; k++) {
                    printf("%d,", tmp_mem[j][k]);
                    fprintf(fpt, "%d,", tmp_mem[j][k]);
                }
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                printf("%d,", tmp_rule[i]);
                fprintf(fpt, "%d,", tmp_rule[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                printf("%d,", tmp_rule_op[i]);
                fprintf(fpt, "%d,", tmp_rule_op[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].F2->numRules; i++) {
                printf("%d,", tmp_rule_in[i]);
                fprintf(fpt, "%d,", tmp_rule_in[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                printf("%d,", tmp_rough[i]);
                fprintf(fpt, "%d,", tmp_rough[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                printf("%d,", tmp_rough_op[i]);
                fprintf(fpt, "%d,", tmp_rough_op[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].R3->numRoughSets; i++) {
                printf("%d,", tmp_rough_in[i]);
                fprintf(fpt, "%d,", tmp_rough_in[i]);
            }
            printf("\n");
            fprintf(fpt, "\n");
            //
            if(frnn_MOP_Predict_dfrnn->tag_GEP) {
            }
            //
            free(tmp_rule);
            for(int i = 0; i < frnn_MOP_Predict_dfrnn->dfrnn[iL][iLS].M1->numInput; i++) {
                free(tmp_mem[i]);
            }
            free(tmp_mem);
            free(tmp_rough);
            free(tmp_rough_op);
            free(tmp_rough_in);
            //
            free(tmp_rule_op);
            free(tmp_rule_in);
        }
    }
    //
    return;
}

static void readData_stock_MOP_Predict_DFRNN(char* fname, int trainNo, int testNo, int endNo)
{
    FILE* fpt;
    if((fpt = fopen(fname, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }
    trainDataSize_MOP_Predict_DFRNN = 0;
    testDataSize_MOP_Predict_DFRNN = 0;
    //
    char tmp_delim[] = " ,\t\r\n";
    int max_buf_size = 1000 * 20 + 1;
    char* buf = (char*)malloc(max_buf_size * sizeof(char));
    char* p;
    //
    char StrLine[1024];
    int seq = 0;
    for(seq = 1; seq < trainNo; seq++) {
        // fgets(StrLine, 1024, fpt);
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
    }
    for(seq = trainNo; seq < testNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_MOP_Predict_DFRNN(StrLine);
        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        int tmp_size_pre = 0;
        int tmp_size = 0;
        for(int iK = 0; iK < numAttr_MOP_Predict_DFRNN; iK++) {
            if(!fgets(buf, max_buf_size, fpt_data)) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2001);
            }
            int tmp_cnt = -1;
            double elem;
            for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                if(tmp_cnt == -1) {
                    if(sscanf(p, "%d", &tmp_size) != 1) {
                        printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                        exit(2002);
                    }
                    if(iK && tmp_size != tmp_size_pre) {
                        printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                               __FILE__, __LINE__, iK, tmp_size, tmp_size_pre);
                        exit(2003);
                    }
                } else {
                    if(sscanf(p, "%lf", &elem) != 1) {
                        printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                        exit(2004);
                    }
                    allData_MOP_Predict_DFRNN[iK][trainDataSize_MOP_Predict_DFRNN + tmp_cnt] = elem;
                }
                tmp_cnt++;
            }
            if(tmp_size != tmp_cnt) {
                printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                       __FILE__, __LINE__, iK, tmp_size, tmp_cnt);
                exit(2005);
            }
            tmp_size_pre = tmp_size;
        }
        trainDataSize_MOP_Predict_DFRNN += tmp_size;
        fclose(fpt_data);
    }
    for(seq = testNo; seq < endNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_MOP_Predict_DFRNN(StrLine);
        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        int tmp_size_pre = 0;
        int tmp_size = 0;
        for(int iK = 0; iK < numAttr_MOP_Predict_DFRNN; iK++) {
            if(!fgets(buf, max_buf_size, fpt_data)) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2001);
            }
            int tmp_cnt = -1;
            double elem;
            for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                if(tmp_cnt == -1) {
                    if(sscanf(p, "%d", &tmp_size) != 1) {
                        printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                        exit(2002);
                    }
                    if(iK && tmp_size != tmp_size_pre) {
                        printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                               __FILE__, __LINE__, iK, tmp_size, tmp_size_pre);
                        exit(2003);
                    }
                } else {
                    if(sscanf(p, "%lf", &elem) != 1) {
                        printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                        exit(2004);
                    }
                    allData_MOP_Predict_DFRNN[iK][trainDataSize_MOP_Predict_DFRNN +
                                                  testDataSize_MOP_Predict_DFRNN + tmp_cnt] = elem;
                }
                tmp_cnt++;
            }
            if(tmp_size != tmp_cnt) {
                printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                       __FILE__, __LINE__, iK, tmp_size, tmp_cnt);
                exit(2005);
            }
            tmp_size_pre = tmp_size;
        }
        testDataSize_MOP_Predict_DFRNN += tmp_size;
        fclose(fpt_data);
    }
    allDataSize_MOP_Predict_DFRNN = trainDataSize_MOP_Predict_DFRNN + testDataSize_MOP_Predict_DFRNN;
    valDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN * 0.2;
    testDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN * 0.2;
    trainDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN -
                                      valDataSize_MOP_Predict_DFRNN -
                                      testDataSize_MOP_Predict_DFRNN;
    //
    free(buf);
    fclose(fpt);
}

static void readData_general_MOP_Predict_DFRNN(char* fname, int tag_classification)
{
    FILE* fpt;
    if((fpt = fopen(fname, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }
    allDataSize_MOP_Predict_DFRNN = 0;
    trainDataSize_MOP_Predict_DFRNN = 0;
    testDataSize_MOP_Predict_DFRNN = 0;
    //
    char tmp_delim[] = " ,\t\r\n";
    int max_buf_size = 100 * MAX_ATTR_NUM_Pred_DFRNN + 1;
    char* buf = (char*)malloc(max_buf_size * sizeof(char));
    char* p;
    int tmp_cnt;
    int elem_int;
    double elem;
    // get size
    if(fgets(buf, max_buf_size, fpt) == NULL) {
        printf("%s(%d): No  line\n", __FILE__, __LINE__);
        exit(-1);
    }
    tmp_cnt = 0;
    for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
        if(sscanf(p, "%d", &elem_int) != 1) {
            printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
            exit(1001);
        }
        if(tmp_cnt == 0) {
            allDataSize_MOP_Predict_DFRNN = elem_int;
        } else if(tmp_cnt == 1) {
            numAttr_MOP_Predict_DFRNN = elem_int;
        } else {
            if(tag_classification && tmp_cnt == 2) {
                num_class_MOP_Predict_DFRNN = elem_int;
            } else {
                printf("\n%s(%d):too many data...\n", __FILE__, __LINE__);
                exit(1002);
            }
        }
        tmp_cnt++;
    }
    //get data
    int seq = 0;
    for(seq = 0; seq < allDataSize_MOP_Predict_DFRNN; seq++) {
        if(fgets(buf, max_buf_size, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        tmp_cnt = 0;
        for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
            if(sscanf(p, "%lf", &elem) != 1) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2004);
            }
            allData_MOP_Predict_DFRNN[tmp_cnt][seq] = elem;
            tmp_cnt++;
        }
        if(numAttr_MOP_Predict_DFRNN != tmp_cnt) {
            printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                   __FILE__, __LINE__, seq, numAttr_MOP_Predict_DFRNN, tmp_cnt);
            exit(2005);
        }
    }
    //
    if(tag_classification)
        numAttr_MOP_Predict_DFRNN--;
    //
    if(!tag_classification) {
        //trainDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN * 2 / 3;
        //testDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN - trainDataSize_MOP_Predict_DFRNN;
        valDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN * 0.2;
        testDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN * 0.2;
        trainDataSize_MOP_Predict_DFRNN = allDataSize_MOP_Predict_DFRNN -
                                          valDataSize_MOP_Predict_DFRNN -
                                          testDataSize_MOP_Predict_DFRNN;
    } else {
        int tmp_stratified_ind[MAX_DATA_LEN_MOP_PREDICT_DFRNN];
        int tmp_ind1[MAX_DATA_LEN_MOP_PREDICT_DFRNN];
        int tmp_ind2[MAX_DATA_LEN_MOP_PREDICT_DFRNN];
        int tmp_ind3[MAX_DATA_LEN_MOP_PREDICT_DFRNN];
        int tmp_cnt = 0;
        for(int n = 0; n < num_class_MOP_Predict_DFRNN; n++) {
            for(int i = 0; i < allDataSize_MOP_Predict_DFRNN; i++)
                if(allData_MOP_Predict_DFRNN[numAttr_MOP_Predict_DFRNN][i] == n)
                    tmp_stratified_ind[tmp_cnt++] = i;
        }
        int tmp_cnt1 = 0;
        int tmp_cnt2 = 0;
        int tmp_cnt3 = 0;
        for(int i = 0; i < allDataSize_MOP_Predict_DFRNN; i++) {
            if(i % repNum_MOP_Predict_DFRNN == repNo_MOP_Predict_DFRNN) {
                tmp_ind2[tmp_cnt2] = tmp_stratified_ind[i];
                tmp_cnt2++;
            } else if((i + 1) % repNum_MOP_Predict_DFRNN == repNo_MOP_Predict_DFRNN) {
                tmp_ind3[tmp_cnt3] = tmp_stratified_ind[i];
                tmp_cnt3++;
            } else {
                tmp_ind1[tmp_cnt1] = tmp_stratified_ind[i];
                tmp_cnt1++;
            }
        }
        trainDataSize_MOP_Predict_DFRNN = tmp_cnt1;
        testDataSize_MOP_Predict_DFRNN = tmp_cnt2;
        valDataSize_MOP_Predict_DFRNN = tmp_cnt3;
        for(int i = 0; i < tmp_cnt3; i++) tmp_ind1[tmp_cnt1 + i] = tmp_ind3[i];
        for(int i = 0; i < tmp_cnt2; i++) tmp_ind1[tmp_cnt1 + tmp_cnt3 + i] = tmp_ind2[i];
        for(int i = 0; i < allDataSize_MOP_Predict_DFRNN; i++) tmp_ind2[i] = i;
        for(int i = 0; i < allDataSize_MOP_Predict_DFRNN; i++) {
            if(tmp_ind2[i] != tmp_ind1[i]) {
                int the_ind = -1;
                for(int j = i + 1; j < allDataSize_MOP_Predict_DFRNN; j++) {
                    if(tmp_ind2[j] == tmp_ind1[i]) {
                        the_ind = j;
                        break;
                    }
                }
                if(the_ind == -1) {
                    printf("\n%s(%d): something is wrong -- the index (%d) has not found, exiting ...\n",
                           __FILE__, __LINE__, tmp_ind1[i]);
                    exit(2005671);
                }
                int ti1 = tmp_ind2[i];
                int ti2 = tmp_ind2[the_ind];
                // swap
                tmp_ind2[i] = ti2;
                tmp_ind2[the_ind] = ti1;
                for(int n = 0; n <= numAttr_MOP_Predict_DFRNN; n++) {
                    double tmp_v = allData_MOP_Predict_DFRNN[n][i];
                    allData_MOP_Predict_DFRNN[n][i] = allData_MOP_Predict_DFRNN[n][the_ind];
                    allData_MOP_Predict_DFRNN[n][the_ind] = tmp_v;
                }
            }
        }
    }
    //
    free(buf);
    fclose(fpt);
}

static void normalizeData_MOP_Predict_DFRNN()
{
    for(int i = 0; i < numAttr_MOP_Predict_DFRNN; i++) {
        trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][0];
        trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][0];
        trainStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] = 0;
        trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = 0;
        for(int j = 0; j < trainDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            if(trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] > tmp_dt)
                trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = tmp_dt;
            if(trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] < tmp_dt)
                trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = tmp_dt;
            trainStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] += tmp_dt;
        }
        trainStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] /= trainDataSize_MOP_Predict_DFRNN;
        for(int j = 0; j < trainDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            double tmp_mn = trainStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN];
            trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] += (tmp_dt - tmp_mn) * (tmp_dt - tmp_mn);
        }
        trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] /= trainDataSize_MOP_Predict_DFRNN;
        trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = sqrt(trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN]);
        //
        valStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][trainDataSize_MOP_Predict_DFRNN];
        valStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][trainDataSize_MOP_Predict_DFRNN];
        valStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] = 0;
        valStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = 0;
        for(int j = trainDataSize_MOP_Predict_DFRNN;
            j < trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            if(valStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] > tmp_dt)
                valStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = tmp_dt;
            if(valStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] < tmp_dt)
                valStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = tmp_dt;
            valStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] += tmp_dt;
        }
        valStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] /= valDataSize_MOP_Predict_DFRNN;
        for(int j = trainDataSize_MOP_Predict_DFRNN;
            j < trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            double tmp_mn = valStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN];
            valStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] += (tmp_dt - tmp_mn) * (tmp_dt - tmp_mn);
        }
        valStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] /= valDataSize_MOP_Predict_DFRNN;
        valStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = sqrt(valStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN]);
        //
        testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][allDataSize_MOP_Predict_DFRNN - 1];
        testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = allData_MOP_Predict_DFRNN[i][allDataSize_MOP_Predict_DFRNN - 1];
        testStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] = 0;
        testStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = 0;
        for(int j = trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN;
            j < allDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            if(testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] > tmp_dt)
                testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = tmp_dt;
            if(testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] < tmp_dt)
                testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = tmp_dt;
            testStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] += tmp_dt;
        }
        testStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN] /= testDataSize_MOP_Predict_DFRNN;
        for(int j = trainDataSize_MOP_Predict_DFRNN + valDataSize_MOP_Predict_DFRNN;
            j < allDataSize_MOP_Predict_DFRNN; j++) {
            double tmp_dt = allData_MOP_Predict_DFRNN[i][j];
            double tmp_mn = testStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN];
            testStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] += (tmp_dt - tmp_mn) * (tmp_dt - tmp_mn);
        }
        testStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] /= testDataSize_MOP_Predict_DFRNN;
        testStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] = sqrt(testStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN]);
        //////////////////////////////////////////////////////////////////////////
        if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_Z_SCORE_MOP_Predict_DFRNN) {
            if(trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN]) {
                for(int j = 0; j < allDataSize_MOP_Predict_DFRNN; j++) {
                    allData_MOP_Predict_DFRNN[i][j] -= trainStat_MOP_Predict_DFRNN[i][DATA_MEAN_MOP_Predict_DFRNN];
                    allData_MOP_Predict_DFRNN[i][j] /= trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN];
                }
            } else {
                for(int j = 0; j < allDataSize_MOP_Predict_DFRNN; j++) {
                    allData_MOP_Predict_DFRNN[i][j] = 0;
                }
            }
            trainFact_MOP_Predict_DFRNN[i] = trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN] *
                                             trainStat_MOP_Predict_DFRNN[i][DATA_STD_MOP_Predict_DFRNN];
        } else if(NORMALIZE_MOP_Predict_DFRNN == NORMALIZE_MIN_MAX_MOP_Predict_DFRNN) {
            if(trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] >
               trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN]) {
                for(int j = 0; j < allDataSize_MOP_Predict_DFRNN; j++) {
                    allData_MOP_Predict_DFRNN[i][j] -= trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN];
                    allData_MOP_Predict_DFRNN[i][j] /= (trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] -
                                                        trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN]);
                }
            } else {
                for(int j = 0; j < allDataSize_MOP_Predict_DFRNN; j++) {
                    allData_MOP_Predict_DFRNN[i][j] = 0;
                }
            }
            trainFact_MOP_Predict_DFRNN[i] = (trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] -
                                              trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN]) *
                                             (trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] -
                                              trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN]);
        }
        //////////////////////////////////////////////////////////////////////////
        //trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = trainData_MOP_Predict_DFRNN[i][0];
        //trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = trainData_MOP_Predict_DFRNN[i][0];
        //for(int j = 0; j < trainDataSize_MOP_Predict_DFRNN; j++) {
        //    double tmp_dt = trainData_MOP_Predict_DFRNN[i][j];
        //    if(trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] > tmp_dt)
        //        trainStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = tmp_dt;
        //    if(trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] < tmp_dt)
        //        trainStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = tmp_dt;
        //}
        ////
        //testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = testData_MOP_Predict_DFRNN[i][0];
        //testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = testData_MOP_Predict_DFRNN[i][0];
        //for(int j = 0; j < testDataSize_MOP_Predict_DFRNN; j++) {
        //    double tmp_dt = testData_MOP_Predict_DFRNN[i][j];
        //    if(testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] > tmp_dt)
        //        testStat_MOP_Predict_DFRNN[i][DATA_MIN_MOP_Predict_DFRNN] = tmp_dt;
        //    if(testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] < tmp_dt)
        //        testStat_MOP_Predict_DFRNN[i][DATA_MAX_MOP_Predict_DFRNN] = tmp_dt;
        //}
    }
    //
    return;
}

static void get_Evaluation_Indicators_MOP_Predict_DFRNN(int num_class, MY_FLT_TYPE* N_TP, MY_FLT_TYPE* N_FP, MY_FLT_TYPE* N_TN,
        MY_FLT_TYPE* N_FN, MY_FLT_TYPE* N_wrong, MY_FLT_TYPE* N_sum,
        MY_FLT_TYPE* mean_prc, MY_FLT_TYPE* std_prc, MY_FLT_TYPE* mean_rec, MY_FLT_TYPE* std_rec, MY_FLT_TYPE* mean_ber,
        MY_FLT_TYPE* std_ber)
{
    int outSize = num_class;
    //
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE mean_errorRate = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    MY_FLT_TYPE std_errorRate = 0;
    MY_FLT_TYPE min_precision = 1;
    MY_FLT_TYPE min_recall = 1;
    MY_FLT_TYPE min_Fvalue = 1;
    MY_FLT_TYPE max_errorRate = 0;
    MY_FLT_TYPE* tmp_precision = (MY_FLT_TYPE*)malloc(outSize * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_recall = (MY_FLT_TYPE*)malloc(outSize * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_Fvalue = (MY_FLT_TYPE*)malloc(outSize * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_errorRate = (MY_FLT_TYPE*)malloc(outSize * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < outSize; i++) {
        if(N_TP[i] > 0) {
            tmp_precision[i] = N_TP[i] / (N_TP[i] + N_FP[i]);
        } else {
            tmp_precision[i] = 0;
        }
        if(N_TP[i] + N_FN[i] > 0) {
            tmp_recall[i] = N_TP[i] / (N_TP[i] + N_FN[i]);
            tmp_errorRate[i] = N_FN[i] / (N_TP[i] + N_FN[i]);
        } else {
            tmp_recall[i] = 0;
            tmp_errorRate[i] = 1;
        }
        if(tmp_recall[i] + tmp_precision[i] > 0)
            tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
                            (tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
        else
            tmp_Fvalue[i] = 0;
        mean_precision += tmp_precision[i];
        mean_recall += tmp_recall[i];
        mean_Fvalue += tmp_Fvalue[i];
        mean_errorRate += tmp_errorRate[i];
        if(min_precision > tmp_precision[i]) min_precision = tmp_precision[i];
        if(min_recall > tmp_recall[i]) min_recall = tmp_recall[i];
        if(min_Fvalue > tmp_Fvalue[i]) min_Fvalue = tmp_Fvalue[i];
        if(max_errorRate < tmp_errorRate[i]) max_errorRate = tmp_errorRate[i];
#if STATUS_OUT_INDEICES_MOP_PREDICT_DFRNN == FLAG_ON_MOP_Predict_DFRNN
        printf("%f %f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i], tmp_errorRate[i]);
#endif
    }
    mean_precision /= outSize;
    mean_recall /= outSize;
    mean_Fvalue /= outSize;
    mean_errorRate /= outSize;
    for(int i = 0; i < outSize; i++) {
        std_precision += (tmp_precision[i] - mean_precision) * (tmp_precision[i] - mean_precision);
        std_recall += (tmp_recall[i] - mean_recall) * (tmp_recall[i] - mean_recall);
        std_Fvalue += (tmp_Fvalue[i] - mean_Fvalue) * (tmp_Fvalue[i] - mean_Fvalue);
        std_errorRate += (tmp_errorRate[i] - mean_errorRate) * (tmp_errorRate[i] - mean_errorRate);
    }
    std_precision /= outSize;
    std_recall /= outSize;
    std_Fvalue /= outSize;
    std_errorRate /= outSize;
    std_precision = sqrt(std_precision);
    std_recall = sqrt(std_recall);
    std_Fvalue = sqrt(std_Fvalue);
    std_errorRate = sqrt(std_errorRate);
    //
    double mean_err_rt = 0.0;
    double max_err_rt = 0.0;
    for(int i = 0; i < outSize; i++) {
        double tmp_rt = N_wrong[i] / N_sum[i];
        mean_err_rt += tmp_rt;
        if(max_err_rt < tmp_rt)
            max_err_rt = tmp_rt;
    }
    mean_err_rt /= outSize;
    //
    if(mean_prc) mean_prc[0] = mean_precision;
    if(std_prc) std_prc[0] = std_precision;
    if(mean_rec) mean_rec[0] = mean_recall;
    if(std_rec) std_rec[0] = std_recall;
    if(mean_ber) mean_ber[0] = mean_errorRate;
    if(std_ber) std_ber[0] = std_errorRate;
    //
    free(tmp_precision);
    free(tmp_recall);
    free(tmp_Fvalue);
    free(tmp_errorRate);
    //
    return;
}

#if CURRENT_PROB_MOP_PREDICT_DFRNN == STOCK_TRADING_MOP_PREDICT_DFRNN
static void genTradingLabel_MOP_Predict_DFRNN()
{
    for(int i = 0; i < win_size_cases_MOP_Predict_DFRNN; i++) {
        for(int j = 0; j < MAX_DATA_LEN_MOP_PREDICT_DFRNN; j++) {
            train_trading_label_MOP_Predict_DFRNN[i][j] = CLASS_IND_HOLD_MOP_PREDICT_DFRNN;
            test_trading_label_MOP_Predict_DFRNN[i][j] = CLASS_IND_HOLD_MOP_PREDICT_DFRNN;
        }
    }
    for(int i = 0; i < win_size_cases_MOP_Predict_DFRNN; i++) {
        int win_size = i * 2 + win_size_min_MOP_Predict_DFRNN;
        // train data
        for(int j = 0; j < trainDataSize_MOP_Predict_DFRNN - win_size + 1; j++) {
            int ind_start = j;
            int ind_final = j + win_size - 1;
            int ind_middl = j + win_size / 2;
            double val_mid = allData_MOP_Predict_DFRNN[0][ind_middl];
            int ind_min = j;
            int ind_max = j;
            double val_min = allData_MOP_Predict_DFRNN[0][j];
            double val_max = allData_MOP_Predict_DFRNN[0][j];
            for(int k = ind_start + 1; k <= ind_final; k++) {
                if(allData_MOP_Predict_DFRNN[0][k] < val_min) {
                    val_min = allData_MOP_Predict_DFRNN[0][k];
                    ind_min = k;
                }
                if(allData_MOP_Predict_DFRNN[0][k] > val_max) {
                    val_max = allData_MOP_Predict_DFRNN[0][k];
                    ind_max = k;
                }
            }
            if(ind_middl == ind_min || val_min == val_mid)
                train_trading_label_MOP_Predict_DFRNN[i][ind_middl] = CLASS_IND_BUY_MOP_PREDICT_DFRNN;
            if(ind_middl == ind_max || val_max == val_mid)
                train_trading_label_MOP_Predict_DFRNN[i][ind_middl] = CLASS_IND_SELL_MOP_PREDICT_DFRNN;
        }
        // test data
        for(int j = trainDataSize_MOP_Predict_DFRNN - win_size; j < allDataSize_MOP_Predict_DFRNN - win_size + 1; j++) {
            int ind_start = j;
            int ind_final = j + win_size - 1;
            int ind_middl = j + win_size / 2;
            double val_mid = allData_MOP_Predict_DFRNN[0][ind_middl];
            int ind_min = j;
            int ind_max = j;
            double val_min = allData_MOP_Predict_DFRNN[0][j];
            double val_max = allData_MOP_Predict_DFRNN[0][j];
            for(int k = ind_start + 1; k <= ind_final; k++) {
                if(allData_MOP_Predict_DFRNN[0][k] < val_min) {
                    val_min = allData_MOP_Predict_DFRNN[0][k];
                    ind_min = k;
                }
                if(allData_MOP_Predict_DFRNN[0][k] > val_max) {
                    val_max = allData_MOP_Predict_DFRNN[0][k];
                    ind_max = k;
                }
            }
            if(ind_middl == ind_min || val_min == val_mid)
                test_trading_label_MOP_Predict_DFRNN[i][ind_middl] = CLASS_IND_BUY_MOP_PREDICT_DFRNN;
            if(ind_middl == ind_max || val_max == val_mid)
                test_trading_label_MOP_Predict_DFRNN[i][ind_middl] = CLASS_IND_SELL_MOP_PREDICT_DFRNN;
        }
    }
}
#endif

//////////////////////////////////////////////////////////////////////////
#define IM1_Predict_DFRNN 2147483563
#define IM2_Predict_DFRNN 2147483399
#define AM_Predict_DFRNN (1.0/IM1_Predict_DFRNN)
#define IMM1_Predict_DFRNN (IM1_Predict_DFRNN-1)
#define IA1_Predict_DFRNN 40014
#define IA2_Predict_DFRNN 40692
#define IQ1_Predict_DFRNN 53668
#define IQ2_Predict_DFRNN 52774
#define IR1_Predict_DFRNN 12211
#define IR2_Predict_DFRNN 3791
#define NTAB_Predict_DFRNN 32
#define NDIV_Predict_DFRNN (1+IMM1_Predict_DFRNN/NTAB_Predict_DFRNN)
#define EPS_Predict_DFRNN 1.2e-7
#define RNMX_Predict_DFRNN (1.0-EPS_Predict_DFRNN)

//the random generator in [0,1)
static double rnd_uni_Predict_DFRNN(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_Predict_DFRNN];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_Predict_DFRNN + 7; j >= 0; j--) {
            k = (*idum) / IQ1_Predict_DFRNN;
            *idum = IA1_Predict_DFRNN * (*idum - k * IQ1_Predict_DFRNN) - k * IR1_Predict_DFRNN;
            if(*idum < 0) *idum += IM1_Predict_DFRNN;
            if(j < NTAB_Predict_DFRNN) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_Predict_DFRNN;
    *idum = IA1_Predict_DFRNN * (*idum - k * IQ1_Predict_DFRNN) - k * IR1_Predict_DFRNN;
    if(*idum < 0) *idum += IM1_Predict_DFRNN;
    k = idum2 / IQ2_Predict_DFRNN;
    idum2 = IA2_Predict_DFRNN * (idum2 - k * IQ2_Predict_DFRNN) - k * IR2_Predict_DFRNN;
    if(idum2 < 0) idum2 += IM2_Predict_DFRNN;
    j = iy / NDIV_Predict_DFRNN;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_Predict_DFRNN;       //printf("%lf\n", AM_Predict_DFRNN*iy);
    if((temp = AM_Predict_DFRNN * iy) > RNMX_Predict_DFRNN) return RNMX_Predict_DFRNN;
    else return temp;
}/*------End of rnd_uni_Classify_CNN()--------------------------*/

static int rnd_Predict_DFRNN(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_Predict_DFRNN(&rnd_uni_init_Predict_DFRNN) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}

/* Fisher¨CYates shuffle algorithm */
static void shuffle_Predict_DFRNN(int* x, int size)
{
    int i, aux, k = 0;
    for(i = size - 1; i > 0; i--) {
        /* get a value between cero and i  */
        k = rnd_Predict_DFRNN(0, i);
        /* exchange of values */
        aux = x[i];
        x[i] = x[k];
        x[k] = aux;
    }
    //
    return;
}

static void trimLine_MOP_Predict_DFRNN(char line[])
{
    int i = 0;

    while(line[i] != '\0') {
        if(line[i] == '\r' || line[i] == '\n') {
            line[i] = '\0';
            break;
        }
        i++;
    }
}

int get_setting_MOP_Predict_DFRNN(char* wholestr, const char* candidstr, int& val, int* vec)
{
    //
    char tmp_str[MAX_STR_LEN_MOP_Predict_DFRNN];
    sprintf(tmp_str, "%s", wholestr);
    char tmp_delim[] = "_";
    char* p;
    int elem_int;
    int flag_found = 0;
    for(p = strtok(tmp_str, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
        if(flag_found == 1) {
            if(!strcmp(candidstr, "Deep")) {
                char tmp_str1[MAX_STR_LEN_MOP_Predict_DFRNN];
                sprintf(tmp_str1, "%s", p);
                char* p1;
                val = 0;
                for(p1 = strtok(tmp_str1, "+"); p1; p1 = strtok(NULL, "+")) {
                    if(sscanf(p, "%d", &elem_int) != 1) {
                        printf("\n%s(%d): setting value not found error...\n", __FILE__, __LINE__);
                        exit(65871001);
                    }
                    vec[val] = elem_int;
                    val++;
                }
                flag_found = 2;
                break;
            } else {
                if(sscanf(p, "%d", &elem_int) != 1) {
                    printf("\n%s(%d): setting value not found error...\n", __FILE__, __LINE__);
                    exit(65871001);
                }
                val = elem_int;
                flag_found = 2;
                break;
            }
        }
        if(!strcmp(p, candidstr)) flag_found = 1;
    }
    //
    return (flag_found == 2);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
