#include "MOP_Predict_FRNN.h"
#include <float.h>
#include <math.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
#endif

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_Predict_FRNN 0
#define FLAG_ON_MOP_Predict_FRNN 1
#define STATUS_OUT_INDEICES_MOP_Predict_FRNN FLAG_OFF_MOP_Predict_FRNN
#define STATUS_OUT_INDEICES_MOP_PREDICT_FRNN FLAG_OFF_MOP_Predict_FRNN

//#define OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_FRNN
#define PRINT_ERROR_PARA_MOP_Predict_FRNN

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN_MOP_Predict_FRNN 1024
#define MAX_OUT_NUM_MOP_Predict_FRNN 1024
#define VIOLATION_PENALTY_Predict_FRNN 1e6

//////////////////////////////////////////////////////////////////////////
#define TAG_VALI_MOP_PREDICT_FRNN -2
#define TAG_NULL_MOP_PREDICT_FRNN 0
#define TAG_INVA_MOP_PREDICT_FRNN -1

//////////////////////////////////////////////////////////////////////////
#define MAX_DATA_LEN_MOP_PREDICT_FRNN 10000
#define MAX_ATTR_NUM 64

//////////////////////////////////////////////////////////////////////////
#define THRESHOLD_NUM_ROUGH_NODES 1

//////////////////////////////////////////////////////////////////////////
enum ENUM_DATA_INTERVAL_TYPE {
    D_INTV_T_MONTHLY,
    D_INTV_T_QUARTERLY,
    D_INTV_T_HOURLY,
    D_INTV_T_YEARLY,
    D_INTV_T_WEEKLY,
    D_INTV_T_DAILY
};
int d_intv_t_MOP_PREDICT_FRNN;

#define PARA_M_MONTHLY   12
#define PARA_M_QUARTERLY 4
#define PARA_M_HOURLY    24
#define PARA_M_YEARLY    1
#define PARA_M_WEEKLY    1
#define PARA_M_DAILY     1
int p_m_MOP_PREDICT_FRNN;

//////////////////////////////////////////////////////////////////////////
#define NUM_LABEL_TRADING_MOP_PREDICT_FRNN 3
#define CLASS_IND_BUY_MOP_PREDICT_FRNN  0
#define CLASS_IND_HOLD_MOP_PREDICT_FRNN 1
#define CLASS_IND_SELL_MOP_PREDICT_FRNN 2
#if CURRENT_PROB_MOP_PREDICT_FRNN == STOCK_TRADING_MOP_PREDICT_FRNN
#define win_size_max_MOP_Predict_FRNN 35
#define win_size_min_MOP_Predict_FRNN 7
#define win_size_cases_MOP_Predict_FRNN 15 // 7 9 11 13 15 17 ... 35
int train_trading_label_MOP_Predict_FRNN[win_size_cases_MOP_Predict_FRNN][MAX_DATA_LEN_MOP_PREDICT_FRNN];
int test_trading_label_MOP_Predict_FRNN[win_size_cases_MOP_Predict_FRNN][MAX_DATA_LEN_MOP_PREDICT_FRNN];
#endif

//////////////////////////////////////////////////////////////////////////
int NDIM_MOP_Predict_FRNN = 0;
int NOBJ_MOP_Predict_FRNN = 0;
char prob_name_MOP_Predict_FRNN[128];
int tag_classification_MOP_Predict_FRNN;
int num_class_MOP_Predict_FRNN;

int num_in__predict_MOP_Predict_FRNN;
int ind_in__predict_MOP_Predict_FRNN[MAX_ATTR_NUM];
int num_out_predict_MOP_Predict_FRNN;
int ind_out_predict_MOP_Predict_FRNN[MAX_ATTR_NUM];

int flag_multi_in_cnsq_MOP_Predict_FRNN;
int flag_fuse_in_cnsq_MOP_Predict_FRNN;
int type_frnn_cnsq_MOP_Predict_FRNN;

//////////////////////////////////////////////////////////////////////////
#define DATA_MIN_MOP_Predict_FRNN 0
#define DATA_MAX_MOP_Predict_FRNN 1
#define DATA_MEAN_MOP_Predict_FRNN 2
#define DATA_STD_MOP_Predict_FRNN 3
int numAttr; // not including the label for classification
double allData_MOP_Predict_FRNN[MAX_ATTR_NUM][MAX_DATA_LEN_MOP_PREDICT_FRNN];
double trainStat_MOP_Predict_FRNN[MAX_ATTR_NUM][4];
double trainFact_MOP_Predict_FRNN[MAX_ATTR_NUM];
double testStat_MOP_Predict_FRNN[MAX_ATTR_NUM][4];
int allDataSize_MOP_Predict_FRNN = 0;
int trainDataSize_MOP_Predict_FRNN = 0;
int testDataSize_MOP_Predict_FRNN = 0;
int NORMALIZE_Z_SCORE_MOP_Predict_FRNN = 0;
int NORMALIZE_MIN_MAX_MOP_Predict_FRNN = 1;
int NORMALIZE_MOP_Predict_FRNN = -1;
int flag_T2_MOP_Predict_FRNN = -1;

int  repNum_MOP_Predict_FRNN;
int  repNo_MOP_Predict_FRNN;

MY_FLT_TYPE total_penalty_MOP_Predict_FRNN = 0.0;
MY_FLT_TYPE penaltyVal_MOP_Predict_FRNN = 1e6;

frnn_MOP_Predict_FRNN* frnn_MOP_Predict = NULL;

//////////////////////////////////////////////////////////////////////////
static void ff_Predict_FRNN_c(double* individual, int tag_train_test);
static double simplicity_MOP_Predict_FRNN();
static double generality_MOP_Predict_FRNN();
static double get_profit_MOP_Predict_FRNN(int tag_train_test);
static void readData_stock_MOP_Predict_FRNN(char* fname, int trainNo, int testNo, int endNo);
static void readData_general_MOP_Predict_FRNN(char* fname, int tag_classification);
static void normalizeData_MOP_Predict_FRNN();
static void get_Evaluation_Indicators_MOP_Predict_FRNN(int num_class, MY_FLT_TYPE* N_TP, MY_FLT_TYPE* N_FP, MY_FLT_TYPE* N_TN,
        MY_FLT_TYPE* N_FN, MY_FLT_TYPE* N_wrong, MY_FLT_TYPE* N_sum,
        MY_FLT_TYPE* mean_prc, MY_FLT_TYPE* std_prc, MY_FLT_TYPE* mean_rec, MY_FLT_TYPE* std_rec, MY_FLT_TYPE* mean_ber,
        MY_FLT_TYPE* std_ber);
#if CURRENT_PROB_MOP_PREDICT_FRNN == STOCK_TRADING_MOP_PREDICT_FRNN
static void genTradingLabel_MOP_Predict_FRNN();
#endif
//
int     seed_Predict_FRNN = 237;
long    rnd_uni_init_Predict_FRNN = -(long)seed_Predict_FRNN;
static double rnd_uni_Predict_FRNN(long* idum);
static int rnd_Predict_FRNN(int low, int high);
static void shuffle_Predict_FRNN(int* x, int size);
static void trimLine_MOP_Predict_FRNN(char line[]);
static int  get_setting_MOP_Predict_FRNN(char* wholestr, const char* candidstr, int& val);

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Predict_FRNN(char* pro, int curN, int numN, int trainNo, int testNo, int endNo, int my_rank)
{
    //
    sprintf(prob_name_MOP_Predict_FRNN, "%s", pro);
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    seed_Predict_FRNN = 237 + my_rank;
    seed_Predict_FRNN = seed_Predict_FRNN % 1235;
    rnd_uni_init_Predict_FRNN = -(long)seed_Predict_FRNN;
    for(int i = 0; i < curN; i++) {
        seed_Predict_FRNN = (seed_Predict_FRNN + 111) % 1235;
        rnd_uni_init_Predict_FRNN = -(long)seed_Predict_FRNN;
    }
    //
    repNo_MOP_Predict_FRNN = curN;
    repNum_MOP_Predict_FRNN = numN;
    //
    flag_multi_in_cnsq_MOP_Predict_FRNN = FLAG_STATUS_OFF;
    flag_fuse_in_cnsq_MOP_Predict_FRNN = FLAG_STATUS_OFF;
    get_setting_MOP_Predict_FRNN(pro, "FuseF", flag_fuse_in_cnsq_MOP_Predict_FRNN);
    NORMALIZE_MOP_Predict_FRNN = NORMALIZE_MIN_MAX_MOP_Predict_FRNN;
    get_setting_MOP_Predict_FRNN(pro, "Tnorm", NORMALIZE_MOP_Predict_FRNN);
    flag_T2_MOP_Predict_FRNN = 1;
    get_setting_MOP_Predict_FRNN(pro, "flagT2", flag_T2_MOP_Predict_FRNN);
    type_frnn_cnsq_MOP_Predict_FRNN = FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_ADAPT;
    get_setting_MOP_Predict_FRNN(pro, "Tcnsq", type_frnn_cnsq_MOP_Predict_FRNN);
    num_class_MOP_Predict_FRNN = 0;
    //
    char filename[MAX_STR_LEN_MOP_Predict_FRNN];
    if(strstr(pro, "Stock_")) {
        tag_classification_MOP_Predict_FRNN = 0;
        numAttr = 6;
        sprintf(filename, "../Data_all/AllFileNames_FRNN");
        readData_stock_MOP_Predict_FRNN(filename, trainNo, testNo, endNo);
    } else if(strstr(pro, "Classify_")) {
        tag_classification_MOP_Predict_FRNN = 1;
        char* ret = strstr(pro, "Classify_");
        ret += strlen("Classify_");
        sprintf(filename, "../Data_all/UCI_Data/%s", ret);
        readData_general_MOP_Predict_FRNN(filename, tag_classification_MOP_Predict_FRNN);
    } else if(strstr(pro, "TimeSeries_")) {
        tag_classification_MOP_Predict_FRNN = 0;
        char* ret = strstr(pro, "TimeSeries_");
        ret += strlen("TimeSeries_");
        char tmp_str[MAX_STR_LEN_MOP_Predict_FRNN];
        sprintf(tmp_str, "%s", ret);
        for(int i = 0; i < MAX_STR_LEN_MOP_Predict_FRNN; i++) {
            if(tmp_str[i] == '_') {
                tmp_str[i] = '\0';
                break;
            }
        }
        sprintf(filename, "../Data_all/UCI_Data/%s", tmp_str);
        readData_general_MOP_Predict_FRNN(filename, tag_classification_MOP_Predict_FRNN);
    } else {
        printf("\n%s(%d): Unknown problem name ~ %s, the dataset cannot be found, exiting...\n",
               __FILE__, __LINE__, pro);
        exit(-9124);
    }
    if(tag_classification_MOP_Predict_FRNN && num_class_MOP_Predict_FRNN == 0) {
        printf("\n%s(%d): The number of classes is 0 for classification, the data file should include the number of classes, exiting...\n",
               __FILE__, __LINE__);
        exit(-91224);
    }
    num_in__predict_MOP_Predict_FRNN = numAttr;
    for(int i = 0; i < numAttr; i++) ind_in__predict_MOP_Predict_FRNN[i] = i;
    if(tag_classification_MOP_Predict_FRNN) {
        num_out_predict_MOP_Predict_FRNN = 1;
        flag_multi_in_cnsq_MOP_Predict_FRNN = FLAG_STATUS_ON;
    } else {
        if(strstr(pro, "Stock_")) {
            num_out_predict_MOP_Predict_FRNN = 1;
            ind_out_predict_MOP_Predict_FRNN[0] = 0;
        } else if(strstr(pro, "TimeSeries_")) {
            if(strstr(pro, "gnfuv")) {
                num_out_predict_MOP_Predict_FRNN = 1;
                ind_out_predict_MOP_Predict_FRNN[0] = 0;
                ind_out_predict_MOP_Predict_FRNN[1] = 1;
            } else if(strstr(pro, "hungaryChickenpox")) {
                num_out_predict_MOP_Predict_FRNN = 1;
                ind_out_predict_MOP_Predict_FRNN[0] = 0;
            } else if(strstr(pro, "SML2010-DATA")) {
                num_out_predict_MOP_Predict_FRNN = 1;
                ind_out_predict_MOP_Predict_FRNN[0] = 0;
                ind_out_predict_MOP_Predict_FRNN[1] = 1;
            } else if(strstr(pro, "traffic")) {
                num_out_predict_MOP_Predict_FRNN = 1;
                ind_out_predict_MOP_Predict_FRNN[0] = 0;
            } else if(strstr(pro, "Daily_Demand_Forecasting_Orders")) {
                num_out_predict_MOP_Predict_FRNN = 1;
                ind_out_predict_MOP_Predict_FRNN[0] = 0;
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
    d_intv_t_MOP_PREDICT_FRNN = D_INTV_T_DAILY;
    p_m_MOP_PREDICT_FRNN = 1;
    if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_MONTHLY)
        p_m_MOP_PREDICT_FRNN = PARA_M_MONTHLY;
    else if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_QUARTERLY)
        p_m_MOP_PREDICT_FRNN = PARA_M_QUARTERLY;
    else if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_HOURLY)
        p_m_MOP_PREDICT_FRNN = PARA_M_HOURLY;
    else if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_YEARLY)
        p_m_MOP_PREDICT_FRNN = PARA_M_YEARLY;
    else if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_WEEKLY)
        p_m_MOP_PREDICT_FRNN = PARA_M_WEEKLY;
    else if(d_intv_t_MOP_PREDICT_FRNN == D_INTV_T_DAILY)
        p_m_MOP_PREDICT_FRNN = PARA_M_DAILY;
    // Normalization
    for(int i = 0; i < MAX_ATTR_NUM; i++) trainFact_MOP_Predict_FRNN[i] = 1;
    normalizeData_MOP_Predict_FRNN();
    //////////////////////////////////////////////////////////////////////////
#if CURRENT_PROB_MOP_PREDICT_FRNN == STOCK_TRADING_MOP_PREDICT_FRNN
    genTradingLabel_MOP_Predict_FRNN();
#endif
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    frnn_MOP_Predict = (frnn_MOP_Predict_FRNN*)calloc(1, sizeof(frnn_MOP_Predict_FRNN));
    if(strstr(pro, "evoFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoGFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoDFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoGFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_OFF;
    } else if(strstr(pro, "evoBFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoBGFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoBDFRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoBFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoBDFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else if(strstr(pro, "evoBGFGRNN")) {
        frnn_MOP_Predict->tag_GEP = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_DIF = FLAG_STATUS_OFF;
        frnn_MOP_Predict->tag_GEPr = FLAG_STATUS_ON;
        frnn_MOP_Predict->tag_multiKindInput = FLAG_STATUS_ON;
    } else {
        printf("\n%s(%d): Unknown problem name ~ %s, exiting...\n",
               __FILE__, __LINE__, pro);
        exit(-91284);
    }
    if(frnn_MOP_Predict->tag_multiKindInput == FLAG_STATUS_OFF) {
        num_in__predict_MOP_Predict_FRNN = 1;
        ind_in__predict_MOP_Predict_FRNN[0] = 0;
    }
    if(flag_fuse_in_cnsq_MOP_Predict_FRNN == FLAG_STATUS_ON)
        flag_multi_in_cnsq_MOP_Predict_FRNN = FLAG_STATUS_ON;
    if(flag_fuse_in_cnsq_MOP_Predict_FRNN == FLAG_STATUS_ON &&
       tag_classification_MOP_Predict_FRNN) {
        printf("%s(%d): For the classification problem, there is no need to set feature fusing, exiting...\n",
               __FILE__, __LINE__);
        exit(-972849);
    }
    if(flag_fuse_in_cnsq_MOP_Predict_FRNN == FLAG_STATUS_ON) {
        // remove the target ind in the input features
        int i = 0;
        while(i < num_in__predict_MOP_Predict_FRNN) {
            for(int o = 0; o < num_out_predict_MOP_Predict_FRNN; o++) {
                if(ind_in__predict_MOP_Predict_FRNN[i] == ind_out_predict_MOP_Predict_FRNN[o]) {
                    num_in__predict_MOP_Predict_FRNN--;
                    if(num_in__predict_MOP_Predict_FRNN <= 0) {
                        printf("%s(%d): There are no remaining input features, exiting...\n", __FILE__, __LINE__);
                        exit(-97281);
                    }
                    for(int ii = i; ii < num_in__predict_MOP_Predict_FRNN; ii++) {
                        ind_in__predict_MOP_Predict_FRNN[ii] = ind_in__predict_MOP_Predict_FRNN[ii + 1];
                    }
                }
            }
            i++;
        }
    }
    frnn_Predict_FRNN_setup(frnn_MOP_Predict);
    //
    NDIM_MOP_Predict_FRNN = frnn_MOP_Predict->numParaLocal;
    NOBJ_MOP_Predict_FRNN = 3;
    //
    return;
}
void SetLimits_MOP_Predict_FRNN(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    if(frnn_MOP_Predict->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn_MOP_Predict->num_GEP; n++) {
            for(int i = 0; i < frnn_MOP_Predict->GEP0[n]->numParaLocal; i++) {
                minLimit[count] = frnn_MOP_Predict->GEP0[n]->xMin[i];
                maxLimit[count] = frnn_MOP_Predict->GEP0[n]->xMax[i];
                count++;
            }
        }
    }
    for(int i = 0; i < frnn_MOP_Predict->M1->numParaLocal; i++) {
        minLimit[count] = frnn_MOP_Predict->M1->xMin[i];
        maxLimit[count] = frnn_MOP_Predict->M1->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_MOP_Predict->F2->numParaLocal; i++) {
        minLimit[count] = frnn_MOP_Predict->F2->xMin[i];
        maxLimit[count] = frnn_MOP_Predict->F2->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_MOP_Predict->R3->numParaLocal; i++) {
        minLimit[count] = frnn_MOP_Predict->R3->xMin[i];
        maxLimit[count] = frnn_MOP_Predict->R3->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_MOP_Predict->OL->numParaLocal; i++) {
        minLimit[count] = frnn_MOP_Predict->OL->xMin[i];
        maxLimit[count] = frnn_MOP_Predict->OL->xMax[i];
        count++;
    }
    //
    return;
}

int CheckLimits_MOP_Predict_FRNN(double* x, int nx)
{
    int count = 0;
    //
    if(frnn_MOP_Predict->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn_MOP_Predict->num_GEP; n++) {
            for(int i = 0; i < frnn_MOP_Predict->GEP0[n]->numParaLocal; i++) {
                if(x[count] < frnn_MOP_Predict->GEP0[n]->xMin[i] ||
                   x[count] > frnn_MOP_Predict->GEP0[n]->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->GEP0[%d] %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, n, i, x[count], frnn_MOP_Predict->GEP0[n]->xMin[i], frnn_MOP_Predict->GEP0[n]->xMax[i]);
                    return 0;
                }
                count++;
            }
        }
    }
    for(int i = 0; i < frnn_MOP_Predict->M1->numParaLocal; i++) {
        if(x[count] < frnn_MOP_Predict->M1->xMin[i] ||
           x[count] > frnn_MOP_Predict->M1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->M1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_MOP_Predict->M1->xMin[i], frnn_MOP_Predict->M1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_MOP_Predict->F2->numParaLocal; i++) {
        if(x[count] < frnn_MOP_Predict->F2->xMin[i] ||
           x[count] > frnn_MOP_Predict->F2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->F2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_MOP_Predict->F2->xMin[i], frnn_MOP_Predict->F2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_MOP_Predict->R3->numParaLocal; i++) {
        if(x[count] < frnn_MOP_Predict->R3->xMin[i] ||
           x[count] > frnn_MOP_Predict->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->R3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_MOP_Predict->R3->xMin[i], frnn_MOP_Predict->R3->xMax[i]);
            return 0;
        }
        count++;
    }
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    for(int i = 0; i < frnn_MOP_Predict->OL->numParaLocal; i++) {
        if(x[count] < frnn_MOP_Predict->OL->xMin[i] ||
           x[count] > frnn_MOP_Predict->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->OL %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_MOP_Predict->OL->xMin[i], frnn_MOP_Predict->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#else
    if(frnn_MOP_Predict->flagConnectStatus != FLAG_STATUS_OFF ||
       frnn_MOP_Predict->flagConnectWeight != FLAG_STATUS_ON ||
       frnn_MOP_Predict->typeCoding != PARA_CODING_DIRECT) {
        printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
               __FILE__, __LINE__, frnn_MOP_Predict->flagConnectStatus, frnn_MOP_Predict->flagConnectWeight, frnn_MOP_Predict->typeCoding);
        exit(-275082);
    }
    int tmp_offset = frnn_MOP_Predict->OL->numOutput * frnn_MOP_Predict->OL->numInput;
    count += tmp_offset;
    for(int i = tmp_offset; i < frnn_MOP_Predict->OL->numParaLocal; i++) {
        if(x[count] < frnn_MOP_Predict->OL->xMin[i] ||
           x[count] > frnn_MOP_Predict->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - frnn_MOP_Predict: frnn_MOP_Predict->OL %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_MOP_Predict->OL->xMin[i], frnn_MOP_Predict->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
    //
    return 1;
}

void Fitness_MOP_Predict_FRNN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    ff_Predict_FRNN_c(individual, TRAIN_TAG_MOP_PREDICT_FRNN);
    //
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
    if(!tag_classification_MOP_Predict_FRNN) {
        fitness[0] = sqrt(frnn_MOP_Predict->sum_wrong / frnn_MOP_Predict->sum_all);
    } else {
        MY_FLT_TYPE mean_prc = 0;
        get_Evaluation_Indicators_MOP_Predict_FRNN(num_class_MOP_Predict_FRNN,
                frnn_MOP_Predict->N_TP, frnn_MOP_Predict->N_FP, frnn_MOP_Predict->N_TN, frnn_MOP_Predict->N_FN,
                frnn_MOP_Predict->N_wrong, frnn_MOP_Predict->N_sum,
                &mean_prc, NULL, NULL, NULL, NULL, NULL);
        fitness[0] = 1 - mean_prc;
    }
#else
    fitness[0] = get_profit_MOP_Predict_FRNN(TRAIN_TAG_MOP_PREDICT_FRNN);
#endif
    fitness[1] = simplicity_MOP_Predict_FRNN();
    //
    fitness[2] = generality_MOP_Predict_FRNN();
    //
    for(int i = 0; i < NOBJ_MOP_Predict_FRNN; i++)
        fitness[i] += total_penalty_MOP_Predict_FRNN;
    //
    return;
}

void Fitness_MOP_Predict_FRNN_test(double* individual, double* fitness)
{
    ff_Predict_FRNN_c(individual, TEST_TAG_MOP_PREDICT_FRNN);
    //
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
    if(!tag_classification_MOP_Predict_FRNN) {
        fitness[0] = sqrt(frnn_MOP_Predict->sum_wrong / frnn_MOP_Predict->sum_all);
    } else {
        MY_FLT_TYPE mean_prc = 0;
        get_Evaluation_Indicators_MOP_Predict_FRNN(num_class_MOP_Predict_FRNN,
                frnn_MOP_Predict->N_TP, frnn_MOP_Predict->N_FP, frnn_MOP_Predict->N_TN, frnn_MOP_Predict->N_FN,
                frnn_MOP_Predict->N_wrong, frnn_MOP_Predict->N_sum,
                &mean_prc, NULL, NULL, NULL, NULL, NULL);
        fitness[0] = mean_prc;
    }
#else
    fitness[0] = get_profit_MOP_Predict_FRNN(TEST_TAG_MOP_PREDICT_FRNN);
#endif
    fitness[1] = simplicity_MOP_Predict_FRNN();
    //
    fitness[2] = generality_MOP_Predict_FRNN();
    //
    for(int i = 0; i < NOBJ_MOP_Predict_FRNN; i++)
        fitness[i] += total_penalty_MOP_Predict_FRNN;
    //
    return;
}

static void ff_Predict_FRNN_c(double* individual, int tag_train_test)
{
    int num_in = 0;
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
        num_in = trainDataSize_MOP_Predict_FRNN;
    } else {
        num_in = testDataSize_MOP_Predict_FRNN;
    }
    //
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
    int num_sample = num_in - (frnn_MOP_Predict->numInput - 1) * frnn_MOP_Predict->lenGap - 1 -
                     frnn_MOP_Predict->numOutput * frnn_MOP_Predict->lenGap + 1;
    if(tag_train_test == TEST_TAG_MOP_PREDICT_FRNN)
        num_sample = num_in - (frnn_MOP_Predict->numOutput - 1) * frnn_MOP_Predict->lenGap;
    if(tag_classification_MOP_Predict_FRNN)
        num_sample = num_in;
#else
    int num_sample = num_in - frnn_MOP_Predict->numInput + 1;
    if(tag_train_test == TEST_TAG_MOP_PREDICT_FRNN)
        num_sample = num_in;
#endif
    frnn_MOP_Predict->sum_all = 0;
    frnn_MOP_Predict->sum_wrong = 0;
    for(int i = 0; i < frnn_MOP_Predict->numOutput * frnn_MOP_Predict->num_multiKindOutput; i++) {
        frnn_MOP_Predict->N_sum[i] = 0;
        frnn_MOP_Predict->N_wrong[i] = 0;
        frnn_MOP_Predict->e_sum[i] = 0;
        frnn_MOP_Predict->N_TP[i] = 0;
        frnn_MOP_Predict->N_TN[i] = 0;
        frnn_MOP_Predict->N_FP[i] = 0;
        frnn_MOP_Predict->N_FN[i] = 0;
    }
    frnn_Predict_FRNN_init(frnn_MOP_Predict, individual, ASSIGN_MODE_FRNN);
    //
    const int len_valIn = 100 * MAX_ATTR_NUM;
    MY_FLT_TYPE valIn[len_valIn];
    MY_FLT_TYPE valOut[MAX_OUT_NUM_MOP_Predict_FRNN];
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    MY_FLT_TYPE*** matA = NULL;
    MY_FLT_TYPE*** matB = NULL;
    MY_FLT_TYPE*** matLeft = NULL;
    MY_FLT_TYPE*** matRight = NULL;
    int tmp_offset_samp = 0;
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
        matA = (MY_FLT_TYPE***)malloc(frnn_MOP_Predict->num_multiKindOutput * sizeof(MY_FLT_TYPE**));
        matB = (MY_FLT_TYPE***)malloc(frnn_MOP_Predict->num_multiKindOutput * sizeof(MY_FLT_TYPE**));
        matLeft = (MY_FLT_TYPE***)malloc(frnn_MOP_Predict->num_multiKindOutput * sizeof(MY_FLT_TYPE**));
        matRight = (MY_FLT_TYPE***)malloc(frnn_MOP_Predict->num_multiKindOutput * sizeof(MY_FLT_TYPE**));
        for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
            matA[n] = (MY_FLT_TYPE**)malloc(frnn_MOP_Predict->numOutput * sizeof(MY_FLT_TYPE*));
            matB[n] = (MY_FLT_TYPE**)malloc(frnn_MOP_Predict->numOutput * sizeof(MY_FLT_TYPE*));
            matLeft[n] = (MY_FLT_TYPE**)malloc(frnn_MOP_Predict->numOutput * sizeof(MY_FLT_TYPE*));
            matRight[n] = (MY_FLT_TYPE**)malloc(frnn_MOP_Predict->numOutput * sizeof(MY_FLT_TYPE*));
            for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                matA[n][i] = (MY_FLT_TYPE*)calloc(num_sample * frnn_MOP_Predict->OL->numInput, sizeof(MY_FLT_TYPE));
                matB[n][i] = (MY_FLT_TYPE*)calloc(num_sample, sizeof(MY_FLT_TYPE));
                matLeft[n][i] = (MY_FLT_TYPE*)calloc(frnn_MOP_Predict->OL->numInput * frnn_MOP_Predict->OL->numInput, sizeof(MY_FLT_TYPE));
                matRight[n][i] = (MY_FLT_TYPE*)calloc(frnn_MOP_Predict->OL->numInput, sizeof(MY_FLT_TYPE));
            }
        }
    }
#endif
    //
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_FRNN
    FILE *fpt = NULL;
    char tmp_fnm[128];
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
        sprintf(tmp_fnm, "tmpFile/OUT_%s_train.csv", prob_name_MOP_Predict_FRNN);
    else
        sprintf(tmp_fnm, "tmpFile/OUT_%s_test.csv", prob_name_MOP_Predict_FRNN);
    fpt = fopen(tmp_fnm, "w");
#endif
    //
    for(int m = 0; m < num_sample; m++) {
        for(int n = 0; n < frnn_MOP_Predict->num_multiKindInput; n++) {
            int t_i = ind_in__predict_MOP_Predict_FRNN[n];
            int tmp_ind_os = n * frnn_MOP_Predict->numInput;
            if(tag_classification_MOP_Predict_FRNN) {
                if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
                    for(int i = 0; i < frnn_MOP_Predict->numInput; i++)
                        valIn[tmp_ind_os + i] = allData_MOP_Predict_FRNN[t_i][m + i * frnn_MOP_Predict->lenGap];
                else
                    for(int i = 0; i < frnn_MOP_Predict->numInput; i++)
                        valIn[tmp_ind_os + i] =
                            allData_MOP_Predict_FRNN[t_i][trainDataSize_MOP_Predict_FRNN + m + i * frnn_MOP_Predict->lenGap];
            } else {
                if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
                    for(int i = 0; i < frnn_MOP_Predict->numInput; i++)
                        valIn[tmp_ind_os + i] = allData_MOP_Predict_FRNN[t_i][m + i * frnn_MOP_Predict->lenGap];
                else
                    for(int i = 0; i < frnn_MOP_Predict->numInput; i++)
                        valIn[tmp_ind_os + i] =
                            allData_MOP_Predict_FRNN[t_i][trainDataSize_MOP_Predict_FRNN + m + i * frnn_MOP_Predict->lenGap -
                                                          (frnn_MOP_Predict->numInput - 1) * frnn_MOP_Predict->lenGap - 1];
            }
        }
        ff_frnn_Predict_FRNN(frnn_MOP_Predict, valIn, valOut, NULL);
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
        if(tag_train_test == TEST_TAG_MOP_PREDICT_FRNN)
#endif
#ifdef PRINT_ERROR_PARA_MOP_Predict_FRNN
            for(int i = 0; i < frnn_MOP_Predict->OL->numOutput; i++) {
                if(CHECK_INVALID(frnn_MOP_Predict->OL->valOutputFinal[i])) {
                    printf("%s(%d): Invalid output %d ~ %lf, exiting...\n",
                           __FILE__, __LINE__, i, frnn_MOP_Predict->OL->valOutputFinal[i]);
                    print_para_memberLayer(frnn_MOP_Predict->M1);
                    print_data_memberLayer(frnn_MOP_Predict->M1);
                    print_para_fuzzyLayer(frnn_MOP_Predict->F2);
                    print_data_fuzzyLayer(frnn_MOP_Predict->F2);
                    print_para_roughLayer(frnn_MOP_Predict->R3);
                    print_data_roughLayer(frnn_MOP_Predict->R3);
                    print_para_outReduceLayer(frnn_MOP_Predict->OL);
                    print_data_outReduceLayer(frnn_MOP_Predict->OL);
                    exit(-94628);
                }
            }
#endif
        MY_FLT_TYPE* cur_out = valOut;
        double* true_out;
        int true_label;
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
        if(!tag_classification_MOP_Predict_FRNN) {
            MY_FLT_TYPE tmp_dif1 = 0.0;
            for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
                int tmp_o = ind_out_predict_MOP_Predict_FRNN[n];
                if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
                    true_out = &allData_MOP_Predict_FRNN[tmp_o][m +
                               frnn_MOP_Predict->numInput * frnn_MOP_Predict->lenGap];
                else
                    true_out = &allData_MOP_Predict_FRNN[tmp_o][trainDataSize_MOP_Predict_FRNN + m];
                for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                    tmp_dif1 += (cur_out[n * frnn_MOP_Predict->numOutput + i] - true_out[i * frnn_MOP_Predict->lenGap]) *
                                (cur_out[n * frnn_MOP_Predict->numOutput + i] - true_out[i * frnn_MOP_Predict->lenGap]) *
                                trainFact_MOP_Predict_FRNN[tmp_o];
                }
            }
            frnn_MOP_Predict->sum_all++;
            frnn_MOP_Predict->sum_wrong += tmp_dif1 / frnn_MOP_Predict->numOutput / frnn_MOP_Predict->num_multiKindOutput;
        } else {
            if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
                true_out = &allData_MOP_Predict_FRNN[numAttr][m];
            else
                true_out = &allData_MOP_Predict_FRNN[numAttr][trainDataSize_MOP_Predict_FRNN + m];
            int cur_label = 0;
            MY_FLT_TYPE cur_val = valOut[0];
            for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
                if(cur_val < valOut[j]) {
                    cur_val = valOut[j];
                    cur_label = j;
                }
            }
            true_label = true_out[0];
            for(int j = 0; j < frnn_MOP_Predict->numOutput; j++) {
                if(j == cur_label && j == true_label) frnn_MOP_Predict->N_TP[j]++;
                if(j == cur_label && j != true_label) frnn_MOP_Predict->N_FP[j]++;
                if(j != cur_label && j == true_label) frnn_MOP_Predict->N_FN[j]++;
                if(j != cur_label && j != true_label) frnn_MOP_Predict->N_TN[j]++;
            }
            frnn_MOP_Predict->sum_all++;
            frnn_MOP_Predict->N_sum[true_label]++;
            if(cur_label != true_label) {
                frnn_MOP_Predict->sum_wrong++;
                frnn_MOP_Predict->N_wrong[true_label]++;
            }
        }
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_FRNN
        fprintf(fpt, "%e,%e\n", true_out[0], cur_out[0]);
#endif
#else
        int cur_label = 0;
        MY_FLT_TYPE cur_val = valOut[0];
        for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
            if(cur_val < valOut[j]) {
                cur_val = valOut[j];
                cur_label = j;
            }
        }
        true_label = 0;
        if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
            frnn_MOP_Predict->trading_actions[m + frnn_MOP_Predict->numInput - 1] = cur_label;
            true_label = train_trading_label_MOP_Predict_FRNN[2][m + frnn_MOP_Predict->numInput - 1];
        } else {
            frnn_MOP_Predict->trading_actions[m] = cur_label;
            true_label = test_trading_label_MOP_Predict_FRNN[2][m];
        }
        for(int j = 0; j < frnn_MOP_Predict->numOutput; j++) {
            if(j == cur_label && j == true_label) frnn_MOP_Predict->N_TP[j]++;
            if(j == cur_label && j != true_label) frnn_MOP_Predict->N_FP[j]++;
            if(j != cur_label && j == true_label) frnn_MOP_Predict->N_FN[j]++;
            if(j != cur_label && j != true_label) frnn_MOP_Predict->N_TN[j]++;
        }
        frnn_MOP_Predict->sum_all++;
        frnn_MOP_Predict->N_sum[true_label]++;
        if(cur_label != true_label) {
            frnn_MOP_Predict->sum_wrong++;
            frnn_MOP_Predict->N_wrong[true_label]++;
        }
#endif
        //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
        if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
            for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
                int tmp_o = ind_out_predict_MOP_Predict_FRNN[n];
                if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN)
                    true_out = &allData_MOP_Predict_FRNN[tmp_o][m +
                               frnn_MOP_Predict->numInput * frnn_MOP_Predict->lenGap];
                else
                    true_out = &allData_MOP_Predict_FRNN[tmp_o][trainDataSize_MOP_Predict_FRNN + m];
                for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                    for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                        int ind_cur = tmp_offset_samp * frnn_MOP_Predict->OL->numInput + j;
                        matA[n][i][ind_cur] = frnn_MOP_Predict->OL->valInputFinal[n * frnn_MOP_Predict->numOutput + i][j];
                    }
                    if(!tag_classification_MOP_Predict_FRNN)
                        matB[n][i][tmp_offset_samp] = true_out[i * frnn_MOP_Predict->lenGap];
                    else if(i == true_label)
                        matB[n][i][tmp_offset_samp] = 1;
                    else
                        matB[n][i][tmp_offset_samp] = -1;
                }
            }
        }
#else
        if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
            for(int i = 0; i < frnn_MOP_Predict->OL->numOutput; i++) {
                for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                    int ind_cur = tmp_offset_samp * frnn_MOP_Predict->OL->numInput + j;
                    matA[0][i][ind_cur] = frnn_MOP_Predict->OL->valInputFinal[i][j];
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
#ifdef OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Predict_FRNN
    fclose(fpt);
#endif
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
        frnn_MOP_Predict->sum_all = 0;
        frnn_MOP_Predict->sum_wrong = 0;
        for(int i = 0; i < frnn_MOP_Predict->numOutput * frnn_MOP_Predict->num_multiKindOutput; i++) {
            frnn_MOP_Predict->N_sum[i] = 0;
            frnn_MOP_Predict->N_wrong[i] = 0;
            frnn_MOP_Predict->e_sum[i] = 0;
            frnn_MOP_Predict->N_TP[i] = 0;
            frnn_MOP_Predict->N_TN[i] = 0;
            frnn_MOP_Predict->N_FP[i] = 0;
            frnn_MOP_Predict->N_FN[i] = 0;
        }
        //
        //printf("tmp_offset_samp = %d\n", tmp_offset_samp);
        for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
            for(int iOut = 0; iOut < frnn_MOP_Predict->numOutput; iOut++) {
                MY_FLT_TYPE lambda = 9.3132e-10;
                MY_FLT_TYPE tmp_max = 0;
                int tmp_max_flag = 0;
                for(int i = 0; i < frnn_MOP_Predict->OL->numInput; i++) {
                    for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                        int tmp_o0 = i * frnn_MOP_Predict->OL->numInput + j;
                        for(int k = 0; k < tmp_offset_samp; k++) {
                            int tmp_i1 = k * frnn_MOP_Predict->OL->numInput + i;
                            int tmp_i2 = k * frnn_MOP_Predict->OL->numInput + j;
                            matLeft[n][iOut][tmp_o0] += matA[n][iOut][tmp_i1] * matA[n][iOut][tmp_i2];
                        }
                        //if(i == j)
                        //    matLeft[tmp_o0] += lambda * fabs(matLeft[tmp_o0]);
                        if(i == j) {
                            if(!tmp_max_flag) {
                                tmp_max = matLeft[n][iOut][tmp_o0];
                                tmp_max_flag = 1;
                            } else {
                                if(tmp_max < matLeft[n][iOut][tmp_o0])
                                    tmp_max = matLeft[n][iOut][tmp_o0];
                            }
                        }
                    }
                }
                for(int i = 0; i < frnn_MOP_Predict->OL->numInput; i++) {
                    int tmp_o0 = i * frnn_MOP_Predict->OL->numInput + i;
                    matLeft[n][iOut][tmp_o0] += lambda;// *tmp_max;
                }
                for(int i = 0; i < frnn_MOP_Predict->OL->numInput; i++) {
                    for(int k = 0; k < tmp_offset_samp; k++) {
                        int tmp_i1 = k * frnn_MOP_Predict->OL->numInput + i;
                        matRight[n][iOut][i] += matA[n][iOut][tmp_i1] * matB[n][iOut][k];
                    }
                }
                int N = frnn_MOP_Predict->OL->numInput;
                int NRHS = 1;
                int LDA = N;
                int LDB = NRHS;
                int nn = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
                int* ipiv = (int*)calloc(N, sizeof(int));
                info = LAPACKE_dgesv(matStoreType, nn, nrhs, matLeft[n][iOut], lda, ipiv, matRight[n][iOut], ldb);
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
        for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
            for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                    if(CHECK_INVALID(matRight[n][i][j])) {
                        printf("%s(%d): Error - invalid value of matRight[%d][%d][%d] = %lf",
                               __FILE__, __LINE__, n, i, j, matRight[n][i][j]);
                        exit(-112);
                    }
                    frnn_MOP_Predict->OL->connectWeight[n * frnn_MOP_Predict->numOutput + i][j] = matRight[n][i][j];
                }
            }
        }
        frnn_Predict_FRNN_init(frnn_MOP_Predict, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
        //
        for(int m = 0; m < tmp_offset_samp; m++) {
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
            for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
                for(int j = 0; j < frnn_MOP_Predict->numOutput; j++) {
                    valOut[n * frnn_MOP_Predict->numOutput + j] = 0;
                    for(int k = 0; k < frnn_MOP_Predict->OL->numInput; k++) {
                        int ind_cur = m * frnn_MOP_Predict->OL->numInput + k;
                        valOut[n * frnn_MOP_Predict->numOutput + j] +=
                            matA[n][j][ind_cur] *
                            frnn_MOP_Predict->OL->connectWeight[n * frnn_MOP_Predict->numOutput + j][k];
                    }
                    if(CHECK_INVALID(valOut[n * frnn_MOP_Predict->numOutput + j])) {
                        printf("%d~%lf", j, valOut[n * frnn_MOP_Predict->numOutput + j]);
                    }
                }
            }
            MY_FLT_TYPE* cur_out = valOut;
            MY_FLT_TYPE** true_out = matB[0];
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
            if(!tag_classification_MOP_Predict_FRNN) {
                MY_FLT_TYPE tmp_loss = 0.0;
                for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
                    int tmp_o = ind_out_predict_MOP_Predict_FRNN[n];
                    true_out = matB[n];
                    for(int i = 0; i < frnn_MOP_Predict->numOutput; i++)
                        tmp_loss += (cur_out[n * frnn_MOP_Predict->numOutput + i] - true_out[i][m]) *
                                    (cur_out[n * frnn_MOP_Predict->numOutput + i] - true_out[i][m]) *
                                    trainFact_MOP_Predict_FRNN[tmp_o];
                }
                frnn_MOP_Predict->sum_all++;
                frnn_MOP_Predict->sum_wrong += tmp_loss / frnn_MOP_Predict->numOutput / frnn_MOP_Predict->num_multiKindOutput;
            } else {
                int cur_label = 0;
                MY_FLT_TYPE cur_val = valOut[0];
                for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
                    if(cur_val < valOut[j]) {
                        cur_val = valOut[j];
                        cur_label = j;
                    }
                }
                int true_label = 0;
                MY_FLT_TYPE true_val = true_out[0][m];
                for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
                    if(true_val < true_out[j][m]) {
                        true_val = true_out[j][m];
                        true_label = j;
                    }
                }
                for(int j = 0; j < frnn_MOP_Predict->numOutput; j++) {
                    if(j == cur_label && j == true_label) frnn_MOP_Predict->N_TP[j]++;
                    if(j == cur_label && j != true_label) frnn_MOP_Predict->N_FP[j]++;
                    if(j != cur_label && j == true_label) frnn_MOP_Predict->N_FN[j]++;
                    if(j != cur_label && j != true_label) frnn_MOP_Predict->N_TN[j]++;
                }
                frnn_MOP_Predict->sum_all++;
                frnn_MOP_Predict->N_sum[true_label]++;
                if(cur_label != true_label) {
                    frnn_MOP_Predict->sum_wrong++;
                    frnn_MOP_Predict->N_wrong[true_label]++;
                }
            }
#else
            int cur_label = 0;
            MY_FLT_TYPE cur_val = valOut[0];
            for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
                if(cur_val < valOut[j]) {
                    cur_val = valOut[j];
                    cur_label = j;
                }
            }
            if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
                frnn_MOP_Predict->trading_actions[m + frnn_MOP_Predict->numInput - 1] = cur_label;
            } else {
                frnn_MOP_Predict->trading_actions[m] = cur_label;
            }
            int true_label = 0;
            MY_FLT_TYPE true_val = true_out[0][m];
            for(int j = 1; j < frnn_MOP_Predict->numOutput; j++) {
                if(true_val < true_out[j][m]) {
                    true_val = true_out[j][m];
                    true_label = j;
                }
            }
            for(int j = 0; j < frnn_MOP_Predict->numOutput; j++) {
                if(j == cur_label && j == true_label) frnn_MOP_Predict->N_TP[j]++;
                if(j == cur_label && j != true_label) frnn_MOP_Predict->N_FP[j]++;
                if(j != cur_label && j == true_label) frnn_MOP_Predict->N_FN[j]++;
                if(j != cur_label && j != true_label) frnn_MOP_Predict->N_TN[j]++;
            }
            frnn_MOP_Predict->sum_all++;
            frnn_MOP_Predict->N_sum[true_label]++;
            if(cur_label != true_label) {
                frnn_MOP_Predict->sum_wrong++;
                frnn_MOP_Predict->N_wrong[true_label]++;
            }
#endif
        }
        //
        for(int n = 0; n < frnn_MOP_Predict->num_multiKindOutput; n++) {
            for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                free(matA[n][i]);
                free(matB[n][i]);
                free(matLeft[n][i]);
                free(matRight[n][i]);
            }
            free(matA[n]);
            free(matB[n]);
            free(matLeft[n]);
            free(matRight[n]);
        }
        free(matA);
        free(matB);
        free(matLeft);
        free(matRight);
    }
#endif
    //
    return;
}

void Finalize_MOP_Predict_FRNN()
{
    frnn_Predict_FRNN_free(frnn_MOP_Predict);
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
void frnn_Predict_FRNN_setup(frnn_MOP_Predict_FRNN* frnn)
{
    frnn->num_multiKindInput = num_in__predict_MOP_Predict_FRNN;
    if(frnn->tag_multiKindInput == FLAG_STATUS_ON) {
        if(NORMALIZE_MOP_Predict_FRNN != NORMALIZE_MIN_MAX_MOP_Predict_FRNN &&
           NORMALIZE_MOP_Predict_FRNN != NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
            printf("%s(%d): If different kinds of inputs are utilized, all inputs should be normalized, exiting...\n",
                   __FILE__, __LINE__);
            exit(-1759);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //
    frnn->numInput = 3;
    frnn->lenGap = 1;
    if(tag_classification_MOP_Predict_FRNN)
        frnn->numInput = 1;
#if CURRENT_PROB_MOP_PREDICT_FRNN == PREDICT_CLASSIFY_MOP_PREDICT_FRNN
    int numOutput = 1;
    if(tag_classification_MOP_Predict_FRNN)
        numOutput = num_class_MOP_Predict_FRNN;
#else
    int numOutput = NUM_LABEL_TRADING_MOP_PREDICT_FRNN;
#endif
    frnn->num_multiKindOutput = num_out_predict_MOP_Predict_FRNN;
    if(frnn->tag_multiKindInput == FLAG_STATUS_OFF)
        frnn->num_multiKindOutput = 1;
    frnn->numOutput = numOutput;
    //
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    if(!flag_T2_MOP_Predict_FRNN) typeFuzzySet = FUZZY_SET_I;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE; // MUL_EACH_IN_TO_ONE_RULE; //
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;// FIXED_ROUGH_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int flagConnectStatus = FLAG_STATUS_OFF;
    int flagConnectWeight = FLAG_STATUS_OFF;
    if(frnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    if(frnn->num_multiKindOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    if(tag_classification_MOP_Predict_FRNN) {
        consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
        //centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    //
    frnn->typeFuzzySet = typeFuzzySet;
    frnn->typeRules = typeRules;
    frnn->typeInRuleCorNum = typeInRuleCorNum;
    frnn->typeTypeReducer = typeTypeReducer;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
    frnn->centroid_num_tag = centroid_num_tag;
    frnn->flagConnectStatus = flagConnectStatus;
    frnn->flagConnectWeight = flagConnectWeight;
    //
#if MF_RULE_NUM_MOP_PREDICT_FRNN_CUR == MF_RULE_NUM_MOP_PREDICT_FRNN_LESS
    int numFuzzyRules = 50;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#else
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#endif
    if(numFuzzyRules > DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL)
        numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    //numRoughSets = 2 * frnn->numOutput * frnn->num_multiKindOutput;
    //numRoughSets = numFuzzyRules / 2;
    //if(numRoughSets < 3)
    numRoughSets = 10;
    //
    int GEP_head_len = 8;
    frnn->GEP_head_len = GEP_head_len;
    //
    frnn->layerNum = 8;
    frnn->numRules = numFuzzyRules;
    frnn->numRoughs = numRoughSets;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;
    //
    int numInputConsequenceNode = 0;
    if(type_frnn_cnsq_MOP_Predict_FRNN == FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_FIXED) {
        numInputConsequenceNode = 0;
        consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
        frnn->consequenceNodeStatus = consequenceNodeStatus;
    } else if(type_frnn_cnsq_MOP_Predict_FRNN == FRNN_CONSEQUENCE_MOP_PREDICT_FRNN_ADAPT) {
        if(flag_multi_in_cnsq_MOP_Predict_FRNN == FLAG_STATUS_OFF)
            numInputConsequenceNode = frnn->numInput;
        else
            numInputConsequenceNode = frnn->numInput * frnn->num_multiKindInput;
        consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
        frnn->consequenceNodeStatus = consequenceNodeStatus;
    } else {
        printf("%s(%d): Invalid ``type_frnn_cnsq_MOP_Predict_FRNN''-%d, exiting...\n",
               __FILE__, __LINE__, type_frnn_cnsq_MOP_Predict_FRNN);
        exit(-12759);
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    flagConnectStatus = FLAG_STATUS_OFF;
    frnn->flagConnectStatus = flagConnectStatus;
    flagConnectWeight = FLAG_STATUS_ON;
    frnn->flagConnectWeight = flagConnectWeight;
    tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;
    //numInputConsequenceNode = 0;
    //consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
    //frnn->consequenceNodeStatus = consequenceNodeStatus;
    centroid_num_tag = CENTROID_ALL_ONESET;
    if(frnn->num_multiKindOutput > 1 || frnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
    }
    frnn->centroid_num_tag = centroid_num_tag;
    if(numRoughSets < numOutput) {
        numRoughSets = numOutput;
        frnn->numRoughs = numRoughSets;
    }
#endif
    //
    MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {
            inputMin[i] = 0;
            inputMax[i] = 1;
        } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
            inputMin[i] = -10;
            inputMax[i] = 10;
        } else {
            inputMin[i] = 0;
            inputMax[i] = 120;
        }
    }
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        frnn->num_GEP = frnn->numInput * frnn->num_multiKindInput;
        frnn->GEP0 = (codingGEP**)malloc(frnn->num_GEP * sizeof(codingGEP*));
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->GEP0[n] = setupCodingGEP(frnn->numInput, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                           frnn->GEP_head_len,
                                           FLAG_STATUS_OFF,
                                           PARA_MIN_VAL_GEP_CFRNN_MODEL,
                                           PARA_MAX_VAL_GEP_CFRNN_MODEL);
        }
        for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
            inputMin[i] = -10;
            inputMax[i] = 10;
        }
    }
    if(frnn->tag_DIF == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_multiKindInput; n++) {
            int tmp_ind_os = n * frnn->numInput;
            if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {
                inputMin[tmp_ind_os + 0] = 0;
                inputMax[tmp_ind_os + 0] = 1;
            } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
                inputMin[tmp_ind_os + 0] = -10;
                inputMax[tmp_ind_os + 0] = 10;
            } else {
                inputMin[tmp_ind_os + 0] = 0;
                inputMax[tmp_ind_os + 0] = 120;
            }
            for(int i = 1; i < frnn->numInput; i++) {
                if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {
                    inputMin[tmp_ind_os + i] = -1;
                    inputMax[tmp_ind_os + i] = 1;
                } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
                    inputMin[tmp_ind_os + i] = -10;
                    inputMax[tmp_ind_os + i] = 10;
                } else {
                    inputMin[tmp_ind_os + i] = -10;
                    inputMax[tmp_ind_os + i] = 10;
                }
            }
        }
    }
    int* numMemship = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
#if MF_RULE_NUM_MOP_PREDICT_FRNN_CUR == MF_RULE_NUM_MOP_PREDICT_FRNN_LESS
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#else
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#endif
    }
    int* flagAdapMemship = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        flagAdapMemship[i] = FLAG_STATUS_ON;
    }
    frnn->M1 = setupMemberLayer(frnn->numInput * frnn->num_multiKindInput, inputMin, inputMax,
                                numMemship, flagAdapMemship, frnn->typeFuzzySet,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    frnn->F2 = setupFuzzyLayer(frnn->numInput * frnn->num_multiKindInput, frnn->M1->numMembershipFun, frnn->numRules,
                               frnn->typeFuzzySet, frnn->typeRules,
                               frnn->typeInRuleCorNum, frnn->tag_GEPr,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, FLAG_STATUS_OFF);
    frnn->R3 = setupRoughLayer(frnn->numRules, frnn->numRoughs, frnn->typeFuzzySet,
                               FLAG_STATUS_ON,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    MY_FLT_TYPE outputMin[MAX_OUT_NUM_MOP_Predict_FRNN];
    MY_FLT_TYPE outputMax[MAX_OUT_NUM_MOP_Predict_FRNN];
    for(int i = 0; i < frnn->numOutput * frnn->num_multiKindOutput; i++) {
        if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {

            outputMin[i] = 0;
            outputMax[i] = 1;
        } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
            outputMin[i] = -10;
            outputMax[i] = 10;
        } else {
            outputMin[i] = 0;
            outputMax[i] = 120;
        }
    }
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {
            inputMin[i] = 0;
            inputMax[i] = 1;
        } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {
            inputMin[i] = -10;
            inputMax[i] = 10;
        } else {
            inputMin[i] = 0;
            inputMax[i] = 120;
        }
    }
    frnn->OL = setupOutReduceLayer(frnn->R3->numRoughSets, frnn->numOutput * frnn->num_multiKindOutput, outputMin, outputMax,
                                   frnn->typeFuzzySet, frnn->typeTypeReducer,
                                   frnn->consequenceNodeStatus, frnn->centroid_num_tag,
                                   numInputConsequenceNode, inputMin, inputMax,
                                   frnn->flagConnectStatus, frnn->flagConnectWeight,
                                   tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    //
    free(inputMin);
    free(inputMax);
    free(numMemship);
    free(flagAdapMemship);
    //
    frnn->numParaLocal = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal += frnn->GEP0[n]->numParaLocal;
        }
    }
    frnn->numParaLocal += frnn->M1->numParaLocal;
    frnn->numParaLocal += frnn->F2->numParaLocal;
    frnn->numParaLocal += frnn->R3->numParaLocal + frnn->OL->numParaLocal;
    //
    frnn->numParaLocal_disc = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal_disc += frnn->GEP0[n]->numParaLocal_disc;
        }
    }
    frnn->numParaLocal_disc += frnn->M1->numParaLocal_disc;
    frnn->numParaLocal_disc += frnn->F2->numParaLocal_disc;
    frnn->numParaLocal_disc += frnn->R3->numParaLocal_disc + frnn->OL->numParaLocal_disc;
    frnn->layerNum = 4;
    //
    frnn->xType = (int*)malloc(frnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            memcpy(&frnn->xType[tmp_cnt_p], frnn->GEP0[n]->xType, frnn->GEP0[n]->numParaLocal * sizeof(int));
            tmp_cnt_p += frnn->GEP0[n]->numParaLocal;
        }
    }
    memcpy(&frnn->xType[tmp_cnt_p], frnn->M1->xType, frnn->M1->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->M1->numParaLocal;
    memcpy(&frnn->xType[tmp_cnt_p], frnn->F2->xType, frnn->F2->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->F2->numParaLocal;
    memcpy(&frnn->xType[tmp_cnt_p], frnn->R3->xType, frnn->R3->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->R3->numParaLocal;
    memcpy(&frnn->xType[tmp_cnt_p], frnn->OL->xType, frnn->OL->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->OL->numParaLocal;
    //tmp_cnt_p = 0;
    //for(int i = 0; i < cfrnn->numParaLocal; i++) {
    //    if(cfrnn->xType[i] != VAR_TYPE_CONTINUOUS)
    //        tmp_cnt_p++;
    //}
    //printf("%d ~ %d \n", tmp_cnt_p, cfrnn->numParaLocal_disc);

    frnn->e = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_sum = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_wrong = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->e_sum = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_TP = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_TN = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FP = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FN = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->money_in_hand = 100000;
    frnn->trading_actions = (int*)calloc(MAX_DATA_LEN_MOP_PREDICT_FRNN, sizeof(int));
    frnn->num_stock_held = 0;

    frnn->featureMapTagInitial = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    frnn->dataflowInitial = (MY_FLT_TYPE*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        frnn->featureMapTagInitial[i] = 1;
        frnn->dataflowInitial[i] = 1;
    }

    if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->numInput * frnn->num_multiKindInput * frnn->numRules * frnn->numRoughs *
                                          frnn->numOutput * frnn->num_multiKindOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->numInput * frnn->num_multiKindInput * frnn->numRules +
                                            frnn->numRules * frnn->numRoughs);
    } else {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->numRules * frnn->numRoughs * frnn->numOutput *
                                          frnn->num_multiKindOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->numRules +
                                            frnn->numRules * frnn->numRoughs);
    }
    //
    return;
}

void frnn_Predict_FRNN_free(frnn_MOP_Predict_FRNN* frnn)
{
    freeOutReduceLayer(frnn->OL);
    freeRoughLayer(frnn->R3);
    freeFuzzyLayer(frnn->F2);
    freeMemberLayer(frnn->M1);
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int i = 0; i < frnn->num_GEP; i++) {
            freeCodingGEP(frnn->GEP0[i]);
        }
        free(frnn->GEP0);
    }

    free(frnn->xType);

    free(frnn->e);

    free(frnn->N_sum);
    free(frnn->N_wrong);
    free(frnn->e_sum);

    free(frnn->N_TP);
    free(frnn->N_TN);
    free(frnn->N_FP);
    free(frnn->N_FN);

    free(frnn->trading_actions);

    free(frnn->featureMapTagInitial);
    free(frnn->dataflowInitial);

    free(frnn);

    return;
}

void frnn_Predict_FRNN_init(frnn_MOP_Predict_FRNN* frnn, double* x, int mode)
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

    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            assignCodingGEP(frnn->GEP0[n], &x[count], mode);
            count += frnn->GEP0[n]->numParaLocal;
        }
    }
    assignMemberLayer(frnn->M1, &x[count], mode);
    count += frnn->M1->numParaLocal;
    assignFuzzyLayer(frnn->F2, &x[count], mode);
    count += frnn->F2->numParaLocal;
    assignRoughLayer(frnn->R3, &x[count], mode);
    count += frnn->R3->numParaLocal;
    assignOutReduceLayer(frnn->OL, &x[count], mode);
    count += frnn->OL->numParaLocal;
    //
    return;
}

void ff_frnn_Predict_FRNN(frnn_MOP_Predict_FRNN* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                          MY_FLT_TYPE** inputConsequenceNode)
{
    int len_valIn = frnn->numInput * frnn->num_multiKindInput;
    MY_FLT_TYPE* tmpIn = (MY_FLT_TYPE*)malloc(len_valIn * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmpOut = (MY_FLT_TYPE*)malloc(frnn->num_GEP * sizeof(MY_FLT_TYPE));
    memcpy(tmpIn, valIn, frnn->numInput * frnn->num_multiKindInput * sizeof(MY_FLT_TYPE));
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            int tmp_ind = n / frnn->numInput;
            decodingGEP(frnn->GEP0[n], &tmpIn[tmp_ind * frnn->numInput], &tmpOut[n]);
            //printf("%lf ", tmpOut[n]);
        }
        ff_memberLayer(frnn->M1, tmpOut, frnn->dataflowInitial);
    } else if(frnn->tag_DIF == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_multiKindInput; n++) {
            int tmp_ind_os = n * frnn->numInput;
            for(int i = 1; i < frnn->numInput; i++) {
                tmpIn[tmp_ind_os + i] = valIn[tmp_ind_os + i - 1] - valIn[tmp_ind_os + i];
            }
        }
        ff_memberLayer(frnn->M1, tmpIn, frnn->dataflowInitial);
    } else {
        ff_memberLayer(frnn->M1, valIn, frnn->dataflowInitial);
    }
    free(tmpIn);
    free(tmpOut);
    //
    ff_fuzzyLayer(frnn->F2, frnn->M1->degreeMembership, frnn->M1->dataflowStatus);
    ff_roughLayer(frnn->R3, frnn->F2->degreeRules, frnn->F2->dataflowStatus);
    //
    if(frnn_MOP_Predict->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        if(flag_multi_in_cnsq_MOP_Predict_FRNN == FLAG_STATUS_ON) {
            for(int n = 0; n < frnn->num_multiKindOutput; n++) {
                for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                    memcpy(frnn_MOP_Predict->OL->inputConsequenceNode[n * frnn_MOP_Predict->numOutput + i],
                           valIn,
                           frnn_MOP_Predict->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
                }
            }
        } else {
            for(int n = 0; n < frnn->num_multiKindOutput; n++) {
                int tmp_o = ind_out_predict_MOP_Predict_FRNN[n];
                for(int i = 0; i < frnn_MOP_Predict->numOutput; i++) {
                    memcpy(frnn_MOP_Predict->OL->inputConsequenceNode[n * frnn_MOP_Predict->numOutput + i],
                           &valIn[tmp_o * frnn->numInput],
                           frnn_MOP_Predict->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
                }
            }
        }
    }
    ff_outReduceLayer(frnn->OL, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
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
    memcpy(valOut, frnn->OL->valOutputFinal, frnn->OL->numOutput * sizeof(MY_FLT_TYPE));
    //
    return;
}

static double simplicity_MOP_Predict_FRNN()
{
    //
    double f_simpl = 0.0;
    double f_simpl_gl = 0.0;
    double f_simpl_fl = 0.0;
    double f_simpl_rl = 0.0;
    total_penalty_MOP_Predict_FRNN = 0.0;
    //
    int *tmp_rule, *tmp_rough, **tmp_mem;
    tmp_rule = (int*)calloc(frnn_MOP_Predict->F2->numRules, sizeof(int));
    tmp_rough = (int*)calloc(frnn_MOP_Predict->R3->numRoughSets, sizeof(int));
    tmp_mem = (int**)malloc(frnn_MOP_Predict->M1->numInput * sizeof(int*));
    for(int i = 0; i < frnn_MOP_Predict->M1->numInput; i++) {
        tmp_mem[i] = (int*)calloc(frnn_MOP_Predict->M1->numMembershipFun[i], sizeof(int));
    }

    if(frnn_MOP_Predict->tag_GEP) {
        for(int i = 0; i < frnn_MOP_Predict->num_GEP; i++) {
            int tmp_g = 0;
            for(int j = 0; j < frnn_MOP_Predict->GEP0[i]->check_head; j++) {
                if(frnn_MOP_Predict->GEP0[i]->check_op[j] >= 0) {
                    tmp_g++;
                }
            }
            f_simpl_gl += (double)tmp_g / frnn_MOP_Predict->GEP0[i]->GEP_head_length;
        }
    }
    if(frnn_MOP_Predict->tag_GEPr == FLAG_STATUS_OFF) {
        for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
            tmp_rule[i] = 0;
            for(int j = 0; j < frnn_MOP_Predict->M1->numInput; j++) {
                int tmp_count = 0;
                for(int k = 0; k < frnn_MOP_Predict->M1->numMembershipFun[j]; k++) {
                    if(frnn_MOP_Predict->F2->connectStatusAll[i][j][k]) {
                        tmp_count++;
                        tmp_mem[j][k]++;
                    }
                }
                if(tmp_count) {
                    tmp_rule[i]++;
                }
            }
            f_simpl_fl += (double)tmp_rule[i] / frnn_MOP_Predict->M1->numInput;
        }
        for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
            tmp_rough[i] = 0;
            for(int j = 0; j < frnn_MOP_Predict->F2->numRules; j++) {
                if(tmp_rule[j] && frnn_MOP_Predict->R3->connectStatus[i][j]) {
                    tmp_rough[i]++;
                }
            }
            f_simpl_rl += (double)tmp_rough[i] / frnn_MOP_Predict->F2->numRules;
        }
    } else {
        for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
            for(int j = 0; j < frnn_MOP_Predict->M1->numInput; j++) {
                for(int k = 0; k < frnn_MOP_Predict->M1->numMembershipFun[j]; k++) {
                    if(frnn_MOP_Predict->F2->connectStatusAll[i][j][k]) {
                        tmp_mem[j][k]++;
                    }
                }
            }
            //
            tmp_rule[i] = 0;
            for(int j = 0; j < frnn_MOP_Predict->F2->ruleGEP[i]->check_head; j++) {
                if(frnn_MOP_Predict->F2->ruleGEP[i]->check_op[j] >= 0) {
                    tmp_rule[i]++;
                }
            }
            f_simpl_fl += (double)tmp_rule[i] / frnn_MOP_Predict->F2->ruleGEP[i]->GEP_head_length;
        }
        //
        for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
            tmp_rough[i] = 0;
            for(int j = 0; j < frnn_MOP_Predict->F2->numRules; j++) {
                if(frnn_MOP_Predict->R3->connectStatus[i][j]) {
                    tmp_rough[i]++;
                }
            }
            f_simpl_rl += (double)tmp_rough[i] / frnn_MOP_Predict->F2->numRules;
        }
    }
    if(frnn_MOP_Predict->tag_GEP)
        f_simpl = (f_simpl_gl + f_simpl_fl + f_simpl_rl) /
                  (frnn_MOP_Predict->num_GEP + frnn_MOP_Predict->F2->numRules + frnn_MOP_Predict->R3->numRoughSets);
    else
        f_simpl = (f_simpl_fl + f_simpl_rl) /
                  (frnn_MOP_Predict->F2->numRules + frnn_MOP_Predict->R3->numRoughSets);
    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    int tmp_sum = 0;
    for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
        tmp_sum += tmp_rule[i];
    }
    if(tmp_sum == 0) {
        total_penalty_MOP_Predict_FRNN += penaltyVal_MOP_Predict_FRNN;
    }
    //tmp_sum = 0;
    //for(int i = 0; i < NUM_CLASS_MOP_Predict_FRNN; i++) {
    //    tmp_sum += tmp2[i];
    //}
    //if(tmp_sum == 0.0) {
    //    f_prcsn += 1e6;
    //    f_simpl += 1e6;
    //}
    tmp_sum = 0;
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        if(tmp_rough[i])
            tmp_sum++;
        //tmp_sum += tmp_rough[i];
        //if(tmp_rough[i] == 0)
        //    total_penalty_MOP_Predict_FRNN += penaltyVal_MOP_Predict_FRNN;
    }
    if(tmp_sum < THRESHOLD_NUM_ROUGH_NODES) {
        total_penalty_MOP_Predict_FRNN += penaltyVal_MOP_Predict_FRNN * (THRESHOLD_NUM_ROUGH_NODES - tmp_sum);
    }
    //
    free(tmp_rule);
    free(tmp_rough);
    for(int i = 0; i < frnn_MOP_Predict->M1->numInput; i++) {
        free(tmp_mem[i]);
    }
    free(tmp_mem);
    //
    return f_simpl;
}

static double generality_MOP_Predict_FRNN()
{
    double tmp_sum = 0.0;
    int tmp_cnt = 0;
    //
    if(frnn_MOP_Predict->OL->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < frnn_MOP_Predict->OL->numOutput; i++) {
            if(frnn_MOP_Predict->OL->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                continue;
            }
            for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                for(int k = 0; k < frnn_MOP_Predict->OL->dim_degree; k++) {
                    if(frnn_MOP_Predict->OL->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                        continue;
                    }
                    for(int m = 0; m < frnn_MOP_Predict->OL->numInputConsequenceNode; m++) {
                        tmp_sum += fabs(frnn_MOP_Predict->OL->paraConsequenceNode[i][j][k][m]);
                        tmp_cnt++;
                    }
                }
            }
        }
        //printf("Tag 1\n");
    }
    if(frnn_MOP_Predict->OL->flagConnectWeightAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < frnn_MOP_Predict->OL->numOutput; i++) {
            for(int j = 0; j < frnn_MOP_Predict->OL->numInput; j++) {
                tmp_sum += fabs(frnn_MOP_Predict->OL->connectWeight[i][j]);
                tmp_cnt++;
            }
        }
        //printf("Tag 2\n");
    }
    if(tmp_cnt)
        tmp_sum /= tmp_cnt;
    //
    return tmp_sum;
}

static double get_profit_MOP_Predict_FRNN(int tag_train_test)
{
    //
    frnn_MOP_Predict->money_init = 100000;
    frnn_MOP_Predict->money_in_hand = 100000;
    frnn_MOP_Predict->num_stock_held = 0;
    //
    double total_profit = 0;
    double* all_close_prices = NULL;
    int num_close_prices = 0;
    int* trading_actions = frnn_MOP_Predict->trading_actions;
    //
    if(tag_train_test == TRAIN_TAG_MOP_PREDICT_FRNN) {
        all_close_prices = allData_MOP_Predict_FRNN[0];
        num_close_prices = trainDataSize_MOP_Predict_FRNN - frnn_MOP_Predict->numInput + 1;
    } else {
        all_close_prices = &allData_MOP_Predict_FRNN[0][trainDataSize_MOP_Predict_FRNN];
        num_close_prices = testDataSize_MOP_Predict_FRNN;
    }
    //
    for(int i = 0; i < num_close_prices; i++) {
        int cur_i = i + frnn_MOP_Predict->numInput - 1;
        if(tag_train_test == TEST_TAG_MOP_PREDICT_FRNN)
            cur_i = i;
        if(trading_actions[cur_i] == CLASS_IND_BUY_MOP_PREDICT_FRNN) {
            if(frnn_MOP_Predict->money_in_hand > all_close_prices[cur_i]) {
                frnn_MOP_Predict->money_in_hand -= all_close_prices[cur_i];
                frnn_MOP_Predict->num_stock_held++;
            }
        } else if(trading_actions[cur_i] == CLASS_IND_SELL_MOP_PREDICT_FRNN) {
            if(frnn_MOP_Predict->num_stock_held > 0) {
                frnn_MOP_Predict->money_in_hand += all_close_prices[cur_i];
                frnn_MOP_Predict->num_stock_held--;
            }
        }
    }
    //
    total_profit = (frnn_MOP_Predict->money_in_hand - frnn_MOP_Predict->money_init) / frnn_MOP_Predict->money_init;
    total_profit = 1 - total_profit;
    //
    return total_profit;
}

void statistics_MOP_Predict_FRNN(FILE* fpt)
{
    //
    print_para_memberLayer(frnn_MOP_Predict->M1);
    print_data_memberLayer(frnn_MOP_Predict->M1);
    print_para_fuzzyLayer(frnn_MOP_Predict->F2);
    print_data_fuzzyLayer(frnn_MOP_Predict->F2);
    print_para_roughLayer(frnn_MOP_Predict->R3);
    print_data_roughLayer(frnn_MOP_Predict->R3);
    print_para_outReduceLayer(frnn_MOP_Predict->OL);
    print_data_outReduceLayer(frnn_MOP_Predict->OL);
    //
    int *tmp_rule, **tmp_mem;
    tmp_rule = (int*)calloc(frnn_MOP_Predict->F2->numRules, sizeof(int));
    tmp_mem = (int**)malloc(frnn_MOP_Predict->M1->numInput * sizeof(int*));
    for(int i = 0; i < frnn_MOP_Predict->M1->numInput; i++) {
        tmp_mem[i] = (int*)calloc(frnn_MOP_Predict->M1->numMembershipFun[i], sizeof(int));
    }
    int *tmp_rough, *tmp_rough_op, *tmp_rough_in;
    tmp_rough = (int*)calloc(frnn_MOP_Predict->R3->numRoughSets, sizeof(int));
    tmp_rough_op = (int*)calloc(frnn_MOP_Predict->R3->numRoughSets, sizeof(int));
    tmp_rough_in = (int*)calloc(frnn_MOP_Predict->R3->numRoughSets, sizeof(int));
    int *tmp_rule_op, *tmp_rule_in;
    tmp_rule_op = (int*)calloc(frnn_MOP_Predict->F2->numRules, sizeof(int));
    tmp_rule_in = (int*)calloc(frnn_MOP_Predict->F2->numRules, sizeof(int));
    //
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_MOP_Predict->F2->numRules; j++) {
            if(frnn_MOP_Predict->R3->connectStatus[i][j]) {
                tmp_rule[j]++;
            }
        }
    }
    //
    if(frnn_MOP_Predict->tag_GEPr == FLAG_STATUS_OFF) {
        for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
            for(int j = 0; j < frnn_MOP_Predict->M1->numInput; j++) {
                for(int k = 0; k < frnn_MOP_Predict->M1->numMembershipFun[j]; k++) {
                    if(frnn_MOP_Predict->F2->connectStatusAll[i][j][k]) {
                        tmp_mem[j][k] += tmp_rule[i];
                        tmp_rule_in[i]++;
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
            for(int j = 0; j < frnn_MOP_Predict->F2->ruleGEP[i]->check_tail; j++) {
                if(frnn_MOP_Predict->F2->ruleGEP[i]->check_vInd[j] >= 0) {
                    tmp_rule_in[i]++;
                    int cur_in = frnn_MOP_Predict->F2->ruleGEP[i]->check_vInd[j];
                    for(int k = 0; k < frnn_MOP_Predict->M1->numMembershipFun[cur_in]; k++) {
                        if(frnn_MOP_Predict->F2->connectStatusAll[i][cur_in][k]) {
                            tmp_mem[cur_in][k] += tmp_rule[i];
                            break;
                        }
                    }
                }
                if(frnn_MOP_Predict->F2->ruleGEP[i]->check_op[j] >= 0) {
                    tmp_rule_op[i]++;
                }
            }
        }
    }
    //
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_MOP_Predict->F2->numRules; j++) {
            if(frnn_MOP_Predict->R3->connectStatus[i][j]) {
                tmp_rough[i]++;
                tmp_rough_op[i] += tmp_rule_op[j];
                tmp_rough_in[i] += tmp_rule_in[j];
            }
        }
    }
    //
    for(int j = 0; j < frnn_MOP_Predict->M1->numInput; j++) {
        for(int k = 0; k < frnn_MOP_Predict->M1->numMembershipFun[j]; k++) {
            printf("%d,", tmp_mem[j][k]);
            fprintf(fpt, "%d,", tmp_mem[j][k]);
        }
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
        printf("%d,", tmp_rule[i]);
        fprintf(fpt, "%d,", tmp_rule[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
        printf("%d,", tmp_rule_op[i]);
        fprintf(fpt, "%d,", tmp_rule_op[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->F2->numRules; i++) {
        printf("%d,", tmp_rule_in[i]);
        fprintf(fpt, "%d,", tmp_rule_in[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough[i]);
        fprintf(fpt, "%d,", tmp_rough[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough_op[i]);
        fprintf(fpt, "%d,", tmp_rough_op[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    for(int i = 0; i < frnn_MOP_Predict->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough_in[i]);
        fprintf(fpt, "%d,", tmp_rough_in[i]);
    }
    printf("\n");
    fprintf(fpt, "\n");
    //
    if(frnn_MOP_Predict->tag_GEP) {
    }
    //
    free(tmp_rule);
    for(int i = 0; i < frnn_MOP_Predict->M1->numInput; i++) {
        free(tmp_mem[i]);
    }
    free(tmp_mem);
    free(tmp_rough);
    free(tmp_rough_op);
    free(tmp_rough_in);
    //
    free(tmp_rule_op);
    free(tmp_rule_in);
    //
    return;
}

static void readData_stock_MOP_Predict_FRNN(char* fname, int trainNo, int testNo, int endNo)
{
    FILE* fpt;
    if((fpt = fopen(fname, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }
    trainDataSize_MOP_Predict_FRNN = 0;
    testDataSize_MOP_Predict_FRNN = 0;
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
        trimLine_MOP_Predict_FRNN(StrLine);
        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        int tmp_size_pre = 0;
        int tmp_size = 0;
        for(int iK = 0; iK < numAttr; iK++) {
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
                    allData_MOP_Predict_FRNN[iK][trainDataSize_MOP_Predict_FRNN + tmp_cnt] = elem;
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
        trainDataSize_MOP_Predict_FRNN += tmp_size;
        fclose(fpt_data);
    }
    for(seq = testNo; seq < endNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_MOP_Predict_FRNN(StrLine);
        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        int tmp_size_pre = 0;
        int tmp_size = 0;
        for(int iK = 0; iK < numAttr; iK++) {
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
                    allData_MOP_Predict_FRNN[iK][trainDataSize_MOP_Predict_FRNN +
                                                 testDataSize_MOP_Predict_FRNN + tmp_cnt] = elem;
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
        testDataSize_MOP_Predict_FRNN += tmp_size;
        fclose(fpt_data);
    }
    allDataSize_MOP_Predict_FRNN = trainDataSize_MOP_Predict_FRNN + testDataSize_MOP_Predict_FRNN;
    //
    free(buf);
    fclose(fpt);
}

static void readData_general_MOP_Predict_FRNN(char* fname, int tag_classification)
{
    FILE* fpt;
    if((fpt = fopen(fname, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }
    allDataSize_MOP_Predict_FRNN = 0;
    trainDataSize_MOP_Predict_FRNN = 0;
    testDataSize_MOP_Predict_FRNN = 0;
    //
    char tmp_delim[] = " ,\t\r\n";
    int max_buf_size = 100 * MAX_ATTR_NUM + 1;
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
            allDataSize_MOP_Predict_FRNN = elem_int;
        } else if(tmp_cnt == 1) {
            numAttr = elem_int;
        } else {
            if(tag_classification && tmp_cnt == 2) {
                num_class_MOP_Predict_FRNN = elem_int;
            } else {
                printf("\n%s(%d):too many data...\n", __FILE__, __LINE__);
                exit(1002);
            }
        }
        tmp_cnt++;
    }
    //get data
    int seq = 0;
    for(seq = 0; seq < allDataSize_MOP_Predict_FRNN; seq++) {
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
            allData_MOP_Predict_FRNN[tmp_cnt][seq] = elem;
            tmp_cnt++;
        }
        if(numAttr != tmp_cnt) {
            printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                   __FILE__, __LINE__, seq, numAttr, tmp_cnt);
            exit(2005);
        }
    }
    //
    if(tag_classification)
        numAttr--;
    //
    if(!tag_classification) {
        trainDataSize_MOP_Predict_FRNN = allDataSize_MOP_Predict_FRNN * 2 / 3;
        testDataSize_MOP_Predict_FRNN = allDataSize_MOP_Predict_FRNN - trainDataSize_MOP_Predict_FRNN;
    } else {
        int tmp_stratified_ind[MAX_DATA_LEN_MOP_PREDICT_FRNN];
        int tmp_ind1[MAX_DATA_LEN_MOP_PREDICT_FRNN];
        int tmp_ind2[MAX_DATA_LEN_MOP_PREDICT_FRNN];
        int tmp_cnt = 0;
        for(int n = 0; n < num_class_MOP_Predict_FRNN; n++) {
            for(int i = 0; i < allDataSize_MOP_Predict_FRNN; i++)
                if(allData_MOP_Predict_FRNN[numAttr][i] == n)
                    tmp_stratified_ind[tmp_cnt++] = i;
        }
        int tmp_cnt1 = 0;
        int tmp_cnt2 = 0;
        for(int i = 0; i < allDataSize_MOP_Predict_FRNN; i++) {
            if(i % repNum_MOP_Predict_FRNN == repNo_MOP_Predict_FRNN) {
                tmp_ind2[tmp_cnt2] = tmp_stratified_ind[i];
                tmp_cnt2++;
            } else {
                tmp_ind1[tmp_cnt1] = tmp_stratified_ind[i];
                tmp_cnt1++;
            }
        }
        trainDataSize_MOP_Predict_FRNN = tmp_cnt1;
        testDataSize_MOP_Predict_FRNN = tmp_cnt2;
        for(int i = 0; i < tmp_cnt2; i++) tmp_ind1[trainDataSize_MOP_Predict_FRNN + i] = tmp_ind2[i];
        for(int i = 0; i < allDataSize_MOP_Predict_FRNN; i++) tmp_ind2[i] = i;
        for(int i = 0; i < allDataSize_MOP_Predict_FRNN; i++) {
            if(tmp_ind2[i] != tmp_ind1[i]) {
                int the_ind = -1;
                for(int j = i + 1; j < allDataSize_MOP_Predict_FRNN; j++) {
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
                for(int n = 0; n <= numAttr; n++) {
                    double tmp_v = allData_MOP_Predict_FRNN[n][i];
                    allData_MOP_Predict_FRNN[n][i] = allData_MOP_Predict_FRNN[n][the_ind];
                    allData_MOP_Predict_FRNN[n][the_ind] = tmp_v;
                }
            }
        }
    }
    //
    free(buf);
    fclose(fpt);
}

static void normalizeData_MOP_Predict_FRNN()
{
    for(int i = 0; i < numAttr; i++) {
        trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = allData_MOP_Predict_FRNN[i][0];
        trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = allData_MOP_Predict_FRNN[i][0];
        trainStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] = 0;
        trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] = 0;
        for(int j = 0; j < trainDataSize_MOP_Predict_FRNN; j++) {
            double tmp_dt = allData_MOP_Predict_FRNN[i][j];
            if(trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] > tmp_dt)
                trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = tmp_dt;
            if(trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] < tmp_dt)
                trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = tmp_dt;
            trainStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] += tmp_dt;
        }
        trainStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] /= trainDataSize_MOP_Predict_FRNN;
        for(int j = 0; j < trainDataSize_MOP_Predict_FRNN; j++) {
            double tmp_dt = allData_MOP_Predict_FRNN[i][j];
            double tmp_mn = trainStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN];
            trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] += (tmp_dt - tmp_mn) * (tmp_dt - tmp_mn);
        }
        trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] /= trainDataSize_MOP_Predict_FRNN;
        trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] = sqrt(trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN]);
        //
        testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = allData_MOP_Predict_FRNN[i][trainDataSize_MOP_Predict_FRNN];
        testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = allData_MOP_Predict_FRNN[i][trainDataSize_MOP_Predict_FRNN];
        testStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] = 0;
        testStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] = 0;
        for(int j = trainDataSize_MOP_Predict_FRNN; j < allDataSize_MOP_Predict_FRNN; j++) {
            double tmp_dt = allData_MOP_Predict_FRNN[i][j];
            if(testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] > tmp_dt)
                testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = tmp_dt;
            if(testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] < tmp_dt)
                testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = tmp_dt;
            testStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] += tmp_dt;
        }
        testStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN] /= testDataSize_MOP_Predict_FRNN;
        for(int j = trainDataSize_MOP_Predict_FRNN; j < allDataSize_MOP_Predict_FRNN; j++) {
            double tmp_dt = allData_MOP_Predict_FRNN[i][j];
            double tmp_mn = testStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN];
            testStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] += (tmp_dt - tmp_mn) * (tmp_dt - tmp_mn);
        }
        testStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] /= testDataSize_MOP_Predict_FRNN;
        testStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] = sqrt(testStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN]);
        //////////////////////////////////////////////////////////////////////////
        if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_Z_SCORE_MOP_Predict_FRNN) {

            if(trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN]) {
                for(int j = 0; j < allDataSize_MOP_Predict_FRNN; j++) {
                    allData_MOP_Predict_FRNN[i][j] -= trainStat_MOP_Predict_FRNN[i][DATA_MEAN_MOP_Predict_FRNN];
                    allData_MOP_Predict_FRNN[i][j] /= trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN];
                }
            } else {
                for(int j = 0; j < allDataSize_MOP_Predict_FRNN; j++) {
                    allData_MOP_Predict_FRNN[i][j] = 0;
                }
            }
            trainFact_MOP_Predict_FRNN[i] = trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN] *
                                            trainStat_MOP_Predict_FRNN[i][DATA_STD_MOP_Predict_FRNN];
        } else if(NORMALIZE_MOP_Predict_FRNN == NORMALIZE_MIN_MAX_MOP_Predict_FRNN) {
            if(trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] >
               trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN]) {
                for(int j = 0; j < allDataSize_MOP_Predict_FRNN; j++) {
                    allData_MOP_Predict_FRNN[i][j] -= trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN];
                    allData_MOP_Predict_FRNN[i][j] /= (trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] -
                                                       trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN]);
                }
            } else {
                for(int j = 0; j < allDataSize_MOP_Predict_FRNN; j++) {
                    allData_MOP_Predict_FRNN[i][j] = 0;
                }
            }
            trainFact_MOP_Predict_FRNN[i] = (trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] -
                                             trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN]) *
                                            (trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] -
                                             trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN]);
        }
        //////////////////////////////////////////////////////////////////////////
        //trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = trainData_MOP_Predict_FRNN[i][0];
        //trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = trainData_MOP_Predict_FRNN[i][0];
        //for(int j = 0; j < trainDataSize_MOP_Predict_FRNN; j++) {
        //    double tmp_dt = trainData_MOP_Predict_FRNN[i][j];
        //    if(trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] > tmp_dt)
        //        trainStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = tmp_dt;
        //    if(trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] < tmp_dt)
        //        trainStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = tmp_dt;
        //}
        ////
        //testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = testData_MOP_Predict_FRNN[i][0];
        //testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = testData_MOP_Predict_FRNN[i][0];
        //for(int j = 0; j < testDataSize_MOP_Predict_FRNN; j++) {
        //    double tmp_dt = testData_MOP_Predict_FRNN[i][j];
        //    if(testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] > tmp_dt)
        //        testStat_MOP_Predict_FRNN[i][DATA_MIN_MOP_Predict_FRNN] = tmp_dt;
        //    if(testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] < tmp_dt)
        //        testStat_MOP_Predict_FRNN[i][DATA_MAX_MOP_Predict_FRNN] = tmp_dt;
        //}
    }
    //
    return;
}

static void get_Evaluation_Indicators_MOP_Predict_FRNN(int num_class, MY_FLT_TYPE* N_TP, MY_FLT_TYPE* N_FP, MY_FLT_TYPE* N_TN,
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
#if STATUS_OUT_INDEICES_MOP_PREDICT_FRNN == FLAG_ON_MOP_Predict_FRNN
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

#if CURRENT_PROB_MOP_PREDICT_FRNN == STOCK_TRADING_MOP_PREDICT_FRNN
static void genTradingLabel_MOP_Predict_FRNN()
{
    for(int i = 0; i < win_size_cases_MOP_Predict_FRNN; i++) {
        for(int j = 0; j < MAX_DATA_LEN_MOP_PREDICT_FRNN; j++) {
            train_trading_label_MOP_Predict_FRNN[i][j] = CLASS_IND_HOLD_MOP_PREDICT_FRNN;
            test_trading_label_MOP_Predict_FRNN[i][j] = CLASS_IND_HOLD_MOP_PREDICT_FRNN;
        }
    }
    for(int i = 0; i < win_size_cases_MOP_Predict_FRNN; i++) {
        int win_size = i * 2 + win_size_min_MOP_Predict_FRNN;
        // train data
        for(int j = 0; j < trainDataSize_MOP_Predict_FRNN - win_size + 1; j++) {
            int ind_start = j;
            int ind_final = j + win_size - 1;
            int ind_middl = j + win_size / 2;
            double val_mid = allData_MOP_Predict_FRNN[0][ind_middl];
            int ind_min = j;
            int ind_max = j;
            double val_min = allData_MOP_Predict_FRNN[0][j];
            double val_max = allData_MOP_Predict_FRNN[0][j];
            for(int k = ind_start + 1; k <= ind_final; k++) {
                if(allData_MOP_Predict_FRNN[0][k] < val_min) {
                    val_min = allData_MOP_Predict_FRNN[0][k];
                    ind_min = k;
                }
                if(allData_MOP_Predict_FRNN[0][k] > val_max) {
                    val_max = allData_MOP_Predict_FRNN[0][k];
                    ind_max = k;
                }
            }
            if(ind_middl == ind_min || val_min == val_mid)
                train_trading_label_MOP_Predict_FRNN[i][ind_middl] = CLASS_IND_BUY_MOP_PREDICT_FRNN;
            if(ind_middl == ind_max || val_max == val_mid)
                train_trading_label_MOP_Predict_FRNN[i][ind_middl] = CLASS_IND_SELL_MOP_PREDICT_FRNN;
        }
        // test data
        for(int j = trainDataSize_MOP_Predict_FRNN - win_size; j < allDataSize_MOP_Predict_FRNN - win_size + 1; j++) {
            int ind_start = j;
            int ind_final = j + win_size - 1;
            int ind_middl = j + win_size / 2;
            double val_mid = allData_MOP_Predict_FRNN[0][ind_middl];
            int ind_min = j;
            int ind_max = j;
            double val_min = allData_MOP_Predict_FRNN[0][j];
            double val_max = allData_MOP_Predict_FRNN[0][j];
            for(int k = ind_start + 1; k <= ind_final; k++) {
                if(allData_MOP_Predict_FRNN[0][k] < val_min) {
                    val_min = allData_MOP_Predict_FRNN[0][k];
                    ind_min = k;
                }
                if(allData_MOP_Predict_FRNN[0][k] > val_max) {
                    val_max = allData_MOP_Predict_FRNN[0][k];
                    ind_max = k;
                }
            }
            if(ind_middl == ind_min || val_min == val_mid)
                test_trading_label_MOP_Predict_FRNN[i][ind_middl] = CLASS_IND_BUY_MOP_PREDICT_FRNN;
            if(ind_middl == ind_max || val_max == val_mid)
                test_trading_label_MOP_Predict_FRNN[i][ind_middl] = CLASS_IND_SELL_MOP_PREDICT_FRNN;
        }
    }
}
#endif

//////////////////////////////////////////////////////////////////////////
#define IM1_Predict_FRNN 2147483563
#define IM2_Predict_FRNN 2147483399
#define AM_Predict_FRNN (1.0/IM1_Predict_FRNN)
#define IMM1_Predict_FRNN (IM1_Predict_FRNN-1)
#define IA1_Predict_FRNN 40014
#define IA2_Predict_FRNN 40692
#define IQ1_Predict_FRNN 53668
#define IQ2_Predict_FRNN 52774
#define IR1_Predict_FRNN 12211
#define IR2_Predict_FRNN 3791
#define NTAB_Predict_FRNN 32
#define NDIV_Predict_FRNN (1+IMM1_Predict_FRNN/NTAB_Predict_FRNN)
#define EPS_Predict_FRNN 1.2e-7
#define RNMX_Predict_FRNN (1.0-EPS_Predict_FRNN)

//the random generator in [0,1)
static double rnd_uni_Predict_FRNN(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_Predict_FRNN];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_Predict_FRNN + 7; j >= 0; j--) {
            k = (*idum) / IQ1_Predict_FRNN;
            *idum = IA1_Predict_FRNN * (*idum - k * IQ1_Predict_FRNN) - k * IR1_Predict_FRNN;
            if(*idum < 0) *idum += IM1_Predict_FRNN;
            if(j < NTAB_Predict_FRNN) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_Predict_FRNN;
    *idum = IA1_Predict_FRNN * (*idum - k * IQ1_Predict_FRNN) - k * IR1_Predict_FRNN;
    if(*idum < 0) *idum += IM1_Predict_FRNN;
    k = idum2 / IQ2_Predict_FRNN;
    idum2 = IA2_Predict_FRNN * (idum2 - k * IQ2_Predict_FRNN) - k * IR2_Predict_FRNN;
    if(idum2 < 0) idum2 += IM2_Predict_FRNN;
    j = iy / NDIV_Predict_FRNN;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_Predict_FRNN;       //printf("%lf\n", AM_Predict_FRNN*iy);
    if((temp = AM_Predict_FRNN * iy) > RNMX_Predict_FRNN) return RNMX_Predict_FRNN;
    else return temp;
}/*------End of rnd_uni_Classify_CNN()--------------------------*/

static int rnd_Predict_FRNN(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_Predict_FRNN(&rnd_uni_init_Predict_FRNN) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}

/* Fisher¨CYates shuffle algorithm */
static void shuffle_Predict_FRNN(int* x, int size)
{
    int i, aux, k = 0;
    for(i = size - 1; i > 0; i--) {
        /* get a value between cero and i  */
        k = rnd_Predict_FRNN(0, i);
        /* exchange of values */
        aux = x[i];
        x[i] = x[k];
        x[k] = aux;
    }
    //
    return;
}

static void trimLine_MOP_Predict_FRNN(char line[])
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

static int get_setting_MOP_Predict_FRNN(char* wholestr, const char* candidstr, int& val)
{
    //
    char tmp_str[MAX_STR_LEN_MOP_Predict_FRNN];
    sprintf(tmp_str, "%s", wholestr);
    char tmp_delim[] = "_";
    char* p;
    int elem_int;
    int flag_found = 0;
    for(p = strtok(tmp_str, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
        if(flag_found == 1) {
            if(sscanf(p, "%d", &elem_int) != 1) {
                printf("\n%s(%d): setting value not found error...\n", __FILE__, __LINE__);
                exit(65871001);
            }
            val = elem_int;
            flag_found = 2;
            break;
        }
        if(!strcmp(p, candidstr)) flag_found = 1;
    }
    //
    return (flag_found == 2);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
