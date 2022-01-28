#include "MOP_ActivityDetection.h"
#include <float.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
//#define COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
#endif

//////////////////////////////////////////////////////////////////////////
#define FUZZY_RULE_OBJ_OFF_MOP_ACTIVITYDETECTION 0
#define FUZZY_RULE_OBJ_ON_MOP_ACTIVITYDETECTION  1
#define FUZZY_RULE_OBJ_STATUS_MOP_ACTIVITYDETECTION  FUZZY_RULE_OBJ_OFF_MOP_ACTIVITYDETECTION

#define NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION 0
#define NETWORK_SIMPLICITY_CONNECTION_MOP_ACTIVITYDETECTION 1
//#define NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION
#define NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION NETWORK_SIMPLICITY_CONNECTION_MOP_ACTIVITYDETECTION

#define NO_INDICATORS_MOP_ACTIVITYDETECTION 0
#define MEAN_INDICATORS_MOP_ACTIVITYDETECTION 1
#define MIN_INDICATORS_MOP_ACTIVITYDETECTION 2
#define OBJ_INDICATORS_MOP_ACTIVITYDETECTION MEAN_INDICATORS_MOP_ACTIVITYDETECTION

//////////////////////////////////////////////////////////////////////////
#define NUM_label_MHEALTH 12
#define IND_label_MHEALTH 23

//////////////////////////////////////////////////////////////////////////
int NDIM_ActivityDetection_FRNN_Classify = 0;
int NOBJ_ActivityDetection_FRNN_Classify = 0;

#define nSubject_MHEALTH 10
MY_FLT_TYPE*** data_MHEALTH_all = NULL;
int nrows_MHEALTH[nSubject_MHEALTH] = {
    35174,
    35532,
    35380,
    35328,
    33947,
    32205,
    34253,
    33332,
    34354,
    33690
};

#define MAX_NUM_SAMPLES_ONE_CLASS 3430

#define FLAG_OFF_MOP_ACTIVITYDETECTION 0
#define FLAG_ON_MOP_ACTIVITYDETECTION  1
#define STATUS_OUTPUT_MOP_ACTIVITYDETECTION FLAG_OFF_MOP_ACTIVITYDETECTION

#define STATUS_OUT_INDEICES_MOP_ACTIVITYDETECTION FLAG_OFF_MOP_ACTIVITYDETECTION

#define STATUS_OUT_ASSOCIATION_MEM_MOP_ACTIVITYDETECTION FLAG_ON_MOP_ACTIVITYDETECTION

//////////////////////////////////////////////////////////////////////////
MY_FLT_TYPE max_feature_MHEALTH_all[nSubject_MHEALTH][nCol_MHEALTH];
MY_FLT_TYPE min_feature_MHEALTH_all[nSubject_MHEALTH][nCol_MHEALTH];
MY_FLT_TYPE max_feature_MHEALTH_train[nCol_MHEALTH];
MY_FLT_TYPE min_feature_MHEALTH_train[nCol_MHEALTH];
MY_FLT_TYPE max_feature_MHEALTH_test[nCol_MHEALTH];
MY_FLT_TYPE min_feature_MHEALTH_test[nCol_MHEALTH];
int   num_input_MHEALTH = nCol_MHEALTH;
int   num_output_MHEALTH = NUM_label_MHEALTH;

int ind_subject_MHEALTH_train[nSubject_MHEALTH];
int ind_subject_MHEALTH_test[nSubject_MHEALTH];
int num_subject_MHEALTH_train = 0;
int num_subject_MHEALTH_test = 0;
#ifdef COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
int ind_subject_MHEALTH_validation[nSubject_MHEALTH];
int num_subject_MHEALTH_validation = 0;
#endif
int ind_all_types_MHEALTH[nSubject_MHEALTH][NUM_label_MHEALTH][MAX_NUM_SAMPLES_ONE_CLASS];
int num_all_types_MHEALTH[nSubject_MHEALTH][NUM_label_MHEALTH];

#define VIOLATION_PENALTY_ACT_C 1e6

int num_samples_MHEALTH_selected_train = 100;// MAX_NUM_SAMPLES_ONE_CLASS;
int num_samples_MHEALTH_selected_test = MAX_NUM_SAMPLES_ONE_CLASS;

//////////////////////////////////////////////////////////////////////////
#define MAX_BUF_SIZE 10000 //
#define MAX_STR_LEN  256 //

static void get_obj_constraints_evoFRNN_Activity(double* fitness,
        int count_no_act, MY_FLT_TYPE tmp_all_err, MY_FLT_TYPE fire_lv_rules, MY_FLT_TYPE sum_weights);
static int** allocINT(int nrow, int ncol);
static MY_FLT_TYPE** allocFLOAT(int nrow, int ncol);
static void readData_MHEALTH(MY_FLT_TYPE** pDATA, char fname[], int nrow, int ncol, MY_FLT_TYPE max_val[],
                             MY_FLT_TYPE min_val[]);

//////////////////////////////////////////////////////////////////////////
FRNN_ACT_C* frnn_act_c = NULL;
static void ff_ActivityDetection_FRNN_Classify(double* individual, int num_subject, int* ind_subject, int num_selected,
        MY_FLT_TYPE*** data_MHEALTH_cur, int& count_no_act, MY_FLT_TYPE& tmp_all_err, MY_FLT_TYPE& fire_lv_rules,
        MY_FLT_TYPE& sum_weights,
        int tag_train_test);

//////////////////////////////////////////////////////////////////////////
int curSubject_MHEALTH;
int numSubject_MHEALTH;
#ifdef COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
int curSubjectValidation_MHEALTH;
#endif
//////////////////////////////////////////////////////////////////////////
void Initialize_ActivityDetection_FRNN_Classify(int curN, int numN, int my_rank)
{
    //
    curSubject_MHEALTH = curN;
    numSubject_MHEALTH = numN;
#ifdef COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
    curSubjectValidation_MHEALTH = (curSubject_MHEALTH - 1 + numSubject_MHEALTH) % numSubject_MHEALTH;
#endif
    if(numSubject_MHEALTH != nSubject_MHEALTH) {
        printf("%s(%d): The number of runs is not consistant with the test instance requirement (%d), exiting...\n",
               __FILE__, __LINE__, nSubject_MHEALTH);
        exit(-1);
    }
    //
    seed_FRNN_MODEL = 237 + my_rank;
    seed_FRNN_MODEL = seed_FRNN_MODEL % 1235;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    // initialize
    num_input_MHEALTH = 0;
    num_output_MHEALTH = NUM_label_MHEALTH;
    // pre-process
    data_MHEALTH_all = (MY_FLT_TYPE***)malloc(nSubject_MHEALTH * sizeof(MY_FLT_TYPE**));
    for(int n = 0; n < nSubject_MHEALTH; n++) {
        char fname[MAX_STR_LEN];
        for(int i = 0; i < nCol_MHEALTH; i++) {
            max_feature_MHEALTH_all[n][i] = -FLT_MAX;
            min_feature_MHEALTH_all[n][i] = FLT_MAX;
        }
        sprintf(fname, "../Data_all/Data_MHEALTH/mHealth_subject%d.dat", n + 1);
        data_MHEALTH_all[n] = allocFLOAT(nrows_MHEALTH[n], nCol_MHEALTH);
        readData_MHEALTH(data_MHEALTH_all[n], fname, nrows_MHEALTH[n], nCol_MHEALTH,
                         max_feature_MHEALTH_all[n], min_feature_MHEALTH_all[n]);
    }
    //
    num_subject_MHEALTH_train = 0;
    num_subject_MHEALTH_test = 0;
    for(int n = 0; n < nSubject_MHEALTH; n++) {
#ifdef COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
        if(n == curSubject_MHEALTH) {
            ind_subject_MHEALTH_test[num_subject_MHEALTH_test++] = n;
        } else if(n == curSubjectValidation_MHEALTH) {
            ind_subject_MHEALTH_validation[num_subject_MHEALTH_validation++] = n;
        } else {
            ind_subject_MHEALTH_train[num_subject_MHEALTH_train++] = n;
        }
#else
        if(n == curSubject_MHEALTH) {
            ind_subject_MHEALTH_test[num_subject_MHEALTH_test++] = n;
        } else {
            ind_subject_MHEALTH_train[num_subject_MHEALTH_train++] = n;
        }
#endif
    }
    //
    for(int n = 0; n < nSubject_MHEALTH; n++) {
        for(int i = 0; i < NUM_label_MHEALTH; i++) {
            num_all_types_MHEALTH[n][i] = 0;
        }
        for(int i = 0; i < nrows_MHEALTH[n]; i++) {
            int cur_label = (int)data_MHEALTH_all[n][i][IND_label_MHEALTH];
            ind_all_types_MHEALTH[n][cur_label][num_all_types_MHEALTH[n][cur_label]] = i;
            num_all_types_MHEALTH[n][cur_label]++;
        }
    }
    //
    for(int i = 0; i < nCol_MHEALTH; i++) {
        max_feature_MHEALTH_train[i] = -FLT_MAX;
        min_feature_MHEALTH_train[i] = FLT_MAX;
        max_feature_MHEALTH_test[i] = -FLT_MAX;
        min_feature_MHEALTH_test[i] = FLT_MAX;
    }
    for(int n = 0; n < num_subject_MHEALTH_train; n++) {
        int cur_ind_subject = ind_subject_MHEALTH_train[n];
        for(int i = 0; i < nCol_MHEALTH; i++) {
            if(max_feature_MHEALTH_train[i] < max_feature_MHEALTH_all[cur_ind_subject][i])
                max_feature_MHEALTH_train[i] = max_feature_MHEALTH_all[cur_ind_subject][i];
            if(min_feature_MHEALTH_train[i] > min_feature_MHEALTH_all[cur_ind_subject][i])
                min_feature_MHEALTH_train[i] = min_feature_MHEALTH_all[cur_ind_subject][i];
        }
    }
    for(int n = 0; n < num_subject_MHEALTH_test; n++) {
        int cur_ind_subject = ind_subject_MHEALTH_test[n];
        for(int i = 0; i < nCol_MHEALTH; i++) {
            if(max_feature_MHEALTH_test[i] < max_feature_MHEALTH_all[cur_ind_subject][i])
                max_feature_MHEALTH_test[i] = max_feature_MHEALTH_all[cur_ind_subject][i];
            if(min_feature_MHEALTH_test[i] > min_feature_MHEALTH_all[cur_ind_subject][i])
                min_feature_MHEALTH_test[i] = min_feature_MHEALTH_all[cur_ind_subject][i];
        }
    }
    //
    num_input_MHEALTH = nCol_MHEALTH - 1;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    frnn_act_c = (FRNN_ACT_C*)calloc(1, sizeof(FRNN_ACT_C));
    int numInput = num_input_MHEALTH;
    MY_FLT_TYPE inputMin[nCol_MHEALTH];
    MY_FLT_TYPE inputMax[nCol_MHEALTH];
    for(int i = 0; i < nCol_MHEALTH; i++) {
        inputMin[i] = min_feature_MHEALTH_train[i];
        inputMax[i] = max_feature_MHEALTH_train[i];
    }
    int numMemship[nCol_MHEALTH];
    for(int i = 0; i < nCol_MHEALTH; i++) {
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL; // DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
    }
    int flagAdapMemship[nCol_MHEALTH];
    for(int i = 0; i < nCol_MHEALTH; i++) {
        flagAdapMemship[i] = 1;
    }
    int numOutput = num_output_MHEALTH;
    MY_FLT_TYPE outputMin[NUM_label_MHEALTH];
    MY_FLT_TYPE outputMax[NUM_label_MHEALTH];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE;
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int numFuzzyRules = 120;// 120;// DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = 60;// 60;
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int numInputConsequenceNode = numInput;
    MY_FLT_TYPE inputMin_cnsq[nCol_MHEALTH];
    MY_FLT_TYPE inputMax_cnsq[nCol_MHEALTH];
    for(int i = 0; i < nCol_MHEALTH; i++) {
        inputMin_cnsq[i] = 0;
        inputMax_cnsq[i] = 1;
    }
    int flagConnectStatusAdap = FLAG_STATUS_OFF;
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int flagConnectWeightAdap = FLAG_STATUS_ON;
#else
    int flagConnectWeightAdap = FLAG_STATUS_ON;
#endif
    frnn_act_c_setup(frnn_act_c, numInput, inputMin, inputMax, numMemship, flagAdapMemship,
                     numOutput, outputMin, outputMax, typeFuzzySet, typeRules,
                     typeInRuleCorNum, typeTypeReducer, numFuzzyRules, numRoughSets,
                     consequenceNodeStatus, centroid_num_tag,
                     numInputConsequenceNode, inputMin_cnsq, inputMax_cnsq,
                     flagConnectStatusAdap, flagConnectWeightAdap);
    //
    NDIM_ActivityDetection_FRNN_Classify = frnn_act_c->M1->numParaLocal +
                                           frnn_act_c->F2->numParaLocal +
                                           frnn_act_c->R3->numParaLocal +
                                           frnn_act_c->OL->numParaLocal;
#if OBJ_INDICATORS_MOP_ACTIVITYDETECTION == MEAN_INDICATORS_MOP_ACTIVITYDETECTION
    NOBJ_ActivityDetection_FRNN_Classify = 3;
#elif OBJ_INDICATORS_MOP_ACTIVITYDETECTION == MIN_INDICATORS_MOP_ACTIVITYDETECTION
    NOBJ_ActivityDetection_FRNN_Classify = 3;
#else
#if FUZZY_RULE_OBJ_STATUS_MOP_ACTIVITYDETECTION == FUZZY_RULE_OBJ_ON_MOP_ACTIVITYDETECTION
    NOBJ_ActivityDetection_FRNN_Classify = 3;
#else
    NOBJ_ActivityDetection_FRNN_Classify = 2;
#endif
#endif
    //
    return;
}

void SetLimits_ActivityDetection_FRNN_Classify(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    for(int i = 0; i < frnn_act_c->M1->numParaLocal; i++) {
        minLimit[count] = frnn_act_c->M1->xMin[i];
        maxLimit[count] = frnn_act_c->M1->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_act_c->F2->numParaLocal; i++) {
        minLimit[count] = frnn_act_c->F2->xMin[i];
        maxLimit[count] = frnn_act_c->F2->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_act_c->R3->numParaLocal; i++) {
        minLimit[count] = frnn_act_c->R3->xMin[i];
        maxLimit[count] = frnn_act_c->R3->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_act_c->OL->numParaLocal; i++) {
        minLimit[count] = frnn_act_c->OL->xMin[i];
        maxLimit[count] = frnn_act_c->OL->xMax[i];
        count++;
    }
    return;
}

int CheckLimits_ActivityDetection_FRNN_Classify(double* x, int nx)
{
    int count = 0;
    for(int i = 0; i < frnn_act_c->M1->numParaLocal; i++) {
        if(x[count] < frnn_act_c->M1->xMin[i] ||
           x[count] > frnn_act_c->M1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->M1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_act_c->M1->xMin[i], frnn_act_c->M1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_act_c->F2->numParaLocal; i++) {
        if(x[count] < frnn_act_c->F2->xMin[i] ||
           x[count] > frnn_act_c->F2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->F2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_act_c->F2->xMin[i], frnn_act_c->F2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_act_c->R3->numParaLocal; i++) {
        if(x[count] < frnn_act_c->R3->xMin[i] ||
           x[count] > frnn_act_c->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->R3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_act_c->R3->xMin[i], frnn_act_c->R3->xMax[i]);
            return 0;
        }
        count++;
    }
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    for(int i = 0; i < frnn_act_c->OL->numParaLocal; i++) {
        if(x[count] < frnn_act_c->OL->xMin[i] ||
           x[count] > frnn_act_c->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->O4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_act_c->OL->xMin[i], frnn_act_c->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#else
    if(frnn_act_c->flagConnectStatus != FLAG_STATUS_OFF ||
       frnn_act_c->flagConnectWeight != FLAG_STATUS_ON ||
       frnn_act_c->typeCoding != PARA_CODING_DIRECT) {
        printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
               __FILE__, __LINE__, frnn_act_c->flagConnectStatus, frnn_act_c->flagConnectWeight, frnn_act_c->typeCoding);
        exit(-275082);
    }
    int tmp_offset = frnn_act_c->OL->numOutput * frnn_act_c->OL->numInput;
    count += tmp_offset;
    for(int i = tmp_offset; i < frnn_act_c->OL->numParaLocal; i++) {
        if(x[count] < frnn_act_c->OL->xMin[i] ||
           x[count] > frnn_act_c->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->O4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_act_c->OL->xMin[i], frnn_act_c->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
    return 1;
}

void Fitness_ActivityDetection_FRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    int count_no_act;
    MY_FLT_TYPE tmp_all_err;
    MY_FLT_TYPE fire_lv_rules;
    MY_FLT_TYPE sum_weights;
    ff_ActivityDetection_FRNN_Classify(individual, num_subject_MHEALTH_train, ind_subject_MHEALTH_train,
                                       num_samples_MHEALTH_selected_train,
                                       data_MHEALTH_all, count_no_act, tmp_all_err, fire_lv_rules, sum_weights,
                                       TRAIN_TAG_MOP_ACTIVITY);
    //
    get_obj_constraints_evoFRNN_Activity(fitness, count_no_act, tmp_all_err, fire_lv_rules, sum_weights);
    //
    return;
}

void Fitness_ActivityDetection_FRNN_Classify_test(double* individual, double* fitness)
{
    int count_no_act;
    MY_FLT_TYPE tmp_all_err;
    MY_FLT_TYPE fire_lv_rules;
    MY_FLT_TYPE sum_weights;
    ff_ActivityDetection_FRNN_Classify(individual, num_subject_MHEALTH_test, ind_subject_MHEALTH_test,
                                       num_samples_MHEALTH_selected_test,
                                       data_MHEALTH_all, count_no_act, tmp_all_err, fire_lv_rules, sum_weights,
                                       TEST_TAG_MOP_ACTIVITY);
    //
    get_obj_constraints_evoFRNN_Activity(fitness, count_no_act, tmp_all_err, fire_lv_rules, sum_weights);
    //
    return;
}

void ff_ActivityDetection_FRNN_Classify(double* individual, int num_subject, int* ind_subject, int num_selected,
                                        MY_FLT_TYPE*** data_MHEALTH_cur,
                                        int& count_no_act, MY_FLT_TYPE& tmp_all_err, MY_FLT_TYPE& fire_lv_rules, MY_FLT_TYPE& sum_weights,
                                        int tag_train_test)
{
    count_no_act = 0;
    tmp_all_err = 0;
    fire_lv_rules = 0;
    sum_weights = 0;
    frnn_act_c->sum_all = 0;
    frnn_act_c->sum_wrong = 0;
    for(int i = 0; i < frnn_act_c->numOutput; i++) {
        frnn_act_c->N_sum[i] = 0;
        frnn_act_c->N_wrong[i] = 0;
        frnn_act_c->e_sum[i] = 0;
        frnn_act_c->N_TP[i] = 0;
        frnn_act_c->N_TN[i] = 0;
        frnn_act_c->N_FP[i] = 0;
        frnn_act_c->N_FN[i] = 0;
    }
    frnn_act_c_init(frnn_act_c, individual, ASSIGN_MODE_FRNN);
#if STATUS_OUT_ASSOCIATION_MEM_MOP_ACTIVITYDETECTION == FLAG_ON_MOP_ACTIVITYDETECTION

#endif
    MY_FLT_TYPE valIn[nCol_MHEALTH];
    MY_FLT_TYPE valOut[NUM_label_MHEALTH];
    MY_FLT_TYPE paraIn[NUM_label_MHEALTH][nCol_MHEALTH];
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    MY_FLT_TYPE* matA = NULL;
    MY_FLT_TYPE* matB = NULL;
    MY_FLT_TYPE* matLeft = NULL;
    MY_FLT_TYPE* matRight = NULL;
    int tmp_offset_samp = 0;
    int tmp_offset_samp_validation = 0;
    if(tag_train_test == TRAIN_TAG_MOP_ACTIVITY) {
        matA = (MY_FLT_TYPE*)calloc(num_subject * NUM_label_MHEALTH * MAX_NUM_SAMPLES_ONE_CLASS * frnn_act_c->OL->numInput,
                                    sizeof(MY_FLT_TYPE));
        matB = (MY_FLT_TYPE*)calloc(num_subject * NUM_label_MHEALTH * MAX_NUM_SAMPLES_ONE_CLASS * frnn_act_c->OL->numOutput,
                                    sizeof(MY_FLT_TYPE));
        matLeft = (MY_FLT_TYPE*)calloc(frnn_act_c->OL->numInput * frnn_act_c->OL->numInput, sizeof(MY_FLT_TYPE));
        matRight = (MY_FLT_TYPE*)calloc(frnn_act_c->OL->numInput * frnn_act_c->OL->numOutput, sizeof(MY_FLT_TYPE));
    }
#endif
    //for(int i = 0; i < nRow_KDD99_test; i++) {
    for(int a = 0; a < num_subject; a++) {
        int m = ind_subject[a];
        for(int b = 0; b < NUM_label_MHEALTH; b++) {
            for(int n = 0; n < num_all_types_MHEALTH[m][b] && n < num_selected; n++) {
                //int tmp_ind = (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (num_all_types_MHEALTH[m][b] - n - 1e-6));
                int tmp_ind = (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (num_all_types_MHEALTH[m][b] - n - 1e-6));
                int i = ind_all_types_MHEALTH[m][b][tmp_ind];
                ind_all_types_MHEALTH[m][b][tmp_ind] = ind_all_types_MHEALTH[m][b][num_all_types_MHEALTH[m][b] - n - 1];
                ind_all_types_MHEALTH[m][b][num_all_types_MHEALTH[m][b] - n - 1] = i;
                for(int j = 0; j < frnn_act_c->numInput; j++) {
                    valIn[j] = data_MHEALTH_cur[m][i][j];
                }
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    for(int k = 0; k < frnn_act_c->numInput; k++) {
                        //paraIn[j][k] = valIn[k];
                        paraIn[j][k] = (valIn[k] - frnn_act_c->M1->inputMin[k]) /
                                       (frnn_act_c->M1->inputMax[k] - frnn_act_c->M1->inputMin[k]);
                        //if(/*paraIn[j][tmp_c] < 0 || paraIn[j][tmp_c] > 1*/1) {
                        //    printf("%d %f (%f %f %f)\n", k, paraIn[j][tmp_c],
                        //           frnn_id_c->M1->inputMin[k], valIn[k], frnn_id_c->M1->inputMax[k]);
                        //}
                    }
                }
                ff_frnn_act_c(frnn_act_c, valIn, valOut, paraIn);
                /*float tmpSum = 0;
                for(int j = 0; j < NUM_label_KDD99; j++) tmpSum += valOut[j];
                if(tmpSum > 0) {
                	printf("No. sample - %d - ", i);
                	for(int j = 0; j < NUM_label_KDD99; j++) printf("%e ", valOut[j]);
                	printf("\n");
                }*/
                int cur_label = 0;
                MY_FLT_TYPE cur_out = valOut[0];
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    if(cur_out < valOut[j]) {
                        cur_out = valOut[j];
                        cur_label = j;
                    }
                }
                //for(int j = 0; j < frnn_act_c->M1->numInput; j++) {
                //    if(frnn_act_c->M1->flag_adapMembershipFun[j] == FLAG_STATUS_OFF) continue;
                //    for(int k = 0; k < frnn_act_c->M1->numMembershipFun[j]; k++) {
                //        if((frnn_act_c->typeFuzzySet == FUZZY_SET_I && frnn_act_c->M1->degreeMembership[j][k][0] == 0) ||
                //           (frnn_act_c->typeFuzzySet == FUZZY_INTERVAL_TYPE_II && frnn_act_c->M1->degreeMembership[j][k][1] == 0))
                //            count_no_act++;
                //    }
                //}
                int true_label = (int)data_MHEALTH_cur[m][i][IND_label_MHEALTH];
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    if(j == cur_label && j == true_label) frnn_act_c->N_TP[j]++;
                    if(j == cur_label && j != true_label) frnn_act_c->N_FP[j]++;
                    if(j != cur_label && j == true_label) frnn_act_c->N_FN[j]++;
                    if(j != cur_label && j != true_label) frnn_act_c->N_TN[j]++;
                }
                frnn_act_c->sum_all++;
                frnn_act_c->N_sum[true_label]++;
                if(cur_label != true_label) {
                    frnn_act_c->sum_wrong++;
                    frnn_act_c->N_wrong[true_label]++;
                }
                //
                MY_FLT_TYPE softmax_outs[NUM_label_MHEALTH];
                MY_FLT_TYPE softmax_sum = 0;
                MY_FLT_TYPE softmax_degr[NUM_label_MHEALTH];
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j] - cur_out));
                    softmax_sum += softmax_outs[j];
                }
                if(softmax_sum > 0) {
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        softmax_degr[j] = softmax_outs[j] / softmax_sum;
                    }
                }
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    if(true_label == j) {
                        tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
                    } else {
                        tmp_all_err += softmax_degr[j] * softmax_degr[j];
                    }
                }
                //
                MY_FLT_TYPE tmp_fire = 0;
                int tempN;
                if(frnn_act_c->typeFuzzySet == FUZZY_SET_I) tempN = 1;
                else tempN = 2;
                for(int j = 0; j < frnn_act_c->F2->numRules; j++) {
                    for(int k = 0; k < tempN; k++) {
                        tmp_fire += frnn_act_c->F2->degreeRules[j][k];
                    }
                    ////////////////////////////////////////////////////////////////////////// count_no_act
                }
                if(tmp_fire == 0)
                    count_no_act++;
                tmp_fire /= frnn_act_c->F2->numRules * tempN;
                fire_lv_rules += tmp_fire;
                //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
                if(tag_train_test == TRAIN_TAG_MOP_ACTIVITY) {
                    for(int j = 0; j < frnn_act_c->OL->numInput; j++) {
                        int ind_cur = tmp_offset_samp * frnn_act_c->OL->numInput + j;
                        matA[ind_cur] = frnn_act_c->OL->valInputFinal[0][j];
                    }
                    for(int j = 0; j < frnn_act_c->OL->numOutput; j++) {
                        int ind_cur = tmp_offset_samp * frnn_act_c->OL->numOutput + j;
                        if(j == (int)data_MHEALTH_cur[m][i][IND_label_MHEALTH])
                            matB[ind_cur] = 1;
                        else
                            matB[ind_cur] = -1;
                    }
                }
                tmp_offset_samp++;
#endif
            }
        }
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_test == TRAIN_TAG_MOP_ACTIVITY) {
        tmp_all_err = 0;
        frnn_act_c->sum_all = 0;
        frnn_act_c->sum_wrong = 0;
        for(int i = 0; i < frnn_act_c->numOutput; i++) {
            frnn_act_c->N_sum[i] = 0;
            frnn_act_c->N_wrong[i] = 0;
            frnn_act_c->e_sum[i] = 0;
            frnn_act_c->N_TP[i] = 0;
            frnn_act_c->N_TN[i] = 0;
            frnn_act_c->N_FP[i] = 0;
            frnn_act_c->N_FN[i] = 0;
        }
        //
        //printf("tmp_offset_samp = %d\n", tmp_offset_samp);
        MY_FLT_TYPE lambda = 9.3132e-10;
        MY_FLT_TYPE tmp_max = 0;
        int tmp_max_flag = 0;
        for(int i = 0; i < frnn_act_c->OL->numInput; i++) {
            for(int j = 0; j < frnn_act_c->OL->numInput; j++) {
                int tmp_o0 = i * frnn_act_c->OL->numInput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * frnn_act_c->OL->numInput + i;
                    int tmp_i2 = k * frnn_act_c->OL->numInput + j;
                    matLeft[tmp_o0] += matA[tmp_i1] * matA[tmp_i2];
                }
                //if(i == j)
                //    matLeft[tmp_o0] += lambda * fabs(matLeft[tmp_o0]);
                if(i == j) {
                    if(!tmp_max_flag) {
                        tmp_max = matLeft[tmp_o0];
                    } else {
                        if(tmp_max < matLeft[tmp_o0])
                            tmp_max < matLeft[tmp_o0];
                    }
                }
            }
        }
        //printf("tmp_max = %lf\n", tmp_max);
        for(int i = 0; i < frnn_act_c->OL->numInput; i++) {
            int tmp_o0 = i * frnn_act_c->OL->numInput + i;
            matLeft[tmp_o0] += lambda;// *tmp_max;
        }
        for(int i = 0; i < frnn_act_c->OL->numInput; i++) {
            for(int j = 0; j < frnn_act_c->OL->numOutput; j++) {
                int tmp_o0 = i * frnn_act_c->OL->numOutput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * frnn_act_c->OL->numInput + i;
                    int tmp_i2 = k * frnn_act_c->OL->numOutput + j;
                    matRight[tmp_o0] += matA[tmp_i1] * matB[tmp_i2];
                }
            }
        }
        int N = frnn_act_c->OL->numInput;
        int NRHS = frnn_act_c->OL->numOutput;
        int LDA = N;
        int LDB = NRHS;
        int n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
        int* ipiv = (int*)calloc(N, sizeof(int));
        info = LAPACKE_dgesv(matStoreType, n, nrhs, matLeft, lda, ipiv, matRight, ldb);
        if(info > 0) {
            printf("The diagonal element of the triangular factor of A,\n");
            printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
            printf("the solution could not be computed.\n");
            exit(1);
        }
        //
        for(int i = 0; i < frnn_act_c->OL->numInput; i++) {
            for(int j = 0; j < frnn_act_c->OL->numOutput; j++) {
                int ind_cur = i * frnn_act_c->OL->numOutput + j;
                if(CHECK_INVALID(matRight[ind_cur])) {
                    printf("%s(%d): Error - invalid value of matRight[%d] = %lf",
                           __FILE__, __LINE__, ind_cur, matRight[ind_cur]);
                    exit(-112);
                }
                frnn_act_c->OL->connectWeight[j][i] = matRight[ind_cur];
                sum_weights += fabs(matRight[ind_cur]);
            }
        }
        frnn_act_c_init(frnn_act_c, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
        //
#ifdef COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
        MY_FLT_TYPE tmp_fire_lv_rules = 0;
        for(int a = 0; a < num_subject_MHEALTH_validation; a++) {
            int m = ind_subject_MHEALTH_validation[a];
            for(int b = 0; b < NUM_label_MHEALTH; b++) {
                for(int n = 0; n < num_all_types_MHEALTH[m][b] && n < num_selected; n++) {
                    //int tmp_ind = (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (num_all_types_MHEALTH[m][b] - n - 1e-6));
                    int tmp_ind = (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (num_all_types_MHEALTH[m][b] - n - 1e-6));
                    int i = ind_all_types_MHEALTH[m][b][tmp_ind];
                    ind_all_types_MHEALTH[m][b][tmp_ind] = ind_all_types_MHEALTH[m][b][num_all_types_MHEALTH[m][b] - n - 1];
                    ind_all_types_MHEALTH[m][b][num_all_types_MHEALTH[m][b] - n - 1] = i;
                    for(int j = 0; j < frnn_act_c->numInput; j++) {
                        valIn[j] = data_MHEALTH_cur[m][i][j];
                    }
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        for(int k = 0; k < frnn_act_c->numInput; k++) {
                            paraIn[j][k] = (valIn[k] - frnn_act_c->M1->inputMin[k]) /
                                           (frnn_act_c->M1->inputMax[k] - frnn_act_c->M1->inputMin[k]);
                            //if(/*paraIn[j][tmp_c] < 0 || paraIn[j][tmp_c] > 1*/1) {
                            //    printf("%d %f (%f %f %f)\n", k, paraIn[j][tmp_c],
                            //           frnn_id_c->M1->inputMin[k], valIn[k], frnn_id_c->M1->inputMax[k]);
                            //}
                        }
                    }
                    ff_frnn_act_c(frnn_act_c, valIn, valOut, paraIn);
                    /*float tmpSum = 0;
                    for(int j = 0; j < NUM_label_KDD99; j++) tmpSum += valOut[j];
                    if(tmpSum > 0) {
                    	printf("No. sample - %d - ", i);
                    	for(int j = 0; j < NUM_label_KDD99; j++) printf("%e ", valOut[j]);
                    	printf("\n");
                    }*/
                    int cur_label = 0;
                    MY_FLT_TYPE cur_out = valOut[0];
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        if(CHECK_INVALID(valOut[j])) {
                            printf("Error: Invalid output %d~%lf", j, valOut[j]);
                            exit(-1);
                        }
                        if(cur_out < valOut[j]) {
                            cur_out = valOut[j];
                            cur_label = j;
                        }
                    }
                    //for(int j = 0; j < frnn_act_c->M1->numInput; j++) {
                    //    if(frnn_act_c->M1->flag_adapMembershipFun[j] == FLAG_STATUS_OFF) continue;
                    //    for(int k = 0; k < frnn_act_c->M1->numMembershipFun[j]; k++) {
                    //        if((frnn_act_c->typeFuzzySet == FUZZY_SET_I && frnn_act_c->M1->degreeMembership[j][k][0] == 0) ||
                    //           (frnn_act_c->typeFuzzySet == FUZZY_INTERVAL_TYPE_II && frnn_act_c->M1->degreeMembership[j][k][1] == 0))
                    //            count_no_act++;
                    //    }
                    //}
                    int true_label = (int)data_MHEALTH_cur[m][i][IND_label_MHEALTH];
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        if(j == cur_label && j == true_label) frnn_act_c->N_TP[j]++;
                        if(j == cur_label && j != true_label) frnn_act_c->N_FP[j]++;
                        if(j != cur_label && j == true_label) frnn_act_c->N_FN[j]++;
                        if(j != cur_label && j != true_label) frnn_act_c->N_TN[j]++;
                    }
                    frnn_act_c->sum_all++;
                    frnn_act_c->N_sum[true_label]++;
                    if(cur_label != true_label) {
                        frnn_act_c->sum_wrong++;
                        frnn_act_c->N_wrong[true_label]++;
                    }
                    //
                    MY_FLT_TYPE softmax_outs[NUM_label_MHEALTH];
                    MY_FLT_TYPE softmax_sum = 0;
                    MY_FLT_TYPE softmax_degr[NUM_label_MHEALTH];
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j] - cur_out));
                        softmax_sum += softmax_outs[j];
                    }
                    if(softmax_sum > 0) {
                        for(int j = 0; j < frnn_act_c->numOutput; j++) {
                            softmax_degr[j] = softmax_outs[j] / softmax_sum;
                        }
                    }
                    for(int j = 0; j < frnn_act_c->numOutput; j++) {
                        if(true_label == j) {
                            tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
                        } else {
                            tmp_all_err += softmax_degr[j] * softmax_degr[j];
                        }
                    }
                    //
                    MY_FLT_TYPE tmp_fire = 0;
                    int tempN;
                    if(frnn_act_c->typeFuzzySet == FUZZY_SET_I) tempN = 1;
                    else tempN = 2;
                    for(int j = 0; j < frnn_act_c->F2->numRules; j++) {
                        for(int k = 0; k < tempN; k++) {
                            tmp_fire += frnn_act_c->F2->degreeRules[j][k];
                        }
                        ////////////////////////////////////////////////////////////////////////// count_no_act
                    }
                    if(tmp_fire == 0)
                        count_no_act++;
                    tmp_fire /= frnn_act_c->F2->numRules * tempN;
                    tmp_fire_lv_rules += tmp_fire;
                }
            }
            tmp_offset_samp_validation++;
        }
        MY_FLT_TYPE tmp_ratio = (MY_FLT_TYPE)tmp_offset_samp / tmp_offset_samp_validation;
        tmp_all_err *= (int)tmp_ratio;
        tmp_fire_lv_rules *= (int)tmp_ratio;
        fire_lv_rules += tmp_fire_lv_rules;
        frnn_act_c->sum_all *= (int)tmp_ratio;
        frnn_act_c->sum_wrong *= (int)tmp_ratio;
        for(int i = 0; i < frnn_act_c->numOutput; i++) {
            frnn_act_c->N_sum[i] *= (int)tmp_ratio;
            frnn_act_c->N_wrong[i] *= (int)tmp_ratio;
            frnn_act_c->e_sum[i] *= (int)tmp_ratio;
            frnn_act_c->N_TP[i] *= (int)tmp_ratio;
            frnn_act_c->N_TN[i] *= (int)tmp_ratio;
            frnn_act_c->N_FP[i] *= (int)tmp_ratio;
            frnn_act_c->N_FN[i] *= (int)tmp_ratio;
        }
#endif
        for(int m = 0; m < tmp_offset_samp; m++) {
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
            for(int j = 0; j < frnn_act_c->OL->numOutput; j++) {
                valOut[j] = 0;
                for(int k = 0; k < frnn_act_c->OL->numInput; k++) {
                    int ind_cur = m * frnn_act_c->OL->numInput + k;
                    valOut[j] += matA[ind_cur] * frnn_act_c->OL->connectWeight[j][k];
                }
                if(CHECK_INVALID(valOut[j])) {
                    printf("%d~%lf", j, valOut[j]);
                }
            }
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 0; j < frnn_act_c->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE cur_true_out = matB[m * frnn_act_c->OL->numOutput];
            for(int j = 0; j < frnn_act_c->numOutput; j++) {
                if(cur_true_out < matB[m * frnn_act_c->OL->numOutput + j]) {
                    cur_true_out = matB[m * frnn_act_c->OL->numOutput + j];
                    true_label = j;
                }
            }
            for(int j = 0; j < frnn_act_c->numOutput; j++) {
                if(j == cur_label && j == true_label) frnn_act_c->N_TP[j]++;
                if(j == cur_label && j != true_label) frnn_act_c->N_FP[j]++;
                if(j != cur_label && j == true_label) frnn_act_c->N_FN[j]++;
                if(j != cur_label && j != true_label) frnn_act_c->N_TN[j]++;
            }
            frnn_act_c->sum_all++;
            frnn_act_c->N_sum[true_label]++;
            if(cur_label != true_label) {
                frnn_act_c->sum_wrong++;
                frnn_act_c->N_wrong[true_label]++;
            }
            //
            MY_FLT_TYPE softmax_outs[NUM_label_MHEALTH];
            MY_FLT_TYPE softmax_sum = 0;
            MY_FLT_TYPE softmax_degr[NUM_label_MHEALTH];
            for(int j = 0; j < frnn_act_c->numOutput; j++) {
                softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j] - cur_out));
                softmax_sum += softmax_outs[j];
            }
            if(softmax_sum > 0) {
                for(int j = 0; j < frnn_act_c->numOutput; j++) {
                    softmax_degr[j] = softmax_outs[j] / softmax_sum;
                }
            }
            for(int j = 0; j < frnn_act_c->numOutput; j++) {
                if(true_label == j) {
                    tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
                } else {
                    tmp_all_err += softmax_degr[j] * softmax_degr[j];
                }
            }
        }
        //
        free(matA);
        free(matB);
        free(matLeft);
        free(matRight);
        free(ipiv);
    }
#endif
//
    tmp_all_err /= frnn_act_c->sum_all;
    tmp_all_err /= frnn_act_c->numOutput;
    //fire_lv_fules /= frnn_act_c->sum_all;
    fire_lv_rules /= frnn_act_c->sum_all;
    //
    return;
}

void get_obj_constraints_evoFRNN_Activity(double* fitness,
        int count_no_act, MY_FLT_TYPE tmp_all_err, MY_FLT_TYPE fire_lv_rules, MY_FLT_TYPE sum_weights)
{
    //
    MY_FLT_TYPE sum_precision = frnn_act_c->sum_wrong / frnn_act_c->sum_all;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE min_precision = 1;
    MY_FLT_TYPE min_recall = 1;
    MY_FLT_TYPE min_Fvalue = 1;
    MY_FLT_TYPE tmp_precision[NUM_label_MHEALTH];
    MY_FLT_TYPE tmp_recall[NUM_label_MHEALTH];
    MY_FLT_TYPE tmp_Fvalue[NUM_label_MHEALTH];
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < NUM_label_MHEALTH; i++) {
        if(frnn_act_c->N_TP[i] > 0) {
            tmp_precision[i] = frnn_act_c->N_TP[i] / (frnn_act_c->N_TP[i] + frnn_act_c->N_FP[i]);
            tmp_recall[i] = frnn_act_c->N_TP[i] / (frnn_act_c->N_TP[i] + frnn_act_c->N_FN[i]);
            tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
                            (tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
        } else {
            tmp_precision[i] = 0;
            tmp_recall[i] = 0;
            tmp_Fvalue[i] = 0;
        }
        mean_precision += tmp_precision[i];
        mean_recall += tmp_recall[i];
        mean_Fvalue += tmp_Fvalue[i];
        if(min_precision > tmp_precision[i]) min_precision = tmp_precision[i];
        if(min_recall > tmp_recall[i]) min_recall = tmp_recall[i];
        if(min_Fvalue > tmp_Fvalue[i]) min_Fvalue = tmp_Fvalue[i];
#if STATUS_OUT_INDEICES_MOP_ACTIVITYDETECTION == FLAG_ON_MOP_ACTIVITYDETECTION
        printf("%f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i]);
#endif
    }
    ////
    //MY_FLT_TYPE count_violation = 0;
    //int cur_dataflow = 0;
    //for(int i = 0; i < NUM_label_MHEALTH; i++) {
    //    cur_dataflow += frnn_act_c->OL->dataflowStatus[i];
    //    if(frnn_act_c->OL->dataflowStatus[i] == 0) count_violation++;
    //}
    //
    MY_FLT_TYPE rule_Complexity = 0;
    for(int i = 0; i < frnn_act_c->F2->numRules; i++) {
        MY_FLT_TYPE tmp_rc = 0;
        for(int j = 0; j < frnn_act_c->F2->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < frnn_act_c->F2->numMembershipFun[j]; k++) {
                if(frnn_act_c->F2->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp_rc += ac_flag;
        }
        rule_Complexity += tmp_rc / frnn_act_c->F2->numInput;
    }
    rule_Complexity /= frnn_act_c->F2->numRules;
//    //
//    MY_FLT_TYPE count_connections = 0;
//    for(int i = 0; i < frnn_act_c->F2->numRules; i++) {
//        for(int j = 0; j < frnn_act_c->F2->numInput; j++) {
//            for(int k = 0; k < frnn_act_c->F2->numMembershipFun[j]; k++) {
//#if FRNN_FUZZY_RULE_LAYER == FRNN_FUZZY_RULE_LAYER_2
//                if(frnn_act_c->F2->connectStatus[i][j] == 0)
//                    continue;
//#endif
//                if(frnn_act_c->F2->connectStatusAll[i][j][k])
//                    count_connections++;
//            }
//        }
//    }
//    for(int i = 0; i < frnn_act_c->R3->numRoughSets; i++) {
//        for(int j = 0; j < frnn_act_c->R3->numInput; j++) {
//            if(frnn_act_c->R3->connectStatus[i][j])
//                count_connections++;
//        }
//    }
//    //for(int i = 0; i < frnn_act_c->OL->numOutput; i++) {
//    //    for(int j = 0; j < frnn_act_c->OL->numInput; j++) {
//    //        if(frnn_act_c->OL->connectStatus[i][j])
//    //            count_connections++;
//    //    }
//    //}
//    //
    //
    double f_simpl = 0.0;
    double f_simpl_F = 0.0;
    double f_simpl_R = 0.0;
    //total_penalty_EVO5_FRNN = 0.0;
    //for (int i = 0; i < MAX_NUM_FUZZY_RULE_EVO5_FRNN; i++) {
    //  if (flag_fuzzy_rule[i])
    //      f_simpl++;
    //}
    //int flag_no_fuzzy_rule = 0;
    //if (f_simpl == 0)
    //  flag_no_fuzzy_rule = 1;
    int *tmp1 = (int*)calloc(frnn_act_c->F2->numRules, sizeof(int));
    int *tmp2 = (int*)calloc(frnn_act_c->R3->numRoughSets, sizeof(int));
    for(int i = 0; i < frnn_act_c->F2->numRules; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < frnn_act_c->F2->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < frnn_act_c->F2->numMembershipFun[j]; k++) {
                if(frnn_act_c->F2->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp1[i] += ac_flag;
        }
        f_simpl_F += (double)tmp1[i] / frnn_act_c->F2->numInput;
    }
    f_simpl_F /= frnn_act_c->F2->numRules;
    for(int i = 0; i < frnn_act_c->R3->numRoughSets; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < frnn_act_c->R3->numInput; j++) {
            if(tmp1[j] && frnn_act_c->R3->connectStatus[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl_R += (double)tmp2[i] / frnn_act_c->R3->numInput;
    }
    f_simpl_R /= frnn_act_c->R3->numRoughSets;
    f_simpl = f_simpl_F + f_simpl_R;
    f_simpl /= 2;
    //f_simpl /= MAX_NUM_FUZZY_RULE_EVO5_FRNN;
    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    int count_violation = 0;
    //int tmp_sum = 0;
    //for(int i = 0; i < frnn_act_c->F2->numRules; i++) {
    //    tmp_sum += tmp1[i];
    //}
    //if(tmp_sum == 0.0) {
    //    count_violation++;
    //}
    //for(int i = 0; i < frnn_act_c->R3->numRoughSets; i++) {
    //    if(tmp2[i] == 0) {
    //        count_violation++;
    //    }
    //}
    for(int i = 0; i < NUM_label_MHEALTH; i++) {
        if(frnn_act_c->OL->dataflowStatus[i] == 0) count_violation++;
    }
#if STATUS_OUT_INDEICES_MOP_ACTIVITYDETECTION == FLAG_ON_MOP_ACTIVITYDETECTION
    printf("count_violation=%d\tcount_no_act=%d\n", count_violation, count_no_act);
#endif
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_ACT_C/* + count_no_act * 100*/);
    free(tmp1);
    free(tmp2);
    //////////////////////////////////////////////////////////////////////////
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
#if OBJ_INDICATORS_MOP_ACTIVITYDETECTION == MEAN_INDICATORS_MOP_ACTIVITYDETECTION
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    //fitness[0] = 1 - mean_recall / NUM_label_MHEALTH + val_violation;
    //fitness[1] = sum_weights + val_violation;
    fitness[0] = 1 - mean_precision / NUM_label_MHEALTH + val_violation;
    fitness[1] = 1 - mean_recall / NUM_label_MHEALTH + val_violation;
#else
    fitness[0] = 1 - mean_precision / NUM_label_MHEALTH + val_violation;
    fitness[1] = 1 - mean_recall / NUM_label_MHEALTH + val_violation;
#endif
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION
    fitness[2] = cur_dataflow / frnn_act_c->dataflowMax + val_violation;
#else
    fitness[2] = f_simpl + val_violation;
#endif
#elif OBJ_INDICATORS_MOP_ACTIVITYDETECTION == MIN_INDICATORS_MOP_ACTIVITYDETECTION
    fitness[0] = 1 - min_precision + val_violation;
    fitness[1] = 1 - min_recall + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION
    fitness[2] = cur_dataflow / frnn_act_c->dataflowMax + val_violation;
#else
    fitness[2] = count_connections / frnn_act_c->connectionMax + val_violation;
#endif
#else
#if FUZZY_RULE_OBJ_STATUS_MOP_ACTIVITYDETECTION == FUZZY_RULE_OBJ_ON_MOP_ACTIVITYDETECTION
    fitness[0] = tmp_all_err + val_violation;
    fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION
    fitness[2] = cur_dataflow / frnn_act_c->dataflowMax + val_violation;
#else
    fitness[2] = count_connections / frnn_act_c->connectionMax + val_violation;
#endif
#else
    fitness[0] = tmp_all_err + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_ACTIVITYDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_ACTIVITYDETECTION
    fitness[1] = cur_dataflow / frnn_act_c->dataflowMax + val_violation;
#else
    fitness[1] = count_connections / frnn_act_c->connectionMax + val_violation;
#endif
#endif
#endif

    return;
}

void Finalize_ActivityDetection_FRNN_Classify()
{
    for(int n = 0; n < nSubject_MHEALTH; n++) {
        for(int i = 0; i < nrows_MHEALTH[n]; i++) {
            free(data_MHEALTH_all[n][i]);
        }
        free(data_MHEALTH_all[n]);
    }
    free(data_MHEALTH_all);
    //
    frnn_act_c_free(frnn_act_c);
    return;
}

//////////////////////////////////////////////////////////////////////////
void frnn_act_c_setup(FRNN_ACT_C* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                      int* flagAdapMemship,
                      int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
                      int typeFuzzySet, int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
                      int consequenceNodeStatus, int centroid_num_tag,
                      int numInputConsequenceNode, MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq,
                      int flagConnectStatus, int flagConnectWeight)
{
    frnn->typeFuzzySet = typeFuzzySet;
    frnn->typeRules = typeRules;
    frnn->typeInRuleCorNum = typeInRuleCorNum;
    frnn->typeTypeReducer = typeTypeReducer;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
    frnn->flagConnectStatus = flagConnectStatus;
    frnn->flagConnectWeight = flagConnectWeight;

    frnn->layerNum = 4;

    frnn->numInput = numInput;
    frnn->numOutput = numOutput;

    int tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;

    frnn->M1 = setupMemberLayer(numInput, inputMin, inputMax, numMemship, flagAdapMemship, typeFuzzySet,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->F2 = setupFuzzyLayer(numInput, frnn->M1->numMembershipFun, numFuzzyRules, typeFuzzySet, typeRules, typeInRuleCorNum,
                               FLAG_STATUS_OFF,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->R3 = setupRoughLayer(frnn->F2->numRules, numRoughSets, typeFuzzySet,
                               1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->OL = setupOutReduceLayer(frnn->R3->numRoughSets, numOutput, outputMin, outputMax,
                                   typeFuzzySet, typeTypeReducer,
                                   consequenceNodeStatus, centroid_num_tag, numInputConsequenceNode, inputMin_cnsq, inputMax_cnsq,
                                   flagConnectStatus, flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);

    frnn->e = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_wrong = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->e_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_TP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_TN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->numInput * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->OL->numOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->numInput * frnn->F2->numRules +
                                            frnn->F2->numRules * frnn->R3->numRoughSets/* +
                                            frnn->R3->numRoughSets * frnn->O4->numOutput*/);
    } else {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->OL->numOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->F2->numRules +
                                            frnn->F2->numRules * frnn->R3->numRoughSets/* +
                                            frnn->R3->numRoughSets * frnn->O4->numOutput*/);
    }

    return;
}

void frnn_act_c_free(FRNN_ACT_C* frnn)
{
    freeMemberLayer(frnn->M1);
    freeFuzzyLayer(frnn->F2);
    freeRoughLayer(frnn->R3);
    freeOutReduceLayer(frnn->OL);

    free(frnn->e);

    free(frnn->N_sum);
    free(frnn->N_wrong);
    free(frnn->e_sum);

    free(frnn->N_TP);
    free(frnn->N_TN);
    free(frnn->N_FP);
    free(frnn->N_FN);

    free(frnn);

    return;
}

void frnn_act_c_init(FRNN_ACT_C* frnn, double* x, int mode)
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

    assignMemberLayer(frnn->M1, &x[count], mode);
    count += frnn->M1->numParaLocal;
    memcpy(frnn->F2->numMembershipFunCur, frnn->M1->numMembershipFunCur, frnn->M1->numInput * sizeof(int));
    assignFuzzyLayer(frnn->F2, &x[count], mode);
    count += frnn->F2->numParaLocal;
    assignRoughLayer(frnn->R3, &x[count], mode);
    count += frnn->R3->numParaLocal;
    assignOutReduceLayer(frnn->OL, &x[count], mode);

    return;
}

void ff_frnn_act_c(FRNN_ACT_C* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE inputConsequenceNode[][nCol_MHEALTH])
{
    MY_FLT_TYPE* dataflowStatus = (MY_FLT_TYPE*)calloc(frnn->M1->numInput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->M1->numInput; i++) dataflowStatus[i] = 1;
    ff_memberLayer(frnn->M1, valIn, dataflowStatus);
    free(dataflowStatus);
    ff_fuzzyLayer(frnn->F2, frnn->M1->degreeMembership, frnn->M1->dataflowStatus);
    ff_roughLayer(frnn->R3, frnn->F2->degreeRules, frnn->F2->dataflowStatus);
    if(frnn->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < frnn->OL->numOutput; i++) {
            for(int j = 0; j < frnn->OL->numInputConsequenceNode; j++) {
                frnn->OL->inputConsequenceNode[i][j] = inputConsequenceNode[i][j];
            }
        }
    }
    ff_outReduceLayer(frnn->OL, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
    memcpy(valOut, frnn->OL->valOutputFinal, frnn->OL->numOutput * sizeof(MY_FLT_TYPE));

    return;
}

//////////////////////////////////////////////////////////////////////////
static int** allocINT(int nrow, int ncol)
{
    int** tmp = NULL;
    if((tmp = (int**)malloc(nrow * sizeof(int*))) == NULL) {
        printf("%s(%d): ERROR!! --> malloc: no memory for matrix*\n", __FILE__, __LINE__);
        exit(-123320);
    } else {
        for(int i = 0; i < nrow; i++) {
            if((tmp[i] = (int*)malloc(ncol * sizeof(int))) == NULL) {
                printf("%s(%d): ERROR!! --> malloc: no memory for vector\n", __FILE__, __LINE__);
                exit(-123323);
            }
        }
    }
    return tmp;
}

static MY_FLT_TYPE** allocFLOAT(int nrow, int ncol)
{
    MY_FLT_TYPE** tmp = NULL;
    if((tmp = (MY_FLT_TYPE**)malloc(nrow * sizeof(MY_FLT_TYPE*))) == NULL) {
        printf("%s(%d): ERROR!! --> malloc: no memory for matrix*\n", __FILE__, __LINE__);
        exit(-123321);
    } else {
        for(int i = 0; i < nrow; i++) {
            if((tmp[i] = (MY_FLT_TYPE*)malloc(ncol * sizeof(MY_FLT_TYPE))) == NULL) {
                printf("%s(%d): ERROR!! --> malloc: no memory for vector\n", __FILE__, __LINE__);
                exit(-123322);
            }
        }
    }
    return tmp;
}

static void readData_MHEALTH(MY_FLT_TYPE** pDATA, char fname[], int nrow, int ncol, MY_FLT_TYPE max_val[],
                             MY_FLT_TYPE min_val[])
{
    FILE* fpt = fopen(fname, "r");
    if(fpt) {
        char buf[MAX_BUF_SIZE];
        char* p;
        for(int i = 0; i < nrow; i++) {
            int cur_c = 0;
            if(fgets(buf, MAX_BUF_SIZE, fpt)) {
                for(p = strtok(buf, " ,\t\r\n"); p; p = strtok(NULL, " ,\t\r\n")) {
                    int tmp_flag = 0;
                    double tmp_val;
                    if(cur_c >= ncol) {
                        printf("%s(%d): More data items for this row - %d (%d) - (%s), exiting...\n",
                               __FILE__, __LINE__, i, cur_c, p);
                        exit(-111000);
                    }
                    if(cur_c == IND_label_MHEALTH) {
                        sscanf(p, "%lf", &tmp_val);
                        tmp_val--;
                    } else {
                        sscanf(p, "%lf", &tmp_val);
                    }
                    if(tmp_val > max_val[cur_c]) max_val[cur_c] = tmp_val;
                    if(tmp_val < min_val[cur_c]) min_val[cur_c] = tmp_val;
                    pDATA[i][cur_c] = tmp_val;
                    cur_c++;
                }
                if(cur_c != ncol) {
                    printf("%s(%d): Number of data items is not consistent for this row - %d - (%d), exiting...\n",
                           __FILE__, __LINE__, i, cur_c);
                    exit(-111005);
                }
            } else {
                printf("%s(%d): No more data, exiting...\n", __FILE__, __LINE__);
                exit(-111006);
            }
        }
        fclose(fpt);
    } else {
        printf("Open file %s error, exiting...\n", fname);
        exit(-111007);
    }
    return;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////