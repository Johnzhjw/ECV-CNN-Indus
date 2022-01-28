#include "MOP_cnn_data.h"
#include "MOP_evoCNN.h"
#include <float.h>

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_evoCNN 0
#define FLAG_ON_MOP_evoCNN 1
#define STATUS_OUT_INDEICES_MOP_evoCNN FLAG_ON_MOP_evoCNN

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN 1024
#define MAX_LAB_NUM 1024
#define VIOLATION_PENALTY_evoCNN_C 1e6

//////////////////////////////////////////////////////////////////////////
int NDIM_evoCNN_Classify = 0;
int NOBJ_evoCNN_Classify = 0;
int NDIM_evoCNN_Classify_BP = 0;
int NOBJ_evoCNN_Classify_BP = 0;

//////////////////////////////////////////////////////////////////////////
int num_class_MNIST = 10;

LabelArr allLabels_train;
ImgArr   allImgs_train;
LabelArr allLabels_test;
ImgArr   allImgs_test;

ImgArr   allMaxInImgs_evoCNN;
ImgArr   allMinInImgs_evoCNN;
ImgArr   allMeanInImgs_evoCNN;
ImgArr   allStdInImgs_evoCNN;
ImgArr   allRangeInImgs_evoCNN;

CNN_evoCNN_C* cnn_evoCNN_c = NULL;

static int** allocINT(int nrow, int ncol);
static MY_FLT_TYPE** allocFLOAT(int nrow, int ncol);
static void ff_evoCNN_Classify(double* individual, ImgArr inputData, LabelArr outputData);
static void bp_evoCNN_Classify(double* individual, ImgArr inputData, LabelArr outputData);
static void getIndicators_Classify(MY_FLT_TYPE& mean_p, MY_FLT_TYPE& mean_r, MY_FLT_TYPE& mean_F, MY_FLT_TYPE& std_p,
                                   MY_FLT_TYPE& std_r, MY_FLT_TYPE& std_F);

//////////////////////////////////////////////////////////////////////////
void Initialize_evoCNN_Classify(int curN, int numN)
{
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    //
    char fname[MAX_STR_LEN];
    sprintf(fname, "../Data_all/Data_MNIST/train-images.idx3-ubyte");
    allImgs_train = read_Img_IDX_FILE(fname);
    sprintf(fname, "../Data_all/Data_MNIST/train-labels.idx1-ubyte");
    allLabels_train = read_Label_IDX_FILE(fname, num_class_MNIST);
    sprintf(fname, "../Data_all/Data_MNIST/t10k-images.idx3-ubyte");
    allImgs_test = read_Img_IDX_FILE(fname);
    sprintf(fname, "../Data_all/Data_MNIST/t10k-labels.idx1-ubyte");
    allLabels_test = read_Label_IDX_FILE(fname, num_class_MNIST);

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    cnn_evoCNN_c = (CNN_evoCNN_C*)calloc(1, sizeof(CNN_evoCNN_C));
    cnn_evoCNN_c_setup(cnn_evoCNN_c);
    //
    NDIM_evoCNN_Classify =
        cnn_evoCNN_c->C1->numParaLocal +
        cnn_evoCNN_c->P2->numParaLocal +
        cnn_evoCNN_c->C3->numParaLocal +
        cnn_evoCNN_c->P4->numParaLocal +
        cnn_evoCNN_c->O5->numParaLocal;
    NOBJ_evoCNN_Classify = 2;
    //
    NDIM_evoCNN_Classify_BP = NDIM_evoCNN_Classify;
    NOBJ_evoCNN_Classify_BP = NOBJ_evoCNN_Classify;
    //
    return;
}

void SetLimits_evoCNN_Classify(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    for(int i = 0; i < cnn_evoCNN_c->C1->numParaLocal; i++) {
        minLimit[count] = cnn_evoCNN_c->C1->xMin[i];
        maxLimit[count] = cnn_evoCNN_c->C1->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->P2->numParaLocal; i++) {
        minLimit[count] = cnn_evoCNN_c->P2->xMin[i];
        maxLimit[count] = cnn_evoCNN_c->P2->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->C3->numParaLocal; i++) {
        minLimit[count] = cnn_evoCNN_c->C3->xMin[i];
        maxLimit[count] = cnn_evoCNN_c->C3->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->P4->numParaLocal; i++) {
        minLimit[count] = cnn_evoCNN_c->P4->xMin[i];
        maxLimit[count] = cnn_evoCNN_c->P4->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->O5->numParaLocal; i++) {
        minLimit[count] = cnn_evoCNN_c->O5->xMin[i];
        maxLimit[count] = cnn_evoCNN_c->O5->xMax[i];
        count++;
    }
    return;
}

int CheckLimits_evoCNN_Classify(double* x, int nx)
{
    int count = 0;
    for(int i = 0; i < cnn_evoCNN_c->C1->numParaLocal; i++) {
        if(x[count] < cnn_evoCNN_c->C1->xMin[i] ||
           x[count] > cnn_evoCNN_c->C1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCNN_c->C1->xMin[i], cnn_evoCNN_c->C1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->P2->numParaLocal; i++) {
        if(x[count] < cnn_evoCNN_c->P2->xMin[i] ||
           x[count] > cnn_evoCNN_c->P2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCNN_c->P2->xMin[i], cnn_evoCNN_c->P2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->C3->numParaLocal; i++) {
        if(x[count] < cnn_evoCNN_c->C3->xMin[i] ||
           x[count] > cnn_evoCNN_c->C3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCNN_c->C3->xMin[i], cnn_evoCNN_c->C3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->P4->numParaLocal; i++) {
        if(x[count] < cnn_evoCNN_c->P4->xMin[i] ||
           x[count] > cnn_evoCNN_c->P4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCNN_c->P4->xMin[i], cnn_evoCNN_c->P4->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCNN_c->O5->numParaLocal; i++) {
        if(x[count] < cnn_evoCNN_c->O5->xMin[i] ||
           x[count] > cnn_evoCNN_c->O5->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->O5 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCNN_c->O5->xMin[i], cnn_evoCNN_c->O5->xMax[i]);
            return 0;
        }
        count++;
    }
    return 1;
}

void Fitness_evoCNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    ff_evoCNN_Classify(individual, allImgs_train, allLabels_train);
    //
    int len_lab = num_class_MNIST;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    getIndicators_Classify(mean_precision, mean_recall, mean_Fvalue, std_precision, std_recall, std_Fvalue);
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cnn_evoCNN_c->O5->dataflowStatus[i];
        if(cnn_evoCNN_c->O5->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_evoCNN_C);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
    fitness[0] = 1 - mean_recall + val_violation;
    fitness[1] = std_recall + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
    return;
}

void Fitness_evoCNN_Classify_test(double* individual, double* fitness)
{
    ff_evoCNN_Classify(individual, allImgs_test, allLabels_test);
    //
    int len_lab = num_class_MNIST;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    getIndicators_Classify(mean_precision, mean_recall, mean_Fvalue, std_precision, std_recall, std_Fvalue);
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cnn_evoCNN_c->O5->dataflowStatus[i];
        if(cnn_evoCNN_c->O5->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_evoCNN_C);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
    fitness[0] = 1 - mean_recall + val_violation;
    fitness[1] = std_recall + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
    return;
}

static void ff_evoCNN_Classify(double* individual, ImgArr inputData, LabelArr outputData)
{
    int num_sample = inputData->ImgNum;
    int len_lab = num_class_MNIST;
    cnn_evoCNN_c->sum_all = 0;
    cnn_evoCNN_c->sum_wrong = 0;
    for(int i = 0; i < cnn_evoCNN_c->numOutput; i++) {
        cnn_evoCNN_c->N_sum[i] = 0;
        cnn_evoCNN_c->N_wrong[i] = 0;
        cnn_evoCNN_c->e_sum[i] = 0;
        cnn_evoCNN_c->N_TP[i] = 0;
        cnn_evoCNN_c->N_TN[i] = 0;
        cnn_evoCNN_c->N_FP[i] = 0;
        cnn_evoCNN_c->N_FN[i] = 0;
    }
    cnn_evoCNN_c_init(cnn_evoCNN_c, individual, ASSIGN_MODE_FRNN);
    //
    MY_FLT_TYPE*** valIn;
    MY_FLT_TYPE valOut[MAX_LAB_NUM];
    for(int m = 0; m < num_sample; m++) {
        valIn = &inputData->ImgPtr[m].ImgData;
        ff_cnn_evoCNN_c(cnn_evoCNN_c, valIn, valOut, NULL);
        int cur_label = 0;
        MY_FLT_TYPE cur_out = valOut[0];
        for(int j = 1; j < cnn_evoCNN_c->numOutput; j++) {
            if(cur_out < valOut[j]) {
                cur_out = valOut[j];
                cur_label = j;
            }
        }
        int true_label = 0;
        MY_FLT_TYPE tmp_max_lab_val = outputData->LabelPtr[m].LabelData[0];
        for(int j = 1; j < cnn_evoCNN_c->numOutput; j++) {
            if(tmp_max_lab_val < outputData->LabelPtr[m].LabelData[j]) {
                tmp_max_lab_val = outputData->LabelPtr[m].LabelData[j];
                true_label = j;
            }
        }
        for(int j = 0; j < cnn_evoCNN_c->numOutput; j++) {
            if(j == cur_label && j == true_label) cnn_evoCNN_c->N_TP[j]++;
            if(j == cur_label && j != true_label) cnn_evoCNN_c->N_FP[j]++;
            if(j != cur_label && j == true_label) cnn_evoCNN_c->N_FN[j]++;
            if(j != cur_label && j != true_label) cnn_evoCNN_c->N_TN[j]++;
        }
        cnn_evoCNN_c->sum_all++;
        cnn_evoCNN_c->N_sum[true_label]++;
        if(cur_label != true_label) {
            cnn_evoCNN_c->sum_wrong++;
            cnn_evoCNN_c->N_wrong[true_label]++;
        }
    }
    //
    return;
}

static void bp_evoCNN_Classify(double* individual, ImgArr inputData, LabelArr outputData)
{
    int num_sample = inputData->ImgNum;
    int len_lab = num_class_MNIST;
    cnn_evoCNN_c->sum_all = 0;
    cnn_evoCNN_c->sum_wrong = 0;
    for(int i = 0; i < cnn_evoCNN_c->numOutput; i++) {
        cnn_evoCNN_c->N_sum[i] = 0;
        cnn_evoCNN_c->N_wrong[i] = 0;
        cnn_evoCNN_c->e_sum[i] = 0;
        cnn_evoCNN_c->N_TP[i] = 0;
        cnn_evoCNN_c->N_TN[i] = 0;
        cnn_evoCNN_c->N_FP[i] = 0;
        cnn_evoCNN_c->N_FN[i] = 0;
    }
    cnn_evoCNN_c_init(cnn_evoCNN_c, individual, INIT_BP_MODE_FRNN);
    //
    FRNNOpts opts;
    opts.numepochs = 100;
    opts.curepochs = 0;
    opts.batch_size = 128;
    opts.all_sample_num = num_sample;
    opts.cur_sample_num = 0;
    opts.alpha = (MY_FLT_TYPE)0.018;
    //
    MY_FLT_TYPE*** valIn;
    MY_FLT_TYPE valOut[MAX_LAB_NUM];
    int* tmp_ind = (int*)malloc(num_sample * sizeof(int));
    for(int i = 0; i < num_sample; i++) tmp_ind[i] = i;
    for(int iEp = 0; iEp < opts.numepochs; iEp++) {
        //opts.alpha *= (MY_FLT_TYPE)0.95;
        shuffle_FRNN_MODEL(tmp_ind, num_sample);
        for(int tmp_m = 0; tmp_m < num_sample; tmp_m++) {
            int m = tmp_ind[tmp_m];
            valIn = &inputData->ImgPtr[m].ImgData;
            if(opts.cur_sample_num % opts.batch_size == 0) opts.tag_init = 1;
            else opts.tag_init = 0;
            opts.cur_sample_num++;
            if(opts.cur_sample_num % opts.batch_size == 0 || opts.cur_sample_num == opts.all_sample_num) opts.tag_update = 1;
            else opts.tag_update = 0;
            ff_cnn_evoCNN_c(cnn_evoCNN_c, valIn, valOut, NULL);
            bp_cnn_evoCNN_c(cnn_evoCNN_c, valIn, outputData->LabelPtr[m].LabelData, NULL, opts);
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 1; j < cnn_evoCNN_c->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE tmp_max_lab_val = outputData->LabelPtr[m].LabelData[0];
            for(int j = 1; j < cnn_evoCNN_c->numOutput; j++) {
                if(tmp_max_lab_val < outputData->LabelPtr[m].LabelData[j]) {
                    tmp_max_lab_val = outputData->LabelPtr[m].LabelData[j];
                    true_label = j;
                }
            }
            for(int j = 0; j < cnn_evoCNN_c->numOutput; j++) {
                if(j == cur_label && j == true_label) cnn_evoCNN_c->N_TP[j]++;
                if(j == cur_label && j != true_label) cnn_evoCNN_c->N_FP[j]++;
                if(j != cur_label && j == true_label) cnn_evoCNN_c->N_FN[j]++;
                if(j != cur_label && j != true_label) cnn_evoCNN_c->N_TN[j]++;
            }
            cnn_evoCNN_c->sum_all++;
            cnn_evoCNN_c->N_sum[true_label]++;
            if(cur_label != true_label) {
                cnn_evoCNN_c->sum_wrong++;
                cnn_evoCNN_c->N_wrong[true_label]++;
            }
        }
        //

    }
    free(tmp_ind);
    //
    return;
}

void Finalize_evoCNN_Classify()
{
    cnn_evoCNN_c_free(cnn_evoCNN_c);

    free_Img(allImgs_train);
    free_Label(allLabels_train);
    free_Img(allImgs_test);
    free_Label(allLabels_test);

    return;
}

//////////////////////////////////////////////////////////////////////////
void Fitness_evoCNN_Classify_BP(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    MY_FLT_TYPE* tmp_save = (MY_FLT_TYPE*)malloc(nx * sizeof(MY_FLT_TYPE));
    memcpy(tmp_save, individual, nx * sizeof(MY_FLT_TYPE));
    for(int i = 0; i < nx; i++) {
        printf("%lf ", tmp_save[i]);
    }
    printf("\n");
    printf("\n");
    printf("\n");
    bp_evoCNN_Classify(individual, allImgs_train, allLabels_train);
    //
    cnn_evoCNN_c_init(cnn_evoCNN_c, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
    for(int i = 0; i < nx; i++) {
        printf("%lf ", tmp_save[i] - individual[i]);
    }
    printf("\n");
    free(tmp_save);
    ff_evoCNN_Classify(individual, allImgs_train, allLabels_train);
    //
    int len_lab = num_class_MNIST;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    getIndicators_Classify(mean_precision, mean_recall, mean_Fvalue, std_precision, std_recall, std_Fvalue);
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cnn_evoCNN_c->O5->dataflowStatus[i];
        if(cnn_evoCNN_c->O5->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_evoCNN_C);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
    fitness[0] = 1 - mean_recall + val_violation;
    fitness[1] = std_recall + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;

    return;
}

//////////////////////////////////////////////////////////////////////////
void cnn_evoCNN_c_setup(CNN_evoCNN_C* cnn)
{
    int numOutput = num_class_MNIST;
    //
    int typeFuzzySet = FUZZY_SET_I;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE;
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = (int)sqrt(numFuzzyRules);
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int flagConnectWeight = 0;
    //
    cnn->typeFuzzySet = typeFuzzySet;
    cnn->typeRules = typeRules;
    cnn->typeInRuleCorNum = typeInRuleCorNum;
    cnn->typeTypeReducer = typeTypeReducer;
    cnn->consequenceNodeStatus = consequenceNodeStatus;
    cnn->centroid_num_tag = centroid_num_tag;
    cnn->flagConnectWeight = flagConnectWeight;
    //
    cnn->layerNum = 5;
    cnn->numOutput = numOutput;
    //
    int inputHeightMax = allImgs_train->ImgPtr[0].r;
    int inputWidthMax = allImgs_train->ImgPtr[0].c;
    cnn->inputHeightMax = inputHeightMax;
    cnn->inputWidthMax = inputWidthMax;
    //
    int channelsIn_C1 = 1;
    int channelsOut_C1 = 6;
    int channelsInOut_P2 = channelsOut_C1;
    int channelsIn_C3 = channelsInOut_P2;
    int channelsOut_C3 = 12;
    int channelsInOut_P4 = channelsOut_C3;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    int tmp_flag_kernelFlagAdap = 0;
    int tmp_default_kernelFlag = KERNEL_FLAG_OPERATE;
    int tmp_flag_actFuncAdap = 0;
    int tmp_default_actFunc = ACT_FUNC_LEAKYRELU;
    int tmp_flag_paddingTypeAdap = 0;
    int tmp_default_paddingType = PADDING_SAME;
    int tmp_flag_poolTypeAdap = 0;
    int tmp_default_poolType = POOL_MAX;
    //
    cnn->C1 = setupConvLayer(inputHeightMax, inputWidthMax, channelsIn_C1, channelsOut_C1, channelsIn_C1,
                             tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                             1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                             tmp_flag_actFuncAdap, tmp_default_actFunc,
                             tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cnn->P2 = setupPoolLayer(cnn->C1->featureMapHeightMax, cnn->C1->featureMapWidthMax, channelsInOut_P2, cnn->C1->channelsOutMax,
                             1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                             MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                             MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                             tmp_flag_poolTypeAdap, tmp_default_poolType);
    cnn->C3 = setupConvLayer(cnn->P2->featureMapHeightMax, cnn->P2->featureMapWidthMax, channelsIn_C3, channelsOut_C3,
                             cnn->P2->channelsInOutMax,
                             tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                             1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                             tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                             tmp_flag_actFuncAdap, tmp_default_actFunc,
                             tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cnn->P4 = setupPoolLayer(cnn->C3->featureMapHeightMax, cnn->C3->featureMapWidthMax, channelsInOut_P4, cnn->C3->channelsOutMax,
                             1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                             MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                             MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                             tmp_flag_poolTypeAdap, tmp_default_poolType);
    int layerNum = 3;
    int numNodesAll[3] = { 4, 7, 2 };
    int flagActFuncICFC = FLAG_STATUS_OFF;
    int flagActFuncAdapICFC = FLAG_STATUS_OFF;
    int defaultActFuncTypeICFC = ACT_FUNC_LEAKYRELU;
    int flagConnectAdap = FLAG_STATUS_OFF;
    int flag_wt_positiveICFC = FLAG_STATUS_ON;
    cnn->O5 = setupInterCFCLayer(cnn->P4->channelsInOutMax, cnn->P4->featureMapHeightMax, cnn->P4->featureMapWidthMax,
                                 cnn->numOutput,
                                 flagActFuncICFC, flagActFuncAdapICFC, defaultActFuncTypeICFC,
                                 flagConnectAdap,
                                 tmp_typeCoding, layerNum, numNodesAll, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0, 0, 0);

    cnn->e = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cnn->N_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cnn->N_wrong = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cnn->e_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cnn->N_TP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cnn->N_TN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cnn->N_FP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cnn->N_FN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cnn->featureMapTagInitial = (int***)calloc(channelsIn_C1, sizeof(int**));
    cnn->dataflowInitial = (MY_FLT_TYPE***)calloc(channelsIn_C1, sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < channelsIn_C1; i++) {
        cnn->featureMapTagInitial[i] = (int**)calloc(inputHeightMax, sizeof(int*));
        cnn->dataflowInitial[i] = (MY_FLT_TYPE**)calloc(inputHeightMax, sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < inputHeightMax; j++) {
            cnn->featureMapTagInitial[i][j] = (int*)calloc(inputWidthMax, sizeof(int));
            cnn->dataflowInitial[i][j] = (MY_FLT_TYPE*)calloc(inputWidthMax, sizeof(MY_FLT_TYPE));
            for(int k = 0; k < inputWidthMax; k++) {
                cnn->featureMapTagInitial[i][j][k] = 1;
                cnn->dataflowInitial[i][j][k] = (MY_FLT_TYPE)(1.0 / (inputHeightMax * inputWidthMax));
            }
        }
    }

    //if (typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
    //    cnn->dataflowMax = (float)(cnn->M1->numInput * cnn->F2->numRules * cnn->R3->numRoughSets * cnn->O4->numOutput);
    //    cnn->connectionMax = (float)(cnn->M1->numInput * cnn->F2->numRules +
    //        cnn->F2->numRules * cnn->R3->numRoughSets +
    //        cnn->R3->numRoughSets * cnn->O4->numOutput);
    //}
    //else {
    //    cnn->dataflowMax = (float)(cnn->M1->outputSize * cnn->F2->numRules * cnn->R3->numRoughSets * cnn->O4->numOutput);
    //    cnn->connectionMax = (float)(cnn->M1->outputSize * cnn->F2->numRules +
    //        cnn->F2->numRules * cnn->R3->numRoughSets +
    //        cnn->R3->numRoughSets * cnn->O4->numOutput);
    //}

    return;
}

void cnn_evoCNN_c_free(CNN_evoCNN_C* cnn)
{
    freeConvLayer(cnn->C1);
    freePoolLayer(cnn->P2);
    freeConvLayer(cnn->C3);
    freePoolLayer(cnn->P4);
    freeInterCFCLayer(cnn->O5);

    free(cnn->e);

    free(cnn->N_sum);
    free(cnn->N_wrong);
    free(cnn->e_sum);

    free(cnn->N_TP);
    free(cnn->N_TN);
    free(cnn->N_FP);
    free(cnn->N_FN);

    int inputHeightMax = allImgs_train->ImgPtr[0].r;
    int inputWidthMax = allImgs_train->ImgPtr[0].c;
    int channelsIn_C1 = 1;
    for(int i = 0; i < channelsIn_C1; i++) {
        for(int j = 0; j < inputHeightMax; j++) {
            free(cnn->featureMapTagInitial[i][j]);
            free(cnn->dataflowInitial[i][j]);
        }
        free(cnn->featureMapTagInitial[i]);
        free(cnn->dataflowInitial[i]);
    }
    free(cnn->featureMapTagInitial);
    free(cnn->dataflowInitial);

    free(cnn);

    return;
}

void cnn_evoCNN_c_init(CNN_evoCNN_C* cnn, double* x, int mode)
{
    int count = 0;
    switch(mode) {
    case INIT_MODE_FRNN:
    case INIT_BP_MODE_FRNN:
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

    assignConvLayer(cnn->C1, &x[count], mode);
    count += cnn->C1->numParaLocal;
    assignPoolLayer(cnn->P2, &x[count], mode);
    count += cnn->P2->numParaLocal;
    assignConvLayer(cnn->C3, &x[count], mode);
    count += cnn->C3->numParaLocal;
    assignPoolLayer(cnn->P4, &x[count], mode);
    count += cnn->P4->numParaLocal;
    assignInterCFCLayer(cnn->O5, &x[count], mode);

    return;
}

void ff_cnn_evoCNN_c(CNN_evoCNN_C* cnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode)
{
    ff_convLayer(cnn->C1, valIn, cnn->featureMapTagInitial, &cnn->inputHeightMax, &cnn->inputWidthMax, cnn->dataflowInitial);
    ff_poolLayer(cnn->P2, cnn->C1->featureMapData, cnn->C1->featureMapTag,
                 cnn->C1->featureMapHeight, cnn->C1->featureMapWidth, cnn->C1->dataflowStatus);
    ff_convLayer(cnn->C3, cnn->P2->featureMapData, cnn->P2->featureMapTag,
                 cnn->P2->featureMapHeight, cnn->P2->featureMapWidth, cnn->P2->dataflowStatus);
    ff_poolLayer(cnn->P4, cnn->C3->featureMapData, cnn->C3->featureMapTag,
                 cnn->C3->featureMapHeight, cnn->C3->featureMapWidth, cnn->C3->dataflowStatus);
    ff_icfcLayer(cnn->O5, cnn->P4->featureMapData, cnn->P4->featureMapTag, cnn->P4->featureMapHeight, cnn->P4->featureMapWidth,
                 cnn->P4->dataflowStatus);
    memcpy(valOut, cnn->O5->outputData, cnn->O5->numOutput * sizeof(MY_FLT_TYPE));

    return;
}

void bp_cnn_evoCNN_c(CNN_evoCNN_C* cnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* tarOut, MY_FLT_TYPE** inputConsequenceNode,
                     FRNNOpts opts)
{
    MY_FLT_TYPE* cnnOut = (MY_FLT_TYPE*)calloc(cnn->O5->numOutput, sizeof(MY_FLT_TYPE));
    memcpy(cnnOut, cnn->O5->outputData, cnn->O5->numOutput * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE tmpMax = cnnOut[0];
    for(int i = 1; i < cnn->O5->numOutput; i++) {
        if(tmpMax < cnnOut[i])
            tmpMax = cnnOut[i];
        if(CHECK_INVALID(cnnOut[i])) {
            printf("%e\n", cnnOut[i]);
        }
    }
    for(int i = 1; i < cnn->O5->numOutput; i++) {
        printf("%e\n", cnnOut[i]);
    }
    MY_FLT_TYPE tmpSum = 1e-6;
    for(int i = 0; i < cnn->O5->numOutput; i++) {
        cnnOut[i] = exp(cnnOut[i] - tmpMax);
        tmpSum += cnnOut[i];
    }
    for(int i = 0; i < cnn->O5->numOutput; i++) {
        cnnOut[i] /= tmpSum;
    }
    for(int i = 0; i < cnn->O5->numOutput; i++) {
        cnn->O5->outputDelta[i] = cnnOut[i] - tarOut[i];
    }
    //
    free(cnnOut);
    //
    MY_FLT_TYPE* individual = (MY_FLT_TYPE*)calloc(NDIM_evoCNN_Classify_BP, sizeof(MY_FLT_TYPE));
    cnn_evoCNN_c_init(cnn, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
    MY_FLT_TYPE tmp_min = 1e30;
    MY_FLT_TYPE tmp_max = -1e30;
    for(int i = 1; i < NDIM_evoCNN_Classify_BP; i++) {
        if(tmp_min > individual[i]) tmp_min = individual[i];
        if(tmp_max < individual[i]) tmp_max = individual[i];
    }
    printf("tmp_min = %e\ttmp_max = %e\n", tmp_min, tmp_max);
    free(individual);
    //
    bp_derivative_poolLayer(cnn->P4);
    bp_derivative_convLayer(cnn->C3);
    bp_derivative_poolLayer(cnn->P2);
    bp_derivative_convLayer(cnn->C1);
    //
    bp_delta_icfcLayer(cnn->O5, cnn->P4->featureMapDelta, cnn->P4->featureMapDerivative, cnn->P4->featureMapTag);
    bp_delta_poolLayer(cnn->P4, cnn->C3->featureMapDelta, cnn->C3->featureMapDerivative, cnn->C3->featureMapTag);
    bp_delta_convLayer(cnn->C3, cnn->P2->featureMapDelta, cnn->P2->featureMapDerivative, cnn->P2->featureMapTag);
    bp_delta_poolLayer(cnn->P2, cnn->C1->featureMapDelta, cnn->C1->featureMapDerivative, cnn->C1->featureMapTag);
    //
    bp_update_convLayer(cnn->C1, valIn, cnn->featureMapTagInitial, opts);
    bp_update_convLayer(cnn->C3, cnn->P2->featureMapData, cnn->P2->featureMapTag, opts);
    bp_update_icfcLayer(cnn->O5, cnn->P4->featureMapData, cnn->P4->featureMapTag, opts);
    //
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

static void getIndicators_Classify(MY_FLT_TYPE& mean_p, MY_FLT_TYPE& mean_r, MY_FLT_TYPE& mean_F, MY_FLT_TYPE& std_p,
                                   MY_FLT_TYPE& std_r, MY_FLT_TYPE& std_F)
{
    int len_lab = num_class_MNIST;
    MY_FLT_TYPE sum_precision = cnn_evoCNN_c->sum_wrong / cnn_evoCNN_c->sum_all;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    MY_FLT_TYPE tmp_precision[MAX_LAB_NUM];
    MY_FLT_TYPE tmp_recall[MAX_LAB_NUM];
    MY_FLT_TYPE tmp_Fvalue[MAX_LAB_NUM];
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < len_lab; i++) {
        if(cnn_evoCNN_c->N_TP[i] > 0) {
            tmp_precision[i] = cnn_evoCNN_c->N_TP[i] / (cnn_evoCNN_c->N_TP[i] + cnn_evoCNN_c->N_FP[i]);
            tmp_recall[i] = cnn_evoCNN_c->N_TP[i] / (cnn_evoCNN_c->N_TP[i] + cnn_evoCNN_c->N_FN[i]);
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
#if STATUS_OUT_INDEICES_MOP_evoCNN == FLAG_ON_MOP_evoCNN
        printf("%f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i]);
#endif
    }
    mean_precision /= len_lab;
    mean_recall /= len_lab;
    mean_Fvalue /= len_lab;
    for(int i = 0; i < len_lab; i++) {
        std_precision += (tmp_precision[i] - mean_precision) * (tmp_precision[i] - mean_precision);
        std_recall += (tmp_recall[i] - mean_recall) * (tmp_recall[i] - mean_recall);
        std_Fvalue += (tmp_Fvalue[i] - mean_Fvalue) * (tmp_Fvalue[i] - mean_Fvalue);
    }
    std_precision /= len_lab;
    std_precision = sqrt(std_precision);
    std_recall /= len_lab;
    std_recall = sqrt(std_recall);
    std_Fvalue /= len_lab;
    std_Fvalue = sqrt(std_Fvalue);
    //
    mean_p = mean_precision;
    mean_r = mean_recall;
    mean_F = mean_Fvalue;
    std_p = std_precision;
    std_r = std_recall;
    std_F = std_Fvalue;
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////