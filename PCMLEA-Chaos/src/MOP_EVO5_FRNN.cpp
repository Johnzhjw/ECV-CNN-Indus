#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOP_EVO5_FRNN.h"
//#include <Eigen/Dense>
//#include <Eigen/SVD>
//#include <iostream>

//using namespace Eigen;
//using namespace std;

#define PARA_MAX_EVO5_FRNN_MEM_REAL 120.0
#define PARA_MAX_EVO5_FRNN_MEM_REAL_DIFF 10.0
#define PARA_MAX_EVO5_FRNN_MEM_SIGMA 50.0
#define PARA_MAX_EVO5_FRNN_MEM_BELL_A 50.0
#define PARA_MAX_EVO5_FRNN_MEM_BELL_B 50.0
#define PARA_MAX_EVO5_FRNN_MEM_SIGMOID 50.0
#define PARA_MAX_EVO5_FRNN_MEM_RATIO 2.0
#define PARA_MAX_EVO5_FRNN_W 1.0
#define PARA_MAX_EVO5_FRNN_BIAS 120.0

#define LOWER_EVO5_FRNN 0
#define UPPER_EVO5_FRNN 1

#define penaltyVal_EVO5_FRNN (1e6)

static int seed_EVO5_FRNN;
static long rnd_uni_init_EVO5_FRNN;
static double trainData_EVO5_FRNN[2000];
static double testData_EVO5_FRNN[2000];
static int trainDataSize_EVO5_FRNN;
static int testDataSize_EVO5_FRNN;

// FRNN
static int flag_cluster_bin_rule_EVO5_FRNN[MAX_NUM_FUZZY_RULE_EVO5_FRNN][N_INPUT_EVO5_FRNN];
static int type_fuzzy_mem_func_EVO5_FRNN[N_INPUT_EVO5_FRNN][MAX_NUM_CLUSTER_EVO5_FRNN];
static int flag_fuzzy_bin_rough_EVO5_FRNN[NUM_CLASS_EVO5_FRNN][MAX_NUM_FUZZY_RULE_EVO5_FRNN];
static double
para_fuzzy_mem_func_EVO5_FRNN[N_INPUT_EVO5_FRNN][MAX_NUM_CLUSTER_EVO5_FRNN][MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN];
static double para_fuzzy_rough_EVO5_FRNN[NUM_CLASS_EVO5_FRNN][MAX_NUM_FUZZY_RULE_EVO5_FRNN];
static double para_class_output_EVO5_FRNN[NUM_CLASS_EVO5_FRNN][2][NUM_PARA_CONSE_EVO5_FRNN];
static double para_predict_EVO5_FRNN;

static double L1_input_output_EVO5_FRNN[N_INPUT_EVO5_FRNN];
static double L2_output_EVO5_FRNN[N_INPUT_EVO5_FRNN][MAX_NUM_CLUSTER_EVO5_FRNN][2];
static double L2_5_T_S_norms_EVO5_FRNN[N_INPUT_EVO5_FRNN][MAX_NUM_CLUSTER_EVO5_FRNN][2];
static double L3_fuzzy_rule_values_EVO5_FRNN[MAX_NUM_FUZZY_RULE_EVO5_FRNN][2];
static double L4_output_EVO5_FRNN[MAX_NUM_FUZZY_RULE_EVO5_FRNN][2];
static double L5_output_y_EVO5_FRNN[2];
static double L6_predict_EVO5_FRNN;

static double total_penalty_EVO5_FRNN;

//
static void trimLine_EVO5_FRNN(char line[]);
static void preprocess_EVO5_FRNN(double* individual);
static double precision_EVO5_FRNN(double dataSet[], int dataLen);
static double simplicity_EVO5_FRNN();

void Initialize_data_EVO5_FRNN(int curN, int numN, int trainNo, int testNo, int endNo)
{
    seed_EVO5_FRNN = 237;
    rnd_uni_init_EVO5_FRNN = -(long)seed_EVO5_FRNN;
    for(int i = 0; i < curN; i++) {
        seed_EVO5_FRNN = (seed_EVO5_FRNN + 111) % 1235;
        rnd_uni_init_EVO5_FRNN = -(long)seed_EVO5_FRNN;
    }

    char filename[1024] = "../Data_all/AllFileNames_FRNN";
    FILE* fpt;

    if((fpt = fopen(filename, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }

    trainDataSize_EVO5_FRNN = 0;
    testDataSize_EVO5_FRNN = 0;

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
        trimLine_EVO5_FRNN(StrLine);

        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        // fgets(StrLine, 1024, fpt_data);
        if(fgets(StrLine, 1024, fpt_data) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        int tmp;
        int tmp_size;
        double elem;

        tmp = fscanf(fpt_data, "%d", &tmp_size);
        if(tmp == EOF) {
            printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
            exit(2000);
        }

        for(int i = 0; i < tmp_size; i++) {
            tmp = fscanf(fpt_data, "%lf", &elem);
            if(tmp == EOF) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2000);
            }
            trainData_EVO5_FRNN[trainDataSize_EVO5_FRNN + i] = elem;
        }

        trainDataSize_EVO5_FRNN += tmp_size;

        fclose(fpt_data);
    }
    for(seq = testNo; seq < endNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_EVO5_FRNN(StrLine);

        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        // fgets(StrLine, 1024, fpt_data);
        if(fgets(StrLine, 1024, fpt_data) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        int tmp;
        int tmp_size;
        double elem;

        tmp = fscanf(fpt_data, "%d", &tmp_size);
        if(tmp == EOF) {
            printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
            exit(2000);
        }

        for(int i = 0; i < tmp_size; i++) {
            tmp = fscanf(fpt_data, "%lf", &elem);
            if(tmp == EOF) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2000);
            }
            testData_EVO5_FRNN[testDataSize_EVO5_FRNN + i] = elem;
        }

        testDataSize_EVO5_FRNN += tmp_size;

        fclose(fpt_data);
    }

    fclose(fpt);

    // EVO5_FRNN结构的初始化
    // None

    return;
}

void SetLimits_EVO5_FRNN(double* minLimit, double* maxLimit, int nx)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO5_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO5_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO5_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO5_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO5_FRNN;
    int offset6 = offset5 + DIM_CONSEQUENCE_EVO5_FRNN;
    int offset7 = offset6 + DIM_CONSE_WEIGHT_EVO5_FRNN;

    for(int i = offset0; i < offset1; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 3 - 1e-6;
    }
    for(int i = offset1; i < offset2; i++) {
        minLimit[i] = 0;
        maxLimit[i] = MAX_NUM_CLUSTER_EVO5_FRNN + 1 - 1e-6;
    }
    for(int i = offset2; i < offset3; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 2 - 1e-6;
    }
    int tag = 0;
    double cur_min, cur_max;
    for(int i = offset3; i < offset4; i += MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN) {
        if(tag < MAX_NUM_CLUSTER_EVO5_FRNN) {
            cur_min = 0.0;
            cur_max = PARA_MAX_EVO5_FRNN_MEM_REAL;
        } else {
            cur_min = -PARA_MAX_EVO5_FRNN_MEM_REAL_DIFF;
            cur_max = PARA_MAX_EVO5_FRNN_MEM_REAL_DIFF;
        }
        tag++;
        minLimit[i + 0] = 1e-2;
        minLimit[i + 1] = cur_min;
        minLimit[i + 2] = 1e-2;
        minLimit[i + 3] = cur_min;
        minLimit[i + 4] = -PARA_MAX_EVO5_FRNN_MEM_SIGMOID;
        minLimit[i + 5] = cur_min;
        minLimit[i + 6] = 1.0 + 1e-6;
        minLimit[i + 7] = 0.5 + 1e-6;
        maxLimit[i + 0] = PARA_MAX_EVO5_FRNN_MEM_SIGMA;
        maxLimit[i + 1] = cur_max;
        maxLimit[i + 2] = PARA_MAX_EVO5_FRNN_MEM_SIGMA;
        maxLimit[i + 3] = cur_max;
        maxLimit[i + 4] = PARA_MAX_EVO5_FRNN_MEM_SIGMOID;
        maxLimit[i + 5] = cur_max;
        maxLimit[i + 6] = 2.0 - 1e-6;
        maxLimit[i + 7] = 1.0 - 1e-6;
    }
    for(int i = offset4; i < offset5; i++) {
        minLimit[i] = 0;
        maxLimit[i] = PARA_MAX_EVO5_FRNN_W;
    }
    for(int i = offset5; i < offset6; i += NUM_PARA_CONSE_EVO5_FRNN) {
        for(int j = 0; j < NUM_PARA_CONSE_EVO5_FRNN; j++) {
            if(j < N_INPUT_EVO5_FRNN) {
                minLimit[i + j] = 0.0;
                maxLimit[i + j] = PARA_MAX_EVO5_FRNN_W;
            } else {
                minLimit[i + j] = 0.0;
                maxLimit[i + j] = PARA_MAX_EVO5_FRNN_BIAS;
            }
        }
    }
    for(int i = offset6; i < offset7; i++) {
        minLimit[i] = 0.0;
        maxLimit[i] = 1.0;
    }

    return;
}

int CheckLimits_EVO5_FRNN(double* x, int nx)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO5_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO5_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO5_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO5_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO5_FRNN;
    int offset6 = offset5 + DIM_CONSEQUENCE_EVO5_FRNN;
    int offset7 = offset6 + DIM_CONSE_WEIGHT_EVO5_FRNN;

    for(int i = offset0; i < offset1; i++) {
        if(x[i] < 0 || x[i] > 3 - 1e-6 || isnan(x[i])) {
            printf("%s(%d): Check limits FAIL - EVO5_FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 3 - 1e-6);
            return false;
        }
    }
    for(int i = offset1; i < offset2; i++) {
        if(x[i] < 0 || x[i] > MAX_NUM_CLUSTER_EVO5_FRNN + 1 - 1e-6 || isnan(x[i])) {
            printf("%s(%d): Check limits FAIL - EVO5_FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, MAX_NUM_CLUSTER_EVO5_FRNN + 1 - 1e-6);
            return false;
        }
    }
    for(int i = offset2; i < offset3; i++) {
        if(x[i] < 0 || x[i] > 2 - 1e-6 || isnan(x[i])) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 2 - 1e-6);
            return false;
        }
    }
    int tag = 0;
    double cur_min, cur_max;
    for(int i = offset3; i < offset4; i += MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN) {
        if(tag < MAX_NUM_CLUSTER_EVO5_FRNN) {
            cur_min = 0.0;
            cur_max = PARA_MAX_EVO5_FRNN_MEM_REAL;
        } else {
            cur_min = -PARA_MAX_EVO5_FRNN_MEM_REAL_DIFF;
            cur_max = PARA_MAX_EVO5_FRNN_MEM_REAL_DIFF;
        }
        tag++;
        double tmp_min[MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN], tmp_max[MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN];
        tmp_min[0] = 1e-2;
        tmp_min[1] = cur_min;
        tmp_min[2] = 1e-2;
        tmp_min[3] = cur_min;
        tmp_min[4] = -PARA_MAX_EVO5_FRNN_MEM_SIGMOID;
        tmp_min[5] = cur_min;
        tmp_min[6] = 1.0 + 1e-6;
        tmp_min[7] = 0.5 + 1e-6;
        tmp_max[0] = PARA_MAX_EVO5_FRNN_MEM_SIGMA;
        tmp_max[1] = cur_max;
        tmp_max[2] = PARA_MAX_EVO5_FRNN_MEM_SIGMA;
        tmp_max[3] = cur_max;
        tmp_max[4] = PARA_MAX_EVO5_FRNN_MEM_SIGMOID;
        tmp_max[5] = cur_max;
        tmp_max[6] = 2.0 - 1e-6;
        tmp_max[7] = 1.0 - 1e-6;
        for(int j = 0; j < MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN; j++) {
            if(x[i + j] < tmp_min[j] || x[i + j] > tmp_max[j] || isnan(x[i + j])) {
                printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i + j, x[i + j], tmp_min[j], tmp_max[j]);
                return false;
            }
        }
    }
    for(int i = offset4; i < offset5; i++) {
        if(x[i] < 0 || x[i] > PARA_MAX_EVO5_FRNN_W || isnan(x[i])) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, PARA_MAX_EVO5_FRNN_W);
            return false;
        }
    }
    for(int i = offset5; i < offset6; i += NUM_PARA_CONSE_EVO5_FRNN) {
        double tmp_min, tmp_max;
        for(int j = 0; j < NUM_PARA_CONSE_EVO5_FRNN; j++) {
            if(j < N_INPUT_EVO5_FRNN) {
                tmp_min = 0.0;
                tmp_max = PARA_MAX_EVO5_FRNN_W;
            } else {
                tmp_min = 0.0;
                tmp_max = PARA_MAX_EVO5_FRNN_BIAS;
            }
            if(x[i + j] < tmp_min || x[i + j] > tmp_max || isnan(x[i + j])) {
                printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i + j, x[i + j], tmp_min, tmp_max);
                return false;
            }
        }
    }
    for(int i = offset6; i < offset7; i++) {
        if(x[i] < 0.0 || x[i] > 1.0 || isnan(x[i])) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 1.0);
            return false;
        }
    }

    return true;
}

static void preprocess_EVO5_FRNN(double* individual)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO5_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO5_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO5_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO5_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO5_FRNN;
    int offset6 = offset5 + DIM_CONSEQUENCE_EVO5_FRNN;
    int offset7 = offset6 + DIM_CONSE_WEIGHT_EVO5_FRNN;

    //int i;
    int tmp_i, tmp_j, tmp_k;
    for(int i = offset0; i < offset1; i++) {
        tmp_i = (i - offset0) / MAX_NUM_CLUSTER_EVO5_FRNN;
        tmp_j = (i - offset0 - tmp_i * MAX_NUM_CLUSTER_EVO5_FRNN);
        type_fuzzy_mem_func_EVO5_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset1; i < offset2; i++) {
        tmp_i = (i - offset1) / N_INPUT_EVO5_FRNN;
        tmp_j = (i - offset1 - tmp_i * N_INPUT_EVO5_FRNN);
        flag_cluster_bin_rule_EVO5_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset2; i < offset3; i++) {
        tmp_i = (i - offset2) / MAX_NUM_FUZZY_RULE_EVO5_FRNN;
        tmp_j = (i - offset2 - tmp_i * MAX_NUM_FUZZY_RULE_EVO5_FRNN);
        flag_fuzzy_bin_rough_EVO5_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset3; i < offset4; i++) {
        tmp_i = (i - offset3) / (MAX_NUM_CLUSTER_EVO5_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN);
        tmp_j = (i - offset3 - tmp_i * MAX_NUM_CLUSTER_EVO5_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN) /
                MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN;
        tmp_k = (i - offset3 - tmp_i * MAX_NUM_CLUSTER_EVO5_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN - tmp_j *
                 MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN);
        para_fuzzy_mem_func_EVO5_FRNN[tmp_i][tmp_j][tmp_k] = individual[i];
    }
    for(int i = offset4; i < offset5; i++) {
        tmp_i = (i - offset4) / MAX_NUM_FUZZY_RULE_EVO5_FRNN;
        tmp_j = (i - offset4 - tmp_i * MAX_NUM_FUZZY_RULE_EVO5_FRNN);
        para_fuzzy_rough_EVO5_FRNN[tmp_i][tmp_j] = individual[i];
    }
    for(int i = offset5; i < offset6; i++) {
        tmp_i = (i - offset5) / (NUM_PARA_CONSE_EVO5_FRNN);
        tmp_j = (i - offset5 - tmp_i * (NUM_PARA_CONSE_EVO5_FRNN)) / NUM_PARA_CONSE_EVO5_FRNN;
        tmp_k = (i - offset5 - tmp_i * (NUM_PARA_CONSE_EVO5_FRNN) - tmp_j * NUM_PARA_CONSE_EVO5_FRNN);
        para_class_output_EVO5_FRNN[tmp_i][tmp_j][tmp_k] = individual[i];
    }
    for(int i = offset6; i < offset7; i++) {
        para_predict_EVO5_FRNN = individual[i];
    }
    para_predict_EVO5_FRNN = 0.5;

    return;
}

void Fitness_EVO5_FRNN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    preprocess_EVO5_FRNN(individual);

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    //double f_normp = 0.0;

    f_prcsn = precision_EVO5_FRNN(trainData_EVO5_FRNN, trainDataSize_EVO5_FRNN);

    //
    f_simpl = simplicity_EVO5_FRNN();

    fitness[0] = f_prcsn + total_penalty_EVO5_FRNN;
    fitness[1] = f_simpl + total_penalty_EVO5_FRNN;

    return;
}

void Fitness_EVO5_FRNN_testSet(double* individual, double* fitness)
{
    preprocess_EVO5_FRNN(individual);

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    //double f_normp = 0.0;

    f_prcsn = precision_EVO5_FRNN(testData_EVO5_FRNN, testDataSize_EVO5_FRNN);

    //
    f_simpl = simplicity_EVO5_FRNN();

    fitness[0] = f_prcsn + total_penalty_EVO5_FRNN;
    fitness[1] = f_simpl + total_penalty_EVO5_FRNN;

    return;
}

static void trimLine_EVO5_FRNN(char line[])
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

static double precision_EVO5_FRNN(double dataSet[], int dataLen)
{
    double f_prcsn = 0.0;

    //0 1 2
    int offset_data[N_INPUT_EVO5_FRNN];// = { 0, 1, 2 };
    for(int i = 0; i < N_INPUT_EVO5_FRNN; i++) {
        offset_data[i] = i;
    }

    for(int i = offset_data[N_INPUT_EVO5_FRNN - 1]; i < dataLen - 1; i++) {
        //Layer 1
        for(int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
            L1_input_output_EVO5_FRNN[j] = dataSet[i - offset_data[j]];
        }
        for(int j = 1; j < N_INPUT_EVO5_FRNN; j++) {
            L1_input_output_EVO5_FRNN[j] = dataSet[i - offset_data[j - 1]] -
                                           dataSet[i - offset_data[j]];
        }
        //Layer 2
        for(int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
            double tmp_func_para[MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN];
            for(int k = 0; k < MAX_NUM_CLUSTER_EVO5_FRNN; k++) {
                for(int l = 0; l < MAX_NUM_CLUSTER_MEM_PPARA_EVO5_FRNN; l++) {
                    tmp_func_para[l] = para_fuzzy_mem_func_EVO5_FRNN[j][k][l];
                }
                double sigma1 = tmp_func_para[0];
                double c1 = tmp_func_para[1];
                double sigma2 = tmp_func_para[2];
                double c2 = tmp_func_para[3];
                double gamma = tmp_func_para[4];
                double c3 = tmp_func_para[5];
                double a1 = tmp_func_para[6];
                double a2 = tmp_func_para[7];
                int t11, t12, t21, t22;
                double cur_in = L1_input_output_EVO5_FRNN[j];
                double cur_out;
                switch(type_fuzzy_mem_func_EVO5_FRNN[j][k]) {
                case 0:
                    cur_out = exp(-(cur_in - c1) * (cur_in - c1) / 2.0 / (sigma1 * sigma1));
                    break;
                case 1:
                    if(cur_in <= c1) {
                        t11 = 1;
                        t12 = 0;
                    } else {
                        t11 = 0;
                        t12 = 1;
                    }
                    if(cur_in >= c2) {
                        t21 = 1;
                        t22 = 0;
                    } else {
                        t21 = 0;
                        t22 = 1;
                    }
                    cur_out = (t11 * exp(-(cur_in - c1) * (cur_in - c1) / 2.0 / (sigma1 * sigma1)) + t12) *
                              (t21 * exp(-(cur_in - c2) * (cur_in - c2) / 2.0 / (sigma2 * sigma2)) + t22);
                    break;
                case 2:
                    cur_out = 1.0 / (1.0 + exp(-gamma * (cur_in - c3)));
                    break;
                default:
                    printf("%s(%d): Unknown MF, exiting...\n", __FILE__, __LINE__);
                    exit(-1234567);
                    break;
                }
                L2_output_EVO5_FRNN[j][k][UPPER_EVO5_FRNN] = pow((1.0 - pow((1.0 - cur_out), a1)), (1.0 / a1));
                L2_output_EVO5_FRNN[j][k][LOWER_EVO5_FRNN] = pow((1.0 - pow((1.0 - cur_out), a2)), (1.0 / a2));
            }
        }
        //Layer2.5
        for(int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
            for(int k = 0; k < MAX_NUM_CLUSTER_EVO5_FRNN; k++) {
                L2_5_T_S_norms_EVO5_FRNN[j][k][LOWER_EVO5_FRNN] =
                    L2_output_EVO5_FRNN[j][k][LOWER_EVO5_FRNN];
                L2_5_T_S_norms_EVO5_FRNN[j][k][UPPER_EVO5_FRNN] =
                    L2_output_EVO5_FRNN[j][k][UPPER_EVO5_FRNN];
            }
        }
        //Layer 3
        memset(L3_fuzzy_rule_values_EVO5_FRNN, 0, MAX_NUM_FUZZY_RULE_EVO5_FRNN * 2 * sizeof(double));
        for(int a = 0; a < MAX_NUM_FUZZY_RULE_EVO5_FRNN; a++) {
            int flag_tmp = 0;
            double product0 = 1.0;
            double product1 = 1.0;
            for(int b = 0; b < N_INPUT_EVO5_FRNN; b++) {
                if(flag_cluster_bin_rule_EVO5_FRNN[a][b]) {
                    int tmp_ind = flag_cluster_bin_rule_EVO5_FRNN[a][b] - 1;
                    product0 *= L2_5_T_S_norms_EVO5_FRNN[b][tmp_ind][LOWER_EVO5_FRNN];
                    product1 *= L2_5_T_S_norms_EVO5_FRNN[b][tmp_ind][UPPER_EVO5_FRNN];
                    flag_tmp++;
                }
            }
            if(flag_tmp) {
                L3_fuzzy_rule_values_EVO5_FRNN[a][LOWER_EVO5_FRNN] = product0;
                L3_fuzzy_rule_values_EVO5_FRNN[a][UPPER_EVO5_FRNN] = product1;
            } else {
                L3_fuzzy_rule_values_EVO5_FRNN[a][LOWER_EVO5_FRNN] = 0.0;
                L3_fuzzy_rule_values_EVO5_FRNN[a][UPPER_EVO5_FRNN] = 0.0;
            }
        }
        double tmp_sum_L4[2] = { 0, 0 };
        double tmp_sum_L4_all = 0.0;
        //Layer 4
        for(int j = 0; j < NUM_CLASS_EVO5_FRNN; j++) {
            L4_output_EVO5_FRNN[j][0] = 0.0;
            L4_output_EVO5_FRNN[j][1] = 0.0;
            for(int k = 0; k < MAX_NUM_FUZZY_RULE_EVO5_FRNN; k++) {
                if(flag_fuzzy_bin_rough_EVO5_FRNN[j][k]) {
                    L4_output_EVO5_FRNN[j][LOWER_EVO5_FRNN] +=
                        para_fuzzy_rough_EVO5_FRNN[j][k] * L3_fuzzy_rule_values_EVO5_FRNN[k][LOWER_EVO5_FRNN];
                    L4_output_EVO5_FRNN[j][UPPER_EVO5_FRNN] +=
                        para_fuzzy_rough_EVO5_FRNN[j][k] * L3_fuzzy_rule_values_EVO5_FRNN[k][UPPER_EVO5_FRNN];
                }
            }
            tmp_sum_L4[LOWER_EVO5_FRNN] += L4_output_EVO5_FRNN[j][LOWER_EVO5_FRNN];
            tmp_sum_L4[UPPER_EVO5_FRNN] += L4_output_EVO5_FRNN[j][UPPER_EVO5_FRNN];
        }
        tmp_sum_L4_all = tmp_sum_L4[LOWER_EVO5_FRNN] + tmp_sum_L4[UPPER_EVO5_FRNN];
        ////
        //for (int j = 0; j < NUM_CLASS_EVO5_FRNN; j++) {
        //  if (tmp_sum_L4 > 0)
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_EVO5_FRNN - 1]) = L4_output[j] / tmp_sum_L4;
        //  else
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_EVO5_FRNN - 1]) = L4_output[j];
        //}
        //O_targetOutput(i - offset_data[N_INPUT_EVO5_FRNN - 1], 0) = trainData[i + 1];
        //Layer 5
        L5_output_y_EVO5_FRNN[LOWER_EVO5_FRNN] = 0.0;
        L5_output_y_EVO5_FRNN[UPPER_EVO5_FRNN] = 0.0;
        double tmp_y[2];
        double tmp[2];
        for(int j = 0; j < NUM_CLASS_EVO5_FRNN; j++) {
            tmp[0] = 0;
            for(int k = 0; k < N_INPUT_EVO5_FRNN; k++) {
                tmp[0] += para_class_output_EVO5_FRNN[j][0][k] * dataSet[i - offset_data[k]];
            }
            tmp[0] += para_class_output_EVO5_FRNN[j][0][N_INPUT_EVO5_FRNN];
            tmp_y[LOWER_EVO5_FRNN] = tmp[0];
            tmp_y[UPPER_EVO5_FRNN] = tmp[0];
            tmp_y[LOWER_EVO5_FRNN] *= L4_output_EVO5_FRNN[j][LOWER_EVO5_FRNN];
            tmp_y[UPPER_EVO5_FRNN] *= L4_output_EVO5_FRNN[j][UPPER_EVO5_FRNN];
            //double tmp = L4_output[j] * L1_input_output[j];
            if(tmp_sum_L4_all > 0) {
                tmp_y[LOWER_EVO5_FRNN] /= tmp_sum_L4_all;
                tmp_y[UPPER_EVO5_FRNN] /= tmp_sum_L4_all;
            }
            L5_output_y_EVO5_FRNN[LOWER_EVO5_FRNN] += tmp_y[LOWER_EVO5_FRNN];
            L5_output_y_EVO5_FRNN[UPPER_EVO5_FRNN] += tmp_y[UPPER_EVO5_FRNN];
        }
        //Layer 6
        L6_predict_EVO5_FRNN = L5_output_y_EVO5_FRNN[LOWER_EVO5_FRNN] + L5_output_y_EVO5_FRNN[UPPER_EVO5_FRNN];
        ////
        //double tmp_mean = 0.0;
        //double tmp_std = 0.0;
        //for (int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
        //  tmp_mean += L1_input_output[j];
        //}
        //tmp_mean /= N_INPUT_EVO5_FRNN;
        //for (int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
        //  tmp_std += (L1_input_output[j] - tmp_mean)*(L1_input_output[j] - tmp_mean);
        //}
        //tmp_std = sqrt(tmp_std / N_INPUT_EVO5_FRNN);
        //L5_output_y = L5_output_y*tmp_std + tmp_mean;
        // Error
        f_prcsn += (L6_predict_EVO5_FRNN - dataSet[i + 1]) * (L6_predict_EVO5_FRNN - dataSet[i + 1]);
    }

    //MatrixXd L4TL4 = L4DataAllSamples * L4DataAllSamples.transpose();
    //MatrixXd L4TY = L4DataAllSamples * O_targetOutput;
    //JacobiSVD<MatrixXd> svd(L4TL4, ComputeThinU | ComputeThinV);
    //MatrixXd L5Weights = svd.solve(L4TY);
    //MatrixXd O_predicted = L5Weights.transpose() * L4DataAllSamples;
    ////cout << "Before:" << endl << L4TL4 << endl;
    ////cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    ////cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
    ////cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
    ////MatrixXd tmpMat = L4TL4;
    ////tmpMat.setZero(L4TL4.rows(), L4TL4.cols());
    ////for (int i = 0; i < tmpMat.rows(); i++) {
    ////    tmpMat(i, i) = svd.singularValues()(i);
    ////}
    ////cout << "After:" << endl << svd.matrixU() * tmpMat * svd.matrixV().transpose() << endl;

    // precision objective
    //f_prcsn = 0.0;
    //for (int i = offset_data[N_INPUT_EVO5_FRNN - 1]; i < trainDataSize - 1; i++) {
    //  // Error
    //  f_prcsn += (O_predicted(0, i - offset_data[N_INPUT_EVO5_FRNN - 1]) - trainData[i + 1]) *
    //      (O_predicted(0, i - offset_data[N_INPUT_EVO5_FRNN - 1]) - trainData[i + 1]);
    //}
    f_prcsn = sqrt(f_prcsn / (dataLen - offset_data[N_INPUT_EVO5_FRNN - 1] - 1));

    return f_prcsn;
}

static double simplicity_EVO5_FRNN()
{
    //
    double f_simpl = 0.0;
    total_penalty_EVO5_FRNN = 0.0;
    //for (int i = 0; i < MAX_NUM_FUZZY_RULE_EVO5_FRNN; i++) {
    //  if (flag_fuzzy_rule[i])
    //      f_simpl++;
    //}
    //int flag_no_fuzzy_rule = 0;
    //if (f_simpl == 0)
    //  flag_no_fuzzy_rule = 1;
    int tmp1[MAX_NUM_FUZZY_RULE_EVO5_FRNN], tmp2[NUM_CLASS_EVO5_FRNN],
        tmp3[N_INPUT_EVO5_FRNN][MAX_NUM_CLUSTER_EVO5_FRNN];
    for(int i = 0; i < N_INPUT_EVO5_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_CLUSTER_EVO5_FRNN; j++) {
            tmp3[i][j] = 0;
        }
    }
    for(int i = 0; i < MAX_NUM_FUZZY_RULE_EVO5_FRNN; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < N_INPUT_EVO5_FRNN; j++) {
            if(flag_cluster_bin_rule_EVO5_FRNN[i][j]) {
                tmp1[i]++;
                int tmp_ind = flag_cluster_bin_rule_EVO5_FRNN[i][j] - 1;
                tmp3[j][tmp_ind]++;
            }
        }
        f_simpl += (double)tmp1[i] / N_INPUT_EVO5_FRNN;
    }
    for(int i = 0; i < NUM_CLASS_EVO5_FRNN; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < MAX_NUM_FUZZY_RULE_EVO5_FRNN; j++) {
            if(tmp1[j] && flag_fuzzy_bin_rough_EVO5_FRNN[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl += (double)tmp2[i] / MAX_NUM_FUZZY_RULE_EVO5_FRNN;
    }
    f_simpl /= (MAX_NUM_FUZZY_RULE_EVO5_FRNN + NUM_CLASS_EVO5_FRNN);
    //f_simpl /= MAX_NUM_FUZZY_RULE_EVO5_FRNN;

    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    int tmp_sum = 0;
    for(int i = 0; i < MAX_NUM_FUZZY_RULE_EVO5_FRNN; i++) {
        tmp_sum += tmp1[i];
    }
    if(tmp_sum == 0.0) {
        total_penalty_EVO5_FRNN += penaltyVal_EVO5_FRNN;
    }
    //tmp_sum = 0;
    //for(int i = 0; i < NUM_CLASS_EVO5_FRNN; i++) {
    //    tmp_sum += tmp2[i];
    //}
    //if(tmp_sum == 0.0) {
    //    f_prcsn += 1e6;
    //    f_simpl += 1e6;
    //}
    for(int i = 0; i < NUM_CLASS_EVO5_FRNN; i++) {
        if(tmp2[i] == 0.0) {
            total_penalty_EVO5_FRNN += penaltyVal_EVO5_FRNN;
        }
    }

    return f_simpl;
}