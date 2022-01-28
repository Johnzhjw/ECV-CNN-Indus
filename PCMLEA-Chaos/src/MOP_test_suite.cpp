#include "MOP_test_suite.h"

//////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include "assert.h"
#include <string.h>

//////////////////////////////////////////////////////////////////////////

// #define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029
#define MYSIGN(x) ((x)>0?1.0:-1.0)

int EMO_test_suite_nvar = 0, EMO_test_suite_nobj = 0; //  the number of variables and objectives
int EMO_test_suite_position_parameters = 0;
char EMO_test_suite_testInstName[1024];

//////////////////////////////////////////////////////////////////////////
void(*pointer_InitPara)(char*, int, int, int) = NULL;
void (*pointer_SetLimits)(double*, double*, int) = NULL;
int (*pointer_CheckLimits)(double*, int) = NULL;
void (*pointer_Fitness)(double*, double*, double*, int, int) = NULL;
void (*pointer_adjust_constraints)(double, double) = NULL;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Implementation
//////////////////////////////////////////////////////////////////////////
void EMO_initialization(char* pro, int& nobj, int& ndim, int curN, int numN, int my_rank, int para_1, int para_2, int para_3)
{
    strcpy(EMO_test_suite_testInstName, pro);

    if(!strncmp(pro, "LSMOP", 5)) {
        EMO_test_suite_nobj = nobj;
        EMO_test_suite_position_parameters = 2 * (nobj - 1);
        LSMOP_initialization(pro, nobj);
        ndim = LSMOP_D;
    } else if(!strcmp(pro, "RS")) {
        loadData();
        //ndim = 943 * 10;
        ndim = DIM_RS;
        nobj = 2;
    } else if(!strcmp(pro, "IWSN")) {
        ndim = DIM_IWSN;
        nobj = IWSNOBJ;
    } else if(!strcmp(pro, "HDSN")) {
        ndim = DIM_HDSN;
        nobj = HDSNOBJ;
    } else if(!strcmp(pro, "HDSN_URBAN")) {
        ndim = DIM_HDSN_URBAN;
        nobj = HDSNOBJ_URBAN;
    } else if(!strcmp(pro, "RecSys_SmartCity")) {
        Initialize_data_RS_SC(curN, numN);
        ndim = DIM_RS_SC;
        nobj = DIM_OBJ_RS_SC;
    } else if(!strcmp(pro, "WDCN")) {
        ndim = DIM_WDCN;
        nobj = WDCNOBJ;
    } else if(!strcmp(pro, "IWSN_S_1F")) {
        ndim = DIM_IWSN_S_1F;
        nobj = DIM_OBJ_IWSN_S_1F;
    } else if(!strncmp(pro, "FeatureSelection_", 17)) {  //Classify
        Initialize_ClassifierFunc(pro, curN, numN);
        nobj = NOBJ_CLASSIFY;
        ndim = DIM_ClassifierFunc;
    } else if(!strncmp(pro, "FeatureSelectionTREE_", 21)) {
        f_Initialize_ClassifierTreeFunc(pro, curN, numN);
        nobj = NOBJ_CLASSIFY_TREE;
        ndim = m_DIM_ClassifierTreeFunc;
    } else if(!strncmp(pro, "FINANCE", 7)) {
        Initialize_data_LeNet(curN, numN, para_1, para_2, para_3);
        nobj = OBJ_LeNet;
        ndim = DIM_LeNet;
    } else if(!strncmp(pro, "ENSEMBLE_FINANCE", 16)) {
        Initialize_data_LeNet(curN, numN, para_1, para_2, para_3);
        char tmp[128];
        sprintf(tmp, "%s", &pro[9]);
        load_storedCNN(tmp);
        nobj = OBJ_LeNet_ENSEMBLE;
        ndim = NDIM_LeNet_ENSEMBLE;
    } else if(!strncmp(pro, "EVO1_FRNN", 9)) {
        Initialize_data_EVO1_FRNN(curN, numN, para_1, para_2, para_3);
        nobj = DIM_OBJ_EVO1_FRNN;
        ndim = DIM_EVO1_FRNN;
    } else if(!strncmp(pro, "EVO2_FRNN", 9)) {
        Initialize_data_EVO2_FRNN(curN, numN, para_1, para_2, para_3);
        nobj = DIM_OBJ_EVO2_FRNN;
        ndim = DIM_EVO2_FRNN;
    } else if(!strncmp(pro, "EVO3_FRNN", 9)) {
        Initialize_data_EVO3_FRNN(curN, numN, para_1, para_2, para_3);
        nobj = DIM_OBJ_EVO3_FRNN;
        ndim = DIM_EVO3_FRNN;
    } else if(!strncmp(pro, "EVO4_FRNN", 9)) {
        Initialize_data_EVO4_FRNN(curN, numN, para_1, para_2, para_3);
        nobj = DIM_OBJ_EVO4_FRNN;
        ndim = DIM_EVO4_FRNN;
    } else if(!strncmp(pro, "EVO5_FRNN", 9)) {
        Initialize_data_EVO5_FRNN(curN, numN, para_1, para_2, para_3);
        nobj = DIM_OBJ_EVO5_FRNN;
        ndim = DIM_EVO5_FRNN;
    } else if(!strncmp(pro, "evoFRNN_Predict_", 16) ||
              !strncmp(pro, "evoGFRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFRNN_Predict_", 17) ||
              !strncmp(pro, "evoFGRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoGFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFRNN_Predict_", 17) ||
              !strncmp(pro, "evoBGFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFGRNN_Predict_", 19) ||
              !strncmp(pro, "evoBGFGRNN_Predict_", 19)) {
        Initialize_MOP_Predict_FRNN(pro, curN, numN, para_1, para_2, para_3, my_rank);
        nobj = NOBJ_MOP_Predict_FRNN;
        ndim = NDIM_MOP_Predict_FRNN;
    } else if(!strncmp(pro, "evoDeepFRNN_Predict_", 20) ||
              !strncmp(pro, "evoDeepGFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepFGRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepGFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepBGFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFGRNN_Predict_", 23) ||
              !strncmp(pro, "evoDeepBGFGRNN_Predict_", 23)) {
        Initialize_MOP_Predict_DFRNN(pro, curN, numN, para_1, para_2, para_3, my_rank);
        nobj = NOBJ_MOP_Predict_DFRNN;
        ndim = NDIM_MOP_Predict_DFRNN;
    } else if(!strncmp(pro, "ARRANGE2D", 9)) {
        Initialize_data_ARRANGE2D(curN, numN);
        nobj = NOBJ_ARRANGE2D;
        ndim = NDIM_ARRANGE2D;
    } else if(!strcmp(pro, "Classify_CNN_Indus")) {
        Initialize_data_Classify_CNN(curN, numN, para_1, para_2, para_3);
        nobj = NOBJ_Classify_CNN_Indus;
        ndim = NDIM_Classify_CNN_Indus;
    } else if(!strcmp(pro, "Classify_CNN_Indus_BP")) {
        Initialize_data_Classify_CNN(curN, numN, para_1, para_2, para_3);
        nobj = NOBJ_Classify_CNN_Indus_BP;
        ndim = NDIM_Classify_CNN_Indus_BP;
    } else if(!strcmp(pro, "Classify_NN_Indus")) {
        Initialize_data_Classify_NN(curN, numN, para_1, para_2, para_3);
        nobj = NOBJ_Classify_NN_Indus;
        ndim = NDIM_Classify_NN_Indus;
    } else if(!strcmp(pro, "Classify_NN_Indus_BP")) {
        Initialize_data_Classify_NN(curN, numN, para_1, para_2, para_3);
        nobj = NOBJ_Classify_NN_Indus_BP;
        ndim = NDIM_Classify_NN_Indus_BP;
    } else if(!strcmp(pro, "IntrusionDetection_FRNN_Classify")) {
        Initialize_IntrusionDetection_FRNN_Classify(curN, numN);
        ndim = NDIM_IntrusionDetection_FRNN_Classify;
        nobj = NOBJ_IntrusionDetection_FRNN_Classify;
    } else if(!strcmp(pro, "ActivityDetection_FRNN_Classify")) {
        Initialize_ActivityDetection_FRNN_Classify(curN, numN, my_rank);
        ndim = NDIM_ActivityDetection_FRNN_Classify;
        nobj = NOBJ_ActivityDetection_FRNN_Classify;
    } else if(!strcmp(pro, "evoCNN_MNIST_Classify")) {
        Initialize_evoCNN_Classify(curN, numN);
        ndim = NDIM_evoCNN_Classify;
        nobj = NOBJ_evoCNN_Classify;
    } else if(!strcmp(pro, "evoCNN_MNIST_Classify_BP")) {
        Initialize_evoCNN_Classify(curN, numN);
        ndim = NDIM_evoCNN_Classify_BP;
        nobj = NOBJ_evoCNN_Classify_BP;
    } else if(!strcmp(pro, "evoCFRNN_Classify")) {
        Initialize_evoCFRNN_Classify(curN, numN);
        ndim = NDIM_evoCFRNN_Classify;
        nobj = NOBJ_evoCFRNN_Classify;
    } else if(!strcmp(pro, "Classify_CFRNN_Indus") ||
              !strcmp(pro, "Classify_CFRNN_MNIST") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST") ||
              !strcmp(pro, "Classify_CFRNN_MNIST_FM") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST_FM")) {
        Initialize_Classify_CFRNN(pro, curN, numN, my_rank);
        ndim = NDIM_Classify_CFRNN;
        nobj = NOBJ_Classify_CFRNN;
    } else if(!strcmp(pro, "EdgeComputation")) {
        Initialize_data_EdgeComputation(curN, numN);
        ndim = NDIM_EdgeComputation;
        nobj = NOBJ_EdgeComputation;
    } else if(strstr(pro, "evoMobileSink")) {
        Initialize_MOP_Mob_Sink(pro, curN, numN, my_rank);
        ndim = NDIM_MOP_Mob_Sink;
        nobj = NOBJ_MOP_Mob_Sink;
    }

    //////////////////////////////////////////////////////////////////////////
    EMO_test_suite_nvar = ndim;
    EMO_test_suite_nobj = nobj;
    //////////////////////////////////////////////////////////////////////////
    // function pointers
    pointer_InitPara = NULL;
    pointer_SetLimits = NULL;
    pointer_CheckLimits = NULL;
    pointer_Fitness = NULL;
    pointer_adjust_constraints = NULL;
    if(!strcmp(pro, "IWSN")) {
        pointer_SetLimits = SetLimits_IWSN;
        pointer_CheckLimits = CheckLimits_IWSN;
        pointer_Fitness = Fitness_IWSN;
    } else if(!strcmp(pro, "HDSN")) {
        pointer_SetLimits = SetLimits_HDSN;
        pointer_CheckLimits = CheckLimits_HDSN;
        pointer_Fitness = Fitness_HDSN;
    } else if(!strcmp(pro, "HDSN_URBAN")) {
        pointer_SetLimits = SetLimits_HDSN_URBAN;
        pointer_CheckLimits = CheckLimits_HDSN_URBAN;
        pointer_Fitness = Fitness_HDSN_URBAN;
    } else if(!strcmp(pro, "WDCN")) {
        pointer_SetLimits = SetLimits_WDCN;
        pointer_CheckLimits = CheckLimits_WDCN;
        pointer_Fitness = Fitness_WDCN;
    } else if(!strcmp(pro, "IWSN_S_1F")) {
        pointer_SetLimits = SetLimits_IWSN_S_1F;
        pointer_CheckLimits = CheckLimits_IWSN_S_1F;
        pointer_Fitness = Fitness_IWSN_S_1F;
        //printf("%d ", pointer_adjust_constraints);
        pointer_adjust_constraints = adjust_constraints_IWSN_S_1F;
        //printf("%d\n", pointer_adjust_constraints);
    } else if(!strcmp(pro, "RS")) {
        pointer_SetLimits = SetLimitsRS;
        pointer_CheckLimits = CheckLimitsRS;
        pointer_Fitness = getFitnessRS;
    } else if(!strcmp(pro, "RecSys_SmartCity")) {
        pointer_SetLimits = SetLimits_RS_SC;
        pointer_CheckLimits = CheckLimits_RS_SC;
        pointer_Fitness = Fitness_RS_SC;
    } else if(!strncmp(pro, "FeatureSelection_", 17)) {
        pointer_SetLimits = SetLimits_ClassifierFunc;
        pointer_CheckLimits = CheckLimits_ClassifierFunc;
        pointer_Fitness =
            Fitness_ClassifierFunc; //////////////////////////////////////////////////////////////////////////
    } else if(!strncmp(pro, "FeatureSelectionTREE_", 21)) {
        pointer_SetLimits = f_SetLimits_ClassifierTreeFunc;
        pointer_CheckLimits = f_CheckLimits_ClassifierTreeFunc;
        pointer_Fitness = f_Fitness_ClassifierTreeFunc;
    } else if(!strncmp(pro, "FINANCE", 7)) {
        pointer_SetLimits = SetLimits_LeNet;
        pointer_CheckLimits = CheckLimits_LeNet;
        pointer_Fitness = Fitness_LeNet;
    } else if(!strncmp(pro, "ENSEMBLE_FINANCE", 16)) {
        pointer_SetLimits = SetLimits_LeNet_ensemble;
        pointer_CheckLimits = CheckLimits_LeNet_ensemble;
        pointer_Fitness = Fitness_LeNet_ensemble;
    } else if(!strncmp(pro, "EVO1_FRNN", 9)) {
        pointer_SetLimits = SetLimits_EVO1_FRNN;
        pointer_CheckLimits = CheckLimits_EVO1_FRNN;
        pointer_Fitness = Fitness_EVO1_FRNN;
    } else if(!strncmp(pro, "EVO2_FRNN", 9)) {
        pointer_SetLimits = SetLimits_EVO2_FRNN;
        pointer_CheckLimits = CheckLimits_EVO2_FRNN;
        pointer_Fitness = Fitness_EVO2_FRNN;
    } else if(!strncmp(pro, "EVO3_FRNN", 9)) {
        pointer_SetLimits = SetLimits_EVO3_FRNN;
        pointer_CheckLimits = CheckLimits_EVO3_FRNN;
        pointer_Fitness = Fitness_EVO3_FRNN;
    } else if(!strncmp(pro, "EVO4_FRNN", 9)) {
        pointer_SetLimits = SetLimits_EVO4_FRNN;
        pointer_CheckLimits = CheckLimits_EVO4_FRNN;
        pointer_Fitness = Fitness_EVO4_FRNN;
    } else if(!strncmp(pro, "EVO5_FRNN", 9)) {
        pointer_SetLimits = SetLimits_EVO5_FRNN;
        pointer_CheckLimits = CheckLimits_EVO5_FRNN;
        pointer_Fitness = Fitness_EVO5_FRNN;
    } else if(!strncmp(pro, "evoFRNN_Predict_", 16) ||
              !strncmp(pro, "evoGFRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFRNN_Predict_", 17) ||
              !strncmp(pro, "evoFGRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoGFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFRNN_Predict_", 17) ||
              !strncmp(pro, "evoBGFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFGRNN_Predict_", 19) ||
              !strncmp(pro, "evoBGFGRNN_Predict_", 19)) {
        pointer_SetLimits = SetLimits_MOP_Predict_FRNN;
        pointer_CheckLimits = CheckLimits_MOP_Predict_FRNN;
        pointer_Fitness = Fitness_MOP_Predict_FRNN;
    } else if(!strncmp(pro, "evoDeepFRNN_Predict_", 20) ||
              !strncmp(pro, "evoDeepGFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepFGRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepGFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepBGFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFGRNN_Predict_", 23) ||
              !strncmp(pro, "evoDeepBGFGRNN_Predict_", 23)) {
        pointer_SetLimits = SetLimits_MOP_Predict_DFRNN;
        pointer_CheckLimits = CheckLimits_MOP_Predict_DFRNN;
        pointer_Fitness = Fitness_MOP_Predict_DFRNN;
    } else if(!strncmp(pro, "ARRANGE2D", 9)) {
        pointer_SetLimits = SetLimits_ARRANGE2D;
        pointer_CheckLimits = CheckLimits_ARRANGE2D;
        pointer_Fitness = Fitness_ARRANGE2D;
    } else if(!strcmp(pro, "Classify_CNN_Indus")) {
        pointer_SetLimits = SetLimits_Classify_CNN;
        pointer_CheckLimits = CheckLimits_Classify_CNN;
        pointer_Fitness = Fitness_Classify_CNN;
    } else if(!strcmp(pro, "Classify_CNN_Indus_BP")) {
        pointer_SetLimits = SetLimits_Classify_CNN;
        pointer_CheckLimits = CheckLimits_Classify_CNN;
        pointer_Fitness = Fitness_Classify_CNN_Indus_BP;
    } else if(!strcmp(pro, "Classify_NN_Indus")) {
        pointer_SetLimits = SetLimits_Classify_NN;
        pointer_CheckLimits = CheckLimits_Classify_NN;
        pointer_Fitness = Fitness_Classify_NN;
    } else if(!strcmp(pro, "Classify_NN_Indus_BP")) {
        pointer_SetLimits = SetLimits_Classify_NN;
        pointer_CheckLimits = CheckLimits_Classify_NN;
        pointer_Fitness = Fitness_Classify_NN_Indus_BP;
    } else if(!strcmp(pro, "DTLZ1")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz1;
    } else if(!strcmp(pro, "DTLZ2")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz2;
    } else if(!strcmp(pro, "DTLZ3")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz3;
    } else if(!strcmp(pro, "DTLZ4")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz4;
    } else if(!strcmp(pro, "DTLZ5")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz5;
    } else if(!strcmp(pro, "DTLZ6")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz6;
    } else if(!strcmp(pro, "DTLZ7")) {
        pointer_InitPara = InitPara_DTLZ;
        pointer_SetLimits = SetLimits_DTLZ;
        pointer_CheckLimits = CheckLimits_DTLZ;
        pointer_Fitness = dtlz7;
    } else if(!strcmp(pro, "UF1")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF1;
    } else if(!strcmp(pro, "UF2")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF2;
    } else if(!strcmp(pro, "UF3")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF3;
    } else if(!strcmp(pro, "UF4")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF4;
    } else if(!strcmp(pro, "UF5")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF5;
    } else if(!strcmp(pro, "UF6")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF6;
    } else if(!strcmp(pro, "UF7")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF7;
    } else if(!strcmp(pro, "UF8")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF8;
    } else if(!strcmp(pro, "UF9")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF9;
    } else if(!strcmp(pro, "UF10")) {
        pointer_InitPara = InitPara_UF;
        pointer_SetLimits = SetLimits_UF;
        pointer_CheckLimits = CheckLimits_UF;
        pointer_Fitness = UF10;
    } else if(!strcmp(pro, "WFG1")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG1;
    } else if(!strcmp(pro, "WFG2")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG2;
    } else if(!strcmp(pro, "WFG3")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG3;
    } else if(!strcmp(pro, "WFG4")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG4;
    } else if(!strcmp(pro, "WFG5")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG5;
    } else if(!strcmp(pro, "WFG6")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG6;
    } else if(!strcmp(pro, "WFG7")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG7;
    } else if(!strcmp(pro, "WFG8")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG8;
    } else if(!strcmp(pro, "WFG9")) {
        pointer_InitPara = InitPara_WFG;
        pointer_SetLimits = SetLimits_WFG;
        pointer_CheckLimits = CheckLimits_WFG;
        pointer_Fitness = WFG9;
    } else if(!strcmp(pro, "LSMOP1")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP1;
    } else if(!strcmp(pro, "LSMOP2")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP2;
    } else if(!strcmp(pro, "LSMOP3")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP3;
    } else if(!strcmp(pro, "LSMOP4")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP4;
    } else if(!strcmp(pro, "LSMOP5")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP5;
    } else if(!strcmp(pro, "LSMOP6")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP6;
    } else if(!strcmp(pro, "LSMOP7")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP7;
    } else if(!strcmp(pro, "LSMOP8")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP8;
    } else if(!strcmp(pro, "LSMOP9")) {
        pointer_InitPara = InitPara_LSMOP;
        pointer_SetLimits = SetLimits_LSMOP;
        pointer_CheckLimits = CheckLimits_LSMOP;
        pointer_Fitness = LSMOP9;
    } else if(!strcmp(pro, "SEC18_MaOP1")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP1;
    } else if(!strcmp(pro, "SEC18_MaOP2")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP2;
    } else if(!strcmp(pro, "SEC18_MaOP3")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP3;
    } else if(!strcmp(pro, "SEC18_MaOP4")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP4;
    } else if(!strcmp(pro, "SEC18_MaOP5")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP5;
    } else if(!strcmp(pro, "SEC18_MaOP6")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP6;
    } else if(!strcmp(pro, "SEC18_MaOP7")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP7;
    } else if(!strcmp(pro, "SEC18_MaOP8")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP8;
    } else if(!strcmp(pro, "SEC18_MaOP9")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP9;
    } else if(!strcmp(pro, "SEC18_MaOP10")) {
        pointer_InitPara = InitPara_SEC18_MaOP;
        pointer_SetLimits = SetLimits_SEC18_MaOP;
        pointer_CheckLimits = CheckLimits_SEC18_MaOP;
        pointer_Fitness = SEC18_MaOP10;
    } else if(!strcmp(pro, "IntrusionDetection_FRNN_Classify")) {
        pointer_SetLimits = SetLimits_IntrusionDetection_FRNN_Classify;
        pointer_CheckLimits = CheckLimits_IntrusionDetection_FRNN_Classify;
        pointer_Fitness = Fitness_IntrusionDetection_FRNN_Classify;
    } else if(!strcmp(pro, "ActivityDetection_FRNN_Classify")) {
        pointer_SetLimits = SetLimits_ActivityDetection_FRNN_Classify;
        pointer_CheckLimits = CheckLimits_ActivityDetection_FRNN_Classify;
        pointer_Fitness = Fitness_ActivityDetection_FRNN_Classify;
    } else if(!strcmp(pro, "evoCNN_MNIST_Classify")) {
        pointer_SetLimits = SetLimits_evoCNN_Classify;
        pointer_CheckLimits = CheckLimits_evoCNN_Classify;
        pointer_Fitness = Fitness_evoCNN_Classify;
    } else if(!strcmp(pro, "evoCNN_MNIST_Classify_BP")) {
        pointer_SetLimits = SetLimits_evoCNN_Classify;
        pointer_CheckLimits = CheckLimits_evoCNN_Classify;
        pointer_Fitness = Fitness_evoCNN_Classify_BP;
    } else if(!strcmp(pro, "evoCFRNN_Classify")) {
        pointer_SetLimits = SetLimits_evoCFRNN_Classify;
        pointer_CheckLimits = CheckLimits_evoCFRNN_Classify;
        pointer_Fitness = Fitness_evoCFRNN_Classify;
    } else if(!strcmp(pro, "Classify_CFRNN_Indus") ||
              !strcmp(pro, "Classify_CFRNN_MNIST") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST") ||
              !strcmp(pro, "Classify_CFRNN_MNIST_FM") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST_FM")) {
        pointer_SetLimits = SetLimits_Classify_CFRNN;
        pointer_CheckLimits = CheckLimits_Classify_CFRNN;
        pointer_Fitness = Fitness_Classify_CFRNN;
    } else if(!strcmp(pro, "EdgeComputation")) {
        pointer_SetLimits = SetLimits_EdgeComputation;
        pointer_CheckLimits = CheckLimits_EdgeComputation;
        pointer_Fitness = Fitness_EdgeComputation;
    } else if(strstr(pro, "evoMobileSink")) {
        pointer_SetLimits = SetLimits_MOP_Mob_Sink;
        pointer_CheckLimits = CheckLimits_MOP_Mob_Sink;
        pointer_Fitness = Fitness_MOP_Mob_Sink;
    }
    //////////////////////////////////////////////////////////////////////////
    else {
        fprintf(stderr, "Unknown problem %s\n", pro);
        exit(-1);
    }
    //////////////////////////////////////////////////////////////////////////
    if(pointer_InitPara)
        pointer_InitPara(EMO_test_suite_testInstName, EMO_test_suite_nobj, EMO_test_suite_nvar, EMO_test_suite_position_parameters);

    return;
}

void EMO_finalization(char* pro)
{
    if(!strncmp(pro, "LSMOP", 5)) {
        LSMOP_finalization();
    } else if(!strcmp(pro, "RS")) {
        freeData();
    } else if(!strncmp(pro, "FeatureSelection_", 17)) {
        freeMemoryCLASS();
    } else if(!strncmp(pro, "FeatureSelectionTREE_", 21)) {
        f_freeMemoryTreeCLASS();
    } else if(!strncmp(pro, "FINANCE", 7)) {
        Finalize_LeNet();
    } else if(!strncmp(pro, "ENSEMBLE_FINANCE", 16)) {
        Finalize_LeNet_ensemble();
    } else if(!strcmp(pro, "Classify_CNN_Indus") ||
              !strcmp(pro, "Classify_CNN_Indus_BP")) {
        Finalize_Classify_CNN();
    } else if(!strcmp(pro, "Classify_NN_Indus") ||
              !strcmp(pro, "Classify_NN_Indus_BP")) {
        Finalize_Classify_NN();
    } else if(!strcmp(pro, "IntrusionDetection_FRNN_Classify")) {
        Finalize_IntrusionDetection_FRNN_Classify();
    } else if(!strcmp(pro, "ActivityDetection_FRNN_Classify")) {
        Finalize_ActivityDetection_FRNN_Classify();
    } else if(!strcmp(pro, "evoCNN_MNIST_Classify") ||
              !strcmp(pro, "evoCNN_MNIST_Classify_BP")) {
        Finalize_evoCNN_Classify();
    } else if(!strcmp(pro, "Classify_CFRNN_Indus") ||
              !strcmp(pro, "Classify_CFRNN_MNIST") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST") ||
              !strcmp(pro, "Classify_CFRNN_MNIST_FM") ||
              !strcmp(pro, "Classify_CFRNN_FashionMNIST_FM")) {
        Finalize_Classify_CFRNN();
    } else if(!strcmp(pro, "EdgeComputation")) {
        Finalize_EdgeComputation();
    } else if(!strncmp(pro, "evoFRNN_Predict_", 16) ||
              !strncmp(pro, "evoGFRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFRNN_Predict_", 17) ||
              !strncmp(pro, "evoFGRNN_Predict_", 17) ||
              !strncmp(pro, "evoDFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoGFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFRNN_Predict_", 17) ||
              !strncmp(pro, "evoBGFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFRNN_Predict_", 18) ||
              !strncmp(pro, "evoBFGRNN_Predict_", 18) ||
              !strncmp(pro, "evoBDFGRNN_Predict_", 19) ||
              !strncmp(pro, "evoBGFGRNN_Predict_", 19)) {
        Finalize_MOP_Predict_FRNN();
    } else if(!strncmp(pro, "evoDeepFRNN_Predict_", 20) ||
              !strncmp(pro, "evoDeepGFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepFGRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepDFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepGFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFRNN_Predict_", 21) ||
              !strncmp(pro, "evoDeepBGFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBFGRNN_Predict_", 22) ||
              !strncmp(pro, "evoDeepBDFGRNN_Predict_", 23) ||
              !strncmp(pro, "evoDeepBGFGRNN_Predict_", 23)) {
        Finalize_MOP_Predict_DFRNN();
    } else if(strstr(pro, "evoMobileSink")) {
        Finalize_MOP_Mob_Sink();
    }

    pointer_InitPara = NULL;
    pointer_SetLimits = NULL;
    pointer_CheckLimits = NULL;
    pointer_Fitness = NULL;
    pointer_adjust_constraints = NULL;

    return;
}

void EMO_setLimits(char* pro, double* minLimit, double* maxLimit, int dim)
{
    pointer_SetLimits(minLimit, maxLimit, dim);
    return;
}

int checkLimits(char* pro, double* x, int dim)
{
    for(int i = 0; i < dim; i++) {
        if(CHECK_INVALID(x[i]))
            return false;
    }
    return pointer_CheckLimits(x, dim);
}

void EMO_adjust_constraint_penalty(double cur_iter, double max_iter)
{
    if(pointer_adjust_constraints) {
        pointer_adjust_constraints(cur_iter, max_iter);
    }
    return;
}

void EMO_evaluate_problems(char* pro, double* xreal, double* obj, int dim, int nx, int nobj)
{
    EMO_test_suite_nobj = nobj;
    EMO_test_suite_nvar = dim;
    EMO_test_suite_position_parameters = 2 * (EMO_test_suite_nobj - 1);
    //if(EMO_nobj==3)	EMO_position_parameters=4;
    //else if(EMO_nobj==6) EMO_position_parameters=10;
    //else if(EMO_nobj==8) EMO_position_parameters=7;
    //else if(EMO_nobj==10) EMO_position_parameters=9;

    for(int i = 0; i < nx; i++) {
        assert(checkLimits(pro, &xreal[i * dim], dim));
        //assert(pointer_checkLimits(&xreal[i * dim], dim));
        pointer_Fitness(&xreal[i * dim], &obj[i * EMO_test_suite_nobj], NULL, dim, EMO_test_suite_nobj);
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//#include <time.h>

//#define READ_PS_FILE
//#define PRINT_STATISTICS_MOP_Predict_FRNN
//#define TAG_UPDATE_testInstance_txt

//#define DEBUG_TEST_SUITE

#ifdef DEBUG_TEST_SUITE

static int mat_inds[88][3];

static void readData_general_test_suite(const char* fname)
{
    FILE* fpt;
    if((fpt = fopen(fname, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }
    //
    const int MAX_ATTR_NUM = 100;
    int n_row;
    int n_col;
    char tmp_delim[] = " ,\t\r\n";
    int max_buf_size = 100 * MAX_ATTR_NUM + 1;
    char* buf = (char*)malloc(max_buf_size * sizeof(char));
    char* p;
    int tmp_cnt;
    int elem_int;
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
            n_row = elem_int;
        } else if(tmp_cnt == 1) {
            n_col = elem_int;
        } else {
            if(tmp_cnt == 2) {
                printf("\n%s(%d):too many data...\n", __FILE__, __LINE__);
                exit(1002);
            }
        }
        tmp_cnt++;
    }
    //get data
    int seq = 0;
    for(seq = 0; seq < n_row; seq++) {
        if(fgets(buf, max_buf_size, fpt) == NULL) {
            printf("%s(%d): No  line\n", __FILE__, __LINE__);
            exit(-1);
        }
        tmp_cnt = 0;
        for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
            if(sscanf(p, "%d", &elem_int) != 1) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2004);
            }
            mat_inds[seq][tmp_cnt] = elem_int;
            tmp_cnt++;
        }
        if(n_col != tmp_cnt) {
            printf("\n%s(%d): the number of data items is not consistant (%d)(%d != %d), exiting ...\n",
                   __FILE__, __LINE__, seq, n_col, tmp_cnt);
            exit(2005);
        }
    }
    //
    free(buf);
    fclose(fpt);
}

int main(int argc, char** argv)
{
    int mpi_rank;
    int mpi_size;
    char my_name[1024];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Get_processor_name(my_name, &name_len);

    int fun_start_num = 475; // the start number of test function, you can find the explanation in testInstance.txt
    int fun_end_num = 617;   // the end number of test function
#ifdef TAG_UPDATE_testInstance_txt
    fun_start_num = 1;
    fun_end_num = 671;
#endif
    int num_run = 20;         // the number of running test instrct_grp_ana_vals.Dependently
    int iRun;

    int seq;
    char prob[256];
    int ndim;
    int nobj;
    int para_1 = 0;
    int para_2 = 0;
    int para_3 = 0;

    char tmp_str[1024];
    FILE* readf = fopen("testInstance.txt", "r");
#ifdef TAG_UPDATE_testInstance_txt
    FILE* wrtef = fopen("testInst.txt", "w");
#endif
    system("mkdir -p tmpFile");
#ifdef READ_PS_FILE
    readData_general_test_suite("table_inds.txt");
#endif
    int iPro;
    for(iPro = 1; iPro < fun_start_num; iPro++) {
        if(!fgets(tmp_str, sizeof(tmp_str), readf)) {
            if(mpi_rank == 0)
                printf("%s(%d): Reading file error - no more line, exiting...\n", __FILE__, __LINE__);
            exit(-1);
        }
    }
    for(iPro = fun_start_num; iPro <= fun_end_num; iPro++) {
        int nb;
        if(fgets(tmp_str, sizeof(tmp_str), readf)) {
            nb = sscanf(tmp_str, "%d %s %d %d %d %d %d", &seq, prob, &ndim, &nobj, &para_1, &para_2, &para_3);
            if(nb < 4) {
                if(mpi_rank == 0)
                    printf("%s(%d): Reading file error - no more para, exiting...\n", __FILE__, __LINE__);
                exit(-2);
            }
        } else {
            if(mpi_rank == 0)
                printf("%s(%d): Reading file error - no more line, exiting...\n", __FILE__, __LINE__);
            exit(-1);
        }

#ifdef TAG_UPDATE_testInstance_txt
        if(nb == 4)
            fprintf(wrtef, "%d\t%s\t%d\t%d\n", iPro, prob, ndim, nobj);
        else
            fprintf(wrtef, "%d\t%s\t%d\t%d\t%d\t%d\t%d\n", iPro, prob, ndim, nobj, para_1, para_2, para_3);
        continue;
#endif
        //
        if(!strncmp(prob, "evoFRNN_Predict_", 16) ||
           !strncmp(prob, "evoGFRNN_Predict_", 17) ||
           !strncmp(prob, "evoDFRNN_Predict_", 17) ||
           !strncmp(prob, "evoFGRNN_Predict_", 17) ||
           !strncmp(prob, "evoDFGRNN_Predict_", 18) ||
           !strncmp(prob, "evoGFGRNN_Predict_", 18) ||
           !strncmp(prob, "evoBFRNN_Predict_", 17) ||
           !strncmp(prob, "evoBGFRNN_Predict_", 18) ||
           !strncmp(prob, "evoBDFRNN_Predict_", 18) ||
           !strncmp(prob, "evoBFGRNN_Predict_", 18) ||
           !strncmp(prob, "evoBDFGRNN_Predict_", 19) ||
           !strncmp(prob, "evoBGFGRNN_Predict_", 19)) {
            num_run = 10;
        }
        if(!strncmp(prob, "evoDeepFRNN_Predict_", 20) ||
           !strncmp(prob, "evoDeepGFRNN_Predict_", 21) ||
           !strncmp(prob, "evoDeepDFRNN_Predict_", 21) ||
           !strncmp(prob, "evoDeepFGRNN_Predict_", 21) ||
           !strncmp(prob, "evoDeepDFGRNN_Predict_", 22) ||
           !strncmp(prob, "evoDeepGFGRNN_Predict_", 22) ||
           !strncmp(prob, "evoDeepBFRNN_Predict_", 21) ||
           !strncmp(prob, "evoDeepBGFRNN_Predict_", 22) ||
           !strncmp(prob, "evoDeepBDFRNN_Predict_", 22) ||
           !strncmp(prob, "evoDeepBFGRNN_Predict_", 22) ||
           !strncmp(prob, "evoDeepBDFGRNN_Predict_", 23) ||
           !strncmp(prob, "evoDeepBGFGRNN_Predict_", 23)) {
            num_run = 10;
        }
        for(iRun = 1; iRun <= num_run; iRun++) {
#ifdef READ_PS_FILE
            int cur_row = iPro - fun_start_num;
            if(mat_inds[cur_row][0] != iRun) continue;
#endif

            EMO_initialization(prob, nobj, ndim, iRun - 1, num_run, mpi_rank, para_1, para_2, para_3);

            printf("\n--   run %d   --\n\n\n-- PROBLEM %s\n--  variables: %d\n--  objectives: %d\n\n",
                   iRun, prob, ndim, nobj);

            double* individual = (double*)calloc(ndim, sizeof(double));
            double* fitnessess = (double*)calloc(nobj, sizeof(double));

            double* minLimit = (double*)calloc(ndim, sizeof(double));
            double* maxLimit = (double*)calloc(ndim, sizeof(double));

            EMO_setLimits(prob, minLimit, maxLimit, ndim);

            srand((unsigned int)(iRun + 1));

            int npop = 120;
            if(nobj == 2) npop = 100;
            if(nobj == 3) npop = 120;
            ////////////////////////////////////////////////////////////////////////
            //npop = 120;
            int flag_evo_pred = 0;
            if(!strncmp(prob, "evoFRNN_Predict_", 16) ||
               !strncmp(prob, "evoGFRNN_Predict_", 17) ||
               !strncmp(prob, "evoDFRNN_Predict_", 17) ||
               !strncmp(prob, "evoFGRNN_Predict_", 17) ||
               !strncmp(prob, "evoDFGRNN_Predict_", 18) ||
               !strncmp(prob, "evoGFGRNN_Predict_", 18) ||
               !strncmp(prob, "evoBFRNN_Predict_", 17) ||
               !strncmp(prob, "evoBGFRNN_Predict_", 18) ||
               !strncmp(prob, "evoBDFRNN_Predict_", 18) ||
               !strncmp(prob, "evoBFGRNN_Predict_", 18) ||
               !strncmp(prob, "evoBDFGRNN_Predict_", 19) ||
               !strncmp(prob, "evoBGFGRNN_Predict_", 19)) {
                flag_evo_pred = 1;
                npop = 12;
            }
            if(!strncmp(prob, "evoDeepFRNN_Predict_", 20) ||
               !strncmp(prob, "evoDeepGFRNN_Predict_", 21) ||
               !strncmp(prob, "evoDeepDFRNN_Predict_", 21) ||
               !strncmp(prob, "evoDeepFGRNN_Predict_", 21) ||
               !strncmp(prob, "evoDeepDFGRNN_Predict_", 22) ||
               !strncmp(prob, "evoDeepGFGRNN_Predict_", 22) ||
               !strncmp(prob, "evoDeepBFRNN_Predict_", 21) ||
               !strncmp(prob, "evoDeepBGFRNN_Predict_", 22) ||
               !strncmp(prob, "evoDeepBDFRNN_Predict_", 22) ||
               !strncmp(prob, "evoDeepBFGRNN_Predict_", 22) ||
               !strncmp(prob, "evoDeepBDFGRNN_Predict_", 23) ||
               !strncmp(prob, "evoDeepBGFGRNN_Predict_", 23)) {
                flag_evo_pred = 2;
                npop = 12;
            }

#ifdef READ_PS_FILE
            char tmp_fn[1024];
            if(mat_inds[cur_row][1] == 26)
                sprintf(tmp_fn,
                        "../trial08/PS/DPCCMOLSIA_MP_III_FUN_%s_OBJ%d_VAR%d_MPI96_RUN%d", prob, nobj, ndim,
                        mat_inds[cur_row][0]);
            else
                sprintf(tmp_fn,
                        "../trial08/PS/trace/DPCCMOLSIA_MP_III_FUN_%s_OBJ%d_VAR%d_MPI96_RUN%d_key%d", prob, nobj, ndim,
                        mat_inds[cur_row][0], mat_inds[cur_row][1] - 1);
            FILE* fpt = fopen(tmp_fn, "r");
            if(!fpt) {
                printf("%s(%d): Open file error ! Exiting ...\n", __FILE__, __LINE__);
                exit(-1375);
            }
            npop = 12;
#endif
            //////////////////////////////////////////////////////////////////////////
            clock_t start = clock();
            for(int i = 0; i < npop; i++) {
#ifdef READ_PS_FILE
                if(mat_inds[cur_row][2] - 1 != i) continue;
#endif
                for(int a = 0; a < ndim; a++) {
#ifdef READ_PS_FILE
                    fscanf(fpt, "%lf", &individual[a]);
#else
                    int tmp = rand();
                    individual[a] = minLimit[a] + tmp / (RAND_MAX + 0.0) * (maxLimit[a] - minLimit[a]);
#endif
                }
                //
                EMO_evaluate_problems(prob, individual, fitnessess, ndim, 1, nobj);
                printf("iRun %d - ID: %d\n", iRun, i + 1);
                for(int a = 0; a < nobj; a++) {
                    printf("%.6lf ", fitnessess[a]);
                }
                printf("\n");

                char tfn[1024];
                FILE* tfp = NULL;
                sprintf(tfn, "tmpFile/%d_FUN_%s_RUN_%d.tmp", seq, prob, iRun);
                tfp = fopen(tfn, "a");
                for(int a = 0; a < nobj; a++) {
                    fprintf(tfp, "%e ", fitnessess[a]);
                }
                fprintf(tfp, "\n");
                fclose(tfp);
                //
                int tmp_flag_test = 0;
                if(flag_evo_pred) {
                    if(flag_evo_pred == 1)
                        Fitness_MOP_Predict_FRNN_test(individual, fitnessess);
                    else
                        Fitness_MOP_Predict_DFRNN_test(individual, fitnessess);
#ifdef PRINT_STATISTICS_MOP_Predict_FRNN
                    sprintf(tfn, "tmpFile/%d_FUN_%s_RUN_%d_statistics.tmp", seq, prob, mat_inds[cur_row][0]);
                    tfp = fopen(tfn, "w");
                    if(flag_evo_pred == 1)
                        statistics_MOP_Predict_FRNN(tfp);
                    else
                        statistics_MOP_Predict_DFRNN(tfp);
                    fclose(tfp);
#endif
                    tmp_flag_test = 1;
                } else if(strstr(prob, "evoMobileSink")) {
                    Fitness_MOP_Mob_Sink_test(individual, fitnessess);
                    tmp_flag_test = 1;
                }
                if(tmp_flag_test) {
                    for(int a = 0; a < nobj; a++) {
                        printf("%.6lf ", fitnessess[a]);
                    }
                    printf("\n");
                    sprintf(tfn, "tmpFile/%d_FUN_%s_RUN_%d_test.tmp", seq, prob, iRun);
                    tfp = fopen(tfn, "a");
                    for(int a = 0; a < nobj; a++) {
                        fprintf(tfp, "%e ", fitnessess[a]);
                    }
                    fprintf(tfp, "\n");
                    fclose(tfp);
                }
            }
            clock_t finish = clock();
            printf("Time elapsed in ms: %ld / %ld\n", finish - start, CLOCKS_PER_SEC);

            free(individual);
            free(fitnessess);
            free(minLimit);
            free(maxLimit);
#ifdef READ_PS_FILE
            if(fpt)
                fclose(fpt);
#endif

            EMO_finalization(prob);
            printf("Done.\n");
        }
    }

    fclose(readf);
#ifdef TAG_UPDATE_testInstance_txt
    fclose(wrtef);
#endif

    MPI_Finalize();

    return 0;
}
#endif
