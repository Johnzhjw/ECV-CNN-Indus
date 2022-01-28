#include "MOP_cnn_data.h"
#include "MOP_HandleData.h"
#include "MOP_evoCFRNN.h"
#include <float.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
//#define COMPUTE_OBJECTIVES_FOR_VALIDATION_SET
#endif

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_evoCFRNN 0
#define FLAG_ON_MOP_evoCFRNN 1
#define STATUS_OUT_INDEICES_MOP_evoCFRNN FLAG_ON_MOP_evoCFRNN

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN_MOP_evoCFRNN 1024
#define MAX_LAB_NUM_MOP_evoCFRNN 1024
#define VIOLATION_PENALTY_evoCFRNN_C 1e6

//////////////////////////////////////////////////////////////////////////
int NDIM_evoCFRNN_Classify = 0;
int NOBJ_evoCFRNN_Classify = 0;

//////////////////////////////////////////////////////////////////////////
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
const int num_class_MOP_evoCFRNN = 10;
LabelArr allLabels_train_MOP_evoCFRNN;
ImgArr   allImgs_train_MOP_evoCFRNN;
LabelArr allLabels_test_MOP_evoCFRNN;
ImgArr   allImgs_test_MOP_evoCFRNN;
#else
const int num_class_MOP_evoCFRNN = 2;
DataSet allSamples_train_MOP_evoCFRNN;
DataSet allSamples_test_MOP_evoCFRNN;
#endif

CNN_evoCFRNN_C* cnn_evoCFRNN_c = NULL;

static int** allocINT_MOP_evoCFRNN(int nrow, int ncol);
static MY_FLT_TYPE** allocFLOAT_MOP_evoCFRNN(int nrow, int ncol);
static void ff_evoCFRNN_Classify(double* individual, ImgArr inputData, LabelArr outputData, int tag_train_test);
static void ff_evoCFRNN_Classify(double* individual, DataSet curData, int tag_train_test);
static void getIndicators_MOP_evoCFRNN_Classify(MY_FLT_TYPE& mean_p, MY_FLT_TYPE& mean_r, MY_FLT_TYPE& mean_F,
        MY_FLT_TYPE& std_p, MY_FLT_TYPE& std_r, MY_FLT_TYPE& std_F);
static void getNetworkComplexity(MY_FLT_TYPE& f_simpl);

//////////////////////////////////////////////////////////////////////////
void Initialize_evoCFRNN_Classify(int curN, int numN)
{
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    //
    char fname[MAX_STR_LEN_MOP_evoCFRNN];
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
    sprintf(fname, "../Data_all/Data_MNIST/train-images.idx3-ubyte");
    allImgs_train_MOP_evoCFRNN = read_Img_IDX_FILE(fname);
    sprintf(fname, "../Data_all/Data_MNIST/train-labels.idx1-ubyte");
    allLabels_train_MOP_evoCFRNN = read_Label_IDX_FILE(fname, num_class_MOP_evoCFRNN);
    sprintf(fname, "../Data_all/Data_MNIST/t10k-images.idx3-ubyte");
    allImgs_test_MOP_evoCFRNN = read_Img_IDX_FILE(fname);
    sprintf(fname, "../Data_all/Data_MNIST/t10k-labels.idx1-ubyte");
    allLabels_test_MOP_evoCFRNN = read_Label_IDX_FILE(fname, num_class_MOP_evoCFRNN);
#else
    sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/train");
    handleData_read(allSamples_train_MOP_evoCFRNN, fname, HANDLEDATATYPE_STORE_ALL);
    //sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/train.save");
    //handleData_save(allSamples_train_MOP_evoCFRNN, fname);
    handleData_normalize(allSamples_train_MOP_evoCFRNN,
                         allSamples_train_MOP_evoCFRNN.min_val,
                         allSamples_train_MOP_evoCFRNN.max_val);
    //sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/train_norm.save");
    //handleData_save(allSamples_train_MOP_evoCFRNN, fname);
    sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/test");
    handleData_read(allSamples_test_MOP_evoCFRNN, fname, HANDLEDATATYPE_STORE_ALL);
    //sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/test.save");
    //handleData_save(allSamples_test_MOP_evoCFRNN, fname);
    handleData_normalize(allSamples_test_MOP_evoCFRNN,
                         allSamples_train_MOP_evoCFRNN.min_val,
                         allSamples_train_MOP_evoCFRNN.max_val);
    //sprintf(fname, "../Data_all/Data_MedicalInsuranceFraud/test_norm.save");
    //handleData_save(allSamples_test_MOP_evoCFRNN, fname);
#endif
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    cnn_evoCFRNN_c = (CNN_evoCFRNN_C*)calloc(1, sizeof(CNN_evoCFRNN_C));
    cnn_evoCFRNN_c_setup(cnn_evoCFRNN_c);
    //
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    NDIM_evoCFRNN_Classify =
        cnn_evoCFRNN_c->M1->numParaLocal +
        cnn_evoCFRNN_c->F2->numParaLocal +
        cnn_evoCFRNN_c->R3->numParaLocal +
        cnn_evoCFRNN_c->OL->numParaLocal;
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    NDIM_evoCFRNN_Classify =
        cnn_evoCFRNN_c->C1->numParaLocal +
        cnn_evoCFRNN_c->P2->numParaLocal +
        cnn_evoCFRNN_c->C3->numParaLocal +
        cnn_evoCFRNN_c->P4->numParaLocal +
        cnn_evoCFRNN_c->M5->numParaLocal +
        cnn_evoCFRNN_c->F6->numParaLocal +
        cnn_evoCFRNN_c->R7->numParaLocal +
        cnn_evoCFRNN_c->OL->numParaLocal;
#else
    NDIM_evoCFRNN_Classify =
        cnn_evoCFRNN_c->C1->numParaLocal +
        cnn_evoCFRNN_c->P2->numParaLocal +
        cnn_evoCFRNN_c->C3->numParaLocal +
        cnn_evoCFRNN_c->P4->numParaLocal +
        cnn_evoCFRNN_c->I5->numParaLocal +
        cnn_evoCFRNN_c->M6->numParaLocal +
        cnn_evoCFRNN_c->F7->numParaLocal +
        cnn_evoCFRNN_c->R8->numParaLocal +
        cnn_evoCFRNN_c->OL->numParaLocal;
#endif
    NOBJ_evoCFRNN_Classify = 3;
    //
    return;
}
void SetLimits_evoCFRNN_Classify(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    for(int i = 0; i < cnn_evoCFRNN_c->M1->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->M1->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->M1->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F2->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->F2->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->F2->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R3->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->R3->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->R3->xMax[i];
        count++;
    }
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    for(int i = 0; i < cnn_evoCFRNN_c->C1->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->C1->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->C1->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P2->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->P2->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->P2->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->C3->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->C3->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->C3->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P4->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->P4->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->P4->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->M5->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->M5->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->M5->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F6->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->F6->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->F6->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R7->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->R7->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->R7->xMax[i];
        count++;
    }
#else
    for(int i = 0; i < cnn_evoCFRNN_c->C1->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->C1->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->C1->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P2->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->P2->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->P2->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->C3->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->C3->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->C3->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P4->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->P4->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->P4->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->I5->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->I5->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->I5->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->M6->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->M6->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->M6->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F7->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->F7->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->F7->xMax[i];
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R8->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->R8->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->R8->xMax[i];
        count++;
    }
#endif
    for(int i = 0; i < cnn_evoCFRNN_c->OL->numParaLocal; i++) {
        minLimit[count] = cnn_evoCFRNN_c->OL->xMin[i];
        maxLimit[count] = cnn_evoCFRNN_c->OL->xMax[i];
        count++;
    }
    return;
}
int CheckLimits_evoCFRNN_Classify(double* x, int nx)
{
    int count = 0;
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    for(int i = 0; i < cnn_evoCFRNN_c->M1->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->M1->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->M1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->M1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->M1->xMin[i], cnn_evoCFRNN_c->M1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F2->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->F2->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->F2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->F2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->F2->xMin[i], cnn_evoCFRNN_c->F2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R3->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->R3->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->R3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->R3->xMin[i], cnn_evoCFRNN_c->R3->xMax[i]);
            return 0;
        }
        count++;
    }
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    for(int i = 0; i < cnn_evoCFRNN_c->C1->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->C1->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->C1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->C1->xMin[i], cnn_evoCFRNN_c->C1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P2->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->P2->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->P2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->P2->xMin[i], cnn_evoCFRNN_c->P2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->C3->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->C3->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->C3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->C3->xMin[i], cnn_evoCFRNN_c->C3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P4->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->P4->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->P4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->P4->xMin[i], cnn_evoCFRNN_c->P4->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->M5->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->M5->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->M5->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->M5 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->M5->xMin[i], cnn_evoCFRNN_c->M5->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F6->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->F6->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->F6->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->F6 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->F6->xMin[i], cnn_evoCFRNN_c->F6->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R7->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->R7->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->R7->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->R7 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->R7->xMin[i], cnn_evoCFRNN_c->R7->xMax[i]);
            return 0;
        }
        count++;
    }
#else
    for(int i = 0; i < cnn_evoCFRNN_c->C1->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->C1->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->C1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->C1->xMin[i], cnn_evoCFRNN_c->C1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P2->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->P2->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->P2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->P2->xMin[i], cnn_evoCFRNN_c->P2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->C3->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->C3->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->C3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->C3->xMin[i], cnn_evoCFRNN_c->C3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->P4->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->P4->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->P4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->P4->xMin[i], cnn_evoCFRNN_c->P4->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->I5->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->I5->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->I5->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->I5 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->I5->xMin[i], cnn_evoCFRNN_c->I5->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->M6->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->M6->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->M6->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->M6 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->M6->xMin[i], cnn_evoCFRNN_c->M6->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->F7->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->F7->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->F7->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->F7 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->F7->xMin[i], cnn_evoCFRNN_c->F7->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cnn_evoCFRNN_c->R8->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->R8->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->R8->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->R8 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->R8->xMin[i], cnn_evoCFRNN_c->R8->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    for(int i = 0; i < cnn_evoCFRNN_c->OL->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->OL->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->OL %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->OL->xMin[i], cnn_evoCFRNN_c->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#else
    if(cnn_evoCFRNN_c->flagConnectStatus != FLAG_STATUS_OFF ||
       cnn_evoCFRNN_c->flagConnectWeight != FLAG_STATUS_ON ||
       cnn_evoCFRNN_c->typeCoding != PARA_CODING_DIRECT) {
        printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
               __FILE__, __LINE__, cnn_evoCFRNN_c->flagConnectStatus, cnn_evoCFRNN_c->flagConnectWeight, cnn_evoCFRNN_c->typeCoding);
        exit(-275082);
    }
    int tmp_offset = cnn_evoCFRNN_c->OL->numOutput * cnn_evoCFRNN_c->OL->numInput;
    count += tmp_offset;
    for(int i = tmp_offset; i < cnn_evoCFRNN_c->OL->numParaLocal; i++) {
        if(x[count] < cnn_evoCFRNN_c->OL->xMin[i] ||
           x[count] > cnn_evoCFRNN_c->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->O4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cnn_evoCFRNN_c->OL->xMin[i], cnn_evoCFRNN_c->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
    //
    return 1;
}
void Fitness_evoCFRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M)
{
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
    ff_evoCFRNN_Classify(individual, allImgs_train_MOP_evoCFRNN, allLabels_train_MOP_evoCFRNN, TRAIN_TAG_MOP_evoCFRNN_Classify);
#else
    ff_evoCFRNN_Classify(individual, allSamples_train_MOP_evoCFRNN, TRAIN_TAG_MOP_evoCFRNN_Classify);
#endif
    //
    int len_lab = num_class_MOP_evoCFRNN;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    getIndicators_MOP_evoCFRNN_Classify(mean_precision, mean_recall, mean_Fvalue, std_precision, std_recall, std_Fvalue);
    MY_FLT_TYPE f_simpl = 0;
    getNetworkComplexity(f_simpl);
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cnn_evoCFRNN_c->OL->dataflowStatus[i];
        if(cnn_evoCFRNN_c->OL->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_evoCFRNN_C);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
    fitness[0] = 1 - mean_recall + val_violation;
    fitness[1] = std_recall + val_violation;
    fitness[2] = f_simpl;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
    return;
}

void Fitness_evoCFRNN_Classify_test(double* individual, double* fitness)
{
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
    ff_evoCFRNN_Classify(individual, allImgs_test_MOP_evoCFRNN, allLabels_test_MOP_evoCFRNN, TEST_TAG_MOP_evoCFRNN_Classify);
#else
    ff_evoCFRNN_Classify(individual, allSamples_test_MOP_evoCFRNN, TEST_TAG_MOP_evoCFRNN_Classify);
#endif
    //
    int len_lab = num_class_MOP_evoCFRNN;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    getIndicators_MOP_evoCFRNN_Classify(mean_precision, mean_recall, mean_Fvalue, std_precision, std_recall, std_Fvalue);
    MY_FLT_TYPE f_simpl = 0;
    getNetworkComplexity(f_simpl);
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cnn_evoCFRNN_c->OL->dataflowStatus[i];
        if(cnn_evoCFRNN_c->OL->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_evoCFRNN_C);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
    fitness[0] = 1 - mean_recall + val_violation;
    fitness[1] = std_recall + val_violation;
    fitness[2] = f_simpl;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
    return;
}

static void ff_evoCFRNN_Classify(double* individual, ImgArr inputData, LabelArr outputData, int tag_train_test)
{
    int num_sample = inputData->ImgNum;
    int len_lab = num_class_MOP_evoCFRNN;
    cnn_evoCFRNN_c->sum_all = 0;
    cnn_evoCFRNN_c->sum_wrong = 0;
    for(int i = 0; i < cnn_evoCFRNN_c->numOutput; i++) {
        cnn_evoCFRNN_c->N_sum[i] = 0;
        cnn_evoCFRNN_c->N_wrong[i] = 0;
        cnn_evoCFRNN_c->e_sum[i] = 0;
        cnn_evoCFRNN_c->N_TP[i] = 0;
        cnn_evoCFRNN_c->N_TN[i] = 0;
        cnn_evoCFRNN_c->N_FP[i] = 0;
        cnn_evoCFRNN_c->N_FN[i] = 0;
    }
    cnn_evoCFRNN_c_init(cnn_evoCFRNN_c, individual, ASSIGN_MODE_FRNN);
    //
    MY_FLT_TYPE*** valIn;
    MY_FLT_TYPE valOut[MAX_LAB_NUM_MOP_evoCFRNN];
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    MY_FLT_TYPE* matA = NULL;
    MY_FLT_TYPE* matB = NULL;
    MY_FLT_TYPE* matLeft = NULL;
    MY_FLT_TYPE* matRight = NULL;
    int tmp_offset_samp = 0;
    int tmp_offset_samp_validation = 0;
    if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
        matA = (MY_FLT_TYPE*)calloc(num_sample * cnn_evoCFRNN_c->OL->numInput, sizeof(MY_FLT_TYPE));
        matB = (MY_FLT_TYPE*)calloc(num_sample * cnn_evoCFRNN_c->OL->numOutput, sizeof(MY_FLT_TYPE));
        matLeft = (MY_FLT_TYPE*)calloc(cnn_evoCFRNN_c->OL->numInput * cnn_evoCFRNN_c->OL->numInput, sizeof(MY_FLT_TYPE));
        matRight = (MY_FLT_TYPE*)calloc(cnn_evoCFRNN_c->OL->numInput * cnn_evoCFRNN_c->OL->numOutput, sizeof(MY_FLT_TYPE));
    }
#endif
    for(int m = 0; m < num_sample; m += 1) {
        valIn = &inputData->ImgPtr[m].ImgData;
        ff_cnn_evoCFRNN_c(cnn_evoCFRNN_c, valIn, valOut, NULL);
        int cur_label = 0;
        MY_FLT_TYPE cur_out = valOut[0];
        for(int j = 1; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(cur_out < valOut[j]) {
                cur_out = valOut[j];
                cur_label = j;
            }
        }
        int true_label = 0;
        MY_FLT_TYPE tmp_max_lab_val = outputData->LabelPtr[m].LabelData[0];
        for(int j = 1; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(tmp_max_lab_val < outputData->LabelPtr[m].LabelData[j]) {
                tmp_max_lab_val = outputData->LabelPtr[m].LabelData[j];
                true_label = j;
            }
        }
        for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(j == cur_label && j == true_label) cnn_evoCFRNN_c->N_TP[j]++;
            if(j == cur_label && j != true_label) cnn_evoCFRNN_c->N_FP[j]++;
            if(j != cur_label && j == true_label) cnn_evoCFRNN_c->N_FN[j]++;
            if(j != cur_label && j != true_label) cnn_evoCFRNN_c->N_TN[j]++;
        }
        cnn_evoCFRNN_c->sum_all++;
        cnn_evoCFRNN_c->N_sum[true_label]++;
        if(cur_label != true_label) {
            cnn_evoCFRNN_c->sum_wrong++;
            cnn_evoCFRNN_c->N_wrong[true_label]++;
        }
        //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
        if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numInput; j++) {
                int ind_cur = tmp_offset_samp * cnn_evoCFRNN_c->OL->numInput + j;
                matA[ind_cur] = cnn_evoCFRNN_c->OL->valInputFinal[0][j];
            }
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int ind_cur = tmp_offset_samp * cnn_evoCFRNN_c->OL->numOutput + j;
                if(j == true_label)
                    matB[ind_cur] = 1;
                else
                    matB[ind_cur] = -1;
            }
        }
        tmp_offset_samp++;
#endif
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
        cnn_evoCFRNN_c->sum_all = 0;
        cnn_evoCFRNN_c->sum_wrong = 0;
        for(int i = 0; i < cnn_evoCFRNN_c->numOutput; i++) {
            cnn_evoCFRNN_c->N_sum[i] = 0;
            cnn_evoCFRNN_c->N_wrong[i] = 0;
            cnn_evoCFRNN_c->e_sum[i] = 0;
            cnn_evoCFRNN_c->N_TP[i] = 0;
            cnn_evoCFRNN_c->N_TN[i] = 0;
            cnn_evoCFRNN_c->N_FP[i] = 0;
            cnn_evoCFRNN_c->N_FN[i] = 0;
        }
        //
        //printf("tmp_offset_samp = %d\n", tmp_offset_samp);
        MY_FLT_TYPE lambda = 9.3132e-10;
        MY_FLT_TYPE tmp_max = 0;
        int tmp_max_flag = 0;
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numInput; j++) {
                int tmp_o0 = i * cnn_evoCFRNN_c->OL->numInput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * cnn_evoCFRNN_c->OL->numInput + i;
                    int tmp_i2 = k * cnn_evoCFRNN_c->OL->numInput + j;
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
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            int tmp_o0 = i * cnn_evoCFRNN_c->OL->numInput + i;
            matLeft[tmp_o0] += lambda;// *tmp_max;
        }
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int tmp_o0 = i * cnn_evoCFRNN_c->OL->numOutput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * cnn_evoCFRNN_c->OL->numInput + i;
                    int tmp_i2 = k * cnn_evoCFRNN_c->OL->numOutput + j;
                    matRight[tmp_o0] += matA[tmp_i1] * matB[tmp_i2];
                }
            }
        }
        int N = cnn_evoCFRNN_c->OL->numInput;
        int NRHS = cnn_evoCFRNN_c->OL->numOutput;
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
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int ind_cur = i * cnn_evoCFRNN_c->OL->numOutput + j;
                if(CHECK_INVALID(matRight[ind_cur])) {
                    printf("%s(%d): Error - invalid value of matRight[%d] = %lf",
                           __FILE__, __LINE__, ind_cur, matRight[ind_cur]);
                    exit(-112);
                }
                cnn_evoCFRNN_c->OL->connectWeight[j][i] = matRight[ind_cur];
                //sum_weights += fabs(matRight[ind_cur]);
            }
        }
        cnn_evoCFRNN_c_init(cnn_evoCFRNN_c, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
        for(int m = 0; m < tmp_offset_samp; m++) {
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                valOut[j] = 0;
                for(int k = 0; k < cnn_evoCFRNN_c->OL->numInput; k++) {
                    int ind_cur = m * cnn_evoCFRNN_c->OL->numInput + k;
                    valOut[j] += matA[ind_cur] * cnn_evoCFRNN_c->OL->connectWeight[j][k];
                }
                if(CHECK_INVALID(valOut[j])) {
                    printf("%d~%lf", j, valOut[j]);
                }
            }
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE cur_true_out = matB[m * cnn_evoCFRNN_c->OL->numOutput];
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(cur_true_out < matB[m * cnn_evoCFRNN_c->OL->numOutput + j]) {
                    cur_true_out = matB[m * cnn_evoCFRNN_c->OL->numOutput + j];
                    true_label = j;
                }
            }
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(j == cur_label && j == true_label) cnn_evoCFRNN_c->N_TP[j]++;
                if(j == cur_label && j != true_label) cnn_evoCFRNN_c->N_FP[j]++;
                if(j != cur_label && j == true_label) cnn_evoCFRNN_c->N_FN[j]++;
                if(j != cur_label && j != true_label) cnn_evoCFRNN_c->N_TN[j]++;
            }
            cnn_evoCFRNN_c->sum_all++;
            cnn_evoCFRNN_c->N_sum[true_label]++;
            if(cur_label != true_label) {
                cnn_evoCFRNN_c->sum_wrong++;
                cnn_evoCFRNN_c->N_wrong[true_label]++;
            }
            //
            //MY_FLT_TYPE softmax_outs[num_class_MOP_evoCFRNN];
            //MY_FLT_TYPE softmax_sum = 0;
            //MY_FLT_TYPE softmax_degr[num_class_MOP_evoCFRNN];
            //for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //    softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j] - cur_out));
            //    softmax_sum += softmax_outs[j];
            //}
            //if(softmax_sum > 0) {
            //    for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //        softmax_degr[j] = softmax_outs[j] / softmax_sum;
            //    }
            //}
            //for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //    if(true_label == j) {
            //        tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
            //    } else {
            //        tmp_all_err += softmax_degr[j] * softmax_degr[j];
            //    }
            //}
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
    return;
}

static void ff_evoCFRNN_Classify(double* individual, DataSet curData, int tag_train_test)
{
    int num_sample = curData.numSample;
    int len_lab = curData.numClass;
    cnn_evoCFRNN_c->sum_all = 0;
    cnn_evoCFRNN_c->sum_wrong = 0;
    for(int i = 0; i < cnn_evoCFRNN_c->numOutput; i++) {
        cnn_evoCFRNN_c->N_sum[i] = 0;
        cnn_evoCFRNN_c->N_wrong[i] = 0;
        cnn_evoCFRNN_c->e_sum[i] = 0;
        cnn_evoCFRNN_c->N_TP[i] = 0;
        cnn_evoCFRNN_c->N_TN[i] = 0;
        cnn_evoCFRNN_c->N_FP[i] = 0;
        cnn_evoCFRNN_c->N_FN[i] = 0;
    }
    cnn_evoCFRNN_c_init(cnn_evoCFRNN_c, individual, ASSIGN_MODE_FRNN);
    //
    MY_FLT_TYPE*** valIn = (MY_FLT_TYPE***)malloc(1 * sizeof(MY_FLT_TYPE**));
    MY_FLT_TYPE valOut[MAX_LAB_NUM_MOP_evoCFRNN];
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    MY_FLT_TYPE* matA = NULL;
    MY_FLT_TYPE* matB = NULL;
    MY_FLT_TYPE* matLeft = NULL;
    MY_FLT_TYPE* matRight = NULL;
    int tmp_offset_samp = 0;
    int tmp_offset_samp_validation = 0;
    if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
        matA = (MY_FLT_TYPE*)calloc(num_sample * cnn_evoCFRNN_c->OL->numInput, sizeof(MY_FLT_TYPE));
        matB = (MY_FLT_TYPE*)calloc(num_sample * cnn_evoCFRNN_c->OL->numOutput, sizeof(MY_FLT_TYPE));
        matLeft = (MY_FLT_TYPE*)calloc(cnn_evoCFRNN_c->OL->numInput * cnn_evoCFRNN_c->OL->numInput, sizeof(MY_FLT_TYPE));
        matRight = (MY_FLT_TYPE*)calloc(cnn_evoCFRNN_c->OL->numInput * cnn_evoCFRNN_c->OL->numOutput, sizeof(MY_FLT_TYPE));
    }
#endif
    for(int m = 0; m < num_sample; m += 1) {
        valIn[0] = &curData.Data[m];
        ff_cnn_evoCFRNN_c(cnn_evoCFRNN_c, valIn, valOut, NULL);
        int cur_label = 0;
        MY_FLT_TYPE cur_out = valOut[0];
        for(int j = 1; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(cur_out < valOut[j]) {
                cur_out = valOut[j];
                cur_label = j;
            }
        }
        int true_label = 0;
        MY_FLT_TYPE tmp_max_lab_val = curData.Label[m][0];
        for(int j = 1; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(tmp_max_lab_val < curData.Label[m][j]) {
                tmp_max_lab_val = curData.Label[m][j];
                true_label = j;
            }
        }
        for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            if(j == cur_label && j == true_label) cnn_evoCFRNN_c->N_TP[j]++;
            if(j == cur_label && j != true_label) cnn_evoCFRNN_c->N_FP[j]++;
            if(j != cur_label && j == true_label) cnn_evoCFRNN_c->N_FN[j]++;
            if(j != cur_label && j != true_label) cnn_evoCFRNN_c->N_TN[j]++;
        }
        cnn_evoCFRNN_c->sum_all++;
        cnn_evoCFRNN_c->N_sum[true_label]++;
        if(cur_label != true_label) {
            cnn_evoCFRNN_c->sum_wrong++;
            cnn_evoCFRNN_c->N_wrong[true_label]++;
        }
        //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
        if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numInput; j++) {
                int ind_cur = tmp_offset_samp * cnn_evoCFRNN_c->OL->numInput + j;
                matA[ind_cur] = cnn_evoCFRNN_c->OL->valInputFinal[0][j];
            }
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int ind_cur = tmp_offset_samp * cnn_evoCFRNN_c->OL->numOutput + j;
                if(j == true_label)
                    matB[ind_cur] = 1;
                else
                    matB[ind_cur] = -1;
            }
        }
        tmp_offset_samp++;
#endif
    }
    //
    free(valIn);
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_test == TRAIN_TAG_MOP_evoCFRNN_Classify) {
        cnn_evoCFRNN_c->sum_all = 0;
        cnn_evoCFRNN_c->sum_wrong = 0;
        for(int i = 0; i < cnn_evoCFRNN_c->numOutput; i++) {
            cnn_evoCFRNN_c->N_sum[i] = 0;
            cnn_evoCFRNN_c->N_wrong[i] = 0;
            cnn_evoCFRNN_c->e_sum[i] = 0;
            cnn_evoCFRNN_c->N_TP[i] = 0;
            cnn_evoCFRNN_c->N_TN[i] = 0;
            cnn_evoCFRNN_c->N_FP[i] = 0;
            cnn_evoCFRNN_c->N_FN[i] = 0;
        }
        //
        //printf("tmp_offset_samp = %d\n", tmp_offset_samp);
        MY_FLT_TYPE lambda = 9.3132e-10;
        MY_FLT_TYPE tmp_max = 0;
        int tmp_max_flag = 0;
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numInput; j++) {
                int tmp_o0 = i * cnn_evoCFRNN_c->OL->numInput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * cnn_evoCFRNN_c->OL->numInput + i;
                    int tmp_i2 = k * cnn_evoCFRNN_c->OL->numInput + j;
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
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            int tmp_o0 = i * cnn_evoCFRNN_c->OL->numInput + i;
            matLeft[tmp_o0] += lambda;// *tmp_max;
        }
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int tmp_o0 = i * cnn_evoCFRNN_c->OL->numOutput + j;
                for(int k = 0; k < tmp_offset_samp; k++) {
                    int tmp_i1 = k * cnn_evoCFRNN_c->OL->numInput + i;
                    int tmp_i2 = k * cnn_evoCFRNN_c->OL->numOutput + j;
                    matRight[tmp_o0] += matA[tmp_i1] * matB[tmp_i2];
                }
            }
        }
        int N = cnn_evoCFRNN_c->OL->numInput;
        int NRHS = cnn_evoCFRNN_c->OL->numOutput;
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
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numInput; i++) {
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                int ind_cur = i * cnn_evoCFRNN_c->OL->numOutput + j;
                if(CHECK_INVALID(matRight[ind_cur])) {
                    printf("%s(%d): Error - invalid value of matRight[%d] = %lf",
                           __FILE__, __LINE__, ind_cur, matRight[ind_cur]);
                    exit(-112);
                }
                cnn_evoCFRNN_c->OL->connectWeight[j][i] = matRight[ind_cur];
                //sum_weights += fabs(matRight[ind_cur]);
            }
        }
        cnn_evoCFRNN_c_init(cnn_evoCFRNN_c, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
        for(int m = 0; m < tmp_offset_samp; m++) {
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
            for(int j = 0; j < cnn_evoCFRNN_c->OL->numOutput; j++) {
                valOut[j] = 0;
                for(int k = 0; k < cnn_evoCFRNN_c->OL->numInput; k++) {
                    int ind_cur = m * cnn_evoCFRNN_c->OL->numInput + k;
                    valOut[j] += matA[ind_cur] * cnn_evoCFRNN_c->OL->connectWeight[j][k];
                }
                if(CHECK_INVALID(valOut[j])) {
                    printf("%d~%lf", j, valOut[j]);
                }
            }
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE cur_true_out = matB[m * cnn_evoCFRNN_c->OL->numOutput];
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(cur_true_out < matB[m * cnn_evoCFRNN_c->OL->numOutput + j]) {
                    cur_true_out = matB[m * cnn_evoCFRNN_c->OL->numOutput + j];
                    true_label = j;
                }
            }
            for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
                if(j == cur_label && j == true_label) cnn_evoCFRNN_c->N_TP[j]++;
                if(j == cur_label && j != true_label) cnn_evoCFRNN_c->N_FP[j]++;
                if(j != cur_label && j == true_label) cnn_evoCFRNN_c->N_FN[j]++;
                if(j != cur_label && j != true_label) cnn_evoCFRNN_c->N_TN[j]++;
            }
            cnn_evoCFRNN_c->sum_all++;
            cnn_evoCFRNN_c->N_sum[true_label]++;
            if(cur_label != true_label) {
                cnn_evoCFRNN_c->sum_wrong++;
                cnn_evoCFRNN_c->N_wrong[true_label]++;
            }
            //
            //MY_FLT_TYPE softmax_outs[num_class_MOP_evoCFRNN];
            //MY_FLT_TYPE softmax_sum = 0;
            //MY_FLT_TYPE softmax_degr[num_class_MOP_evoCFRNN];
            //for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //    softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j] - cur_out));
            //    softmax_sum += softmax_outs[j];
            //}
            //if(softmax_sum > 0) {
            //    for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //        softmax_degr[j] = softmax_outs[j] / softmax_sum;
            //    }
            //}
            //for(int j = 0; j < cnn_evoCFRNN_c->numOutput; j++) {
            //    if(true_label == j) {
            //        tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
            //    } else {
            //        tmp_all_err += softmax_degr[j] * softmax_degr[j];
            //    }
            //}
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
    return;
}

void Finalize_evoCFRNN_Classify()
{
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
    free_Img(allImgs_train_MOP_evoCFRNN);
    free_Label(allLabels_train_MOP_evoCFRNN);
    free_Img(allImgs_test_MOP_evoCFRNN);
    free_Label(allLabels_test_MOP_evoCFRNN);
#else
    handleData_free(allSamples_train_MOP_evoCFRNN);
    handleData_free(allSamples_test_MOP_evoCFRNN);
#endif

    cnn_evoCFRNN_c_free(cnn_evoCFRNN_c);
    return;
}

//////////////////////////////////////////////////////////////////////////
void cnn_evoCFRNN_c_setup(CNN_evoCFRNN_C* cfrnn)
{
    int numOutput = num_class_MOP_evoCFRNN;
    //
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE;
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;// FIXED_ROUGH_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int flagConnectStatus = FLAG_STATUS_OFF;
    int flagConnectWeight = FLAG_STATUS_ON;
    //
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = (int)sqrt(numFuzzyRules);
    //
    cfrnn->typeFuzzySet = typeFuzzySet;
    cfrnn->typeRules = typeRules;
    cfrnn->typeInRuleCorNum = typeInRuleCorNum;
    cfrnn->typeTypeReducer = typeTypeReducer;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
    cfrnn->centroid_num_tag = centroid_num_tag;
    cfrnn->flagConnectStatus = flagConnectStatus;
    cfrnn->flagConnectWeight = flagConnectWeight;
    //
    cfrnn->layerNum = 8;
    cfrnn->numOutput = numOutput;
    //
#if DATASET_MOP_EVO_CFRNN_CUR == DATASET_MNIST_MOP_EVO_CFRNN
    int inputHeightMax = allImgs_train_MOP_evoCFRNN->ImgPtr[0].r;
    int inputWidthMax = allImgs_train_MOP_evoCFRNN->ImgPtr[0].c;
#else
    int inputHeightMax = 1;
    int inputWidthMax = allSamples_train_MOP_evoCFRNN.numFeature;
#endif
    cfrnn->inputHeightMax = inputHeightMax;
    cfrnn->inputWidthMax = inputWidthMax;
    //
    int channelsIn_C1 = 1;
    cfrnn->inputChannel = channelsIn_C1;
    cfrnn->numValIn = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
    int channelsOut_C1 = 6;
    int channelsInOut_P2 = channelsOut_C1;
    int channelsIn_C3 = channelsInOut_P2;
    int channelsOut_C3 = 12;
    int channelsInOut_P4 = channelsOut_C3;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    int tmp_flagConnectStatusAdap = FLAG_STATUS_ON;
    int tmp_flag_kernelFlagAdap = FLAG_STATUS_OFF;
    int tmp_default_kernelFlag = KERNEL_FLAG_OPERATE;
    int tmp_flag_actFuncAdap = FLAG_STATUS_OFF;
    int tmp_default_actFunc = ACT_FUNC_LEAKYRELU;
    int tmp_flag_paddingTypeAdap = FLAG_STATUS_OFF;
    int tmp_default_paddingType = PADDING_SAME;
    int tmp_flag_poolTypeAdap = FLAG_STATUS_OFF;
    int tmp_default_poolType = POOL_MAX;
    //
    cfrnn->typeCoding = tmp_typeCoding;
    //
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    cfrnn->layerNum = 4;
    int* numMemship = (int*)calloc(cfrnn->numValIn, sizeof(int));
    int* flagAdapMemship = (int*)calloc(cfrnn->numValIn, sizeof(int));
    MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(cfrnn->numValIn, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(cfrnn->numValIn, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < cfrnn->numValIn; i++) {
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
        flagAdapMemship[i] = 1;
        inputMin[i] = 0;
        inputMax[i] = 1;
    }
    cfrnn->M1 = setupMemberLayer(cfrnn->numValIn, inputMin, inputMax, numMemship, flagAdapMemship, cfrnn->typeFuzzySet,
                                 tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    free(numMemship);
    free(flagAdapMemship);
    free(inputMin);
    free(inputMax);
    cfrnn->F2 = setupFuzzyLayer(cfrnn->M1->numInput, cfrnn->M1->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet,
                                cfrnn->typeRules, cfrnn->typeInRuleCorNum, FLAG_STATUS_OFF, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    cfrnn->R3 = setupRoughLayer(cfrnn->F2->numRules, numRoughSets, cfrnn->typeFuzzySet, tmp_flagConnectStatusAdap,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE* outputMin = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* outputMax = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = cfrnn->numValIn;
    cfrnn->OL = setupOutReduceLayer(cfrnn->R3->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer,
                                    cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag, numInputConsequenceNode, inputMin, inputMax,
                                    cfrnn->flagConnectStatus, cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    free(outputMin);
    free(outputMax);
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    cfrnn->layerNum = 8;
    cfrnn->C1 = setupConvLayer(inputHeightMax, inputWidthMax, channelsIn_C1, channelsOut_C1, channelsIn_C1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P2 = setupPoolLayer(cfrnn->C1->featureMapHeightMax, cfrnn->C1->featureMapWidthMax, channelsInOut_P2,
                               cfrnn->C1->channelsOutMax,
                               1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    cfrnn->C3 = setupConvLayer(cfrnn->P2->featureMapHeightMax, cfrnn->P2->featureMapWidthMax, channelsIn_C3, channelsOut_C3,
                               cfrnn->P2->channelsInOutMax,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P4 = setupPoolLayer(cfrnn->C3->featureMapHeightMax, cfrnn->C3->featureMapWidthMax, channelsInOut_P4,
                               cfrnn->C3->channelsOutMax,
                               1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    int numMemship[100];
    for(int i = 0; i < cfrnn->P4->channelsInOutMax; i++) {
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
    }
    int flagAdapMemship[100];
    for(int i = 0; i < cfrnn->P4->channelsInOutMax; i++) {
        flagAdapMemship[i] = 1;
    }
    cfrnn->M5 = setupMember2DLayer(cfrnn->P4->channelsInOutMax,
                                   tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1,
                                   cfrnn->P4->featureMapHeightMax, cfrnn->P4->featureMapWidthMax,
                                   0, MAT_SIMILARITY_T_COS, numMemship, flagAdapMemship, cfrnn->typeFuzzySet);
    cfrnn->F6 = setupFuzzyLayer(cfrnn->P4->channelsInOutMax, cfrnn->M5->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet,
                                cfrnn->typeRules, cfrnn->typeInRuleCorNum, TODO, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    cfrnn->R7 = setupRoughLayer(cfrnn->F6->numRules, numRoughSets, cfrnn->typeFuzzySet, FLAG_STATUS_ON, tmp_typeCoding,
                                MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE outputMin[MAX_LAB_NUM_MOP_evoCFRNN];
    MY_FLT_TYPE outputMax[MAX_LAB_NUM_MOP_evoCFRNN];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = cfrnn->P4->channelsInOutMax;
    cfrnn->OL = setupOutReduceLayer(cfrnn->R7->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer, cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag, numInputConsequenceNode,
                                    TODO, TODO,
                                    cfrnn->flagConnectStatus,
                                    cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
#else
    cfrnn->layerNum = 9;
    cfrnn->C1 = setupConvLayer(inputHeightMax, inputWidthMax, channelsIn_C1, channelsOut_C1, channelsIn_C1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P2 = setupPoolLayer(cfrnn->C1->featureMapHeightMax, cfrnn->C1->featureMapWidthMax, channelsInOut_P2,
                               cfrnn->C1->channelsOutMax,
                               1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    cfrnn->C3 = setupConvLayer(cfrnn->P2->featureMapHeightMax, cfrnn->P2->featureMapWidthMax, channelsIn_C3, channelsOut_C3,
                               cfrnn->P2->channelsInOutMax,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               1, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P4 = setupPoolLayer(cfrnn->C3->featureMapHeightMax, cfrnn->C3->featureMapWidthMax, channelsInOut_P4,
                               cfrnn->C3->channelsOutMax,
                               1, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    int layerNum = 3;
    int numNodesAll[3] = { 4, 7, 2 };
    int numOutICFC = 12;
    int flagActFuncICFC = FLAG_STATUS_OFF;
    int flagActFuncAdapICFC = FLAG_STATUS_OFF;
    int defaultActFuncTypeICFC = ACT_FUNC_LEAKYRELU;
    int flagConnectAdap = FLAG_STATUS_OFF;
    int flag_wt_positiveICFC = FLAG_STATUS_ON;
    cfrnn->I5 = setupInterCFCLayer(cfrnn->P4->channelsInOutMax, cfrnn->P4->featureMapHeightMax, cfrnn->P4->featureMapWidthMax,
                                   numOutICFC,
                                   flagActFuncICFC, flagActFuncAdapICFC, defaultActFuncTypeICFC, flagConnectAdap,
                                   tmp_typeCoding, layerNum, numNodesAll, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                                   flag_wt_positiveICFC, 1);
    MY_FLT_TYPE inputMin[100];
    MY_FLT_TYPE inputMax[100];
    for(int i = 0; i < cfrnn->I5->numOutput; i++) {
        inputMin[i] = 0;
        inputMax[i] = 1;
    }
    int numMemship[100];
    for(int i = 0; i < cfrnn->I5->numOutput; i++) {
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
    }
    int flagAdapMemship[100];
    for(int i = 0; i < cfrnn->I5->numOutput; i++) {
        flagAdapMemship[i] = 1;
    }
    cfrnn->M6 = setupMemberLayer(cfrnn->I5->numOutput,
                                 inputMin, inputMax,
                                 numMemship, flagAdapMemship, cfrnn->typeFuzzySet,
                                 tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    cfrnn->F7 = setupFuzzyLayer(cfrnn->I5->numOutput, cfrnn->M6->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet,
                                cfrnn->typeRules, cfrnn->typeInRuleCorNum, TODO, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    cfrnn->R8 = setupRoughLayer(cfrnn->F7->numRules, numRoughSets, cfrnn->typeFuzzySet,
                                1,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE outputMin[MAX_LAB_NUM_MOP_evoCFRNN];
    MY_FLT_TYPE outputMax[MAX_LAB_NUM_MOP_evoCFRNN];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = cfrnn->I5->numOutput;
    cfrnn->OL = setupOutReduceLayer(cfrnn->R8->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer, cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag, numInputConsequenceNode,
                                    TODO, TODO,
                                    cfrnn->flagConnectStatus,
                                    cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
#endif

    cfrnn->e = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->N_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_wrong = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->e_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->N_TP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_TN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_FP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_FN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->featureMapTagInitial = (int***)calloc(channelsIn_C1, sizeof(int**));
    cfrnn->dataflowInitial = (MY_FLT_TYPE***)calloc(channelsIn_C1, sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < channelsIn_C1; i++) {
        cfrnn->featureMapTagInitial[i] = (int**)calloc(inputHeightMax, sizeof(int*));
        cfrnn->dataflowInitial[i] = (MY_FLT_TYPE**)calloc(inputHeightMax, sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < inputHeightMax; j++) {
            cfrnn->featureMapTagInitial[i][j] = (int*)calloc(inputWidthMax, sizeof(int));
            cfrnn->dataflowInitial[i][j] = (MY_FLT_TYPE*)calloc(inputWidthMax, sizeof(MY_FLT_TYPE));
            for(int k = 0; k < inputWidthMax; k++) {
                cfrnn->featureMapTagInitial[i][j][k] = 1;
                cfrnn->dataflowInitial[i][j][k] = (MY_FLT_TYPE)(1.0 / (inputHeightMax * inputWidthMax));
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

void cnn_evoCFRNN_c_free(CNN_evoCFRNN_C* cfrnn)
{
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    freeMemberLayer(cfrnn->M1);
    freeFuzzyLayer(cfrnn->F2);
    freeRoughLayer(cfrnn->R3);
    freeOutReduceLayer(cfrnn->OL);
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    freeConvLayer(cfrnn->C1);
    freePoolLayer(cfrnn->P2);
    freeConvLayer(cfrnn->C3);
    freePoolLayer(cfrnn->P4);
    freeMember2DLayer(cfrnn->M5);
    freeFuzzyLayer(cfrnn->F6);
    freeRoughLayer(cfrnn->R7);
    freeOutReduceLayer(cfrnn->OL);
#else
    freeConvLayer(cfrnn->C1);
    freePoolLayer(cfrnn->P2);
    freeConvLayer(cfrnn->C3);
    freePoolLayer(cfrnn->P4);
    freeInterCFCLayer(cfrnn->I5);
    freeMemberLayer(cfrnn->M6);
    freeFuzzyLayer(cfrnn->F7);
    freeRoughLayer(cfrnn->R8);
    freeOutReduceLayer(cfrnn->OL);
#endif

    free(cfrnn->e);

    free(cfrnn->N_sum);
    free(cfrnn->N_wrong);
    free(cfrnn->e_sum);

    free(cfrnn->N_TP);
    free(cfrnn->N_TN);
    free(cfrnn->N_FP);
    free(cfrnn->N_FN);

    for(int i = 0; i < cfrnn->inputChannel; i++) {
        for(int j = 0; j < cfrnn->inputHeightMax; j++) {
            free(cfrnn->featureMapTagInitial[i][j]);
            free(cfrnn->dataflowInitial[i][j]);
        }
        free(cfrnn->featureMapTagInitial[i]);
        free(cfrnn->dataflowInitial[i]);
    }
    free(cfrnn->featureMapTagInitial);
    free(cfrnn->dataflowInitial);

    free(cfrnn);

    return;
}

void cnn_evoCFRNN_c_init(CNN_evoCFRNN_C* cfrnn, double* x, int mode)
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

#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    assignMemberLayer(cfrnn->M1, &x[count], mode);
    count += cfrnn->M1->numParaLocal;
    assignFuzzyLayer(cfrnn->F2, &x[count], mode);
    count += cfrnn->F2->numParaLocal;
    assignRoughLayer(cfrnn->R3, &x[count], mode);
    count += cfrnn->R3->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    assignConvLayer(cfrnn->C1, &x[count], mode);
    count += cfrnn->C1->numParaLocal;
    assignPoolLayer(cfrnn->P2, &x[count], mode);
    count += cfrnn->P2->numParaLocal;
    assignConvLayer(cfrnn->C3, &x[count], mode);
    count += cfrnn->C3->numParaLocal;
    assignPoolLayer(cfrnn->P4, &x[count], mode);
    count += cfrnn->P4->numParaLocal;
    assignMember2DLayer(cfrnn->M5, &x[count], mode);
    count += cfrnn->M5->numParaLocal;
    assignFuzzyLayer(cfrnn->F6, &x[count], mode);
    count += cfrnn->F6->numParaLocal;
    assignRoughLayer(cfrnn->R7, &x[count], mode);
    count += cfrnn->R7->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#else
    assignConvLayer(cfrnn->C1, &x[count], mode);
    count += cfrnn->C1->numParaLocal;
    assignPoolLayer(cfrnn->P2, &x[count], mode);
    count += cfrnn->P2->numParaLocal;
    assignConvLayer(cfrnn->C3, &x[count], mode);
    count += cfrnn->C3->numParaLocal;
    assignPoolLayer(cfrnn->P4, &x[count], mode);
    count += cfrnn->P4->numParaLocal;
    assignInterCFCLayer(cfrnn->I5, &x[count], mode);
    count += cfrnn->I5->numParaLocal;
    assignMemberLayer(cfrnn->M6, &x[count], mode);
    count += cfrnn->M6->numParaLocal;
    assignFuzzyLayer(cfrnn->F7, &x[count], mode);
    count += cfrnn->F7->numParaLocal;
    assignRoughLayer(cfrnn->R8, &x[count], mode);
    count += cfrnn->R8->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#endif

    return;
}

void ff_cnn_evoCFRNN_c(CNN_evoCFRNN_C* cfrnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode)
{
#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    MY_FLT_TYPE* vecInput = (MY_FLT_TYPE*)calloc(cfrnn->numValIn, sizeof(MY_FLT_TYPE));
    int tmp_offset = 0;
    for(int i = 0; i < cfrnn->inputChannel; i++) {
        for(int r = 0; r < cfrnn->inputHeightMax; r++) {
            for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                vecInput[tmp_offset++] = valIn[i][r][c];
            }
        }
    }
    ff_memberLayer(cfrnn->M1, vecInput, cfrnn->dataflowInitial[0][0]);
    ff_fuzzyLayer(cfrnn->F2, cfrnn->M1->degreeMembership, cfrnn->M1->dataflowStatus);
    ff_roughLayer(cfrnn->R3, cfrnn->F2->degreeRules, cfrnn->F2->dataflowStatus);
    if(cnn_evoCFRNN_c->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numOutput; i++) {
            memcpy(cnn_evoCFRNN_c->OL->inputConsequenceNode[i],
                   vecInput,
                   cnn_evoCFRNN_c->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    ff_outReduceLayer(cfrnn->OL, cfrnn->R3->degreeRough, cfrnn->R3->dataflowStatus);
    free(vecInput);
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    ff_convLayer(cfrnn->C1, valIn, cfrnn->featureMapTagInitial, &cfrnn->inputHeightMax, &cfrnn->inputWidthMax,
                 cfrnn->dataflowInitial);
    ff_poolLayer(cfrnn->P2, cfrnn->C1->featureMapData, cfrnn->C1->featureMapTag,
                 cfrnn->C1->featureMapHeight, cfrnn->C1->featureMapWidth, cfrnn->C1->dataflowStatus);
    ff_convLayer(cfrnn->C3, cfrnn->P2->featureMapData, cfrnn->P2->featureMapTag,
                 cfrnn->P2->featureMapHeight, cfrnn->P2->featureMapWidth, cfrnn->P2->dataflowStatus);
    ff_poolLayer(cfrnn->P4, cfrnn->C3->featureMapData, cfrnn->C3->featureMapTag,
                 cfrnn->C3->featureMapHeight, cfrnn->C3->featureMapWidth, cfrnn->C3->dataflowStatus);
    ff_member2DLayer(cfrnn->M5, cfrnn->P4->featureMapData, cfrnn->P4->featureMapTag,
                     cfrnn->P4->featureMapHeight, cfrnn->P4->featureMapWidth, cfrnn->P4->dataflowStatus);
    ff_fuzzyLayer(cfrnn->F6, cfrnn->M5->degreeMembership, cfrnn->M5->dataflowStatus);
    ff_roughLayer(cfrnn->R7, cfrnn->F6->degreeRules, cfrnn->F6->dataflowStatus);
    if(cnn_evoCFRNN_c->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numOutput; i++) {
            memcpy(cnn_evoCFRNN_c->OL->inputConsequenceNode[i],
                   cfrnn->M5->mean_featureMapDataIn,
                   cnn_evoCFRNN_c->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    ff_outReduceLayer(cfrnn->OL, cfrnn->R7->degreeRough, cfrnn->R7->dataflowStatus);
#else
    ff_convLayer(cfrnn->C1, valIn, cfrnn->featureMapTagInitial, &cfrnn->inputHeightMax, &cfrnn->inputWidthMax,
                 cfrnn->dataflowInitial);
    ff_poolLayer(cfrnn->P2, cfrnn->C1->featureMapData, cfrnn->C1->featureMapTag,
                 cfrnn->C1->featureMapHeight, cfrnn->C1->featureMapWidth, cfrnn->C1->dataflowStatus);
    ff_convLayer(cfrnn->C3, cfrnn->P2->featureMapData, cfrnn->P2->featureMapTag,
                 cfrnn->P2->featureMapHeight, cfrnn->P2->featureMapWidth, cfrnn->P2->dataflowStatus);
    ff_poolLayer(cfrnn->P4, cfrnn->C3->featureMapData, cfrnn->C3->featureMapTag,
                 cfrnn->C3->featureMapHeight, cfrnn->C3->featureMapWidth, cfrnn->C3->dataflowStatus);
    ff_icfcLayer(cfrnn->I5, cfrnn->P4->featureMapData, cfrnn->P4->featureMapTag,
                 cfrnn->P4->featureMapHeight, cfrnn->P4->featureMapWidth, cfrnn->P4->dataflowStatus);
    ff_memberLayer(cfrnn->M6, cfrnn->I5->outputData, cfrnn->I5->dataflowStatus);
    ff_fuzzyLayer(cfrnn->F7, cfrnn->M6->degreeMembership, cfrnn->M6->dataflowStatus);
    ff_roughLayer(cfrnn->R8, cfrnn->F7->degreeRules, cfrnn->F7->dataflowStatus);
    if(cnn_evoCFRNN_c->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cnn_evoCFRNN_c->OL->numOutput; i++) {
            memcpy(cnn_evoCFRNN_c->OL->inputConsequenceNode[i],
                   cfrnn->I5->outputData,
                   cnn_evoCFRNN_c->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    ff_outReduceLayer(cfrnn->OL, cfrnn->R8->degreeRough, cfrnn->R8->dataflowStatus);
#endif
    memcpy(valOut, cfrnn->OL->valOutputFinal, cfrnn->OL->numOutput * sizeof(MY_FLT_TYPE));

    return;
}

//////////////////////////////////////////////////////////////////////////
static int** allocINT_MOP_evoCFRNN(int nrow, int ncol)
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

static MY_FLT_TYPE** allocFLOAT_MOP_evoCFRNN(int nrow, int ncol)
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

static void getIndicators_MOP_evoCFRNN_Classify(MY_FLT_TYPE& mean_p, MY_FLT_TYPE& mean_r, MY_FLT_TYPE& mean_F,
        MY_FLT_TYPE& std_p, MY_FLT_TYPE& std_r, MY_FLT_TYPE& std_F)
{
    int len_lab = num_class_MOP_evoCFRNN;
    MY_FLT_TYPE sum_precision = cnn_evoCFRNN_c->sum_wrong / cnn_evoCFRNN_c->sum_all;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE std_Fvalue = 0;
    MY_FLT_TYPE tmp_precision[MAX_LAB_NUM_MOP_evoCFRNN];
    MY_FLT_TYPE tmp_recall[MAX_LAB_NUM_MOP_evoCFRNN];
    MY_FLT_TYPE tmp_Fvalue[MAX_LAB_NUM_MOP_evoCFRNN];
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < len_lab; i++) {
        if(cnn_evoCFRNN_c->N_TP[i] > 0) {
            tmp_precision[i] = cnn_evoCFRNN_c->N_TP[i] / (cnn_evoCFRNN_c->N_TP[i] + cnn_evoCFRNN_c->N_FP[i]);
            tmp_recall[i] = cnn_evoCFRNN_c->N_TP[i] / (cnn_evoCFRNN_c->N_TP[i] + cnn_evoCFRNN_c->N_FN[i]);
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
#if STATUS_OUT_INDEICES_MOP_evoCFRNN == FLAG_ON_MOP_evoCFRNN
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

#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
static void getNetworkComplexity(MY_FLT_TYPE& f_simpl)
{
    //
    f_simpl = 0.0;
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
    FuzzyLayer* FLptr = cnn_evoCFRNN_c->F2;
    RoughLayer* RLptr = cnn_evoCFRNN_c->R3;
    int *tmp1 = (int*)calloc(FLptr->numRules, sizeof(int));
    int *tmp2 = (int*)calloc(RLptr->numRoughSets, sizeof(int));
    for(int i = 0; i < FLptr->numRules; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < FLptr->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < FLptr->numMembershipFun[j]; k++) {
                if(FLptr->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp1[i] += ac_flag;
        }
        f_simpl_F += (double)tmp1[i] / FLptr->numInput;
    }
    f_simpl_F /= FLptr->numRules;
    for(int i = 0; i < RLptr->numRoughSets; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < RLptr->numInput; j++) {
            if(tmp1[j] && RLptr->connectStatus[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl_R += (double)tmp2[i] / RLptr->numInput;
    }
    f_simpl_R /= RLptr->numRoughSets;
    f_simpl = f_simpl_F + f_simpl_R;
    f_simpl /= 2;
    //
    return;
}
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
static void getNetworkComplexity(MY_FLT_TYPE& f_simpl)
{
    //
    f_simpl = 0.0;
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
    FuzzyLayer* FLptr = cnn_evoCFRNN_c->F6;
    RoughLayer* RLptr = cnn_evoCFRNN_c->R7;
    int *tmp1 = (int*)calloc(FLptr->numRules, sizeof(int));
    int *tmp2 = (int*)calloc(RLptr->numRoughSets, sizeof(int));
    for(int i = 0; i < FLptr->numRules; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < FLptr->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < FLptr->numMembershipFun[j]; k++) {
                if(FLptr->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp1[i] += ac_flag;
        }
        f_simpl_F += (double)tmp1[i] / FLptr->numInput;
    }
    f_simpl_F /= FLptr->numRules;
    for(int i = 0; i < RLptr->numRoughSets; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < RLptr->numInput; j++) {
            if(tmp1[j] && RLptr->connectStatus[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl_R += (double)tmp2[i] / RLptr->numInput;
    }
    f_simpl_R /= RLptr->numRoughSets;
    f_simpl = f_simpl_F + f_simpl_R;
    f_simpl /= 2;
    //
    return;
}
#else
static void getNetworkComplexity(MY_FLT_TYPE& f_simpl)
{
    //
    f_simpl = 0.0;
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
    FuzzyLayer* FLptr = cnn_evoCFRNN_c->F7;
    RoughLayer* RLptr = cnn_evoCFRNN_c->R8;
    int *tmp1 = (int*)calloc(FLptr->numRules, sizeof(int));
    int *tmp2 = (int*)calloc(RLptr->numRoughSets, sizeof(int));
    for(int i = 0; i < FLptr->numRules; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < FLptr->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < FLptr->numMembershipFun[j]; k++) {
                if(FLptr->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp1[i] += ac_flag;
        }
        f_simpl_F += (double)tmp1[i] / FLptr->numInput;
    }
    f_simpl_F /= FLptr->numRules;
    for(int i = 0; i < RLptr->numRoughSets; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < RLptr->numInput; j++) {
            if(tmp1[j] && RLptr->connectStatus[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl_R += (double)tmp2[i] / RLptr->numInput;
    }
    f_simpl_R /= RLptr->numRoughSets;
    f_simpl = f_simpl_F + f_simpl_R;
    f_simpl /= 2;
    //
    return;
}
#endif
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////