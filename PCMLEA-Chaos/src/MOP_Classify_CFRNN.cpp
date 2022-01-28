#include "MOP_cnn_data.h"
#include "MOP_Classify_CFRNN.h"
#include <float.h>
#include <math.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
#endif

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_Classify_CFRNN 0
#define FLAG_ON_MOP_Classify_CFRNN 1
#define STATUS_OUT_INDEICES_MOP_Classify_CFRNN FLAG_OFF_MOP_Classify_CFRNN

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN_MOP_Classify_CFRNN 1024
#define MAX_LAB_NUM_MOP_Classify_CFRNN 1024
#define VIOLATION_PENALTY_Classify_CFRNN 1e6

//////////////////////////////////////////////////////////////////////////
#define TAG_VALI_MOP_CLASSIFY_CFRNN -2
#define TAG_NULL_MOP_CLASSIFY_CFRNN 0
#define TAG_INVA_MOP_CLASSIFY_CFRNN -1

//////////////////////////////////////////////////////////////////////////
int NDIM_Classify_CFRNN = 0;
int NOBJ_Classify_CFRNN = 0;

//////////////////////////////////////////////////////////////////////////
int NUM_class_data_MOP_Classify_CFRNN;

LabelArr allLabels_train_MOP_Classify_CFRNN;
ImgArr   allImgs_train_MOP_Classify_CFRNN;
LabelArr allLabels_test_MOP_Classify_CFRNN;
ImgArr   allImgs_test_MOP_Classify_CFRNN;

ImgArr   allMaxInImgs_MOP_Classify_CFRNN;
ImgArr   allMinInImgs_MOP_Classify_CFRNN;
ImgArr   allMeanInImgs_MOP_Classify_CFRNN;
ImgArr   allStdInImgs_MOP_Classify_CFRNN;
ImgArr   allRangeInImgs_MOP_Classify_CFRNN;

int** ind_per_class_train_MOP_Classify_CFRNN;
int*  num_per_class_train_MOP_Classify_CFRNN;
int** ind_per_class_test_MOP_Classify_CFRNN;
int*  num_per_class_test_MOP_Classify_CFRNN;

int num_selected_train_MOP_Classify_CFRNN;
int num_selected_test_MOP_Classify_CFRNN;

int  repNum_MOP_Classify_CFRNN;
int  repNo_MOP_Classify_CFRNN;

cfrnn_Classify_CFRNN* cfrnn_Classify = NULL;

static int** allocINT_MOP_Classify_CFRNN(int nrow, int ncol);
static MY_FLT_TYPE** allocFLOAT_MOP_Classify_CFRNN(int nrow, int ncol);
static void ff_Classify_CFRNN_c(double* individual, ImgArr inputData, LabelArr outputData,
                                int** ind_per_class, int* num_per_class, int num_selected_per_class,
                                int tag_train_test);
static void getIndicators_MOP_Classify_CFRNN_c(MY_FLT_TYPE& mean_prc, MY_FLT_TYPE& std_prc, MY_FLT_TYPE& mean_rec,
        MY_FLT_TYPE& std_rec, MY_FLT_TYPE& mean_ber, MY_FLT_TYPE& std_ber);
static void getFitness_MOP_Classify_CFRNN_c(double* fitness);
static void read_int_from_file_MOP_Classify_CFRNN(const char* filename, int* vec_int, int num_int);
static void generate_img_statistics_MOP_Classify_CFRNN(ImgArr& imgmin, ImgArr& imgmax, ImgArr& imgmean, ImgArr& imgstd,
        ImgArr& imgrange,
        ImgArr allimgs, LabelArr alllabels);
//
int     seed_Classify_CFRNN = 237;
long    rnd_uni_init_Classify_CFRNN = -(long)seed_Classify_CFRNN;
static double rnd_uni_Classify_CFRNN(long* idum);
static int rnd_Classify_CFRNN(int low, int high);
static void shuffle_Classify_CFRNN(int* x, int size);

//////////////////////////////////////////////////////////////////////////
void Initialize_Classify_CFRNN(char* pro, int curN, int numN, int my_rank)
{
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    seed_Classify_CFRNN = 237 + my_rank;
    seed_Classify_CFRNN = seed_Classify_CFRNN % 1235;
    rnd_uni_init_Classify_CFRNN = -(long)seed_Classify_CFRNN;
    for(int i = 0; i < curN; i++) {
        seed_Classify_CFRNN = (seed_Classify_CFRNN + 111) % 1235;
        rnd_uni_init_Classify_CFRNN = -(long)seed_Classify_CFRNN;
    }
    //
    repNo_MOP_Classify_CFRNN = curN;
    repNum_MOP_Classify_CFRNN = numN;
    //
    char filename[MAX_STR_LEN_MOP_Classify_CFRNN];
    if(!strcmp(pro, "Classify_CFRNN_MNIST")) {
        NUM_class_data_MOP_Classify_CFRNN = 10;
        sprintf(filename, "../Data_all/Data_MNIST/train-images.idx3-ubyte");
        allImgs_train_MOP_Classify_CFRNN = read_Img_IDX_FILE(filename);
        sprintf(filename, "../Data_all/Data_MNIST/train-labels.idx1-ubyte");
        allLabels_train_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
        sprintf(filename, "../Data_all/Data_MNIST/t10k-images.idx3-ubyte");
        allImgs_test_MOP_Classify_CFRNN = read_Img_IDX_FILE(filename);
        sprintf(filename, "../Data_all/Data_MNIST/t10k-labels.idx1-ubyte");
        allLabels_test_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
    } else if(!strcmp(pro, "Classify_CFRNN_FashionMNIST")) {
        NUM_class_data_MOP_Classify_CFRNN = 10;
        sprintf(filename, "../Data_all/Data_FashionMNIST/train-images-idx3-ubyte");
        allImgs_train_MOP_Classify_CFRNN = read_Img_IDX_FILE(filename);
        sprintf(filename, "../Data_all/Data_FashionMNIST/train-labels-idx1-ubyte");
        allLabels_train_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
        sprintf(filename, "../Data_all/Data_FashionMNIST/t10k-images-idx3-ubyte");
        allImgs_test_MOP_Classify_CFRNN = read_Img_IDX_FILE(filename);
        sprintf(filename, "../Data_all/Data_FashionMNIST/t10k-labels-idx1-ubyte");
        allLabels_test_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
    } else if(!strcmp(pro, "Classify_CFRNN_MNIST_FM")) {
        NUM_class_data_MOP_Classify_CFRNN = 10;
        int nChns = 16;
        int tH = 5;
        int tW = 5;
        int nSamp_train = 60000;
        int nSamp_test = 10000;
        sprintf(filename, "../Data_all/Data_MNIST/mnist_train_fm");
        allImgs_train_MOP_Classify_CFRNN = read_Img_tmp(filename, nSamp_train, nChns, tH, tW);
        sprintf(filename, "../Data_all/Data_MNIST/train-labels.idx1-ubyte");
        allLabels_train_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
        sprintf(filename, "../Data_all/Data_MNIST/mnist_test_fm");
        allImgs_test_MOP_Classify_CFRNN = read_Img_tmp(filename, nSamp_test, nChns, tH, tW);
        sprintf(filename, "../Data_all/Data_MNIST/t10k-labels.idx1-ubyte");
        allLabels_test_MOP_Classify_CFRNN = read_Label_IDX_FILE(filename, NUM_class_data_MOP_Classify_CFRNN);
    } else {
        int NUM_feature_MOP_Classify_CFRNN;
        int NUM_samples_train_MOP_Classify_CFRNN;
        int NUM_samples_test_MOP_Classify_CFRNN;
        int IMG_side_len_MOP_Classify_CFRNN;
        //
        sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_num_class");
        read_int_from_file_MOP_Classify_CFRNN(filename, &NUM_class_data_MOP_Classify_CFRNN, 1);
        //
        sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_side_length_F%d", curN + 1);
        read_int_from_file_MOP_Classify_CFRNN(filename, &IMG_side_len_MOP_Classify_CFRNN, 1);
        //
        sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_num_feature_F%d", curN + 1);
        read_int_from_file_MOP_Classify_CFRNN(filename, &NUM_feature_MOP_Classify_CFRNN, 1);
        //
        int* tmp_vec_ind = (int*)malloc(NUM_feature_MOP_Classify_CFRNN * sizeof(int));
        sprintf(filename, "../Data_all/Data_CNN_Indus/Mat_ind_F%d.txt", curN + 1);
        read_int_from_file_MOP_Classify_CFRNN(filename, tmp_vec_ind, NUM_feature_MOP_Classify_CFRNN);
        //
        sprintf(filename, "../Data_all/Data_CNN_Indus/F%d/SECOM_nrow_train_F%d", curN + 1, curN + 1);
        read_int_from_file_MOP_Classify_CFRNN(filename, &NUM_samples_train_MOP_Classify_CFRNN, 1);
        //
        sprintf(filename, "../Data_all/Data_CNN_Indus/F%d/SECOM_nrow_test_F%d", curN + 1, curN + 1);
        read_int_from_file_MOP_Classify_CFRNN(filename, &NUM_samples_test_MOP_Classify_CFRNN, 1);
        //
        int NUM_samples_all_MOP_Classify_CFRNN =
            NUM_samples_train_MOP_Classify_CFRNN +
            NUM_samples_test_MOP_Classify_CFRNN;
        //
        int** tmp_mat_ind = (int**)malloc(IMG_side_len_MOP_Classify_CFRNN * sizeof(int*));
        for(int i = 0; i < IMG_side_len_MOP_Classify_CFRNN; i++)
            tmp_mat_ind[i] = (int*)malloc(IMG_side_len_MOP_Classify_CFRNN * sizeof(int));
        for(int r = 0; r < IMG_side_len_MOP_Classify_CFRNN; r++) {
            for(int c = 0; c < IMG_side_len_MOP_Classify_CFRNN; c++) {
                int cur_count = r * IMG_side_len_MOP_Classify_CFRNN + c;
                if(cur_count < NUM_feature_MOP_Classify_CFRNN) {
                    tmp_mat_ind[r][c] = tmp_vec_ind[cur_count];
                } else {
                    tmp_mat_ind[r][c] = -1;
                }
            }
        }
        //
        char filetrain[MAX_STR_LEN_MOP_Classify_CFRNN];
        char filetest[MAX_STR_LEN_MOP_Classify_CFRNN];
        sprintf(filetrain, "../Data_all/Data_CNN_Indus/F%d/SECOM_samples_train_F%d", curN + 1, curN + 1);
        sprintf(filetest, "../Data_all/Data_CNN_Indus/F%d/SECOM_samples_test_F%d", curN + 1, curN + 1);
        allImgs_train_MOP_Classify_CFRNN = read_Img_table(filetrain, filetest,
                                           NUM_samples_train_MOP_Classify_CFRNN, 0,
                                           NUM_feature_MOP_Classify_CFRNN, IMG_side_len_MOP_Classify_CFRNN, IMG_side_len_MOP_Classify_CFRNN,
                                           tmp_mat_ind);
        allImgs_test_MOP_Classify_CFRNN = read_Img_table(filetrain, filetest,
                                          0, NUM_samples_test_MOP_Classify_CFRNN,
                                          NUM_feature_MOP_Classify_CFRNN, IMG_side_len_MOP_Classify_CFRNN, IMG_side_len_MOP_Classify_CFRNN,
                                          tmp_mat_ind);
        sprintf(filetrain, "../Data_all/Data_CNN_Indus/F%d/SECOM_labels_train_F%d", curN + 1, curN + 1);
        sprintf(filetest, "../Data_all/Data_CNN_Indus/F%d/SECOM_labels_test_F%d", curN + 1, curN + 1);
        allLabels_train_MOP_Classify_CFRNN = read_Lable_tabel(filetrain, filetest,
                                             NUM_samples_train_MOP_Classify_CFRNN, 0,
                                             NUM_class_data_MOP_Classify_CFRNN);
        allLabels_test_MOP_Classify_CFRNN = read_Lable_tabel(filetrain, filetest,
                                            0, NUM_samples_test_MOP_Classify_CFRNN,
                                            NUM_class_data_MOP_Classify_CFRNN);
        //
        free(tmp_vec_ind);
        for(int i = 0; i < IMG_side_len_MOP_Classify_CFRNN; i++) {
            free(tmp_mat_ind[i]);
        }
        free(tmp_mat_ind);
    }
    //
    ind_per_class_train_MOP_Classify_CFRNN = allocINT_MOP_Classify_CFRNN(
                allLabels_train_MOP_Classify_CFRNN->LabelPtr[0].l,
                allLabels_train_MOP_Classify_CFRNN->LabelNum);
    num_per_class_train_MOP_Classify_CFRNN = (int*)calloc(allLabels_train_MOP_Classify_CFRNN->LabelPtr[0].l, sizeof(int));
    ind_per_class_test_MOP_Classify_CFRNN = allocINT_MOP_Classify_CFRNN(
            allLabels_test_MOP_Classify_CFRNN->LabelPtr[0].l,
            allLabels_test_MOP_Classify_CFRNN->LabelNum);
    num_per_class_test_MOP_Classify_CFRNN = (int*)calloc(allLabels_test_MOP_Classify_CFRNN->LabelPtr[0].l, sizeof(int));
    for(int i = 0; i < allLabels_train_MOP_Classify_CFRNN->LabelNum; i++) {
        int cur_label = 0;
        MY_FLT_TYPE cur_out = allLabels_train_MOP_Classify_CFRNN->LabelPtr[i].LabelData[0];
        for(int j = 1; j < allLabels_train_MOP_Classify_CFRNN->LabelPtr[i].l; j++) {
            MY_FLT_TYPE tmp_out = allLabels_train_MOP_Classify_CFRNN->LabelPtr[i].LabelData[j];
            if(cur_out < tmp_out) {
                cur_out = tmp_out;
                cur_label = j;
            }
        }
        ind_per_class_train_MOP_Classify_CFRNN[cur_label][num_per_class_train_MOP_Classify_CFRNN[cur_label]] = i;
        num_per_class_train_MOP_Classify_CFRNN[cur_label]++;
    }
    for(int i = 0; i < allLabels_test_MOP_Classify_CFRNN->LabelNum; i++) {
        int cur_label = 0;
        MY_FLT_TYPE cur_out = allLabels_test_MOP_Classify_CFRNN->LabelPtr[i].LabelData[0];
        for(int j = 1; j < allLabels_test_MOP_Classify_CFRNN->LabelPtr[i].l; j++) {
            MY_FLT_TYPE tmp_out = allLabels_test_MOP_Classify_CFRNN->LabelPtr[i].LabelData[j];
            if(cur_out < tmp_out) {
                cur_out = tmp_out;
                cur_label = j;
            }
        }
        ind_per_class_test_MOP_Classify_CFRNN[cur_label][num_per_class_test_MOP_Classify_CFRNN[cur_label]] = i;
        num_per_class_test_MOP_Classify_CFRNN[cur_label]++;
    }
    //
    //int sum_tmp = 0;
    //for(int i = 0; i < allLabels_train_MOP_Classify_CFRNN->LabelPtr[0].l; i++) {
    //    printf("%d ", num_per_class_train_MOP_Classify_CFRNN[i]);
    //    sum_tmp += num_per_class_train_MOP_Classify_CFRNN[i];
    //}
    //printf("sum = %d\n", sum_tmp);
    //sum_tmp = 0;
    //for(int i = 0; i < allLabels_test_MOP_Classify_CFRNN->LabelPtr[0].l; i++) {
    //    printf("%d ", num_per_class_test_MOP_Classify_CFRNN[i]);
    //    sum_tmp += num_per_class_test_MOP_Classify_CFRNN[i];
    //}
    //printf("sum = %d\n", sum_tmp);
    //
    int tmp_min_train = allLabels_train_MOP_Classify_CFRNN->LabelNum;
    for(int i = 0; i < NUM_class_data_MOP_Classify_CFRNN; i++) {
        if(tmp_min_train > num_per_class_train_MOP_Classify_CFRNN[i])
            tmp_min_train = num_per_class_train_MOP_Classify_CFRNN[i];
    }
    if(!strcmp(pro, "Classify_CFRNN_MNIST") ||
       !strcmp(pro, "Classify_CFRNN_FashionMNIST")) {
        num_selected_train_MOP_Classify_CFRNN = allLabels_train_MOP_Classify_CFRNN->LabelNum;
    } else {
        num_selected_train_MOP_Classify_CFRNN = tmp_min_train;
    }
    num_selected_test_MOP_Classify_CFRNN = allLabels_test_MOP_Classify_CFRNN->LabelNum;
    //
    generate_img_statistics_MOP_Classify_CFRNN(allMinInImgs_MOP_Classify_CFRNN, allMaxInImgs_MOP_Classify_CFRNN,
            allMeanInImgs_MOP_Classify_CFRNN, allStdInImgs_MOP_Classify_CFRNN,
            allRangeInImgs_MOP_Classify_CFRNN,
            allImgs_train_MOP_Classify_CFRNN, allLabels_train_MOP_Classify_CFRNN);
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    cfrnn_Classify = (cfrnn_Classify_CFRNN*)calloc(1, sizeof(cfrnn_Classify_CFRNN));
    cfrnn_Classify_CFRNN_setup(cfrnn_Classify);
    //
    NDIM_Classify_CFRNN = cfrnn_Classify->numParaLocal;
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    NOBJ_Classify_CFRNN = 3;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    NOBJ_Classify_CFRNN = 2;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    NOBJ_Classify_CFRNN = 2;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    NOBJ_Classify_CFRNN = 3;
#endif
    //
    return;
}
void SetLimits_Classify_CFRNN(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    for(int i = 0; i < cfrnn_Classify->C1->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->C1->xMin[i];
        maxLimit[count] = cfrnn_Classify->C1->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P2->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->P2->xMin[i];
        maxLimit[count] = cfrnn_Classify->P2->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->C3->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->C3->xMin[i];
        maxLimit[count] = cfrnn_Classify->C3->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P4->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->P4->xMin[i];
        maxLimit[count] = cfrnn_Classify->P4->xMax[i];
        count++;
    }
    for(int n = 0; n < cfrnn_Classify->P4->channelsInOutMax; n++) {
        for(int i = 0; i < cfrnn_Classify->M5[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->M5[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->M5[n]->xMax[i];
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F6[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->F6[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->F6[n]->xMax[i];
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R7->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->R7->xMin[i];
        maxLimit[count] = cfrnn_Classify->R7->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->OL->xMin[i];
        maxLimit[count] = cfrnn_Classify->OL->xMax[i];
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    for(int i = 0; i < cfrnn_Classify->C1->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->C1->xMin[i];
        maxLimit[count] = cfrnn_Classify->C1->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P2->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->P2->xMin[i];
        maxLimit[count] = cfrnn_Classify->P2->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->C3->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->C3->xMin[i];
        maxLimit[count] = cfrnn_Classify->C3->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P4->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->P4->xMin[i];
        maxLimit[count] = cfrnn_Classify->P4->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->OL->xMin[i];
        maxLimit[count] = cfrnn_Classify->OL->xMax[i];
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    for(int n = 0; n < cfrnn_Classify->regionNum; n++) {
        for(int i = 0; i < cfrnn_Classify->M1[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->M1[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->M1[n]->xMax[i];
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F2[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->F2[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->F2[n]->xMax[i];
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R3->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->R3->xMin[i];
        maxLimit[count] = cfrnn_Classify->R3->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->OL->xMin[i];
        maxLimit[count] = cfrnn_Classify->OL->xMax[i];
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    for(int n = 0; n < cfrnn_Classify->inputChannel; n++) {
        for(int i = 0; i < cfrnn_Classify->M1[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->M1[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->M1[n]->xMax[i];
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F2[n]->numParaLocal; i++) {
            minLimit[count] = cfrnn_Classify->F2[n]->xMin[i];
            maxLimit[count] = cfrnn_Classify->F2[n]->xMax[i];
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R3->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->R3->xMin[i];
        maxLimit[count] = cfrnn_Classify->R3->xMax[i];
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
        minLimit[count] = cfrnn_Classify->OL->xMin[i];
        maxLimit[count] = cfrnn_Classify->OL->xMax[i];
        count++;
    }
#endif
    return;
}

int CheckLimits_Classify_CFRNN(double* x, int nx)
{
    int count = 0;
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    for(int i = 0; i < cfrnn_Classify->C1->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->C1->xMin[i] ||
           x[count] > cfrnn_Classify->C1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->C1->xMin[i], cfrnn_Classify->C1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P2->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->P2->xMin[i] ||
           x[count] > cfrnn_Classify->P2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->P2->xMin[i], cfrnn_Classify->P2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->C3->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->C3->xMin[i] ||
           x[count] > cfrnn_Classify->C3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->C3->xMin[i], cfrnn_Classify->C3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P4->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->P4->xMin[i] ||
           x[count] > cfrnn_Classify->P4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->P4->xMin[i], cfrnn_Classify->P4->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int n = 0; n < cfrnn_Classify->P4->channelsInOutMax; n++) {
        for(int i = 0; i < cfrnn_Classify->M5[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->M5[n]->xMin[i] ||
               x[count] > cfrnn_Classify->M5[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->M6 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->M5[n]->xMin[i], cfrnn_Classify->M5[n]->xMax[i]);
                return 0;
            }
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F6[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->F6[n]->xMin[i] ||
               x[count] > cfrnn_Classify->F6[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->F7 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->F6[n]->xMin[i], cfrnn_Classify->F6[n]->xMax[i]);
                return 0;
            }
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R7->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->R7->xMin[i] ||
           x[count] > cfrnn_Classify->R7->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->R8 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->R7->xMin[i], cfrnn_Classify->R7->xMax[i]);
            return 0;
        }
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    for(int i = 0; i < cfrnn_Classify->C1->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->C1->xMin[i] ||
           x[count] > cfrnn_Classify->C1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->C1->xMin[i], cfrnn_Classify->C1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P2->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->P2->xMin[i] ||
           x[count] > cfrnn_Classify->P2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->P2->xMin[i], cfrnn_Classify->P2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->C3->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->C3->xMin[i] ||
           x[count] > cfrnn_Classify->C3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->C3->xMin[i], cfrnn_Classify->C3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < cfrnn_Classify->P4->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->P4->xMin[i] ||
           x[count] > cfrnn_Classify->P4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->P4->xMin[i], cfrnn_Classify->P4->xMax[i]);
            return 0;
        }
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    for(int n = 0; n < cfrnn_Classify->regionNum; n++) {
        for(int i = 0; i < cfrnn_Classify->M1[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->M1[n]->xMin[i] ||
               x[count] > cfrnn_Classify->M1[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->M1[n]->xMin[i], cfrnn_Classify->M1[n]->xMax[i]);
                return 0;
            }
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F2[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->F2[n]->xMin[i] ||
               x[count] > cfrnn_Classify->F2[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->F2[n]->xMin[i], cfrnn_Classify->F2[n]->xMax[i]);
                return 0;
            }
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R3->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->R3->xMin[i] ||
           x[count] > cfrnn_Classify->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->R3->xMin[i], cfrnn_Classify->R3->xMax[i]);
            return 0;
        }
        count++;
    }
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    for(int n = 0; n < cfrnn_Classify->inputChannel; n++) {
        for(int i = 0; i < cfrnn_Classify->M1[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->M1[n]->xMin[i] ||
               x[count] > cfrnn_Classify->M1[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C1 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->M1[n]->xMin[i], cfrnn_Classify->M1[n]->xMax[i]);
                return 0;
            }
            count++;
        }
        for(int i = 0; i < cfrnn_Classify->F2[n]->numParaLocal; i++) {
            if(x[count] < cfrnn_Classify->F2[n]->xMin[i] ||
               x[count] > cfrnn_Classify->F2[n]->xMax[i]) {
                printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->P2 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], cfrnn_Classify->F2[n]->xMin[i], cfrnn_Classify->F2[n]->xMax[i]);
                return 0;
            }
            count++;
        }
    }
    for(int i = 0; i < cfrnn_Classify->R3->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->R3->xMin[i] ||
           x[count] > cfrnn_Classify->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->C3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->R3->xMin[i], cfrnn_Classify->R3->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
    //
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->OL->xMin[i] ||
           x[count] > cfrnn_Classify->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->OL %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->OL->xMin[i], cfrnn_Classify->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#else
    if(cfrnn_Classify->flagConnectStatus != FLAG_STATUS_OFF ||
       cfrnn_Classify->flagConnectWeight != FLAG_STATUS_ON ||
       cfrnn_Classify->typeCoding != PARA_CODING_DIRECT) {
        printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
               __FILE__, __LINE__, cfrnn_Classify->flagConnectStatus, cfrnn_Classify->flagConnectWeight, cfrnn_Classify->typeCoding);
        exit(-275082);
    }
    int tmp_offset = cfrnn_Classify->OL->numOutput * cfrnn_Classify->OL->numInput;
    count += tmp_offset;
    for(int i = tmp_offset; i < cfrnn_Classify->OL->numParaLocal; i++) {
        if(x[count] < cfrnn_Classify->OL->xMin[i] ||
           x[count] > cfrnn_Classify->OL->xMax[i]) {
            printf("%s(%d): Check limits FAIL - evoCNN_Classify: cnn_evoCNN_c->OL %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], cfrnn_Classify->OL->xMin[i], cfrnn_Classify->OL->xMax[i]);
            return 0;
        }
        count++;
    }
#endif
    //
    return 1;
}

void Fitness_Classify_CFRNN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    ff_Classify_CFRNN_c(individual, allImgs_train_MOP_Classify_CFRNN, allLabels_train_MOP_Classify_CFRNN,
                        ind_per_class_train_MOP_Classify_CFRNN, num_per_class_train_MOP_Classify_CFRNN,
                        num_selected_train_MOP_Classify_CFRNN,
                        TRAIN_TAG_MOP_CLASSIFY_CFRNN);
    //
    getFitness_MOP_Classify_CFRNN_c(fitness);
    //
    return;
}

void Fitness_Classify_CFRNN_test(double* individual, double* fitness)
{
    ff_Classify_CFRNN_c(individual, allImgs_test_MOP_Classify_CFRNN, allLabels_test_MOP_Classify_CFRNN,
                        ind_per_class_test_MOP_Classify_CFRNN, num_per_class_test_MOP_Classify_CFRNN,
                        num_selected_test_MOP_Classify_CFRNN,
                        TEST_TAG_MOP_CLASSIFY_CFRNN);
    //
    getFitness_MOP_Classify_CFRNN_c(fitness);
    //
    return;
}

void ff_Classify_CFRNN_c(double* individual, ImgArr inputData, LabelArr outputData,
                         int** ind_per_class, int* num_per_class, int num_selected_per_class,
                         int tag_train_test)
{
    int num_sample = inputData->ImgNum;
    int len_lab = NUM_class_data_MOP_Classify_CFRNN;
    cfrnn_Classify->sum_all = 0;
    cfrnn_Classify->sum_wrong = 0;
    for(int i = 0; i < cfrnn_Classify->numOutput; i++) {
        cfrnn_Classify->N_sum[i] = 0;
        cfrnn_Classify->N_wrong[i] = 0;
        cfrnn_Classify->e_sum[i] = 0;
        cfrnn_Classify->N_TP[i] = 0;
        cfrnn_Classify->N_TN[i] = 0;
        cfrnn_Classify->N_FP[i] = 0;
        cfrnn_Classify->N_FN[i] = 0;
    }
    cfrnn_Classify_CFRNN_init(cfrnn_Classify, individual, ASSIGN_MODE_FRNN);
    //double* copy_individual = (double*)calloc(NDIM_Classify_CFRNN, sizeof(double));
    //memcpy(copy_individual, individual, NDIM_Classify_CFRNN * sizeof(double));
    ////cfrnn_Classify->OL->connectStatus[0][0] = -2;
    ////cfrnn_Classify->OL->connectWeight[0][0] = -1;
    //cfrnn_Classify_CFRNN_init(cfrnn_Classify, individual, OUTPUT_ALL_MODE_FRNN);
    //int count = 0;
    //for(int i = 0; i < NDIM_Classify_CFRNN; i++) {
    //    if(individual[i] != copy_individual[i]) {
    //        count++;
    //        if(individual[i] != (int)individual[i] ||
    //           copy_individual[i] - individual[i] > 1)
    //            printf("(%d~%lf~%lf)", i, individual[i], copy_individual[i]);
    //    }
    //}
    //printf("\nNUM = (%d~%d)\n", count, cfrnn_Classify->numParaLocal_disc);
    ////
    //int tmp_offset = 0;
    //for(int n = 0; n < cfrnn_Classify->regionNum; n++) {
    //    count = 0;
    //    for(int i = 0; i < cfrnn_Classify->M1[n]->numParaLocal; i++) {
    //        int tmp_ind = tmp_offset + i;
    //        if(individual[tmp_ind] != copy_individual[tmp_ind]) {
    //            count++;
    //            if(individual[tmp_ind] != (int)individual[tmp_ind] ||
    //               copy_individual[tmp_ind] - individual[tmp_ind] > 1)
    //                printf("(%d~%lf~%lf)", i, individual[tmp_ind], copy_individual[tmp_ind]);
    //        }
    //    }
    //    tmp_offset += cfrnn_Classify->M1[n]->numParaLocal;
    //    printf("\nM%d~NUM = (%d~%d)\n", n, count, cfrnn_Classify->M1[n]->numParaLocal_disc);
    //    count = 0;
    //    for(int i = 0; i < cfrnn_Classify->F2[n]->numParaLocal; i++) {
    //        int tmp_ind = tmp_offset + i;
    //        if(individual[tmp_ind] != copy_individual[tmp_ind]) {
    //            count++;
    //            if(individual[tmp_ind] != (int)individual[tmp_ind] ||
    //               copy_individual[tmp_ind] - individual[tmp_ind] > 1)
    //                printf("(%d~%lf~%lf)", i, individual[tmp_ind], copy_individual[tmp_ind]);
    //        }
    //    }
    //    tmp_offset += cfrnn_Classify->F2[n]->numParaLocal;
    //    printf("\nF%d~NUM = (%d~%d)\n", n, count, cfrnn_Classify->F2[n]->numParaLocal_disc);
    //}
    //count = 0;
    //for(int i = 0; i < cfrnn_Classify->R3->numParaLocal; i++) {
    //    int tmp_ind = tmp_offset + i;
    //    if(individual[tmp_ind] != copy_individual[tmp_ind]) {
    //        count++;
    //        if(individual[tmp_ind] != (int)individual[tmp_ind] ||
    //           copy_individual[tmp_ind] - individual[tmp_ind] > 1)
    //            printf("(%d~%lf~%lf)", i, individual[tmp_ind], copy_individual[tmp_ind]);
    //    }
    //}
    //tmp_offset += cfrnn_Classify->R3->numParaLocal;
    //printf("\nR~NUM = (%d~%d)\n", count, cfrnn_Classify->R3->numParaLocal_disc);
    //count = 0;
    //for(int i = 0; i < cfrnn_Classify->OL->numParaLocal; i++) {
    //    int tmp_ind = tmp_offset + i;
    //    if(individual[tmp_ind] != copy_individual[tmp_ind]) {
    //        count++;
    //        if(individual[tmp_ind] != (int)individual[tmp_ind] ||
    //           copy_individual[tmp_ind] - individual[tmp_ind] > 1)
    //            printf("(%d~%lf~%lf)", i, individual[tmp_ind], copy_individual[tmp_ind]);
    //    }
    //}
    //printf("\nO~NUM = (%d~%d)\n", count, cfrnn_Classify->OL->numParaLocal_disc);
    ////
    //int test = 0;
    //free(copy_individual);
    //if(mpi_rank_MOP_Classify_CFRNN == 0) printf("cfrnn_Classify_CFRNN_init.\n");
    //
    MY_FLT_TYPE*** valIn;
    MY_FLT_TYPE valOut[MAX_LAB_NUM_MOP_Classify_CFRNN];
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    int matStoreType = LAPACK_ROW_MAJOR;
    if(matStoreType != LAPACK_ROW_MAJOR) {
        printf("%s(%d): The data should be stored in row wise for LAPACKE, exiting...\n",
               __FILE__, __LINE__);
        exit(-123457);
    }
    MY_FLT_TYPE* matA = NULL;
    MY_FLT_TYPE* matB = NULL;
    MY_FLT_TYPE* matLeft = NULL;
    MY_FLT_TYPE* matRight = NULL;
    if(tag_train_test == TRAIN_TAG_MOP_CLASSIFY_CFRNN) {
        matA = (MY_FLT_TYPE*)calloc(num_sample * cfrnn_Classify->OL->numInput, sizeof(MY_FLT_TYPE));
        matB = (MY_FLT_TYPE*)calloc(num_sample * cfrnn_Classify->OL->numOutput, sizeof(MY_FLT_TYPE));
        matLeft = (MY_FLT_TYPE*)calloc(cfrnn_Classify->OL->numInput * cfrnn_Classify->OL->numInput, sizeof(MY_FLT_TYPE));
        matRight = (MY_FLT_TYPE*)calloc(cfrnn_Classify->OL->numInput * cfrnn_Classify->OL->numOutput, sizeof(MY_FLT_TYPE));
    }
    int tmp_count_samples = 0;
#endif
    for(int iClass = 0; iClass < len_lab; iClass++) {
        shuffle_Classify_CFRNN(ind_per_class[iClass], num_per_class[iClass]);
        for(int n = 0; n < num_per_class[iClass] && n < num_selected_per_class; n++) {
            int m = ind_per_class[iClass][n];
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
            valIn = (MY_FLT_TYPE***)malloc(cfrnn_Classify->inputChannel * sizeof(MY_FLT_TYPE**));
            for(int curChn = 0; curChn < cfrnn_Classify->inputChannel; curChn++)
                valIn[curChn] = inputData[curChn].ImgPtr[m].ImgData;
            ff_cfrnn_Classify_CFRNN(cfrnn_Classify, valIn, valOut, NULL);
            free(valIn);
#else
            valIn = &inputData->ImgPtr[m].ImgData;
            ff_cfrnn_Classify_CFRNN(cfrnn_Classify, valIn, valOut, NULL);
#endif
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 1; j < cfrnn_Classify->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE tmp_max_lab_val = outputData->LabelPtr[m].LabelData[0];
            for(int j = 1; j < cfrnn_Classify->numOutput; j++) {
                if(tmp_max_lab_val < outputData->LabelPtr[m].LabelData[j]) {
                    tmp_max_lab_val = outputData->LabelPtr[m].LabelData[j];
                    true_label = j;
                }
            }
            for(int j = 0; j < cfrnn_Classify->numOutput; j++) {
                if(j == cur_label && j == true_label) cfrnn_Classify->N_TP[j]++;
                if(j == cur_label && j != true_label) cfrnn_Classify->N_FP[j]++;
                if(j != cur_label && j == true_label) cfrnn_Classify->N_FN[j]++;
                if(j != cur_label && j != true_label) cfrnn_Classify->N_TN[j]++;
            }
            cfrnn_Classify->sum_all++;
            cfrnn_Classify->N_sum[true_label]++;
            if(cur_label != true_label) {
                cfrnn_Classify->sum_wrong++;
                cfrnn_Classify->N_wrong[true_label]++;
            }
            //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
            if(tag_train_test == TRAIN_TAG_MOP_CLASSIFY_CFRNN) {
                for(int j = 0; j < cfrnn_Classify->OL->numInput; j++) {
                    int ind_cur;// = j * num_sample + m;
                    if(matStoreType == LAPACK_ROW_MAJOR)
                        ind_cur = tmp_count_samples * cfrnn_Classify->OL->numInput + j;
                    matA[ind_cur] = cfrnn_Classify->OL->valInputFinal[0][j];
                }
                for(int j = 0; j < cfrnn_Classify->OL->numOutput; j++) {
                    int ind_cur;// = j * num_sample + m;
                    if(matStoreType == LAPACK_ROW_MAJOR)
                        ind_cur = tmp_count_samples * cfrnn_Classify->OL->numOutput + j;
                    matB[ind_cur] = (outputData->LabelPtr[m].LabelData[j] - 1) * 2 + 1;
                }
                tmp_count_samples++;
            }
#endif
        }
    }
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    if(tag_train_test == TRAIN_TAG_MOP_CLASSIFY_CFRNN) {
        cfrnn_Classify->sum_all = 0;
        cfrnn_Classify->sum_wrong = 0;
        for(int i = 0; i < cfrnn_Classify->numOutput; i++) {
            cfrnn_Classify->N_sum[i] = 0;
            cfrnn_Classify->N_wrong[i] = 0;
            cfrnn_Classify->e_sum[i] = 0;
            cfrnn_Classify->N_TP[i] = 0;
            cfrnn_Classify->N_TN[i] = 0;
            cfrnn_Classify->N_FP[i] = 0;
            cfrnn_Classify->N_FN[i] = 0;
        }
        //
        MY_FLT_TYPE lambda = 9.3132e-10;
        for(int i = 0; i < cfrnn_Classify->OL->numInput; i++) {
            for(int j = 0; j < cfrnn_Classify->OL->numInput; j++) {
                int tmp_o0;// = j * cfrnn_Classify->OL->numInput + i;
                if(matStoreType == LAPACK_ROW_MAJOR)
                    tmp_o0 = i * cfrnn_Classify->OL->numInput + j;
                for(int k = 0; k < tmp_count_samples; k++) {
                    int tmp_i1;// = i * num_sample + k;
                    int tmp_i2;// = j * num_sample + k;
                    if(matStoreType == LAPACK_ROW_MAJOR) {
                        tmp_i1 = k * cfrnn_Classify->OL->numInput + i;
                        tmp_i2 = k * cfrnn_Classify->OL->numInput + j;
                    }
                    matLeft[tmp_o0] += matA[tmp_i1] * matA[tmp_i2];
                }
                if(i == j)
                    matLeft[tmp_o0] += lambda;
            }
        }
        for(int i = 0; i < cfrnn_Classify->OL->numInput; i++) {
            for(int j = 0; j < cfrnn_Classify->OL->numOutput; j++) {
                int tmp_o0;// = j * cfrnn_Classify->OL->numInput + i;
                if(matStoreType == LAPACK_ROW_MAJOR)
                    tmp_o0 = i * cfrnn_Classify->OL->numOutput + j;
                for(int k = 0; k < tmp_count_samples; k++) {
                    int tmp_i1;// = i * num_sample + k;
                    int tmp_i2;// = j * num_sample + k;
                    if(matStoreType == LAPACK_ROW_MAJOR) {
                        tmp_i1 = k * cfrnn_Classify->OL->numInput + i;
                        tmp_i2 = k * cfrnn_Classify->OL->numOutput + j;
                    }
                    matRight[tmp_o0] += matA[tmp_i1] * matB[tmp_i2];
                }
            }
        }
        int N = cfrnn_Classify->OL->numInput;
        int NRHS = cfrnn_Classify->OL->numOutput;
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
        for(int i = 0; i < cfrnn_Classify->OL->numInput; i++) {
            for(int j = 0; j < cfrnn_Classify->OL->numOutput; j++) {
                int ind_cur;// = j * cfrnn_Classify->OL->numInput + i;
                if(matStoreType == LAPACK_ROW_MAJOR)
                    ind_cur = i * cfrnn_Classify->OL->numOutput + j;
                cfrnn_Classify->OL->connectWeight[j][i] = matRight[ind_cur];
            }
        }
        cfrnn_Classify_CFRNN_init(cfrnn_Classify, individual, OUTPUT_CONTINUOUS_MODE_FRNN);
        //
        for(int m = 0; m < tmp_count_samples; m++) {
            //if(mpi_rank_MOP_Classify_CFRNN == 0 && m >= 1317 && m < 1320)
            //    printf("for(int m = 0; m < num_sample; m++) - m = %d.\n", m);
            for(int j = 0; j < cfrnn_Classify->OL->numOutput; j++) {
                valOut[j] = 0;
                for(int k = 0; k < cfrnn_Classify->OL->numInput; k++) {
                    int ind_cur;// = k * num_sample + m;
                    if(matStoreType == LAPACK_ROW_MAJOR)
                        ind_cur = m * cfrnn_Classify->OL->numInput + k;
                    valOut[j] += matA[ind_cur] * cfrnn_Classify->OL->connectWeight[j][k];
                }
                if(CHECK_INVALID(valOut[j])) {
                    printf("%d~%lf", j, valOut[j]);
                }
            }
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 1; j < cfrnn_Classify->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
            }
            int true_label = 0;
            MY_FLT_TYPE tmp_max_lab_val = matB[m * cfrnn_Classify->OL->numOutput + 0];//outputData->LabelPtr[m].LabelData[0];
            for(int j = 1; j < cfrnn_Classify->numOutput; j++) {
                int ind_cur;// = j * num_sample + m;
                if(matStoreType == LAPACK_ROW_MAJOR)
                    ind_cur = m * cfrnn_Classify->OL->numOutput + j;
                if(tmp_max_lab_val < matB[ind_cur]) {
                    tmp_max_lab_val = matB[ind_cur];
                    true_label = j;
                }
            }
            for(int j = 0; j < cfrnn_Classify->numOutput; j++) {
                if(j == cur_label && j == true_label) cfrnn_Classify->N_TP[j]++;
                if(j == cur_label && j != true_label) cfrnn_Classify->N_FP[j]++;
                if(j != cur_label && j == true_label) cfrnn_Classify->N_FN[j]++;
                if(j != cur_label && j != true_label) cfrnn_Classify->N_TN[j]++;
            }
            cfrnn_Classify->sum_all++;
            cfrnn_Classify->N_sum[true_label]++;
            if(cur_label != true_label) {
                cfrnn_Classify->sum_wrong++;
                cfrnn_Classify->N_wrong[true_label]++;
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
    return;
}

void Finalize_Classify_CFRNN()
{
    for(int i = 0; i < allLabels_train_MOP_Classify_CFRNN->LabelPtr[0].l; i++) {
        free(ind_per_class_train_MOP_Classify_CFRNN[i]);
    }
    free(ind_per_class_train_MOP_Classify_CFRNN);
    free(num_per_class_train_MOP_Classify_CFRNN);

    cfrnn_Classify_CFRNN_free(cfrnn_Classify);

    free_Img(allImgs_train_MOP_Classify_CFRNN);
    free_Label(allLabels_train_MOP_Classify_CFRNN);
    free_Img(allImgs_test_MOP_Classify_CFRNN);
    free_Label(allLabels_test_MOP_Classify_CFRNN);

    free_Img(allMaxInImgs_MOP_Classify_CFRNN);
    free_Img(allMinInImgs_MOP_Classify_CFRNN);
    free_Img(allMeanInImgs_MOP_Classify_CFRNN);
    free_Img(allStdInImgs_MOP_Classify_CFRNN);
    free_Img(allRangeInImgs_MOP_Classify_CFRNN);

    return;
}

//////////////////////////////////////////////////////////////////////////
void cfrnn_Classify_CFRNN_setup(cfrnn_Classify_CFRNN* cfrnn)
{
    int numOutput = NUM_class_data_MOP_Classify_CFRNN;
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
#if MF_RULE_NUM_MOP_CLASSIFY_CFRNN_CUR == MF_RULE_NUM_MOP_CLASSIFY_CFRNN_LESS
    int numFuzzyRules = 10;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#else
    int numFuzzyRules = 10;// DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = 20;// (int)sqrt(numFuzzyRules);
#endif
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
    int channelsIn_C1 = 1;
    int channelsOut_C1 = 6;
    int channelsInOut_P2 = channelsOut_C1;
    int channelsIn_C3 = channelsInOut_P2;
    int channelsOut_C3 = 16;
    int channelsInOut_P4 = channelsOut_C3;
    //
    cfrnn->inputChannel = channelsIn_C1;
    int inputHeightMax = allImgs_train_MOP_Classify_CFRNN->ImgPtr[0].r;
    int inputWidthMax = allImgs_train_MOP_Classify_CFRNN->ImgPtr[0].c;
    cfrnn->inputHeightMax = inputHeightMax;
    cfrnn->inputWidthMax = inputWidthMax;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    int tmp_flag_kernelFlagAdap = 0;
    int tmp_default_kernelFlag = KERNEL_FLAG_OPERATE;
    int tmp_flag_actFuncAdap = 0;
    int tmp_default_actFunc = ACT_FUNC_LEAKYRELU;
    int tmp_flag_paddingTypeAdap = 0;
    int tmp_default_paddingType = PADDING_VALID;
    int tmp_flag_poolTypeAdap = 0;
    int tmp_default_poolType = POOL_MAX;
    //
    cfrnn->typeCoding = tmp_typeCoding;
    //
    cfrnn->numParaLocal = 0;
    cfrnn->numParaLocal_disc = 0;
    //
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    cfrnn->C1 = setupConvLayer(inputHeightMax, inputWidthMax, channelsIn_C1, channelsOut_C1, channelsIn_C1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               0, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P2 = setupPoolLayer(cfrnn->C1->featureMapHeightMax, cfrnn->C1->featureMapWidthMax, channelsInOut_P2,
                               cfrnn->C1->channelsOutMax,
                               0, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    cfrnn->C3 = setupConvLayer(cfrnn->P2->featureMapHeightMax, cfrnn->P2->featureMapWidthMax, channelsIn_C3, channelsOut_C3,
                               cfrnn->P2->channelsInOutMax,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               0, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P4 = setupPoolLayer(cfrnn->C3->featureMapHeightMax, cfrnn->C3->featureMapWidthMax, channelsInOut_P4,
                               cfrnn->C3->channelsOutMax,
                               0, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    cfrnn->M5 = (MemberLayer**)calloc(cfrnn->P4->channelsInOutMax, sizeof(MemberLayer*));
    cfrnn->F6 = (FuzzyLayer**)calloc(cfrnn->P4->channelsInOutMax, sizeof(FuzzyLayer*));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        int numPixels = cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
        MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int i = 0; i < numPixels; i++) {
            inputMin[i] = 0;
            inputMax[i] = 1;
        }
        int* numMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
#if MF_RULE_NUM_MOP_CLASSIFY_CFRNN_CUR == MF_RULE_NUM_MOP_CLASSIFY_CFRNN_LESS
            numMemship[i] = 1;
#else
            numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#endif
        }
        int* flagAdapMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
            flagAdapMemship[i] = 1;
        }
        cfrnn->M5[n] = setupMemberLayer(numPixels, inputMin, inputMax,
                                        numMemship, flagAdapMemship, cfrnn->typeFuzzySet,
                                        tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        cfrnn->F6[n] = setupFuzzyLayer(numPixels, cfrnn->M5[n]->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet, cfrnn->typeRules,
                                       cfrnn->typeInRuleCorNum, TODO, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        free(inputMin);
        free(inputMax);
        free(numMemship);
        free(flagAdapMemship);
    }
    int numRules = 0;
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) numRules += cfrnn->F6[n]->numRules;
    cfrnn->R7 = setupRoughLayer(numRules, numRoughSets, cfrnn->typeFuzzySet,
                                FLAG_STATUS_ON,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE outputMin[MAX_LAB_NUM_MOP_Classify_CFRNN];
    MY_FLT_TYPE outputMax[MAX_LAB_NUM_MOP_Classify_CFRNN];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = 0;
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++)
        numInputConsequenceNode += cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++)
        numInputConsequenceNode += cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    numInputConsequenceNode += cfrnn->P4->channelsInOutMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    numInputConsequenceNode += cfrnn->P4->channelsInOutMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    numInputConsequenceNode += cfrnn->P4->channelsInOutMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    numInputConsequenceNode += cfrnn->P4->channelsInOutMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#endif
    cfrnn->OL = setupOutReduceLayer(cfrnn->R7->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer,
                                    cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag,
                                    numInputConsequenceNode, TODO, TODO,
                                    cfrnn->flagConnectStatus, cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    //
    cfrnn->numParaLocal =
        cfrnn->C1->numParaLocal +
        cfrnn->P2->numParaLocal +
        cfrnn->C3->numParaLocal +
        cfrnn->P4->numParaLocal;
    for(int i = 0; i < cfrnn->P4->channelsInOutMax; i++) {
        cfrnn->numParaLocal +=
            cfrnn->M5[i]->numParaLocal +
            cfrnn->F6[i]->numParaLocal;
    }
    cfrnn->numParaLocal +=
        cfrnn->R7->numParaLocal +
        cfrnn->OL->numParaLocal;
    cfrnn->numParaLocal_disc =
        cfrnn->C1->numParaLocal_disc +
        cfrnn->P2->numParaLocal_disc +
        cfrnn->C3->numParaLocal_disc +
        cfrnn->P4->numParaLocal_disc;
    for(int i = 0; i < cfrnn->P4->channelsInOutMax; i++) {
        cfrnn->numParaLocal_disc +=
            cfrnn->M5[i]->numParaLocal_disc +
            cfrnn->F6[i]->numParaLocal_disc;
    }
    cfrnn->numParaLocal_disc +=
        cfrnn->R7->numParaLocal_disc +
        cfrnn->OL->numParaLocal_disc;
    cfrnn->layerNum = 8;
    //
    cfrnn->xType = (int*)malloc(cfrnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->C1->xType, cfrnn->C1->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->C1->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->P2->xType, cfrnn->P2->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->P2->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->C3->xType, cfrnn->C3->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->C3->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->P4->xType, cfrnn->P4->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->P4->numParaLocal;
    for(int i = 0; i < cfrnn->P4->channelsInOutMax; i++) {
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->M5[i]->xType, cfrnn->M5[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->M5[i]->numParaLocal;
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->F6[i]->xType, cfrnn->F6[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->F6[i]->numParaLocal;
    }
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->R7->xType, cfrnn->R7->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->R7->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->OL->xType, cfrnn->OL->numParaLocal * sizeof(int));
    //tmp_cnt_p = 0;
    //for(int i = 0; i < cfrnn->numParaLocal; i++) {
    //    if(cfrnn->xType[i] != VAR_TYPE_CONTINUOUS)
    //        tmp_cnt_p++;
    //}
    //printf("%d ~ %d \n", tmp_cnt_p, cfrnn->numParaLocal_disc);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    cfrnn->C1 = setupConvLayer(inputHeightMax, inputWidthMax, channelsIn_C1, channelsOut_C1, channelsIn_C1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               0, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P2 = setupPoolLayer(cfrnn->C1->featureMapHeightMax, cfrnn->C1->featureMapWidthMax, channelsInOut_P2,
                               cfrnn->C1->channelsOutMax,
                               0, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    cfrnn->C3 = setupConvLayer(cfrnn->P2->featureMapHeightMax, cfrnn->P2->featureMapWidthMax, channelsIn_C3, channelsOut_C3,
                               cfrnn->P2->channelsInOutMax,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 0,
                               0, DEFAULT_CONV_KERNEL_HEIGHT_CFRNN_MODEL, DEFAULT_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MIN_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MAX_CONV_KERNEL_HEIGHT_CFRNN_MODEL, MIN_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               MAX_CONV_KERNEL_WIDTH_CFRNN_MODEL,
                               tmp_flag_kernelFlagAdap, tmp_default_kernelFlag,
                               tmp_flag_actFuncAdap, tmp_default_actFunc,
                               tmp_flag_paddingTypeAdap, tmp_default_paddingType);
    cfrnn->P4 = setupPoolLayer(cfrnn->C3->featureMapHeightMax, cfrnn->C3->featureMapWidthMax, channelsInOut_P4,
                               cfrnn->C3->channelsOutMax,
                               0, DEFAULT_POOL_REGION_HEIGHT_CFRNN_MODEL, DEFAULT_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MIN_POOL_REGION_HEIGHT_CFRNN_MODEL, MAX_POOL_REGION_HEIGHT_CFRNN_MODEL, MIN_POOL_REGION_WIDTH_CFRNN_MODEL,
                               MAX_POOL_REGION_WIDTH_CFRNN_MODEL,
                               tmp_flag_poolTypeAdap, tmp_default_poolType);
    int numInputConsequenceNode = 0;
    int tmp_flag_actFunc = 0;
    int tmp_flag_connectAdap = 0;
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++)
        numInputConsequenceNode += cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    cfrnn->OL = setupFCLayer(numInputConsequenceNode, numInputConsequenceNode, cfrnn->numOutput,
                             tmp_flag_actFunc, TODO, TODO, tmp_flag_connectAdap);
    //
    cfrnn->numParaLocal =
        cfrnn->C1->numParaLocal +
        cfrnn->P2->numParaLocal +
        cfrnn->C3->numParaLocal +
        cfrnn->P4->numParaLocal;
    cfrnn->numParaLocal +=
        cfrnn->OL->numParaLocal;
    cfrnn->numParaLocal_disc =
        cfrnn->C1->numParaLocal_disc +
        cfrnn->P2->numParaLocal_disc +
        cfrnn->C3->numParaLocal_disc +
        cfrnn->P4->numParaLocal_disc;
    cfrnn->numParaLocal_disc +=
        cfrnn->OL->numParaLocal_disc;
    cfrnn->layerNum = 5;
    //
    cfrnn->xType = (int*)malloc(cfrnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->C1->xType, cfrnn->C1->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->C1->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->P2->xType, cfrnn->P2->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->P2->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->C3->xType, cfrnn->C3->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->C3->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->P4->xType, cfrnn->P4->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->P4->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->OL->xType, cfrnn->OL->numParaLocal * sizeof(int));
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    cfrnn->regionNum = 16;
    cfrnn->regionNum_side = sqrt(cfrnn->regionNum);
    cfrnn->regionNum = cfrnn->regionNum_side * cfrnn->regionNum_side;
    cfrnn->region_row_len = (int*)calloc(cfrnn->regionNum_side, sizeof(int));
    cfrnn->region_col_len = (int*)calloc(cfrnn->regionNum_side, sizeof(int));
    cfrnn->region_row_offset = (int*)calloc(cfrnn->regionNum_side, sizeof(int));
    cfrnn->region_col_offset = (int*)calloc(cfrnn->regionNum_side, sizeof(int));
    int reg_quo_row = inputHeightMax / cfrnn->regionNum_side;
    int reg_rem_row = inputHeightMax % cfrnn->regionNum_side;
    for(int tmp_r_id = 0; tmp_r_id < cfrnn->regionNum_side; tmp_r_id++) {
        if(tmp_r_id > 0) {
            cfrnn->region_row_offset[tmp_r_id] = cfrnn->region_row_offset[tmp_r_id - 1] + cfrnn->region_row_len[tmp_r_id - 1];
        }
        cfrnn->region_row_len[tmp_r_id] = reg_quo_row;
        if(tmp_r_id < reg_rem_row) cfrnn->region_row_len[tmp_r_id]++;
    }
    int reg_quo_col = inputWidthMax / cfrnn->regionNum_side;
    int reg_rem_col = inputWidthMax % cfrnn->regionNum_side;
    for(int tmp_c_id = 0; tmp_c_id < cfrnn->regionNum_side; tmp_c_id++) {
        if(tmp_c_id > 0) {
            cfrnn->region_col_offset[tmp_c_id] = cfrnn->region_col_offset[tmp_c_id - 1] + cfrnn->region_col_len[tmp_c_id - 1];
        }
        cfrnn->region_col_len[tmp_c_id] = reg_quo_col;
        if(tmp_c_id < reg_rem_col) cfrnn->region_col_len[tmp_c_id]++;
    }
    cfrnn->M1 = (MemberLayer**)calloc(cfrnn->regionNum, sizeof(MemberLayer*));
    cfrnn->F2 = (FuzzyLayer**)calloc(cfrnn->regionNum, sizeof(FuzzyLayer*));
    for(int n = 0; n < cfrnn->regionNum; n++) {
        int tmp_r_id = n / cfrnn->regionNum_side;
        int tmp_c_id = n % cfrnn->regionNum_side;
        int numPixels = cfrnn->inputChannel * cfrnn->region_row_len[tmp_r_id] * cfrnn->region_col_len[tmp_c_id];
        MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int i = 0; i < numPixels; i++) {
            inputMin[i] = 0;
            inputMax[i] = 1;
        }
        int* numMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
            numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
        }
        int* flagAdapMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
            flagAdapMemship[i] = 1;
        }
        cfrnn->M1[n] = setupMemberLayer(numPixels, inputMin, inputMax,
                                        numMemship, flagAdapMemship, cfrnn->typeFuzzySet,
                                        tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        cfrnn->F2[n] = setupFuzzyLayer(numPixels, cfrnn->M1[n]->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet, cfrnn->typeRules,
                                       cfrnn->typeInRuleCorNum, FLAG_STATUS_OFF, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        free(inputMin);
        free(inputMax);
        free(numMemship);
        free(flagAdapMemship);
    }
    int numRules = 0;
    for(int n = 0; n < cfrnn->regionNum; n++) {
        numRules += cfrnn->F2[n]->numRules;
    }
    cfrnn->R3 = setupRoughLayer(numRules, numRoughSets, cfrnn->typeFuzzySet,
                                1,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE outputMin[MAX_LAB_NUM_MOP_Classify_CFRNN];
    MY_FLT_TYPE outputMax[MAX_LAB_NUM_MOP_Classify_CFRNN];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = 0;
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    numInputConsequenceNode = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    numInputConsequenceNode = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#endif
    MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(numInputConsequenceNode + 1, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(numInputConsequenceNode + 1, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < numInputConsequenceNode + 1; i++) {
        inputMin[i] = 0;
        inputMax[i] = 1;
    }
    cfrnn->OL = setupOutReduceLayer(cfrnn->R3->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer, cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag,
                                    numInputConsequenceNode, inputMin, inputMax, cfrnn->flagConnectStatus,
                                    cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    free(inputMin);
    free(inputMax);
    //
    cfrnn->numParaLocal = 0;
    for(int n = 0; n < cfrnn->regionNum; n++) {
        cfrnn->numParaLocal +=
            cfrnn->M1[n]->numParaLocal +
            cfrnn->F2[n]->numParaLocal;
    }
    cfrnn->numParaLocal +=
        cfrnn->R3->numParaLocal +
        cfrnn->OL->numParaLocal;
    cfrnn->numParaLocal_disc = 0;
    for(int n = 0; n < cfrnn->regionNum; n++) {
        cfrnn->numParaLocal_disc +=
            cfrnn->M1[n]->numParaLocal_disc +
            cfrnn->F2[n]->numParaLocal_disc;
    }
    cfrnn->numParaLocal_disc +=
        cfrnn->R3->numParaLocal_disc +
        cfrnn->OL->numParaLocal_disc;
    cfrnn->layerNum = 4;
    //
    cfrnn->xType = (int*)malloc(cfrnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    for(int i = 0; i < cfrnn->regionNum; i++) {
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->M1[i]->xType, cfrnn->M1[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->M1[i]->numParaLocal;
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->F2[i]->xType, cfrnn->F2[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->F2[i]->numParaLocal;
    }
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->R3->xType, cfrnn->R3->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->R3->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->OL->xType, cfrnn->OL->numParaLocal * sizeof(int));
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    cfrnn->inputChannel = 16;
    cfrnn->inputHeightMax = 5;
    cfrnn->inputWidthMax = 5;
    cfrnn->M1 = (MemberLayer**)calloc(cfrnn->inputChannel, sizeof(MemberLayer*));
    cfrnn->F2 = (FuzzyLayer**)calloc(cfrnn->inputChannel, sizeof(FuzzyLayer*));
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        int numPixels = cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        MY_FLT_TYPE* inputMin = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE* inputMax = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int i = 0; i < numPixels; i++) {
            inputMin[i] = 0;
            inputMax[i] = 1;
        }
        int* numMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
            numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
        }
        int* flagAdapMemship = (int*)calloc(numPixels, sizeof(int));
        for(int i = 0; i < numPixels; i++) {
            flagAdapMemship[i] = 1;
        }
        cfrnn->M1[n] = setupMemberLayer(numPixels, inputMin, inputMax,
                                        numMemship, flagAdapMemship, cfrnn->typeFuzzySet,
                                        tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        cfrnn->F2[n] = setupFuzzyLayer(numPixels, cfrnn->M1[n]->numMembershipFun, numFuzzyRules, cfrnn->typeFuzzySet, cfrnn->typeRules,
                                       cfrnn->typeInRuleCorNum, FLAG_STATUS_OFF, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
        free(inputMin);
        free(inputMax);
        free(numMemship);
        free(flagAdapMemship);
    }
    int numRules = 0;
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        numRules += cfrnn->F2[n]->numRules;
    }
    cfrnn->R3 = setupRoughLayer(numRules, numRoughSets, cfrnn->typeFuzzySet,
                                1,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    MY_FLT_TYPE outputMin[MAX_LAB_NUM_MOP_Classify_CFRNN];
    MY_FLT_TYPE outputMax[MAX_LAB_NUM_MOP_Classify_CFRNN];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int numInputConsequenceNode = 0;
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    numInputConsequenceNode = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    numInputConsequenceNode = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    numInputConsequenceNode += cfrnn->inputChannel;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE
    numInputConsequenceNode += 0;
    consequenceNodeStatus = NO_CONSEQUENCE_CENTROID;
    cfrnn->consequenceNodeStatus = consequenceNodeStatus;
#endif
    cfrnn->OL = setupOutReduceLayer(cfrnn->R3->numRoughSets, cfrnn->numOutput, outputMin, outputMax,
                                    cfrnn->typeFuzzySet, cfrnn->typeTypeReducer, cfrnn->consequenceNodeStatus, cfrnn->centroid_num_tag,
                                    numInputConsequenceNode, TODO, TODO, cfrnn->flagConnectStatus,
                                    cfrnn->flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
//
    cfrnn->numParaLocal = 0;
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        cfrnn->numParaLocal +=
            cfrnn->M1[n]->numParaLocal +
            cfrnn->F2[n]->numParaLocal;
    }
    cfrnn->numParaLocal +=
        cfrnn->R3->numParaLocal +
        cfrnn->OL->numParaLocal;
    cfrnn->numParaLocal_disc = 0;
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        cfrnn->numParaLocal_disc +=
            cfrnn->M1[n]->numParaLocal_disc +
            cfrnn->F2[n]->numParaLocal_disc;
    }
    cfrnn->numParaLocal_disc +=
        cfrnn->R3->numParaLocal_disc +
        cfrnn->OL->numParaLocal_disc;
    cfrnn->layerNum = 4;
//
    cfrnn->xType = (int*)malloc(cfrnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    for(int i = 0; i < cfrnn->inputChannel; i++) {
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->M1[i]->xType, cfrnn->M1[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->M1[i]->numParaLocal;
        memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->F2[i]->xType, cfrnn->F2[i]->numParaLocal * sizeof(int));
        tmp_cnt_p += cfrnn->F2[i]->numParaLocal;
    }
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->R3->xType, cfrnn->R3->numParaLocal * sizeof(int));
    tmp_cnt_p += cfrnn->R3->numParaLocal;
    memcpy(&cfrnn->xType[tmp_cnt_p], cfrnn->OL->xType, cfrnn->OL->numParaLocal * sizeof(int));
#endif

    cfrnn->e = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->N_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_wrong = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->e_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->N_TP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_TN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_FP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    cfrnn->N_FN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    cfrnn->featureMapTagInitial = (int***)calloc(cfrnn->inputChannel, sizeof(int**));
    cfrnn->dataflowInitial = (MY_FLT_TYPE***)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < cfrnn->inputChannel; i++) {
        cfrnn->featureMapTagInitial[i] = (int**)calloc(inputHeightMax, sizeof(int*));
        cfrnn->dataflowInitial[i] = (MY_FLT_TYPE**)calloc(inputHeightMax, sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < inputHeightMax; j++) {
            cfrnn->featureMapTagInitial[i][j] = (int*)calloc(inputWidthMax, sizeof(int));
            cfrnn->dataflowInitial[i][j] = (MY_FLT_TYPE*)calloc(inputWidthMax, sizeof(MY_FLT_TYPE));
            for(int k = 0; k < inputWidthMax; k++) {
                cfrnn->featureMapTagInitial[i][j][k] = 1;
                cfrnn->dataflowInitial[i][j][k] = (MY_FLT_TYPE)(1.0 / (cfrnn->inputChannel * inputHeightMax * inputWidthMax));
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

void cfrnn_Classify_CFRNN_free(cfrnn_Classify_CFRNN* cfrnn)
{
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    freeOutReduceLayer(cfrnn->OL);
    freeRoughLayer(cfrnn->R7);
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        freeFuzzyLayer(cfrnn->F6[n]);
        freeMemberLayer(cfrnn->M5[n]);
    }
    free(cfrnn->F6);
    free(cfrnn->M5);
    freePoolLayer(cfrnn->P4);
    freeConvLayer(cfrnn->C3);
    freePoolLayer(cfrnn->P2);
    freeConvLayer(cfrnn->C1);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    freeFCLayer(cfrnn->OL);
    freePoolLayer(cfrnn->P4);
    freeConvLayer(cfrnn->C3);
    freePoolLayer(cfrnn->P2);
    freeConvLayer(cfrnn->C1);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    freeOutReduceLayer(cfrnn->OL);
    freeRoughLayer(cfrnn->R3);
    for(int n = 0; n < cfrnn->regionNum; n++) {
        freeFuzzyLayer(cfrnn->F2[n]);
        freeMemberLayer(cfrnn->M1[n]);
    }
    free(cfrnn->F2);
    free(cfrnn->M1);
    free(cfrnn->region_row_len);
    free(cfrnn->region_col_len);
    free(cfrnn->region_row_offset);
    free(cfrnn->region_col_offset);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    freeOutReduceLayer(cfrnn->OL);
    freeRoughLayer(cfrnn->R3);
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        freeFuzzyLayer(cfrnn->F2[n]);
        freeMemberLayer(cfrnn->M1[n]);
    }
    free(cfrnn->F2);
    free(cfrnn->M1);
#endif

    free(cfrnn->xType);

    free(cfrnn->e);

    free(cfrnn->N_sum);
    free(cfrnn->N_wrong);
    free(cfrnn->e_sum);

    free(cfrnn->N_TP);
    free(cfrnn->N_TN);
    free(cfrnn->N_FP);
    free(cfrnn->N_FN);

    int inputHeightMax = cfrnn->inputHeightMax;
    int inputWidthMax = cfrnn->inputWidthMax;
    int channelsIn = cfrnn->inputChannel;
    for(int i = 0; i < channelsIn; i++) {
        for(int j = 0; j < inputHeightMax; j++) {
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

void cfrnn_Classify_CFRNN_init(cfrnn_Classify_CFRNN* cfrnn, double* x, int mode)
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

#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    assignConvLayer(cfrnn->C1, &x[count], mode);
    count += cfrnn->C1->numParaLocal;
    assignPoolLayer(cfrnn->P2, &x[count], mode);
    count += cfrnn->P2->numParaLocal;
    assignConvLayer(cfrnn->C3, &x[count], mode);
    count += cfrnn->C3->numParaLocal;
    assignPoolLayer(cfrnn->P4, &x[count], mode);
    count += cfrnn->P4->numParaLocal;
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        assignMemberLayer(cfrnn->M5[n], &x[count], mode);
        count += cfrnn->M5[n]->numParaLocal;
        assignFuzzyLayer(cfrnn->F6[n], &x[count], mode);
        count += cfrnn->F6[n]->numParaLocal;
    }
    assignRoughLayer(cfrnn->R7, &x[count], mode);
    count += cfrnn->R7->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    assignConvLayer(cfrnn->C1, &x[count], mode);
    count += cfrnn->C1->numParaLocal;
    assignPoolLayer(cfrnn->P2, &x[count], mode);
    count += cfrnn->P2->numParaLocal;
    assignConvLayer(cfrnn->C3, &x[count], mode);
    count += cfrnn->C3->numParaLocal;
    assignPoolLayer(cfrnn->P4, &x[count], mode);
    count += cfrnn->P4->numParaLocal;
    assignFCLayer(cfrnn->OL, &x[count], mode);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    for(int n = 0; n < cfrnn->regionNum; n++) {
        assignMemberLayer(cfrnn->M1[n], &x[count], mode);
        count += cfrnn->M1[n]->numParaLocal;
        assignFuzzyLayer(cfrnn->F2[n], &x[count], mode);
        count += cfrnn->F2[n]->numParaLocal;
    }
    assignRoughLayer(cfrnn->R3, &x[count], mode);
    count += cfrnn->R3->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        assignMemberLayer(cfrnn->M1[n], &x[count], mode);
        count += cfrnn->M1[n]->numParaLocal;
        assignFuzzyLayer(cfrnn->F2[n], &x[count], mode);
        count += cfrnn->F2[n]->numParaLocal;
    }
    assignRoughLayer(cfrnn->R3, &x[count], mode);
    count += cfrnn->R3->numParaLocal;
    assignOutReduceLayer(cfrnn->OL, &x[count], mode);
#endif
    return;
}

void ff_cfrnn_Classify_CFRNN(cfrnn_Classify_CFRNN* cfrnn, MY_FLT_TYPE*** valIn, MY_FLT_TYPE* valOut,
                             MY_FLT_TYPE** inputConsequenceNode)
{
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    ff_convLayer(cfrnn->C1, valIn, cfrnn->featureMapTagInitial, &cfrnn->inputHeightMax, &cfrnn->inputWidthMax,
                 cfrnn->dataflowInitial);
    ff_poolLayer(cfrnn->P2, cfrnn->C1->featureMapData, cfrnn->C1->featureMapTag,
                 cfrnn->C1->featureMapHeight, cfrnn->C1->featureMapWidth, cfrnn->C1->dataflowStatus);
    ff_convLayer(cfrnn->C3, cfrnn->P2->featureMapData, cfrnn->P2->featureMapTag,
                 cfrnn->P2->featureMapHeight, cfrnn->P2->featureMapWidth, cfrnn->P2->dataflowStatus);
    ff_poolLayer(cfrnn->P4, cfrnn->C3->featureMapData, cfrnn->C3->featureMapTag,
                 cfrnn->C3->featureMapHeight, cfrnn->C3->featureMapWidth, cfrnn->C3->dataflowStatus);
    int numPixels = cfrnn->P4->channelsInOutMax * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    MY_FLT_TYPE* input_M5 = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* inputNormed_M5 = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* dataflowStatus_M5 = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE tmp_min = 0;
    MY_FLT_TYPE tmp_max = 0;
    int tmp_flag = 0;
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    if(tmp_flag == 0) {
                        tmp_min = cfrnn->P4->featureMapData[n][h][w];
                        tmp_max = cfrnn->P4->featureMapData[n][h][w];
                        tmp_flag = 1;
                    } else {
                        if(tmp_min > cfrnn->P4->featureMapData[n][h][w])
                            tmp_min = cfrnn->P4->featureMapData[n][h][w];
                        if(tmp_max < cfrnn->P4->featureMapData[n][h][w])
                            tmp_max = cfrnn->P4->featureMapData[n][h][w];
                    }
                }
            }
        }
    }
    //
    MY_FLT_TYPE tmp_small_val = 1e-6;
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                int cur_ind = n * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax +
                              h * cfrnn->P4->featureMapWidthMax + w;
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    inputNormed_M5[cur_ind] =
                        (cfrnn->P4->featureMapData[n][h][w] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                    input_M5[cur_ind] = cfrnn->P4->featureMapData[n][h][w];
                } else {
                    inputNormed_M5[cur_ind] = 0;
                    input_M5[cur_ind] = 0;
                }
                dataflowStatus_M5[cur_ind] = cfrnn->P4->dataflowStatus[n][h][w];
            }
        }
        int tmp_offset = n * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
        ff_memberLayer(cfrnn->M5[n], &inputNormed_M5[tmp_offset], &dataflowStatus_M5[tmp_offset]);
        ff_fuzzyLayer(cfrnn->F6[n], cfrnn->M5[n]->degreeMembership, cfrnn->M5[n]->dataflowStatus);
    }
    //
    int tmp_size_dim_2 = 1;
    if(cfrnn->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) tmp_size_dim_2 = 2;
    int numRules = cfrnn->R7->numInput;
    int tmp_offset = 0;
    MY_FLT_TYPE** degreesInput = (MY_FLT_TYPE**)calloc(numRules, sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < numRules; i++) {
        degreesInput[i] = (MY_FLT_TYPE*)calloc(tmp_size_dim_2, sizeof(MY_FLT_TYPE));
    }
    MY_FLT_TYPE* dataflowStatus_F6 = (MY_FLT_TYPE*)calloc(numRules, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int i = 0; i < cfrnn->F6[n]->numRules; i++) {
            for(int j = 0; j < tmp_size_dim_2; j++) {
                degreesInput[tmp_offset + i][j] = cfrnn->F6[n]->degreeRules[i][j];
            }
            dataflowStatus_F6[tmp_offset + i] = cfrnn->F6[n]->dataflowStatus[i];
        }
        tmp_offset += cfrnn->F6[n]->numRules;
    }
    ff_roughLayer(cfrnn->R7, degreesInput, dataflowStatus_F6);
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   input_M5,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   inputNormed_M5,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    MY_FLT_TYPE* meanFM = (MY_FLT_TYPE*)calloc(cfrnn->P4->channelsInOutMax, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        int tmp_cnt = 0;
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    meanFM[n] += cfrnn->P4->featureMapData[n][h][w];
                    tmp_cnt++;
                }
            }
        }
        if(tmp_cnt)
            meanFM[n] /= tmp_cnt;
    }
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   meanFM,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    free(meanFM);
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    MY_FLT_TYPE* meanFM = (MY_FLT_TYPE*)calloc(cfrnn->P4->channelsInOutMax, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    meanFM[n] += cfrnn->P4->featureMapData[n][h][w];
                }
            }
        }
        meanFM[n] /= cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    }
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   meanFM,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    free(meanFM);
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    MY_FLT_TYPE* meanFM = (MY_FLT_TYPE*)calloc(cfrnn->P4->channelsInOutMax, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        int tmp_cnt = 0;
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                int cur_ind = n * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax +
                              h * cfrnn->P4->featureMapWidthMax + w;
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    meanFM[n] += inputNormed_M5[cur_ind];
                    tmp_cnt++;
                }
            }
        }
        if(tmp_cnt)
            meanFM[n] /= tmp_cnt;
    }
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   meanFM,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    free(meanFM);
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    MY_FLT_TYPE* meanFM = (MY_FLT_TYPE*)calloc(cfrnn->P4->channelsInOutMax, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                int cur_ind = n * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax +
                              h * cfrnn->P4->featureMapWidthMax + w;
                if(cfrnn->P4->dataflowStatus[n][h][w] > 0) {
                    meanFM[n] += inputNormed_M5[cur_ind];
                }
            }
        }
        meanFM[n] /= cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    }
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            memcpy(cfrnn_Classify->OL->inputConsequenceNode[i],
                   meanFM,
                   cfrnn_Classify->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
    free(meanFM);
#endif
    ff_outReduceLayer(cfrnn->OL, cfrnn->R7->degreeRough, cfrnn->R7->dataflowStatus);
    for(int i = 0; i < cfrnn->OL->numOutput; i++) {
        if(CHECK_INVALID(cfrnn->OL->valOutputFinal[i])) {
            printf("%s(%d): Invalid output, exiting...\n",
                   __FILE__, __LINE__);
        }
    }

    memcpy(valOut, cfrnn->OL->valOutputFinal, cfrnn->OL->numOutput * sizeof(MY_FLT_TYPE));
    //
    free(input_M5);
    free(inputNormed_M5);
    free(dataflowStatus_M5);
    for(int i = 0; i < numRules; i++) {
        free(degreesInput[i]);
    }
    free(degreesInput);
    free(dataflowStatus_F6);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    ff_convLayer(cfrnn->C1, valIn, cfrnn->featureMapTagInitial, &cfrnn->inputHeightMax, &cfrnn->inputWidthMax,
                 cfrnn->dataflowInitial);
    ff_poolLayer(cfrnn->P2, cfrnn->C1->featureMapData, cfrnn->C1->featureMapTag,
                 cfrnn->C1->featureMapHeight, cfrnn->C1->featureMapWidth, cfrnn->C1->dataflowStatus);
    ff_convLayer(cfrnn->C3, cfrnn->P2->featureMapData, cfrnn->P2->featureMapTag,
                 cfrnn->P2->featureMapHeight, cfrnn->P2->featureMapWidth, cfrnn->P2->dataflowStatus);
    ff_poolLayer(cfrnn->P4, cfrnn->C3->featureMapData, cfrnn->C3->featureMapTag,
                 cfrnn->C3->featureMapHeight, cfrnn->C3->featureMapWidth, cfrnn->C3->dataflowStatus);
    int numPixels = cfrnn->P4->channelsInOutMax * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax;
    MY_FLT_TYPE* inputFlatten = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
    int* tagFlatten = (int*)calloc(numPixels, sizeof(int));
    MY_FLT_TYPE* dataflowStatusFlatten = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->P4->channelsInOutMax; n++) {
        for(int h = 0; h < cfrnn->P4->featureMapHeightMax; h++) {
            for(int w = 0; w < cfrnn->P4->featureMapWidthMax; w++) {
                int cur_ind = n * cfrnn->P4->featureMapHeightMax * cfrnn->P4->featureMapWidthMax +
                              h * cfrnn->P4->featureMapWidthMax + w;
                inputFlatten[cur_ind] = cfrnn->P4->featureMapData[n][h][w];
                tagFlatten[cur_ind] = 1;
                dataflowStatusFlatten[cur_ind] = cfrnn->P4->dataflowStatus[n][h][w];
            }
        }
    }
    ff_fcLayer(cfrnn->OL, inputFlatten, tagFlatten, numPixels, dataflowStatusFlatten);

    memcpy(valOut, cfrnn->OL->outputData, cfrnn->OL->numOutput * sizeof(MY_FLT_TYPE));
    //
    free(inputFlatten);
    free(tagFlatten);
    free(dataflowStatusFlatten);
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    for(int n = 0; n < cfrnn->regionNum; n++) {
        int tmp_r_id = n / cfrnn->regionNum_side;
        int tmp_c_id = n % cfrnn->regionNum_side;
        int numPixels = cfrnn->inputChannel * cfrnn->region_row_len[tmp_r_id] * cfrnn->region_col_len[tmp_c_id];
        int tmp_count = 0;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE* vec_in_flow = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            for(int r = cfrnn->region_row_offset[tmp_r_id]; r < cfrnn->region_row_offset[tmp_r_id] + cfrnn->region_row_len[tmp_r_id]; r++) {
                for(int c = cfrnn->region_col_offset[tmp_c_id]; c < cfrnn->region_col_offset[tmp_c_id] + cfrnn->region_col_len[tmp_c_id]; c++) {
                    vec_in_data[tmp_count] = valIn[ch][r][c];
                    vec_in_flow[tmp_count] = cfrnn->dataflowInitial[ch][r][c];
                    tmp_count++;
                }
            }
        }
        ff_memberLayer(cfrnn->M1[n], vec_in_data, vec_in_flow);
        ff_fuzzyLayer(cfrnn->F2[n], cfrnn->M1[n]->degreeMembership, cfrnn->M1[n]->dataflowStatus);
        free(vec_in_data);
        free(vec_in_flow);
    }
    int tmp_size_dim_2 = 1;
    if(cfrnn->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) tmp_size_dim_2 = 2;
    int numRules = cfrnn->R3->numInput;
    int tmp_offset = 0;
    MY_FLT_TYPE** degreesInput_R = (MY_FLT_TYPE**)calloc(numRules, sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < numRules; i++) {
        degreesInput_R[i] = (MY_FLT_TYPE*)calloc(tmp_size_dim_2, sizeof(MY_FLT_TYPE));
    }
    MY_FLT_TYPE* dataflowStatus_F = (MY_FLT_TYPE*)calloc(numRules, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->regionNum; n++) {
        for(int i = 0; i < cfrnn->F2[n]->numRules; i++) {
            for(int j = 0; j < tmp_size_dim_2; j++) {
                degreesInput_R[tmp_offset + i][j] = cfrnn->F2[n]->degreeRules[i][j];
            }
            dataflowStatus_F[tmp_offset + i] = cfrnn->F2[n]->dataflowStatus[i];
        }
        tmp_offset += cfrnn->F2[n]->numRules;
    }
    ff_roughLayer(cfrnn->R3, degreesInput_R, dataflowStatus_F);
    for(int i = 0; i < numRules; i++) {
        free(degreesInput_R[i]);
    }
    free(degreesInput_R);
    free(dataflowStatus_F);
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        int numPixels = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        int tmp_count = 0;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            tmp_count = ch * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        vec_in_data[tmp_count] = valIn[ch][r][c];
                    }
                    tmp_count++;
                }
            }
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_data,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_data);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        int numPixels = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        int tmp_count = 0;
        MY_FLT_TYPE tmp_small_val = 1e-6;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            tmp_count = ch * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        vec_in_data[tmp_count] = (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                    } else {
                        vec_in_data[tmp_count] = 0;
                    }
                    tmp_count++;
                }
            }
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_data,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_data);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            int tmp_cnt = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        vec_in_FM[ch] += valIn[ch][r][c];
                        tmp_cnt++;
                    }
                }
            }
            if(tmp_cnt)
                vec_in_FM[ch] /= tmp_cnt;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_FM);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        vec_in_FM[ch] += valIn[ch][r][c];
                    }
                }
            }
            vec_in_FM[ch] /= cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_FM);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE tmp_small_val = 1e-6;
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            int tmp_cnt = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_min < tmp_max)
                            vec_in_FM[ch] += (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                        else
                            vec_in_FM[ch] += 0;
                        tmp_cnt++;
                    }
                }
            }
            if(tmp_cnt)
                vec_in_FM[ch] /= tmp_cnt;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE tmp_small_val = 1e-6;
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_min < tmp_max)
                            vec_in_FM[ch] += (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                        else
                            vec_in_FM[ch] += 0;
                    }
                }
            }
            vec_in_FM[ch] /= cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#endif
    ff_outReduceLayer(cfrnn->OL, cfrnn->R3->degreeRough, cfrnn->R3->dataflowStatus);

    memcpy(valOut, cfrnn->OL->valOutputFinal, cfrnn->OL->numOutput * sizeof(MY_FLT_TYPE));
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        int numPixels = cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        int tmp_count = 0;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE* vec_in_flow = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        int ch = n;
        for(int r = 0; r < cfrnn->inputHeightMax; r++) {
            for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                vec_in_data[tmp_count] = valIn[ch][r][c];
                vec_in_flow[tmp_count] = cfrnn->dataflowInitial[ch][r][c];
                tmp_count++;
            }
        }
        ff_memberLayer(cfrnn->M1[n], vec_in_data, vec_in_flow);
        ff_fuzzyLayer(cfrnn->F2[n], cfrnn->M1[n]->degreeMembership, cfrnn->M1[n]->dataflowStatus);
        free(vec_in_data);
        free(vec_in_flow);
    }
    int tmp_size_dim_2 = 1;
    if(cfrnn->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) tmp_size_dim_2 = 2;
    int numRules = cfrnn->R3->numInput;
    int tmp_offset = 0;
    MY_FLT_TYPE** degreesInput_R = (MY_FLT_TYPE**)calloc(numRules, sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < numRules; i++) {
        degreesInput_R[i] = (MY_FLT_TYPE*)calloc(tmp_size_dim_2, sizeof(MY_FLT_TYPE));
    }
    MY_FLT_TYPE* dataflowStatus_F = (MY_FLT_TYPE*)calloc(numRules, sizeof(MY_FLT_TYPE));
    for(int n = 0; n < cfrnn->inputChannel; n++) {
        for(int i = 0; i < cfrnn->F2[n]->numRules; i++) {
            for(int j = 0; j < tmp_size_dim_2; j++) {
                degreesInput_R[tmp_offset + i][j] = cfrnn->F2[n]->degreeRules[i][j];
            }
            dataflowStatus_F[tmp_offset + i] = cfrnn->F2[n]->dataflowStatus[i];
        }
        tmp_offset += cfrnn->F2[n]->numRules;
    }
    ff_roughLayer(cfrnn->R3, degreesInput_R, dataflowStatus_F);
    for(int i = 0; i < numRules; i++) {
        free(degreesInput_R[i]);
    }
    free(degreesInput_R);
    free(dataflowStatus_F);
#if CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        int numPixels = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        int tmp_count = 0;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            tmp_count = ch * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        vec_in_data[tmp_count] = valIn[ch][r][c];
                    }
                    tmp_count++;
                }
            }
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_data,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_data);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        int numPixels = cfrnn->inputChannel * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        int tmp_count = 0;
        MY_FLT_TYPE tmp_small_val = 1e-6;
        MY_FLT_TYPE* vec_in_data = (MY_FLT_TYPE*)calloc(numPixels, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            tmp_count = ch * cfrnn->inputHeightMax * cfrnn->inputWidthMax;
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c]) {
                        vec_in_data[tmp_count] = (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                    } else {
                        vec_in_data[tmp_count] = 0;
                    }
                    tmp_count++;
                }
            }
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_data,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_data);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            int tmp_cnt = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        vec_in_FM[ch] += valIn[ch][r][c];
                        tmp_cnt++;
                    }
                }
            }
            if(tmp_cnt)
                vec_in_FM[ch] /= tmp_cnt;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_FM);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        vec_in_FM[ch] += valIn[ch][r][c];
                    }
                }
            }
            vec_in_FM[ch] /= cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
        free(vec_in_FM);
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE tmp_small_val = 1e-6;
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            int tmp_cnt = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_min < tmp_max)
                            vec_in_FM[ch] += (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                        else
                            vec_in_FM[ch] += 0;
                        tmp_cnt++;
                    }
                }
            }
            if(tmp_cnt)
                vec_in_FM[ch] /= tmp_cnt;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#elif CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR == CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM
    if(cfrnn_Classify->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        MY_FLT_TYPE* vec_in_FM = (MY_FLT_TYPE*)calloc(cfrnn->inputChannel, sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE tmp_small_val = 1e-6;
        for(int ch = 0; ch < cfrnn->inputChannel; ch++) {
            MY_FLT_TYPE tmp_min = 1e30;
            MY_FLT_TYPE tmp_max = -1e30;
            int tmp_flag = 0;
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_flag == 0) {
                            tmp_min = valIn[ch][r][c];
                            tmp_max = valIn[ch][r][c];
                            tmp_flag = 1;
                        } else {
                            if(tmp_min > valIn[ch][r][c]) tmp_min = valIn[ch][r][c];
                            if(tmp_max < valIn[ch][r][c]) tmp_max = valIn[ch][r][c];
                        }
                    }
                }
            }
            for(int r = 0; r < cfrnn->inputHeightMax; r++) {
                for(int c = 0; c < cfrnn->inputWidthMax; c++) {
                    if(cfrnn->featureMapTagInitial[ch][r][c] > 0) {
                        if(tmp_min < tmp_max)
                            vec_in_FM[ch] += (valIn[ch][r][c] - tmp_min) / (tmp_max - tmp_min + tmp_small_val);
                        else
                            vec_in_FM[ch] += 0;
                    }
                }
            }
            vec_in_FM[ch] /= cfrnn->inputHeightMax * cfrnn->inputWidthMax;
        }
        for(int i = 0; i < cfrnn->OL->numOutput; i++) {
            memcpy(cfrnn->OL->inputConsequenceNode[i],
                   vec_in_FM,
                   cfrnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        }
    }
#endif
    ff_outReduceLayer(cfrnn->OL, cfrnn->R3->degreeRough, cfrnn->R3->dataflowStatus);

    memcpy(valOut, cfrnn->OL->valOutputFinal, cfrnn->OL->numOutput * sizeof(MY_FLT_TYPE));
#endif
    return;
}

//////////////////////////////////////////////////////////////////////////
static int** allocINT_MOP_Classify_CFRNN(int nrow, int ncol)
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

static MY_FLT_TYPE** allocFLOAT_MOP_Classify_CFRNN(int nrow, int ncol)
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

static void getIndicators_MOP_Classify_CFRNN_c(MY_FLT_TYPE& mean_prc, MY_FLT_TYPE& std_prc, MY_FLT_TYPE& mean_rec,
        MY_FLT_TYPE& std_rec, MY_FLT_TYPE& mean_ber, MY_FLT_TYPE& std_ber)
{
    int len_lab = NUM_class_data_MOP_Classify_CFRNN;
    MY_FLT_TYPE sum_precision = cfrnn_Classify->sum_wrong / cfrnn_Classify->sum_all;
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
    MY_FLT_TYPE* tmp_precision = (MY_FLT_TYPE*)malloc(len_lab * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_recall = (MY_FLT_TYPE*)malloc(len_lab * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_Fvalue = (MY_FLT_TYPE*)malloc(len_lab * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmp_errorRate = (MY_FLT_TYPE*)malloc(len_lab * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < len_lab; i++) {
        if(cfrnn_Classify->N_TP[i] > 0) {
            tmp_precision[i] = cfrnn_Classify->N_TP[i] / (cfrnn_Classify->N_TP[i] + cfrnn_Classify->N_FP[i]);
        } else {
            tmp_precision[i] = 0;
        }
        if(cfrnn_Classify->N_TP[i] + cfrnn_Classify->N_FN[i] > 0) {
            tmp_recall[i] = cfrnn_Classify->N_TP[i] / (cfrnn_Classify->N_TP[i] + cfrnn_Classify->N_FN[i]);
            tmp_errorRate[i] = cfrnn_Classify->N_FN[i] / (cfrnn_Classify->N_TP[i] + cfrnn_Classify->N_FN[i]);
        } else {
            tmp_recall[i] = 0;
            tmp_errorRate[i] = 1;
        }
        if(tmp_precision[i] + tmp_recall[i] > 0) {
            tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
                            (tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
        } else {
            tmp_Fvalue[i] = 0;
        }
        mean_precision += tmp_precision[i];
        mean_recall += tmp_recall[i];
        mean_Fvalue += tmp_Fvalue[i];
        mean_errorRate += tmp_errorRate[i];
#if STATUS_OUT_INDEICES_MOP_Classify_CFRNN == FLAG_ON_MOP_Classify_CFRNN
        printf("%f %f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i], tmp_errorRate[i]);
#endif
    }
    mean_precision /= len_lab;
    mean_recall /= len_lab;
    mean_Fvalue /= len_lab;
    mean_errorRate /= len_lab;
    for(int i = 0; i < len_lab; i++) {
        std_precision += (tmp_precision[i] - mean_precision) * (tmp_precision[i] - mean_precision);
        std_recall += (tmp_recall[i] - mean_recall) * (tmp_recall[i] - mean_recall);
        std_Fvalue += (tmp_Fvalue[i] - mean_Fvalue) * (tmp_Fvalue[i] - mean_Fvalue);
        std_errorRate += (tmp_errorRate[i] - mean_errorRate) * (tmp_errorRate[i] - mean_errorRate);
    }
    std_precision /= len_lab;
    std_precision = sqrt(std_precision);
    std_recall /= len_lab;
    std_recall = sqrt(std_recall);
    std_Fvalue /= len_lab;
    std_Fvalue = sqrt(std_Fvalue);
    std_errorRate /= len_lab;
    std_errorRate = sqrt(std_errorRate);
    //
    mean_prc = mean_precision;
    std_prc = std_precision;
    mean_rec = mean_recall;
    std_rec = std_recall;
    mean_ber = mean_errorRate;
    std_ber = std_errorRate;
    //
    free(tmp_precision);
    free(tmp_recall);
    free(tmp_Fvalue);
    free(tmp_errorRate);
    //
    return;
}

void getFitness_MOP_Classify_CFRNN_c(double* fitness)
{
    //
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE std_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE std_recall = 0;
    MY_FLT_TYPE mean_ber = 0;
    MY_FLT_TYPE std_ber = 0;
    getIndicators_MOP_Classify_CFRNN_c(mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);
    //
    int len_lab = NUM_class_data_MOP_Classify_CFRNN;
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < len_lab; i++) {
        cur_dataflow += cfrnn_Classify->OL->dataflowStatus[i];
        if(cfrnn_Classify->OL->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_Classify_CFRNN);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
#if CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I
    fitness[0] = 1 - mean_recall + val_violation;
    //
    MY_FLT_TYPE f_FC = 0;
    for(int n = 0; n < cfrnn_Classify->P4->channelsInOutMax; n++) {
        MY_FLT_TYPE num_connect_cur = 0;
        MY_FLT_TYPE num_connect_all = 0;
        for(int i = 0; i < cfrnn_Classify->F6[n]->numRules; i++) {
            for(int j = 0; j < cfrnn_Classify->F6[n]->numInput; j++) {
                int tmp_flag = 0;
                for(int k = 0; k < cfrnn_Classify->F6[n]->numMembershipFun[j]; k++) {
                    if(cfrnn_Classify->F6[n]->connectStatusAll[i][j][k]) tmp_flag = 1;
                }
                if(tmp_flag) num_connect_cur++;
                num_connect_all++;
            }
        }
        f_FC += num_connect_cur / num_connect_all;
    }
    fitness[1] = f_FC / cfrnn_Classify->P4->channelsInOutMax;
    //
    MY_FLT_TYPE f_IO = 0;
    int tmp_offset_RL = 0;
    MY_FLT_TYPE num_FT = 0;
    for(int n = 0; n < cfrnn_Classify->P4->channelsInOutMax; n++) {
        for(int j = 0; j < cfrnn_Classify->F6[n]->numInput; j++) {
            MY_FLT_TYPE* IO_FLi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->F6[n]->numRules, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < cfrnn_Classify->F6[n]->numRules; i++) {
                for(int k = 0; k < cfrnn_Classify->F6[n]->numMembershipFun[j]; k++) {
                    if(cfrnn_Classify->F6[n]->connectStatusAll[i][j][k]) IO_FLi[i] = 1;
                }
            }
            MY_FLT_TYPE* IO_RLi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->R7->numRoughSets, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < cfrnn_Classify->R7->numRoughSets; i++) {
                for(int k = tmp_offset_RL; k < tmp_offset_RL + cfrnn_Classify->F6[n]->numRules; k++) {
                    if(cfrnn_Classify->R7->connectStatus[i][k]) IO_RLi[i] += IO_FLi[k - tmp_offset_RL];
                }
            }
            MY_FLT_TYPE tmp_max = IO_RLi[0];
            MY_FLT_TYPE tmp_sum = IO_RLi[0];
            for(int i = 1; i < cfrnn_Classify->R7->numRoughSets; i++) {
                if(tmp_max < IO_RLi[i]) tmp_max = IO_RLi[i];
                tmp_sum += IO_RLi[i];
            }
            f_IO += (tmp_sum - tmp_max) / (tmp_max + 1e-6);
            //MY_FLT_TYPE* IO_OUTi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->OL->numOutput, sizeof(MY_FLT_TYPE));
            //for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            //    for(int k = 0; k < cfrnn_Classify->R7->numRoughSets; k++) {
            //        if(cfrnn_Classify->OL->connectStatus[i][k]) IO_OUTi[i] += IO_RLi[k];
            //    }
            //}
            //MY_FLT_TYPE tmp_max = IO_OUTi[0];
            //MY_FLT_TYPE tmp_sum = IO_OUTi[0];
            //for(int i = 1; i < cfrnn_Classify->numOutput; i++) {
            //    if(tmp_max < IO_OUTi[i]) tmp_max = IO_OUTi[i];
            //    tmp_sum += IO_OUTi[i];
            //}
            //f_IO += (tmp_sum - tmp_max) / (tmp_max + 1e-6);
            free(IO_FLi);
            free(IO_RLi);
            //free(IO_OUTi);
        }
        tmp_offset_RL += cfrnn_Classify->F6[n]->numRules;
        num_FT += cfrnn_Classify->F6[n]->numInput;
    }
    f_IO /= num_FT * (cfrnn_Classify->R7->numRoughSets - 1);
    fitness[2] = f_IO;
    //fitness[1] = std_ber + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II
    fitness[0] = mean_ber + val_violation;
    fitness[1] = std_ber + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III
    fitness[0] = mean_ber + val_violation;
    fitness[1] = std_ber + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#elif CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR == CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV
    fitness[0] = 1 - mean_recall + val_violation;
    //
    MY_FLT_TYPE f_FC = 0;
    for(int n = 0; n < cfrnn_Classify->inputChannel; n++) {
        MY_FLT_TYPE num_connect_cur = 0;
        MY_FLT_TYPE num_connect_all = 0;
        for(int i = 0; i < cfrnn_Classify->F2[n]->numRules; i++) {
            for(int j = 0; j < cfrnn_Classify->F2[n]->numInput; j++) {
                int tmp_flag = 0;
                for(int k = 0; k < cfrnn_Classify->F2[n]->numMembershipFun[j]; k++) {
                    if(cfrnn_Classify->F2[n]->connectStatusAll[i][j][k]) tmp_flag = 1;
                }
                if(tmp_flag) num_connect_cur++;
                num_connect_all++;
            }
        }
        f_FC += num_connect_cur / num_connect_all;
    }
    fitness[1] = f_FC / cfrnn_Classify->inputChannel;
    //
    MY_FLT_TYPE f_IO = 0;
    int tmp_offset_RL = 0;
    MY_FLT_TYPE num_FT = 0;
    for(int n = 0; n < cfrnn_Classify->inputChannel; n++) {
        for(int j = 0; j < cfrnn_Classify->F2[n]->numInput; j++) {
            MY_FLT_TYPE* IO_FLi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->F2[n]->numRules, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < cfrnn_Classify->F2[n]->numRules; i++) {
                for(int k = 0; k < cfrnn_Classify->F2[n]->numMembershipFun[j]; k++) {
                    if(cfrnn_Classify->F2[n]->connectStatusAll[i][j][k]) IO_FLi[i] = 1;
                }
            }
            MY_FLT_TYPE* IO_RLi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->R3->numRoughSets, sizeof(MY_FLT_TYPE));
            for(int i = 0; i < cfrnn_Classify->R3->numRoughSets; i++) {
                for(int k = tmp_offset_RL; k < tmp_offset_RL + cfrnn_Classify->F2[n]->numRules; k++) {
                    if(cfrnn_Classify->R3->connectStatus[i][k]) IO_RLi[i] += IO_FLi[k - tmp_offset_RL];
                }
            }
            MY_FLT_TYPE tmp_max = IO_RLi[0];
            MY_FLT_TYPE tmp_sum = IO_RLi[0];
            for(int i = 1; i < cfrnn_Classify->R3->numRoughSets; i++) {
                if(tmp_max < IO_RLi[i]) tmp_max = IO_RLi[i];
                tmp_sum += IO_RLi[i];
            }
            f_IO += (tmp_sum - tmp_max) / (tmp_max + 1e-6);
            //MY_FLT_TYPE* IO_OUTi = (MY_FLT_TYPE*)calloc(cfrnn_Classify->OL->numOutput, sizeof(MY_FLT_TYPE));
            //for(int i = 0; i < cfrnn_Classify->OL->numOutput; i++) {
            //    for(int k = 0; k < cfrnn_Classify->R7->numRoughSets; k++) {
            //        if(cfrnn_Classify->OL->connectStatus[i][k]) IO_OUTi[i] += IO_RLi[k];
            //    }
            //}
            //MY_FLT_TYPE tmp_max = IO_OUTi[0];
            //MY_FLT_TYPE tmp_sum = IO_OUTi[0];
            //for(int i = 1; i < cfrnn_Classify->numOutput; i++) {
            //    if(tmp_max < IO_OUTi[i]) tmp_max = IO_OUTi[i];
            //    tmp_sum += IO_OUTi[i];
            //}
            //f_IO += (tmp_sum - tmp_max) / (tmp_max + 1e-6);
            free(IO_FLi);
            free(IO_RLi);
            //free(IO_OUTi);
        }
        tmp_offset_RL += cfrnn_Classify->F2[n]->numRules;
        num_FT += cfrnn_Classify->F2[n]->numInput;
    }
    f_IO /= num_FT * (cfrnn_Classify->R3->numRoughSets - 1);
    fitness[2] = f_IO;
    //fitness[1] = std_ber + val_violation;
    //fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#endif
}
static void read_int_from_file_MOP_Classify_CFRNN(const char* filename, int* vec_int, int num_int)
{
    FILE* fp = NULL;
    if((fp = fopen(filename, "r")) != NULL) {
        int tmpVal;
        for(int i = 0; i < num_int; i++) {
            int tmp = fscanf(fp, "%d", &tmpVal);
            if(tmp == EOF) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(9);
            }
            vec_int[i] = tmpVal;
        }
        fclose(fp);
        fp = NULL;
    } else {
        printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, filename);
        exit(-1);
    }
    return;
}

void generate_img_statistics_MOP_Classify_CFRNN(ImgArr& imgmin, ImgArr& imgmax, ImgArr& imgmean, ImgArr& imgstd,
        ImgArr& imgrange,
        ImgArr allimgs, LabelArr alllabels)
{
    int num_imgs = allimgs->ImgNum;
    int num_class = alllabels->LabelPtr[0].l;
    int n_rows = allimgs->ImgPtr[0].r;
    int n_cols = allimgs->ImgPtr[0].c;
    //////////////////////////////////////////////////////////////////////////
    imgmin = (ImgArr)malloc(sizeof(MinstImgArr));
    imgmax = (ImgArr)malloc(sizeof(MinstImgArr));
    imgmean = (ImgArr)malloc(sizeof(MinstImgArr));
    imgstd = (ImgArr)malloc(sizeof(MinstImgArr));
    imgrange = (ImgArr)malloc(sizeof(MinstImgArr));
    imgmin->ImgNum = num_class;
    imgmin->ImgPtr = (MinstImg*)calloc(num_class, sizeof(MinstImg));
    imgmax->ImgNum = num_class;
    imgmax->ImgPtr = (MinstImg*)calloc(num_class, sizeof(MinstImg));
    imgmean->ImgNum = num_class;
    imgmean->ImgPtr = (MinstImg*)calloc(num_class, sizeof(MinstImg));
    imgstd->ImgNum = num_class;
    imgstd->ImgPtr = (MinstImg*)calloc(num_class, sizeof(MinstImg));
    imgrange->ImgNum = num_class;
    imgrange->ImgPtr = (MinstImg*)calloc(num_class, sizeof(MinstImg));
    int* tmp_count = (int*)calloc(num_class, sizeof(int));
    for(int iClass = 0; iClass < num_class; iClass++) {
        imgmin->ImgPtr[iClass].r = n_rows;
        imgmax->ImgPtr[iClass].r = n_rows;
        imgmean->ImgPtr[iClass].r = n_rows;
        imgstd->ImgPtr[iClass].r = n_rows;
        imgrange->ImgPtr[iClass].r = n_rows;
        imgmin->ImgPtr[iClass].c = n_cols;
        imgmax->ImgPtr[iClass].c = n_cols;
        imgmean->ImgPtr[iClass].c = n_cols;
        imgstd->ImgPtr[iClass].c = n_cols;
        imgrange->ImgPtr[iClass].c = n_cols;
        imgmin->ImgPtr[iClass].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        imgmax->ImgPtr[iClass].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        imgmean->ImgPtr[iClass].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        imgstd->ImgPtr[iClass].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        imgrange->ImgPtr[iClass].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(int r = 0; r < n_rows; ++r) {
            imgmin->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            imgmax->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            imgmean->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            imgstd->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            imgrange->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            for(int c = 0; c < n_cols; c++) {
                imgmin->ImgPtr[iClass].ImgData[r][c] = (MY_FLT_TYPE)(1e30);
                imgmax->ImgPtr[iClass].ImgData[r][c] = (MY_FLT_TYPE)(-1e30);
                imgstd->ImgPtr[iClass].ImgData[r][c] = 0;
                imgmean->ImgPtr[iClass].ImgData[r][c] = 0;
                imgrange->ImgPtr[iClass].ImgData[r][c] = 0;
            }
        }
    }
    for(int i = 0; i < num_imgs; i++) {
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
        for(int lab = 1; lab < num_class; lab++) {
            if(tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
                tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        int iClass = cur_lab_i;
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; c++) {
                MY_FLT_TYPE tmp_f = allimgs->ImgPtr[i].ImgData[r][c];
                if(imgmin->ImgPtr[iClass].ImgData[r][c] > tmp_f)
                    imgmin->ImgPtr[iClass].ImgData[r][c] = tmp_f;
                if(imgmax->ImgPtr[iClass].ImgData[r][c] < tmp_f)
                    imgmax->ImgPtr[iClass].ImgData[r][c] = tmp_f;
                imgmean->ImgPtr[iClass].ImgData[r][c] += tmp_f;
            }
        }
        tmp_count[iClass]++;
    }
    for(int iClass = 0; iClass < num_class; iClass++) {
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; c++) {
                if(tmp_count[iClass]) {
                    imgrange->ImgPtr[iClass].ImgData[r][c] =
                        imgmax->ImgPtr[iClass].ImgData[r][c] -
                        imgmin->ImgPtr[iClass].ImgData[r][c];
                    //if (imgrange->ImgPtr[iClass].ImgData[r][c] != 1) {
                    //	printf("%e\n", imgrange->ImgPtr[iClass].ImgData[r][c]);
                    //}
                    imgmean->ImgPtr[iClass].ImgData[r][c] /= tmp_count[iClass];
                }
            }
        }
    }
    for(int i = 0; i < num_imgs; i++) {
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
        for(int lab = 1; lab < num_class; lab++) {
            if(tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
                tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        int iClass = cur_lab_i;
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; c++) {
                MY_FLT_TYPE tmp_f = allimgs->ImgPtr[i].ImgData[r][c];
                MY_FLT_TYPE tmp_m = imgmean->ImgPtr[iClass].ImgData[r][c];
                imgstd->ImgPtr[iClass].ImgData[r][c] += (tmp_f - tmp_m) * (tmp_f - tmp_m);
            }
        }
        //tmp_count[iClass]++;
    }
    for(int iClass = 0; iClass < num_class; iClass++) {
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; c++) {
                if(tmp_count[iClass]) {
                    imgstd->ImgPtr[iClass].ImgData[r][c] /= tmp_count[iClass];
                    imgstd->ImgPtr[iClass].ImgData[r][c] = sqrt(imgstd->ImgPtr[iClass].ImgData[r][c]);
                }
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    free(tmp_count);
    //
    return;
}

#define IM1_Classify_CFRNN 2147483563
#define IM2_Classify_CFRNN 2147483399
#define AM_Classify_CFRNN (1.0/IM1_Classify_CFRNN)
#define IMM1_Classify_CFRNN (IM1_Classify_CFRNN-1)
#define IA1_Classify_CFRNN 40014
#define IA2_Classify_CFRNN 40692
#define IQ1_Classify_CFRNN 53668
#define IQ2_Classify_CFRNN 52774
#define IR1_Classify_CFRNN 12211
#define IR2_Classify_CFRNN 3791
#define NTAB_Classify_CFRNN 32
#define NDIV_Classify_CFRNN (1+IMM1_Classify_CFRNN/NTAB_Classify_CFRNN)
#define EPS_Classify_CFRNN 1.2e-7
#define RNMX_Classify_CFRNN (1.0-EPS_Classify_CFRNN)

//the random generator in [0,1)
static double rnd_uni_Classify_CFRNN(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_Classify_CFRNN];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_Classify_CFRNN + 7; j >= 0; j--) {
            k = (*idum) / IQ1_Classify_CFRNN;
            *idum = IA1_Classify_CFRNN * (*idum - k * IQ1_Classify_CFRNN) - k * IR1_Classify_CFRNN;
            if(*idum < 0) *idum += IM1_Classify_CFRNN;
            if(j < NTAB_Classify_CFRNN) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_Classify_CFRNN;
    *idum = IA1_Classify_CFRNN * (*idum - k * IQ1_Classify_CFRNN) - k * IR1_Classify_CFRNN;
    if(*idum < 0) *idum += IM1_Classify_CFRNN;
    k = idum2 / IQ2_Classify_CFRNN;
    idum2 = IA2_Classify_CFRNN * (idum2 - k * IQ2_Classify_CFRNN) - k * IR2_Classify_CFRNN;
    if(idum2 < 0) idum2 += IM2_Classify_CFRNN;
    j = iy / NDIV_Classify_CFRNN;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_Classify_CFRNN;    //printf("%lf\n", AM_Classify_CFRNN*iy);
    if((temp = AM_Classify_CFRNN * iy) > RNMX_Classify_CFRNN) return RNMX_Classify_CFRNN;
    else return temp;
}/*------End of rnd_uni_Classify_CNN()--------------------------*/

static int rnd_Classify_CFRNN(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_Classify_CFRNN(&rnd_uni_init_Classify_CFRNN) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}

/* Fisher¨CYates shuffle algorithm */
static void shuffle_Classify_CFRNN(int* x, int size)
{
    int i, aux, k = 0;
    for(i = size - 1; i > 0; i--) {
        /* get a value between cero and i  */
        k = rnd_Classify_CFRNN(0, i);
        /* exchange of values */
        aux = x[i];
        x[i] = x[k];
        x[k] = aux;
    }
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////