#ifndef __MINST_
#define __MINST_
/*
MINST数据库是一个手写图像数据库，里面
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_NN_FLT_TYPE.h"

typedef struct MinstImg {
    int c;           // 图像宽
    int r;           // 图像高
    MY_FLT_TYPE** ImgData; // 图像数据二维动态数组
} MinstImg;

typedef struct MinstImgArr {
    int ImgNum;        // 存储图像的数目
    MinstImg* ImgPtr;  // 存储图像数组指针
}*ImgArr;              // 存储图像数据的数组

typedef struct MinstLabel {
    int l;            // 输出标记的长
    MY_FLT_TYPE* LabelData; // 输出标记数据
} MinstLabel;

typedef struct MinstLabelArr {
    int LabelNum;
    MinstLabel* LabelPtr;
}*LabelArr;              // 存储图像标记的数组

LabelArr read_Label_IDX_FILE(const char* filename, int len_label); // 读入图像标记
LabelArr read_Label(const char* filename); // 读入图像标记
LabelArr read_Lable_tabel(const char* filetrain, const char* filetest,
                          int num_train, int num_test, int num_class); // 读入图像标记
void     free_Label(LabelArr arr_label);

ImgArr read_Img_IDX_FILE(const char* filename); // 读入图像
ImgArr read_Img(const char* filename); // 读入图像
ImgArr read_Img_tmp(const char* filename, int nSamp, int nCh, int tHeight, int tWidth); // 读入图像
ImgArr read_Img_table(const char* filetrain, const char* filetest,
                      int num_train, int num_test, int num_feature, int num_row, int num_col, int** pixel_index); // 读入图像
void   free_Img(ImgArr arr_img);
void interpolate_Img(ImgArr& imgarr, LabelArr& labarr, ImgArr allimgs, LabelArr alllabels,
                     int* num4neighbor, int maxSize_neighborhood, int* flag_train, int num_train, int* flag_class);

void save_Img(ImgArr imgarr, const char filedir[512]); // 将图像数据保存成文件

typedef struct LabelIndex {
    int len;            // 某标记的个体数
    int* IndexData; // index数组
} LabelIndex;

typedef struct ArrLabelIndex {
    int LabelNum;
    LabelIndex* LabelIndexPtr;
} ArrLabelIndex;             // 存储图像标记的数组

typedef struct SetArrLabelIndex {
    int ArrNum;
    ArrLabelIndex* ArrLabelIndexPtr;
}*SetArrLabelIndexPtr;              // 存储图像标记的数组

SetArrLabelIndexPtr getLabelIndex(LabelArr* arr_Label, int len_arr);
void freeSetArrLabelIndexPtr(SetArrLabelIndexPtr dataPtr);

///////////////////////////////////////////////////////////////////////////
double rnd_uni_CNN(long* idum);
extern int     seed_CNN;
extern long    rnd_uni_init_CNN;
int rnd_CNN(int low, int high);
/* Fisher–Yates shuffle algorithm */
void shuffle_CNN(int* x, int size);
double gaussrand_CNN(double a, double b);

#endif
