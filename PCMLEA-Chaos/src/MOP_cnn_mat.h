// 这里库文件主要存在关于二维矩阵数组的操作
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_NN_FLT_TYPE.h"

#define full 0
#define same 1
#define valid 2

typedef struct Mat2DSize {
	int c; // 列（宽）
	int r; // 行（高）
} nSize;

MY_FLT_TYPE** rotate180(MY_FLT_TYPE** mat, nSize matSize);// 矩阵翻转180度

void addmat(MY_FLT_TYPE** res, MY_FLT_TYPE** mat1, nSize matSize1, MY_FLT_TYPE** mat2, nSize matSize2);// 矩阵相加

MY_FLT_TYPE** correlation(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type);// 互相关

MY_FLT_TYPE** cov(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type); // 卷积操作

// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
MY_FLT_TYPE** UpSample(MY_FLT_TYPE** mat, nSize matSize, int upc, int upr);

// 给二维矩阵边缘扩大，增加addw大小的0值边
MY_FLT_TYPE** matEdgeExpand(MY_FLT_TYPE** mat, nSize matSize, int addc, int addr);

// 给二维矩阵边缘缩小，擦除shrinkc大小的边
MY_FLT_TYPE** matEdgeShrink(MY_FLT_TYPE** mat, nSize matSize, int shrinkc, int shrinkr);

void savemat(MY_FLT_TYPE** mat, nSize matSize, const char* filename);// 保存矩阵数据

void multifactor(MY_FLT_TYPE** res, MY_FLT_TYPE** mat, nSize matSize, MY_FLT_TYPE factor);// 矩阵乘以系数

MY_FLT_TYPE summat(MY_FLT_TYPE** mat, nSize matSize);// 矩阵各元素的和

char* combine_strings(char* a, char* b);

char* intTochar(int i);

#endif
