#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_cnn.h"
#include "MOP_cnn_data.h"
#include "MOP_Classify_CNN.h"
#include "MOP_LeNet.h"
#include "mpi.h"

#define FLAG_OFF_MOP_CLASSIFY_CNN 0
#define FLAG_ON_MOP_CLASSIFY_CNN 1

#define STATUS_OUT_INDEICES_MOP_CLASSIFY_CNN FLAG_OFF_MOP_CLASSIFY_CNN

#define TAG_VALI_MOP_CLASSIFY_CNN -2
#define TAG_NULL_MOP_CLASSIFY_CNN 0
#define TAG_INVA_MOP_CLASSIFY_CNN -1

// 以下都是测试函数，可以不用管
// 测试Minst模块是否工作正常
// void test_minst()
// {
//     LabelArr testLabel = read_Lable("Minst/t10k-labels.idx1-ubyte");
//     ImgArr testImg = read_Img("Minst/t10k-images.idx3-ubyte");
//     save_Img(testImg, "Minst/testImgs/");
// }
// 测试Mat模块是否工作正常
// void test_mat()
// {
//     int i, j;
//     nSize srcSize = { 6, 6 };
//     nSize mapSize = { 4, 4 };
//     srand((unsigned)time(NULL));
//     float** src = (float**)malloc(srcSize.r * sizeof(float*));
//     for (i = 0; i < srcSize.r; i++) {
//         src[i] = (float*)malloc(srcSize.c * sizeof(float));
//         for (j = 0; j < srcSize.c; j++) {
//             src[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }
//     float** map = (float**)malloc(mapSize.r * sizeof(float*));
//     for (i = 0; i < mapSize.r; i++) {
//         map[i] = (float*)malloc(mapSize.c * sizeof(float));
//         for (j = 0; j < mapSize.c; j++) {
//             map[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }

//     nSize cov1size = { srcSize.c + mapSize.c - 1, srcSize.r + mapSize.r - 1 };
//     float** cov1 = cov(map, mapSize, src, srcSize, full);
//     //nSize cov2size={srcSize.c,srcSize.r};
//     //float** cov2=cov(map,mapSize,src,srcSize,same);
//     nSize cov3size = { srcSize.c - (mapSize.c - 1), srcSize.r - (mapSize.r - 1) };
//     float** cov3 = cov(map, mapSize, src, srcSize, valid);

//     savemat(src, srcSize, "PicTrans/src.ma");
//     savemat(map, mapSize, "PicTrans/map.ma");
//     savemat(cov1, cov1size, "PicTrans/cov1.ma");
//     //savemat(cov2,cov2size,"PicTrans/cov2.ma");
//     savemat(cov3, cov3size, "PicTrans/cov3.ma");

//     float** sample = UpSample(src, srcSize, 2, 2);
//     nSize samSize = { srcSize.c * 2, srcSize.r * 2 };
//     savemat(sample, samSize, "PicTrans/sam.ma");
// }
// void test_mat1()
// {
//     int i, j;
//     nSize srcSize = { 12, 12 };
//     nSize mapSize = { 5, 5 };
//     float** src = (float**)malloc(srcSize.r * sizeof(float*));
//     for (i = 0; i < srcSize.r; i++) {
//         src[i] = (float*)malloc(srcSize.c * sizeof(float));
//         for (j = 0; j < srcSize.c; j++) {
//             src[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }
//     float** map1 = (float**)malloc(mapSize.r * sizeof(float*));
//     for (i = 0; i < mapSize.r; i++) {
//         map1[i] = (float*)malloc(mapSize.c * sizeof(float));
//         for (j = 0; j < mapSize.c; j++) {
//             map1[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }
//     float** map2 = (float**)malloc(mapSize.r * sizeof(float*));
//     for (i = 0; i < mapSize.r; i++) {
//         map2[i] = (float*)malloc(mapSize.c * sizeof(float));
//         for (j = 0; j < mapSize.c; j++) {
//             map2[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }
//     float** map3 = (float**)malloc(mapSize.r * sizeof(float*));
//     for (i = 0; i < mapSize.r; i++) {
//         map3[i] = (float*)malloc(mapSize.c * sizeof(float));
//         for (j = 0; j < mapSize.c; j++) {
//             map3[i][j] = (float)((rnd_uni_LeNet(&rnd_uni_init_LeNet) - 0.5) * 2);
//         }
//     }

//     float** cov1 = cov(map1, mapSize, src, srcSize, valid);
//     float** cov2 = cov(map2, mapSize, src, srcSize, valid);
//     nSize covsize = { srcSize.c - (mapSize.c - 1), srcSize.r - (mapSize.r - 1) };
//     float** cov3 = cov(map3, mapSize, src, srcSize, valid);
//     addmat(cov1, cov1, covsize, cov2, covsize);
//     addmat(cov1, cov1, covsize, cov3, covsize);

//     savemat(src, srcSize, "PicTrans/src.ma");
//     savemat(map1, mapSize, "PicTrans/map1.ma");
//     savemat(map2, mapSize, "PicTrans/map2.ma");
//     savemat(map3, mapSize, "PicTrans/map3.ma");
//     savemat(cov1, covsize, "PicTrans/cov1.ma");
//     savemat(cov2, covsize, "PicTrans/cov2.ma");
//     savemat(cov3, covsize, "PicTrans/cov3.ma");

// }
// // 测试cnn模块是否工作正常
// void test_cnn()
// {
//     LabelArr testLabel = read_Lable("Minst/train-labels.idx1-ubyte");
//     ImgArr testImg = read_Img("Minst/train-images.idx3-ubyte");

//     nSize inputSize = { testImg->ImgPtr[0].c, testImg->ImgPtr[0].r };
//     int outSize = testLabel->LabelPtr[0].l;

//     CNN* cnn = (CNN*)malloc(sizeof(CNN));
//     cnnsetup(cnn, inputSize, outSize);

//     CNNOpts opts;
//     opts.numepochs = 1;
//     opts.alpha = 1;
//     int trainNum = 5000;
//     cnntrain(cnn, testImg, testLabel, opts, trainNum);

//     FILE  *fp = NULL;
//     fp = fopen("PicTrans/cnnL.ma", "wb");
//     if (fp == NULL)
//         printf("write file failed\n");
//     fwrite(cnn->L, sizeof(float), trainNum, fp);
//     fclose(fp);
// }

static void trimLine(char line[])
{
	int i = 0;

	while (line[i] != '\0') {
		if (line[i] == '\r' || line[i] == '\n') {
			line[i] = '\0';
			break;
		}
		i++;
	}
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define IM1_Classify_CNN 2147483563
#define IM2_Classify_CNN 2147483399
#define AM_Classify_CNN (1.0/IM1_Classify_CNN)
#define IMM1_Classify_CNN (IM1_Classify_CNN-1)
#define IA1_Classify_CNN 40014
#define IA2_Classify_CNN 40692
#define IQ1_Classify_CNN 53668
#define IQ2_Classify_CNN 52774
#define IR1_Classify_CNN 12211
#define IR2_Classify_CNN 3791
#define NTAB_Classify_CNN 32
#define NDIV_Classify_CNN (1+IMM1_Classify_CNN/NTAB_Classify_CNN)
#define EPS_Classify_CNN 1.2e-7
#define RNMX_Classify_CNN (1.0-EPS_Classify_CNN)

//the random generator in [0,1)
double rnd_uni_Classify_CNN(long* idum)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB_Classify_CNN];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB_Classify_CNN + 7; j >= 0; j--) {
			k = (*idum) / IQ1_Classify_CNN;
			*idum = IA1_Classify_CNN * (*idum - k * IQ1_Classify_CNN) - k * IR1_Classify_CNN;
			if (*idum < 0) *idum += IM1_Classify_CNN;
			if (j < NTAB_Classify_CNN) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1_Classify_CNN;
	*idum = IA1_Classify_CNN * (*idum - k * IQ1_Classify_CNN) - k * IR1_Classify_CNN;
	if (*idum < 0) *idum += IM1_Classify_CNN;
	k = idum2 / IQ2_Classify_CNN;
	idum2 = IA2_Classify_CNN * (idum2 - k * IQ2_Classify_CNN) - k * IR2_Classify_CNN;
	if (idum2 < 0) idum2 += IM2_Classify_CNN;
	j = iy / NDIV_Classify_CNN;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1_Classify_CNN;   //printf("%lf\n", AM_Classify_CNN*iy);
	if ((temp = AM_Classify_CNN * iy) > RNMX_Classify_CNN) return RNMX_Classify_CNN;
	else return temp;
}/*------End of rnd_uni_Classify_CNN()--------------------------*/
int     seed_Classify_CNN = 237;
long    rnd_uni_init_Classify_CNN = -(long)seed_Classify_CNN;

int rnd_Classify_CNN(int low, int high)
{
	int res;
	if (low >= high) {
		res = low;
	}
	else {
		res = low + (int)(rnd_uni_Classify_CNN(&rnd_uni_init_Classify_CNN) * (high - low + 1));
		if (res > high) {
			res = high;
		}
	}
	return (res);
}

/* Fisher–Yates shuffle algorithm */
void shuffle_Classify_CNN(int* x, int size)
{
	int i, aux, k = 0;
	for (i = size - 1; i > 0; i--) {
		/* get a value between cero and i  */
		k = rnd_Classify_CNN(0, i);
		/* exchange of values */
		aux = x[i];
		x[i] = x[k];
		x[k] = aux;
	}
	//
	return;
}

void get_Evaluation_Indicators(int num_class, MY_FLT_TYPE* N_TP, MY_FLT_TYPE* N_FP, MY_FLT_TYPE* N_TN, MY_FLT_TYPE* N_FN, MY_FLT_TYPE* N_wrong, MY_FLT_TYPE* N_sum,
	MY_FLT_TYPE& mean_prc, MY_FLT_TYPE& std_prc, MY_FLT_TYPE& mean_rec, MY_FLT_TYPE& std_rec, MY_FLT_TYPE& mean_ber, MY_FLT_TYPE& std_ber)
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
	for (int i = 0; i < outSize; i++) {
		if (N_TP[i] > 0) {
			tmp_precision[i] = N_TP[i] / (N_TP[i] + N_FP[i]);
		}
		else {
			tmp_precision[i] = 0;
		}
		if (N_TP[i] + N_FN[i] > 0) {
			tmp_recall[i] = N_TP[i] / (N_TP[i] + N_FN[i]);
			tmp_errorRate[i] = N_FN[i] / (N_TP[i] + N_FN[i]);
		}
		else {
			tmp_recall[i] = 0;
			tmp_errorRate[i] = 1;
		}
		if (tmp_recall[i] + tmp_precision[i] > 0)
			tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
			(tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
		else
			tmp_Fvalue[i] = 0;
		mean_precision += tmp_precision[i];
		mean_recall += tmp_recall[i];
		mean_Fvalue += tmp_Fvalue[i];
		mean_errorRate += tmp_errorRate[i];
		if (min_precision > tmp_precision[i]) min_precision = tmp_precision[i];
		if (min_recall > tmp_recall[i]) min_recall = tmp_recall[i];
		if (min_Fvalue > tmp_Fvalue[i]) min_Fvalue = tmp_Fvalue[i];
		if (max_errorRate < tmp_errorRate[i]) max_errorRate = tmp_errorRate[i];
#if STATUS_OUT_INDEICES_MOP_CLASSIFY_CNN == FLAG_ON_MOP_CLASSIFY_CNN
		printf("%f %f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i], tmp_errorRate[i]);
#endif
	}
	mean_precision /= outSize;
	mean_recall /= outSize;
	mean_Fvalue /= outSize;
	mean_errorRate /= outSize;
	for (int i = 0; i < outSize; i++) {
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
	for (int i = 0; i < outSize; i++) {
		double tmp_rt = N_wrong[i] / N_sum[i];
		mean_err_rt += tmp_rt;
		if (max_err_rt < tmp_rt)
			max_err_rt = tmp_rt;
	}
	mean_err_rt /= outSize;
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

////////
////////
LabelArr allLabels;
ImgArr   allImgs;

int* flag_samples;
int* indx_samples;
int num_samples;

int  repNum;
int  repNo;

CNN* cnn_Classify;

int max_size_neighborhood;

LabelArr allLabels_interp;
ImgArr   allImgs_interp;

int* flag_samples_interp;
int* indx_samples_interp;
int num_samples_interp;

ImgArr   allMaxInImgs;
ImgArr   allMinInImgs;
ImgArr   allMeanInImgs;
ImgArr   allStdInImgs;
ImgArr   allRangeInImgs;

int NUM_pixel_Classify_CNN_Indus;
int NUM_feature_Classify_CNN_Indus;
int NUM_samples_all_Classify_CNN_Indus;
int NUM_samples_train_Classify_CNN_Indus;
int NUM_samples_test_Classify_CNN_Indus;
int NUM_class_Classify_CNN_Indus;
int IMG_side_len_Classify_CNN_Indus;
int NDIM_Classify_CNN_Indus;// DIM_LeNet
int NOBJ_Classify_CNN_Indus;// 2

int NDIM_Classify_CNN_Indus_BP;// NDIM_Classify_CNN_Indus
int NOBJ_Classify_CNN_Indus_BP;// 6

//
void bubbleSort_Classify_CNN(double* data, int arrayFx[], int len)
{
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len - i - 1; j++) {
			int ind1 = arrayFx[j];
			int ind2 = arrayFx[j + 1];
			if (data[ind1] < data[ind2]) {
				int tmp = arrayFx[j];
				arrayFx[j] = arrayFx[j + 1];
				arrayFx[j + 1] = tmp;
			}
		}
	}
	//
	return;
}

void read_int_from_file(const char* filename, int* vec_int, int num_int)
{
	FILE* fp = NULL;
	if ((fp = fopen(filename, "r")) != NULL) {
		int tmpVal;
		for (int i = 0; i < num_int; i++) {
			int tmp = fscanf(fp, "%d", &tmpVal);
			if (tmp == EOF) {
				printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
				exit(9);
			}
			vec_int[i] = tmpVal;
		}
		fclose(fp);
		fp = NULL;
	}
	else {
		printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, filename);
		exit(-1);
	}
	return;
}

void bootstrapInitialize_Classify_CNN()
{
	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////
	int numSample = allImgs->ImgNum;
	int lenLabel = allLabels->LabelPtr[0].l;
	for (int i = 0; i < numSample; i++) flag_samples[i] = TAG_NULL_MOP_CLASSIFY_CNN;
	for (int i = 0; i < numSample; i++) indx_samples[i] = TAG_INVA_MOP_CLASSIFY_CNN;
	int* selectIndicator = (int*)calloc(numSample * repNum, sizeof(int));
	int* sizeClass = (int*)calloc(lenLabel, sizeof(int));
	int  count = 0;
	for (int i = 0; i < lenLabel; i++) {
		for (int j = 0; j < numSample; j++) {
			if ((int)allLabels->LabelPtr[j].LabelData[i]) {
				for (int k = count; k < count + repNum; k++) {
					selectIndicator[k] = j;
				}
				count += repNum;
				sizeClass[i]++;
			}
		}
	}
	//int* tmpIDX = (int*)calloc(N_row_whole_data, sizeof(int));
	int tmpIDX[100000];
	int ind_begin = 0;
	int ind_end = 0;
	for (int i = 0; i < lenLabel; i++) {
		int len_sub = sizeClass[i] * repNum;
		shuffle_Classify_CNN(&selectIndicator[ind_begin], len_sub);
		ind_begin += len_sub;
	}
	for (int n = 0; n < repNum; n++) {
		int cur_offset = 0;
		int k = 0;
		for (int i = 0; i < lenLabel; i++) {
			ind_begin = cur_offset + sizeClass[i] * n;
			ind_end = ind_begin + sizeClass[i];
			for (int j = ind_begin; j < ind_end; j++) {
				tmpIDX[k++] = selectIndicator[j];
			}
			cur_offset += sizeClass[i] * repNum;
		}
		if (n == repNo) {
			for (int i = 0; i < numSample; i++) {
				flag_samples[tmpIDX[i]]++;
				indx_samples[i] = tmpIDX[i];
			}
			break;
		}
	}
	shuffle_Classify_CNN(indx_samples, numSample);
	num_samples = numSample;
	free(selectIndicator);
	free(sizeClass);
	return;
}
void multifoldInitialize_Classify_CNN()
{
	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////
	int numSample = allImgs->ImgNum;
	int lenLabel = allLabels->LabelPtr[0].l;
	for (int i = 0; i < numSample; i++) flag_samples[i] = TAG_NULL_MOP_CLASSIFY_CNN;
	for (int i = 0; i < numSample; i++) indx_samples[i] = TAG_INVA_MOP_CLASSIFY_CNN;
	int* stratufiedIndx = (int*)calloc(numSample, sizeof(int));
	int* validIndxTag = (int*)calloc(numSample, sizeof(int));
	int* testIndxTag = (int*)calloc(numSample, sizeof(int));

	int count = 0;
	for (int i = 0; i < lenLabel; i++) {
		for (int j = 0; j < numSample; j++) {
			if ((int)allLabels->LabelPtr[j].LabelData[i]) {
				stratufiedIndx[count++] = j;
			}
		}
	}

	int n1 = repNo;
	while (n1 < numSample) {
		testIndxTag[stratufiedIndx[n1]] = 1;
		//printf("%d ", stratufiedIndx[n1]);
		n1 += repNum;
	}
	int n2 = (repNo - 1 + repNum) % repNum;
	while (n2 < numSample) {
		validIndxTag[stratufiedIndx[n2]] = 1;
		//printf("%d ", stratufiedIndx[n2]);
		n2 += repNum;
	}
	//printf("\n");
	//for(n = 0; n < N_row_whole_data; n++) {
	//    if(testIndxTag[n])
	//        printf("%d ", n);
	//}
	//printf("\n");
	//printf("\n");

	count = 0;
	for (int n = 0; n < numSample; n++) {
		if (!testIndxTag[n] && !validIndxTag[n]) {
			flag_samples[n]++;
			indx_samples[count++] = n;
		}
		if (validIndxTag[n]) {
			flag_samples[n] = TAG_VALI_MOP_CLASSIFY_CNN;
		}
	}
	num_samples = count;

	//for(n = 0; n < count; n++) {
	//    printf("%d ", index[n][0]);
	//}
	//printf("\n");

	shuffle_Classify_CNN(indx_samples, count);

	free(stratufiedIndx);
	free(validIndxTag);
	free(testIndxTag);

	return;
}

void generate_img_statistics(ImgArr& imgmin, ImgArr& imgmax, ImgArr& imgmean, ImgArr& imgstd, ImgArr& imgrange,
	ImgArr allimgs, LabelArr alllabels, int* flag_train)
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
	for (int iClass = 0; iClass < num_class; iClass++) {
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
		for (int r = 0; r < n_rows; ++r) {
			imgmin->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
			imgmax->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
			imgmean->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
			imgstd->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
			imgrange->ImgPtr[iClass].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
			for (int c = 0; c < n_cols; c++) {
				imgmin->ImgPtr[iClass].ImgData[r][c] = (MY_FLT_TYPE)(1e30);
				imgmax->ImgPtr[iClass].ImgData[r][c] = (MY_FLT_TYPE)(-1e30);
				imgstd->ImgPtr[iClass].ImgData[r][c] = 0;
				imgmean->ImgPtr[iClass].ImgData[r][c] = 0;
				imgrange->ImgPtr[iClass].ImgData[r][c] = 0;
			}
		}
	}
	for (int i = 0; i < num_imgs; i++) {
		if (flag_train[i] <= 0) continue;
		int cur_lab_i = 0;
		MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
		for (int lab = 1; lab < num_class; lab++) {
			if (tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
				tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
				cur_lab_i = lab;
			}
		}
		int iClass = cur_lab_i;
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; c++) {
				MY_FLT_TYPE tmp_f = allimgs->ImgPtr[i].ImgData[r][c];
				if (imgmin->ImgPtr[iClass].ImgData[r][c] > tmp_f)
					imgmin->ImgPtr[iClass].ImgData[r][c] = tmp_f;
				if (imgmax->ImgPtr[iClass].ImgData[r][c] < tmp_f)
					imgmax->ImgPtr[iClass].ImgData[r][c] = tmp_f;
				imgmean->ImgPtr[iClass].ImgData[r][c] += tmp_f;
			}
		}
		tmp_count[iClass]++;
	}
	for (int iClass = 0; iClass < num_class; iClass++) {
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; c++) {
				if (tmp_count[iClass]) {
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
	for (int i = 0; i < num_imgs; i++) {
		if (flag_train[i] <= 0) continue;
		int cur_lab_i = 0;
		MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
		for (int lab = 1; lab < num_class; lab++) {
			if (tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
				tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
				cur_lab_i = lab;
			}
		}
		int iClass = cur_lab_i;
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; c++) {
				MY_FLT_TYPE tmp_f = allimgs->ImgPtr[i].ImgData[r][c];
				MY_FLT_TYPE tmp_m = imgmean->ImgPtr[iClass].ImgData[r][c];
				imgstd->ImgPtr[iClass].ImgData[r][c] += (tmp_f - tmp_m) * (tmp_f - tmp_m);
			}
		}
		//tmp_count[iClass]++;
	}
	for (int iClass = 0; iClass < num_class; iClass++) {
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; c++) {
				if (tmp_count[iClass]) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Initialize_data_Classify_CNN(int curN, int numN, int trainNo, int testNo, int endNo)
{
	repNum = numN;
	repNo = curN;

	seed_CNN = 237;
	rnd_uni_init_CNN = -(long)seed_CNN;
	for (int i = 0; i < curN; i++) {
		seed_CNN = (seed_CNN + 111) % 1235;
		rnd_uni_init_CNN = -(long)seed_CNN;
	}
	seed_Classify_CNN = 237;
	rnd_uni_init_Classify_CNN = -(long)seed_Classify_CNN;

	char filename[1024];
	//
	sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_num_class");
	read_int_from_file(filename, &NUM_class_Classify_CNN_Indus, 1);
	//
	sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_side_length_F%d", curN + 1);
	read_int_from_file(filename, &IMG_side_len_Classify_CNN_Indus, 1);
	//
	sprintf(filename, "../Data_all/Data_CNN_Indus/SECOM_num_feature_F%d", curN + 1);
	read_int_from_file(filename, &NUM_feature_Classify_CNN_Indus, 1);
	//
	int* tmp_vec_ind = (int*)malloc(NUM_feature_Classify_CNN_Indus * sizeof(int));
	sprintf(filename, "../Data_all/Data_CNN_Indus/Mat_ind_F%d.txt", curN + 1);
	read_int_from_file(filename, tmp_vec_ind, NUM_feature_Classify_CNN_Indus);
	//
	sprintf(filename, "../Data_all/Data_CNN_Indus/F%d/SECOM_nrow_train_F%d", curN + 1, curN + 1);
	read_int_from_file(filename, &NUM_samples_train_Classify_CNN_Indus, 1);
	//
	sprintf(filename, "../Data_all/Data_CNN_Indus/F%d/SECOM_nrow_test_F%d", curN + 1, curN + 1);
	read_int_from_file(filename, &NUM_samples_test_Classify_CNN_Indus, 1);
	//
	NUM_samples_all_Classify_CNN_Indus =
		NUM_samples_train_Classify_CNN_Indus +
		NUM_samples_test_Classify_CNN_Indus;
	//
#if TAG_RAND_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	for (int i = 0; i < NUM_feature_Classify_CNN_Indus; i++) tmp_vec_ind[i] = i;
	shuffle_CNN(tmp_vec_ind, NUM_feature_Classify_CNN_Indus);
	//shuffle_Classify_CNN(tmp_vec_ind, NUM_feature_Classify_CNN_Indus);
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	for (int i = 0; i < NUM_feature_Classify_CNN_Indus; i++) tmp_vec_ind[i] = i;
	shuffle_CNN(tmp_vec_ind, NUM_feature_Classify_CNN_Indus);
	//shuffle_Classify_CNN(tmp_vec_ind, NUM_feature_Classify_CNN_Indus);
#endif
	int** tmp_mat_ind = (int**)malloc(IMG_side_len_Classify_CNN_Indus * sizeof(int*));
	for (int i = 0; i < IMG_side_len_Classify_CNN_Indus; i++)
		tmp_mat_ind[i] = (int*)malloc(IMG_side_len_Classify_CNN_Indus * sizeof(int));
	for (int r = 0; r < IMG_side_len_Classify_CNN_Indus; r++) {
		for (int c = 0; c < IMG_side_len_Classify_CNN_Indus; c++) {
			int cur_count = r * IMG_side_len_Classify_CNN_Indus + c;
			if (cur_count < NUM_feature_Classify_CNN_Indus) {
				tmp_mat_ind[r][c] = tmp_vec_ind[cur_count];
			}
			else {
				tmp_mat_ind[r][c] = -1;
			}
		}
	}
	//
	char filetrain[1024];
	char filetest[1024];
	sprintf(filetrain, "../Data_all/Data_CNN_Indus/F%d/SECOM_samples_train_F%d", curN + 1, curN + 1);
	sprintf(filetest, "../Data_all/Data_CNN_Indus/F%d/SECOM_samples_test_F%d", curN + 1, curN + 1);
	allImgs = read_Img_table(filetrain, filetest, NUM_samples_train_Classify_CNN_Indus, NUM_samples_test_Classify_CNN_Indus,
		NUM_feature_Classify_CNN_Indus, IMG_side_len_Classify_CNN_Indus, IMG_side_len_Classify_CNN_Indus,
		tmp_mat_ind);
	sprintf(filetrain, "../Data_all/Data_CNN_Indus/F%d/SECOM_labels_train_F%d", curN + 1, curN + 1);
	sprintf(filetest, "../Data_all/Data_CNN_Indus/F%d/SECOM_labels_test_F%d", curN + 1, curN + 1);
	allLabels = read_Lable_tabel(filetrain, filetest, NUM_samples_train_Classify_CNN_Indus, NUM_samples_test_Classify_CNN_Indus,
		NUM_class_Classify_CNN_Indus);
	//
	free(tmp_vec_ind);
	for (int i = 0; i < IMG_side_len_Classify_CNN_Indus; i++) {
		free(tmp_mat_ind[i]);
	}
	free(tmp_mat_ind);
	//#if TAG_RAND_Classify_CNN_Indus == FLAG_OFF_Classify_CNN_Indus
//    sprintf(filename, "../Data_all/Data_CNN_Indus/Data_ubyte/AllFileNames");
//#else
//    sprintf(filename, "../Data_all/Data_CNN_Indus/Data_ubyte_rand/AllFileNames");
//#endif
//    FILE* fpt;
//
//    if((fpt = fopen(filename, "r")) == NULL) {
//        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
//        exit(10000);
//    }
//
//    char StrLine[1024];
//    if(fgets(StrLine, 1024, fpt) == NULL) {
//        printf("%s(%d): No more line\n", __FILE__, __LINE__);
//        exit(-1);
//    }
//    trimLine(StrLine);
//    //printf("train set imgs %d --- %s\n", iSet + 1, StrLine);
//    allImgs = read_Img(StrLine);
//    if(fgets(StrLine, 1024, fpt) == NULL) {
//        printf("%s(%d): No more line\n", __FILE__, __LINE__);
//        exit(-1);
//    }
//    trimLine(StrLine);
//    //printf("train set label %d --- %s\n", iSet + 1, StrLine);
//    allLabels = read_Lable(StrLine);

	//
	flag_samples = (int*)calloc(allImgs->ImgNum, sizeof(int));
	indx_samples = (int*)calloc(allImgs->ImgNum, sizeof(int));
	num_samples = NUM_samples_train_Classify_CNN_Indus;
	////bootstrapInitialize_Classify_CNN();
	//multifoldInitialize_Classify_CNN();
	for (int i = 0; i < NUM_samples_all_Classify_CNN_Indus; i++) flag_samples[i] = TAG_NULL_MOP_CLASSIFY_CNN;
	for (int i = 0; i < NUM_samples_all_Classify_CNN_Indus; i++) indx_samples[i] = TAG_INVA_MOP_CLASSIFY_CNN;
	for (int i = 0; i < NUM_samples_train_Classify_CNN_Indus; i++) flag_samples[i] = 1;
	for (int i = 0; i < NUM_samples_train_Classify_CNN_Indus; i++) indx_samples[i] = i;
	shuffle_Classify_CNN(indx_samples, NUM_samples_train_Classify_CNN_Indus);

	//
	generate_img_statistics(allMinInImgs, allMaxInImgs, allMeanInImgs, allStdInImgs, allRangeInImgs,
		allImgs, allLabels, flag_samples);

	//
	int num_train = 0;
	for (int i = 0; i < allImgs->ImgNum; i++) {
		if (flag_samples[i] > 0)
			num_train++;
	}
	max_size_neighborhood = 1;
	int* num4neighbor = (int*)calloc(max_size_neighborhood + 1, sizeof(int));
	num4neighbor[0] = 0;
	int* flag_class = (int*)calloc(allLabels->LabelPtr[0].l, sizeof(int));
	flag_class[1] = 1;
	interpolate_Img(allImgs_interp, allLabels_interp, allImgs, allLabels,
		num4neighbor, max_size_neighborhood, flag_samples, num_train, flag_class);
	free(num4neighbor);
	free(flag_class);
	flag_samples_interp = (int*)calloc(allImgs_interp->ImgNum, sizeof(int));
	indx_samples_interp = (int*)calloc(allImgs_interp->ImgNum + allImgs->ImgNum - num_train, sizeof(int));
	int tmp_count1 = 0;
	int tmp_count2 = 0;
	for (int i = 0; i < allImgs->ImgNum; i++) {
		if (flag_samples[i] > 0) {
			flag_samples_interp[tmp_count1++] = flag_samples[i];
			for (int j = 0; j < flag_samples[i]; j++)
				indx_samples_interp[tmp_count2++] = tmp_count1 - 1;
		}
	}
	for (int i = num_train; i < allImgs_interp->ImgNum; i++) {
		flag_samples_interp[i] = 1;
		indx_samples_interp[tmp_count2++] = i;
	}
	num_samples_interp = tmp_count2;

	//////////////////////////////////////////////////////////////////////////
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int outSize = allLabels->LabelPtr[0].l;
	// CNN结构的初始化
	cnn_Classify = (CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn_Classify, inputSize, outSize);

	//printf("DIM_LeNet = %d\n", DIM_LeNet);
	//int t_cnt = 0;
	//for(int i = 0; i < allImgs->ImgNum; i++) {
	//    printf("%d ", flag_samples[i]);
	//    if(flag_samples[i])
	//        t_cnt++;
	//}
	//printf("\n%d -- %lf\n", t_cnt, (double)t_cnt / allImgs->ImgNum);

	NUM_pixel_Classify_CNN_Indus = inputSize.r * inputSize.c;
	NOBJ_Classify_CNN_Indus = 2;
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_OFF_Classify_CNN_Indus
	NDIM_Classify_CNN_Indus = DIM_LeNet;
#else
	NDIM_Classify_CNN_Indus = DIM_LeNet + NUM_feature_Classify_CNN_Indus;
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	NDIM_Classify_CNN_Indus++;
	//NOBJ_Classify_CNN_Indus++;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	NDIM_Classify_CNN_Indus += NUM_feature_Classify_CNN_Indus;
	//NOBJ_Classify_CNN_Indus++;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	NDIM_Classify_CNN_Indus++;
	//NOBJ_Classify_CNN_Indus++;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	NDIM_Classify_CNN_Indus += NUM_feature_Classify_CNN_Indus;
	//NOBJ_Classify_CNN_Indus++;
#endif
	//
	NDIM_Classify_CNN_Indus_BP = NDIM_Classify_CNN_Indus;
	NOBJ_Classify_CNN_Indus_BP = 6;
	//
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	seed_CNN = 237 + mpi_rank;
	rnd_uni_init_CNN = -(long)seed_CNN;
	for (int i = 0; i < curN; i++) {
		seed_CNN = (seed_CNN + 111) % 1235;
		rnd_uni_init_CNN = -(long)seed_CNN;
	}
	//
	return;
}

void Finalize_Classify_CNN()
{
	//
	free_Img(allImgs);
	free_Label(allLabels);

	//
	free(flag_samples);
	free(indx_samples);

	//
	cnnfree(cnn_Classify);

	//
	free_Img(allImgs_interp);
	free_Label(allLabels_interp);

	//
	free_Img(allMaxInImgs);
	free_Img(allMinInImgs);
	free_Img(allMeanInImgs);
	free_Img(allStdInImgs);
	free_Img(allRangeInImgs);

	//
	free(flag_samples_interp);
	free(indx_samples_interp);

	return;
}

void Fitness_Classify_CNN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	cnninit(cnn_Classify, individual, ASSIGN_MODE);

	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int outSize = allLabels->LabelPtr[0].l;

	int** pixel_index = NULL;

	int tmp_offset = DIM_LeNet;
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	int tmp_size = NUM_feature_Classify_CNN_Indus;
	double* tmp_w = (double*)malloc(tmp_size * sizeof(double));
	int* tmp_ind = (int*)malloc(tmp_size * sizeof(int));
	for (int r = 0; r < tmp_size; r++) {
		tmp_w[r] = individual[tmp_offset + r];
		tmp_ind[r] = r;
	}
	bubbleSort_Classify_CNN(tmp_w, tmp_ind, tmp_size);
	pixel_index = (int**)malloc(inputSize.r * sizeof(int*));
	int tmp_count = 0;
	for (int r = 0; r < inputSize.r; r++) {
		pixel_index[r] = (int*)malloc(inputSize.c * sizeof(int));
		for (int j = 0; j < inputSize.c; j++) {
			if (tmp_count < NUM_feature_Classify_CNN_Indus)
				pixel_index[r][j] = tmp_ind[tmp_count++];
			else
				pixel_index[r][j] = tmp_count++;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus != GENERALIZATION_NONE_Classify_CNN_Indus
	MY_FLT_TYPE*** noise_level = (MY_FLT_TYPE***)malloc(outSize * sizeof(MY_FLT_TYPE**));
	for (int le = 0; le < outSize; le++) {
		noise_level[le] = (MY_FLT_TYPE**)malloc(inputSize.r * sizeof(MY_FLT_TYPE*));
		for (int r = 0; r < inputSize.r; r++) {
			noise_level[le][r] = (MY_FLT_TYPE*)malloc(inputSize.c * sizeof(MY_FLT_TYPE));
		}
	}
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (int le = 0; le < outSize; le++) {
		for (int r = 0; r < inputSize.r; r++) {
			for (int c = 0; c < inputSize.c; c++) {
				if (r * inputSize.c + inputSize.c < NUM_feature_Classify_CNN_Indus)
					noise_level[le][r][c] = allRangeInImgs->ImgPtr[le].ImgData[r][c] * (MY_FLT_TYPE)(individual[tmp_offset]);
				else
					noise_level[le][r][c] = 0;
			}
		}
	}
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (int le = 0; le < outSize; le++) {
		for (int r = 0; r < inputSize.r; r++) {
			for (int c = 0; c < inputSize.c; c++) {
				int tmp_ind = tmp_offset + r * inputSize.c + c;
				if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
					noise_level[le][r][c] = allRangeInImgs->ImgPtr[le].ImgData[r][c] * (MY_FLT_TYPE)(individual[tmp_ind]);
				else
					noise_level[le][r][c] = 0;
			}
		}
	}
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	for (int le = 0; le < outSize; le++) {
		for (int r = 0; r < inputSize.r; r++) {
			for (int c = 0; c < inputSize.c; c++) {
				if (r * inputSize.c + inputSize.c < NUM_feature_Classify_CNN_Indus)
					noise_level[le][r][c] = (MY_FLT_TYPE)(individual[tmp_offset]);
				else
					noise_level[le][r][c] = 0;
			}
		}
	}
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	for (int le = 0; le < outSize; le++) {
		for (int r = 0; r < inputSize.r; r++) {
			for (int c = 0; c < inputSize.c; c++) {
				int tmp_ind = tmp_offset + r * inputSize.c + c;
				if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
					noise_level[le][r][c] = (MY_FLT_TYPE)(individual[tmp_ind]);
				else
					noise_level[le][r][c] = 0;
			}
		}
	}
#endif

	// CNNOpts opts;
	// opts.numepochs = 1;
	// opts.alpha = 1.0;

	//int* vec_index = (int*)calloc(outSize, sizeof(int));
	for (int r = 0; r < outSize; r++) {
		cnn_Classify->e[r] = 0.0;
		cnn_Classify->N_sum[r] = 0.0;
		cnn_Classify->N_wrong[r] = 0.0;
		cnn_Classify->e_sum[r] = 0.0;
		cnn_Classify->N_TP[r] = 0.0;
		cnn_Classify->N_TN[r] = 0.0;
		cnn_Classify->N_FP[r] = 0.0;
		cnn_Classify->N_FN[r] = 0.0;
	}

	//i = (int)(rnd_uni_LeNet(&rnd_uni_init_LeNet)*len_TrainArr) % len_TrainArr;
	//for (j = 0; j < outSize; j++){
	//  int tmp_len = arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].len;
	//  int tmp_ind = (int)(rnd_uni_LeNet(&rnd_uni_init_LeNet)*tmp_len) % tmp_len;
	//  vec_index[j] = arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData[tmp_ind];
	//}
	//cnn_err_train(cnn_Classify, allImgs, allLabels, flag_samples);
	//int num_selected = 10;
	//cnn_err_train_less(cnn_Classify, allImgs, allLabels, flag_samples, num_selected);
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus != GENERALIZATION_NONE_Classify_CNN_Indus
	cnn_err_train_with_noise(cnn_Classify, allImgs_interp, allLabels_interp, flag_samples_interp, pixel_index,
		noise_level);
#else
	cnn_err_train(cnn_Classify, allImgs_interp, allLabels_interp, flag_samples_interp, pixel_index);
#endif
	//float incorrectRatio = 0.0;
	//incorrectRatio = cnntest(cnn, trainImg[i], trainLabel[i], trainNum);
	//cnntest_selected(cnn, trainImg[i], trainLabel[i], outSize, vec_index);
	//printf("train %d finished!!\nincorrectRatio = %lf%%\n", i + 1, incorrectRatio * 100);
	//int j, k;
	//for (j = 0; j < outSize; j++){
	//  int tmp = arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].len;
	//  printf("train: Label %d %d:%d - \n", i, j, tmp);
	//  for (k = 0; k < tmp; k++){
	//      printf("%d ",
	//          arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData[k]);
	//  }
	//  printf("\n");
	//}

	//
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	free(tmp_w);
	free(tmp_ind);
	for (int r = 0; r < inputSize.r; r++) {
		free(pixel_index[r]);
	}
	free(pixel_index);
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus != GENERALIZATION_NONE_Classify_CNN_Indus
	for (int le = 0; le < outSize; le++) {
		for (int r = 0; r < inputSize.r; r++) {
			free(noise_level[le][r]);
		}
		free(noise_level[le]);
	}
	free(noise_level);
#endif

	//int j;
	//for (j = 0; j < outSize; j++){
	//  printf("train: Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
	//      j, cnn->N_TP[j], cnn->N_TN[j], cnn->N_FP[j], cnn->N_FN[j],
	//      cnn->N_sum[j], cnn->N_wrong[j], cnn->e_sum[j],
	//      cnn->N_wrong[j] / cnn->N_sum[j], cnn->e_sum[j] / cnn->N_sum[j]);
	//}
	//float e_sum_all = 0.0;
	//float N_wrong_all = 0.0;
	//float N_all = 0.0;
	//float N_wr_weighted = 0.0;
	//float e_max = 0.0;
	//float N_wr_max = 0.0;
	//for (i = 0; i < outSize; i++){
	//  e_sum_all += cnn->e_sum[i];
	//  N_wrong_all += cnn->N_wrong[i];
	//  N_all += cnn->N_sum[i];
	//  if (e_max < cnn->e_sum[i] / sum_len)
	//      e_max = cnn->e_sum[i] / sum_len;
	//  if (N_wr_max < cnn->N_wrong[i] / cnn->N_sum[i])
	//      N_wr_max = cnn->N_wrong[i] / cnn->N_sum[i];
	//  N_wr_weighted += cnn->N_wrong[i] / cnn->N_sum[i];
	//  //fitness[i] = cnn->e_sum[i] / sum_len;
	//  //fitness[i] = cnn->N_wrong[i] / cnn->N_sum[i];
	//}
	//e_sum_all /= (outSize*N_all);
	//N_wrong_all /= N_all;
	//N_wr_weighted /= outSize;
	//fitness[0] = N_wrong_all;
	//fitness[1] = N_wr_max;
	//fitness[2] = N_wr_weighted;
	//for(i = 0; i < outSize; i++) {
	//    fitness[i] = cnn->N_wrong[i] / cnn->N_sum[i];
	//}

	//
	MY_FLT_TYPE mean_precision = 0;
	MY_FLT_TYPE std_precision = 0;
	MY_FLT_TYPE mean_recall = 0;
	MY_FLT_TYPE std_recall = 0;
	MY_FLT_TYPE mean_ber = 0;
	MY_FLT_TYPE std_ber = 0;
	get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
		cnn_Classify->N_wrong, cnn_Classify->N_sum,
		mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

	////fitness[0] = mean_err_rt;
	////fitness[1] = max_err_rt;
	//fitness[0] = 1.0 - mean_precision;
	//fitness[1] = std_precision;

	MY_FLT_TYPE mean_err = 0;
	MY_FLT_TYPE std_err = 0;
	for (int r = 0; r < outSize; r++) {
		if (cnn_Classify->N_sum[r])
			mean_err += cnn_Classify->e_sum[r] / cnn_Classify->N_sum[r];
	}
	mean_err /= outSize;
	for (int r = 0; r < outSize; r++) {
		std_err += (cnn_Classify->e_sum[r] / cnn_Classify->N_sum[r] - mean_err) *
			(cnn_Classify->e_sum[r] / cnn_Classify->N_sum[r] - mean_err);
	}
	std_err /= outSize;
	std_err = sqrt(std_err);

	fitness[0] = mean_ber;
	fitness[1] = std_ber;
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#else
	//float tmp_val = 0;
	//for(int r = 0; r < DIM_ALL_PARA_CNN; r++) {
	//    tmp_val += fabs(individual[r]);
	//}
	//tmp_val /= DIM_ALL_PARA_CNN;
	//tmp_val /= MAX_WEIGHT_BIAS_CNN;
	//fitness[1] = std_precision;// tmp_val;
#endif

	//cnninit(cnn, individual, OUTPUT_MODE);

	return;
}

void Fitness_Classify_CNN_validation(double* individual, double* fitness)
{
	//printf("Dim info: %d %d %d %d %d %d %d\n",
	//  NUM_PARA_C1_M, NUM_PARA_C1_B,
	//  NUM_PARA_C3_M, NUM_PARA_C3_B,
	//  NUM_PARA_O5_M, NUM_PARA_O5_B,
	//  DIM_LeNet);

	cnninit(cnn_Classify, individual, ASSIGN_MODE);

	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int outSize = allLabels->LabelPtr[0].l;

	int** pixel_index = NULL;

	int tmp_offset = DIM_LeNet;
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	int tmp_size = NUM_feature_Classify_CNN_Indus;
	double* tmp_w = (double*)malloc(tmp_size * sizeof(double));
	int* tmp_ind = (int*)malloc(tmp_size * sizeof(int));
	for (int r = 0; r < tmp_size; r++) {
		tmp_w[r] = individual[tmp_offset + r];
		tmp_ind[r] = r;
	}
	bubbleSort_Classify_CNN(tmp_w, tmp_ind, tmp_size);
	pixel_index = (int**)malloc(inputSize.r * sizeof(int*));
	int tmp_count = 0;
	for (int r = 0; r < inputSize.r; r++) {
		pixel_index[r] = (int*)malloc(inputSize.c * sizeof(int));
		for (int j = 0; j < inputSize.c; j++) {
			if (tmp_count < NUM_feature_Classify_CNN_Indus)
				pixel_index[r][j] = tmp_ind[tmp_count++];
			else
				pixel_index[r][j] = tmp_count++;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif

	int i;
	for (i = 0; i < outSize; i++) {
		cnn_Classify->e[i] = 0.0;
		cnn_Classify->N_sum[i] = 0.0;
		cnn_Classify->N_wrong[i] = 0.0;
		cnn_Classify->e_sum[i] = 0.0;
		cnn_Classify->N_TP[i] = 0.0;
		cnn_Classify->N_TN[i] = 0.0;
		cnn_Classify->N_FP[i] = 0.0;
		cnn_Classify->N_FN[i] = 0.0;
	}
	//
	cnn_err_test(cnn_Classify, allImgs, allLabels, flag_samples, TAG_VALI_MOP_CLASSIFY_CNN, pixel_index);
	//
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	free(tmp_w);
	free(tmp_ind);
	for (int i = 0; i < inputSize.r; i++) {
		free(pixel_index[i]);
	}
	free(pixel_index);
#endif
	//
	MY_FLT_TYPE mean_precision = 0;
	MY_FLT_TYPE std_precision = 0;
	MY_FLT_TYPE mean_recall = 0;
	MY_FLT_TYPE std_recall = 0;
	MY_FLT_TYPE mean_ber = 0;
	MY_FLT_TYPE std_ber = 0;
	get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
		cnn_Classify->N_wrong, cnn_Classify->N_sum,
		mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

	//fitness[0] = mean_err_rt;
	//fitness[1] = max_err_rt;
	fitness[0] = mean_ber;
	fitness[1] = std_ber;
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#else
	//float tmp_val = 0;
	//for(int r = 0; r < DIM_ALL_PARA_CNN; r++) {
	//    tmp_val += fabs(individual[r]);
	//}
	//tmp_val /= DIM_ALL_PARA_CNN;
	//tmp_val /= MAX_WEIGHT_BIAS_CNN;
	//fitness[1] = std_precision;// tmp_val;
#endif

	//cnninit(cnn, individual, OUTPUT_MODE);

	return;
}

void Fitness_Classify_CNN_test(double* individual, double* fitness)
{
	//printf("Dim info: %d %d %d %d %d %d %d\n",
	//  NUM_PARA_C1_M, NUM_PARA_C1_B,
	//  NUM_PARA_C3_M, NUM_PARA_C3_B,
	//  NUM_PARA_O5_M, NUM_PARA_O5_B,
	//  DIM_LeNet);

	cnninit(cnn_Classify, individual, ASSIGN_MODE);

	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int outSize = allLabels->LabelPtr[0].l;

	int** pixel_index = NULL;

	int tmp_offset = DIM_LeNet;
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	int tmp_size = NUM_feature_Classify_CNN_Indus;
	double* tmp_w = (double*)malloc(tmp_size * sizeof(double));
	int* tmp_ind = (int*)malloc(tmp_size * sizeof(int));
	for (int r = 0; r < tmp_size; r++) {
		tmp_w[r] = individual[tmp_offset + r];
		tmp_ind[r] = r;
	}
	bubbleSort_Classify_CNN(tmp_w, tmp_ind, tmp_size);
	pixel_index = (int**)malloc(inputSize.r * sizeof(int*));
	int tmp_count = 0;
	for (int r = 0; r < inputSize.r; r++) {
		pixel_index[r] = (int*)malloc(inputSize.c * sizeof(int));
		for (int j = 0; j < inputSize.c; j++) {
			if (tmp_count < NUM_feature_Classify_CNN_Indus)
				pixel_index[r][j] = tmp_ind[tmp_count++];
			else
				pixel_index[r][j] = tmp_count++;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif

	int i;
	for (i = 0; i < outSize; i++) {
		cnn_Classify->e[i] = 0.0;
		cnn_Classify->N_sum[i] = 0.0;
		cnn_Classify->N_wrong[i] = 0.0;
		cnn_Classify->e_sum[i] = 0.0;
		cnn_Classify->N_TP[i] = 0.0;
		cnn_Classify->N_TN[i] = 0.0;
		cnn_Classify->N_FP[i] = 0.0;
		cnn_Classify->N_FN[i] = 0.0;
	}
	//
	cnn_err_test(cnn_Classify, allImgs, allLabels, flag_samples, TAG_NULL_MOP_CLASSIFY_CNN, pixel_index);
	//
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	free(tmp_w);
	free(tmp_ind);
	for (int i = 0; i < inputSize.r; i++) {
		free(pixel_index[i]);
	}
	free(pixel_index);
#endif
	//
	MY_FLT_TYPE mean_precision = 0;
	MY_FLT_TYPE std_precision = 0;
	MY_FLT_TYPE mean_recall = 0;
	MY_FLT_TYPE std_recall = 0;
	MY_FLT_TYPE mean_ber = 0;
	MY_FLT_TYPE std_ber = 0;
	get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
		cnn_Classify->N_wrong, cnn_Classify->N_sum,
		mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

	//fitness[0] = mean_err_rt;
	//fitness[1] = max_err_rt;
	fitness[0] = mean_ber;
	fitness[1] = std_ber;
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	//fitness[2] = (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_offset])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	double tmp_nl = 0.0;
	for (int r = 0; r < inputSize.r; r++) {
		for (int c = 0; c < inputSize.c; c++) {
			int tmp_ind = tmp_offset + r * inputSize.c + c;
			if (tmp_ind - tmp_offset < NUM_feature_Classify_CNN_Indus)
				tmp_nl += (MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN - fabs(individual[tmp_ind])) / MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		}
	}
	//fitness[2] = tmp_nl / NUM_feature_Classify_CNN_Indus;
#else
	//float tmp_val = 0;
	//for(int r = 0; r < DIM_ALL_PARA_CNN; r++) {
	//    tmp_val += fabs(individual[r]);
	//}
	//tmp_val /= DIM_ALL_PARA_CNN;
	//tmp_val /= MAX_WEIGHT_BIAS_CNN;
	//fitness[1] = std_precision;// tmp_val;
#endif

	//cnninit(cnn, individual, OUTPUT_MODE);

	return;
}

void Fitness_raw_Classify_CNN(double* individual, double* fitness_raw)
{
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int outSize = allLabels->LabelPtr[0].l;

	double* indiv = (double*)calloc(NDIM_Classify_CNN_Indus, sizeof(double));
	double* fitness = (double*)calloc(NOBJ_Classify_CNN_Indus, sizeof(double));

	memcpy(indiv, individual, NDIM_Classify_CNN_Indus * sizeof(double));

#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	int tmp_offset_n = NDIM_Classify_CNN_Indus - 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	int tmp_offset_n = NDIM_Classify_CNN_Indus - NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	int tmp_offset_n = NDIM_Classify_CNN_Indus - 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	int tmp_offset_n = NDIM_Classify_CNN_Indus - NUM_feature_Classify_CNN_Indus;
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus != GENERALIZATION_NONE_Classify_CNN_Indus
	for (int i = tmp_offset_n; i < NDIM_Classify_CNN_Indus; i++) {
		indiv[i] = 0;
	}
#endif

	Fitness_Classify_CNN(indiv, fitness, NULL, NDIM_Classify_CNN_Indus, NOBJ_Classify_CNN_Indus);

	int tmp_offset = 0;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_TP[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_FP[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_TN[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_FN[i];
	}
	tmp_offset += outSize;

	Fitness_Classify_CNN_test(indiv, fitness);

	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_TP[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_FP[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_TN[i];
	}
	tmp_offset += outSize;
	for (int i = 0; i < outSize; i++) {
		fitness_raw[tmp_offset + i] = cnn_Classify->N_FN[i];
	}

	free(indiv);
	free(fitness);

	return;
}

void SetLimits_Classify_CNN(double* minLimit, double* maxLimit, int nx)
{
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int tmp_offset = 0;
	int k;
	for (k = tmp_offset; k < tmp_offset + DIM_ALL_PARA_CNN; k++) {
		minLimit[k] = -MAX_WEIGHT_BIAS_CNN;
		maxLimit[k] = MAX_WEIGHT_BIAS_CNN;
	}
	tmp_offset += DIM_ALL_PARA_CNN;
#if OPTIMIZE_STRUCTURE_CNN == 1
	for (k = tmp_offset; k < tmp_offset + DIM_ALL_STRU_CNN; k++) {
		minLimit[k] = 0.0;
		maxLimit[k] = 2.0 - 1e-6;
	}
	tmp_offset += DIM_ALL_STRU_CNN;
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		minLimit[k] = 0.0;
		maxLimit[k] = 1.0;
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + 1; k++) {
		minLimit[k] = -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		maxLimit[k] = MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
	}
	tmp_offset += 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		minLimit[k] = -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		maxLimit[k] = MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + 1; k++) {
		minLimit[k] = -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		maxLimit[k] = MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
	}
	tmp_offset += 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		minLimit[k] = -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
		maxLimit[k] = MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN;
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif
	//
	return;
}

int  CheckLimits_Classify_CNN(double* x, int nx)
{
	nSize inputSize = { allImgs->ImgPtr[0].c, allImgs->ImgPtr[0].r };
	int tmp_offset = 0;
	int k;
	for (k = tmp_offset; k < tmp_offset + DIM_ALL_PARA_CNN; k++) {
		if (x[k] < -MAX_WEIGHT_BIAS_CNN || x[k] > MAX_WEIGHT_BIAS_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_WEIGHT_BIAS_CNN, MAX_WEIGHT_BIAS_CNN);
			return false;
		}
	}
	tmp_offset += DIM_ALL_PARA_CNN;
#if OPTIMIZE_STRUCTURE_CNN == 1
	for (k = tmp_offset; k < tmp_offset + DIM_ALL_STRU_CNN; k++) {
		if (x[k] < 0 || x[k] > 2.0 - 1e-6) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], 0, 2.0 - 1e-6);
			return false;
		}
	}
	tmp_offset += DIM_ALL_STRU_CNN;
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		if (x[k] < 0.0 || x[k] > 1.0) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], 0.0, 1.0);
			return false;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + 1; k++) {
		if (x[k] < -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN || x[k] > MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN, MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN);
			return false;
		}
	}
	tmp_offset += 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		if (x[k] < -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN || x[k] > MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN, MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN);
			return false;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + 1; k++) {
		if (x[k] < -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN || x[k] > MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN, MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN);
			return false;
		}
	}
	tmp_offset += 1;
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
	for (k = tmp_offset; k < tmp_offset + NUM_feature_Classify_CNN_Indus; k++) {
		if (x[k] < -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN || x[k] > MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN, MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN);
			return false;
		}
	}
	tmp_offset += NUM_feature_Classify_CNN_Indus;
#endif
	return true;
}

void Fitness_Classify_CNN_Indus_BP(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	int num_samples = allImgs->ImgNum;
	int outSize = allLabels->LabelPtr[0].l;
	int maxNum_epochs = 100000;

	// CNN训练
	cnninit(cnn_Classify, NULL, INIT_MODE);
	CNNOpts opts;
	opts.numepochs = 1;
	opts.batchSize = 128;
	opts.alpha = (MY_FLT_TYPE)0.2;
	int flag_stop = 0;
	int cur_epochs = 0;
	MY_FLT_TYPE pre_prc = 0;
	int count_prc = 0;
	int th_prc = 10;

	while (cur_epochs < maxNum_epochs && !flag_stop) {
		for (int i = 0; i < outSize; i++) {
			cnn_Classify->e[i] = 0.0;
			cnn_Classify->N_sum[i] = 0.0;
			cnn_Classify->N_wrong[i] = 0.0;
			cnn_Classify->e_sum[i] = 0.0;
			cnn_Classify->N_TP[i] = 0.0;
			cnn_Classify->N_TN[i] = 0.0;
			cnn_Classify->N_FP[i] = 0.0;
			cnn_Classify->N_FN[i] = 0.0;
		}
		//
		shuffle_Classify_CNN(indx_samples_interp, num_samples_interp);
		//cnn_train_bp(cnn_Classify, allImgs, allLabels, opts, indx_samples);
		//if(num_samples_interp)
		cnn_train_bp(cnn_Classify, allImgs_interp, allLabels_interp, opts, indx_samples_interp, num_samples_interp);
		// 保存训练误差
		char filename[256];
		FILE* fp = NULL;
		sprintf(filename, "cnnL_R%dE%d.ma", repNo, cur_epochs);
		fp = fopen(filename, "wb");
		if (fp == NULL)
			printf("write file failed\n");
		fwrite(cnn_Classify->L, sizeof(MY_FLT_TYPE), num_samples_interp, fp);
		fclose(fp);
		free(cnn_Classify->L);

		for (int i = 0; i < outSize; i++) {
			cnn_Classify->e[i] = 0.0;
			cnn_Classify->N_sum[i] = 0.0;
			cnn_Classify->N_wrong[i] = 0.0;
			cnn_Classify->e_sum[i] = 0.0;
			cnn_Classify->N_TP[i] = 0.0;
			cnn_Classify->N_TN[i] = 0.0;
			cnn_Classify->N_FP[i] = 0.0;
			cnn_Classify->N_FN[i] = 0.0;
		}
		cnn_err_test(cnn_Classify, allImgs, allLabels, flag_samples, TAG_VALI_MOP_CLASSIFY_CNN, NULL);
		//
		MY_FLT_TYPE mean_precision = 0;
		MY_FLT_TYPE std_precision = 0;
		MY_FLT_TYPE mean_recall = 0;
		MY_FLT_TYPE std_recall = 0;
		MY_FLT_TYPE mean_ber = 0;
		MY_FLT_TYPE std_ber = 0;
		get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
			cnn_Classify->N_wrong, cnn_Classify->N_sum,
			mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

		if (mean_precision < pre_prc) {
			count_prc++;
		}
		else {
			count_prc = 0;
		}
		if (count_prc >= th_prc) {
			flag_stop = 1;
		}
		pre_prc = mean_precision;

		cur_epochs++;
	}
	printf("train finished!!\n");
	char filename[256];
	sprintf(filename, "train%d.cnn", repNo);
	savecnn(cnn_Classify, filename);
	//
	//for(i = 0; i < outSize; i++) {
	//    printf("train: Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
	//           i, cnn_Classify->N_TP[i], cnn_Classify->N_TN[i], cnn_Classify->N_FP[i], cnn_Classify->N_FN[i],
	//           cnn_Classify->N_sum[i], cnn_Classify->N_wrong[i], cnn_Classify->e_sum[i],
	//           cnn_Classify->N_wrong[i] / cnn_Classify->N_sum[i], cnn_Classify->e_sum[i]);
	//}
	//
	// CNN测试
	//importcnn(cnn, "minst.cnn");
	for (int i = 0; i < outSize; i++) {
		cnn_Classify->e[i] = 0.0;
		cnn_Classify->N_sum[i] = 0.0;
		cnn_Classify->N_wrong[i] = 0.0;
		cnn_Classify->e_sum[i] = 0.0;
		cnn_Classify->N_TP[i] = 0.0;
		cnn_Classify->N_TN[i] = 0.0;
		cnn_Classify->N_FP[i] = 0.0;
		cnn_Classify->N_FN[i] = 0.0;
	}
	//cnn_err_train(cnn_Classify, allImgs, allLabels, flag_samples);
	cnn_err_train(cnn_Classify, allImgs_interp, allLabels_interp, flag_samples_interp, NULL);
	////
	//double mean_err_rt = 0.0;
	//double max_err_rt = 0.0;
	//for(int i = 0; i < outSize; i++) {
	//    double tmp_rt = cnn_Classify->N_wrong[i] / cnn_Classify->N_sum[i];
	//    mean_err_rt += tmp_rt;
	//    if(max_err_rt < tmp_rt)
	//        max_err_rt = tmp_rt;
	//}
	//mean_err_rt /= outSize;
	////
	//fitness[0] = mean_err_rt;
	//fitness[1] = max_err_rt;
	//
	{
		MY_FLT_TYPE mean_precision = 0;
		MY_FLT_TYPE std_precision = 0;
		MY_FLT_TYPE mean_recall = 0;
		MY_FLT_TYPE std_recall = 0;
		MY_FLT_TYPE mean_ber = 0;
		MY_FLT_TYPE std_ber = 0;
		get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
			cnn_Classify->N_wrong, cnn_Classify->N_sum,
			mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

		//fitness[0] = mean_err_rt;
		//fitness[1] = max_err_rt;
		fitness[0] = mean_ber;
		fitness[1] = std_ber;
	}
	//
	for (int i = 0; i < outSize; i++) {
		cnn_Classify->e[i] = 0.0;
		cnn_Classify->N_sum[i] = 0.0;
		cnn_Classify->N_wrong[i] = 0.0;
		cnn_Classify->e_sum[i] = 0.0;
		cnn_Classify->N_TP[i] = 0.0;
		cnn_Classify->N_TN[i] = 0.0;
		cnn_Classify->N_FP[i] = 0.0;
		cnn_Classify->N_FN[i] = 0.0;
	}
	cnn_err_test(cnn_Classify, allImgs, allLabels, flag_samples, TAG_VALI_MOP_CLASSIFY_CNN, NULL);
	////
	//mean_err_rt = 0.0;
	//max_err_rt = 0.0;
	//for(int i = 0; i < outSize; i++) {
	//    double tmp_rt = cnn_Classify->N_wrong[i] / cnn_Classify->N_sum[i];
	//    mean_err_rt += tmp_rt;
	//    if(max_err_rt < tmp_rt)
	//        max_err_rt = tmp_rt;
	//}
	//mean_err_rt /= outSize;
	////
	//fitness[2] = mean_err_rt;
	//fitness[3] = max_err_rt;
	//
	{
		MY_FLT_TYPE mean_precision = 0;
		MY_FLT_TYPE std_precision = 0;
		MY_FLT_TYPE mean_recall = 0;
		MY_FLT_TYPE std_recall = 0;
		MY_FLT_TYPE mean_ber = 0;
		MY_FLT_TYPE std_ber = 0;
		get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
			cnn_Classify->N_wrong, cnn_Classify->N_sum,
			mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

		//fitness[0] = mean_err_rt;
		//fitness[1] = max_err_rt;
		fitness[2] = mean_ber;
		fitness[3] = std_ber;
	}
	//
	for (int i = 0; i < outSize; i++) {
		cnn_Classify->e[i] = 0.0;
		cnn_Classify->N_sum[i] = 0.0;
		cnn_Classify->N_wrong[i] = 0.0;
		cnn_Classify->e_sum[i] = 0.0;
		cnn_Classify->N_TP[i] = 0.0;
		cnn_Classify->N_TN[i] = 0.0;
		cnn_Classify->N_FP[i] = 0.0;
		cnn_Classify->N_FN[i] = 0.0;
	}
	cnn_err_test(cnn_Classify, allImgs, allLabels, flag_samples, TAG_NULL_MOP_CLASSIFY_CNN, NULL);
	////
	//mean_err_rt = 0.0;
	//max_err_rt = 0.0;
	//for(int i = 0; i < outSize; i++) {
	//    double tmp_rt = cnn_Classify->N_wrong[i] / cnn_Classify->N_sum[i];
	//    mean_err_rt += tmp_rt;
	//    if(max_err_rt < tmp_rt)
	//        max_err_rt = tmp_rt;
	//}
	//mean_err_rt /= outSize;
	////
	//fitness[2] = mean_err_rt;
	//fitness[3] = max_err_rt;
	//
	{
		MY_FLT_TYPE mean_precision = 0;
		MY_FLT_TYPE std_precision = 0;
		MY_FLT_TYPE mean_recall = 0;
		MY_FLT_TYPE std_recall = 0;
		MY_FLT_TYPE mean_ber = 0;
		MY_FLT_TYPE std_ber = 0;
		get_Evaluation_Indicators(outSize, cnn_Classify->N_TP, cnn_Classify->N_FP, cnn_Classify->N_TN, cnn_Classify->N_FN,
			cnn_Classify->N_wrong, cnn_Classify->N_sum,
			mean_precision, std_precision, mean_recall, std_recall, mean_ber, std_ber);

		//fitness[0] = mean_err_rt;
		//fitness[1] = max_err_rt;
		fitness[4] = mean_ber;
		fitness[5] = std_ber;
	}

	return;
}