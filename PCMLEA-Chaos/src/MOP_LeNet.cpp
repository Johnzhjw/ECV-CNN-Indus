#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_cnn.h"
#include "MOP_cnn_data.h"
#include "MOP_LeNet.h"

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

////////
////////
LabelArr* trainLabel;
ImgArr* trainImg;
LabelArr* testLabel;
ImgArr* testImg;
int len_TrainArr;
int len_TestArr;

SetArrLabelIndexPtr arr_index_train;
SetArrLabelIndexPtr arr_index_test;

CNN* cnn;

double* storedCNN;
int     num_storedCNN;
MY_FLT_TYPE* all_y_train;
MY_FLT_TYPE* all_y_test;
////////

void Initialize_data_LeNet(int curN, int numN, int trainNo, int testNo, int endNo)
{
	seed_CNN = 237;
	rnd_uni_init_CNN = -(long)seed_CNN;
	for (int i = 0; i < curN; i++) {
		seed_CNN = (seed_CNN + 111) % 1235;
		rnd_uni_init_CNN = -(long)seed_CNN;
	}

	len_TrainArr = (testNo - trainNo) / 2;
	len_TestArr = (endNo - testNo) / 2;
	trainImg = (ImgArr*)calloc(len_TrainArr, sizeof(ImgArr));
	trainLabel = (LabelArr*)calloc(len_TrainArr, sizeof(LabelArr));
	testImg = (ImgArr*)calloc(len_TestArr, sizeof(ImgArr));
	testLabel = (LabelArr*)calloc(len_TestArr, sizeof(LabelArr));

	char filename[1024] = "../Data_all/AllFileNames";
	FILE* fpt;

	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}

	char StrLine[1024];
	int seq = 0;
	for (seq = 1; seq < trainNo; seq++) {
		if (fgets(StrLine, 1024, fpt) == NULL) {
			printf("%s(%d): No more line\n", __FILE__, __LINE__);
			exit(-1);
		}
	}
	int iSet;
	for (seq = trainNo; seq < testNo; seq += 2) {
		iSet = (seq - trainNo) / 2;
		if (fgets(StrLine, 1024, fpt) == NULL) {
			printf("%s(%d): No more line\n", __FILE__, __LINE__);
			exit(-1);
		}
		trimLine(StrLine);
		//printf("train set imgs %d --- %s\n", iSet + 1, StrLine);
		trainImg[iSet] = read_Img(StrLine);
		if (fgets(StrLine, 1024, fpt) == NULL) {
			printf("%s(%d): No more line\n", __FILE__, __LINE__);
			exit(-1);
		}
		trimLine(StrLine);
		//printf("train set label %d --- %s\n", iSet + 1, StrLine);
		trainLabel[iSet] = read_Label(StrLine);
	}
	for (seq = testNo; seq < endNo; seq += 2) {
		iSet = (seq - testNo) / 2;
		if (fgets(StrLine, 1024, fpt) == NULL) {
			printf("%s(%d): No more line\n", __FILE__, __LINE__);
			exit(-1);
		}
		trimLine(StrLine);
		//printf("test set imgs %d --- %s\n", iSet + 1, StrLine);
		testImg[iSet] = read_Img(StrLine);
		if (fgets(StrLine, 1024, fpt) == NULL) {
			printf("%s(%d): No more line\n", __FILE__, __LINE__);
			exit(-1);
		}
		trimLine(StrLine);
		//printf("test set label %d --- %s\n", iSet + 1, StrLine);
		testLabel[iSet] = read_Label(StrLine);
	}

	arr_index_train = getLabelIndex(trainLabel, len_TrainArr);
	arr_index_test = getLabelIndex(testLabel, len_TestArr);

	nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	int outSize = testLabel[0]->LabelPtr[0].l;

	// CNN结构的初始化
	cnn = (CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn, inputSize, outSize);

	return;
}

void Finalize_LeNet()
{
	//
	freeSetArrLabelIndexPtr(arr_index_train);
	freeSetArrLabelIndexPtr(arr_index_test);
	int i;
	for (i = 0; i < len_TrainArr; i++) {
		free_Img(trainImg[i]);
		free_Label(trainLabel[i]);
	}
	free(trainImg);
	free(trainLabel);
	for (i = 0; i < len_TestArr; i++) {
		free_Img(testImg[i]);
		free_Label(testLabel[i]);
	}
	free(testImg);
	free(testLabel);

	//
	cnnfree(cnn);

	return;
}

/*主函数*/
void Fitness_LeNet_BP(double* fitness)
{
	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	int outSize = testLabel[0]->LabelPtr[0].l;

	// CNN训练
	cnninit(cnn, NULL, INIT_MODE);
	CNNOpts opts;
	opts.numepochs = 1;
	opts.alpha = 1.0;
	int i;
	for (i = 0; i < outSize; i++) {
		cnn->N_sum[i] = 0.0;
		cnn->N_wrong[i] = 0.0;
		cnn->e_sum[i] = 0.0;
		cnn->N_TP[i] = 0.0;
		cnn->N_TN[i] = 0.0;
		cnn->N_FP[i] = 0.0;
		cnn->N_FN[i] = 0.0;
	}
	int sum_len = 0;
	for (i = 0; i < len_TrainArr; i++) {
		int trainNum = trainImg[i]->ImgNum; //55000;
		sum_len += trainNum;
		//cnntrain(cnn, trainImg[i], trainLabel[i], opts, trainNum);
		cnntrain_selected(cnn, trainImg[i], trainLabel[i], opts, trainNum, arr_index_train->ArrLabelIndexPtr[i]);
		printf("train finished!!\n");
		char filename[256];
		sprintf(filename, "train%02d.cnn", i);
		savecnn(cnn, filename);

		// 保存训练误差
		FILE* fp = NULL;
		sprintf(filename, "PicTrans/cnnL%2d.ma", i);
		fp = fopen(filename, "wb");
		if (fp == NULL)
			printf("write file failed\n");
		fwrite(cnn->L, sizeof(MY_FLT_TYPE), trainNum, fp);
		fclose(fp);

		free(cnn->L);
	}
	for (i = 0; i < outSize; i++) {
		printf("train: Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
			i, cnn->N_TP[i], cnn->N_TN[i], cnn->N_FP[i], cnn->N_FN[i],
			cnn->N_sum[i], cnn->N_wrong[i], cnn->e_sum[i],
			cnn->N_wrong[i] / cnn->N_sum[i], cnn->e_sum[i] / sum_len);
	}

	for (i = 0; i < outSize; i++) {
		cnn->N_sum[i] = 0.0;
		cnn->N_wrong[i] = 0.0;
		cnn->e_sum[i] = 0.0;
		cnn->N_TP[i] = 0.0;
		cnn->N_TN[i] = 0.0;
		cnn->N_FP[i] = 0.0;
		cnn->N_FN[i] = 0.0;
	}
	sum_len = 0;
	// CNN测试
	//importcnn(cnn, "minst.cnn");
	int testNum = testImg[0]->ImgNum; //10000;
	sum_len += testNum;
	MY_FLT_TYPE incorrectRatio = 0.0;
	incorrectRatio = cnntest(cnn, testImg[0], testLabel[0], testNum);
	printf("test finished!!\nincorrectRatio = %lf%%\n", incorrectRatio * 100);
	for (i = 0; i < outSize; i++) {
		printf("test: Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
			i, cnn->N_TP[i], cnn->N_TN[i], cnn->N_FP[i], cnn->N_FN[i],
			cnn->N_sum[i], cnn->N_wrong[i], cnn->e_sum[i],
			cnn->N_wrong[i] / cnn->N_sum[i], cnn->e_sum[i] / sum_len);

		fitness[i] = cnn->e_sum[i] / sum_len;
	}

	return;
}

void SetLimits_LeNet(double* minLimit, double* maxLimit, int nx)
{
	int k;
	for (k = 0; k < DIM_LeNet; k++) {
		minLimit[k] = -MAX_WEIGHT_BIAS_CNN;
		maxLimit[k] = MAX_WEIGHT_BIAS_CNN;
	}
}

int CheckLimits_LeNet(double* x, int nx)
{
	int k;
	for (k = 0; k < DIM_LeNet; k++) {
		if (x[k] < -MAX_WEIGHT_BIAS_CNN || x[k] > MAX_WEIGHT_BIAS_CNN) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], -MAX_WEIGHT_BIAS_CNN, MAX_WEIGHT_BIAS_CNN);
			return false;
		}
	}
	return true;
}

void InitiateIndiv_LeNet(double* individual)
{
	cnninit(cnn, NULL, INIT_MODE);
	cnninit(cnn, individual, OUTPUT_MODE);
}

void Fitness_LeNet(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	//printf("Dim info: %d %d %d %d %d %d %d\n",
	//  NUM_PARA_C1_M, NUM_PARA_C1_B,
	//  NUM_PARA_C3_M, NUM_PARA_C3_B,
	//  NUM_PARA_O5_M, NUM_PARA_O5_B,
	//  DIM_LeNet);

	cnninit(cnn, individual, ASSIGN_MODE);

	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	int outSize = testLabel[0]->LabelPtr[0].l;

	// CNNOpts opts;
	// opts.numepochs = 1;
	// opts.alpha = 1.0;

	//int* vec_index = (int*)calloc(outSize, sizeof(int));
	int i;
	for (i = 0; i < outSize; i++) {
		cnn->N_sum[i] = 0.0;
		cnn->N_wrong[i] = 0.0;
		cnn->e_sum[i] = 0.0;
		cnn->N_TP[i] = 0.0;
		cnn->N_TN[i] = 0.0;
		cnn->N_FP[i] = 0.0;
		cnn->N_FN[i] = 0.0;
	}
	int sum_len = 0;
	for (i = 0; i < len_TrainArr; i++) {
		//i = (int)(rnd_uni_LeNet(&rnd_uni_init_LeNet)*len_TrainArr) % len_TrainArr;
		//for (j = 0; j < outSize; j++){
		//  int tmp_len = arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].len;
		//  int tmp_ind = (int)(rnd_uni_LeNet(&rnd_uni_init_LeNet)*tmp_len) % tmp_len;
		//  vec_index[j] = arr_index_train->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData[tmp_ind];
		//}
		int trainNum = trainImg[i]->ImgNum; //55000;
		sum_len += trainNum;
		//cnntrain(cnn, trainImg[i], trainLabel[i], opts, trainNum);
		cnntest(cnn, trainImg[i], trainLabel[i], trainNum);
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
	}

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
	for (i = 0; i < outSize; i++) {
		fitness[i] = cnn->N_wrong[i] / cnn->N_sum[i];
	}

	//cnninit(cnn, individual, OUTPUT_MODE);

	return;
}

void Fitness_LeNet_test(double* individual, double* fitness)
{
	//printf("Dim info: %d %d %d %d %d %d %d\n",
	//  NUM_PARA_C1_M, NUM_PARA_C1_B,
	//  NUM_PARA_C3_M, NUM_PARA_C3_B,
	//  NUM_PARA_O5_M, NUM_PARA_O5_B,
	//  DIM_LeNet);

	cnninit(cnn, individual, ASSIGN_MODE);

	// nSize inputSize = { testImg[0]->ImgPtr[0].c, testImg[0]->ImgPtr[0].r };
	int outSize = testLabel[0]->LabelPtr[0].l;

	int i;
	for (i = 0; i < outSize; i++) {
		cnn->N_sum[i] = 0.0;
		cnn->N_wrong[i] = 0.0;
		cnn->e_sum[i] = 0.0;
		cnn->N_TP[i] = 0.0;
		cnn->N_TN[i] = 0.0;
		cnn->N_FP[i] = 0.0;
		cnn->N_FN[i] = 0.0;
	}
	int sum_len = 0;
	for (i = 0; i < len_TestArr; i++) {
		int testNum = testImg[i]->ImgNum; //10000;
		sum_len += testNum;
		// float incorrectRatio = 0.0;
		// incorrectRatio =
		cnntest(cnn, testImg[i], testLabel[i], testNum);
		//printf("test %d finished!!\nincorrectRatio = %lf%%\n", i + 1, incorrectRatio * 100);
		//int j, k;
		//for (j = 0; j < outSize; j++){
		//  int tmp = arr_index_test->ArrLabelIndexPtr[i].LabelIndexPtr[j].len;
		//  printf("test : Label %d %d:%d - \n", i, j, tmp);
		//  for (k = 0; k < tmp; k++){
		//      printf("%d ",
		//          arr_index_test->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData[k]);
		//  }
		//  printf("\n");
		//}
	}
	for (i = 0; i < outSize; i++) {
		//printf("test : Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
		//  i, cnn->N_TP[i], cnn->N_TN[i], cnn->N_FP[i], cnn->N_FN[i],
		//  cnn->N_sum[i], cnn->N_wrong[i], cnn->e_sum[i],
		//  cnn->N_wrong[i] / cnn->N_sum[i], cnn->e_sum[i] / cnn->N_sum[i]);

		fitness[i] = cnn->N_wrong[i] / cnn->N_sum[i];
	}

	return;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//
//
int NDIM_LeNet_ENSEMBLE;
void Fitness_ensemble_initialize(ImgArr* inputData, LabelArr* outputData, int arrSize, MY_FLT_TYPE* all_y);
void Fitness_ensemble(ImgArr* inputData, LabelArr* outputData, int arrSize,
	MY_FLT_TYPE* all_y, double* indivWeight, double* fitness);

void load_storedCNN(char* pro)
{
	num_storedCNN = 120;
	storedCNN = (double*)calloc(DIM_LeNet * num_storedCNN, sizeof(double));

	int outSize = testLabel[0]->LabelPtr[0].l;
	NDIM_LeNet_ENSEMBLE = num_storedCNN * outSize;

	char filename[1024];
	sprintf(filename, "PS/DPCCMOLSEA_VAR_%s_OBJ%d_VAR%d_RUN1", pro, OBJ_LeNet, DIM_LeNet);
	FILE* fpt = fopen(filename, "r");
	if (!fpt) {
		printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, filename);
		exit(999);
	}
	//int i;
	int tmp;
	double elem;
	for (int i = 0; i < num_storedCNN * DIM_LeNet; i++) {
		tmp = fscanf(fpt, "%lf", &elem);
		if (tmp == EOF) {
			printf("\n%s(%d): storedCNNs are not enough...\n", __FILE__, __LINE__);
			exit(99);
		}
		storedCNN[i] = elem;
	}
	fclose(fpt);

	int count = 0;
	for (int i = 0; i < len_TrainArr; i++)
		count += trainImg[i]->ImgNum;
	all_y_train = (MY_FLT_TYPE*)calloc(num_storedCNN * count * outSize, sizeof(MY_FLT_TYPE));
	count = 0;
	for (int i = 0; i < len_TestArr; i++)
		count += testImg[i]->ImgNum;
	all_y_test = (MY_FLT_TYPE*)calloc(num_storedCNN * count * outSize, sizeof(MY_FLT_TYPE));

	Fitness_ensemble_initialize(trainImg, trainLabel, len_TrainArr, all_y_train);
	Fitness_ensemble_initialize(testImg, testLabel, len_TestArr, all_y_test);
}

void Finalize_LeNet_ensemble()
{
	//
	freeSetArrLabelIndexPtr(arr_index_train);
	freeSetArrLabelIndexPtr(arr_index_test);
	int i;
	for (i = 0; i < len_TrainArr; i++) {
		free_Img(trainImg[i]);
		free_Label(trainLabel[i]);
	}
	free(trainImg);
	free(trainLabel);
	for (i = 0; i < len_TestArr; i++) {
		free_Img(testImg[i]);
		free_Label(testLabel[i]);
	}
	free(testImg);
	free(testLabel);

	//
	cnnfree(cnn);

	free(storedCNN);
	free(all_y_train);
	free(all_y_test);

	return;
}

void SetLimits_LeNet_ensemble(double* minLimit, double* maxLimit, int nx)
{
	int k;
	for (k = 0; k < NDIM_LeNet_ENSEMBLE; k++) {
		minLimit[k] = 0.0;
		maxLimit[k] = 1.0;
	}
}

int CheckLimits_LeNet_ensemble(double* x, int nx)
{
	int k;
	for (k = 0; k < NDIM_LeNet_ENSEMBLE; k++) {
		if (x[k] < 0.0 || x[k] > 1.0) {
			printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], 0.0, 1.0);
			return false;
		}
	}
	return true;
}

void Fitness_ensemble_initialize(ImgArr* inputData, LabelArr* outputData, int arrSize, MY_FLT_TYPE* all_y)
{
	int outSize = outputData[0]->LabelPtr[0].l;

	//int i;
	int count = 0;
	for (int i = 0; i < arrSize; i++) {
		count += inputData[i]->ImgNum; //10000;
	}
	int num_all_sample = count;

	int mpi_size;
	int mpi_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int* each_size = (int*)calloc(mpi_size, sizeof(int));
	int* recv_size = (int*)calloc(mpi_size, sizeof(int));
	int* disp_size = (int*)calloc(mpi_size, sizeof(int));
	int quo = num_storedCNN / mpi_size;
	int rem = num_storedCNN % mpi_size;
	for (int i = 0; i < mpi_size; i++) {
		each_size[i] = quo;
		if (i < rem) each_size[i]++;
	}
	for (int i = 0; i < mpi_size; i++) {
		recv_size[i] = each_size[i];
		if (i == 0)
			disp_size[i] = 0;
		else
			disp_size[i] = disp_size[i - 1] + recv_size[i - 1];
	}
	MY_FLT_TYPE* tmp_y = (MY_FLT_TYPE*)calloc(num_storedCNN * num_all_sample * outSize, sizeof(MY_FLT_TYPE));
	MY_FLT_TYPE tmp_sum;

	int iIndiv;
	for (iIndiv = disp_size[mpi_rank];
		iIndiv < disp_size[mpi_rank] + recv_size[mpi_rank];
		iIndiv++) {
		cnninit(cnn, &storedCNN[iIndiv * DIM_LeNet], ASSIGN_MODE);
		count = 0;
		int n;
		for (int i = 0; i < arrSize; i++) {
			for (n = 0; n < inputData[i]->ImgNum; n++) {
				cnnff(cnn, inputData[i]->ImgPtr[n].ImgData, cnn->core_mode);
				int m;
				tmp_sum = 0;
				for (m = 0; m < outSize; m++) {
					tmp_sum += cnn->O5->y[m];
				}
				for (m = 0; m < outSize; m++) {
					tmp_y[(iIndiv - disp_size[mpi_rank]) * num_all_sample * outSize + (count + n) * outSize + m] = cnn->O5->y[m] / tmp_sum;
				}
				cnnclear(cnn);
			}
			count += inputData[i]->ImgNum;
		}
	}
	//char filename[128];
	//FILE* fpt;
	//sprintf(filename, "tmp_y_rank_%d", mpi_rank);
	//fpt = fopen(filename, "w");
	//if (!fpt){
	//  printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, filename);
	//  exit(999);
	//}
	//fprintf(fpt, "%d\n", recv_size[mpi_rank]);
	//for (iIndiv = 0; iIndiv < recv_size[mpi_rank]; iIndiv++){
	//  for (i = 0; i < num_all_sample; i++){
	//      for (int j = 0; j < outSize; j++){
	//          fprintf(fpt, "%e ", tmp_y[iIndiv*num_all_sample*outSize + i*outSize + j]);
	//      }
	//      fprintf(fpt, "||| ");
	//  }
	//  fprintf(fpt, "\n");
	//}
	//fclose(fpt);

	for (int i = 0; i < mpi_size; i++) {
		recv_size[i] = each_size[i] * num_all_sample * outSize;
		if (i == 0)
			disp_size[i] = 0;
		else
			disp_size[i] = disp_size[i - 1] + recv_size[i - 1];
	}
	MPI_Allgatherv(tmp_y, recv_size[mpi_rank], MPI_FLOAT,
		all_y, recv_size, disp_size, MPI_FLOAT, MPI_COMM_WORLD);

	//sprintf(filename, "all_y_rank_%d", mpi_rank);
	//fpt = fopen(filename, "w");
	//if (!fpt){
	//  printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, filename);
	//  exit(999);
	//}
	//fprintf(fpt, "%d\n", num_storedCNN);
	//for (iIndiv = 0; iIndiv < num_storedCNN; iIndiv++){
	//  for (i = 0; i < num_all_sample; i++){
	//      for (int j = 0; j < outSize; j++){
	//          fprintf(fpt, "%e ", all_y[iIndiv*num_all_sample*outSize + i*outSize + j]);
	//      }
	//      fprintf(fpt, "||| ");
	//  }
	//  fprintf(fpt, "\n");
	//}
	//fclose(fpt);

	free(each_size);
	free(recv_size);
	free(disp_size);

	free(tmp_y);

	return;
}

void Fitness_ensemble(ImgArr* inputData, LabelArr* outputData, int arrSize,
	MY_FLT_TYPE* all_y, double* indivWeight, double* fitness)
{
	int outSize = outputData[0]->LabelPtr[0].l;
	int i;
	for (i = 0; i < outSize; i++) {
		cnn->N_sum[i] = 0.0;
		cnn->N_wrong[i] = 0.0;
		cnn->e_sum[i] = 0.0;
		cnn->N_TP[i] = 0.0;
		cnn->N_TN[i] = 0.0;
		cnn->N_FP[i] = 0.0;
		cnn->N_FN[i] = 0.0;
	}
	int count = 0;
	for (i = 0; i < arrSize; i++) {
		count += inputData[i]->ImgNum; //10000;
	}
	int num_all_sample = count;
	MY_FLT_TYPE* labelling_results = (MY_FLT_TYPE*)calloc(count * outSize, sizeof(MY_FLT_TYPE));

	int iIndiv;
	for (iIndiv = 0; iIndiv < num_storedCNN; iIndiv++) {
		count = 0;
		int n;
		for (i = 0; i < arrSize; i++) {
			for (n = 0; n < inputData[i]->ImgNum; n++) {
				int m;
				for (m = 0; m < outSize; m++) {
					labelling_results[(count + n) * outSize + m] +=
						all_y[iIndiv * num_all_sample * outSize + (count + n) * outSize + m] *
						(MY_FLT_TYPE)indivWeight[m * num_storedCNN + iIndiv];
				}
			}
			count += inputData[i]->ImgNum;
		}
	}

	count = 0;
	int max_ind, max_ind_t;
	double max_val;
	for (i = 0; i < arrSize; i++) {
		int n;
		int m;
		for (n = 0; n < inputData[i]->ImgNum; n++) {
			max_ind = 0;
			max_val = labelling_results[(count + n) * outSize + 0];
			for (m = 1; m < outSize; m++) {
				if (max_val < labelling_results[(count + n) * outSize + m]) {
					max_val = labelling_results[(count + n) * outSize + m];
					max_ind = m;
				}
			}
			max_ind_t = vecmaxIndex(outputData[i]->LabelPtr[n].LabelData, outSize);

			int a;
			for (a = 0; a < outSize; a++) {
				cnn->e_sum[a] += fabs(cnn->e[a]);
				cnn->N_sum[a] += outputData[i]->LabelPtr[n].LabelData[a];
			}
			if (max_ind != max_ind_t) {
				cnn->N_wrong[max_ind_t]++;
			}
			for (a = 0; a < cnn->O5->outputNum; a++) {
				if (a == max_ind && a == max_ind_t) cnn->N_TP[a]++;
				if (a == max_ind && a != max_ind_t) cnn->N_FP[a]++;
				if (a != max_ind && a == max_ind_t) cnn->N_FN[a]++;
				if (a != max_ind && a != max_ind_t) cnn->N_TN[a]++;
			}
		}

		count += inputData[i]->ImgNum;
	}

	free(labelling_results);

	for (i = 0; i < outSize; i++) {
		//printf("test : Label %d: TP - %lf TN - %lf FP - %lf FN - %lf \nN_sum - %lf N_wrong - %lf e_sum - %lf \nN - %lf, e - %lf\n",
		//  i, cnn->N_TP[i], cnn->N_TN[i], cnn->N_FP[i], cnn->N_FN[i],
		//  cnn->N_sum[i], cnn->N_wrong[i], cnn->e_sum[i],
		//  cnn->N_wrong[i] / cnn->N_sum[i], cnn->e_sum[i] / cnn->N_sum[i]);

		fitness[i] = cnn->N_wrong[i] / cnn->N_sum[i];
	}

	return;
}

void Fitness_LeNet_ensemble(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	Fitness_ensemble(trainImg, trainLabel, len_TrainArr, all_y_train, individual, fitness);

	return;
}

void Fitness_LeNet_test_ensemble(double* indivWeight, double* fitness)
{
	Fitness_ensemble(testImg, testLabel, len_TestArr, all_y_test, indivWeight, fitness);

	return;
}