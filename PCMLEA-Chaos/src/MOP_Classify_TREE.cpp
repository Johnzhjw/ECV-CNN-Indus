#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "MOP_Classify_TREE.h"

//#include <tchar.h>

//macro - opType
#define m_BOOTSTRAP     0 // .632 bootstrap
#define m_BOOTSTRAPplus 1 // .632+ bootstrap
#define m_MULTIFOLD     2 // 多折交叉
#define m_RATIO         3 // 比例划分

#define m_MAX_BUF_SIZE  1000000 //

//
int    m_N_CLASS;
int    m_N_MODEL;
int    m_N_FEATURE;
int    m_DIM_ClassifierTreeFunc;
int    m_opNum;
int    m_opNumCur;
int    v_labelCLASS[256];//每个类别的标识
double v_ratioLabelCLASS[256];
int    v_arrSize[256];

//global variables in this cpp file
double** mx_wholeData;
int      m_M_whole_data;
int      m_N_whole_data;
int      m_N_sample_whole;

double** mx_optimizeData;
int** mx_optimizeIdx;
int      m_N_sample_optimize;

double** mx_trainData;
int** mx_trainIdx;
int      m_N_sample_train;

double** mx_validationData;
int** mx_validationIdx;
int      m_N_sample_validation;

double** mx_testData;
int** mx_testIdx;
int      m_N_sample_test;

double** mx_weights;
int      m_N_sample_filter;
double** mx_filterData;
double** mx_filterDataMinMax;

int** mx_selectionFlag;

int      m_innerTag = 0;
int      m_filterTag = 0;

int      m_class_mpi_rank;
int      m_class_mpi_size;

//stdout = freopen("out.txt", "w", stdout);

//typedef  struct BiTNode
//{
//  int* featureSubset = NULL;
//  int* lLabelSet = NULL;
//  int* rLabelSet = NULL;
//  int  N_LABEL_left = 0;
//  int  N_LABEL_right = 0;
//  int  numFeature = 0;
//  int  model = -1;
//  int  label = -1;
//  int  level = -1;
//  struct BiTNode *parent = NULL, *lchild = NULL, *rchild = NULL;
//}BiTNode, *BiTree;

const char testInstNames[24][128] = {
	"ALLGSE412_poterapiji_TREE",
	"ALLGSE412_pred_poTh_TREE",
	"AMLGSE2191_TREE",
	"BC_CCGSE3726_frozen_TREE",
	"BCGSE349_350_TREE",
	"bladderGSE89_TREE",
	"braintumor_TREE",
	"CMLGSE2535_TREE",
	"DLBCL_TREE",
	"EWSGSE967_TREE",
	"EWSGSE967_3class_TREE",
	"gastricGSE2685_TREE",
	"gastricGSE2685_2razreda_TREE",
	"glioblastoma_TREE",
	"leukemia_TREE",
	"LL_GSE1577_TREE",
	"LL_GSE1577_2razreda_TREE",
	"lung_TREE",
	"lungGSE1987_TREE",
	"meduloblastomiGSE468_TREE",
	"MLL_TREE",
	"prostata_TREE",
	"prostateGSE2443_TREE",
	"SRBCT_TREE"
};

const int numROW[24] = {
	60, 110, 54,  52, 24,  40,
	40,  28, 77,  23, 23,  30,
	30,  50, 72,  29, 19, 203,
	34,  23, 72, 102, 20,  83
};

const int numCOL[24] = {
	8281,  8281, 12626, 22284,
	12626,  5725,  7130, 12626,
	7071,  9946,  9946,  4523,
	4523, 12626,  5148, 15435,
	15435, 12601, 10542,  1466,
	12534, 12534, 12628,  2309
};

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define M_IM1_CLASS 2147483563
#define M_IM2_CLASS 2147483399
#define M_AM_CLASS (1.0/M_IM1_CLASS)
#define M_IMM1_CLASS (M_IM1_CLASS-1)
#define M_IA1_CLASS 40014
#define M_IA2_CLASS 40692
#define M_IQ1_CLASS 53668
#define M_IQ2_CLASS 52774
#define M_IR1_CLASS 12211
#define M_IR2_CLASS 3791
#define M_NTAB_CLASS 32
#define M_NDIV_CLASS (1+M_IMM1_CLASS/M_NTAB_CLASS)
#define M_EPS_CLASS 1.2e-7
#define M_RNMX_CLASS (1.0-M_EPS_CLASS)
double f_rnd_uni_CLASS(long* idum);
//the random generator in [0,1)
double f_rnd_uni_CLASS(long* idum)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[M_NTAB_CLASS];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = M_NTAB_CLASS + 7; j >= 0; j--) {
			k = (*idum) / M_IQ1_CLASS;
			*idum = M_IA1_CLASS * (*idum - k * M_IQ1_CLASS) - k * M_IR1_CLASS;
			if (*idum < 0) *idum += M_IM1_CLASS;
			if (j < M_NTAB_CLASS) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / M_IQ1_CLASS;
	*idum = M_IA1_CLASS * (*idum - k * M_IQ1_CLASS) - k * M_IR1_CLASS;
	if (*idum < 0) *idum += M_IM1_CLASS;
	k = idum2 / M_IQ2_CLASS;
	idum2 = M_IA2_CLASS * (idum2 - k * M_IQ2_CLASS) - k * M_IR2_CLASS;
	if (idum2 < 0) idum2 += M_IM2_CLASS;
	j = iy / M_NDIV_CLASS;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += M_IMM1_CLASS;
	if ((temp = M_AM_CLASS * iy) > M_RNMX_CLASS) return M_RNMX_CLASS;
	else return temp;
}/*------End of f_rnd_uni_CLASS()--------------------------*/
int     m_seed_CLASS = 237;
long    m_rnd_uni_init_CLASS;

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
void f_getSize(char filename[], int& m, int& n);
double** f_allocDouble(int m, int n);
int** f_allocInt(int m, int n);
void f_readData(double*& MEM, char filename[], int& m, int& n);

void f_splitData_outer(int opType, double ratio);
void f_splitData_inner(int opType, int m_opNum, int m_opNumCur, double ratio);
void f_LOOCV_split(int* arrIDX, int nFeature, int opNum, int opNumCur);
void f_cascadedClassifierMineBT(BiTree T, int& nCorrectValidat, int& nNumValidat);
void f_cascadedClassifierMineBT2(BiTree T, int& nCorrectValidat, int& nNumValidat);
//void f_Initialize_ClassifierTreeFunc(char prob[], int curN, int numN);
//void f_SetLimits_ClassifierTreeFunc(double* minLimit, double* maxLimit, int nx);
//BiTree f_treeDecoding(int* codedTree);
int  f_featureAssign(BiTree& T, int* codedFeature);
int  f_labelAssign(BiTree& T);
void f_printBiTree(BiTree T);
//int  f_freeBiTree(BiTree &T);
//void f_Fitness_ClassifierTreeFunc(double* individual, double* fitness);
//int  f_checkLimits_ClassifierTreeFunc(double* x, int nx);
void f_freeMemoryTreeCLASS();
//void f_filter_ReliefF();
//void f_testAccuracy(double* individual, double* fitness);
void f_splitData_test(int opType, int opNum, int opNumCur, double ratio);
void f_bootstrapInitialize(int**& index);

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void f_getSize(char filename[], int& m, int& n)
{
	FILE* fpt = fopen(filename, "r");

	if (fpt) {
		char buf[m_MAX_BUF_SIZE];
		char* p;
		m = 0;
		n = 0;

		while (fgets(buf, m_MAX_BUF_SIZE, fpt)) {
			if (n == 0) {
				for (p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
					n++;
				}
			}
			m++;
		}
		// fgets(buf, m_MAX_BUF_SIZE, fpt);   //读取文件中的一行到buf中
		// for (p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
		//     n++;
		// }
		// while (!feof(fpt)) {
		//     m++;//置于fgets之前
		//     fgets(buf, m_MAX_BUF_SIZE, fpt);   //读取文件中的一行到buf中
		// }
		fclose(fpt);
	}
	else {
		printf("Open file %s error, exiting...\n", filename);
		exit(-1);
	}

	return;
}

double** f_allocDouble(int m, int n)
{
	double** tmp;
	if ((tmp = (double**)calloc(m, sizeof(double*))) == NULL) {
		printf("ERROR!! --> calloc: no memory for matrix\n");
		exit(1);
	}
	for (int i = 0; i < m; i++) {
		if ((tmp[i] = (double*)calloc(n, sizeof(double))) == NULL) {
			printf("ERROR!! --> calloc: no memory for vector\n");
			exit(1);
		}
	}
	return tmp;
}

int** f_allocInt(int m, int n)
{
	int** tmp;
	if ((tmp = (int**)calloc(m, sizeof(int*))) == NULL) {
		printf("ERROR!! --> calloc: no memory for matrix\n");
		exit(1);
	}
	for (int i = 0; i < m; i++) {
		if ((tmp[i] = (int*)calloc(n, sizeof(int))) == NULL) {
			printf("ERROR!! --> calloc: no memory for vector\n");
			exit(1);
		}
	}
	return tmp;
}

void f_readData(double**& MEM, char filename[], int& m, int& n)
{
	//printf("%s\n", filename);
	f_getSize(filename, m, n);
	MEM = f_allocDouble(m, n);
	FILE* fpt;
	fpt = fopen(filename, "r");
	if (!fpt) {
		printf("Load .csv file failed");
		exit(-1);
	}
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (fscanf(fpt, "%lf", &MEM[i][j]) != 1) {
				printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
				exit(-1);
			}
		}
	}
	fclose(fpt);
	//for (i = 0; i < m; i++) {
	//  for (j = 0; j < n && j < 9; j++) {
	//      printf("%lf ", MEM[i][j]);
	//  }
	//  printf("\n");
	//}

	return;
}

//void saveMat(Mat inputMat, char* filename)
//{
//  FILE* fpt = fopen(filename, "w");
//  for (int i = 0; i < inputMat.rows; i++) {
//      for (int j = 0; j < inputMat.cols; j++) {
//          if (j < inputMat.cols - 1)
//              fprintf(fpt, "%f,", inputMat.at<float>(i, j));
//          else
//              fprintf(fpt, "%f\n", inputMat.at<float>(i, j));
//      }
//  }
//  fclose(fpt);
//}

//
//  mx_wholeData;
//  mx_optimizeData;
//  mx_testData;
// int m_opNum, int m_opNumCur --- 外层测试次数 及 外层测试当前次数 初始化随机数
void f_splitData_outer(int opType, double ratio)
{
	//free
	//for (int i = 0; i < m_N_sample_optimize; i++){
	//  if (mx_optimizeData[i]) free(mx_optimizeData[i]);
	//  if (mx_optimizeIdx[i]) free(mx_optimizeIdx[i]);
	//}
	//if (mx_optimizeData) free(mx_optimizeData);
	//if (mx_optimizeIdx) free(mx_optimizeIdx);
	//for (int i = 0; i < m_N_sample_test; i++){
	//  if (mx_testData[i]) free(mx_testData[i]);
	//  if (mx_testIdx[i]) free(mx_testIdx[i]);
	//}
	//if (mx_testData) free(mx_testData);
	//if (mx_testIdx) free(mx_testIdx);

	//m_seed_CLASS = (m_seed_CLASS + 111) % 1235;
	//m_rnd_uni_init_CLASS = -(long)m_seed_CLASS;
	//f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS)

	int count = 0;
	int count2 = 0;
	int testSize;
	int optimizeSize;
	int ccc1;
	int ccc2;
	switch (opType) {
	case m_BOOTSTRAP:
		//用于优化过程的数据
		m_N_sample_optimize = m_M_whole_data;
		mx_optimizeData = f_allocDouble(m_N_sample_optimize, m_N_whole_data);
		mx_optimizeIdx = f_allocInt(m_N_sample_optimize, 1);
		f_bootstrapInitialize(mx_optimizeIdx);
		//for (int i = 0; i < m_N_sample_optimize; i++) {
		//  mx_optimizeIdx[i][0] = f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS) * m_N_sample_optimize;
		//  //printf("%d ", mx_optimizeIdx[i][0]);
		//}
		//for (int j = 0; j < m_M_whole_data; j++){
		//  printf("%2d ", mx_optimizeIdx[j][0]);
		//}
		//printf("\n");
		//记录使用的样本序号
		mx_selectionFlag = f_allocInt(m_N_sample_optimize, 1);
		for (int i = 0; i < m_N_sample_optimize; i++)
			mx_selectionFlag[i][0] = 0;
		count = m_N_sample_optimize;
		//有放回地随机选择
		for (int i = 0; i < m_M_whole_data; i++) {
			memcpy(mx_optimizeData[i], mx_wholeData[mx_optimizeIdx[i][0]], m_N_whole_data * sizeof(double));
			if (0 == mx_selectionFlag[mx_optimizeIdx[i][0]][0]) {
				count--;
				mx_selectionFlag[mx_optimizeIdx[i][0]][0] = 1;
			}
		}
		//未被选择的作为测试数据
		m_N_sample_test = count;
		mx_testData = f_allocDouble(m_N_sample_test, m_N_whole_data);
		mx_testIdx = f_allocInt(m_N_sample_test, 1);
		count = 0;
		for (int i = 0; i < m_M_whole_data; i++) {
			if (0 == mx_selectionFlag[i][0]) {
				memcpy(mx_testData[count], mx_wholeData[i], m_N_whole_data * sizeof(double));
				mx_testIdx[count][0] = i;
				count++;
			}
		}
		for (int i = 0; i < m_N_sample_optimize; i++) free(mx_selectionFlag[i]);
		free(mx_selectionFlag);
		break;

	case m_BOOTSTRAPplus:
		printf("Unavailable currently, EXITING...\n");
		exit(-77);
		break;

	case m_MULTIFOLD:
		//折数为m_opNum
		if (m_opNumCur < 0) {
			printf("m_opNumCur should not be negative, exiting!!!\n");
			exit(-111);
		}
		if (m_opNumCur >= m_opNum) {
			printf("m_opNumCur %d should be less than m_opNum %d\n", m_opNumCur, m_opNum);
			exit(-111);
		}
		//计算用于测试的那一折的样本数
		ccc1 = (int)(((m_opNumCur + 1.0) / m_opNum) * m_M_whole_data);
		if (ccc1 > m_M_whole_data) ccc1 = m_M_whole_data;
		ccc2 = (int)(((m_opNumCur + 0.0) / m_opNum) * m_M_whole_data);
		testSize = ccc1 - ccc2;
		//用于优化的数据初始化
		m_N_sample_optimize = m_M_whole_data - testSize;
		mx_optimizeData = f_allocDouble(m_N_sample_optimize, m_N_whole_data);
		mx_optimizeIdx = f_allocInt(m_N_sample_optimize, 1);
		m_N_sample_test = testSize;
		mx_testData = f_allocDouble(m_N_sample_test, m_N_whole_data);
		mx_testIdx = f_allocInt(m_N_sample_test, 1);
		//选取样本
		count = 0;
		for (int i = 0; i < ccc2; i++) {
			memcpy(mx_optimizeData[count], mx_wholeData[i], m_N_whole_data * sizeof(double));
			mx_optimizeIdx[count][0] = i;
			count++;
		}
		count2 = 0;
		for (int i = ccc2; i < ccc1; i++) {
			memcpy(mx_testData[count2], mx_wholeData[i], m_N_whole_data * sizeof(double));
			mx_testIdx[count2][0] = i;
			count2++;
		}
		for (int i = ccc1; i < m_M_whole_data; i++) {
			memcpy(mx_optimizeData[count], mx_wholeData[i], m_N_whole_data * sizeof(double));
			mx_optimizeIdx[count][0] = i;
			count++;
		}
		break;

	case m_RATIO:
		// double ratio --- 用于优化的样本比例
		if (ratio <= 0.0 || ratio >= 1.0) {
			printf("ratio should be in (0.0,1.0), but not %lf\n", ratio);
			exit(-222);
		}
		//用于优化的样本数目
		optimizeSize = 0;
		for (int i = 0; i < m_N_CLASS; i++) {
			v_arrSize[i] = 0;
			for (int j = 0; j < m_M_whole_data; j++) {
				if ((int)mx_wholeData[j][m_N_FEATURE] == v_labelCLASS[i])
					v_arrSize[i]++;
			}
			int tmp = (int)(v_arrSize[i] * ratio);
			if (tmp == 0) tmp = 1;
			optimizeSize += tmp;
		}
		//初始化
		m_N_sample_optimize = optimizeSize;
		mx_optimizeData = f_allocDouble(m_N_sample_optimize, m_N_whole_data);
		mx_optimizeIdx = f_allocInt(m_N_sample_optimize, 1);
		m_N_sample_test = m_M_whole_data - optimizeSize;
		mx_testData = f_allocDouble(m_N_sample_test, m_N_whole_data);
		mx_testIdx = f_allocInt(m_N_sample_test, 1);
		//记录使用的样本序号
		mx_selectionFlag = f_allocInt(m_N_sample_whole, 1);
		for (int i = 0; i < m_N_sample_whole; i++)
			mx_selectionFlag[i][0] = 0;
		//选取样本
		for (int p = 0; p < m_N_CLASS; p++) {
			count = 0;
			int tmp = (int)(v_arrSize[p] * ratio);
			if (tmp == 0) tmp = 1;
			while (count < tmp) {
				int idx = (int)(f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS) * m_N_sample_whole);
				if ((int)mx_wholeData[idx][m_N_FEATURE] == v_labelCLASS[p] && mx_selectionFlag[idx][0] == 0) {
					memcpy(mx_optimizeData[count], mx_wholeData[idx], m_N_whole_data * sizeof(double));
					mx_optimizeIdx[count][0] = idx;
					count++;
					mx_selectionFlag[idx][0] = 1;
				}
			}
		}
		count = 0;
		for (int i = 0; i < m_M_whole_data; i++) {
			if (mx_selectionFlag[i][0] == 0) {
				memcpy(mx_testData[count], mx_wholeData[i], m_N_whole_data * sizeof(double));
				mx_testIdx[count][0] = i;
				count++;
			}
		}
		//for (int i = 0; i < m_N_sample_test; i++){
		//  for (int j = 0; j < m_N_whole_data; j++){
		//      printf("%lf ", mx_testData[i][j]);
		//  }
		//  printf("\n");
		//}
		for (int i = 0; i < m_N_sample_whole; i++) free(mx_selectionFlag[i]);
		free(mx_selectionFlag);
		break;

	default:
		printf("INVALID OP TYPE, EXITING...\n");
		exit(-77);
		break;
	}

	return;
}

//
//  mx_optimizeData;
//  mx_trainData;
//  mx_validationData;
void f_splitData_inner(int opType, int mem_opNum, int mem_opNumCur, double ratio)
{
	//free
	if (m_innerTag) {
		for (int i = 0; i < m_N_sample_train; i++) {
			free(mx_trainData[i]);
			free(mx_trainIdx[i]);
		}
		free(mx_trainData);
		free(mx_trainIdx);
		for (int i = 0; i < m_N_sample_validation; i++) {
			free(mx_validationData[i]);
			free(mx_validationIdx[i]);
		}
		free(mx_validationData);
		free(mx_validationIdx);

		m_innerTag = 0;
	}

	//m_seed_CLASS = (m_seed_CLASS + m_opNumCur + 100) % 1377;
	//m_rnd_uni_init_CLASS = -(long)m_seed_CLASS;
	//f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS)

	int count = 0;
	int count2 = 0;
	int validationSize;
	int trainSize;
	int ccc1;
	int ccc2;
	switch (opType) {
	case m_BOOTSTRAP:
		//用于train的数据
		m_N_sample_train = m_N_sample_optimize;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		for (int i = 0; i < m_N_sample_train; i++) {
			mx_trainIdx[i][0] = (int)(f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS) * m_N_sample_train);
			//printf("%d ", mx_trainIdx[i][0]);
		}
		//记录使用的样本序号
		mx_selectionFlag = f_allocInt(m_N_sample_train, 1);
		for (int i = 0; i < m_N_sample_train; i++)
			mx_selectionFlag[i][0] = 0;
		count = m_N_sample_train;
		//有放回地随机选择
		for (int i = 0; i < m_N_sample_optimize; i++) {
			memcpy(mx_trainData[i], mx_optimizeData[mx_trainIdx[i][0]], m_N_whole_data * sizeof(double));
			if (0 == mx_selectionFlag[mx_trainIdx[i][0]][0]) {
				count--;
				mx_selectionFlag[mx_trainIdx[i][0]][0] = 1;
			}
		}
		//未被选择的作为validation数据
		m_N_sample_validation = count;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		count = 0;
		for (int i = 0; i < m_N_sample_optimize; i++) {
			if (0 == mx_selectionFlag[i][0]) {
				memcpy(mx_validationData[count], mx_optimizeData[i], m_N_whole_data * sizeof(double));
				mx_validationIdx[count][0] = i;
				count++;
			}
		}
		for (int i = 0; i < m_N_sample_train; i++) free(mx_selectionFlag[i]);
		free(mx_selectionFlag);
		break;

	case m_BOOTSTRAPplus:
		printf("Unavailable currently, EXITING...\n");
		exit(-77);
		break;

	case m_MULTIFOLD:
		//折数为m_opNum
		if (mem_opNumCur < 0) {
			printf("m_opNumCur should not be negative, exiting!!!\n");
			exit(-111);
		}
		if (mem_opNumCur >= mem_opNum) {
			printf("m_opNumCur %d should be less than m_opNum %d\n", mem_opNumCur, mem_opNum);
			exit(-111);
		}
		//计算用于测试的那一折的样本数
		ccc1 = (int)(((mem_opNumCur + 1.0) / mem_opNum) * m_N_sample_optimize);
		if (ccc1 > m_N_sample_optimize) ccc1 = m_N_sample_optimize;
		ccc2 = (int)(((mem_opNumCur + 0.0) / mem_opNum) * m_N_sample_optimize);
		validationSize = ccc1 - ccc2;
		//用于优化的数据初始化
		m_N_sample_train = m_N_sample_optimize - validationSize;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		m_N_sample_validation = validationSize;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		//选取样本
		count = 0;
		for (int i = 0; i < ccc2; i++) {
			memcpy(mx_trainData[count], mx_optimizeData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[count][0] = mx_optimizeIdx[i][0];
			count++;
		}
		count2 = 0;
		for (int i = ccc2; i < ccc1; i++) {
			memcpy(mx_validationData[count2], mx_optimizeData[i], m_N_whole_data * sizeof(double));
			mx_validationIdx[count2][0] = mx_optimizeIdx[i][0];
			count2++;
		}
		for (int i = ccc1; i < m_N_sample_optimize; i++) {
			memcpy(mx_trainData[count], mx_optimizeData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[count][0] = mx_optimizeIdx[i][0];
			count++;
		}
		break;

	case m_RATIO:
		// double ratio --- 用于优化的样本比例
		if (ratio <= 0.0 || ratio >= 1.0) {
			printf("ratio should be in (0.0,1.0), but not %lf\n", ratio);
			exit(-222);
		}
		//用于优化的样本数目
		trainSize = (int)(m_N_sample_optimize * ratio);
		//初始化
		m_N_sample_train = trainSize;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		m_N_sample_validation = m_N_sample_optimize - m_N_sample_train;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		//选取样本
		for (int i = 0; i < m_N_sample_train; i++) {
			memcpy(mx_trainData[i], mx_optimizeData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[i][0] = mx_optimizeIdx[i][0];
		}
		count = 0;
		for (int i = m_N_sample_train; i < m_N_sample_optimize; i++) {
			memcpy(mx_validationData[count], mx_optimizeData[i], m_N_whole_data * sizeof(double));
			mx_validationIdx[count][0] = i;
			count++;
		}
		break;

	default:
		printf("INVALID OP TYPE, EXITING...\n");
		exit(-77);
		break;
	}

	m_innerTag = 1;

	return;
}

//
//  mx_testData;
//  mx_trainData;
//  mx_validationData;
void f_splitData_test(int opType, int mem_opNum, int mem_opNumCur, double ratio)
{
	//free
	if (m_innerTag) {
		for (int i = 0; i < m_N_sample_train; i++) {
			free(mx_trainData[i]);
			free(mx_trainIdx[i]);
		}
		free(mx_trainData);
		free(mx_trainIdx);
		for (int i = 0; i < m_N_sample_validation; i++) {
			free(mx_validationData[i]);
			free(mx_validationIdx[i]);
		}
		free(mx_validationData);
		free(mx_validationIdx);

		m_innerTag = 0;
	}

	//m_seed_CLASS = (m_seed_CLASS + m_opNumCur + 100) % 1377;
	//m_rnd_uni_init_CLASS = -(long)m_seed_CLASS;
	//f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS)

	int count = 0;
	int count2 = 0;
	int validationSize;
	int trainSize;
	int ccc1;
	int ccc2;
	switch (opType) {
	case m_BOOTSTRAP:
		//用于train的数据
		m_N_sample_train = m_N_sample_test;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		for (int i = 0; i < m_N_sample_train; i++) {
			mx_trainIdx[i][0] = (int)(f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS) * m_N_sample_train);
			//printf("%d ", mx_trainIdx[i][0]);
		}
		//记录使用的样本序号
		mx_selectionFlag = f_allocInt(m_N_sample_train, 1);
		for (int i = 0; i < m_N_sample_train; i++)
			mx_selectionFlag[i][0] = 0;
		count = m_N_sample_train;
		//有放回地随机选择
		for (int i = 0; i < m_N_sample_test; i++) {
			memcpy(mx_trainData[i], mx_testData[mx_trainIdx[i][0]], m_N_whole_data * sizeof(double));
			if (0 == mx_selectionFlag[mx_trainIdx[i][0]][0]) {
				count--;
				mx_selectionFlag[mx_trainIdx[i][0]][0] = 1;
			}
		}
		//未被选择的作为validation数据
		m_N_sample_validation = count;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		count = 0;
		for (int i = 0; i < m_N_sample_test; i++) {
			if (0 == mx_selectionFlag[i][0]) {
				memcpy(mx_validationData[count], mx_testData[i], m_N_whole_data * sizeof(double));
				mx_validationIdx[count][0] = i;
				count++;
			}
		}
		for (int i = 0; i < m_N_sample_train; i++) free(mx_selectionFlag[i]);
		free(mx_selectionFlag);
		break;

	case m_BOOTSTRAPplus:
		printf("Unavailable currently, EXITING...\n");
		exit(-77);
		break;

	case m_MULTIFOLD:
		//折数为m_opNum
		if (mem_opNumCur < 0) {
			printf("m_opNumCur should not be negative, exiting!!!\n");
			exit(-111);
		}
		if (mem_opNumCur >= mem_opNum) {
			printf("m_opNumCur %d should be less than m_opNum %d\n", mem_opNumCur, mem_opNum);
			exit(-111);
		}
		//计算用于测试的那一折的样本数
		ccc1 = (int)(((mem_opNumCur + 1.0) / mem_opNum) * m_N_sample_test);
		if (ccc1 > m_N_sample_test) ccc1 = m_N_sample_test;
		ccc2 = (int)(((mem_opNumCur + 0.0) / mem_opNum) * m_N_sample_test);
		validationSize = ccc1 - ccc2;
		//用于优化的数据初始化
		m_N_sample_train = m_N_sample_test - validationSize;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		m_N_sample_validation = validationSize;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		//选取样本
		count = 0;
		for (int i = 0; i < ccc2; i++) {
			memcpy(mx_trainData[count], mx_testData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[count][0] = i;
			count++;
		}
		count2 = 0;
		for (int i = ccc2; i < ccc1; i++) {
			memcpy(mx_validationData[count2], mx_testData[i], m_N_whole_data * sizeof(double));
			mx_validationIdx[count2][0] = i;
			count2++;
		}
		for (int i = ccc1; i < m_N_sample_test; i++) {
			memcpy(mx_trainData[count], mx_testData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[count][0] = i;
			count++;
		}
		break;

	case m_RATIO:
		// double ratio --- 用于优化的样本比例
		if (ratio <= 0.0 || ratio >= 1.0) {
			printf("ratio should be in (0.0,1.0), but not %lf\n", ratio);
			exit(-222);
		}
		//用于优化的样本数目
		trainSize = (int)(m_N_sample_test * ratio);
		//初始化
		m_N_sample_train = trainSize;
		mx_trainData = f_allocDouble(m_N_sample_train, m_N_whole_data);
		mx_trainIdx = f_allocInt(m_N_sample_train, 1);
		m_N_sample_validation = m_N_sample_test - m_N_sample_train;
		mx_validationData = f_allocDouble(m_N_sample_validation, m_N_whole_data);
		mx_validationIdx = f_allocInt(m_N_sample_validation, 1);
		//选取样本
		for (int i = 0; i < m_N_sample_train; i++) {
			memcpy(mx_trainData[i], mx_testData[i], m_N_whole_data * sizeof(double));
			mx_trainIdx[i][0] = i;
		}
		count = 0;
		for (int i = m_N_sample_train; i < m_N_sample_test; i++) {
			memcpy(mx_validationData[count], mx_testData[i], m_N_whole_data * sizeof(double));
			mx_validationIdx[count][0] = i;
			count++;
		}
		break;

	default:
		printf("INVALID OP TYPE, EXITING...\n");
		exit(-77);
		break;
	}

	m_innerTag = 1;

	return;
}

void f_cascadedClassifierMineBT(BiTree T, int& nCorrectValidat, int& nNumValidat)
{
	nNumValidat = m_N_sample_validation;

	if (!T) return;

	int predictedLabels;

	//kNN - mine
	{
		const int K = 1;
		double dists[K];
		int    labels[K];
		nCorrectValidat = 0;
		for (int i = 0; i < m_N_sample_validation; i++) {
			//cout << "SMP_IND: " << i << endl;
			//Traverse the tree
			BiTree node = T;
			// int tag = 1;//1 - left, 2 - right
			while (1) {
				for (int j = 0; j < K; j++) {
					dists[j] = -1;
					labels[j] = -1;
				}
				if (node->label != -1) {
					predictedLabels = v_labelCLASS[node->label];
					break;
				}
				for (int j = 0; j < m_N_sample_train; j++) {
					int flag = 0;
					for (int k = 0; k < node->N_LABEL_left; k++) {
						if ((int)mx_trainData[j][m_N_whole_data - 1] == v_labelCLASS[node->lLabelSet[k]]) {
							flag = 1;
						}
					}
					for (int k = 0; k < node->N_LABEL_right; k++) {
						if ((int)mx_trainData[j][m_N_whole_data - 1] == v_labelCLASS[node->rLabelSet[k]]) {
							flag = 1;
						}
					}
					if (flag) {
						double d = 0.0;
						for (int k = 0; k < m_N_FEATURE; k++) {
							if (node->featureSubset[k]) {
								d += (mx_trainData[j][k] - mx_validationData[i][k]) * (mx_trainData[j][k] - mx_validationData[i][k]);
							}
						}
						int index = -1;
						for (int k = 0; k < K; k++) {
							if (dists[k] < 0) {  //uninitialized
								index = k;
								break;
							}
							if (dists[k] > d) {  //greater than d
								if (index == -1) {
									index = k;
								}
								else {
									if (dists[k] > dists[index]) {  //the greatest one greater than d
										index = k;
									}
								}
							}
						}
						if (index >= 0) {
							dists[index] = d;
							labels[index] = (int)mx_trainData[j][m_N_FEATURE];
						}
					}
				}
				double minD = -1;
				int    minLabel = -1;
				for (int k = 0; k < K; k++) {
					if (minD < 0 || dists[k] < minD) {
						minD = dists[k];
						minLabel = labels[k];
					}
				}
				//cout << minD << endl;
				if (minLabel > 0) {
					for (int k = 0; k < node->N_LABEL_left; k++) {
						if (minLabel == v_labelCLASS[node->lLabelSet[k]]) {
							node = node->lchild;
						}
					}
					for (int k = 0; k < node->N_LABEL_right; k++) {
						if (minLabel == v_labelCLASS[node->rLabelSet[k]]) {
							node = node->rchild;
						}
					}
				}
				else {
					printf("The considered label set doesn't exist in the tran set.\n");
				}
			}
			if (predictedLabels == mx_validationData[i][m_N_FEATURE])
				nCorrectValidat++;
		}
	}

	if (m_innerTag) {
		for (int i = 0; i < m_N_sample_train; i++) {
			free(mx_trainData[i]);
			free(mx_trainIdx[i]);
		}
		free(mx_trainData);
		free(mx_trainIdx);
		for (int i = 0; i < m_N_sample_validation; i++) {
			free(mx_validationData[i]);
			free(mx_validationIdx[i]);
		}
		free(mx_validationData);
		free(mx_validationIdx);

		m_innerTag = 0;
	}

	return;
}

void f_Initialize_ClassifierTreeFunc(char prob[], int curN, int numN)
{
	if (curN <= 0) {
		m_seed_CLASS = 237;
		m_rnd_uni_init_CLASS = -(long)m_seed_CLASS;
	}
	for (int i = 0; i < curN; i++) {
		m_seed_CLASS = (m_seed_CLASS + 111) % 1235;
		m_rnd_uni_init_CLASS = -(long)m_seed_CLASS;
	}

	m_filterTag = 0;

	m_opNum = 24;
	m_opNumCur = curN;

	char filename[1024];
	//strcpy(filename, prob);
	//sprintf(filename, "ALLGSE412_poterapiji.data");
	char tmpSTR0[256];
	char tmpSTR[256];
	sscanf(prob, "%[A-Za-z]%[0-9]", tmpSTR0, tmpSTR);
	int tmpINT;
	sscanf(tmpSTR, "%d", &tmpINT);
	tmpINT--;
	sprintf(filename, "../Data_all/Data_FeatureSelection/%s", testInstNames[tmpINT]);
	///////////////////////////////////////////////////////////////////////////////
	//READING DATA
	///////////////////////////////////////////////////////////////////////////////
	f_readData(mx_wholeData, filename, m_M_whole_data, m_N_whole_data);
	m_N_sample_whole = m_M_whole_data;
	m_N_FEATURE = m_N_whole_data - 1;
	//for (int i = 0; i < m_M_whole_data; i++) printf("%lf ", mx_wholeData[i][m_N_FEATURE]);

	//对于缺失数据(-999999999.876)，用均值替代
	for (int i = 0; i < m_N_FEATURE; i++) {
		double sum = 0.0;
		int count = 0;
		for (int j = 0; j < m_N_sample_whole; j++) {
			if (mx_wholeData[j][i] != -999999999.876) {
				sum += mx_wholeData[j][i];
				count++;
			}
		}
		if (count < m_N_sample_whole) {
			sum /= count;
			for (int j = 0; j < m_N_sample_whole; j++) {
				if (mx_wholeData[j][i] == -999999999.876) {
					mx_wholeData[j][i] = sum;
				}
			}
		}
	}

	m_N_CLASS = 0;
	v_labelCLASS[m_N_CLASS++] = (int)mx_wholeData[0][m_N_FEATURE];
	for (int i = 1; i < m_N_sample_whole; i++) {
		int flag = 1;
		for (int j = 0; j < i; j++) {
			if (mx_wholeData[j][m_N_FEATURE] == mx_wholeData[i][m_N_FEATURE])
				flag = 0;
		}
		if (flag) {
			v_labelCLASS[m_N_CLASS++] = (int)mx_wholeData[i][m_N_FEATURE];
		}
		//printf("%lf ", mx_wholeData[i, m_N_FEATURE]);
	}
	for (int i = 0; i < m_N_CLASS; i++) {
		v_arrSize[i] = 0;
	}
	for (int j = 0; j < m_M_whole_data; j++) {
		for (int i = 0; i < m_N_CLASS; i++)
			if ((int)mx_wholeData[j][m_N_FEATURE] == v_labelCLASS[i])
				v_arrSize[i]++;
	}

	//for (int i = 0; i < m_N_CLASS; i++)
	//  cout << v_labelCLASS[i];
	//cout << endl;

	m_N_MODEL = 1;// only kNN

	m_DIM_ClassifierTreeFunc = m_N_CLASS + m_N_FEATURE;

	//// #define m_BOOTSTRAP     0 // .632 bootstrap
	//// #define m_BOOTSTRAPplus 1 // .632+ bootstrap
	//// #define m_MULTIFOLD     2 // 多折交叉
	//// #define m_RATIO         3 // 比例划分
	f_splitData_outer(m_BOOTSTRAP, 0.7);

	return;
}

void f_SetLimits_ClassifierTreeFunc(double* minLimit, double* maxLimit, int nx)
{
	for (int i = 0; i < m_N_CLASS; i++) {
		minLimit[i] = 0;
		maxLimit[i] = m_N_CLASS - 1 - 1e-6;
	}
	for (int i = m_N_CLASS; i < m_N_CLASS + m_N_FEATURE; i++) {
		minLimit[i] = 0;
		maxLimit[i] = pow(2., m_N_CLASS) - 1e-6;
	}

	return;
}

BiTree f_treeDecoding(int* codedTree)
{
	BiTree queue[256];
	int iQueue = 0;
	int nQueue = 0;
	BiTree T = (BiTree)calloc(1, sizeof(BiTNode));
	T->parent = NULL;
	T->level = 1;
	queue[nQueue++] = T;
	int tag = 1;

	// int iIndex = 0;
	int nIndex = 0;
	int arrIndex[256];
	for (int i = 0; i < m_N_CLASS - 1; i++) {
		if (iQueue < nQueue) {
			BiTree node = queue[iQueue++];
			node->label = -1;
			// iIndex = 0;
			nIndex = 0;
			for (int j = 0; j < m_N_CLASS; j++) {
				if (codedTree[j] == i) {
					arrIndex[nIndex++] = j;
				}
			}
			if (nIndex == 0) {
				node->lchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->lchild->parent = node;
				node->lchild->level = node->level + 1;
				node->rchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->rchild->parent = node;
				node->rchild->level = node->level + 1;
				queue[nQueue++] = node->lchild;
				queue[nQueue++] = node->rchild;
			}
			else if (nIndex == 1) {
				node->lchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->lchild->parent = node;
				node->lchild->level = node->level + 1;
				node->rchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->rchild->parent = node;
				node->rchild->level = node->level + 1;
				node->lchild->label = arrIndex[0];
				queue[nQueue++] = node->rchild;
			}
			else if (nIndex == 2) {
				node->lchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->lchild->parent = node;
				node->lchild->level = node->level + 1;
				node->rchild = (BiTree)calloc(1, sizeof(BiTNode));
				node->rchild->parent = node;
				node->rchild->level = node->level + 1;
				node->lchild->label = arrIndex[0];
				node->rchild->label = arrIndex[1];
			}
			else {
				tag = 0;
				break;
			}
		}
		else {
			tag = 0;
			break;
		}
	}

	if (tag == 0) {
		f_freeBiTree(T);
	}

	return T;
}

//特征编码为一个二进制串，长度为类别数目，相应位为1则此特征与当前类别关联
int  f_featureAssign(BiTree& T, int* codedFeature)
{
	if (T) {
		if (!T->lchild && !T->rchild) {
			int tag = (int)(pow(2., T->label + 1));
			T->numFeature = 0;
			T->featureSubset = (int*)calloc(m_N_FEATURE, sizeof(int));
			for (int i = 0; i < m_N_FEATURE; i++) {
				if (codedFeature[i] % tag >= tag / 2) {
					T->featureSubset[i] = 1;
					T->numFeature++;
				}
			}
		}
		else if (!T->lchild && T->rchild) {
			f_featureAssign(T->rchild, codedFeature);
			T->numFeature = 0;
			T->featureSubset = (int*)calloc(m_N_FEATURE, sizeof(int));
			for (int i = 0; i < m_N_FEATURE; i++) {
				if (T->rchild->featureSubset[i]) {
					T->featureSubset[i] = 1;
					T->numFeature++;
				}
			}
		}
		else if (T->rchild && !T->rchild) {
			f_featureAssign(T->lchild, codedFeature);
			T->numFeature = 0;
			T->featureSubset = (int*)calloc(m_N_FEATURE, sizeof(int));
			for (int i = 0; i < m_N_FEATURE; i++) {
				if (T->lchild->featureSubset[i]) {
					T->featureSubset[i] = 1;
					T->numFeature++;
				}
			}
		}
		else {
			f_featureAssign(T->lchild, codedFeature);
			f_featureAssign(T->rchild, codedFeature);
			T->numFeature = 0;
			T->featureSubset = (int*)calloc(m_N_FEATURE, sizeof(int));
			for (int i = 0; i < m_N_FEATURE; i++) {
				if (T->lchild->featureSubset[i] || T->rchild->featureSubset[i]) {
					T->featureSubset[i] = 1;
					T->numFeature++;
				}
			}
		}
		if (T->numFeature == 0) {}
	}

	return 0;
}

//
int  f_labelAssign(BiTree& T)
{
	if (T) {
		if (!T->lchild && !T->rchild) {
			T->N_LABEL_left = 0;
			T->N_LABEL_right = 0;
			T->lLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			T->rLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
		}
		else if (!T->lchild && T->rchild) {
			f_labelAssign(T->rchild);
			T->N_LABEL_left = 0;
			T->N_LABEL_right = 0;
			T->lLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			T->rLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			for (int i = 0; i < T->rchild->N_LABEL_left; i++) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->lLabelSet[i];
			}
			if (T->rchild->label != -1) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->label;
			}
			for (int i = 0; i < T->rchild->N_LABEL_right; i++) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->rLabelSet[i];
			}
		}
		else if (T->lchild && !T->rchild) {
			f_labelAssign(T->lchild);
			T->N_LABEL_left = 0;
			T->N_LABEL_right = 0;
			T->lLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			T->rLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			for (int i = 0; i < T->lchild->N_LABEL_left; i++) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->lLabelSet[i];
			}
			if (T->lchild->label != -1) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->label;
			}
			for (int i = 0; i < T->lchild->N_LABEL_right; i++) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->rLabelSet[i];
			}
		}
		else {
			f_labelAssign(T->lchild);
			f_labelAssign(T->rchild);
			T->N_LABEL_left = 0;
			T->N_LABEL_right = 0;
			T->lLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			T->rLabelSet = (int*)calloc(m_N_CLASS, sizeof(int));
			for (int i = 0; i < T->lchild->N_LABEL_left; i++) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->lLabelSet[i];
			}
			if (T->lchild->label != -1) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->label;
			}
			for (int i = 0; i < T->lchild->N_LABEL_right; i++) {
				T->lLabelSet[T->N_LABEL_left++] = T->lchild->rLabelSet[i];
			}
			for (int i = 0; i < T->rchild->N_LABEL_left; i++) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->lLabelSet[i];
			}
			if (T->rchild->label != -1) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->label;
			}
			for (int i = 0; i < T->rchild->N_LABEL_right; i++) {
				T->rLabelSet[T->N_LABEL_right++] = T->rchild->rLabelSet[i];
			}
		}
	}

	return 0;
}

void f_printBiTree(BiTree T)
{
	BiTree queue[256];
	int iQueue = 0;
	int nQueue = 0;

	queue[nQueue++] = T;

	while (iQueue < nQueue) {
		BiTree node = queue[iQueue++];
		if (node) {
			printf("%d --- %d --- %d\n", node->level, node->label, node->numFeature);
			if (node->numFeature) {
				int sum = 0;
				for (int i = 0; i < m_N_FEATURE; i++) {
					printf("%d", node->featureSubset[i]);
					if (node->featureSubset[i])
						sum++;
				}
				printf("\n");
				printf("%d\n", sum);
			}
			if (node->N_LABEL_left) {
				printf("lLabelSet: \n");
				for (int i = 0; i < node->N_LABEL_left; i++) {
					printf("%d", node->lLabelSet[i]);
				}
				printf("\n");
			}
			if (node->N_LABEL_right) {
				printf("rLabelSet: \n");
				for (int i = 0; i < node->N_LABEL_right; i++) {
					printf("%d", node->rLabelSet[i]);
				}
				printf("\n");
			}
			if (node->lchild)
				queue[nQueue++] = node->lchild;
			if (node->rchild)
				queue[nQueue++] = node->rchild;
		}
	}

	return;
}

int  f_freeBiTree(BiTree& T)
{
	if (!T) return 0;

	f_freeBiTree(T->lchild);
	f_freeBiTree(T->rchild);
	if (T->featureSubset)
		free(T->featureSubset);
	if (T->lLabelSet)
		free(T->lLabelSet);
	if (T->rLabelSet)
		free(T->rLabelSet);
	free(T);
	T = NULL;

	return 0;
}

void f_Fitness_ClassifierTreeFunc(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	// int numFeature = 0;

	int* codedTree = (int*)calloc(m_N_CLASS, sizeof(int));
	for (int i = 0; i < m_N_CLASS; i++) {
		codedTree[i] = (int)individual[i];
		//cout << codedTree[i] << "---" << individual[i] << " ";
	}
	//cout << endl;
	int* featureTags = (int*)calloc(m_N_FEATURE, sizeof(int));
	for (int i = 0; i < m_N_FEATURE; i++) {
		featureTags[i] = (int)individual[m_N_CLASS + i];
		//cout << featureTags[i] << "---" << individual[m_N_CLASS + i] << " ";
	}
	//cout << endl;

	BiTree T;
	int flag = 0;

	do {
		for (int i = 0; i < m_DIM_ClassifierTreeFunc; i++) {
			individual[i] = 0 + f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS) * (m_N_CLASS - 1 - 1e-6 - 0);
		}

		int tmp[256];
		for (int i = 0; i < m_N_CLASS; i++) {
			tmp[i] = (int)individual[i];
		}
		flag = 0;
		T = f_treeDecoding(tmp);
		if (!T)
			flag = 1;
	} while (flag);
	if (!T)
		printf("Tree Invalid!!!\n");
	//f_printBiTree(T);
	f_featureAssign(T, featureTags);
	f_labelAssign(T);
	//f_printBiTree(T);

	if (T->numFeature) {
		int nFold = m_N_sample_optimize;
		int sumSample = 0;
		int sumCorrect = 0;
		for (int i = 0; i < nFold; i++) {
			int nCorrect = 0;
			int nSample = 0;
			// #define m_BOOTSTRAP     0 // .632 bootstrap
			// #define m_BOOTSTRAPplus 1 // .632+ bootstrap
			// #define m_MULTIFOLD     2 // 多折交叉
			// #define m_RATIO         3 // 比例划分
			f_splitData_inner(m_MULTIFOLD, nFold, i, 0.7);
			f_cascadedClassifierMineBT(T, nCorrect, nSample);
			sumSample += nSample;
			sumCorrect += nCorrect;
		}

		fitness[0] = 1.0 - (double)sumCorrect / sumSample;
		fitness[1] = (double)T->numFeature / m_N_FEATURE;
	}
	else {
		fitness[0] = 10.0;
		fitness[1] = (double)T->numFeature / m_N_FEATURE;
	}
	//printf("%lf\n", (double)sumCorrect / sumSample);
	//printf("%d\n", T->numFeature);

	free(codedTree);
	free(featureTags);
	f_freeBiTree(T);

	return;
}

int f_CheckLimits_ClassifierTreeFunc(double* x, int nx)
{
	for (int i = 0; i < m_N_CLASS; i++) {
		if (x[i] < 0 || x[i] > m_N_CLASS - 1 - 1e-6) {
			return false;
		}
	}
	for (int i = m_N_CLASS; i < m_N_CLASS + m_N_FEATURE; i++) {
		if (x[i] < 0 || x[i] > pow(2., m_N_CLASS) - 1e-6) {
			return false;
		}
	}

	return true;
}

void f_freeMemoryTreeCLASS()
{
	for (int i = 0; i < m_M_whole_data; i++) {
		free(mx_wholeData[i]);
	}
	free(mx_wholeData);

	for (int i = 0; i < m_N_sample_optimize; i++) {
		free(mx_optimizeData[i]);
		free(mx_optimizeIdx[i]);
	}
	free(mx_optimizeData);
	free(mx_optimizeIdx);
	for (int i = 0; i < m_N_sample_test; i++) {
		free(mx_testData[i]);
		free(mx_testIdx[i]);
	}
	free(mx_testData);
	free(mx_testIdx);

	if (m_innerTag) {
		for (int i = 0; i < m_N_sample_train; i++) {
			free(mx_trainData[i]);
			free(mx_trainIdx[i]);
		}
		free(mx_trainData);
		free(mx_trainIdx);
		for (int i = 0; i < m_N_sample_validation; i++) {
			free(mx_validationData[i]);
			free(mx_validationIdx[i]);
		}
		free(mx_validationData);
		free(mx_validationIdx);

		m_innerTag = 0;
	}

	if (m_filterTag) {
		for (int i = 0; i < m_N_sample_filter; i++) {
			free(mx_filterData[i]);
		}
		free(mx_filterData);
		for (int i = 0; i < m_N_FEATURE; i++) {
			free(mx_filterDataMinMax[i]);
			free(mx_weights[i]);
		}
		free(mx_filterDataMinMax);
		free(mx_weights);

		m_filterTag = 0;
	}

	return;
}

void f_filter_ReliefF()
{
	if (m_filterTag) {
		for (int i = 0; i < m_N_sample_filter; i++) {
			free(mx_filterData[i]);
		}
		free(mx_filterData);
		for (int i = 0; i < m_N_FEATURE; i++) {
			free(mx_filterDataMinMax[i]);
			free(mx_weights[i]);
		}
		free(mx_filterDataMinMax);
		free(mx_weights);

		m_filterTag = 0;
	}

	mx_weights = f_allocDouble(m_N_FEATURE, 1);
	for (int i = 0; i < m_N_FEATURE; i++) {
		mx_weights[i][0] = 0.0;
	}

	m_N_sample_filter = m_N_sample_whole - m_N_sample_test;
	mx_filterData = f_allocDouble(m_N_sample_filter, m_N_whole_data);

	mx_selectionFlag = f_allocInt(m_N_sample_whole, 1);
	for (int i = 0; i < m_N_sample_whole; i++)
		mx_selectionFlag[i][0] = 0;
	for (int i = 0; i < m_N_sample_test; i++)
		mx_selectionFlag[mx_testIdx[i][0]][0] = 1;
	int count = 0;
	for (int i = 0; i < m_N_sample_whole; i++)
		if (!mx_selectionFlag[i][0])
			memcpy(mx_filterData[count++], mx_wholeData[i], m_N_whole_data * sizeof(double));

	//////////////////
	mx_filterDataMinMax = f_allocDouble(m_N_FEATURE, 2);
	for (int i = 0; i < m_N_FEATURE; i++) {
		mx_filterDataMinMax[i][0] = 1e308;
		mx_filterDataMinMax[i][1] = -1e308;
	}
	for (int i = 0; i < m_N_CLASS; i++) {
		v_ratioLabelCLASS[i] = 0.0;
	}
	for (int i = 0; i < m_N_sample_filter; i++) {
		for (int j = 0; j < m_N_CLASS; j++) {
			if ((int)mx_filterData[i][m_N_FEATURE] == v_labelCLASS[j]) {
				v_ratioLabelCLASS[j] += 1.0;
			}
		}
		for (int j = 0; j < m_N_FEATURE; j++) {
			if (mx_filterDataMinMax[j][0] > mx_filterData[i][j])
				mx_filterDataMinMax[j][0] = mx_filterData[i][j];
			if (mx_filterDataMinMax[j][1] < mx_filterData[i][j])
				mx_filterDataMinMax[j][1] = mx_filterData[i][j];
		}
	}
	for (int i = 0; i < m_N_CLASS; i++) {
		v_ratioLabelCLASS[i] /= m_N_sample_filter;
		if (v_ratioLabelCLASS[i] == 0)
			printf("ZERO ERROR!\n");
	}
	///////////////////
	const int K = (m_N_sample_filter / 10 > 10) ? 10 : (m_N_sample_filter / 10);
	double** dists = f_allocDouble(m_N_CLASS, K);
	int** indices = f_allocInt(m_N_CLASS, K);
	//////////////////
	srand(0);
	for (int i = 0; i < m_N_sample_filter; i++) {
		int idx = (int)(rand() / (RAND_MAX + 1.0) * m_N_sample_filter);
		int curClass = (int)(mx_filterData[idx][m_N_FEATURE]);
		for (int j = 0; j < m_N_CLASS; j++) {
			for (int k = 0; k < K; k++) {
				dists[j][k] = 1e308;
				indices[j][k] = -1;
			}
		}
		for (int j = 0; j < m_N_sample_filter; j++) {
			double d = 0;
			for (int k = 0; k < m_N_FEATURE; k++) {
				d += (mx_filterData[idx][k] - mx_filterData[j][k]) * (mx_filterData[idx][k] - mx_filterData[j][k]);
			}
			int tmpClass = -1;
			for (int k = 0; k < m_N_CLASS; k++) {
				if ((int)mx_filterData[j][m_N_FEATURE] == v_labelCLASS[k]) {
					tmpClass = k;
				}
			}
			if (tmpClass == -1) {
				printf("%s(%d): index 'tmpClass' is -1, ERROR, ... \n", __FILE__, __LINE__);
				exit(-1);
			}
			int tmpIdx = -1;
			for (int k = 0; k < K; k++) {
				if (dists[tmpClass][k] > d && d > 0) {
					if (tmpIdx == -1) {
						tmpIdx = k;
					}
					else {
						if (dists[tmpClass][k] > dists[tmpClass][tmpIdx]) {
							tmpIdx = k;
						}
					}
				}
			}
			if (tmpIdx >= 0) {
				dists[tmpClass][tmpIdx] = d;
				indices[tmpClass][tmpIdx] = j;
			}
		}
		////////////////////////////////
		for (int j = 0; j < m_N_FEATURE; j++) {
			double sum1 = 0.0;
			double sum2 = 0.0;
			for (int m = 0; m < m_N_CLASS; m++) {
				for (int k = 0; k < K; k++) {
					double tmp = mx_filterData[idx][j] - mx_filterData[indices[m][k]][j];
					tmp /= (mx_filterDataMinMax[j][1] - mx_filterDataMinMax[j][0]);
					if (tmp < 0.0) tmp = -tmp;
					if ((int)mx_filterData[indices[m][k]][m_N_FEATURE] == curClass)
						sum1 += tmp;
					else {
						int tmpClass = -1;
						for (int n = 0; n < m_N_CLASS; n++) {
							if ((int)mx_filterData[indices[m][k]][m_N_FEATURE] == v_labelCLASS[n]) {
								tmpClass = n;
							}
						}
						if (tmpClass == -1) {
							printf("%s(%d): index 'tmpClass' is -1, ERROR, ... \n", __FILE__, __LINE__);
							exit(-1);
						}
						sum2 += v_ratioLabelCLASS[tmpClass] / (1 - v_ratioLabelCLASS[curClass]) * tmp;
					}
				}
			}
			mx_weights[j][0] = mx_weights[j][0] - sum1 / (m_N_sample_filter * K) + sum2 / (m_N_sample_filter * K);
		}
	}

	for (int i = 0; i < m_N_CLASS; i++) {
		free(dists[i]);
		free(indices[i]);
	}
	free(dists);
	free(indices);
	for (int i = 0; i < m_N_sample_whole; i++) {
		free(mx_selectionFlag[i]);
	}
	free(mx_selectionFlag);

	m_filterTag = 1;

	return;
}

void f_testAccuracy(double* individual, double* fitness)
{
	// int numFeature = 0;

	int* codedTree = (int*)calloc(m_N_CLASS, sizeof(int));
	for (int i = 0; i < m_N_CLASS; i++) {
		codedTree[i] = (int)individual[i];
		//cout << codedTree[i] << "---" << individual[i] << " ";
	}
	//cout << endl;
	int* featureTags = (int*)calloc(m_N_FEATURE, sizeof(int));
	for (int i = 0; i < m_N_FEATURE; i++) {
		featureTags[i] = (int)individual[m_N_CLASS + i];
		//cout << featureTags[i] << "---" << individual[m_N_CLASS + i] << " ";
	}
	//cout << endl;

	BiTree T = f_treeDecoding(codedTree);
	if (!T)
		printf("Tree Invalid!!!\n");
	//f_printBiTree(T);
	f_featureAssign(T, featureTags);
	f_labelAssign(T);
	//f_printBiTree(T);

	int nTEST = m_N_sample_test;
	int sumSample = 0;
	int sumCorrect = 0;
	for (int i = 0; i < nTEST; i++) {
		int nCorrect = 0;
		int nSample = 0;
		// #define m_BOOTSTRAP     0 // .632 bootstrap
		// #define m_BOOTSTRAPplus 1 // .632+ bootstrap
		// #define m_MULTIFOLD     2 // 多折交叉
		// #define m_RATIO         3 // 比例划分
		if (T) {
			int predictedLabels;
			nSample = 1;
			//kNN - mine
			{
				const int K = 1;
				double dists[K];
				int    labels[K];
				nCorrect = 0;
				//cout << "SMP_IND: " << i << endl;
				//Traverse the tree
				BiTree node = T;
				// int tag = 1;//1 - left, 2 - right
				while (1) {
					for (int j = 0; j < K; j++) {
						dists[j] = -1;
						labels[j] = -1;
					}
					if (node->label != -1) {
						predictedLabels = v_labelCLASS[node->label];
						break;
					}
					for (int j = 0; j < m_N_sample_optimize; j++) {
						int flag = 0;
						for (int k = 0; k < node->N_LABEL_left; k++) {
							if ((int)mx_optimizeData[j][m_N_whole_data - 1] == v_labelCLASS[node->lLabelSet[k]]) {
								flag = 1;
							}
						}
						for (int k = 0; k < node->N_LABEL_right; k++) {
							if ((int)mx_optimizeData[j][m_N_whole_data - 1] == v_labelCLASS[node->rLabelSet[k]]) {
								flag = 2;
							}
						}
						if (flag) {
							double d = 0.0;
							for (int k = 0; k < m_N_FEATURE; k++) {
								if (node->featureSubset[k]) {
									d += (mx_optimizeData[j][k] - mx_testData[i][k]) * (mx_optimizeData[j][k] - mx_testData[i][k]);
								}
							}
							int index = -1;
							for (int k = 0; k < K; k++) {
								if (dists[k] < 0) {
									index = k;
									break;
								}
								if (dists[k] > d) {
									if (index == -1) {
										index = k;
									}
									else {
										if (dists[k] > dists[index]) {
											index = k;
										}
									}
								}
							}
							if (index >= 0) {
								dists[index] = d;
								labels[index] = (int)mx_optimizeData[j][m_N_FEATURE];
							}
						}
					}
					double minD = -1;
					int    minLabel = -1;
					for (int k = 0; k < K; k++) {
						if (minD < 0 || dists[k] < minD) {
							minD = dists[k];
							minLabel = labels[k];
						}
					}
					//cout << minD << endl;
					if (minLabel > 0) {
						for (int k = 0; k < node->N_LABEL_left; k++) {
							if (minLabel == v_labelCLASS[node->lLabelSet[k]]) {
								node = node->lchild;
							}
						}
						for (int k = 0; k < node->N_LABEL_right; k++) {
							if (minLabel == v_labelCLASS[node->rLabelSet[k]]) {
								node = node->rchild;
							}
						}
					}
					else {
						printf("The considered label set doesn't exist in the tran set.\n");
					}
				}
				if (predictedLabels == mx_testData[i][m_N_FEATURE])
					nCorrect++;
			}
		}
		sumSample += nSample;
		sumCorrect += nCorrect;
	}

	fitness[0] = 1.0 - (double)sumCorrect / sumSample;
	fitness[1] = (double)T->numFeature / m_N_FEATURE;
	//printf("%lf\n", (double)sumCorrect / sumSample);
	//printf("%d\n", T->numFeature);

	free(codedTree);
	free(featureTags);

	return;
}

void f_bootstrapInitialize(int**& index)
{
	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////
	int* v_selectIndicator = (int*)calloc(m_M_whole_data * m_opNum, sizeof(int));

	int count = 0;
	for (int i = 0; i < m_N_CLASS; i++) {
		for (int j = 0; j < m_M_whole_data; j++) {
			if ((int)mx_wholeData[j][m_N_FEATURE] == v_labelCLASS[i]) {
				for (int k = count; k < count + m_opNum; k++) {
					v_selectIndicator[k] = -j;
				}
				count += m_opNum;
			}
		}
	}

	int* tmpIDX = (int*)calloc(m_M_whole_data, sizeof(int));

	srand(0);

	for (int n = 0; n < m_opNum; n++) {
		int ind_begin = 0;
		int ind_end = 0;

		for (int i = 0; i < m_N_CLASS; i++) {
			ind_begin = ind_end;
			ind_end = ind_begin + v_arrSize[i] * m_opNum;

			int j = ind_begin / m_opNum;
			while (j < ind_end / m_opNum) {
				int tmp = (int)(ind_begin + rand() / (RAND_MAX + 1.0) * v_arrSize[i] * m_opNum);
				if (v_selectIndicator[tmp] <= 0) {
					tmpIDX[j] = -v_selectIndicator[tmp];
					v_selectIndicator[tmp] = 1 - v_selectIndicator[tmp];
					j++;
				}
			}
		}

		if (n == m_opNumCur) {
			for (int i = 0; i < m_M_whole_data; i++) {
				index[i][0] = tmpIDX[i];
			}
			break;
		}

		//for (int j = 0; j < M_whole_data; j++){
		//  printf("%2d ", tmpIDX[j]);
		//}
		//printf("\n");
	}

	free(v_selectIndicator);
	free(tmpIDX);
}

//int main()
//{
//  f_Initialize_ClassifierTreeFunc();
//
//  double* LowLimit = (double*)calloc(m_DIM_ClassifierTreeFunc, sizeof(double));
//  double* UppLimit = (double*)calloc(m_DIM_ClassifierTreeFunc, sizeof(double));
//
//  f_SetLimits_ClassifierTreeFunc(LowLimit, UppLimit);
//
//  double* tmpX = (double*)calloc(m_DIM_ClassifierTreeFunc, sizeof(double));
//  double* tmpY = (double*)calloc(m_DIM_ClassifierTreeFunc, sizeof(double));
//  int flag = 0;
//
//  do{
//      for (int i = 0; i < m_DIM_ClassifierTreeFunc; i++){
//          tmpX[i] = LowLimit[i] + f_rnd_uni_CLASS(&m_rnd_uni_init_CLASS)*(UppLimit[i] - LowLimit[i]);
//      }
//
//      int tmp[256];
//      for (int i = 0; i < m_N_CLASS; i++){
//          tmp[i] = (int)tmpX[i];
//      }
//      flag = 0;
//      BiTree T = f_treeDecoding(tmp);
//      if (!T)
//          flag = 1;
//      else
//          f_freeBiTree(T);
//  } while (flag);
//
//  double t = (double)clock();
//  f_Fitness_ClassifierTreeFunc(tmpX, tmpY);
//  t = ((double)clock() - t) / CLOCKS_PER_SEC;
//  printf("CvKNearest --- Times passed in seconds: %lf\n", t);
//
//  free(LowLimit);
//  free(UppLimit);
//  free(tmpX);
//  free(tmpY);
//
//  return (0);
//}