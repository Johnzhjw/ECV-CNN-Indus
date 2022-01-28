#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOP_RecSys_SmartCity.h"

#define VAL_MIN_RS_SC (00.0)
#define VAL_MAX_RS_SC (10.0)

#define SIZE_LIST_RS_SC (4)
#define b_DCG (3)
#define DCG_log_b_3 (1)

static int   seed_rand_RS_SC = 237;
static long  rnd_uni_init_val_RS_SC = -(long)seed_rand_RS_SC;
#define IM1_RS_SC 2147483563
#define IM2_RS_SC 2147483399
#define AM_RS_SC (1.0/IM1_RS_SC)
#define IMM1_RS_SC (IM1_RS_SC-1)
#define IA1_RS_SC 40014
#define IA2_RS_SC 40692
#define IQ1_RS_SC 53668
#define IQ2_RS_SC 52774
#define IR1_RS_SC 12211
#define IR2_RS_SC 3791
#define NTAB_RS_SC 32
#define NDIV_RS_SC (1+IMM1_RS_SC/NTAB_RS_SC)
#define EPS_RS_SC 1.2e-7
#define RNMX_RS_SC (1.0-EPS_RS_SC)

static int nImp_cur_RS_SC;

static int UsrLocAct_RS_SC[N_USR_RS_SC][N_LOC_RS_SC][N_ACT_RS_SC];
static int UsrLoc_RS_SC[N_USR_RS_SC][N_LOC_RS_SC];
static double UsrUsr_RS_SC[N_USR_RS_SC][N_USR_RS_SC];
static double ActAct_RS_SC[N_ACT_RS_SC][N_ACT_RS_SC];
static double LocFea_RS_SC[N_LOC_RS_SC][N_FEA_RS_SC];
static double Lap_UsrUsr_RS_SC[N_USR_RS_SC][N_USR_RS_SC];
static double Lap_ActAct_RS_SC[N_ACT_RS_SC][N_ACT_RS_SC];

static double UsrLocAct_RS_SC_IR[N_USR_RS_SC][N_LOC_RS_SC][N_ACT_RS_SC];
static double ImpUsr_RS_SC[N_USR_RS_SC][N_IMP_RS_SC];
static double ImpLoc_RS_SC[N_LOC_RS_SC][N_IMP_RS_SC];
static double ImpAct_RS_SC[N_ACT_RS_SC][N_IMP_RS_SC];
static double ImpFea_RS_SC[N_FEA_RS_SC][N_IMP_RS_SC];

static int trainFlag_RS_SC[N_USR_RS_SC][N_LOC_RS_SC][N_ACT_RS_SC];
static int testFlag_RS_SC[N_USR_RS_SC][N_LOC_RS_SC][N_ACT_RS_SC];
static int trainNum;
static int testNum;

static double iDCG_UsrLoc4Act_train[N_USR_RS_SC][N_LOC_RS_SC];
static double iDCG_UsrAct4Loc_train[N_USR_RS_SC][N_ACT_RS_SC];
static double iDCG_UsrLoc4Act_test[N_USR_RS_SC][N_LOC_RS_SC];
static double iDCG_UsrAct4Loc_test[N_USR_RS_SC][N_ACT_RS_SC];

static int degree_Loc_train[N_LOC_RS_SC];
static int degree_Act_train[N_ACT_RS_SC];
static int degree_Loc_test[N_LOC_RS_SC];
static int degree_Act_test[N_ACT_RS_SC];

//////////////////////////////////////////////////////////////////////////
//the random generator in [0,1) ~ [vmin, vmax)
static double rnd_uni_gen_RS_SC(long* idum, double vmin, double vmax);
static void shuffle_RS_SC(int* x, int size);
static double* allocDouble_RS_SC(int size);
// static int *allocInt_RS_SC(int size);

//////////////////////////////////////////////////////////////////////////
void Initialize_data_RS_SC(int curN, int numN)
{
	//////////////////////////////////////////////////////////////////////////
	seed_rand_RS_SC = 237;
	for (int i = 0; i < curN; i++) {
		seed_rand_RS_SC = (seed_rand_RS_SC + 111) % 1235;
	}
	rnd_uni_init_val_RS_SC = -(long)seed_rand_RS_SC;

	//////////////////////////////////////////////////////////////////////////
	char filename[1024];
	int theSize[3] = { 0, 0, 0 };
	FILE* fpt;
	int tmp;
	int dim;

	//////////////////////////////////////////////////////////////////////////
	sprintf(filename, "../Data_all/Data_RecSys_SmartCity/UserLocAct");
	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}
	tmp = fscanf(fpt, "%d", &dim);
	if (tmp == EOF) {
		printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
		exit(2000);
	}
	for (int i = 0; i < dim; i++) {
		tmp = fscanf(fpt, "%d", &theSize[i]);
		if (tmp == EOF) {
			printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
			exit(2000);
		}
	}
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				tmp = fscanf(fpt, "%d", &UsrLocAct_RS_SC[i][j][k]);
				if (tmp == EOF) {
					printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
					exit(2000);
				}
			}
		}
	}
	fclose(fpt);
	//////////////////////////////////////////////////////////////////////////
	sprintf(filename, "../Data_all/Data_RecSys_SmartCity/UserLoc");
	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}
	tmp = fscanf(fpt, "%d", &dim);
	if (tmp == EOF) {
		printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
		exit(2000);
	}
	for (int i = 0; i < dim; i++) {
		tmp = fscanf(fpt, "%d", &theSize[i]);
		if (tmp == EOF) {
			printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
			exit(2000);
		}
	}
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			tmp = fscanf(fpt, "%d", &UsrLoc_RS_SC[i][j]);
			if (tmp == EOF) {
				printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
				exit(2000);
			}
		}
	}
	fclose(fpt);
	//////////////////////////////////////////////////////////////////////////
	sprintf(filename, "../Data_all/Data_RecSys_SmartCity/UserUser");
	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}
	tmp = fscanf(fpt, "%d", &dim);
	if (tmp == EOF) {
		printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
		exit(2000);
	}
	for (int i = 0; i < dim; i++) {
		tmp = fscanf(fpt, "%d", &theSize[i]);
		if (tmp == EOF) {
			printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
			exit(2000);
		}
	}
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_USR_RS_SC; j++) {
			tmp = fscanf(fpt, "%le", &UsrUsr_RS_SC[i][j]);
			if (tmp == EOF) {
				printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
				exit(2000);
			}
		}
	}
	fclose(fpt);
	//////////////////////////////////////////////////////////////////////////
	sprintf(filename, "../Data_all/Data_RecSys_SmartCity/ActAct");
	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}
	tmp = fscanf(fpt, "%d", &dim);
	if (tmp == EOF) {
		printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
		exit(2000);
	}
	for (int i = 0; i < dim; i++) {
		tmp = fscanf(fpt, "%d", &theSize[i]);
		if (tmp == EOF) {
			printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
			exit(2000);
		}
	}
	for (int i = 0; i < N_ACT_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			tmp = fscanf(fpt, "%le", &ActAct_RS_SC[i][j]);
			if (tmp == EOF) {
				printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
				exit(2000);
			}
		}
	}
	fclose(fpt);
	//////////////////////////////////////////////////////////////////////////
	sprintf(filename, "../Data_all/Data_RecSys_SmartCity/LocFea");
	if ((fpt = fopen(filename, "r")) == NULL) {
		printf("%s(%d): File open error!\n", __FILE__, __LINE__);
		exit(10000);
	}
	tmp = fscanf(fpt, "%d", &dim);
	if (tmp == EOF) {
		printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
		exit(2000);
	}
	for (int i = 0; i < dim; i++) {
		tmp = fscanf(fpt, "%d", &theSize[i]);
		if (tmp == EOF) {
			printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
			exit(2000);
		}
	}
	for (int i = 0; i < N_LOC_RS_SC; i++) {
		for (int j = 0; j < N_FEA_RS_SC; j++) {
			tmp = fscanf(fpt, "%le", &LocFea_RS_SC[i][j]);
			if (tmp == EOF) {
				printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
				exit(2000);
			}
		}
	}
	fclose(fpt);
	//////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < N_USR_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int j = 0; j < N_USR_RS_SC; j++) {
			tmp_sum += UsrUsr_RS_SC[i][j];
			Lap_UsrUsr_RS_SC[i][j] =
				0.0 - UsrUsr_RS_SC[i][j];
		}
		Lap_UsrUsr_RS_SC[i][i] =
			tmp_sum - UsrUsr_RS_SC[i][i];
	}
	for (int i = 0; i < N_ACT_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			tmp_sum += ActAct_RS_SC[i][j];
			Lap_ActAct_RS_SC[i][j] =
				0.0 - ActAct_RS_SC[i][j];
		}
		Lap_ActAct_RS_SC[i][i] =
			tmp_sum - ActAct_RS_SC[i][i];
	}
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	int indices_tmp[N_ELEM_TENSER_RS_SC];
	for (int i = 0; i < N_ELEM_TENSER_RS_SC; i++) {
		indices_tmp[i] = i;
	}
	shuffle_RS_SC(indices_tmp, N_ELEM_TENSER_RS_SC);
	int tmp_i = 0;
	int tmp_j = 0;
	int tmp_k = 0;
	for (int i = 0; i < N_ELEM_TENSER_RS_SC; i++) {
		tmp_i = (indices_tmp[i]) / (N_LOC_RS_SC * N_ACT_RS_SC);
		tmp_j = (indices_tmp[i] - tmp_i * N_LOC_RS_SC * N_ACT_RS_SC) / N_ACT_RS_SC;
		tmp_k = (indices_tmp[i] - tmp_i * N_LOC_RS_SC * N_ACT_RS_SC - tmp_j * N_ACT_RS_SC);
		if (i < N_ELEM_TENSER_RS_SC / 2) {
			trainFlag_RS_SC[tmp_i][tmp_j][tmp_k] = 1;
			testFlag_RS_SC[tmp_i][tmp_j][tmp_k] = 0;
		}
		else {
			trainFlag_RS_SC[tmp_i][tmp_j][tmp_k] = 0;
			testFlag_RS_SC[tmp_i][tmp_j][tmp_k] = 1;
		}
	}
	trainNum = N_ELEM_TENSER_RS_SC / 2;
	testNum = N_ELEM_TENSER_RS_SC - trainNum;
	//////////////////////////////////////////////////////////////////////////
	char tmp_fn[256];
	sprintf(tmp_fn, "trainFlag_R%d", curN + 1);
	FILE* tmp_fpt = fopen(tmp_fn, "w");
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				fprintf(tmp_fpt, "%d ", trainFlag_RS_SC[i][j][k]);
			}
			fprintf(tmp_fpt, "\n");
		}
	}
	fclose(tmp_fpt);
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	int tmp_vecAct[N_ACT_RS_SC];
	int tmp_vecLoc[N_LOC_RS_SC];
	//
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			int tmp_count = 0;
			memset(tmp_vecAct, 0, N_ACT_RS_SC * sizeof(int));
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (trainFlag_RS_SC[i][j][k]) {
					tmp_vecAct[tmp_count++] = UsrLocAct_RS_SC[i][j][k];
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_vecAct[a] < tmp_vecAct[b]) {
						int tmp_int = tmp_vecAct[a];
						tmp_vecAct[a] = tmp_vecAct[b];
						tmp_vecAct[b] = tmp_int;
					}
				}
			}
			iDCG_UsrLoc4Act_train[i][j] = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				if (a <= b_DCG - 1) {
					iDCG_UsrLoc4Act_train[i][j] += tmp_vecAct[a];
				}
				else {
					iDCG_UsrLoc4Act_train[i][j] += tmp_vecAct[a] / (log(a + 1) / log(b_DCG));
				}
			}
		}
	}
	//
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			int tmp_count = 0;
			memset(tmp_vecLoc, 0, N_LOC_RS_SC * sizeof(int));
			for (int k = 0; k < N_LOC_RS_SC; k++) {
				if (trainFlag_RS_SC[i][k][j]) {
					tmp_vecLoc[tmp_count++] = UsrLocAct_RS_SC[i][k][j];
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = 0; b < tmp_count; b++) {
					if (tmp_vecLoc[a] < tmp_vecLoc[b]) {
						int tmp_int = tmp_vecLoc[a];
						tmp_vecLoc[a] = tmp_vecLoc[b];
						tmp_vecLoc[b] = tmp_int;
					}
				}
			}
			iDCG_UsrAct4Loc_train[i][j] = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				if (a <= b_DCG - 1) {
					iDCG_UsrAct4Loc_train[i][j] += tmp_vecLoc[a];
				}
				else {
					iDCG_UsrAct4Loc_train[i][j] += tmp_vecLoc[a] / (log(a + 1) / log(b_DCG));
				}
			}
		}
	}
	//
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			int tmp_count = 0;
			memset(tmp_vecAct, 0, N_ACT_RS_SC * sizeof(int));
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (testFlag_RS_SC[i][j][k]) {
					tmp_vecAct[tmp_count++] = UsrLocAct_RS_SC[i][j][k];
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_vecAct[a] < tmp_vecAct[b]) {
						int tmp_int = tmp_vecAct[a];
						tmp_vecAct[a] = tmp_vecAct[b];
						tmp_vecAct[b] = tmp_int;
					}
				}
			}
			iDCG_UsrLoc4Act_test[i][j] = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				if (a <= b_DCG - 1) {
					iDCG_UsrLoc4Act_test[i][j] += tmp_vecAct[a];
				}
				else {
					iDCG_UsrLoc4Act_test[i][j] += tmp_vecAct[a] / (log(a + 1) / log(b_DCG));
				}
			}
		}
	}
	//
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			int tmp_count = 0;
			memset(tmp_vecLoc, 0, N_LOC_RS_SC * sizeof(int));
			for (int k = 0; k < N_LOC_RS_SC; k++) {
				if (testFlag_RS_SC[i][k][j]) {
					tmp_vecLoc[tmp_count++] = UsrLocAct_RS_SC[i][k][j];
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = 0; b < tmp_count; b++) {
					if (tmp_vecLoc[a] < tmp_vecLoc[b]) {
						int tmp_int = tmp_vecLoc[a];
						tmp_vecLoc[a] = tmp_vecLoc[b];
						tmp_vecLoc[b] = tmp_int;
					}
				}
			}
			iDCG_UsrAct4Loc_test[i][j] = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				if (a <= b_DCG - 1) {
					iDCG_UsrAct4Loc_test[i][j] += tmp_vecLoc[a];
				}
				else {
					iDCG_UsrAct4Loc_test[i][j] += tmp_vecLoc[a] / (log(a + 1) / log(b_DCG));
				}
			}
		}
	}
	// Degree
	int flag_train = 0;
	int flag_test = 0;
	for (int j = 0; j < N_LOC_RS_SC; j++) {
		degree_Loc_train[j] = 0;
		degree_Loc_test[j] = 0;
		for (int i = 0; i < N_USR_RS_SC; i++) {
			flag_train = 0;
			flag_test = 0;
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (UsrLocAct_RS_SC[i][j][k]) {
					if (trainFlag_RS_SC[i][j][k]) {
						flag_train++;
					}
					else {
						flag_test++;
					}
				}
				if (flag_train && flag_test)
					break;
			}
			if (flag_train) {
				degree_Loc_train[j]++;
			}
			if (flag_test) {
				degree_Loc_test[j]++;
			}
		}
	}
	for (int k = 0; k < N_ACT_RS_SC; k++) {
		degree_Act_train[k] = 0;
		degree_Act_test[k] = 0;
		for (int i = 0; i < N_USR_RS_SC; i++) {
			flag_train = 0;
			flag_test = 0;
			for (int j = 0; j < N_LOC_RS_SC; j++) {
				if (UsrLocAct_RS_SC[i][j][k]) {
					if (trainFlag_RS_SC[i][j][k]) {
						flag_train++;
					}
					else {
						flag_test++;
					}
				}
				if (flag_train && flag_test)
					break;
			}
			if (flag_train) {
				degree_Act_train[k]++;
			}
			if (flag_test) {
				degree_Act_test[k]++;
			}
		}
	}

	return;
}

void Finalize_data_RS_SC()
{
	return;
}

void SetLimits_RS_SC(double* minLimit, double* maxLimit, int nx)
{
	for (int i = 0; i < DIM_RS_SC - 1; i++) {
		minLimit[i] = VAL_MIN_RS_SC;
		maxLimit[i] = VAL_MAX_RS_SC;
	}

	minLimit[DIM_RS_SC - 1] = 1;
	maxLimit[DIM_RS_SC - 1] = N_IMP_RS_SC + 1 - 1e-6;

	return;
}

int CheckLimits_RS_SC(double* x, int nx)
{
	for (int i = 0; i < DIM_RS_SC - 1; i++) {
		if (x[i] < VAL_MIN_RS_SC || x[i] > VAL_MAX_RS_SC) {
			printf("%s(%d): Check limits FAIL - RecSys_SmartCity: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, i, x[i], VAL_MIN_RS_SC, VAL_MAX_RS_SC);
			return false;
		}
	}

	if (x[DIM_RS_SC - 1] < 1 || x[DIM_RS_SC - 1] > N_IMP_RS_SC + 1 - 1e-6) {
		printf("%s(%d): Check limits FAIL - RecSys_SmartCity: %d, %.16e not in [%.16e, %.16e]\n",
			__FILE__, __LINE__, DIM_RS_SC - 1, x[DIM_RS_SC - 1], 1.0, N_IMP_RS_SC + 1 - 1e-6);
		return false;
	}

	return true;
}

void Fitness_RS_SC(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	nImp_cur_RS_SC = (int)individual[DIM_RS_SC - 1];

	int offset_1 = 0;
	int length_1 = N_USR_RS_SC * N_IMP_RS_SC;
	int offset_2 = offset_1 + length_1;
	int length_2 = N_LOC_RS_SC * N_IMP_RS_SC;
	int offset_3 = offset_2 + length_2;
	int length_3 = N_ACT_RS_SC * N_IMP_RS_SC;
	int offset_4 = offset_3 + length_3;
	int length_4 = N_FEA_RS_SC * N_IMP_RS_SC;
	//int offset_5 = offset_4 + length_4;

	memcpy(ImpUsr_RS_SC, &individual[offset_1], length_1 * sizeof(double));
	memcpy(ImpLoc_RS_SC, &individual[offset_2], length_2 * sizeof(double));
	memcpy(ImpAct_RS_SC, &individual[offset_3], length_3 * sizeof(double));
	memcpy(ImpFea_RS_SC, &individual[offset_4], length_4 * sizeof(double));

	//////////////////////////////////////////////////////////////////////////
	double error_tensor = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (!trainFlag_RS_SC[i][j][k])
					continue;
				double tmp_sum = 0.0;
				double tar_val = UsrLocAct_RS_SC[i][j][k];
				for (int l = 0; l < nImp_cur_RS_SC; l++) {
					tmp_sum += ImpUsr_RS_SC[i][l] *
						ImpLoc_RS_SC[j][l] *
						ImpAct_RS_SC[k][l];
				}
				UsrLocAct_RS_SC_IR[i][j][k] = tmp_sum;
				error_tensor += (tmp_sum - tar_val) * (tmp_sum - tar_val);
			}
		}
	}
	error_tensor /= trainNum;
	error_tensor = sqrt(error_tensor);
	//////////////////////////////////////////////////////////////////////////
	double trace_UsrUsr = 0.0;
	double* tmp_mtrx1 = allocDouble_RS_SC(nImp_cur_RS_SC * N_USR_RS_SC);
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		for (int j = 0; j < N_USR_RS_SC; j++) {
			double tmp_sum = 0.0;
			for (int k = 0; k < N_USR_RS_SC; k++) {
				tmp_sum += ImpUsr_RS_SC[k][i] * Lap_UsrUsr_RS_SC[j][k];
			}
			tmp_mtrx1[i * N_USR_RS_SC + j] = tmp_sum;
		}
	}
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int k = 0; k < N_USR_RS_SC; k++) {
			tmp_sum += tmp_mtrx1[i * N_USR_RS_SC + k] * ImpUsr_RS_SC[k][i];
		}
		trace_UsrUsr += fabs(tmp_sum);
	}
	free(tmp_mtrx1);
	trace_UsrUsr /= nImp_cur_RS_SC;
	trace_UsrUsr = sqrt(trace_UsrUsr);
	//////////////////////////////////////////////////////////////////////////
	double error_LocFea = 0.0;
	for (int i = 0; i < N_LOC_RS_SC; i++) {
		for (int j = 0; j < N_FEA_RS_SC; j++) {
			double tmp_sum = 0.0;
			double tar_val = LocFea_RS_SC[i][j];
			for (int k = 0; k < nImp_cur_RS_SC; k++) {
				tmp_sum += ImpLoc_RS_SC[i][k] * ImpFea_RS_SC[j][k];
			}
			error_LocFea += (tmp_sum - tar_val) * (tmp_sum - tar_val);
		}
	}
	error_LocFea /= N_LOC_RS_SC * N_FEA_RS_SC;
	error_LocFea = sqrt(error_LocFea);
	//////////////////////////////////////////////////////////////////////////
	double trace_ActAct = 0.0;
	double* tmp_mtrx2 = allocDouble_RS_SC(nImp_cur_RS_SC * N_ACT_RS_SC);
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			double tmp_sum = 0.0;
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				tmp_sum += ImpAct_RS_SC[k][i] * Lap_ActAct_RS_SC[j][k];
			}
			tmp_mtrx2[i * N_ACT_RS_SC + j] = tmp_sum;
		}
	}
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int k = 0; k < N_ACT_RS_SC; k++) {
			tmp_sum += tmp_mtrx2[i * N_ACT_RS_SC + k] * ImpAct_RS_SC[k][i];
		}
		trace_ActAct += fabs(tmp_sum);
	}
	free(tmp_mtrx2);
	trace_ActAct /= nImp_cur_RS_SC;
	trace_ActAct = sqrt(trace_ActAct);
	//////////////////////////////////////////////////////////////////////////
	double error_UsrLoc = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			double tmp_sum = 0.0;
			double tar_val = UsrLoc_RS_SC[i][j];
			for (int k = 0; k < nImp_cur_RS_SC; k++) {
				tmp_sum += ImpUsr_RS_SC[i][k] * ImpLoc_RS_SC[j][k];
			}
			error_UsrLoc += (tmp_sum - tar_val) * (tmp_sum - tar_val);
		}
	}
	error_UsrLoc /= N_USR_RS_SC * N_LOC_RS_SC;
	error_UsrLoc = sqrt(error_UsrLoc);
	//////////////////////////////////////////////////////////////////////////
	double error_norm = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpUsr_RS_SC[i][j] * ImpUsr_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_LOC_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpLoc_RS_SC[i][j] * ImpLoc_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_ACT_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpAct_RS_SC[i][j] * ImpAct_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_FEA_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpFea_RS_SC[i][j] * ImpFea_RS_SC[i][j];
		}
	}
	error_norm /= N_USR_RS_SC * nImp_cur_RS_SC +
		N_LOC_RS_SC * nImp_cur_RS_SC +
		N_ACT_RS_SC * nImp_cur_RS_SC +
		N_FEA_RS_SC * nImp_cur_RS_SC;
	error_norm = sqrt(error_norm);
	//////////////////////////////////////////////////////////////////////////
	double nDCG_UsrLoc4Act = 0.0;
	int count_UsrLoc4Act = 0;
	int tmp_indAct[N_ACT_RS_SC];
	double tmp_valAct[N_ACT_RS_SC];
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			if (iDCG_UsrLoc4Act_train[i][j] <= 0)
				continue;
			count_UsrLoc4Act++;
			int tmp_count = 0;
			memset(tmp_valAct, 0, N_ACT_RS_SC * sizeof(double));
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (trainFlag_RS_SC[i][j][k]) {
					tmp_valAct[tmp_count] = UsrLocAct_RS_SC_IR[i][j][k];
					tmp_indAct[tmp_count] = k;
					tmp_count++;
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_valAct[a] < tmp_valAct[b]) {
						double tmp_d = tmp_valAct[a];
						tmp_valAct[a] = tmp_valAct[b];
						tmp_valAct[b] = tmp_d;
						int tmp_i = tmp_indAct[a];
						tmp_indAct[a] = tmp_indAct[b];
						tmp_indAct[b] = tmp_i;
					}
				}
			}
			double tmp_DCG = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC && a < tmp_count; a++) {
				if (a <= b_DCG - 1) {
					tmp_DCG += UsrLocAct_RS_SC[i][j][tmp_indAct[a]];
				}
				else {
					tmp_DCG += UsrLocAct_RS_SC[i][j][tmp_indAct[a]] / (log(a + 1) / log(b_DCG));
				}
			}
			nDCG_UsrLoc4Act += tmp_DCG / iDCG_UsrLoc4Act_train[i][j];
		}
	}
	nDCG_UsrLoc4Act /= count_UsrLoc4Act;
	//////////////////////////////////////////////////////////////////////////
	double nDCG_UsrAct4Loc = 0.0;
	double novelty_RS_SC = 0.0;
	double coverage_RS_SC = 0.0;
	int count_UsrAct4Loc = 0;
	int tmp_indLoc[N_LOC_RS_SC];
	double tmp_valLoc[N_LOC_RS_SC];
	int tmp_flag_Loc[N_LOC_RS_SC];
	memset(tmp_flag_Loc, 0, N_LOC_RS_SC * sizeof(int));
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			if (iDCG_UsrAct4Loc_train[i][j] <= 0)
				continue;
			count_UsrAct4Loc++;
			int tmp_count = 0;
			memset(tmp_valLoc, 0, N_LOC_RS_SC * sizeof(double));
			for (int k = 0; k < N_LOC_RS_SC; k++) {
				if (trainFlag_RS_SC[i][k][j]) {
					tmp_valLoc[tmp_count] = UsrLocAct_RS_SC_IR[i][k][j];
					tmp_indLoc[tmp_count] = k;
					tmp_count++;
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_valLoc[a] < tmp_valLoc[b]) {
						double tmp_d = tmp_valLoc[a];
						tmp_valLoc[a] = tmp_valLoc[b];
						tmp_valLoc[b] = tmp_d;
						int tmp_i = tmp_indLoc[a];
						tmp_indLoc[a] = tmp_indLoc[b];
						tmp_indLoc[b] = tmp_i;
					}
				}
			}
			double tmp_DCG = 0.0;
			double prod = 1.0;
			for (int a = 0; a < SIZE_LIST_RS_SC && a < tmp_count; a++) {
				if (a <= b_DCG - 1) {
					tmp_DCG += UsrLocAct_RS_SC[i][tmp_indLoc[a]][j];
				}
				else {
					tmp_DCG += UsrLocAct_RS_SC[i][tmp_indLoc[a]][j] / (log(a + 1) / log(b_DCG));
				}
				if (degree_Loc_train[tmp_indLoc[a]] == 0) {
					prod *= (N_USR_RS_SC / 0.5);
				}
				else {
					prod *= (N_USR_RS_SC / (double)degree_Loc_train[tmp_indLoc[a]]);
				}
				if (tmp_flag_Loc[tmp_indLoc[a]] == 0) {
					tmp_flag_Loc[tmp_indLoc[a]]++;
					coverage_RS_SC++;
				}
			}
			nDCG_UsrAct4Loc += tmp_DCG / iDCG_UsrAct4Loc_train[i][j];
			novelty_RS_SC += log(prod) / log(2.0) / SIZE_LIST_RS_SC;
		}
	}
	nDCG_UsrAct4Loc /= count_UsrAct4Loc;
	novelty_RS_SC /= N_USR_RS_SC * N_ACT_RS_SC;
	coverage_RS_SC /= N_LOC_RS_SC;

	//////////////////////////////////////////////////////////////////////////
	//double const_tensor = 100;
	//double const_UsrUsr = 100;
	//double const_LocFea = 40;
	//double const_ActAct = 5;
	//double const_UsrLoc = 25;
	//double const_norm = VAL_MAX_RS_SC;
	double const_novelty = 4.0;
	fitness[0] = error_tensor;// / const_tensor;
	fitness[1] = trace_UsrUsr;// / const_UsrUsr;
	fitness[2] = error_LocFea;// / const_LocFea;
	fitness[3] = trace_ActAct;// / const_ActAct;
	fitness[4] = error_UsrLoc;// / const_UsrLoc;
	fitness[5] = error_norm;// / const_norm;
	fitness[6] = 1.0 - nDCG_UsrAct4Loc;
	fitness[7] = 1.0 - nDCG_UsrLoc4Act;
	fitness[8] = 1.0 - novelty_RS_SC / const_novelty;
	fitness[9] = 1.0 - coverage_RS_SC;

	return;
}

void Fitness_RS_SC_testSet(double* individual, double* fitness)
{
	nImp_cur_RS_SC = (int)individual[DIM_RS_SC - 1];

	int offset_1 = 0;
	int length_1 = N_USR_RS_SC * N_IMP_RS_SC;
	int offset_2 = offset_1 + length_1;
	int length_2 = N_LOC_RS_SC * N_IMP_RS_SC;
	int offset_3 = offset_2 + length_2;
	int length_3 = N_ACT_RS_SC * N_IMP_RS_SC;
	int offset_4 = offset_3 + length_3;
	int length_4 = N_FEA_RS_SC * N_IMP_RS_SC;
	//int offset_5 = offset_4 + length_4;

	memcpy(ImpUsr_RS_SC, &individual[offset_1], length_1 * sizeof(double));
	memcpy(ImpLoc_RS_SC, &individual[offset_2], length_2 * sizeof(double));
	memcpy(ImpAct_RS_SC, &individual[offset_3], length_3 * sizeof(double));
	memcpy(ImpFea_RS_SC, &individual[offset_4], length_4 * sizeof(double));

	//////////////////////////////////////////////////////////////////////////
	double error_tensor = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (!testFlag_RS_SC[i][j][k])
					continue;
				double tmp_sum = 0.0;
				double tar_val = UsrLocAct_RS_SC[i][j][k];
				for (int l = 0; l < nImp_cur_RS_SC; l++) {
					tmp_sum += ImpUsr_RS_SC[i][l] *
						ImpLoc_RS_SC[j][l] *
						ImpAct_RS_SC[k][l];
				}
				UsrLocAct_RS_SC_IR[i][j][k] = tmp_sum;
				error_tensor += (tmp_sum - tar_val) * (tmp_sum - tar_val);
			}
		}
	}
	error_tensor /= testNum;
	error_tensor = sqrt(error_tensor);
	//////////////////////////////////////////////////////////////////////////
	double trace_UsrUsr = 0.0;
	double* tmp_mtrx1 = allocDouble_RS_SC(nImp_cur_RS_SC * N_USR_RS_SC);
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		for (int j = 0; j < N_USR_RS_SC; j++) {
			double tmp_sum = 0.0;
			for (int k = 0; k < N_USR_RS_SC; k++) {
				tmp_sum += ImpUsr_RS_SC[k][i] * Lap_UsrUsr_RS_SC[j][k];
			}
			tmp_mtrx1[i * N_USR_RS_SC + j] = tmp_sum;
		}
	}
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int k = 0; k < N_USR_RS_SC; k++) {
			tmp_sum += tmp_mtrx1[i * N_USR_RS_SC + k] * ImpUsr_RS_SC[k][i];
		}
		trace_UsrUsr += fabs(tmp_sum);
	}
	free(tmp_mtrx1);
	trace_UsrUsr /= nImp_cur_RS_SC;
	trace_UsrUsr = sqrt(trace_UsrUsr);
	//////////////////////////////////////////////////////////////////////////
	double error_LocFea = 0.0;
	for (int i = 0; i < N_LOC_RS_SC; i++) {
		for (int j = 0; j < N_FEA_RS_SC; j++) {
			double tmp_sum = 0.0;
			double tar_val = LocFea_RS_SC[i][j];
			for (int k = 0; k < nImp_cur_RS_SC; k++) {
				tmp_sum += ImpLoc_RS_SC[i][k] * ImpFea_RS_SC[j][k];
			}
			error_LocFea += (tmp_sum - tar_val) * (tmp_sum - tar_val);
		}
	}
	error_LocFea /= N_LOC_RS_SC * N_FEA_RS_SC;
	error_LocFea = sqrt(error_LocFea);
	//////////////////////////////////////////////////////////////////////////
	double trace_ActAct = 0.0;
	double* tmp_mtrx2 = allocDouble_RS_SC(nImp_cur_RS_SC * N_ACT_RS_SC);
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			double tmp_sum = 0.0;
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				tmp_sum += ImpAct_RS_SC[k][i] * Lap_ActAct_RS_SC[j][k];
			}
			tmp_mtrx2[i * N_ACT_RS_SC + j] = tmp_sum;
		}
	}
	for (int i = 0; i < nImp_cur_RS_SC; i++) {
		double tmp_sum = 0.0;
		for (int k = 0; k < N_ACT_RS_SC; k++) {
			tmp_sum += tmp_mtrx2[i * N_ACT_RS_SC + k] * ImpAct_RS_SC[k][i];
		}
		trace_ActAct += fabs(tmp_sum);
	}
	free(tmp_mtrx2);
	trace_ActAct /= nImp_cur_RS_SC;
	trace_ActAct = sqrt(trace_ActAct);
	//////////////////////////////////////////////////////////////////////////
	double error_UsrLoc = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			double tmp_sum = 0.0;
			double tar_val = UsrLoc_RS_SC[i][j];
			for (int k = 0; k < nImp_cur_RS_SC; k++) {
				tmp_sum += ImpUsr_RS_SC[i][k] * ImpLoc_RS_SC[j][k];
			}
			error_UsrLoc += (tmp_sum - tar_val) * (tmp_sum - tar_val);
		}
	}
	error_UsrLoc /= N_USR_RS_SC * N_LOC_RS_SC;
	error_UsrLoc = sqrt(error_UsrLoc);
	//////////////////////////////////////////////////////////////////////////
	double error_norm = 0.0;
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpUsr_RS_SC[i][j] * ImpUsr_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_LOC_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpLoc_RS_SC[i][j] * ImpLoc_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_ACT_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpAct_RS_SC[i][j] * ImpAct_RS_SC[i][j];
		}
	}
	for (int i = 0; i < N_FEA_RS_SC; i++) {
		for (int j = 0; j < nImp_cur_RS_SC; j++) {
			error_norm += ImpFea_RS_SC[i][j] * ImpFea_RS_SC[i][j];
		}
	}
	error_norm /= N_USR_RS_SC * nImp_cur_RS_SC +
		N_LOC_RS_SC * nImp_cur_RS_SC +
		N_ACT_RS_SC * nImp_cur_RS_SC +
		N_FEA_RS_SC * nImp_cur_RS_SC;
	error_norm = sqrt(error_norm);
	//////////////////////////////////////////////////////////////////////////
	double nDCG_UsrLoc4Act = 0.0;
	int count_UsrLoc4Act = 0;
	int tmp_indAct[N_ACT_RS_SC];
	double tmp_valAct[N_ACT_RS_SC];
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_LOC_RS_SC; j++) {
			if (iDCG_UsrLoc4Act_test[i][j] <= 0)
				continue;
			count_UsrLoc4Act++;
			int tmp_count = 0;
			memset(tmp_valAct, 0, N_ACT_RS_SC * sizeof(double));
			for (int k = 0; k < N_ACT_RS_SC; k++) {
				if (testFlag_RS_SC[i][j][k]) {
					tmp_valAct[tmp_count] = UsrLocAct_RS_SC_IR[i][j][k];
					tmp_indAct[tmp_count] = k;
					tmp_count++;
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_valAct[a] < tmp_valAct[b]) {
						double tmp_d = tmp_valAct[a];
						tmp_valAct[a] = tmp_valAct[b];
						tmp_valAct[b] = tmp_d;
						int tmp_i = tmp_indAct[a];
						tmp_indAct[a] = tmp_indAct[b];
						tmp_indAct[b] = tmp_i;
					}
				}
			}
			double tmp_DCG = 0.0;
			for (int a = 0; a < SIZE_LIST_RS_SC && a < tmp_count; a++) {
				if (a <= b_DCG - 1) {
					tmp_DCG += UsrLocAct_RS_SC[i][j][tmp_indAct[a]];
				}
				else {
					tmp_DCG += UsrLocAct_RS_SC[i][j][tmp_indAct[a]] / (log(a + 1) / log(b_DCG));
				}
			}
			nDCG_UsrLoc4Act += tmp_DCG / iDCG_UsrLoc4Act_test[i][j];
		}
	}
	nDCG_UsrLoc4Act /= count_UsrLoc4Act;
	//////////////////////////////////////////////////////////////////////////
	double nDCG_UsrAct4Loc = 0.0;
	double novelty_RS_SC = 0.0;
	double coverage_RS_SC = 0.0;
	int count_UsrAct4Loc = 0;
	int tmp_indLoc[N_LOC_RS_SC];
	double tmp_valLoc[N_LOC_RS_SC];
	int tmp_flag_Loc[N_LOC_RS_SC];
	memset(tmp_flag_Loc, 0, N_LOC_RS_SC * sizeof(int));
	for (int i = 0; i < N_USR_RS_SC; i++) {
		for (int j = 0; j < N_ACT_RS_SC; j++) {
			if (iDCG_UsrAct4Loc_test[i][j] <= 0)
				continue;
			count_UsrAct4Loc++;
			int tmp_count = 0;
			memset(tmp_valLoc, 0, N_LOC_RS_SC * sizeof(double));
			for (int k = 0; k < N_LOC_RS_SC; k++) {
				if (testFlag_RS_SC[i][k][j]) {
					tmp_valLoc[tmp_count] = UsrLocAct_RS_SC_IR[i][k][j];
					tmp_indLoc[tmp_count] = k;
					tmp_count++;
				}
			}
			for (int a = 0; a < SIZE_LIST_RS_SC; a++) {
				for (int b = a + 1; b < tmp_count; b++) {
					if (tmp_valLoc[a] < tmp_valLoc[b]) {
						double tmp_d = tmp_valLoc[a];
						tmp_valLoc[a] = tmp_valLoc[b];
						tmp_valLoc[b] = tmp_d;
						int tmp_i = tmp_indLoc[a];
						tmp_indLoc[a] = tmp_indLoc[b];
						tmp_indLoc[b] = tmp_i;
					}
				}
			}
			double tmp_DCG = 0.0;
			double prod = 1.0;
			for (int a = 0; a < SIZE_LIST_RS_SC && a < tmp_count; a++) {
				if (a <= b_DCG - 1) {
					tmp_DCG += UsrLocAct_RS_SC[i][tmp_indLoc[a]][j];
				}
				else {
					tmp_DCG += UsrLocAct_RS_SC[i][tmp_indLoc[a]][j] / (log(a + 1) / log(b_DCG));
				}
				if (degree_Loc_test[tmp_indLoc[a]] == 0) {
					prod *= (N_USR_RS_SC / 0.5);
				}
				else {
					prod *= (N_USR_RS_SC / (double)degree_Loc_test[tmp_indLoc[a]]);
				}
				if (tmp_flag_Loc[tmp_indLoc[a]] == 0) {
					tmp_flag_Loc[tmp_indLoc[a]]++;
					coverage_RS_SC++;
				}
			}
			nDCG_UsrAct4Loc += tmp_DCG / iDCG_UsrAct4Loc_test[i][j];
			novelty_RS_SC += log(prod) / log(2.0) / SIZE_LIST_RS_SC;
		}
	}
	nDCG_UsrAct4Loc /= count_UsrAct4Loc;
	novelty_RS_SC /= N_USR_RS_SC * N_ACT_RS_SC;
	coverage_RS_SC /= N_LOC_RS_SC;

	//////////////////////////////////////////////////////////////////////////
	//double const_tensor = 100;
	//double const_UsrUsr = 100;
	//double const_LocFea = 40;
	//double const_ActAct = 5;
	//double const_UsrLoc = 25;
	//double const_norm = VAL_MAX_RS_SC;
	double const_novelty = 4.0;
	fitness[0] = error_tensor;// / const_tensor;
	fitness[1] = trace_UsrUsr;// / const_UsrUsr;
	fitness[2] = error_LocFea;// / const_LocFea;
	fitness[3] = trace_ActAct;// / const_ActAct;
	fitness[4] = error_UsrLoc;// / const_UsrLoc;
	fitness[5] = error_norm;// / const_norm;
	fitness[6] = 1.0 - nDCG_UsrAct4Loc;
	fitness[7] = 1.0 - nDCG_UsrLoc4Act;
	fitness[8] = 1.0 - novelty_RS_SC / const_novelty;
	fitness[9] = 1.0 - coverage_RS_SC;

	return;
}

//the random generator in [0,1)
static double rnd_uni_gen_RS_SC(long* idum, double vmin, double vmax)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB_RS_SC];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB_RS_SC + 7; j >= 0; j--) {
			k = (*idum) / IQ1_RS_SC;
			*idum = IA1_RS_SC * (*idum - k * IQ1_RS_SC) - k * IR1_RS_SC;
			if (*idum < 0) *idum += IM1_RS_SC;
			if (j < NTAB_RS_SC) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1_RS_SC;
	*idum = IA1_RS_SC * (*idum - k * IQ1_RS_SC) - k * IR1_RS_SC;
	if (*idum < 0) *idum += IM1_RS_SC;
	k = idum2 / IQ2_RS_SC;
	idum2 = IA2_RS_SC * (idum2 - k * IQ2_RS_SC) - k * IR2_RS_SC;
	if (idum2 < 0) idum2 += IM2_RS_SC;
	j = iy / NDIV_RS_SC;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1_RS_SC;
	if ((temp = AM_RS_SC * iy) > RNMX_RS_SC) return (vmin + RNMX_RS_SC * (vmax - vmin));
	else return (vmin + temp * (vmax - vmin));
}/*------End of rnd_uni_CLASS()--------------------------*/

/* Fisher¨CYates shuffle algorithm */
static void shuffle_RS_SC(int* x, int size)
{
	int i, aux, k = 0;
	for (i = size - 1; i > 0; i--) {
		/* get a value between zero and i  */
		k = (int)rnd_uni_gen_RS_SC(&rnd_uni_init_val_RS_SC, 0.0, i + 1 - 1e-6);
		/* exchange of values */
		aux = x[i];
		x[i] = x[k];
		x[k] = aux;
	}
}

static double* allocDouble_RS_SC(int size)
{
	double* tmp;
	if ((tmp = (double*)calloc(size, sizeof(double))) == NULL) {
		printf("ERROR!! --> calloc: no memory for vector\n");
		exit(1);
	}
	return tmp;
}

// static int *allocInt_RS_SC(int size)
// {
//     int *tmp;
//     if ((tmp = (int *)calloc(size, sizeof(int))) == NULL) {
//         printf("ERROR!! --> calloc: no memory for vector\n");
//         exit(1);
//     }
//     return tmp;
// }

// static void readData_RS_SC(int* &MEM, char filename[], int* N1, int* N2, int* N3)
// {
//     //printf("%s\n", filename);
//     int theSize[3] = { 0, 0, 0 };
//     FILE* fpt;
//     if ((fpt = fopen(filename, "r")) == NULL) {
//         printf("%s(%d): File open error!\n", __FILE__, __LINE__);
//         exit(10000);
//     }
//     int tmp;
//     int dim;
//     int elem;
//     tmp = fscanf(fpt, "%d", &dim);
//     if (tmp == EOF) {
//         printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//         exit(2000);
//     }
//     for (int i = 0; i < dim; i++) {
//         tmp = fscanf(fpt, "%d", &theSize[i]);
//         if (tmp == EOF) {
//             printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//             exit(2000);
//         }
//     }
//     for (int i = 0; i < dim; i++) {
//         switch (i) {
//         case 0:
//             if (N1) N1[0] = theSize[0];
//             break;
//         case 1:
//             if (N2) N2[0] = theSize[1];
//             break;
//         case 2:
//             if (N3) N3[0] = theSize[2];
//             break;
//         default:
//             break;
//         }
//     }
//     if (theSize[2] == 0) theSize[2]++;
//     MEM = allocInt_RS_SC(theSize[0] * theSize[1] * theSize[2]);
//     for (int i = 0; i < theSize[0]; i++) {
//         for (int j = 0; j < theSize[1]; j++) {
//             for (int k = 0; k < theSize[2]; k++) {
//                 tmp = fscanf(fpt, "%d", &elem);
//                 if (tmp == EOF) {
//                     printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//                     exit(2000);
//                 }
//                 MEM[i * theSize[1] * theSize[2] + j * theSize[2] + k] = elem;
//             }
//         }
//     }
//     fclose(fpt);
//     for (int i = 0; i < theSize[0]; i++) {
//         for (int j = 0; j < theSize[1]; j++) {
//             for (int k = 0; k < theSize[2]; k++) {
//                 printf("%d ", MEM[i * theSize[1] * theSize[2] + j * theSize[2] + k]);
//             }
//             printf("\n");
//         }
//     }
//     printf("\n");
//     printf("\n");

//     return;
// }

// static void readData_RS_SC(double* &MEM, char filename[], int* N1, int* N2, int* N3)
// {
//     //printf("%s\n", filename);
//     int theSize[3] = { 0, 0, 0 };
//     FILE* fpt;
//     if ((fpt = fopen(filename, "r")) == NULL) {
//         printf("%s(%d): File open error!\n", __FILE__, __LINE__);
//         exit(10000);
//     }
//     int tmp;
//     int dim;
//     double elem;
//     tmp = fscanf(fpt, "%d", &dim);
//     if (tmp == EOF) {
//         printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//         exit(2000);
//     }
//     for (int i = 0; i < dim; i++) {
//         tmp = fscanf(fpt, "%d", &theSize[i]);
//         if (tmp == EOF) {
//             printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//             exit(2000);
//         }
//     }
//     for (int i = 0; i < dim; i++) {
//         switch (i) {
//         case 0:
//             if (N1) N1[0] = theSize[0];
//             break;
//         case 1:
//             if (N2) N2[0] = theSize[1];
//             break;
//         case 2:
//             if (N3) N3[0] = theSize[2];
//             break;
//         default:
//             break;
//         }
//     }
//     if (theSize[2] == 0) theSize[2]++;
//     MEM = allocDouble_RS_SC(theSize[0] * theSize[1] * theSize[2]);
//     for (int i = 0; i < theSize[0]; i++) {
//         for (int j = 0; j < theSize[1]; j++) {
//             for (int k = 0; k < theSize[2]; k++) {
//                 tmp = fscanf(fpt, "%le", &elem);
//                 if (tmp == EOF) {
//                     printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
//                     exit(2000);
//                 }
//                 MEM[i * theSize[1] * theSize[2] + j * theSize[2] + k] = elem;
//             }
//         }
//     }
//     fclose(fpt);
//     for (int i = 0; i < theSize[0]; i++) {
//         for (int j = 0; j < theSize[1]; j++) {
//             for (int k = 0; k < theSize[2]; k++) {
//                 printf("%.16e ", MEM[i * theSize[1] * theSize[2] + j * theSize[2] + k]);
//             }
//             printf("\n");
//         }
//     }
//     printf("\n");
//     printf("\n");

//     return;
// }