#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOP_ARRANGE2D.h"

int N_ATTR_ARRANGE2D;

int NDIM_ARRANGE2D;// (N_ATTR_ARRANGE2D)
int NOBJ_ARRANGE2D;//  2

int cur_run_fold_ARRANGE2D;
int num_run_fold_ARRANGE2D;

#define LENGTH_MAT_ATTR_ARRANGE2D (N_ROW_ARRANGE2D*N_COL_ARRANGE2D)

double mat_cor[LENGTH_MAT_ATTR_ARRANGE2D][LENGTH_MAT_ATTR_ARRANGE2D];
int    mat_att[N_ROW_ARRANGE2D][N_COL_ARRANGE2D];
double all_cor[LENGTH_MAT_ATTR_ARRANGE2D];

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define IM1_ARRANGE2D 2147483563
#define IM2_ARRANGE2D 2147483399
#define AM_ARRANGE2D (1.0/IM1_ARRANGE2D)
#define IMM1_ARRANGE2D (IM1_ARRANGE2D-1)
#define IA1_ARRANGE2D 40014
#define IA2_ARRANGE2D 40692
#define IQ1_ARRANGE2D 53668
#define IQ2_ARRANGE2D 52774
#define IR1_ARRANGE2D 12211
#define IR2_ARRANGE2D 3791
#define NTAB_ARRANGE2D 32
#define NDIV_ARRANGE2D (1+IMM1_ARRANGE2D/NTAB_ARRANGE2D)
#define EPS_ARRANGE2D 1.2e-7
#define RNMX_ARRANGE2D (1.0-EPS_ARRANGE2D)

//the random generator in [0,1)
double rnd_uni_ARRANGE2D(long* idum)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB_ARRANGE2D];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB_ARRANGE2D + 7; j >= 0; j--) {
			k = (*idum) / IQ1_ARRANGE2D;
			*idum = IA1_ARRANGE2D * (*idum - k * IQ1_ARRANGE2D) - k * IR1_ARRANGE2D;
			if (*idum < 0) *idum += IM1_ARRANGE2D;
			if (j < NTAB_ARRANGE2D) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1_ARRANGE2D;
	*idum = IA1_ARRANGE2D * (*idum - k * IQ1_ARRANGE2D) - k * IR1_ARRANGE2D;
	if (*idum < 0) *idum += IM1_ARRANGE2D;
	k = idum2 / IQ2_ARRANGE2D;
	idum2 = IA2_ARRANGE2D * (idum2 - k * IQ2_ARRANGE2D) - k * IR2_ARRANGE2D;
	if (idum2 < 0) idum2 += IM2_ARRANGE2D;
	j = iy / NDIV_ARRANGE2D;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1_ARRANGE2D;   //printf("%lf\n", AM_ARRANGE2D*iy);
	if ((temp = AM_ARRANGE2D * iy) > RNMX_ARRANGE2D) return RNMX_ARRANGE2D;
	else return temp;
}/*------End of rnd_uni_ARRANGE2D()--------------------------*/
int     seed_ARRANGE2D = 237;
long    rnd_uni_init_ARRANGE2D = -(long)seed_ARRANGE2D;

void Initialize_data_ARRANGE2D(int curN, int numN)
{
	seed_ARRANGE2D = 237;
	rnd_uni_init_ARRANGE2D = -(long)seed_ARRANGE2D;
	for (int i = 0; i < curN; i++) {
		seed_ARRANGE2D = (seed_ARRANGE2D + 111) % 1235;
		rnd_uni_init_ARRANGE2D = -(long)seed_ARRANGE2D;
	}
	cur_run_fold_ARRANGE2D = curN;
	num_run_fold_ARRANGE2D = numN;
	//
	char fname[1024];
	FILE* fp = NULL;
	sprintf(fname, "../Data_all/Data_CNN_Indus/SECOM_num_feature_F%d", curN + 1);
	if ((fp = fopen(fname, "r")) != NULL) {
		int tmpVal;
		int tmp = fscanf(fp, "%d", &tmpVal);
		if (tmp == EOF) {
			printf("\n%s(%d):weights are not enough...\n", __FILE__, __LINE__);
			exit(9);
		}
		N_ATTR_ARRANGE2D = tmpVal;
		fclose(fp);
		fp = NULL;
	}
	else {
		printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, fname);
		exit(-1);
	}
	//
	sprintf(fname, "../Data_all/Data_CNN_Indus/SECOM_cor_pearson_F%d", curN + 1);
	if ((fp = fopen(fname, "r")) != NULL) {
		double tmpVal;
		for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
			for (int j = 0; j < N_ATTR_ARRANGE2D; j++) {
				int tmp = fscanf(fp, "%lf", &tmpVal);
				if (tmp == EOF) {
					printf("\n%s(%d):weights are not enough...\n", __FILE__, __LINE__);
					exit(9);
				}
				mat_cor[i][j] = fabs(tmpVal);
			}
		}
		fclose(fp);
		fp = NULL;
	}
	else {
		printf("%s(%d): Open file %s error, exiting...\n", __FILE__, __LINE__, fname);
		exit(-1);
	}
	//
	NDIM_ARRANGE2D = N_ATTR_ARRANGE2D;
	NOBJ_ARRANGE2D = 2;
	//
	return;
}

void SetLimits_ARRANGE2D(double* minLimit, double* maxLimit, int nx)
{
	for (int i = 0; i < NDIM_ARRANGE2D; i++) {
		minLimit[i] = 0.0;
		maxLimit[i] = 1.0;
	}
	return;
}

int  CheckLimits_ARRANGE2D(double* x, int nx)
{
	for (int i = 0; i < NDIM_ARRANGE2D; i++) {
		if (x[i] < 0.0 || x[i] > 1.0) {
			printf("%s(%d): Check limits FAIL - ARRANGE2D: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, i, x[i], 0.0, 1.0);
			return false;
		}
	}
	return true;
}

//
int rnd_ARRANGE2D(int low, int high)
{
	int res;
	if (low >= high) {
		res = low;
	}
	else {
		res = low + (int)(rnd_uni_ARRANGE2D(&rnd_uni_init_ARRANGE2D) * (high - low + 1));
		if (res > high) {
			res = high;
		}
	}
	return (res);
}

void qSortGeneral_ARRANGE2D(double* data, int arrayFx[], int left, int right)
{
	int index;
	int temp;
	int i, j;
	double pivot;
	if (left < right) {
		index = rnd_ARRANGE2D(left, right);
		temp = arrayFx[right];
		arrayFx[right] = arrayFx[index];
		arrayFx[index] = temp;
		pivot = data[arrayFx[right]];
		i = left - 1;
		for (j = left; j < right; j++) {
			if (data[arrayFx[j]] >= pivot) {
				i += 1;
				temp = arrayFx[j];
				arrayFx[j] = arrayFx[i];
				arrayFx[i] = temp;
			}
		}
		index = i + 1;
		temp = arrayFx[index];
		arrayFx[index] = arrayFx[right];
		arrayFx[right] = temp;
		qSortGeneral_ARRANGE2D(data, arrayFx, left, index - 1);
		qSortGeneral_ARRANGE2D(data, arrayFx, index + 1, right);
	}
	return;
}

void bubbleSort_ARRANGE2D(double* data, int arrayFx[], int len)
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

void Fitness_ARRANGE2D3OBJ(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	static int tmp_count = 1;

	int the_index[LENGTH_MAT_ATTR_ARRANGE2D];
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		the_index[i] = i;
	}
	//qSortGeneral_ARRANGE2D(individual, the_index, 0, N_ATTR_ARRANGE2D - 1);
	bubbleSort_ARRANGE2D(individual, the_index, N_ATTR_ARRANGE2D);

	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		int the_r = i / N_COL_ARRANGE2D;
		int the_c = i - the_r * N_COL_ARRANGE2D;
		mat_att[the_r][the_c] = the_index[i];
	}

	//
	char fname[128];
	sprintf(fname, "tmp/tmp%03d", tmp_count++);
	FILE* fpt = fopen(fname, "w");

	double mean_cor[3] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		double tmp_cor[3] = { 0.0, 0.0, 0.0 };
		int    tmp_cnt[3] = { 0, 0, 0 };
		int the_r = i / N_COL_ARRANGE2D;
		int the_c = i - the_r * N_COL_ARRANGE2D;
		for (int a = -3; a <= 3; a++) {
			for (int b = -3; b <= 3; b++) {
				int t_r = the_r + a;
				int t_c = the_c + b;
				int max_d = abs(a) > abs(b) ? abs(a) : abs(b);
				int t_ind = t_r * N_COL_ARRANGE2D + t_c;
				if (max_d == 0 ||
					t_r < 0 || t_r >= N_ROW_ARRANGE2D ||
					t_c < 0 || t_c >= N_COL_ARRANGE2D ||
					t_ind >= N_ATTR_ARRANGE2D)
					continue;
				double the_cor = mat_cor[mat_att[the_r][the_c]][mat_att[t_r][t_c]];
				for (int c = max_d - 1; c < 3; c++) {
					tmp_cor[c] += the_cor;
					tmp_cnt[c]++;
				}
			}
		}
		int cur_d1 = abs(the_r - N_ROW_ARRANGE2D / 2);
		int cur_d2 = abs(the_c - N_COL_ARRANGE2D / 2);
		int cur_d = cur_d1;
		if (cur_d < cur_d2) cur_d = cur_d2;
		cur_d++;
		for (int j = 0; j < 3; j++) {
			mean_cor[j] += tmp_cor[j] / tmp_cnt[j] / cur_d;
		}
		fprintf(fpt, "%d %lf %lf %lf\n", the_index[i] + 1,
			tmp_cor[0] / tmp_cnt[0], tmp_cor[1] / tmp_cnt[1], tmp_cor[2] / tmp_cnt[2]);
	}
	fclose(fpt);
	//
	for (int i = 0; i < NOBJ_ARRANGE2D; i++) {
		fitness[i] = 1.0 - mean_cor[i] / N_ATTR_ARRANGE2D;
	}
	//
	return;
}

void Fitness_ARRANGE2D(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	int the_index[LENGTH_MAT_ATTR_ARRANGE2D];
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		the_index[i] = i;
	}
	//qSortGeneral_ARRANGE2D(individual, the_index, 0, N_ATTR_ARRANGE2D - 1);
	bubbleSort_ARRANGE2D(individual, the_index, N_ATTR_ARRANGE2D);
	//
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		int the_r = i / N_COL_ARRANGE2D;
		int the_c = i - the_r * N_COL_ARRANGE2D;
		mat_att[the_r][the_c] = the_index[i];
	}
	//
	//static int tmp_count = 1;
	//if(tmp_count == 801) {
	//    int ccoouunntt = 1;
	//}
	//char fname[128];
	//sprintf(fname, "tmp/tmp_cor%04d", tmp_count);
	//FILE* fpt = fopen(fname, "w");
	////
	//sprintf(fname, "tmp/tmp_ind%04d", tmp_count);
	//tmp_count++;
	//FILE* fpt2 = fopen(fname, "w");
	//for(int i = 0; i < N_ROW_ARRANGE2D; i++) {
	//    for(int j = 0; j < N_COL_ARRANGE2D; j++) {
	//        if(i * N_COL_ARRANGE2D + j < N_ATTR_ARRANGE2D)
	//            fprintf(fpt2, "%d ", mat_att[i][j]);
	//        else
	//            fprintf(fpt2, "%d ", -1);
	//    }
	//    fprintf(fpt2, "\n");
	//}
	//fclose(fpt2);
	//
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		all_cor[i] = 0.0;
		int tmp_cnt = 0;
		int the_r1 = i / N_COL_ARRANGE2D;
		int the_c1 = i - the_r1 * N_COL_ARRANGE2D;
		for (int j = 0; j < N_ATTR_ARRANGE2D; j++) {
			if (i == j) continue;
			int the_r2 = j / N_COL_ARRANGE2D;
			int the_c2 = j - the_r2 * N_COL_ARRANGE2D;
			double the_cor = mat_cor[the_index[i]][the_index[j]];
			double cur_d1 = abs(the_r1 - the_r2);
			double cur_d2 = abs(the_c1 - the_c2);
			//double cur_d = cur_d1;
			//if(cur_d < cur_d2) cur_d = cur_d2;
			double cur_d = sqrt(cur_d1 * cur_d1 + cur_d2 * cur_d2);
			if (cur_d < 0.45) cur_d = 0.5;
			all_cor[i] += the_cor / cur_d;
			tmp_cnt++;
		}
		all_cor[i] /= tmp_cnt;
		//fprintf(fpt, "%d %lf\n", the_index[i] + 1, all_cor[i]);
	}
	//fclose(fpt);
	//
	double fit_mean1 = 0.0;
	double fit_mean2 = 0.0;
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		fit_mean1 += all_cor[i];
	}
	fit_mean1 /= N_ATTR_ARRANGE2D;
	double cen_r = (N_ROW_ARRANGE2D - 1) / 2.0;
	double cen_c = (N_COL_ARRANGE2D - 1) / 2.0;
	for (int i = 0; i < N_ATTR_ARRANGE2D; i++) {
		int the_r = i / N_COL_ARRANGE2D;
		int the_c = i - the_r * N_COL_ARRANGE2D;
		double cur_d1 = fabs(the_r - cen_r);
		double cur_d2 = fabs(the_c - cen_c);
		//double cur_d = cur_d1;
		//if(cur_d < cur_d2) cur_d = cur_d2;
		double cur_d = sqrt(cur_d1 * cur_d1 + cur_d2 * cur_d2);
		if (cur_d < 0.45) cur_d = 0.5;
		fit_mean2 += all_cor[i] / cur_d;
	}
	fit_mean2 /= N_ATTR_ARRANGE2D;
	//
	fitness[0] = 1.0 - fit_mean1;
	fitness[1] = 1.0 - fit_mean2;
	//
	return;
}