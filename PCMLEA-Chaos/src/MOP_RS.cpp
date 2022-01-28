#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "MOP_RS.h"

#define MAX_BUF_SIZE 1000000 //
#define MAX_STR_LEN  256 //

#define sigA 1.0 //sigmoid func a
#define sigC 0.0 //sigmoid func c

#define D_FEATURE 10 //feature number

int DIM_RS;

double* M_ITEM_Feature; //feature data of all items
int     m_M_ITEM_Feature, n_M_ITEM_Feature; //item number dim=D_FEATURE

double* M_PR;
int     m_M_PR, n_M_PR; //numbers of users and items
double  max_PR;

int* Tag_recommendation; //tags of recommended items
int     n_recommended;
int     n_invalid;

double* MMM_similarity;
double* Degree_ITEM;

// static double precision(double *x); //recommendation precision, based on train set
static double precision2(double* x); //recommendation precision
// static double similarity(double *x); //
static double novelty(double* x); //

// static void readData(int* &MEM, char filename[], int &m, int &n); //
static void readData(double*& MEM, char filename[], int& m, int& n); //
static void getSize(char filename[], int& m, int& n); //
static double* allocDouble(int size); //
static int* allocInt(int size); //
// static void shuffle(int *x, int size); //
// static double sigmoidF(double x, double a = sigA, double c = sigC); //
// static void sortDescend(double *x, int *ind, int n, int m); //
// static double cosineSimilarity(double *x, double *y, int len);
static int isValid(double* x, int len, int max);

/*
int main()
{
loadData();

srand(time(NULL));
double x[m_M_PR * D_FEATURE];
int i, j;
for (i = 0; i < m_M_PR; i++) {
for (j = 0; j < D_FEATURE; j++) {
x[i * D_FEATURE + j] = (rand()%RAND_MAX) / (RAND_MAX + 0.0);
}
}
double y[2];

getFitnessRS(x, y); printf("%lf %lf\n", y[0], y[1]);

freeData();

return 0;
}
*/

void SetLimitsRS(double* minL, double* maxL, int nx)
{
	int i, j;

	for (i = 0; i < m_M_PR; i++) {
		for (j = 0; j < D_FEATURE; j++) {
			minL[i * D_FEATURE + j] = 0.0;
			maxL[i * D_FEATURE + j] = 1.0;
		}
	}

	return;
}

int CheckLimitsRS(double* x, int nx)
{
	int i, j;

	for (i = 0; i < m_M_PR; i++) {
		for (j = 0; j < D_FEATURE; j++) {
			if (x[i * D_FEATURE + j] < 0.0) {
				printf("Check limit error, %d %d %lf, below -1.0, exiting...\n", i, j, x[i * D_FEATURE + j]);
				return 0;
			}
			if (x[i * D_FEATURE + j] > 1.0) {
				printf("Check limit error, %d %d %lf, above  1.0, exiting...\n", i, j, x[i * D_FEATURE + j]);
				return 0;
			}
		}
	}

	return 1;
}

void getFitnessRS(double* x, double* y, double* constrainV, int nx, int M)
{
	double tmp = precision2(x);
	y[0] = 0.0 - tmp +
		(n_M_PR - n_recommended + n_invalid) * 1e6; //if (n_invalid) printf("%d ", n_invalid);
	y[1] = 0.0 - novelty(x) +
		(n_M_PR - n_recommended + n_invalid) * 1e6; //printf("c\n");

	return;
}

// static double precision(double *x)
// {
//     int i, j, k;
//     double fit = 0.0;
//     double *result = (double*)calloc(m_M_PR, sizeof(double));
//     for (i = 0; i < m_M_PR; i++)//all users
//         result[i] = 0.0;
//     for (i = 0; i < m_M_ITEM_Feature; i++)//all items
//         Tag_recommendation[i] = 0;

//     for (i = 0; i < m_M_PR; i++) {//for each user
//         //printf("%d ", i);
//         int n = m_M_ITEM_Feature;
//         double *predict = (double*)calloc(n, sizeof(double));
//         int *index = (int*)calloc(n, sizeof(int));
//         for (j = 0; j < n; j++) index[j] = j;
//         for (j = 0; j < n; j++) {//for all items
//             double sum = 0.0;
//             for (k = 0; k < D_FEATURE; k++) {
//                 //printf("%lf %lf\n", x[i * D_FEATURE + k], M_ITEM_Feature[j * D_FEATURE + k]);
//                 sum += x[i * D_FEATURE + k] *
//                        M_ITEM_Feature[j * D_FEATURE + k];
//             }
//             predict[j] = sum;
//         }
//         sortDescend(predict, index, n, NUM_RECOMMEND);
//         double tmp = 0;
//         for (j = 0; j < NUM_RECOMMEND; j++) {
//             //printf("%lf ", predict[j]);
//             tmp += M_PR[i * n + index[j]];
//             Tag_recommendation[index[j]]++;//record the recommended items
//         }
//         //printf("\n");
//         result[i] = tmp;
//         fit += result[i];

//         free(predict);
//         free(index);
//     }

//     n_recommended = 0;
//     for (i = 0; i < n_M_PR; i++) {
//         if (Tag_recommendation[i])
//             n_recommended++;
//     }

//     free(result);

//     return fit / m_M_PR;
// }

static double precision2(double* x)
{
	int i, j;
	double fit = 0.0;
	double* result = (double*)calloc(m_M_PR, sizeof(double));
	for (i = 0; i < m_M_PR; i++) //all users
		result[i] = 0.0;
	for (i = 0; i < n_M_PR; i++) //all items
		Tag_recommendation[i] = 0;

	n_invalid = 0;

	for (i = 0; i < m_M_PR; i++) { //for each user
		//printf("%d ", i);
		int n = n_M_PR;
		double tmp = 0;
		for (j = 0; j < NUM_RECOMMEND; j++) {
			int curInd = ((int)(x[i * NUM_RECOMMEND + j] * n)) % n;
			tmp += M_PR[i * n + curInd];
			Tag_recommendation[curInd]++;//record the recommended items
		}
		//printf("\n");
		result[i] = tmp;
		fit += result[i];

		if (!isValid(&x[i * NUM_RECOMMEND], NUM_RECOMMEND, n))
			n_invalid++;
	}

	n_recommended = 0;
	for (i = 0; i < n_M_PR; i++) {
		if (Tag_recommendation[i])
			n_recommended++;
	}

	free(result);

	return fit / m_M_PR;
}

/*static double similarity(double *x)
{
int i, j, k;
double a;
int N = m_M_PR; //user number
int n = NUM_RECOMMEND;
int m1, m2;

a = 0.0;
for (i = 0; i < N; i++) {
for (j = 0; j < n; j++) {
m1 = (int)(x[i * NUM_RECOMMEND + j] * n_M_PR) % n_M_PR;
for (k = j + 1; k < n; k++) {
m2 = (int)(x[i * NUM_RECOMMEND + k] * n_M_PR) % n_M_PR;
a += MMM_similarity[m1 * D_FEATURE + m2];
}
}
}

return (a / ((n * (n - 1)) / 2) / N);
}*/

static double novelty(double* x)
{
	int i, j;
	int N = m_M_PR;
	int n = NUM_RECOMMEND;
	double a = 0.0;
	int curI;
	double prod;

	for (i = 0; i < N; i++) {
		prod = 1.0;
		for (j = 0; j < n; j++) {
			curI = (int)(x[i * n + j] * n_M_PR) % n_M_PR;
			prod *= (N / Degree_ITEM[curI]);
		}
		a += log(prod) / log(2.0) / n;
	}

	return (a / N);
}

void loadData()
{
	char filename[MAX_STR_LEN];

	//Item feature data
	sprintf(filename, "%s", "../Data_all/Data_RS/PRV");
	readData(M_ITEM_Feature, filename, m_M_ITEM_Feature, n_M_ITEM_Feature);
	MMM_similarity = allocDouble(m_M_ITEM_Feature * m_M_ITEM_Feature);
	/*for (int i = 0; i < m_M_ITEM_Feature; i++) { //calculate similarity
		MMM_similarity[i * m_M_ITEM_Feature + i] = 0.0;
		for (int j = i + 1; j < m_M_ITEM_Feature; j++) {
		MMM_similarity[i * m_M_ITEM_Feature + j] = MMM_similarity[j * m_M_ITEM_Feature + i] =
		cosineSimilarity(&M_ITEM_Feature[i * D_FEATURE], &M_ITEM_Feature[j * D_FEATURE], D_FEATURE);
		}
		}*/
		//Prediction data
	sprintf(filename, "%s", "../Data_all/Data_RS/PR");
	readData(M_PR, filename, m_M_PR, n_M_PR);
	/*double tmp[m_M_PR * n_M_PR];
	memcpy(tmp, M_PR, sizeof(double)*m_M_PR * n_M_PR);
	sortDescend(tmp, NULL, m_M_PR * n_M_PR, NUM_RECOMMEND);
	int i, j;
	max_PR = 0.0;
	for (i = 0; i < NUM_RECOMMEND; i++) {
	max_PR += tmp[i] / NUM_RECOMMEND;
	}
	printf("%lf\n", max_PR);*/
	//Item degree
	sprintf(filename, "%s", "../Data_all/Data_RS/degree_i");
	int a, b;
	readData(Degree_ITEM, filename, a, b);

	Tag_recommendation = allocInt(m_M_ITEM_Feature);

	DIM_RS = m_M_PR * D_FEATURE;

	return;
}

// static void readData(int* &MEM, char filename[], int &m, int &n)
// {
//     //printf("%s\n", filename);
//     getSize(filename, m, n);
//     MEM = allocInt(m * n);
//     FILE* fpt;
//     fpt = fopen(filename, "r");
//     int i, j;
//     for (i = 0; i < m; i++) {
//         for (j = 0; j < n; j++) {
//             fscanf(fpt, "%d", &MEM[i * n + j]);
//         }
//     }
//     fclose(fpt);
//     /*for (i = 0; i < 2; i++) {
//         for (j = 0; j < n && j < 9; j++) {
//         printf("%d ", MEM[i * n + j]);
//         }
//         printf("\n");
//         }*/

//     return;
// }

static void readData(double*& MEM, char filename[], int& m, int& n)
{
	//printf("%s\n", filename);
	getSize(filename, m, n);
	MEM = allocDouble(m * n);
	FILE* fpt;
	fpt = fopen(filename, "r");
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (fscanf(fpt, "%lf", &MEM[i * n + j]) != 1) {
				printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
				exit(-1);
			}
		}
	}
	fclose(fpt);
	/*for (i = 0; i < 2; i++) {
		for (j = 0; j < n && j < 9; j++) {
		printf("%lf ", MEM[i * n + j]);
		}
		printf("\n");
		}*/

	return;
}

static double* allocDouble(int size)
{
	double* tmp;
	if ((tmp = (double*)calloc(size, sizeof(double))) == NULL) {
		printf("ERROR!! --> calloc: no memory for vector\n");
		exit(1);
	}
	return tmp;
}

static int* allocInt(int size)
{
	int* tmp;
	if ((tmp = (int*)calloc(size, sizeof(int))) == NULL) {
		printf("ERROR!! --> calloc: no memory for vector\n");
		exit(1);
	}
	return tmp;
}

void freeData()
{
	free(M_ITEM_Feature);
	free(MMM_similarity);
	free(M_PR);

	free(Tag_recommendation);
	free(Degree_ITEM);

	return;
}

static void getSize(char filename[], int& m, int& n)
{
	FILE* fpt = fopen(filename, "r");

	if (fpt) {
		char* buf = (char*)malloc(MAX_BUF_SIZE * sizeof(char));
		char* p;
		m = 0;
		n = 0;

		while (fgets(buf, MAX_BUF_SIZE, fpt)) {
			if (n == 0) {
				for (p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
					n++;
				}
			}
			m++;
		}
		// fgets(buf, MAX_BUF_SIZE, fpt);   //read one row
		// for (p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
		//     n++;
		// }
		// while (!feof(fpt)) {
		//     m++;//before fgets
		//     fgets(buf, MAX_BUF_SIZE, fpt);   //read one line to buf
		// }
		free(buf);
		fclose(fpt);
	}
	else {
		printf("Open file %s error, exiting...\n", filename);
		exit(-1);
	}

	return;
}

/* Fisherâ€“Yates shuffle algorithm */
// static void shuffle(int *x, int size)
// {
//     int i, aux, k = 0;
//     for (i = size - 1; i > 0; i--) {
//         /* get a value between cero and i  */
//         k = (int)(rand() / (RAND_MAX + 0.0) * i);
//         /* exchange of values */
//         aux = x[i];
//         x[i] = x[k];
//         x[k] = aux;
//     }
// }

// static double sigmoidF(double x, double a, double c)
// {
//     return 1.0 / (1.0 + exp(-1.0 * a * (x - c)));
// }

// static void sortDescend(double *x, int *ind, int n, int m)
// {
//     int i, j;

//     for (i = 0; i < m; i++) {
//         for (j = i + 1; j < n; j++) {
//             if (x[i] < x[j]) {
//                 double tmp = x[i];
//                 x[i] = x[j];
//                 x[j] = tmp;

//                 if (ind) {
//                     int tmp2 = ind[i];
//                     ind[i] = ind[j];
//                     ind[j] = tmp2;
//                 }
//             }
//         }
//     }

//     return;
// }

// static double cosineSimilarity(double *x, double *y, int len)
// {
//     double a, b, c;
//     int i;

//     a = 0.0;
//     b = 0.0;
//     c = 0.0;
//     for (i = 0; i < len; i++) {
//         a += x[i] * y[i];
//         b += x[i] * x[i];
//         c += y[i] * y[i];
//     }

//     return (acos(a / (sqrt(b) * sqrt(c))));
// }

static int isValid(double* x, int len, int max)
{
	int i, j;
	int n = max;
	int cur_i, cur_j;

	for (i = 0; i < len; i++) {
		cur_i = ((int)(x[i] * n)) % n;
		for (j = i + 1; j < len; j++) {
			cur_j = ((int)(x[j] * n)) % n;
			if (cur_i == cur_j)
				return 0;
		}
	}

	return 1;
}