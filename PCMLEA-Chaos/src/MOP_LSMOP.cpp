#include "MOP_LSMOP.h"

//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include "assert.h"
#include <string.h>

//////////////////////////////////////////////////////////////////////////

// #define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029
#define MYSIGN(x) ((x)>0?1.0:-1.0)

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//  Implementation
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// LSMOP

int LSMOP_D;
static int LSMOP_M;
static int N_k = 5;
static int N_ns;
static int* NNg;
static int* Aa, * Ab, * Ac;
static double a = 3.8, c0 = 0.1;
static int* LSMOP_disp_Xs, * LSMOP_len_Xs;

static double sphere_func(double* x, int len);
static double schwefel_func(double* x, int len);
static double rosenbrock_func(double* x, int len);
static double rastrigin_func(double* x, int len);
static double griewank_func(double* x, int len);
static double ackley_func(double* x, int len);

// static double(*g1_func)(double*, int);
// static double(*g2_func)(double*, int);

static void linkage_linear(double* x, double* z);
static void linkage_nonlinear(double* x, double* z);
static void fitness_linear(double* x, double* h, double* g, int* A);
static void fitness_convex(double* x, double* h, double* g, int* A);
static void fitness_discon(double* x, double* h, double* g, int* A);

static double chaos(double c)
{
	return (a * c * (1.0 - c));
}

int LSMOP_nvar = 0, LSMOP_nobj = 0;                //  the number of variables and objectives
int LSMOP_position_parameters = 0;
char LSMOP_testInstName[1024];

void InitPara_LSMOP(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(LSMOP_testInstName, instName);
	LSMOP_nobj = numObj;
	LSMOP_nvar = numVar;
	LSMOP_position_parameters = posPara;

	return;
}

void SetLimits_LSMOP(double* minLimit, double* maxLimit, int dim)
{
	for (int i = 0; i < LSMOP_nobj - 1; i++) {
		minLimit[i] = 0;
		maxLimit[i] = 1;
	}
	for (int i = LSMOP_nobj - 1; i < dim; i++) {
		minLimit[i] = 0;
		maxLimit[i] = 10;
	}
	return;
}

int CheckLimits_LSMOP(double* x, int dim)
{
	for (int i = 0; i < LSMOP_nobj - 1; i++) {
		if (x[i] < 0.0 || x[i] > 1.0) {
			printf("Check limits FAIL - %s\n", LSMOP_testInstName);
			return false;
		}
	}
	for (int i = LSMOP_nobj - 1; i < dim; i++) {
		if (x[i] < 0.0 || x[i] > 10.0) {
			printf("Check limits FAIL - %s\n", LSMOP_testInstName);
			return false;
		}
	}
	return true;
}

void LSMOP_initialization(char* prob, int M)
{
	LSMOP_M = M;
	N_ns = LSMOP_M * 100;
	NNg = (int*)calloc(LSMOP_M, sizeof(int));

	int i, j;
	double* C = (double*)calloc(LSMOP_M, sizeof(double));
	double sum = 0.0;
	C[0] = chaos(c0);
	sum = C[0];
	for (i = 1; i < LSMOP_M; i++) {
		C[i] = chaos(C[i - 1]);
		sum += C[i];
	}
	double sum1 = 0.0;
	for (i = 0; i < LSMOP_M; i++) {
		NNg[i] = (int)(ceil(round(C[i] / sum * N_ns) / N_k));
		sum1 += NNg[i];
		//      printf("%d ",NNg[i]);
	}
	N_ns = (int)(sum1 * N_k);
	LSMOP_D = (LSMOP_M - 1) + N_ns;
	//  printf("D=%d\n",LSMOP_D);

	Aa = (int*)calloc(LSMOP_M * LSMOP_M, sizeof(int));
	Ab = (int*)calloc(LSMOP_M * LSMOP_M, sizeof(int));
	Ac = (int*)calloc(LSMOP_M * LSMOP_M, sizeof(int));

	for (i = 0; i < LSMOP_M; i++) {
		Aa[i * LSMOP_M + i] = 1;
		Ab[i * LSMOP_M + i] = 1;
		if (i < LSMOP_M - 1)
			Ab[i * LSMOP_M + i + 1] = 1;
		for (j = 0; j < LSMOP_M; j++) {
			Ac[i * LSMOP_M + j] = 1;
		}
	}

	LSMOP_disp_Xs = (int*)calloc(LSMOP_M, sizeof(int));
	LSMOP_len_Xs = (int*)calloc(LSMOP_M, sizeof(int));

	LSMOP_disp_Xs[0] = LSMOP_M - 1;
	LSMOP_len_Xs[0] = N_k * NNg[0];
	for (i = 1; i < LSMOP_M; i++) {
		LSMOP_len_Xs[i] = N_k * NNg[i];
		LSMOP_disp_Xs[i] = LSMOP_disp_Xs[i - 1] + LSMOP_len_Xs[i - 1];
	}

	//if (!strcmp(prob, "LSMOP1")) {
	//  g1_func = sphere_func;
	//  g2_func = sphere_func;
	//}
	//else if (!strcmp(prob, "LSMOP2")) {
	//  g1_func = griewank_func;
	//  g2_func = schwefel_func;
	//}
	//else if (!strcmp(prob, "LSMOP3")) {
	//  g1_func = rastrigin_func;
	//  g2_func = rosenbrock_func;
	//}
	//else if (!strcmp(prob, "LSMOP4")) {
	//  g1_func = ackley_func;
	//  g2_func = griewank_func;
	//}
	//else if (!strcmp(prob, "LSMOP5")) {
	//  g1_func = sphere_func;
	//  g2_func = sphere_func;
	//}
	//else if (!strcmp(prob, "LSMOP6")) {
	//  g1_func = rosenbrock_func;
	//  g2_func = schwefel_func;
	//}
	//else if (!strcmp(prob, "LSMOP7")) {
	//  g1_func = ackley_func;
	//  g2_func = rosenbrock_func;
	//}
	//else if (!strcmp(prob, "LSMOP8")) {
	//  g1_func = griewank_func;
	//  g2_func = sphere_func;
	//}
	//else if (!strcmp(prob, "LSMOP9")) {
	//  g1_func = sphere_func;
	//  g2_func = ackley_func;
	//}

	//  //partition
	//#ifdef MPI_INCLUDED
	//  int my_rank;
	//  int group_size;
	//  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//  MPI_Comm_size(MPI_COMM_WORLD, &group_size);
	//  if (my_rank == 0)
	//#endif
	//  {
	//      FILE* fpt;
	//      char fnm[256];
	//      sprintf(fnm, "Actual_partition_%dM_%dD", LSMOP_M, LSMOP_D);
	//      fpt = fopen(fnm, "w");
	//      fprintf(fpt, "%d\t", LSMOP_M - 1);
	//      for (i = 0; i < LSMOP_M; i++)
	//          fprintf(fpt, "%d\t", NNg[i]);
	//      fprintf(fpt, "\n");
	//      fclose(fpt);
	//  }

	free(C);

	return;
}

void LSMOP_setLimits(double* lbound, double* ubound)
{
	int i;
	for (i = 0; i < LSMOP_M - 1; i++) {
		lbound[i] = 0;
		ubound[i] = 1;
	}
	for (i = LSMOP_M - 1; i < LSMOP_D; i++) {
		lbound[i] = 0;
		ubound[i] = 10;
	}

	return;
}

// static bool LSMOP_checkLimits(double* x)
// {
//     int i;
//     for (i = 0; i < LSMOP_M - 1; i++) {
//         if (x[i] < 0.0 || x[i] > 1.0) {
//             printf("var %d = %lf not in [%lf, %lf]\n", i, x[i], 0.0, 1.0);
//             return false;
//         }
//     }
//     for (i = LSMOP_M - 1; i < LSMOP_D; i++) {
//         if (x[i] < 0.0 || x[i] > 10.0) {
//             printf("var %d = %lf not in [%lf, %lf]\n", i, x[i], 0.0, 10.0);
//             return false;
//         }
//     }

//     return true;
// }

//void LSMOP_fitness(char* prob, double* x, double* f)
//{
//  int i, j;
//
//  //if (!LSMOP_checkLimits(x)) {
//  //    printf("Solution x is beyond boundary, exiting...\n");
//  //    exit(-111);
//  //}
//
//  double* z = (double*)calloc(LSMOP_D, sizeof(double));
//
//  if (!strcmp(prob, "LSMOP1") ||
//      !strcmp(prob, "LSMOP2") ||
//      !strcmp(prob, "LSMOP3") ||
//      !strcmp(prob, "LSMOP4")) {
//      linkage_linear(x, z);
//  }
//  else if (!strcmp(prob, "LSMOP5") ||
//      !strcmp(prob, "LSMOP6") ||
//      !strcmp(prob, "LSMOP7") ||
//      !strcmp(prob, "LSMOP8") ||
//      !strcmp(prob, "LSMOP9")) {
//      linkage_nonlinear(x, z);
//  }
//  else {
//      printf("UNKOWN LSMOP INSTANCS, exiting...\n");
//      exit(-112);
//  }
//  /*  for(i=0;i<LSMOP_D;i++)
//  printf("%lf ",z[i]);
//  printf("\n");*/
//
//  double *g = (double*)calloc(LSMOP_M, sizeof(double));
//  for (i = 0; i < LSMOP_M; i++) {
//      int nss = NNg[i];
//      double* segXss = &z[LSMOP_disp_Xs[i]];
//      if (i % 2 == 0) {
//          for (j = 0; j < N_k; j++) {
//              g[i] += g1_func(&segXss[j * nss], nss) / nss;
//          }
//      }
//      else {
//          for (j = 0; j < N_k; j++) {
//              g[i] += g2_func(&segXss[j * nss], nss) / nss;
//          }
//      }
//      g[i] /= N_k;
//  }
//
//  if (!strcmp(prob, "LSMOP1") ||
//      !strcmp(prob, "LSMOP2") ||
//      !strcmp(prob, "LSMOP3") ||
//      !strcmp(prob, "LSMOP4")) {
//      fitness_linear(z, f, g, Aa);
//  }
//  else if (!strcmp(prob, "LSMOP5") ||
//      !strcmp(prob, "LSMOP6") ||
//      !strcmp(prob, "LSMOP7") ||
//      !strcmp(prob, "LSMOP8")) {
//      fitness_convex(z, f, g, Ab);
//  }
//  else if (!strcmp(prob, "LSMOP9")) {
//      fitness_discon(z, f, g, Ac);
//  }
//
//  free(z);
//  free(g);
//
//  return;
//}

void LSMOP1(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_linear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_linear(z, fit, g, Aa);

	free(z);
	free(g);

	return;
}

void LSMOP2(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_linear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += griewank_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += schwefel_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_linear(z, fit, g, Aa);

	free(z);
	free(g);

	return;
}

void LSMOP3(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_linear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += rastrigin_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += rosenbrock_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_linear(z, fit, g, Aa);

	free(z);
	free(g);

	return;
}

void LSMOP4(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_linear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += ackley_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += griewank_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_linear(z, fit, g, Aa);

	free(z);
	free(g);

	return;
}

void LSMOP5(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_nonlinear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_convex(z, fit, g, Ab);

	free(z);
	free(g);

	return;
}

void LSMOP6(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_nonlinear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += rosenbrock_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += schwefel_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_convex(z, fit, g, Ab);

	free(z);
	free(g);

	return;
}

void LSMOP7(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_nonlinear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += ackley_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += rosenbrock_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_convex(z, fit, g, Ab);

	free(z);
	free(g);

	return;
}

void LSMOP8(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_nonlinear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += griewank_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_convex(z, fit, g, Ab);

	free(z);
	free(g);

	return;
}

void LSMOP9(double* x, double* fit, double* constrainV, int nx, int M)
{
	int i, j;

	double* z = (double*)calloc(nx, sizeof(double));

	linkage_nonlinear(x, z);

	double* g = (double*)calloc(M, sizeof(double));
	for (i = 0; i < M; i++) {
		int nss = NNg[i];
		double* segXss = &z[LSMOP_disp_Xs[i]];
		if (i % 2 == 0) {
			for (j = 0; j < N_k; j++) {
				g[i] += sphere_func(&segXss[j * nss], nss) / nss;
			}
		}
		else {
			for (j = 0; j < N_k; j++) {
				g[i] += ackley_func(&segXss[j * nss], nss) / nss;
			}
		}
		g[i] /= N_k;
	}

	fitness_discon(z, fit, g, Ac);

	free(z);
	free(g);

	return;
}

static void linkage_linear(double* x, double* z)
{
	int i;
	for (i = 0; i < LSMOP_M - 1; i++) {
		z[i] = x[i];
	}
	for (i = LSMOP_M - 1; i < LSMOP_D; i++) {
		z[i] = x[i] * (1 + (i + 1.0) / LSMOP_D) - 10 * x[0];
	}

	return;
}
static void linkage_nonlinear(double* x, double* z)
{
	int i;
	for (i = 0; i < LSMOP_M - 1; i++) {
		z[i] = x[i];
	}
	for (i = LSMOP_M - 1; i < LSMOP_D; i++) {
		z[i] = x[i] * (1 + cos(0.5 * PI * (i + 1.0) / LSMOP_D)) - 10 * x[0];
	}

	return;
}

static void fitness_linear(double* x, double* f, double* g, int* A)
{
	int i, j;
	double prod = 1.0;
	double dist;
	for (i = LSMOP_M - 1; i >= 0; i--) {
		dist = 1.0;
		for (j = 0; j < LSMOP_M; j++) {
			dist += A[i * LSMOP_M + j] * g[j];
		}

		if (i)
			f[i] = (1.0 - x[LSMOP_M - 1 - i]);
		else
			f[i] = 1.0;

		f[i] *= prod;
		if (i)
			prod *= x[LSMOP_M - 1 - i];

		f[i] *= dist;
	}

	return;
}
static void fitness_convex(double* x, double* f, double* g, int* A)
{
	int i, j;
	double prod = 1.0;
	double dist;
	for (i = LSMOP_M - 1; i >= 0; i--) {
		dist = 1.0;
		for (j = 0; j < LSMOP_M; j++) {
			dist += A[i * LSMOP_M + j] * g[j];
		}

		if (i)
			f[i] = sin(0.5 * PI * x[LSMOP_M - 1 - i]);
		else
			f[i] = 1.0;

		f[i] *= prod;
		if (i)
			prod *= cos(0.5 * PI * x[LSMOP_M - 1 - i]);

		f[i] *= dist;
	}

	return;
}
static void fitness_discon(double* x, double* f, double* g, int* A)
{
	int i, j;
	double dist;
	double sum = 0.0;

	for (i = 0; i < LSMOP_M - 1; i++) {
		f[i] = x[i];
	}

	dist = 0.0;
	for (j = 0; j < LSMOP_M; j++) {
		dist += A[(LSMOP_M - 1) * LSMOP_M + j] * g[j];
	}

	for (i = 0; i < LSMOP_M - 1; i++) {
		sum += x[i] * (1.0 + sin(3.0 * PI * x[i])) / (2.0 + dist);
	}
	f[LSMOP_M - 1] = (LSMOP_M - sum) * (2.0 + dist);

	return;
}

void LSMOP_finalization()
{
	free(NNg);
	free(Aa);
	free(Ab);
	free(Ac);
	free(LSMOP_disp_Xs);
	free(LSMOP_len_Xs);

	return;
}

static double sphere_func(double* x, int len)
{
	double result = 0.0;
	int D = len;

	int i;
	for (i = 0; i < D; i++) {
		result += x[i] * x[i];
	}

	return result;
}
static double schwefel_func(double* x, int len)
{
	double result = 0.0;
	int D = len;

	int i;
	for (i = 0; i < D; i++) {
		if (result < fabs(x[i]))
			result = fabs(x[i]);
	}

	return result;
}
static double rosenbrock_func(double* x, int len)
{
	double result = 0.0;
	int D = len;

	int i;
	for (i = 0; i < D - 1; i++) {
		result += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) +
			(x[i] - 1.0) * (x[i] - 1.0);
	}

	return result;
}
static double rastrigin_func(double* x, int len)
{
	double result = 0.0;
	int D = len;

	int i;
	for (i = 0; i < D; i++) {
		result += x[i] * x[i] - 10.0 * cos(2 * PI * x[i]) + 10.0;
	}

	return result;
}
static double griewank_func(double* x, int len)
{
	double result = 1.0;
	int D = len;

	double sum = 0.0;
	int i;
	for (i = 0; i < D; i++) {
		sum += x[i] * x[i] / 4000.0;
		result *= cos(x[i] / sqrt(i + 1.0));
	}

	return (sum - result + 1);
}
static double ackley_func(double* x, int len)
{
	double result = 0.0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	int D = len;

	int i;
	for (i = 0; i < D; i++) {
		sum1 += x[i] * x[i];
		sum2 += cos(2.0 * PI * x[i]);
	}
	result = 20.0 - 20.0 * exp(-0.2 * sqrt(sum1 / D)) -
		exp(sum2 / D) + exp(1.0);

	return result;
}