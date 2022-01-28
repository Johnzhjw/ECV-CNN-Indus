#include "MOP_DTLZ.h"

//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <string.h>

//////////////////////////////////////////////////////////////////////////

// #define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029
#define MYSIGN(x) ((x)>0?1.0:-1.0)

int DTLZ_nvar, DTLZ_nobj;                    //  the number of variables and objectives
int DTLZ_position_parameters;
char DTLZ_testInstName[1024];

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Implementation

void InitPara_DTLZ(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(DTLZ_testInstName, instName);
	DTLZ_nobj = numObj;
	DTLZ_nvar = numVar;
	DTLZ_position_parameters = posPara;

	return;
}
void SetLimits_DTLZ(double* minLimit, double* maxLimit, int dim)
{
	for (int i = 0; i < dim; i++) {
		minLimit[i] = 0.0;
		maxLimit[i] = 1.0;
	}

	return;
}

int CheckLimits_DTLZ(double* x, int dim)
{
	for (int i = 0; i < dim; i++) {
		if (x[i] < 0.0 || x[i] > 1.0) {
			printf("Check limits FAIL - %s\n", DTLZ_testInstName);
			return false;
		}
	}
	return true;
}
//////////////////////////////////////////////////////////////////////////
//	DTLZ
void dtlz1(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i] - 0.5), 2.0) - cos(20 * PI * (xreal[i] - 0.5));
	}
	gx = 100 * (sum + dim - nobj + 1) + 1.0;
	sum = gx;
	for (j = 0; j < nobj - 1; j++) {
		sum = sum * xreal[j];
	}
	obj[0] = 0.5 * sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 0; j < nobj - 1 - i; j++) {
			sum = sum * xreal[j];
		}
		sum = sum * (1.0 - xreal[nobj - 1 - i]);
		obj[i] = 0.5 * sum;
	}
	return;
}

void dtlz2(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i] - 0.5), 2.0);
	}
	gx = 1.0 + sum;
	sum = gx;
	for (j = 0; j < nobj - 1; j++) {
		sum = sum * cos(xreal[j] * PI / 2.0);
	}
	obj[0] = sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 0; j < nobj - 1 - i; j++) {
			sum = sum * cos(xreal[j] * PI / 2.0);
		}
		sum = sum * sin(xreal[nobj - 1 - i] * PI / 2.0);
		obj[i] = sum;
	}
	return;
}

void dtlz3(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i] - 0.5), 2.0) - cos(20 * PI * (xreal[i] - 0.5));
	}
	gx = 100 * (sum + dim - nobj + 1) + 1.0;
	sum = gx;
	for (j = 0; j < nobj - 1; j++) {
		sum = sum * cos(xreal[j] * PI / 2.0);
	}
	obj[0] = sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 0; j < nobj - 1 - i; j++) {
			sum = sum * cos(xreal[j] * PI / 2.0);
		}
		sum = sum * sin(xreal[nobj - 1 - i] * PI / 2.0);
		obj[i] = sum;
	}
	return;
}

void dtlz4(double* _xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;
	double* xreal;
	xreal = (double*)malloc(dim * sizeof(double));
	memcpy(xreal, _xreal, dim * sizeof(double));

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i] - 0.5), 2.0);
	}
	for (i = 0; i < nobj - 1; i++) {
		xreal[i] = pow(xreal[i], 100);
	}
	gx = 1.0 + sum;
	sum = gx;
	for (j = 0; j < nobj - 1; j++) {
		sum = sum * cos(xreal[j] * PI / 2.0);
	}
	obj[0] = sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 0; j < nobj - 1 - i; j++) {
			sum = sum * cos(xreal[j] * PI / 2.0);
		}
		sum = sum * sin(xreal[nobj - 1 - i] * PI / 2.0);
		obj[i] = sum;
	}
	free(xreal);
	return;
}

void dtlz5(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;
	double* x;
	x = (double*)malloc((nobj - 1) * sizeof(double));

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i] - 0.5), 2.0);
	}
	for (i = 1; i < nobj - 1; i++) {
		x[i] = PI / (4 * (1 + sum)) * (1 + 2 * sum * xreal[i]);
	}
	gx = 1.0 + sum;
	sum = gx;
	for (j = 1; j < nobj - 1; j++) {
		sum = sum * cos(x[j]);
	}
	sum = sum * cos(xreal[0] * PI / 2.0);
	obj[0] = sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 1; j < nobj - 1 - i; j++) {
			sum = sum * cos(x[j]);
		}
		if (i == nobj - 1) {
			sum = sum * sin(xreal[0] * PI / 2.0);
		}
		else {
			sum = sum * sin(x[nobj - 1 - i]);
			sum = sum * cos(xreal[0] * PI / 2.0);
		}
		obj[i] = sum;
	}
	free(x);
	return;
}

void dtlz6(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0;
	double gx;
	int i, j;
	double* x;
	x = (double*)malloc((nobj - 1) * sizeof(double));

	for (i = nobj - 1; i < dim; i++) {
		sum += pow((xreal[i]), 0.1);
	}
	for (i = 1; i < nobj - 1; i++) {
		x[i] = PI / (4 * (1 + sum)) * (1 + 2 * sum * xreal[i]);
	}
	gx = 1.0 + sum;
	sum = gx;
	for (j = 1; j < nobj - 1; j++) {
		sum = sum * cos(x[j]);
	}
	sum = sum * cos(xreal[0] * PI / 2.0);
	obj[0] = sum;

	for (i = 1; i < nobj; i++) {
		sum = gx;
		for (j = 1; j < nobj - 1 - i; j++) {
			sum = sum * cos(x[j]);
		}
		if (i == nobj - 1) {
			sum = sum * sin(xreal[0] * PI / 2.0);
		}
		else {
			sum = sum * sin(x[nobj - 1 - i]);
			sum = sum * cos(xreal[0] * PI / 2.0);
		}
		obj[i] = sum;
	}
	free(x);
	return;
}

void dtlz7(double* xreal, double* obj, double* constrainV, int dim, int nobj)
{
	double sum = 0, temp = 0;
	double gx;
	int i;

	for (i = nobj - 1; i < dim; i++) {
		sum += xreal[i];
	}
	gx = 1.0 + 9.0 * sum / (dim - nobj + 1.0);
	sum = gx;
	for (i = 0; i < nobj - 1; i++) {
		obj[i] = xreal[i];
	}
	for (i = 0; i < nobj - 1; i++) {
		temp += (obj[i] / (sum + 1)) * (1 + sin(3 * PI * obj[i]));
	}
	temp = nobj - temp;
	obj[nobj - 1] = (sum + 1) * temp;
	return;
}

/*
#ifdef dtlz5_I_M
void test_problem (double *xreal, double *obj, double *constr)
{
double sum=0;
double gx;
int i, j;
double *x;
x=(double*)malloc((EMO_nobj-1)*sizeof(double));

for (i=EMO_nobj-1; i<nreal; i++)
{
sum += pow ((xreal[i]-0.5), 2.0);
}

for (i=0; i<I_number-1; i++)
{
x[i] = xreal[i]*PI/2.0;
}
for (i=I_number-1; i<EMO_nobj-1; i++)
{
x[i] = PI/(4*(1+sum))*(1+2*sum*xreal[i]);
}
gx = 1.0 + 100 * sum;

sum = gx;
for (j=0; j<EMO_nobj-1; j++)
{
sum = sum * cos(x[j]);
}
obj[0] = sum;

for (i=1; i<EMO_nobj; i++)
{
sum = gx;
for (j=0; j<EMO_nobj-1-i; j++)
{
sum = sum * cos(x[j]);
}
sum = sum * sin(x[EMO_nobj-1-i]);
obj[i] = sum;
}
free (x);
return;
}
#endif
*/

//	DTLZ
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////