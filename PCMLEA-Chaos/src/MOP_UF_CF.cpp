#include "MOP_UF_CF.h"

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

int UF_CF_nvar, UF_CF_nobj;                    //  the number of variables and objectives
int UF_CF_position_parameters;
char UF_CF_testInstName[1024];

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Implementation
//////////////////////////////////////////////////////////////////////////
//	UF	&	CF
/****************************************************************************/
// unconstraint test instances
/****************************************************************************/
void InitPara_UF(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(UF_CF_testInstName, instName);
	UF_CF_nobj = numObj;
	UF_CF_nvar = numVar;
	UF_CF_position_parameters = posPara;

	return;
}
void SetLimits_UF(double* minLimit, double* maxLimit, int dim)
{
	if (strcmp(UF_CF_testInstName, "UF1") == 0
		|| strcmp(UF_CF_testInstName, "UF2") == 0
		|| strcmp(UF_CF_testInstName, "UF5") == 0
		|| strcmp(UF_CF_testInstName, "UF6") == 0
		|| strcmp(UF_CF_testInstName, "UF7") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		for (int i = 1; i < dim; i++) {
			minLimit[i] = -1.0;
			maxLimit[i] = 1.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "UF3") == 0) {
		for (int i = 0; i < dim; i++) {
			minLimit[i] = 0.0;
			maxLimit[i] = 1.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "UF4") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		for (int i = 1; i < dim; i++) {
			minLimit[i] = -2.0;
			maxLimit[i] = 2.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "UF8") == 0
		|| strcmp(UF_CF_testInstName, "UF9") == 0
		|| strcmp(UF_CF_testInstName, "UF10") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		minLimit[1] = 0.0;
		maxLimit[1] = 1.0;
		for (int i = 2; i < dim; i++) {
			minLimit[i] = -2.0;
			maxLimit[i] = 2.0;
		}
		return;
	}
}

int CheckLimits_UF(double* x, int dim)
{
	//	UF
	if (strcmp(UF_CF_testInstName, "UF1") == 0
		|| strcmp(UF_CF_testInstName, "UF2") == 0
		|| strcmp(UF_CF_testInstName, "UF5") == 0
		|| strcmp(UF_CF_testInstName, "UF6") == 0
		|| strcmp(UF_CF_testInstName, "UF7") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 1; i < dim; i++) {
			if (x[i] < -1.0 || x[i] > 1.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "UF3") == 0) {
		for (int i = 0; i < dim; i++) {
			if (x[i] < 0.0 || x[i] > 1.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "UF4") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 1; i < dim; i++) {
			if (x[i] < -2.0 || x[i] > 2.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "UF8") == 0
		|| strcmp(UF_CF_testInstName, "UF9") == 0
		|| strcmp(UF_CF_testInstName, "UF10") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0 || x[1] < 0.0 || x[1] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 2; i < dim; i++) {
			if (x[i] < -2.0 || x[i] > 2.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}

	return true;
}

void UF1(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		yj = yj * yj;
		if (j % 2 == 0) {
			sum2 += yj;
			count2++;
		}
		else {
			sum1 += yj;
			count1++;
		}
	}
	f[0] = x[0] + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - sqrt(x[0]) + 2.0 * sum2 / (double)count2;
}

void UF2(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		if (j % 2 == 0) {
			yj = x[j - 1] - 0.3 * x[0] * (x[0] * cos(24.0 * PI * x[0] + 4.0 * j * PI / nx) + 2.0) * sin(6.0 * PI * x[0] + j * PI / nx);
			sum2 += yj * yj;
			count2++;
		}
		else {
			yj = x[j - 1] - 0.3 * x[0] * (x[0] * cos(24.0 * PI * x[0] + 4.0 * j * PI / nx) + 2.0) * cos(6.0 * PI * x[0] + j * PI / nx);
			sum1 += yj * yj;
			count1++;
		}
	}
	f[0] = x[0] + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - sqrt(x[0]) + 2.0 * sum2 / (double)count2;
}

void UF3(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, prod1, prod2, yj, pj;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	prod1 = prod2 = 1.0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (nx - 2.0)));
		pj = cos(20.0 * yj * PI / sqrt(j + 0.0));
		if (j % 2 == 0) {
			sum2 += yj * yj;
			prod2 *= pj;
			count2++;
		}
		else {
			sum1 += yj * yj;
			prod1 *= pj;
			count1++;
		}
	}
	f[0] = x[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / (double)count1;
	f[1] = 1.0 - sqrt(x[0]) + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / (double)count2;
}

void UF4(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj, hj;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		hj = fabs(yj) / (1.0 + exp(2.0 * fabs(yj)));
		if (j % 2 == 0) {
			sum2 += hj;
			count2++;
		}
		else {
			sum1 += hj;
			count1++;
		}
	}
	f[0] = x[0] + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - x[0] * x[0] + 2.0 * sum2 / (double)count2;
}

void UF5(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj, hj, N, E;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	N = 10.0;
	E = 0.1;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		hj = 2.0 * yj * yj - cos(4.0 * PI * yj) + 1.0;
		if (j % 2 == 0) {
			sum2 += hj;
			count2++;
		}
		else {
			sum1 += hj;
			count1++;
		}
	}
	hj = (0.5 / N + E) * fabs(sin(2.0 * N * PI * x[0]));
	f[0] = x[0] + hj + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - x[0] + hj + 2.0 * sum2 / (double)count2;
}

void UF6(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, prod1, prod2, yj, hj, pj, N, E;
	N = 2.0;
	E = 0.1;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	prod1 = prod2 = 1.0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		pj = cos(20.0 * yj * PI / sqrt(j + 0.0));
		if (j % 2 == 0) {
			sum2 += yj * yj;
			prod2 *= pj;
			count2++;
		}
		else {
			sum1 += yj * yj;
			prod1 *= pj;
			count1++;
		}
	}

	hj = 2.0 * (0.5 / N + E) * sin(2.0 * N * PI * x[0]);
	if (hj < 0.0) hj = 0.0;
	f[0] = x[0] + hj + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / (double)count1;
	f[1] = 1.0 - x[0] + hj + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / (double)count2;
}

void UF7(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		if (j % 2 == 0) {
			sum2 += yj * yj;
			count2++;
		}
		else {
			sum1 += yj * yj;
			count1++;
		}
	}
	yj = pow(x[0], 0.2);
	f[0] = yj + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - yj + 2.0 * sum2 / (double)count2;
}

void UF8(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj;

	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		if (j % 3 == 1) {
			sum1 += yj * yj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += yj * yj;
			count2++;
		}
		else {
			sum3 += yj * yj;
			count3++;
		}
	}
	f[0] = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]) + 2.0 * sum1 / (double)count1;
	f[1] = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]) + 2.0 * sum2 / (double)count2;
	f[2] = sin(0.5 * PI * x[0]) + 2.0 * sum3 / (double)count3;
}

void UF9(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj, E;

	E = 0.1;
	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		if (j % 3 == 1) {
			sum1 += yj * yj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += yj * yj;
			count2++;
		}
		else {
			sum3 += yj * yj;
			count3++;
		}
	}
	yj = (1.0 + E) * (1.0 - 4.0 * (2.0 * x[0] - 1.0) * (2.0 * x[0] - 1.0));
	if (yj < 0.0) yj = 0.0;
	f[0] = 0.5 * (yj + 2 * x[0]) * x[1] + 2.0 * sum1 / (double)count1;
	f[1] = 0.5 * (yj - 2 * x[0] + 2.0) * x[1] + 2.0 * sum2 / (double)count2;
	f[2] = 1.0 - x[1] + 2.0 * sum3 / (double)count3;
}

void UF10(double* x, double* f, double* constrainV, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj, hj;

	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		hj = 4.0 * yj * yj - cos(8.0 * PI * yj) + 1.0;
		if (j % 3 == 1) {
			sum1 += hj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += hj;
			count2++;
		}
		else {
			sum3 += hj;
			count3++;
		}
	}
	f[0] = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]) + 2.0 * sum1 / (double)count1;
	f[1] = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]) + 2.0 * sum2 / (double)count2;
	f[2] = sin(0.5 * PI * x[0]) + 2.0 * sum3 / (double)count3;
}

/****************************************************************************/
// constraint test instances
/****************************************************************************/
void InitPara_CF(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(UF_CF_testInstName, instName);
	UF_CF_nobj = numObj;
	UF_CF_nvar = numVar;
	UF_CF_position_parameters = posPara;

	return;
}
void SetLimits_CF(double* minLimit, double* maxLimit, int dim)
{
	if (strcmp(UF_CF_testInstName, "CF1") == 0) {
		for (int i = 0; i < dim; i++) {
			minLimit[i] = 0.0;
			maxLimit[i] = 1.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "CF2") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		for (int i = 1; i < dim; i++) {
			minLimit[i] = -1.0;
			maxLimit[i] = 1.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "CF3") == 0
		|| strcmp(UF_CF_testInstName, "CF4") == 0
		|| strcmp(UF_CF_testInstName, "CF5") == 0
		|| strcmp(UF_CF_testInstName, "CF6") == 0
		|| strcmp(UF_CF_testInstName, "CF7") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		for (int i = 1; i < dim; i++) {
			minLimit[i] = -2.0;
			maxLimit[i] = 2.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "CF8") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		minLimit[1] = 0.0;
		maxLimit[1] = 1.0;
		for (int i = 2; i < dim; i++) {
			minLimit[i] = -4.0;
			maxLimit[i] = 4.0;
		}
		return;
	}
	if (strcmp(UF_CF_testInstName, "CF9") == 0
		|| strcmp(UF_CF_testInstName, "CF10") == 0) {
		minLimit[0] = 0.0;
		maxLimit[0] = 1.0;
		minLimit[1] = 0.0;
		maxLimit[1] = 1.0;
		for (int i = 2; i < dim; i++) {
			minLimit[i] = -2.0;
			maxLimit[i] = 2.0;
		}
		return;
	}
}

int CheckLimits_CF(double* x, int dim)
{
	if (strcmp(UF_CF_testInstName, "CF1") == 0) {
		for (int i = 0; i < dim; i++) {
			if (x[i] < 0.0 || x[i] > 1.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "CF2") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 1; i < dim; i++) {
			if (x[i] < -1.0 || x[i] > 1.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "CF3") == 0
		|| strcmp(UF_CF_testInstName, "CF4") == 0
		|| strcmp(UF_CF_testInstName, "CF5") == 0
		|| strcmp(UF_CF_testInstName, "CF6") == 0
		|| strcmp(UF_CF_testInstName, "CF7") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 1; i < dim; i++) {
			if (x[i] < -2.0 || x[i] > 2.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "CF8") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0 || x[1] < 0.0 || x[1] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 2; i < dim; i++) {
			if (x[i] < -4.0 || x[i] > 4.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}
	if (strcmp(UF_CF_testInstName, "CF9") == 0
		|| strcmp(UF_CF_testInstName, "CF10") == 0) {
		if (x[0] < 0.0 || x[0] > 1.0 || x[1] < 0.0 || x[1] > 1.0) {
			printf("Check limits FAIL - %s\n", UF_CF_testInstName);
			return false;
		}
		for (int i = 2; i < dim; i++) {
			if (x[i] < -2.0 || x[i] > 2.0) {
				printf("Check limits FAIL - %s\n", UF_CF_testInstName);
				return false;
			}
		}
	}

	return true;
}

void CF1(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj, N, a;
	N = 10.0;
	a = 1.0;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (nx - 2.0)));
		if (j % 2 == 1) {
			sum1 += yj * yj;
			count1++;
		}
		else {
			sum2 += yj * yj;
			count2++;
		}
	}
	f[0] = x[0] + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - x[0] + 2.0 * sum2 / (double)count2;
	c[0] = f[1] + f[0] - a * fabs(sin(N * PI * (f[0] - f[1] + 1.0))) - 1.0;
}

void CF2(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, yj, N, a, t;
	N = 2.0;
	a = 1.0;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	for (j = 2; j <= nx; j++) {
		if (j % 2 == 1) {
			yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
			sum1 += yj * yj;
			count1++;
		}
		else {
			yj = x[j - 1] - cos(6.0 * PI * x[0] + j * PI / nx);
			sum2 += yj * yj;
			count2++;
		}
	}
	f[0] = x[0] + 2.0 * sum1 / (double)count1;
	f[1] = 1.0 - sqrt(x[0]) + 2.0 * sum2 / (double)count2;
	t = f[1] + sqrt(f[0]) - a * sin(N * PI * (sqrt(f[0]) - f[1] + 1.0)) - 1.0;
	c[0] = MYSIGN(t) * fabs(t) / (1 + exp(4.0 * fabs(t)));
}

void CF3(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2;
	double sum1, sum2, prod1, prod2, yj, pj, N, a;
	N = 2.0;
	a = 1.0;

	sum1 = sum2 = 0.0;
	count1 = count2 = 0;
	prod1 = prod2 = 1.0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		pj = cos(20.0 * yj * PI / sqrt(j + 0.0));
		if (j % 2 == 0) {
			sum2 += yj * yj;
			prod2 *= pj;
			count2++;
		}
		else {
			sum1 += yj * yj;
			prod1 *= pj;
			count1++;
		}
	}

	f[0] = x[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / (double)count1;
	f[1] = 1.0 - x[0] * x[0] + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / (double)count2;
	c[0] = f[1] + f[0] * f[0] - a * sin(N * PI * (f[0] * f[0] - f[1] + 1.0)) - 1.0;
}

void CF4(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	double sum1, sum2, yj, t;

	sum1 = sum2 = 0.0;
	for (j = 2; j <= nx; j++) {
		yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
		if (j % 2 == 1) {
			sum1 += yj * yj;
		}
		else {
			if (j == 2)
				sum2 += yj < 1.5 - 0.75 * sqrt(2.0) ? fabs(yj) : (0.125 + (yj - 1) * (yj - 1));
			else
				sum2 += yj * yj;
		}
	}
	f[0] = x[0] + sum1;
	f[1] = 1.0 - x[0] + sum2;
	t = x[1] - sin(6.0 * x[0] * PI + 2.0 * PI / nx) - 0.5 * x[0] + 0.25;
	c[0] = MYSIGN(t) * fabs(t) / (1 + exp(4.0 * fabs(t)));
}

void CF5(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	for (j = 2; j <= nx; j++) {
		if (j % 2 == 1) {
			yj = x[j - 1] - 0.8 * x[0] * cos(6.0 * PI * x[0] + j * PI / nx);
			sum1 += 2.0 * yj * yj - cos(4.0 * PI * yj) + 1.0;
		}
		else {
			yj = x[j - 1] - 0.8 * x[0] * sin(6.0 * PI * x[0] + j * PI / nx);
			if (j == 2)
				sum2 += yj < 1.5 - 0.75 * sqrt(2.0) ? fabs(yj) : (0.125 + (yj - 1) * (yj - 1));
			else
				sum2 += 2.0 * yj * yj - cos(4.0 * PI * yj) + 1.0;
		}
	}
	f[0] = x[0] + sum1;
	f[1] = 1.0 - x[0] + sum2;
	c[0] = x[1] - 0.8 * x[0] * sin(6.0 * x[0] * PI + 2.0 * PI / nx) - 0.5 * x[0] + 0.25;
}

void CF6(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	for (j = 2; j <= nx; j++) {
		if (j % 2 == 1) {
			yj = x[j - 1] - 0.8 * x[0] * cos(6.0 * PI * x[0] + j * PI / nx);
			sum1 += yj * yj;
		}
		else {
			yj = x[j - 1] - 0.8 * x[0] * sin(6.0 * PI * x[0] + j * PI / nx);
			sum2 += yj * yj;
		}
	}
	f[0] = x[0] + sum1;
	f[1] = (1.0 - x[0]) * (1.0 - x[0]) + sum2;
	c[0] = x[1] - 0.8 * x[0] * sin(6.0 * x[0] * PI + 2.0 * PI / nx) - MYSIGN((x[0] - 0.5) * (1.0 - x[0])) * sqrt(fabs((
		x[0] - 0.5) * (1.0 - x[0])));
	c[1] = x[3] - 0.8 * x[0] * sin(6.0 * x[0] * PI + 4.0 * PI / nx) - MYSIGN(0.25 * sqrt(1 - x[0]) - 0.5 * (1.0 - x[0])) * sqrt(
		fabs(0.25 * sqrt(1 - x[0]) - 0.5 * (1.0 - x[0])));
}

void CF7(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	double sum1, sum2, yj;

	sum1 = sum2 = 0.0;
	for (j = 2; j <= nx; j++) {
		if (j % 2 == 1) {
			yj = x[j - 1] - cos(6.0 * PI * x[0] + j * PI / nx);
			sum1 += 2.0 * yj * yj - cos(4.0 * PI * yj) + 1.0;
		}
		else {
			yj = x[j - 1] - sin(6.0 * PI * x[0] + j * PI / nx);
			if (j == 2 || j == 4)
				sum2 += yj * yj;
			else
				sum2 += 2.0 * yj * yj - cos(4.0 * PI * yj) + 1.0;
		}
	}
	f[0] = x[0] + sum1;
	f[1] = (1.0 - x[0]) * (1.0 - x[0]) + sum2;
	c[0] = x[1] - sin(6.0 * x[0] * PI + 2.0 * PI / nx) - MYSIGN((x[0] - 0.5) * (1.0 - x[0])) * sqrt(fabs((x[0] - 0.5) *
		(1.0 - x[0])));
	c[1] = x[3] - sin(6.0 * x[0] * PI + 4.0 * PI / nx) - MYSIGN(0.25 * sqrt(1 - x[0]) - 0.5 * (1.0 - x[0])) * sqrt(fabs(0.25 * sqrt(
		1 - x[0]) - 0.5 * (1.0 - x[0])));
}

void CF8(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj, N, a;
	N = 2.0;
	a = 4.0;

	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		if (j % 3 == 1) {
			sum1 += yj * yj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += yj * yj;
			count2++;
		}
		else {
			sum3 += yj * yj;
			count3++;
		}
	}
	f[0] = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]) + 2.0 * sum1 / (double)count1;
	f[1] = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]) + 2.0 * sum2 / (double)count2;
	f[2] = sin(0.5 * PI * x[0]) + 2.0 * sum3 / (double)count3;
	c[0] = (f[0] * f[0] + f[1] * f[1]) / (1 - f[2] * f[2]) - a * fabs(sin(N * PI * ((f[0] * f[0] - f[1] * f[1]) /
		(1 - f[2] * f[2]) + 1.0))) - 1.0;
}

void CF9(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj, N, a;
	N = 2.0;
	a = 3.0;

	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		if (j % 3 == 1) {
			sum1 += yj * yj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += yj * yj;
			count2++;
		}
		else {
			sum3 += yj * yj;
			count3++;
		}
	}
	f[0] = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]) + 2.0 * sum1 / (double)count1;
	f[1] = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]) + 2.0 * sum2 / (double)count2;
	f[2] = sin(0.5 * PI * x[0]) + 2.0 * sum3 / (double)count3;
	c[0] = (f[0] * f[0] + f[1] * f[1]) / (1 - f[2] * f[2]) - a * sin(N * PI * ((f[0] * f[0] - f[1] * f[1]) /
		(1 - f[2] * f[2]) + 1.0)) - 1.0;
}

void CF10(double* x, double* f, double* c, int nx, int nobj)
{
	int j;
	unsigned int count1, count2, count3;
	double sum1, sum2, sum3, yj, hj, N, a;
	N = 2.0;
	a = 1.0;

	sum1 = sum2 = sum3 = 0.0;
	count1 = count2 = count3 = 0;
	for (j = 3; j <= nx; j++) {
		yj = x[j - 1] - 2.0 * x[1] * sin(2.0 * PI * x[0] + j * PI / nx);
		hj = 4.0 * yj * yj - cos(8.0 * PI * yj) + 1.0;
		if (j % 3 == 1) {
			sum1 += hj;
			count1++;
		}
		else if (j % 3 == 2) {
			sum2 += hj;
			count2++;
		}
		else {
			sum3 += hj;
			count3++;
		}
	}
	f[0] = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]) + 2.0 * sum1 / (double)count1;
	f[1] = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]) + 2.0 * sum2 / (double)count2;
	f[2] = sin(0.5 * PI * x[0]) + 2.0 * sum3 / (double)count3;
	c[0] = (f[0] * f[0] + f[1] * f[1]) / (1 - f[2] * f[2]) - a * sin(N * PI * ((f[0] * f[0] - f[1] * f[1]) /
		(1 - f[2] * f[2]) + 1.0)) - 1.0;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////