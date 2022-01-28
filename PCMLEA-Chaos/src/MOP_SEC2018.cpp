// Evaluation.cpp : Defines the entry point for the console application.
//
#include "MOP_SEC2018.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#ifndef __cplusplus
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
//#endif  /* __cplusplus */

#include <math.h>
#define PI  3.1415926535897932384626433832795

int SEC2018_nvar = 0, SEC2018_nobj = 0;                //  the number of variables and objectives
int SEC2018_position_parameters = 0;
char SEC2018_testInstName[1024];

void InitPara_SEC18_MaOP(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(SEC2018_testInstName, instName);
	SEC2018_nobj = numObj;
	SEC2018_nvar = numVar;
	SEC2018_position_parameters = posPara;

	return;
}

void SetLimits_SEC18_MaOP(double* minLimit, double* maxLimit, int dim)
{
	for (int i = 0; i < dim; i++) {
		minLimit[i] = 0.0;
		maxLimit[i] = 1.0;
	}
	return;
}

int CheckLimits_SEC18_MaOP(double* x, int dim)
{
	for (int i = 0; i < dim; i++) {
		if (x[i] < 0.0 || x[i] > 1.0) {
			printf("Check limits FAIL - %s\n", SEC2018_testInstName);
			return false;
		}
	}
	return true;
}

void SEC18_MaOP1(double* x, double* fit, double* constrainV, int nx, int M)//OK
{
	// computing distance function
	double g = 0;
	for (int n = M; n <= nx; n++) {   // nobj-1 --> nobj
		g += (x[n - 1] - 0.5) * (x[n - 1] - 0.5) + 1 - cos(20 * PI * (x[n - 1] - 0.5)); // n --> n-1
	}
	g = g / nx;  // 100 -->10

	// computing position functions
	double prod_x = 1;
	for (int m = M; m >= 1; m--) {
		int id = M - m + 1;                         // id starts with 1 and ends with
		if (m > 1) {
			fit[m - 1] = (1 + g) * (1 - prod_x * (1 - x[id - 1]));
			prod_x = prod_x * x[id - 1];                      // x1 --- x1x2...x(m-1)
		}
		else { // the first objective function   f[0]
			fit[m - 1] = (1 + g) * (1 - prod_x);
		}

		fit[m - 1] = (0.1 + 10 * m) * fit[m - 1];  // m --> m - 1
	}
}

void SEC18_MaOP2(double* x, double* fit, double* constrainV, int nx, int M)//OK
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}

	for (int n = M; n <= nx; n++) {
		if ((n % 5) == 0) {
			g = g + (x[n - 1] - tmp) * (x[n - 1] - tmp);
		}
		else {
			g = g + (x[n - 1] - 0.5) * (x[n - 1] - 0.5); // n --> n-1   2018.5.8
		}
	}

	g = 200 * g;

	double tmp2 = 1;

	for (int m = M; m >= 1; m--) {
		int p = (int)(pow(2, (m % 2) + 1));
		if (m == M) {
			fit[m - 1] = (1 + g) * pow(sin(0.5 * x[0] * PI), p);
		}
		else if (m < M && m >= 2) {
			tmp2 = tmp2 * cos(0.5 * PI * x[M - m - 1]);
			fit[m - 1] = (1 + g) * pow(tmp2 * sin(0.5 * PI * x[M - m]), p);
		}
		else {
			fit[m - 1] = (1 + g) * pow(tmp2 * cos(0.5 * PI * x[M - 2]), p);
		}
	}
}

void SEC18_MaOP3(double* x, double* fit, double* constrainV, int nx, int M)
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}
	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + n * pow(fabs(x[n - 1] - tmp), 0.1);
		}
		else {
			g = g + n * pow(fabs(x[n - 1] - 0.5), 0.1);
		}
	}

	double tmp2 = 1;

	for (int m = M; m >= 1; m--) {
		if (m == M) {
			fit[m - 1] = (1 + g) * sin(0.5 * x[0] * PI);
		}
		else if (m < M && m >= 2) {
			tmp2 = tmp2 * cos(0.5 * PI * x[M - m - 1]);
			fit[m - 1] = (1 + g) * tmp2 * sin(0.5 * PI * x[M - m]);
		}
		else {
			fit[m - 1] = (1 + g) * tmp2 * cos(0.5 * PI * x[M - 2]);
		}
	}
}

void SEC18_MaOP4(double* x, double* fit, double* constrainV, int nx, int M)
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}
	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + (fabs(-0.9 * pow(x[n - 1] - tmp, 2)) + pow(fabs(x[n - 1] - tmp), 0.6));
		}
		else {
			g = g + (fabs(-0.9 * pow(x[n - 1] - 0.5, 2)) + pow(fabs(x[n - 1] - 0.5), 0.6));
		}
	}

	g = g * 20 * sin(x[0] * PI);

	double tmp2 = 1;

	for (int m = M; m >= 1; m--) {
		if (m == M) {
			fit[m - 1] = (1 + g) * sin(0.5 * x[0] * PI);
		}
		else if (m < M && m >= 2) {
			tmp2 = tmp2 * cos(0.5 * PI * x[M - m - 1]);
			fit[m - 1] = (1 + g) * tmp2 * sin(0.5 * PI * x[M - m]);
		}
		else {
			fit[m - 1] = (1 + g) * tmp2 * cos(0.5 * PI * x[M - 2]);
		}
	}
}

void SEC18_MaOP5(double* x, double* fit, double* constrainV, int nx, int M)
{
	double* g = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M; m++) {
		if (m <= 3) {
			double sum = 0;
			for (int j = 3; j <= nx; j++) {
				sum = sum + pow(x[j - 1] - x[0] * x[1], 2);
			}
			g[m - 1] = max(0, -1.4 * cos(2 * x[0] * PI)) + sum;
		}
		else {
			g[m - 1] = exp(pow(x[m - 1] - x[0] * x[1], 2)) - 1;  // order of pow and exp is not correct.
		}
		g[m - 1] = 10 * g[m - 1];
	}
	double alpha1 = cos(0.5 * PI * x[0]) * cos(0.5 * PI * x[1]);
	double alpha2 = cos(0.5 * PI * x[0]) * sin(0.5 * PI * x[1]);//??
	double alpha3 = sin(0.5 * PI * x[0]);
	fit[0] = (1 + g[0]) * alpha1;
	fit[1] = 4 * (1 + g[1]) * alpha2;
	fit[2] = (1 + g[2]) * alpha3;
	for (int m = 4; m <= M; m++) {
		double ratio = 1.0 * m / M;                                                             // m/nobj equals to zero
		fit[m - 1] = (1 + g[m - 1]) * (ratio * alpha1 + (1.0 - ratio) * alpha2 + sin(0.5 * m * PI / M) * alpha3);//??
	}

	free(g);
}

void SEC18_MaOP6(double* x, double* fit, double* constrainV, int nx, int M)
{
	double* g = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M; m++) {
		if (m <= 3) {
			double sum = 0;
			for (int j = 3; j <= nx; j++) {//??
				sum = sum + pow(x[j - 1] - x[0] * x[1], 2);
			}
			g[m - 1] = max(0, 1.4 * sin(4 * x[0] * PI)) + sum;
		}
		else {
			g[m - 1] = exp(pow(x[m - 1] - x[0] * x[1], 2)) - 1;//??
		}
		g[m - 1] = g[m - 1] * 10;
	}

	double alpha1 = x[0] * x[1];
	double alpha2 = x[0] * (1 - x[1]);
	double alpha3 = (1 - x[0]);
	fit[0] = (1 + g[0]) * alpha1;
	fit[1] = 2 * (1 + g[1]) * alpha2;
	fit[2] = 6 * (1 + g[2]) * alpha3;
	for (int m = 4; m <= M; m++) {
		double ratio = 1.0 * m / M;
		fit[m - 1] = (1 + g[m - 1]) * (ratio * alpha1 + (1.0 - ratio) * alpha2 + sin(0.5 * m * PI / M) * alpha3);
	}

	free(g);
}

void SEC18_MaOP7(double* x, double* fit, double* constrainV, int nx, int M)
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}

	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + pow(x[n - 1] - tmp, 2);
		}
		else {
			g = g + pow(x[n - 1] - 0.5, 2);
		}
	}

	g = 100 * g;

	double* alpha = (double*)calloc(M, sizeof(double));
	double tau = sqrt(2) / 2;

	alpha[0] = -pow(2 * x[0] - 1, 3) + 1;
	int  T = (int)(floor((M - 1.0) / 2.0));
	for (int i = 1; i <= T; i++) {
		alpha[2 * i - 1] = x[0] + (2 * x[i] - 1) * tau + tau * pow(fabs(2 * x[i] - 1), 0.5 + x[0]);//??
		alpha[2 * i] = x[0] - (2 * x[i] - 2) * tau + tau * pow(fabs(2 * x[i] - 1), 0.5 + x[0]);
	}
	if (M % 2 == 0) {
		alpha[M - 1] = 1 - alpha[0];
	}

	for (int m = 1; m <= M; m++) {
		fit[m - 1] = (1 + g) * alpha[m - 1];
	}
	free(alpha);
}

void SEC18_MaOP8(double* x, double* fit, double* constrainV, int nx, int M)
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}

	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + pow(x[n - 1] - tmp, 2);
		}
		else {
			g = g + pow(x[n - 1] - 0.5, 2);
		}
	}

	g = g * 100;

	double* alpha = (double*)calloc(M, sizeof(double));
	double tau = sqrt(2) / 2;

	alpha[0] = -pow(2 * x[0] - 1, 3) + 1;
	int  T = (int)(floor((M - 1.0) / 2.0));
	for (int i = 1; i <= T; i++) {
		alpha[2 * i - 1] = x[0] + (2 * x[i]) * tau + tau * pow(fabs(2 * x[i] - 1), 1 - 0.5 * sin(4 * PI * x[0]));//??
		alpha[2 * i] = x[0] - (2 * x[i] - 2) * tau + tau * pow(fabs(2 * x[i] - 1), 1 - 0.5 * sin(4 * PI * x[0]));
	}

	if (M % 2 == 0) {
		alpha[M - 1] = 1 - alpha[0];
	}

	for (int m = 1; m <= M; m++) {
		fit[m - 1] = (1 + g) * alpha[m - 1];
	}
	free(alpha);
}

void SEC18_MaOP9(double* x, double* fit, double* constrainV, int nx, int M)//OK
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}

	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + pow(x[n - 1] - tmp, 2);
		}
		else {
			g = g + pow(x[n - 1] - 0.5, 2);
		}
	}

	g = g * 100;

	double* alpha = (double*)calloc(M, sizeof(double));
	double tau = sqrt(2) / 2;

	alpha[0] = -pow(2 * x[0] - 1, 3) + 1;
	int  T = (int)(floor((M - 1.0) / 2.0));
	for (int i = 1; i <= T; i++) {
		double z = 2 * (2 * x[i] - floor(2 * x[i])) - 1;
		alpha[2 * i - 1] = x[0] + 2 * x[i] * tau + tau * pow(fabs(z), 0.5 + x[0]);
		alpha[2 * i] = x[0] - (2 * x[i] - 2) * tau + tau * pow(fabs(z), 0.5 + x[0]);
	}

	if (M % 2 == 0) {
		alpha[M - 1] = 1 - alpha[0];
	}

	for (int m = 1; m <= M; m++) {
		fit[m - 1] = (1 + g) * alpha[m - 1];
	}
	free(alpha);
}

void SEC18_MaOP10(double* x, double* fit, double* constrainV, int nx, int M)
{
	double g = 0;
	double tmp = 1;
	for (int i = 0; i < M - 1; i++) {
		tmp *= sin(0.5 * PI * x[i]);
	}

	for (int n = M; n <= nx; n++) {
		if (n % 5 == 0) {
			g = g + pow(x[n - 1] - tmp, 2);
		}
		else {
			g = g + pow(x[n - 1] - 0.5, 2);
		}
	}

	g = g * 100;

	double* alpha = (double*)calloc(M, sizeof(double));
	double tau = sqrt(2) / 2, p, z;

	alpha[0] = -pow(2 * x[0] - 1, 3) + 1;
	int  T = (int)(floor((M - 1.0) / 2.0));
	for (int i = 1; i <= T; i++) {
		z = 2 * (2 * x[i] - floor(2 * x[i])) - 1;  // i+1 --> i  2018.5.8
		if (x[i] < 0.5) {//??
			p = 0.5 + x[0];
		}
		else {
			p = 1.5 - x[0];
		}

		alpha[2 * i - 1] = x[0] + 2 * x[i] * tau + tau * pow(fabs(z), p);
		alpha[2 * i] = x[0] - (2 * x[i] - 2) * tau + tau * pow(fabs(z), p);
	}

	if (M % 2 == 0) {
		alpha[M - 1] = 1 - alpha[0];
	}

	for (int m = 1; m <= M; m++) {
		fit[m - 1] = (1 + g) * alpha[m - 1];
	}
	free(alpha);
}