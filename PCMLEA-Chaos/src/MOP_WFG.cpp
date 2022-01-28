#include "MOP_WFG.h"

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

int WFG_nvar, WFG_nobj;                    //  the number of variables and objectives
int WFG_position_parameters;
char WFG_testInstName[1024];

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Implementation
void InitPara_WFG(char* instName, int numObj, int numVar, int posPara)
{
	strcpy(WFG_testInstName, instName);
	WFG_nobj = numObj;
	WFG_nvar = numVar;
	WFG_position_parameters = posPara;

	return;
}
void SetLimits_WFG(double* minLimit, double* maxLimit, int dim)
{
	for (int i = 0; i < dim; i++) {
		minLimit[i] = 0.0;
		maxLimit[i] = 2 * (i + 1.0);
	}
	return;
}
int CheckLimits_WFG(double* x, int dim)
{
	for (int i = 0; i < dim; i++) {
		if (x[i] < 0.0 || x[i] > 2 * (i + 1.0)) {
			printf("Check limits FAIL - %s\n", WFG_testInstName);
			return false;
		}
	}
	return true;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Example problems
//** True if "k" in [1,z.size()), "M" >= 2, and "k" mod ("M"-1) == 0. *******
inline bool ArgsOK(double* z, const int k, const int M)
{
	const int n = WFG_nvar;

	return k >= 1 && k < n&& M >= 2 && k % (M - 1) == 0;
}

////** The WFG1 problem. ******************************************************
//void WFG1(double* x, double* fit, const int k, const int M);
////** The WFG2 problem. ******************************************************
//void WFG2(double* x, double* fit, const int k, const int M);
////** The WFG3 problem. ******************************************************
//void WFG3(double* x, double* fit, const int k, const int M);
////** The WFG4 problem. ******************************************************
//void WFG4(double* x, double* fit, const int k, const int M);
////** The WFG5 problem. ******************************************************
//void WFG5(double* x, double* fit, const int k, const int M);
////** The WFG6 problem. ******************************************************
//void WFG6(double* x, double* fit, const int k, const int M);
////** The WFG7 problem. ******************************************************
//void WFG7(double* x, double* fit, const int k, const int M);
////** The WFG8 problem. ******************************************************
//void WFG8(double* x, double* fit, const int k, const int M);
////** The WFG9 problem. ******************************************************
//void WFG9(double* x, double* fit, const int k, const int M);
//
////** The I1 problem. ********************************************************
//void I1(double* x, double* fit, const int k, const int M);
////** The I2 problem. ********************************************************
//void I2(double* x, double* fit, const int k, const int M);
////** The I3 problem. ********************************************************
//void I3(double* x, double* fit, const int k, const int M);
////** The I4 problem. ********************************************************
//void I4(double* x, double* fit, const int k, const int M);
////** The I5 problem. ********************************************************
//void I5(double* x, double* fit, const int k, const int M);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Example Shapes
//** Construct a vector of length M-1, with values "1,0,0,..." if ***********
//** "degenerate" is true, otherwise with values "1,1,1,..." if   ***********
//** "degenerate" is false.                                       ***********
void WFG_create_A(short* A, const int M, const bool degenerate);

//** Given the vector "x" (the last value of which is the sole distance ****
//** parameter), and the shape function results in "h", calculate the   ****
//** scaled fitness values for a WFG problem.                           ****
void WFG_calculate_f(double* x, int& size_x, double* h, int size_h);

void WFG_normalise_z(double* z, double* y);

//** Given the last transition vector, get the fitness values. *****
void WFG1_shape(double* t_p, int& size);
void WFG2_shape(double* t_p, int& size);
void WFG3_shape(double* t_p, int& size);
void WFG4_shape(double* t_p, int& size);
void I1_shape(double* t_p, int& size);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Example Transitions
void WFG1_t1(double* y, int& size, const int k);
void WFG1_t2(double* y, int& size, const int k);
void WFG1_t3(double* y, int& size);
void WFG1_t4(double* y, int& size, const int k, const int M);
void WFG2_t2(double* y, int& size, const int k);
void WFG2_t3(double* y, int& size, const int k, const int M);
void WFG4_t1(double* y, int& size);
void WFG5_t1(double* y, int& size);
void WFG6_t2(double* y, int& size, const int k, const int M);
void WFG7_t1(double* y, int& size, const int k);
void WFG8_t1(double* y, int& size, const int k);
void WFG9_t1(double* y, int& size);
void WFG9_t2(double* y, int& size, const int k);
void I1_t2(double* y, int& size, const int k);
void I1_t3(double* y, int& size, const int k, const int M);
void I2_t1(double* y, int& size);
void I3_t1(double* y, int& size);
void I4_t3(double* y, int& size_y, const int k, const int M);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Framework Functions
//** Normalise the elements of "z" to the domain [0,1]. *********************
void normalise_z(double* z, int size_z, double* z_max, int size_z_max);

//** Degenerate the values of "t_p" based on the degeneracy vector "A". *****
void calculate_x(double* t_p, int& size_t_p, short* A, int size_A);

//** Calculate the fitness vector using the distance scaling constant D, ****
//** the distance parameter in "x", the shape function values in "h",    ****
//** and the scaling constants in "S".                                   ****
void calculate_f(const double& D, double* x, int& size_x, double* h, int size_h, double* S, int size_S);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Misc
//** Used to correct values in [-epislon,0] to 0, and [1,epsilon] to 1. *****
double correct_to_01(const double& a, const double& epsilon = 1.0e-10);

//** Returns true if all elements of "x" are in [0,1], false otherwise. *****
bool vector_in_01(double* x, int size);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	Shape Functions
//** True if all elements of "x" are in [0,1], and m is in [1, x.size()]. ***
bool shape_args_ok(double* x, int size, const int m);

//** The linear shape function. (m is indexed from 1.) **********************
double linear(double* x, int size, const int m);

//** The convex shape function. (m is indexed from 1.) **********************
double convex(double* x, int size, const int m);

//** The concave shape function. (m is indexed from 1.) *********************
double concave(double* x, int size, const int m);

//** The mixed convex/concave shape function. *******************************
double mixed(double* x, int size, const int A, const double& alpha);

//** The disconnected shape function. ***************************************
double disc(double* x, int size, const int A, const double& alpha, const double& beta);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//	TransFunctions
//** Calculate the minimum of two doubles. **********************************
inline double min(const double& a, const double& b)
{
	double m = a > b ? b : a;
	return m;// std::min< const double >(a, b);
}

//** The polynomial bias transformation function. ***************************
double b_poly(const double& y, const double& alpha);

//** The flat region bias transformation function. **************************
double b_flat(const double& y, const double& A, const double& B, const double& C);

//** The parameter strct_grp_ana_vals.Dependent bias transformation function. ******************
double b_param(const double& y, const double& u,
	const double& A, const double& B, const double& C);

//** The linear shift transformation function. ******************************
double s_linear(const double& y, const double& A);

//** The deceptive shift transformation function. ***************************
double s_decept(const double& y, const double& A, const double& B, const double& C);

//** The multi-modal shift transformation function. *************************
double s_multi(const double& y, const int A, const double& B, const double& C);

//** The weighted sum reduction transformation function. ********************
double r_sum(double* y, int size_y, double* w, int size_w);

//** The non-separable reduction transformation function. *******************
double r_nonsep(double* y, int size, const int A);
//////////////////////////////////////////////////////////
//C

//////////////////////////////////////////////////////////////////////////
//	WFG

//	Example problems
//	//////////////////////////////////////////////////////////////////////////
//** Reduces each paramer in "z" to the domain [0,1]. ***********************
void WFG_normalise_z(double* z, double* y)
{
	for (int i = 0; i < WFG_nvar; i++) {
		const double bound = 2.0 * (i + 1);

		//	assert( z[i] >= 0.0   );
		//	assert( z[i] <= bound );

		y[i] = (z[i] / bound);
	}

	return;
}

void WFG1(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG1_t1(y, size_y, _k);
	WFG1_t2(y, size_y, _k);
	WFG1_t3(y, size_y);
	WFG1_t4(y, size_y, _k, M);

	WFG1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG2(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));
	assert((WFG_nvar - _k) % 2 == 0);

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG1_t1(y, size_y, _k);
	WFG2_t2(y, size_y, _k);
	WFG2_t3(y, size_y, _k, M);

	WFG2_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG3(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));
	assert((WFG_nvar - _k) % 2 == 0);

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG1_t1(y, size_y, _k);
	WFG2_t2(y, size_y, _k);
	WFG2_t3(y, size_y, _k, M);

	WFG3_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG4(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG4_t1(y, size_y);
	WFG2_t3(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG5(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG5_t1(y, size_y);
	WFG2_t3(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG6(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG1_t1(y, size_y, _k);
	WFG6_t2(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG7(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG7_t1(y, size_y, _k);
	WFG1_t1(y, size_y, _k);
	WFG2_t3(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG8(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG8_t1(y, size_y, _k);
	WFG1_t1(y, size_y, _k);
	WFG2_t3(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void WFG9(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	WFG_normalise_z(x, y);
	int size_y = WFG_nvar;

	WFG9_t1(y, size_y);
	WFG9_t2(y, size_y, _k);
	WFG6_t2(y, size_y, _k, M);

	WFG4_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void I1(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	memcpy(y, &x[0], WFG_nvar * sizeof(double));
	int size_y = WFG_nvar;

	I1_t2(y, size_y, _k);
	I1_t3(y, size_y, _k, M);

	I1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void I2(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	memcpy(y, &x[0], WFG_nvar * sizeof(double));
	int size_y = WFG_nvar;

	I2_t1(y, size_y);
	I1_t2(y, size_y, _k);
	I1_t3(y, size_y, _k, M);

	I1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void I3(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	memcpy(y, &x[0], WFG_nvar * sizeof(double));
	int size_y = WFG_nvar;

	I3_t1(y, size_y);
	I1_t2(y, size_y, _k);
	I1_t3(y, size_y, _k, M);

	I1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void I4(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	memcpy(y, &x[0], WFG_nvar * sizeof(double));
	int size_y = WFG_nvar;

	I1_t2(y, size_y, _k);
	I4_t3(y, size_y, _k, M);

	I1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

void I5(
	double* x,
	double* fit,
	double* constrainV,
	int nx,
	int M
)
{
	int _k = WFG_position_parameters;

	assert(ArgsOK(x, _k, M));

	double* y = (double*)calloc(WFG_nvar, sizeof(double));
	memcpy(y, &x[0], WFG_nvar * sizeof(double));
	int size_y = WFG_nvar;

	I3_t1(y, size_y);
	I1_t2(y, size_y, _k);
	I4_t3(y, size_y, _k, M);

	I1_shape(y, size_y);

	memcpy(fit, y, WFG_nobj * sizeof(double));

	free(y);

	return;
}

//	Example Shapes
//////////////////////////////////////////////////////////////////////////
//** Construct a vector of length M-1, with values "1,0,0,..." if ***********
//** "degenerate" is true, otherwise with values "1,1,1,..." if   ***********
//** "degenerate" is false.                                       ***********
void WFG_create_A(short* A, const int M, const bool degenerate)
{
	assert(M >= 2);

	if (degenerate) {
		for (int i = 0; i < M - 1; i++) A[i] = 0;
		A[0] = 1;
	}
	else {
		for (int i = 0; i < M - 1; i++) A[i] = 1;
	}

	return;
}

//** Given the vector "x" (the last value of which is the sole distance ****
//** parameter), and the shape function results in "h", calculate the   ****
//** scaled fitness values for a WFG problem.                           ****
void WFG_calculate_f
(
	double* x,
	int& size_x,
	double* h,
	int size_h
)
{
	assert(vector_in_01(x, size_x));
	assert(vector_in_01(h, size_h));
	assert(size_x == size_h);

	const int M = size_h;

	double* S = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M; m++) {
		S[m - 1] = (m * 2.0);
	}

	calculate_f(1.0, x, size_x, h, size_h, S, M);

	free(S);

	return;
}

void WFG1_shape(double* t_p, int& size)
{
	assert(vector_in_01(t_p, size));
	assert(size >= 2);

	const int M = size;

	short* A = (short*)calloc(M - 1, sizeof(short));
	WFG_create_A(A, M, false);
	calculate_x(t_p, size, A, M - 1);

	double* h = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M - 1; m++) {
		h[m - 1] = (convex(t_p, size, m));
	}
	h[M - 1] = (mixed(t_p, size, 5, 1.0));

	WFG_calculate_f(t_p, size, h, M);

	free(A);
	free(h);

	return;
}

void WFG2_shape(double* t_p, int& size)
{
	assert(vector_in_01(t_p, size));
	assert(size >= 2);

	const int M = size;

	short* A = (short*)calloc(M - 1, sizeof(short));
	WFG_create_A(A, M, false);
	calculate_x(t_p, size, A, M - 1);

	double* h = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M - 1; m++) {
		h[m - 1] = (convex(t_p, size, m));
	}
	h[M - 1] = (disc(t_p, size, 5, 1.0, 1.0));

	WFG_calculate_f(t_p, size, h, M);

	free(A);
	free(h);

	return;
}

void WFG3_shape(double* t_p, int& size)
{
	assert(vector_in_01(t_p, size));
	assert(size >= 2);

	const int M = size;

	short* A = (short*)calloc(M - 1, sizeof(short));
	WFG_create_A(A, M, true);
	calculate_x(t_p, size, A, M - 1);

	double* h = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M; m++) {
		h[m - 1] = (linear(t_p, size, m));
	}

	WFG_calculate_f(t_p, size, h, M);

	free(A);
	free(h);

	return;
}

void WFG4_shape(double* t_p, int& size)
{
	assert(vector_in_01(t_p, size));
	assert(size >= 2);

	const int M = size;

	short* A = (short*)calloc(M - 1, sizeof(short));
	WFG_create_A(A, M, false);
	calculate_x(t_p, size, A, M - 1);

	double* h = (double*)calloc(M, sizeof(double));

	for (int m = 1; m <= M; m++) {
		h[m - 1] = (concave(t_p, size, m));
	}

	WFG_calculate_f(t_p, size, h, M);

	free(A);
	free(h);

	return;
}

void I1_shape(double* t_p, int& size)
{
	assert(vector_in_01(t_p, size));
	assert(size >= 2);

	const int M = size;

	short* A = (short*)calloc(M - 1, sizeof(short));
	WFG_create_A(A, M, false);
	calculate_x(t_p, size, A, M - 1);

	double* h = (double*)calloc(M, sizeof(double));
	double* w = (double*)calloc(M, sizeof(double));
	for (int m = 0; m < M; m++) w[m] = 1.0;

	for (int m = 1; m <= M; m++) {
		h[m - 1] = (concave(t_p, size, m));
	}

	calculate_f(1.0, t_p, size, h, M, w, M);

	free(A);
	free(h);
	free(w);

	return;
}

//	Example Transitions
//	//////////////////////////////////////////////////////////////////////////
//** Construct a vector with the elements v[head], ..., v[tail-1]. **********
void WFG1_t1
(
	double* y,
	int& size,
	const int k
)
{
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);

	for (int i = k; i < n; i++) {
		y[i] = (s_linear(y[i], 0.35));
	}

	return;
}

void WFG1_t2
(
	double* y,
	int& size,
	const int k
)
{
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);

	for (int i = k; i < n; i++) {
		y[i] = (b_flat(y[i], 0.8, 0.75, 0.85));
	}

	return;
}

void WFG1_t3(double* y, int& size)
{
	const int n = size;

	assert(vector_in_01(y, n));

	for (int i = 0; i < n; i++) {
		y[i] = (b_poly(y[i], 0.02));
	}

	return;
}

void WFG1_t4
(
	double* y,
	int& size,
	const int k,
	const int M
)
{
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);
	assert(M >= 2);
	assert(k % (M - 1) == 0);

	double* w = (double*)calloc(n, sizeof(double));

	for (int i = 1; i <= n; i++) {
		w[i - 1] = (2.0 * i);
	}

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (int i = 1; i <= M - 1; i++) {
		const int head = (i - 1) * k / (M - 1);
		const int tail = i * k / (M - 1);

		y[i - 1] = (r_sum(&y_sub[head], tail - head, &w_sub[head], tail - head));
	}

	y[M - 1] = (r_sum(&y_sub[k], n - k, &w_sub[k], n - k));
	size = M;

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void WFG2_t2
(
	double* y,
	int& size,
	const int k
)
{
	const int n = size;
	const int l = n - k;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);
	assert(l % 2 == 0);

	double* y_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));

	for (int i = k + 1; i <= k + l / 2; i++) {
		const int head = k + 2 * (i - k) - 2;
		const int tail = k + 2 * (i - k);

		y[i - 1] = (r_nonsep(&y_sub[head], tail - head, 2));
	}

	size = k + l / 2;

	free(y_sub);

	return;
}

void WFG2_t3
(
	double* y,
	int& size,
	const int k,
	const int M
)
{
	//int i;
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);
	assert(M >= 2);
	assert(k % (M - 1) == 0);

	double* w = (double*)calloc(n, sizeof(double));
	for (int i = 0; i < n; i++) w[i] = 1.0;

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (int i = 1; i <= M - 1; i++) {
		const int head = (i - 1) * k / (M - 1);
		const int tail = i * k / (M - 1);

		y[i - 1] = (r_sum(&y_sub[head], tail - head, &w_sub[head], tail - head));
	}

	y[M - 1] = (r_sum(&y_sub[k], n - k, &w_sub[k], n - k));
	size = M;

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void WFG4_t1(double* y, int& size)
{
	const int n = size;

	assert(vector_in_01(y, size));

	for (int i = 0; i < n; i++) {
		y[i] = (s_multi(y[i], 30, 10, 0.35));
	}

	return;
}

void WFG5_t1(double* y, int& size)
{
	const int n = size;

	assert(vector_in_01(y, size));

	for (int i = 0; i < n; i++) {
		y[i] = (s_decept(y[i], 0.35, 0.001, 0.05));
	}

	return;
}

void WFG6_t2
(
	double* y,
	int& size,
	const int k,
	const int M
)
{
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);
	assert(M >= 2);
	assert(k % (M - 1) == 0);

	double* y_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));

	for (int i = 1; i <= M - 1; i++) {
		const int head = (i - 1) * k / (M - 1);
		const int tail = i * k / (M - 1);

		y[i - 1] = (r_nonsep(&y_sub[head], (tail - head), k / (M - 1)));
	}

	y[M - 1] = (r_nonsep(&y_sub[k], n - k, n - k));
	size = M;

	free(y_sub);

	return;
}

void WFG7_t1
(
	double* y,
	int& size,
	const int k
)
{
	int i;
	const int n = size;

	assert(vector_in_01(y, n));
	assert(k >= 1);
	assert(k < n);

	double* w = (double*)calloc(n, sizeof(double));
	for (i = 0; i < n; i++) w[i] = 1.0;

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (i = 0; i < k; i++) {
		const double u = r_sum(&y_sub[i + 1], n - i - 1, &w_sub[i + 1], n - i - 1);

		y[i] = (b_param(y[i], u, 0.98 / 49.98, 0.02, 50));
	}

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void WFG8_t1
(
	double* y,
	int& size,
	const int k
)
{
	//int i;
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);

	double* w = (double*)calloc(n, sizeof(double));
	for (int i = 0; i < n; i++) w[i] = 1.0;

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (int i = k; i < n; i++) {
		const double u = r_sum(y_sub, i, w_sub, i);

		y[i] = (b_param(y[i], u, 0.98 / 49.98, 0.02, 50));
	}

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void WFG9_t1(double* y, int& size)
{
	int i;
	const int n = size;

	assert(vector_in_01(y, n));

	double* w = (double*)calloc(n, sizeof(double));
	for (i = 0; i < n; i++) w[i] = 1.0;

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (i = 0; i < n - 1; i++) {
		const double u = r_sum(&y_sub[i + 1], n - i - 1, &w_sub[i + 1], n - i - 1);

		y[i] = (b_param(y[i], u, 0.98 / 49.98, 0.02, 50));
	}

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void WFG9_t2
(
	double* y,
	int& size,
	const int k
)
{
	const int n = size;

	assert(vector_in_01(y, size));
	assert(k >= 1);
	assert(k < n);

	for (int i = 0; i < k; i++) {
		y[i] = (s_decept(y[i], 0.35, 0.001, 0.05));
	}

	for (int i = k; i < n; i++) {
		y[i] = (s_multi(y[i], 30, 95, 0.35));
	}

	return;
}

void I1_t2
(
	double* y,
	int& size,
	const int k
)
{
	WFG1_t1(y, size, k);
}

void I1_t3
(
	double* y,
	int& size,
	const int k,
	const int M
)
{
	WFG2_t3(y, size, k, M);
}

void I2_t1(double* y, int& size)
{
	WFG9_t1(y, size);
}

void I3_t1(double* y, int& size)
{
	int i;
	const int n = size;

	assert(vector_in_01(y, size));

	double* w = (double*)calloc(n, sizeof(double));
	for (i = 0; i < n; i++) w[i] = 1.0;

	double* y_sub = (double*)calloc(n, sizeof(double)), * w_sub = (double*)calloc(n, sizeof(double));
	memcpy(y_sub, y, n * sizeof(double));
	memcpy(w_sub, w, n * sizeof(double));

	for (i = 1; i < n; i++) {
		const double u = r_sum(y_sub, i, w_sub, i);

		y[i] = (b_param(y[i], u, 0.98 / 49.98, 0.02, 50));
	}

	free(w);
	free(y_sub);
	free(w_sub);

	return;
}

void I4_t3
(
	double* y,
	int& size_y,
	const int k,
	const int M
)
{
	WFG6_t2(y, size_y, k, M);
}

//	Framework Functions
//	//////////////////////////////////////////////////////////////////////////
void normalise_z
(
	double* z,
	int size_z,
	double* z_max,
	int size_z_max
)
{
	for (int i = 0; i < size_z; i++) {
		assert(z[i] >= 0.0);
		assert(z[i] <= z_max[i]);
		assert(z_max[i] > 0.0);

		z[i] = (z[i] / z_max[i]);
	}

	return;
}

void calculate_x
(
	double* t_p,
	int& size_t_p,
	short* A,
	int size_A
)
{
	assert(vector_in_01(t_p, size_t_p));
	assert(size_t_p != 0);
	assert(size_A == size_t_p - 1);

	for (int i = 0; i < size_t_p - 1; i++) {
		assert(A[i] == 0 || A[i] == 1);

		const double tmp1 = (t_p[size_t_p - 1] > A[i]) ? t_p[size_t_p - 1] : A[i]; // std::max< double >(t_p.back(), A[i]);
		t_p[i] = (tmp1 * (t_p[i] - 0.5) + 0.5);
	}

	return;
}

void  calculate_f
(
	const double& D,
	double* x,
	int& size_x,
	double* h,
	int size_h,
	double* S,
	int size_S
)
{
	assert(D > 0.0);
	assert(vector_in_01(x, size_x));
	assert(vector_in_01(h, size_h));
	assert(size_x == size_h);
	assert(size_h == size_S);

	for (int i = 0; i < size_x; i++) {
		assert(S[i] > 0.0);

		x[i] = (D * x[size_x - 1] + S[i] * h[i]);
	}

	return;
}

//	Misc
//	//////////////////////////////////////////////////////////////////////////
double correct_to_01(const double& a, const double& epsilon)
{
	assert(epsilon >= 0.0);

	const double min = 0.0;
	const double max = 1.0;

	const double min_epsilon = min - epsilon;
	const double max_epsilon = max + epsilon;

	if (a <= min && a >= min_epsilon) {
		return min;
	}
	else if (a >= max && a <= max_epsilon) {
		return max;
	}
	else {
		return a;
	}
}

bool vector_in_01(double* x, int size)
{
	for (int i = 0; i < size; i++) {
		if (x[i] < 0.0 || x[i] > 1.0) {
			return false;
		}
	}

	return true;
}

//	Shape Functions
//	//////////////////////////////////////////////////////////////////////////
//** True if all elements of "x" are in [0,1], and m is in [1, x.size()]. ***
bool shape_args_ok(double* x, int size, const int m)
{
	const int M = size;

	return vector_in_01(x, M) && m >= 1 && m <= M;
}

double linear(double* x, int size, const int m)
{
	assert(shape_args_ok(x, size, m));

	const int M = size;
	double result = 1.0;

	for (int i = 1; i <= M - m; i++) {
		result *= x[i - 1];
	}

	if (m != 1) {
		result *= 1 - x[M - m];
	}

	return correct_to_01(result);
}

double convex(double* x, int size, const int m)
{
	assert(shape_args_ok(x, size, m));

	const int M = size;
	double result = 1.0;

	for (int i = 1; i <= M - m; i++) {
		result *= 1.0 - cos(x[i - 1] * PI / 2.0);
	}

	if (m != 1) {
		result *= 1.0 - sin(x[M - m] * PI / 2.0);
	}

	return correct_to_01(result);
}

double concave(double* x, int size, const int m)
{
	assert(shape_args_ok(x, size, m));

	const int M = size;
	double result = 1.0;

	for (int i = 1; i <= M - m; i++) {
		result *= sin(x[i - 1] * PI / 2.0);
	}

	if (m != 1) {
		result *= cos(x[M - m] * PI / 2.0);
	}

	return correct_to_01(result);
}

double mixed
(
	double* x,
	int size,
	const int A,
	const double& alpha
)
{
	assert(vector_in_01(x, size));
	assert(size != 0);
	assert(A >= 1);
	assert(alpha > 0.0);

	const double tmp = 2.0 * A * PI;

	return correct_to_01(pow(1.0 - x[0] - cos(tmp * x[0] + PI / 2.0) / tmp, alpha));
}

double disc
(
	double* x,
	int size,
	const int A,
	const double& alpha,
	const double& beta
)
{
	assert(vector_in_01(x, size));
	assert(size != 0);
	assert(A >= 1);
	assert(alpha > 0.0);
	assert(beta > 0.0);

	const double tmp1 = A * pow(x[0], beta) * PI;
	return correct_to_01(1.0 - pow(x[0], alpha) * pow(cos(tmp1), 2.0));
}

//	TransFunctions
//	//////////////////////////////////////////////////////////////////////////
double b_poly(const double& y, const double& alpha)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(alpha > 0.0);
	assert(alpha != 1.0);

	return correct_to_01(pow(y, alpha));
}

double b_flat
(
	const double& y,
	const double& A,
	const double& B,
	const double& C
)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(A >= 0.0);
	assert(A <= 1.0);
	assert(B >= 0.0);
	assert(B <= 1.0);
	assert(C >= 0.0);
	assert(C <= 1.0);
	assert(B < C);
	assert(B != 0.0 || A == 0.0);
	assert(B != 0.0 || C != 1.0);
	assert(C != 1.0 || A == 1.0);
	assert(C != 1.0 || B != 0.0);

	const double tmp1 = min(0.0, floor(y - B)) * A * (B - y) / B;
	const double tmp2 = min(0.0, floor(C - y)) * (1.0 - A) * (y - C) / (1.0 - C);

	return correct_to_01(A + tmp1 - tmp2);
}

double b_param
(
	const double& y,
	const double& u,
	const double& A,
	const double& B,
	const double& C
)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(u >= 0.0);
	assert(u <= 1.0);
	assert(A > 0.0);
	assert(A < 1.0);
	assert(B > 0.0);
	assert(B < C);

	const double v = A - (1.0 - 2.0 * u) * fabs(floor(0.5 - u) + A);

	return correct_to_01(pow(y, B + (C - B) * v));
}

double s_linear(const double& y, const double& A)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(A > 0.0);
	assert(A < 1.0);

	return correct_to_01(fabs(y - A) / fabs(floor(A - y) + A));
}

double s_decept
(
	const double& y,
	const double& A,
	const double& B,
	const double& C
)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(A > 0.0);
	assert(A < 1.0);
	assert(B > 0.0);
	assert(B < 1.0);
	assert(C > 0.0);
	assert(C < 1.0);
	assert(A - B > 0.0);
	assert(A + B < 1.0);

	const double tmp1 = floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B);
	const double tmp2 = floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B);

	return correct_to_01(1.0 + (fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B));
}

double s_multi
(
	const double& y,
	const int A,
	const double& B,
	const double& C
)
{
	assert(y >= 0.0);
	assert(y <= 1.0);
	assert(A >= 1);
	assert(B >= 0.0);
	assert((4.0 * A + 2.0) * PI >= 4.0 * B);
	assert(C > 0.0);
	assert(C < 1.0);

	const double tmp1 = fabs(y - C) / (2.0 * (floor(C - y) + C));
	const double tmp2 = (4.0 * A + 2.0) * PI * (0.5 - tmp1);

	return correct_to_01((1.0 + cos(tmp2) + 4.0 * B * pow(tmp1, 2.0)) / (B + 2.0));
}

double r_sum
(
	double* y,
	int size_y,
	double* w,
	int size_w
)
{
	assert(size_y != 0);
	assert(size_w == size_y);
	assert(vector_in_01(y, size_y));

	double numerator = 0.0;
	double denominator = 0.0;

	for (int i = 0; i < size_y; i++) {
		assert(w[i] > 0.0);

		numerator += w[i] * y[i];
		denominator += w[i];
	}

	return correct_to_01(numerator / denominator);
}

double r_nonsep(double* y, int size, const int A)
{
	const int y_len = size;

	assert(y_len != 0);
	assert(vector_in_01(y, y_len));
	assert(A >= 1);
	assert(A <= y_len);
	assert(y_len % A == 0);

	double numerator = 0.0;

	for (int j = 0; j < y_len; j++) {
		numerator += y[j];

		for (int k = 0; k <= A - 2; k++) {
			numerator += fabs(y[j] - y[(j + k + 1) % y_len]);
		}
	}

	const double tmp = ceil(A / 2.0);
	const double denominator = y_len * tmp * (1.0 + 2.0 * A - 2.0 * tmp) / A;

	return correct_to_01(numerator / denominator);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////