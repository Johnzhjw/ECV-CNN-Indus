#ifndef _RAND_H_
#define _RAND_H_

typedef enum {
	RAND_UNIF,
	RAND_CHEBYSHEV,
	RAND_PIECEWISE_LINEAR,
	RAND_SINUS,
	RAND_LOGISTIC,
	RAND_CIRCLE,
	RAND_GAUSS,
	RAND_TENT
} ENUM_RAND_NUM_TYPE;

extern ENUM_RAND_NUM_TYPE rand_type;

extern double(*pointer_gen_rand)();

double gaussrand(double a, double b);
double cauchyrand(double a, double b);
int rnd(int low, int high);
void shuffle(int* x, int size);
int flip_r(float prob);
double rndreal(double low, double high);

double HUPRandomExponential(double mu);
double HUPRandomLevy(double c, double alpha);
double LevyRand(double c, double alpha);

void set_init_rand_para(int seed, int seed_chaos);
void set_init_rand_type();
long get_rnd_uni_init();

int get_select_count_uti_rand();

#endif
