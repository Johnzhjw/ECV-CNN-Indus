#include "MOP_WDCN.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"

#define DIST_TWEEN 3 // 1:0.5:4 meters
#define Height_MAX (4.0/DIST_TWEEN)
#define Height_MIN (0.0/DIST_TWEEN)
#define DIST_PROP_MAX 10.0 //meters
#define DIST_INTERFERENCE_MAX 1.5
#define RADIUS_a 0.1 //meters
#define PPth 0.0
#define pi 3.1415926535897932384626433832795
#define PROP_RADIUS 1.0 //meters
#define PROP_ALPHA  (9.0 * pi / 180.0)
#define CHECK_ALPHA (0.01 * pi) //(9.0*pi/180.0)
#define penalty 1e6

#define MAX_ROW_WDCN 1024
#define MAX_COL_WDCN 256
#define COORDINATES_4_ONEMETER_ROW 31
#define COORDINATES_4_ONEMETER_COL 36

#define EPS_WDCN (1e-5)

#define CHECK_FURTHER

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
static double Mat_Heights[N_RADIO_PER_ROW][N_RADIO_PER_COL];
static int Mat_degrees[N_RADIO_PER_ROW][N_RADIO_PER_COL];
static int Mat_maxDegrees[N_RADIO_PER_ROW][N_RADIO_PER_COL];
static int Val_maxDegree;
static double Mat_prop_inten[N_RADIO_PER_ROW][N_RADIO_PER_COL];
static double Mat_inte_inten[N_RADIO_PER_ROW][N_RADIO_PER_COL];
static int Mat_connect[N_RADIO][N_RADIO];
static int max_avg_degree;
static int r_data, c_data;
static double data_WDCN[MAX_ROW_WDCN][MAX_COL_WDCN];
static double Mat_dists[N_RADIO][N_RADIO];

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void WDCN_OBJ_preliminary();
//static double Radio_propagation_interference_LOS(int t_x, int t_y, int r_x, int r_y, int p_x, int p_y);
//static double Radio_propagation_interference(int t_x, int t_y, int r_x, int r_y, int p_x, int p_y);
static int Connectivity();
static int WDCN_prop_LOS_constrain(int i, int j, int a, int b);
inline double lookup_dataTable_WDCN(double side_d, double forward_d);
static void WDCN_interference_two_signals(int i, int j, int a, int b);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////NN param
////% Input 1
//static double x1_step1_xoffset[2] = { 0, 0 };
//static double x1_step1_gain[2] = { 0.413793103448276, 0.185074626865672 };
//static double x1_step1_ymin = -1;
//
////% Layer 1
//static double b1[10] = { -1.9856608805762781, -1.5221527155556507, -0.14397151272806724, 5.0666388349918936, -0.036136429435158698, 3.3953224960206634, -1.654844189872668, -0.94865128840126811, -11.699095755912252, 2.813987852767331 };
//static double IW1_1[10][2] = {
//    {3.3254834806590172, -0.7264208149638427},
//    {-0.70670017694909648, 0.012230445045149007},
//    {-0.54666412185136626, -5.2984958022187714},
//    {-4.6819772894242604, 9.3515766762089516},
//    {7.277699783132725, -3.374897067877296},
//    {-4.7290183552551683, -5.9898148337883628},
//    {-2.4611750564163737, 2.8425661543107634},
//    {-3.0929299186318842, 0.72224031947596501},
//    {-7.6263254486198706, -8.3132584480696394},
//    {3.7128420720064583, 0.63652787492973673}
//};
//
////% Layer 2
//static double b2 = -3.7306950145575897;
//static double LW2_1[10] = { -0.13237492368856593, -4.195929285293996, -0.098331416187155005, -0.05250580317774195, -0.1537261691917452, 0.071496112170421525, -0.2590971279403016, 0.59512834690093319, 0.23052155536626959, -0.48756371595948977 };
//
////% Output 1
//static double y1_step1_ymin = -1;
//static double y1_step1_gain = 2.02380952380952;
//static double y1_step1_xoffset = 0.0117647058823529;
//
////Func
//static double NN_Map(double x0, double x1)
//{
//    double tmp_x[2];
//    tmp_x[0] = (x0 - x1_step1_xoffset[0]) * x1_step1_gain[0] + x1_step1_ymin;
//    tmp_x[1] = (x1 - x1_step1_xoffset[1]) * x1_step1_gain[1] + x1_step1_ymin;
//
//
//    double tmp_y[10];
//    for (int i = 0; i < 10; i++) {
//        tmp_y[i] = 0;
//        for (int j = 0; j < 2; j++) {
//            tmp_y[i] += IW1_1[i][j] * tmp_x[j];
//        }
//        tmp_y[i] += b1[i];
//        tmp_y[i] = 2.0 / (1.0 + exp(-2.0 * tmp_y[i])) - 1.0;
//    }
//    double tmp_z = 0.0;
//    for (int i = 0; i < 10; i++) {
//        tmp_z += LW2_1[i] * tmp_y[i];
//    }
//    tmp_z += b2;
//
//    tmp_z = (tmp_z - y1_step1_ymin) / y1_step1_gain + y1_step1_xoffset;
//
//    return tmp_z;
//}

//int main()
//{
//    double test[11][2] = { {4.833333333333333,  10.806451612903226},
//        {4.80555555555556,	10.8064516129032},
//        {4.77777777777778,	10.8064516129032},
//        {4.75000000000000,	10.8064516129032},
//        {4.72222222222222,	10.8064516129032},
//        {4.69444444444445,	10.8064516129032},
//        {4.66666666666667,	10.8064516129032},
//        {4.63888888888889,	10.8064516129032},
//        {4.61111111111111,	10.8064516129032},
//        {4.58333333333333,	10.8064516129032},
//        {0, 0}
//    };
//
//    for(int i = 0; i < 11; i++)
//        printf("%lf\n", NN_Map(test[i][0], test[i][1]));
//
//    return 0;
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline double lookup_dataTable_WDCN(double side_d, double forward_d)
{
	if (side_d < 0)
		side_d = -side_d;
	if (forward_d < 0)
		forward_d = -forward_d;
	//side_d = side_d < 0 ? -side_d : side_d;
	//forward_d = forward_d < 0 ? -forward_d : forward_d;

	int tar_r = (int)(COORDINATES_4_ONEMETER_ROW * forward_d);
	int tar_c = (int)(COORDINATES_4_ONEMETER_COL * side_d);

	if (tar_c >= c_data ||
		tar_c < 0 ||
		tar_r < 0)
		return 0.0;
	if (tar_r >= r_data) {
		double w1 = 0.3628;
		double w2 = 0.09763;
		//printf("%lf ", (exp(-0.13 * sqrt((side_d - PROP_RADIUS) * (side_d - PROP_RADIUS) + forward_d * forward_d))));
		return (exp(-(w1 * w1 * side_d * side_d) - w2 * forward_d));
	}
	return data_WDCN[tar_r][tar_c];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SetLimits_WDCN(double* minLimit, double* maxLimit, int nx)
{
	for (int r = 0; r < MAX_ROW_WDCN; r++) {
		for (int c = 0; c < MAX_COL_WDCN; c++) {
			data_WDCN[r][c] = 0;
		}
	}

	char fileName[1024];
	sprintf(fileName, "../Data_all/Data_WDCN/HeatMap");

	FILE* fpt = fopen(fileName, "r");
	if (fpt) {
		fscanf(fpt, "%d", &r_data);
		fscanf(fpt, "%d", &c_data);

		int n_r, n_c;
		for (int r = 0; r < r_data; r++) {
			for (int c = 0; c < c_data; c++) {
				double tmp;
				if (fscanf(fpt, "%lf", &tmp) == 1) {
					n_r = r_data - 1 - r;
					n_c = c_data - 1 - c;
					data_WDCN[n_r][n_c] = tmp;
				}
				else {
					printf("%s(%d): Reading file error, data not enough...\n", __FILE__, __LINE__);
				}
			}
		}
	}
	else {
		printf("%s(%d): File open error of '%s'...\n", __FILE__, __LINE__, fileName);
	}
	//
	int n_r_need = (int)(DIST_PROP_MAX * COORDINATES_4_ONEMETER_ROW) + 1;
	if (r_data < n_r_need) {
		for (int r = r_data; r < n_r_need; r++) {
			double d_f = (double)r / COORDINATES_4_ONEMETER_ROW;
			for (int c = 0; c < c_data; c++) {
				double d_v = (double)c / COORDINATES_4_ONEMETER_COL;
				double tmp = lookup_dataTable_WDCN(d_v, d_f);
				if (tmp > data_WDCN[r - 1][c])
					tmp = data_WDCN[r - 1][c] - EPS_WDCN;
				if (c && tmp > data_WDCN[r][c - 1])
					tmp = data_WDCN[r][c - 1] - EPS_WDCN;
				data_WDCN[r][c] = tmp;
			}
		}
		r_data = n_r_need;
	}

	int i, j;
	int a, b;

	Val_maxDegree = 0;
	int sum_degree = 0;
	double d;
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			int degree = 0;
			for (a = 0; a < N_RADIO_PER_ROW; a++) {
				for (b = 0; b < N_RADIO_PER_COL; b++) {
					if (i == a && j == b) continue;
					d = sqrt((i - a) * (i - a) + (j - b) * (j - b) + 0.0) * DIST_TWEEN;
					if (d <= DIST_PROP_MAX) {
						sum_degree++;
						degree++;
					}
				}
			}
			Mat_maxDegrees[i][j] = degree;
			if (degree > Val_maxDegree)
				Val_maxDegree = degree;
		}
	}
	max_avg_degree = sum_degree / N_RADIO + 1;

	for (i = 0; i < DIM_WDCN; i++) {
		minLimit[i] = Height_MIN;
		maxLimit[i] = Height_MAX;
	}

	return;
}

int CheckLimits_WDCN(double* x, int nx)
{
	int k;
	for (k = 0; k < DIM_WDCN; k++) {
		if (x[k] < Height_MIN || x[k] > Height_MAX) {
			printf("%s(%d):: Check limits FAIL - WDCN: %d, %.16e not in [%.16e, %.16e]\n",
				__FILE__, __LINE__, k, x[k], Height_MIN, Height_MAX);
			return false;
		}
	}

	return true;
}

void Fitness_WDCN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	int i, j;
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			Mat_Heights[i][j] = individual[i * N_RADIO_PER_COL + j];
		}
	}
	WDCN_OBJ_preliminary();
	//double sum_degrees = 0;
	//for (i = 0; i < N_RADIO_PER_ROW; i++) {
	//    for (j = 0; j < N_RADIO_PER_COL; j++) {
	//        sum_degrees += Mat_degrees[i][j];
	//    }
	//}
	//double mean_degree = sum_degrees / (N_RADIO);
	//double std_degree = 0.0;
	//for (i = 0; i < N_RADIO_PER_ROW; i++) {
	//    for (j = 0; j < N_RADIO_PER_COL; j++) {
	//        std_degree += (Mat_degrees[i][j] - mean_degree) * (Mat_degrees[i][j] - mean_degree);
	//    }
	//}
	//std_degree /= (N_RADIO);
	//std_degree = sqrt(std_degree) / max_avg_degree;
	//double mean_prob = 0.0;
	//for (i = 0; i < N_RADIO_PER_ROW; i++) {
	//    for (j = 0; j < N_RADIO_PER_COL; j++) {
	//        if (Mat_degrees[i][j]) {
	//            mean_prob += Mat_prop_inten[i][j] / Mat_degrees[i][j];
	//        }
	//    }
	//}
	//mean_prob /= (N_RADIO);
	//double std_prob = 0.0;
	//for (i = 0; i < N_RADIO_PER_ROW; i++) {
	//    for (j = 0; j < N_RADIO_PER_COL; j++) {
	//        if (Mat_degrees[i][j]) {
	//            std_prob += (Mat_prop_inten[i][j] / Mat_degrees[i][j] - mean_prob) *
	//                        (Mat_prop_inten[i][j] / Mat_degrees[i][j] - mean_prob);
	//        } else {
	//            std_prob += (0.0 - mean_prob) * (0.0 - mean_prob);
	//        }
	//    }
	//}
	//std_prob /= (N_RADIO);
	//std_prob = sqrt(std_prob);

	//double penaVal = (N_RADIO - Connectivity()) * penalty;

	//fitness[0] = 1.0 - mean_degree / max_avg_degree + penaVal;
	//fitness[1] = 1.0 - mean_prob + penaVal;
	//fitness[2] = std_prob + penaVal;

	double min_cov_deg = 1e100;
	double mean_cov_deg = 0.0;
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			double tmp = (double)Mat_degrees[i][j] / Mat_maxDegrees[i][j];
			if (tmp < min_cov_deg)
				min_cov_deg = tmp;
			mean_cov_deg += tmp;
		}
	}
	mean_cov_deg /= N_RADIO;

	double min_prop_inte = 1e100;
	double mean_prop_inte = 0.0;
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			double tmp = 0.0;
			if (Mat_degrees[i][j]) {
				tmp = Mat_prop_inten[i][j] / Mat_degrees[i][j];
			}
			if (tmp < min_prop_inte)
				min_prop_inte = tmp;
			mean_prop_inte += tmp;
		}
	}
	mean_prop_inte /= N_RADIO;

	double max_inte_inte = -1e100;
	double mean_inte_inte = 0.0;
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			double tmp = Mat_inte_inten[i][j] / Mat_maxDegrees[i][j] / 2;
			if (tmp > max_inte_inte)
				max_inte_inte = tmp;
			mean_inte_inte += tmp;
		}
	}
	mean_inte_inte /= N_RADIO;

	double penaVal = (N_RADIO - Connectivity()) * penalty;

	fitness[0] = 1.0 - mean_cov_deg + penaVal;
	fitness[1] = 1.0 - mean_prop_inte + penaVal;
	fitness[2] = mean_inte_inte + penaVal;

	return;
}

static void WDCN_OBJ_preliminary()
{
	int i, j;
	int a, b;
	double d, tmp;
	//
	for (i = 0; i < N_RADIO_PER_ROW; i++) {
		for (j = 0; j < N_RADIO_PER_COL; j++) {
			Mat_degrees[i][j] = 0;
			Mat_prop_inten[i][j] = 0.0;
			Mat_inte_inten[i][j] = 0.0;
		}
	}
	//
	int m, n;
	for (m = 0; m < N_RADIO; m++) {
		i = m / N_RADIO_PER_COL;
		j = m % N_RADIO_PER_COL;
		Mat_dists[m][m] = 0;
		for (n = m + 1; n < N_RADIO; n++) {
			a = n / N_RADIO_PER_COL;
			b = n % N_RADIO_PER_COL;
			d = sqrt((i - a) * (i - a) + (j - b) * (j - b) +
				(Mat_Heights[i][j] - Mat_Heights[a][b]) *
				(Mat_Heights[i][j] - Mat_Heights[a][b]));
			d *= DIST_TWEEN;
			Mat_dists[m][n] = Mat_dists[n][m] = d;
		}
	}
	//
	for (m = 0; m < N_RADIO; m++) {
		i = m / N_RADIO_PER_COL;
		j = m % N_RADIO_PER_COL;
		Mat_connect[m][m] = 0;
		for (n = m + 1; n < N_RADIO; n++) {
			a = n / N_RADIO_PER_COL;
			b = n % N_RADIO_PER_COL;
			d = Mat_dists[m][n];
			if (d <= DIST_PROP_MAX &&
				WDCN_prop_LOS_constrain(i, j, a, b)) {
				Mat_connect[m][n] = 1;
				Mat_connect[n][m] = 1;
				Mat_degrees[i][j]++;
				Mat_degrees[a][b]++;
				//tmp = lookup_dataTable_WDCN(0, d);
				int tmp_r = (int)(d * COORDINATES_4_ONEMETER_ROW),
					tmp_c = 0;
				tmp = 0;
				if (tmp_r >= 0 && tmp_r < r_data &&
					tmp_c >= 0 && tmp_c < c_data)
					tmp = data_WDCN[tmp_r][tmp_c];
				//else {
				//    printf("(%d, %d)\n", tmp_r, tmp_c);
				//}
				Mat_prop_inten[i][j] += tmp;
				Mat_prop_inten[a][b] += tmp;
				WDCN_interference_two_signals(i, j, a, b);
			}
			else {
				Mat_connect[m][n] = 0;
				Mat_connect[n][m] = 0;
			}
		}
	}

	return;
}

static int Connectivity()
{
	int connect = 0;
	int i, j;
	int count = 0;

	int flag[N_RADIO];
	int checked[N_RADIO];
	for (i = 0; i < N_RADIO; i++) flag[i] = 0;
	int status;
	int loop = 1;
	// int cur_loop;
	int pivot;
	for (i = 0; i < N_RADIO; i++) {
		if (flag[i] == 0) {
			count = 1;
			status = 1;
			flag[i] = loop;
			pivot = i;
			for (j = 0; j < N_RADIO; j++) checked[j] = 0;
			while (status) {
				checked[pivot] = 1;//visited
				for (j = 0; j < N_RADIO; j++) {
					if (flag[j] == 0 && Mat_connect[pivot][j]) {
						flag[j] = loop;
						count++;
					}
				}
				status = 0;
				for (j = 0; j < N_RADIO; j++)
					if (flag[j] == loop && checked[j] == 0) { //unvisited
						pivot = j;
						status = 1;
						break;
					}
			}
			loop++;
			if (count > connect) {
				connect = count;
				// cur_loop = loop - 1;
			}
		}
	}

	return (connect);
}

static int WDCN_prop_LOS_constrain(int i, int j, int a, int b)
{
	int m;
	int x, y, x1, y1;
	double k;         //两点成线的斜率//
	double x2;            //用于存储所求点坐标//
	double y2;
	int x3;
	int y3;               //将y2整数化//
	int idxA, idxB, idxS;
	double ab, as, bs;
	double cos_A, sin_A;
	double d;
	x = abs(a - i);       //俩点间的坐标差//
	y = abs(b - j);
	idxA = i * N_RADIO_PER_COL + j;
	idxB = a * N_RADIO_PER_COL + b;
	ab = Mat_dists[idxA][idxB];
	//////////////////////////////////////////////////////////////////////////
	//CASE 1
	if (x >= y) {
		if (i != a) {
			x1 = (a - i) / x;         //存正负数//
			k = ((double)(b - j)) / ((double)(a - i));
			for (m = 1; m < x; m++) {
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				x3 = (int)x2;
				y3 = (int)y2;
				idxS = x3 * N_RADIO_PER_COL + y3;
				as = Mat_dists[idxS][idxA];
				bs = Mat_dists[idxS][idxB];
				cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
				sin_A = sqrt(1 - cos_A * cos_A);
				d = as * sin_A;
				if (d < RADIUS_a) {
					return 0;
				}
				if (y3 < y2) {
					y3 = y3 + 1;
					idxS = x3 * N_RADIO_PER_COL + y3;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					sin_A = sqrt(1 - cos_A * cos_A);
					d = as * sin_A;
					if (d < RADIUS_a) {
						return 0;
					}
				}
			}
		}
	}
	else {
		if (j != b) {
			y1 = (b - j) / y;         //存正负数
			k = ((double)(a - i)) / ((double)(b - j));
			for (m = 1; m < y; m++) {
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				x3 = (int)x2;
				y3 = (int)y2;
				idxS = x3 * N_RADIO_PER_COL + y3;
				as = Mat_dists[idxS][idxA];
				bs = Mat_dists[idxS][idxB];
				cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
				sin_A = sqrt(1 - cos_A * cos_A);
				d = as * sin_A;
				if (d < RADIUS_a) {
					return 0;
				}
				if (x3 < x2) {
					x3 = x3 + 1;
					idxS = x3 * N_RADIO_PER_COL + y3;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					sin_A = sqrt(1 - cos_A * cos_A);
					d = as * sin_A;
					if (d < RADIUS_a) {
						return 0;
					}
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//CASE 2
	int n_step1, n_step2;
	int cur_x, cur_y;
	if (x >= y) {
		if (i != a) {
			x1 = (a - i) / x;         //存正负数
			k = ((double)(b - j)) / ((double)(a - i));
			double alpha_low = atan(fabs(k)) - CHECK_ALPHA;
			double alpha_high = atan(fabs(k)) + CHECK_ALPHA;
			for (m = 1;; m++) {
#ifndef CHECK_FURTHER
				if (m >= x) break;
#endif
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				x3 = (int)x2;
				y3 = (int)y2;
				if (x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					//y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(i - x3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)(abs(x3 - i) * (fabs(k) - tan(alpha_low)) - (y2 - y3));
				n_step2 = (int)(abs(x3 - i) * (tan(alpha_high) - fabs(k)) + (y2 - y3));
				cur_x = x3;
				for (cur_y = y3 - n_step1; cur_y <= y3 + n_step2; cur_y++) {
					if (cur_y < 0 || cur_y >= N_RADIO_PER_COL ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					//if (Radio_propagation_interference_LOS(i, j, a, b, cur_x, cur_y) > PPth)
					//    return 0;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (as > DIST_PROP_MAX)
						continue;
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					double alpha_cur = acos(cos_A);
					if (alpha_cur <= CHECK_ALPHA) {
						int tmp_r = (int)(as * cos_A * COORDINATES_4_ONEMETER_ROW),
							tmp_c = (int)(as * sin(alpha_cur) * COORDINATES_4_ONEMETER_COL);
						// tmp_c less than c_data
						double tmp_p = 0;
						if (tmp_r >= 0 && tmp_r < r_data &&
							tmp_c >= 0 && tmp_c < c_data)
							tmp_p = data_WDCN[tmp_r][tmp_c];
						//else {
						//    printf("(%d, %d)\n", tmp_r, tmp_c);
						//}
						if (tmp_p > PPth) {
							return 0;
						}
					}
				}
			}
			for (m = x - 1;; m--) {
#ifndef CHECK_FURTHER
				if (m <= 0) break;
#endif
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				x3 = (int)x2;
				y3 = (int)y2;
				if (x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					//y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(a - x3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)(abs(a - x3) * (fabs(k) - tan(alpha_low)) - (y2 - y3));
				n_step2 = (int)(abs(a - x3) * (tan(alpha_high) - fabs(k)) + (y2 - y3));
				cur_x = x3;
				for (cur_y = y3 - n_step1; cur_y <= y3 + n_step2; cur_y++) {
					if (cur_y < 0 || cur_y >= N_RADIO_PER_COL ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					//if (Radio_propagation_interference_LOS(a, b, i, j, x3, cur_y) > PPth)
					//    return 0;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (bs > DIST_PROP_MAX)
						continue;
					cos_A = (bs * bs + ab * ab - as * as) / (2 * ab * bs);
					double alpha_cur = acos(cos_A);
					if (alpha_cur <= CHECK_ALPHA) {
						int tmp_r = (int)(bs * cos_A * COORDINATES_4_ONEMETER_ROW),
							tmp_c = (int)(bs * sin(alpha_cur) * COORDINATES_4_ONEMETER_COL);
						// tmp_c less than c_data
						double tmp_p = 0;
						if (tmp_r >= 0 && tmp_r < r_data &&
							tmp_c >= 0 && tmp_c < c_data)
							tmp_p = data_WDCN[tmp_r][tmp_c];
						//else {
						//    printf("(%d, %d)\n", tmp_r, tmp_c);
						//}
						if (tmp_p > PPth) {
							return 0;
						}
					}
				}
			}
		}
	}
	else {
		if (j != b) {
			y1 = (b - j) / y;         //存正负数
			k = ((double)(a - i)) / ((double)(b - j));
			double alpha_low = atan(fabs(k)) - CHECK_ALPHA;
			double alpha_high = atan(fabs(k)) + CHECK_ALPHA;
			for (m = 1;; m++) {
#ifndef CHECK_FURTHER
				if (m >= y) break;
#endif
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				x3 = (int)x2;
				y3 = (int)y2;
				if (//x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(j - y3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)(abs(y3 - j) * (fabs(k) - tan(alpha_low)) - (x2 - x3));
				n_step2 = (int)(abs(y3 - j) * (tan(alpha_high) - fabs(k)) + (x2 - x3));
				cur_y = y3;
				for (cur_x = x3 - n_step1; cur_x <= x3 + n_step2; cur_x++) {
					if (cur_x < 0 || cur_x >= N_RADIO_PER_ROW ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					//if (Radio_propagation_interference_LOS(i, j, a, b, cur_x, cur_y) > PPth)
					//    return 0;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (as > DIST_PROP_MAX)
						continue;
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					double alpha_cur = acos(cos_A);
					if (alpha_cur <= CHECK_ALPHA) {
						int tmp_r = (int)(as * cos_A * COORDINATES_4_ONEMETER_ROW),
							tmp_c = (int)(as * sin(alpha_cur) * COORDINATES_4_ONEMETER_COL);
						// tmp_c less than c_data
						double tmp_p = 0;
						if (tmp_r >= 0 && tmp_r < r_data &&
							tmp_c >= 0 && tmp_c < c_data)
							tmp_p = data_WDCN[tmp_r][tmp_c];
						//else {
						//    printf("(%d, %d)\n", tmp_r, tmp_c);
						//}
						if (tmp_p > PPth) {
							return 0;
						}
					}
				}
			}
			for (m = y - 1;; m--) {
#ifndef CHECK_FURTHER
				if (m <= 0) break;
#endif
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				x3 = (int)x2;
				y3 = (int)y2;
				if (//x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(b - y3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)(abs(b - y3) * (fabs(k) - tan(alpha_low)) - (x2 - x3));
				n_step2 = (int)(abs(b - y3) * (tan(alpha_high) - fabs(k)) + (x2 - x3));
				cur_y = y3;
				for (cur_x = x3 - n_step1; cur_x <= x3 + n_step2; cur_x++) {
					if (cur_x < 0 || cur_x >= N_RADIO_PER_ROW ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					//if (Radio_propagation_interference_LOS(a, b, i, j, cur_x, cur_y) > PPth)
					//    return 0;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (bs > DIST_PROP_MAX)
						continue;
					cos_A = (bs * bs + ab * ab - as * as) / (2 * ab * bs);
					double alpha_cur = acos(cos_A);
					if (alpha_cur <= CHECK_ALPHA) {
						int tmp_r = (int)(bs * cos_A * COORDINATES_4_ONEMETER_ROW),
							tmp_c = (int)(bs * sin(alpha_cur) * COORDINATES_4_ONEMETER_COL);
						// tmp_c less than c_data
						double tmp_p = 0;
						if (tmp_r >= 0 && tmp_r < r_data &&
							tmp_c >= 0 && tmp_c < c_data)
							tmp_p = data_WDCN[tmp_r][tmp_c];
						//else {
						//    printf("(%d, %d)\n", tmp_r, tmp_c);
						//}
						if (tmp_p > PPth) {
							return 0;
						}
					}
				}
			}
		}
	}

	return 1;
}

static void WDCN_interference_two_signals(int i, int j, int a, int b)
{
	int m;
	int x, y, x1, y1;
	double k;         //两点成线的斜率//
	double x2;            //用于存储所求点坐标//
	double y2;
	int x3;
	int y3;               //将y2整数化//
	int idxA, idxB, idxS;
	double ab, as, bs;
	double cos_A, sin_A;
	//double d;
	x = abs(a - i);       //俩点间的坐标差//
	y = abs(b - j);
	idxA = i * N_RADIO_PER_COL + j;
	idxB = a * N_RADIO_PER_COL + b;
	ab = Mat_dists[idxA][idxB];
	//////////////////////////////////////////////////////////////////////////
	int n_step1, n_step2;
	int cur_x, cur_y;
	if (x >= y) {
		if (i != a) {
			x1 = (a - i) / x;         //存正负数
			k = ((double)(b - j)) / ((double)(a - i));
			for (m = 0;; m++) {
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				x3 = (int)x2;
				y3 = (int)y2;
				if (x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					//y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(i - x3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN - (y2 - y3)));
				n_step2 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN + (y2 - y3)));
				cur_x = x3;
				for (cur_y = y3 - n_step1; cur_y <= y3 + n_step2; cur_y++) {
					if (cur_y < 0 || cur_y >= N_RADIO_PER_COL ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (as > DIST_PROP_MAX)
						continue;
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					sin_A = sqrt(1 - cos_A * cos_A);
					int tmp_r = (int)(as * cos_A * COORDINATES_4_ONEMETER_ROW),
						tmp_c = (int)(as * sin_A * COORDINATES_4_ONEMETER_COL);
					// tmp_c less than c_data
					double tmp_p = 0;
					if (tmp_r >= 0 && tmp_r < r_data &&
						tmp_c >= 0 && tmp_c < c_data)
						tmp_p = data_WDCN[tmp_r][tmp_c];
					//else {
					//    printf("(%d, %d)\n", tmp_r, tmp_c);
					//}
					Mat_inte_inten[cur_x][cur_y] += tmp_p;
					//Mat_inte_inten[x3][cur_y] += Radio_propagation_interference(i, j, a, b, x3, cur_y);
				}
			}
			for (m = x;; m--) {
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				x3 = (int)x2;
				y3 = (int)y2;
				if (x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					//y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(a - x3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN - (y2 - y3)));
				n_step2 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN + (y2 - y3)));
				cur_x = x3;
				for (cur_y = y3 - n_step1; cur_y <= y3 + n_step2; cur_y++) {
					if (cur_y < 0 || cur_y >= N_RADIO_PER_COL ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (bs > DIST_PROP_MAX)
						continue;
					cos_A = (bs * bs + ab * ab - as * as) / (2 * ab * bs);
					sin_A = sqrt(1 - cos_A * cos_A);
					int tmp_r = (int)(bs * cos_A * COORDINATES_4_ONEMETER_ROW),
						tmp_c = (int)(bs * sin_A * COORDINATES_4_ONEMETER_COL);
					// tmp_c less than c_data
					double tmp_p = 0;
					if (tmp_r >= 0 && tmp_r < r_data &&
						tmp_c >= 0 && tmp_c < c_data)
						tmp_p = data_WDCN[tmp_r][tmp_c];
					//else {
					//    printf("(%d, %d)\n", tmp_r, tmp_c);
					//}
					Mat_inte_inten[cur_x][cur_y] += tmp_p;
					//Mat_inte_inten[x3][cur_y] += Radio_propagation_interference(a, b, i, j, x3, cur_y);
				}
			}
		}
	}
	else {
		if (j != b) {
			y1 = (b - j) / y;         //存正负数
			k = ((double)(a - i)) / ((double)(b - j));
			for (m = 0;; m++) {
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				x3 = (int)x2;
				y3 = (int)y2;
				if (//x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(j - y3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN - (x2 - x3)));
				n_step2 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN + (x2 - x3)));
				cur_y = y3;
				for (cur_x = x3 - n_step1; cur_x <= x3 + n_step2; cur_x++) {
					if (cur_x < 0 || cur_x >= N_RADIO_PER_ROW ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (as > DIST_PROP_MAX)
						continue;
					cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
					sin_A = sqrt(1 - cos_A * cos_A);
					int tmp_r = (int)(as * cos_A * COORDINATES_4_ONEMETER_ROW),
						tmp_c = (int)(as * sin_A * COORDINATES_4_ONEMETER_COL);
					// tmp_c less than c_data
					double tmp_p = 0;
					if (tmp_r >= 0 && tmp_r < r_data &&
						tmp_c >= 0 && tmp_c < c_data)
						tmp_p = data_WDCN[tmp_r][tmp_c];
					//else {
					//    printf("(%d, %d)\n", tmp_r, tmp_c);
					//}
					Mat_inte_inten[cur_x][cur_y] += tmp_p;
					//Mat_inte_inten[cur_x][y3] += Radio_propagation_interference(i, j, a, b, cur_x, y3);
				}
			}
			for (m = y;; m--) {
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				x3 = (int)x2;
				y3 = (int)y2;
				if (//x3 < 0 || x3 >= N_RADIO_PER_ROW ||
					y3 < 0 || y3 >= N_RADIO_PER_COL ||
					abs(b - y3) * DIST_TWEEN > DIST_PROP_MAX)
					break;
				n_step1 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN - (x2 - x3)));
				n_step2 = (int)((DIST_INTERFERENCE_MAX * sqrt(k * k + 1) / DIST_TWEEN + (x2 - x3)));
				cur_y = y3;
				for (cur_x = x3 - n_step1; cur_x <= x3 + n_step2; cur_x++) {
					if (cur_x < 0 || cur_x >= N_RADIO_PER_ROW ||
						(cur_x == i && cur_y == j) || (cur_x == a && cur_y == b))
						continue;
					idxS = cur_x * N_RADIO_PER_COL + cur_y;
					as = Mat_dists[idxS][idxA];
					bs = Mat_dists[idxS][idxB];
					if (bs > DIST_PROP_MAX)
						continue;
					cos_A = (bs * bs + ab * ab - as * as) / (2 * ab * bs);
					sin_A = sqrt(1 - cos_A * cos_A);
					int tmp_r = (int)(bs * cos_A * COORDINATES_4_ONEMETER_ROW),
						tmp_c = (int)(bs * sin_A * COORDINATES_4_ONEMETER_COL);
					// tmp_c less than c_data
					double tmp_p = 0;
					if (tmp_r >= 0 && tmp_r < r_data &&
						tmp_c >= 0 && tmp_c < c_data)
						tmp_p = data_WDCN[tmp_r][tmp_c];
					//else {
					//    printf("(%d, %d)\n", tmp_r, tmp_c);
					//}
					Mat_inte_inten[cur_x][cur_y] += tmp_p;
					//Mat_inte_inten[cur_x][y3] += Radio_propagation_interference(a, b, i, j, cur_x, y3);
				}
			}
		}
	}

	return;
}

//double Radio_propagation_interference_LOS(int t_x, int t_y, int r_x, int r_y, int p_x, int p_y)
//{
//    double ab, as, bs;
//    double cos_A, sin_A;
//    double d;
//    as = sqrt((t_x - p_x) * (t_x - p_x) + (t_y - p_y) * (t_y - p_y) +
//              (Mat_Heights[t_x][t_y] - Mat_Heights[p_x][p_y]) *
//              (Mat_Heights[t_x][t_y] - Mat_Heights[p_x][p_y])) * DIST_TWEEN;
//    if(as > DIST_PROP_MAX)
//        return 0.0;
//    ab = sqrt((t_x - r_x) * (t_x - r_x) + (t_y - r_y) * (t_y - r_y) +
//              (Mat_Heights[t_x][t_y] - Mat_Heights[r_x][r_y]) *
//              (Mat_Heights[t_x][t_y] - Mat_Heights[r_x][r_y])) * DIST_TWEEN;
//    bs = sqrt((r_x - p_x) * (r_x - p_x) + (r_y - p_y) * (r_y - p_y) +
//              (Mat_Heights[r_x][r_y] - Mat_Heights[p_x][p_y]) *
//              (Mat_Heights[r_x][r_y] - Mat_Heights[p_x][p_y])) * DIST_TWEEN;
//    cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
//    sin_A = sqrt(1 - cos_A * cos_A);
//    d = as * sin_A;
//    double h = as * cos_A;
//    double alpha_cur = acos(cos_A);
//    if(alpha_cur <= CHECK_ALPHA) {
//        return (lookup_dataTable_WDCN(d, h));
//    } else {
//        return 0;
//    }
//    //if (d <= PROP_RADIUS) {
//    //    if (as <= 1.0)
//    //        return 1.0;
//    //    if (as > DIST_PROP_MAX)
//    //        return 0.0;
//    //    //return (exp(-0.13 * as));
//    //    return (lookup_dataTable_WDCN(d, as * cos_A));
//    //} else {
//    //    double h = sqrt(as * as - d * d);
//    //    double alpha_cur = atan((d - PROP_RADIUS) / h);
//    //    if (alpha_cur <= PROP_ALPHA) {
//    //        d = sqrt(h * h + (d - PROP_RADIUS) * (d - PROP_RADIUS));
//    //        if (d <= 1.0)
//    //            return 1.0;
//    //        if (d > DIST_PROP_MAX)
//    //            return 0.0;
//    //        //return (exp(-0.13 * d));
//    //        return (lookup_dataTable_WDCN(as * sin_A, as * cos_A));
//    //    } else {
//    //        return 0;
//    //    }
//    //}
//}
//
//double Radio_propagation_interference(int t_x, int t_y, int r_x, int r_y, int p_x, int p_y)
//{
//    double ab, as, bs;
//    double cos_A, sin_A;
//    double d;
//    as = sqrt((t_x - p_x) * (t_x - p_x) + (t_y - p_y) * (t_y - p_y) +
//              (Mat_Heights[t_x][t_y] - Mat_Heights[p_x][p_y]) *
//              (Mat_Heights[t_x][t_y] - Mat_Heights[p_x][p_y])) * DIST_TWEEN;
//    if(as > DIST_PROP_MAX)
//        return 0.0;
//    ab = sqrt((t_x - r_x) * (t_x - r_x) + (t_y - r_y) * (t_y - r_y) +
//              (Mat_Heights[t_x][t_y] - Mat_Heights[r_x][r_y]) *
//              (Mat_Heights[t_x][t_y] - Mat_Heights[r_x][r_y])) * DIST_TWEEN;
//    bs = sqrt((r_x - p_x) * (r_x - p_x) + (r_y - p_y) * (r_y - p_y) +
//              (Mat_Heights[r_x][r_y] - Mat_Heights[p_x][p_y]) *
//              (Mat_Heights[r_x][r_y] - Mat_Heights[p_x][p_y])) * DIST_TWEEN;
//    cos_A = (as * as + ab * ab - bs * bs) / (2 * ab * as);
//    sin_A = sqrt(1 - cos_A * cos_A);
//    d = as * sin_A;
//    //return (exp(-0.13 * as));
//    return (lookup_dataTable_WDCN(d, as * cos_A));
//    //if (d <= PROP_RADIUS) {
//    //    if (as <= 1.0)
//    //        return 1.0;
//    //    if (as > DIST_PROP_MAX)
//    //        return 0.0;
//    //    //return (exp(-0.13 * as));
//    //    return (lookup_dataTable_WDCN(d, as * cos_A));
//    //} else {
//    //    double h = sqrt(as * as - d * d);
//    //    double alpha_cur = atan((d - PROP_RADIUS) / h);
//    //    if (alpha_cur <= PROP_ALPHA) {
//    //        d = sqrt(h * h + (d - PROP_RADIUS) * (d - PROP_RADIUS));
//    //        if (d <= 1.0)
//    //            return 1.0;
//    //        if (d > DIST_PROP_MAX)
//    //            return 0.0;
//    //        //return (exp(-0.13 * d));
//    //        return (lookup_dataTable_WDCN(as * sin_A, as * cos_A));
//    //    } else {
//    //        return 0;
//    //    }
//    //}
//}

// static double ArcPan(int i, int j, int a, int b)
// {
//     double temp;
//     if (j == b) {
//         if (a > i)
//             return 0.0;
//         else if (a < i)
//             return pi;
//     }
//     temp = (a - i) / sqrt((double)(a - i) * (a - i) + (b - j) * (b - j));
//     if (j > b)
//         return 2 * pi - acos(temp);
//     else
//         return acos(temp);
// }
// static double ArcTilt(int i, int j, int a, int b)
// {
//     double temp;
//     double hs, hp;
//     hs = Mat_Heights[i][j];
//     hp = Mat_Heights[a][b];
//     if (hs == hp) //
//         return 0.0;
//     temp = sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)) /
//            sqrt((double)(a - i) * (a - i) + (b - j) * (b - j) + (hp - hs) * (hp - hs)); //
//     if (hs > hp)
//         return -acos(temp);
//     else
//         return acos(temp);
// }