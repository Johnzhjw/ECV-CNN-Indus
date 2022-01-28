#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MOP_IWSN_1F.h"

#define lon (32)
#define wid (32)
#define hig (2)
#define resIWSN_1F (10.0)
#define X (0)
#define Y (1)
#define Z (2)
#define PAN (2)
#define TILT (3)
#define pi (3.1415926535897932384626433832795)
#define r_direc_min (30.0/resIWSN_1F)
#define r_direc_max (48.0/resIWSN_1F)
#define r_direc_ratio (1.0)
#define N (lon*wid*hig)//总的要覆盖的离散点数
#define ga (-0.5)
#define oq_beta (0.9)
#define angle_min (pi/6/2)
#define angle_max (2*pi/9/2)
#define penaltyVal (1e6)
#define minH (0)
#define angle_ratio_upp (2.0)
#define mu_D (3.0)
#define nu_D (1.0)
#define mu_A (3.0)
#define nu_A (1.0)
#define theta_H (1.0)
#define theta_V (1.3)

static double map[wid][lon] = {
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.40, 0.40, 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
	{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 }
};

#define energy_ini (10.0)
#define E_elec (50.0e-9)
#define e_fs (10.0e-12)
#define e_mp (0.0013e-12)
#define d_th (87.0)
#define E_DA (5.0e-9)
#define d_DA (0.1)
#define l0 (200)
#define n_rn_min (2)
#define d_th_sn (6.0/9.0*d_th)
#define d_th_rn (9.0/9.0*d_th)

// position of the sink node
#define SINK_X (wid/2.0)
#define SINK_Y (lon-1.0)
#define SINK_Z (hig/2.0)

static double qoc[wid][lon][hig];
static int occupVol = 0; //设备所占点

static double radiusRs_DIREC[N_DIREC_1F];
static double radiusRf_DIREC[N_DIREC_1F];
static double pan_angle[N_DIREC_1F];
static double tilt_angle[N_DIREC_1F];
static double angle_range[N_DIREC_1F];

static int pos3D_DIREC[N_DIREC_1F][3];
static bool posFlag_DIREC[N_DIREC_1F];
static int pos3D_RELAY[N_RELAY_1F][3];
static bool posFlag_RELAY[N_RELAY_1F];

static int posCountBad;

static int hopID_SENSOR[N_DIREC_1F];  // 0,1,2,...,N_RELAY-1, N_RELAY - SINK
static int hopID_RELAY[N_RELAY_1F];   // 0,1,2,...,N_RELAY-1, N_RELAY - SINK

static double com_dist_SENSOR[N_DIREC_1F];
static double com_dist_RELAY[N_RELAY_1F];

static double n_data_local_RELAY[N_RELAY_1F];
static double n_data_hop_RELAY[N_RELAY_1F];
static double n_data_fwd_RELAY[N_RELAY_1F];

static double avg_dist_SENSOR_IWSN_1F;
static double std_dist_SENSOR;
static double energy_consumed_RELAY[N_RELAY_1F];

static int n_sn_rn_com[N_DIREC_1F];
static int n_rn_rn_com[N_RELAY_1F];

static int   seed_rand = 237;
static long  rnd_uni_init_val = -(long)seed_rand;
#define IM1_IWSN_1F 2147483563
#define IM2_IWSN_1F 2147483399
#define AM_IWSN_1F (1.0/IM1_IWSN_1F)
#define IMM1_IWSN_1F (IM1_IWSN_1F-1)
#define IA1_IWSN_1F 40014
#define IA2_IWSN_1F 40692
#define IQ1_IWSN_1F 53668
#define IQ2_IWSN_1F 52774
#define IR1_IWSN_1F 12211
#define IR2_IWSN_1F 3791
#define NTAB_IWSN_1F 32
#define NDIV_IWSN_1F (1+IMM1_IWSN_1F/NTAB_IWSN_1F)
#define EPS_IWSN_1F 1.2e-7
#define RNMX_IWSN_1F (1.0-EPS_IWSN_1F)

/***************************************************/
//函数
//the random generator in [0,1) ~ [vmin, vmax)
static double rnd_uni_gen(long* idum, double vmin, double vmax);
static bool TransPos(int a, int b, int* _3D);
static double Coverage();//覆盖率目标函数
static double Lifetime();//生命周期目标函数
static void Qoc();
static double Oq_DIREC(int i, int j, int k, int l);
static int LOS(int i, int j, int h, int a, int b, int c);
static double range(int i, int j, int k, int a, int b, int c); //距离
static double ArcPan(int i, int j, int k, int a, int b, int c); //水平角度
static double ArcTilt(int i, int j, int k, int a, int b, int c); //垂直角度
/***************************************************/

void Fitness_IWSN_1F(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	//if (!checkLimits_IWSN_1F(individual, nx)) {
	//    printf("checkLimits_IWSN_1F FAIL, exiting...\n");
	//    exit(-1);
	//}
	posCountBad = 0;

	for (int i = 0; i < N_DIREC_1F; i++) {
		posFlag_DIREC[i] = TransPos((int)(individual[i * D_DIREC_1F + X]), (int)(individual[i * D_DIREC_1F + Y]), &pos3D_DIREC[i][0]);
		if (!posFlag_DIREC[i])
			posCountBad++;
		pan_angle[i] = individual[i * D_DIREC_1F + PAN];
		tilt_angle[i] = individual[i * D_DIREC_1F + TILT];
	}
	int tmp_offset = N_DIREC_1F * D_DIREC_1F;
	for (int i = 0; i < N_RELAY_1F; i++) {
		posFlag_RELAY[i] = TransPos((int)(individual[tmp_offset + i * D_RELAY_1F + X]),
			(int)(individual[tmp_offset + i * D_RELAY_1F + Y]), &pos3D_RELAY[i][0]);
		if (!posFlag_RELAY[i])
			posCountBad++;
	}

	fitness[0] = 1.0 - Coverage() + (posCountBad)*penaltyVal;
	fitness[1] = Lifetime() + (posCountBad)*penaltyVal;

	return;
}

void SetLimits_IWSN_1F(double* minLimit, double* maxLimit, int nx)
{
	seed_rand = 237;
	rnd_uni_init_val = -(long)seed_rand;

	int i, j, k;
	for (k = 0; k < N_DIREC_1F; k++) {
		minLimit[k * D_DIREC_1F + X] = 0;
		minLimit[k * D_DIREC_1F + Y] = 0;
		minLimit[k * D_DIREC_1F + PAN] = 0;
		minLimit[k * D_DIREC_1F + TILT] = -pi / 2.0;
		maxLimit[k * D_DIREC_1F + X] = wid + 2 * (hig - minH) - 1e-6;
		maxLimit[k * D_DIREC_1F + Y] = lon + 2 * (hig - minH) - 1e-6;
		maxLimit[k * D_DIREC_1F + PAN] = 2 * pi - 1e-6;
		maxLimit[k * D_DIREC_1F + TILT] = pi / 2.0;

		radiusRs_DIREC[k] = rnd_uni_gen(&rnd_uni_init_val, r_direc_min, r_direc_max);
		radiusRf_DIREC[k] = radiusRs_DIREC[k] * r_direc_ratio;
		angle_range[k] = rnd_uni_gen(&rnd_uni_init_val, angle_min, angle_max);
		printf("%lf ", radiusRs_DIREC[k]);
		printf("%lf\n", angle_range[k]);
	}
	occupVol = 0;
	for (i = 0; i < wid; i++) {
		for (j = 0; j < lon; j++) {
			for (k = 0; k < hig; k++) {
				if (k < map[i][j])
					occupVol += 1;
			}
		}
	}
	int tmp_offset = N_DIREC_1F * D_DIREC_1F;
	for (k = 0; k < N_RELAY_1F; k++) {
		minLimit[tmp_offset + k * D_RELAY_1F + X] = 0;
		minLimit[tmp_offset + k * D_RELAY_1F + Y] = 0;
		maxLimit[tmp_offset + k * D_RELAY_1F + X] = wid + 2 * (hig - minH) - 1e-6;
		maxLimit[tmp_offset + k * D_RELAY_1F + Y] = lon + 2 * (hig - minH) - 1e-6;
	}

	return;
}

int CheckLimits_IWSN_1F(double* x, int nx)
{
	int k;

	for (k = 0; k < N_DIREC_1F; k++) {
		if (x[k * D_DIREC_1F + X] < 0 || x[k * D_DIREC_1F + X] > wid + 2 * (hig - minH) - 1e-6) {
			printf("Check limits FAIL - IWSN Coverage: %d: %lf not in [%lf,%lf]\n",
				k * D_DIREC_1F + X, x[k * D_DIREC_1F + X], 0.0, (double)(wid + 2 * (hig - minH) - 1e-6));
			return false;
		}
		if (x[k * D_DIREC_1F + Y] < 0 || x[k * D_DIREC_1F + Y] > lon + 2 * (hig - minH) - 1e-6) {
			printf("Check limits FAIL - IWSN Coverage: %d: %lf not in [%lf,%lf]\n",
				k * D_DIREC_1F + Y, x[k * D_DIREC_1F + Y], 0.0, (double)(lon + 2 * (hig - minH) - 1e-6));
			return false;
		}
		if (x[k * D_DIREC_1F + PAN] < 0 || x[k * D_DIREC_1F + PAN] > 2 * pi - 1e-6) {
			printf("Check limits FAIL - IWSN Coverage: %d: %lf not in [%lf,%lf]\n",
				k * D_DIREC_1F + PAN, x[k * D_DIREC_1F + PAN], 0.0, 2 * pi - 1e-6);
			return false;
		}
		if (x[k * D_DIREC_1F + TILT] < -pi / 2 || x[k * D_DIREC_1F + TILT] > pi / 2) {
			printf("Check limits FAIL - IWSN Coverage: %d: %lf not in [%lf,%lf]\n",
				k * D_DIREC_1F + TILT, x[k * D_DIREC_1F + TILT], -pi / 2, pi / 2);
			return false;
		}
	}
	int tmp_offset = N_DIREC_1F * D_DIREC_1F;
	for (k = 0; k < N_RELAY_1F; k++) {
		if (x[tmp_offset + k * D_RELAY_1F + X] < 0 || x[tmp_offset + k * D_RELAY_1F + X] >
			wid + 2 * (hig - minH) - 1e-6) {
			printf("Check limits FAIL - IWSN Lifetime: %d: %lf not in [%lf,%lf]\n",
				tmp_offset + k * D_RELAY_1F + X, x[tmp_offset + k * D_RELAY_1F + X],
				0.0, (double)(wid + 2 * (hig - minH) - 1e-6));
			return false;
		}
		if (x[tmp_offset + k * D_RELAY_1F + Y] < 0 || x[tmp_offset + k * D_RELAY_1F + Y] >
			lon + 2 * (hig - minH) - 1e-6) {
			printf("Check limits FAIL - IWSN Lifetime: %d: %lf not in [%lf,%lf]\n",
				tmp_offset + k * D_RELAY_1F + Y, x[tmp_offset + k * D_RELAY_1F + Y],
				0.0, (double)(lon + 2 * (hig - minH) - 1e-6));
			return false;
		}
	}

	return true;
}

static bool TransPos(int a, int b, int* _3D)
{
	int bound_left = hig - minH - 1;
	int bound_right = lon + hig - minH - 1;
	int bound_up = hig - minH - 1;
	int bound_down = wid + hig - minH - 1;

	if ((a <= bound_up && b <= bound_left) ||
		(a <= bound_up && b > bound_right) ||
		(a > bound_down && b <= bound_left) ||
		(a > bound_down && b > bound_right)) {
		//printf("(%d,%d)!!!",a,b);
		return false;
	}
	if (!_3D)
		return true;

	if (a > bound_up && a <= bound_down && b <= bound_left) {  //left
		_3D[X] = a - (hig - minH);
		_3D[Y] = 0;
		_3D[Z] = b + minH;
	}
	else if (a > bound_up && a <= bound_down && b > bound_right) {  //right
		_3D[X] = a - (hig - minH);
		_3D[Y] = lon - 1;
		_3D[Z] = lon + 2 * (hig - minH) - 1 - b + minH;
	}
	else if (b > bound_left && b <= bound_right && a <= bound_up) {  //upp
		_3D[X] = 0;
		_3D[Y] = b - (hig - minH);
		_3D[Z] = a + minH;
	}
	else if (b > bound_left && b <= bound_right && a > bound_down) {  //down
		_3D[X] = wid - 1;
		_3D[Y] = b - (hig - minH);
		_3D[Z] = wid + 2 * (hig - minH) - 1 - a + minH;
	}
	else { //center
		_3D[X] = a - (hig - minH);
		_3D[Y] = b - (hig - minH);
		_3D[Z] = hig - 1;
	}

	//printf("(%d,%d)->(%d,%d,%d)",a,b,_3D[X],_3D[Y],_3D[Z]);
	return true;
}

//the random generator in [0,1)
double rnd_uni_gen(long* idum, double vmin, double vmax)
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB_IWSN_1F];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB_IWSN_1F + 7; j >= 0; j--) {
			k = (*idum) / IQ1_IWSN_1F;
			*idum = IA1_IWSN_1F * (*idum - k * IQ1_IWSN_1F) - k * IR1_IWSN_1F;
			if (*idum < 0) *idum += IM1_IWSN_1F;
			if (j < NTAB_IWSN_1F) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1_IWSN_1F;
	*idum = IA1_IWSN_1F * (*idum - k * IQ1_IWSN_1F) - k * IR1_IWSN_1F;
	if (*idum < 0) *idum += IM1_IWSN_1F;
	k = idum2 / IQ2_IWSN_1F;
	idum2 = IA2_IWSN_1F * (idum2 - k * IQ2_IWSN_1F) - k * IR2_IWSN_1F;
	if (idum2 < 0) idum2 += IM2_IWSN_1F;
	j = iy / NDIV_IWSN_1F;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1_IWSN_1F;  //printf("%lf\n", AM_CLASS*iy);
	if ((temp = AM_IWSN_1F * iy) > RNMX_IWSN_1F) return (vmin + RNMX_IWSN_1F * (vmax - vmin));
	else return (vmin + temp * (vmax - vmin));
}/*------End of rnd_uni_CLASS()--------------------------*/

static double Coverage()//cover
{
	Qoc();
	int i, j, k;
	double m = 0.0;
	for (i = 0; i < wid; i++)
		for (j = 0; j < lon; j++)
			for (k = 0; k < hig; k++) {
				m += qoc[i][j][k];
			}
	return (m / (N - occupVol));
}

static void Qoc()
{
	int i, j, k, l;
	double m;
	for (i = 0; i < wid; i++) {
		for (j = 0; j < lon; j++) {
			for (k = 0; k < hig; k++) {
				m = 1.0;
				for (l = 0; l < N_DIREC_1F; l++) {  //
					if (!posFlag_DIREC[l]) continue;
					m *= (1 + ga * Oq_DIREC(i, j, k, l));
				}
				m = (m - 1) / ga;
				if (m > oq_beta) {
					m = 1;
				}
				else {
					m = 0.0;
				}
				qoc[i][j][k] = m; //printf("%f\n",qoc[i][j][k]);
			}
		}
	}
	//for(i=0;i<lon/2;i++)
	//{
	//	for(j=0;j<wid/2;j++)
	//	{
	//		printf("%1.1f ",qoc[i][j]);
	//	}
	//printf("\n");
	//}
}

static double Oq_DIREC(int i, int j, int k, int l) //[0,2pi)
{
	double r;        //range (s,p)
	if (k < map[i][j])  //点在设备中
		return 0.0;
	if (pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j && pos3D_DIREC[l][Z] == k)  //
		return 1.0;
	if (LOS(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k)) {
		r = range(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k);
		//printf("%f\t",r);
		double pan;
		if (pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j)
			pan = 0;
		else
			pan = pan_angle[l] - ArcPan(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k); //anÎª(-2pi,2pi)
		if (pan < 0)
			pan = -pan; //[0,2pi)
		if (pan > pi)
			pan = 2 * pi - pan; //an为[0,pi]，像素点相对传感器的偏转角
		//if (pan < 0)
		//	printf("%f\t", pan);
		double tilt;
		tilt = tilt_angle[l] - ArcTilt(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k);
		if (tilt < 0)
			tilt = -tilt;
		if (tilt > pi)
			tilt = 2 * pi - tilt;
		//if (tilt < 0)
		//	printf("%f\t", tilt);
		double angle_pan;
		angle_pan = (pan * theta_H);
		//printf("%f\n",angle_pan);
		double angle_tilt;
		angle_tilt = (tilt * theta_V);
		//printf("%f\n",angle_tilt);
		double prob_d, prob_pan, prob_tilt;// , prob_ang;
		if (r <= radiusRs_DIREC[l])
			prob_d = 1.0;
		else if (r > radiusRs_DIREC[l] && r < radiusRs_DIREC[l] + radiusRf_DIREC[l]) {
			prob_d = exp(pow((cos(pi * (r - radiusRs_DIREC[l]) / radiusRf_DIREC[l]) - 1.0), mu_D) *
				pow(tan(pi / 2 * (r - radiusRs_DIREC[l]) / radiusRf_DIREC[l]), nu_D));
		}
		else
			return 0.0;
		if (angle_pan <= angle_range[l])
			prob_pan = 1.0;
		else if (angle_pan > angle_range[l] && angle_pan < angle_ratio_upp * angle_range[l]) {
			prob_pan = exp(pow((cos(pi * (angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])) - 1.0), mu_A) *
				pow(tan(pi / 2 * (angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])), nu_A));
			//printf("%lf->%lf\n",(angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),prob_pan);
		}
		else
			return 0.0;
		if (angle_tilt <= angle_range[l])
			prob_tilt = 1.0;
		else if (angle_tilt > angle_range[l] && angle_tilt < angle_ratio_upp * angle_range[l]) {
			prob_tilt = exp(pow((cos(pi * (angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])) - 1.0), mu_A) *
				pow(tan(pi / 2 * (angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])), nu_A));
			//printf("%lf->%lf\n",(angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),prob_tilt);
		}
		else
			return 0.0;
		//printf("%lf->%lf->%lf->%lf\n",(r - radiusRs_DIREC[l]) / radiusRf_DIREC[l],radiusRs_DIREC[l],radiusRs_DIREC[l] + radiusRf_DIREC[l],prob_d);
		//printf("%lf->%lf->%lf->%lf\n",(angle_3D - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),angle_range[l],angle_ratio_upp * angle_range[l],prob_ang);
		return (prob_d * prob_pan * prob_tilt);
	}
	else {
		return 0.0;
	}
}

static int LOS(int i, int j, int h, int a, int b, int c)                          ///LOSº¯Êý
{
	double hh = h;
	double cc = c;
	int m;
	int x, y, x1, y1;
	double z1, z2,
		z;  //z¿Õ¼äÖ±ÏßÉÏµã¸ß¶È   x¡¢yÎªÁ½µãºá×Ý×ø±ê²î£¨ÕýÕûÊý£© x1Îªµ¥Î»²îÁ¿£¨´ø·ûºÅ£© z1¡¢z2µØÐÎµÄ×ø±êµã¸ß¶È
	double k, k1;          //Á½µã³ÉÏßµÄÐ±ÂÊ
	double x2;            //ÓÃÓÚ´æ´¢ËùÇóµã×ø±ê
	double y2;
	int x3;
	int y3;             //½«y2ÕûÐÎ»¯
	x = abs(a - i);    //Á©µã¼äµÄ×ø±ê²î
	y = abs(b - j);
	if (x >= y) {
		if (i != a) {
			x1 = (a - i) / x;         //´æÕý¸ºÊý
			k = ((double)(b - j)) / ((double)(a - i));
			k1 = ((double)(cc - hh)) / (a - i);
			for (m = 1; m < x; m++) {
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				z = k1 * (x2 - i) + hh; ////x¡¢y¡¢z×ø±ê£¨Ö±Ïß£©
				if (y2 == (int)y2) {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					if (z < z1) {
						return 0;
					}
				}
				else {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					y3 = y3 + 1;//不会越界
					z2 = map[x3][y3];
					if (z < z1 || z < z2) {
						return 0;
					}
				}
			}
		}
	}
	else {
		if (j != b) {
			y1 = (b - j) / y;         //´æÕý¸ºÊý
			k = ((double)(a - i)) / ((double)(b - j));
			k1 = ((double)(cc - hh)) / (b - j);
			for (m = 1; m < y; m++) {
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				z = k1 * (y2 - j) + hh;
				if (x2 == (int)x2) {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					if (z < z1) {
						return 0;
					}
				}
				else {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					x3 = x3 + 1;//不会越界
					z2 = map[x3][y3];
					if (z < z1 || z < z2) {
						return 0;
					}
				}
			}
		}
	}
	return 1;
}

static double range(int i, int j, int k, int a, int b, int c) //¼ÆËã¾àÀë
{
	double r;
	r = sqrt((double)((i - a) * (i - a) + (j - b) * (j - b) + (k - c) * (k - c)));
	return r;
}

static double ArcPan(int i, int j, int k, int a, int b, int c)
{
	double temp;
	if (j == b) {
		if (a > i)
			return 0.0;
		else if (a < i)
			return pi;
	}
	temp = (a - i) / sqrt((double)(a - i) * (a - i) + (b - j) * (b - j));
	if (j > b)
		return 2 * pi - acos(temp);
	else
		return acos(temp);
}

static double ArcTilt(int i, int j, int k, int a, int b, int c)
{
	double temp;
	if (k == c)  //·µ»Ø½Ç¶È0
		return 0.0;
	temp = sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)) /
		sqrt((double)(a - i) * (a - i) + (b - j) * (b - j) + (c - k) * (c - k)); //ÇóÓàÏÒ
	if (k > c)
		return -acos(temp);
	else
		return acos(temp);
}

static double Lifetime()
{
	// initialization
	for (int i = 0; i < N_DIREC_1F; i++) {
		n_sn_rn_com[i] = 0;
	}
	for (int i = 0; i < N_RELAY_1F; i++) {
		n_rn_rn_com[i] = 0;
		n_data_local_RELAY[i] = 0;
		n_data_hop_RELAY[i] = 0;
		energy_consumed_RELAY[i] = 0;
	}
	avg_dist_SENSOR_IWSN_1F = 0.0;

	// sensor nodes
	for (int i = 0; i < N_DIREC_1F; i++) {
		double minD = 1.0e99;
		double tmpD;
		int ID = -1;
		for (int j = 0; j < N_RELAY_1F; j++) {
			tmpD = range(pos3D_DIREC[i][X], pos3D_DIREC[i][Y], pos3D_DIREC[i][Z],
				pos3D_RELAY[j][X], pos3D_RELAY[j][Y], pos3D_RELAY[j][Z]);
			if (tmpD * resIWSN_1F < d_th_sn)
				n_sn_rn_com[i]++;
			if (tmpD < minD) {
				minD = tmpD;
				ID = j;
			}
		}
		if (ID == -1) {
			hopID_SENSOR[i] = N_RELAY_1F;
			com_dist_SENSOR[i] = range((int)pos3D_DIREC[i][X], (int)pos3D_DIREC[i][Y], (int)pos3D_DIREC[i][Z],
				(int)SINK_X, (int)SINK_Y, (int)SINK_Z) * resIWSN_1F;
		}
		else {
			hopID_SENSOR[i] = ID;
			com_dist_SENSOR[i] = minD * resIWSN_1F;
			n_data_local_RELAY[ID] += l0;
		}
		avg_dist_SENSOR_IWSN_1F += com_dist_SENSOR[i];
	}
	avg_dist_SENSOR_IWSN_1F /= N_DIREC_1F;
	std_dist_SENSOR = 0.0;
	for (int i = 0; i < N_DIREC_1F; i++) {
		std_dist_SENSOR += (com_dist_SENSOR[i] - avg_dist_SENSOR_IWSN_1F) * (com_dist_SENSOR[i] - avg_dist_SENSOR_IWSN_1F);
	}
	std_dist_SENSOR = sqrt(std_dist_SENSOR / N_DIREC_1F);

	// relay nodes
	double dist2sink[N_RELAY_1F];
	int    ID_RELAY[N_RELAY_1F];
	for (int i = 0; i < N_RELAY_1F; i++) {
		ID_RELAY[i] = i;
		dist2sink[i] = range((int)pos3D_RELAY[i][X], (int)pos3D_RELAY[i][Y], (int)pos3D_RELAY[i][Z],
			(int)SINK_X, (int)SINK_Y, (int)SINK_Z);
	}
	for (int i = 0; i < N_RELAY_1F; i++) {
		for (int j = i + 1; j < N_RELAY_1F; j++) {
			if (dist2sink[i] < dist2sink[j]) {
				double tmpD = dist2sink[i];
				dist2sink[i] = dist2sink[j];
				dist2sink[j] = tmpD;
				int tmpID = ID_RELAY[i];
				ID_RELAY[i] = ID_RELAY[j];
				ID_RELAY[j] = tmpID;
			}
		}
	}

	for (int i = 0; i < N_RELAY_1F; i++) {
		int ID = ID_RELAY[i];
		n_data_fwd_RELAY[ID] = (1.0 - d_DA) * n_data_local_RELAY[ID] + n_data_hop_RELAY[ID];
		double minD = dist2sink[i];
		int tmpID = -1;
		for (int j = i + 1; j < N_RELAY_1F; j++) {
			int id = ID_RELAY[j];
			double tmpD = range(pos3D_RELAY[ID][X], pos3D_RELAY[ID][Y], pos3D_RELAY[ID][Z],
				pos3D_RELAY[id][X], pos3D_RELAY[id][Y], pos3D_RELAY[id][Z]);
			if (tmpD * resIWSN_1F < d_th_rn) {
				n_rn_rn_com[ID]++;
				n_rn_rn_com[id]++;
			}
			if (tmpD < minD) {
				minD = tmpD;
				tmpID = id;
			}
		}
		if (tmpID == -1) {
			hopID_RELAY[ID] = N_RELAY_1F;
			com_dist_RELAY[ID] = dist2sink[i] * resIWSN_1F;
		}
		else {
			hopID_RELAY[ID] = tmpID;
			n_data_hop_RELAY[tmpID] += n_data_fwd_RELAY[ID];
			com_dist_RELAY[ID] = minD * resIWSN_1F;
		}
	}

	double energy_max = -1.0;
	for (int i = 0; i < N_RELAY_1F; i++) {
		energy_consumed_RELAY[i] = (n_data_local_RELAY[i]) * E_DA +
			(n_data_local_RELAY[i] + n_data_hop_RELAY[i]) * E_elec;
		if (com_dist_RELAY[i] < d_th) {
			energy_consumed_RELAY[i] += n_data_fwd_RELAY[i] * E_elec +
				n_data_fwd_RELAY[i] * e_fs *
				com_dist_RELAY[i] * com_dist_RELAY[i];
		}
		else {
			energy_consumed_RELAY[i] += n_data_fwd_RELAY[i] * E_elec +
				n_data_fwd_RELAY[i] * e_mp *
				com_dist_RELAY[i] * com_dist_RELAY[i] * com_dist_RELAY[i] * com_dist_RELAY[i];
		}
		if (energy_consumed_RELAY[i] > energy_max) {
			energy_max = energy_consumed_RELAY[i];
		}
	}

	double LT_min;

	if (energy_max <= 0.0)
		LT_min = 1e-6;
	else
		LT_min = energy_ini / energy_max;

	int n_relia_p = 0;

	for (int i = 0; i < N_DIREC_1F; i++) {
		if (n_sn_rn_com[i] < n_rn_min)
			n_relia_p += (n_rn_min - n_sn_rn_com[i]);
	}
	for (int i = 0; i < N_RELAY_1F; i++) {
		if (n_rn_rn_com[i] < n_rn_min)
			n_relia_p += (n_rn_min - n_rn_rn_com[i]);
	}

	return (1e2 * (std_dist_SENSOR + avg_dist_SENSOR_IWSN_1F) / LT_min + n_relia_p * penaltyVal);
}