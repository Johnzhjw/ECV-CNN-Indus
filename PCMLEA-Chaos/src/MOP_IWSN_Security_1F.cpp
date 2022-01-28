#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "MOP_IWSN_Security_1F.h"

#define lon (32)
#define wid (32)
#define hig (2)
#define resIWSN_S_1F (5.0)
#define X (0)
#define Y (1)
#define Z (2)
#define PAN (2)
#define TILT (3)
#define pi (3.1415926535897932384626433832795)
#define r_direc_min (30.0/resIWSN_S_1F)
#define r_direc_max (48.0/resIWSN_S_1F)
#define r_direc_ratio (1.0)
#define N (lon*wid*hig)//总的要覆盖的离散点数
#define ga (-0.5)
#define oq_beta (0.9)
#define angle_min (pi/6/2)
#define angle_max (2*pi/9/2)
#define penaltyVal (1e6)
#define minH (1)
#define angle_ratio_upp (2.0)
#define mu_D (3.0)
#define nu_D (1.0)
#define mu_A (3.0)
#define nu_A (1.0)
#define theta_H (1.0)
#define theta_V (1.3)

#define INF_DOUBLE_S_1F (9.9E299)

static double map[wid][lon] = {
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
    { 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.00, 0.00, 0.00, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00, 0.00 },
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
//#define E_enc (1.62/8/128*1e-6)
//#define E_dec (2.49/8/128*1e-6)
#define E_M ((5.9e-6)/8)
#define v_d (12.4)
#define lm (200)
#define l0 (4000)
#define n_rn_min (2)
#define d_th_sn (40)
#define d_th_rn (80)

// position of the sink node
#define SINK_X (wid/2.0)
#define SINK_Y (lon-1.0)
#define SINK_Z (hig)

static double qoc[wid][lon][hig];
static int occupVol = 0; //设备所占点

static double radiusRs_DIREC[N_DIREC_S_1F];
static double radiusRf_DIREC[N_DIREC_S_1F];
static double pan_angle[N_DIREC_S_1F];
static double tilt_angle[N_DIREC_S_1F];
static double angle_range[N_DIREC_S_1F];

static int pos3D_DIREC[N_DIREC_S_1F][3];
static bool posFlag_DIREC[N_DIREC_S_1F];
static int pos3D_RELAY[N_RELAY_S_1F][3];
static bool posFlag_RELAY[N_RELAY_S_1F];

static int posCountBad;
static double total_penalty;
static double max_penalty_val = 0.0;
static double cur_penalty_val = 0.0;

static double energy_consumed_RELAY[N_RELAY_S_1F];

static int n_sn_rn_com[N_DIREC_S_1F];
static int n_rn_rn_com[N_RELAY_S_1F];
static int sn_hop_table[N_DIREC_S_1F][N_RELAY_S_1F];
static int rn_hop_table[N_RELAY_S_1F][N_RELAY_S_1F + 1];
static double dist_sn_rn[N_DIREC_S_1F][N_RELAY_S_1F];
static double dist_rn_rn[N_RELAY_S_1F][N_RELAY_S_1F + 1];
static double dist_rn_all[N_RELAY_S_1F][N_RELAY_S_1F + 1];
//static double route_topo_coding[N_DIREC_S_1F][NUM_MAX_ROUTE][LEN_MAX_ROUTE];
//static int    route_topo_all[N_DIREC_S_1F][NUM_MAX_ROUTE][LEN_MAX_ROUTE];
//static int    route_len_all[N_DIREC_S_1F][NUM_MAX_ROUTE];
static double prob_relay_rn_sn[N_RELAY_S_1F + 1][N_DIREC_S_1F];
//static double mean_hops_rn[N_RELAY_S_1F];
//static double mean_hops_sn[N_DIREC_S_1F];

static int n_sn_routing_path[N_DIREC_S_1F];
static int djrp_n_sn_rn_com[N_DIREC_S_1F];
static int djrp_sn_hop_table[N_DIREC_S_1F][N_RELAY_S_1F];
static int djrp_n_rn_rn_com[N_DIREC_S_1F][N_RELAY_S_1F];
static int djrp_rn_hop_table[N_DIREC_S_1F][N_RELAY_S_1F][N_RELAY_S_1F + 1];
static double djrp_prob_relay_rn_sn[N_DIREC_S_1F][N_RELAY_S_1F + 1];

static double dist2sink[N_RELAY_S_1F];
static int    ID_RELAY[N_RELAY_S_1F];
//static int    n_relayed_sensor[N_RELAY_S_1F];
//static int    n_local_sensor[N_RELAY_S_1F];

static double prob_p_sn[wid][lon][hig][N_DIREC_S_1F];

static int   seed_rand = 237;
static long  rnd_uni_init_val = -(long)seed_rand;
#define IM1_IWSN_S_1F 2147483563
#define IM2_IWSN_S_1F 2147483399
#define AM_IWSN_S_1F (1.0/IM1_IWSN_S_1F)
#define IMM1_IWSN_S_1F (IM1_IWSN_S_1F-1)
#define IA1_IWSN_S_1F 40014
#define IA2_IWSN_S_1F 40692
#define IQ1_IWSN_S_1F 53668
#define IQ2_IWSN_S_1F 52774
#define IR1_IWSN_S_1F 12211
#define IR2_IWSN_S_1F 3791
#define NTAB_IWSN_S_1F 32
#define NDIV_IWSN_S_1F (1+IMM1_IWSN_S_1F/NTAB_IWSN_S_1F)
#define EPS_IWSN_S_1F 1.2e-7
#define RNMX_IWSN_S_1F (1.0-EPS_IWSN_S_1F)

/***************************************************/
//函数
//the random generator in [0,1) ~ [vmin, vmax)
static double rnd_uni_gen(long* idum, double vmin, double vmax);
static bool TransPos(int a, int b, int* _3D);
static int AdjustPos_IWSN_S_1F(double* a, double* b);
static int AdjustAngle_IWSN_S_1F(int indx, double x, double y, double* ang_pan, double* ang_tilt);
void AdjustIndiv_whole_IWSN_S_1F(double* individual);
static double Coverage();//覆盖率目标函数
static void Qoc();
static double Oq_DIREC(int i, int j, double h, int l);
static int LOS(int i, int j, double h, int a, int b, double c);
static double range(int i, int j, double k, int a, int b, double c); //距离
static double ArcPan(int i, int j, double k, int a, int b, double c); //水平角度
static double ArcTilt(int i, int j, double k, int a, int b, double c); //垂直角度
static void   Get_topology();
static double Security();
static void   Update_topology_djrp();
static double Dist_relays();
//static double QoS_delay();
static double Lifetime_rn();//生命周期目标函数
static double Lifetime_rn_djrp();//生命周期目标函数
//static double Lifetime_sn();//生命周期目标函数
static void   Constraints();
static void   Constraints_djrp();
/***************************************************/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
#define dinic_min(elem_a, elem_b) ((elem_a)<(elem_b)?(elem_a):(elem_b))
const int dinic_INF = 0x7fffffff;
const int dinic_max_cap = 0x7fff;

struct dinic_edge {
    int y, r, next, op;
} dinic_a[401];
int dinic_head[201], dinic_q[5001], dinic_level[201], dinic_tot = 0;
void dinic_add_edge(int x, int y, int z)
{
    int tmp = dinic_head[x];
    while(tmp != -1) {
        if(tmp % 2 == 1 && y == dinic_a[tmp].y) {
            //dinic_a[tmp].r += z;
            return;
        }
        tmp = dinic_a[tmp].next;
    }
    dinic_a[++dinic_tot].y = y;
    dinic_a[dinic_tot].r = z;
    dinic_a[dinic_tot].next = dinic_head[x];
    dinic_head[x] = dinic_tot;
    dinic_a[dinic_tot].op = dinic_tot + 1;
    dinic_a[++dinic_tot].y = x;
    dinic_a[dinic_tot].r = 0;
    dinic_a[dinic_tot].next = dinic_head[y];
    dinic_head[y] = dinic_tot;
    dinic_a[dinic_tot].op = dinic_tot - 1;
}
bool dinic_bfs(int vs, int vt)
{
    int u, tmp, v, f = 1, r = 1;
    memset(dinic_level, 0, sizeof(dinic_level));
    dinic_q[f] = vs;
    dinic_level[vs] = 1;
    while(f <= r) {
        v = dinic_q[f];
        tmp = dinic_head[v];
        while(tmp != -1) {
            u = dinic_a[tmp].y;
            if(dinic_a[tmp].r && !dinic_level[u]) {
                dinic_level[u] = dinic_level[v] + 1;
                dinic_q[++r] = u;
                if(u == vt)return true;
            }
            tmp = dinic_a[tmp].next;
        }
        f++;
    }
    return false;
}
int dinic_dfs(int v, int vt, int num)
{
    int value, flow, tmp, u, ans = 0;
    if(v == vt || !num)return num;
    tmp = dinic_head[v];
    while(tmp != -1) {
        u = dinic_a[tmp].y;
        value = dinic_a[tmp].r;
        if(dinic_level[u] == dinic_level[v] + 1) {
            flow = dinic_dfs(u, vt, dinic_min(value, num));
            if(flow) {
                dinic_a[tmp].r -= flow;
                dinic_a[dinic_a[tmp].op].r += flow;
                ans += flow;
                num -= flow;
                if(!num)break;
            }
        }
        tmp = dinic_a[tmp].next;
    }
    return ans;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void Fitness_IWSN_S_1F(double* individual, double* fitness, double* constrainV, int nx, int M)
{
#ifdef ADJUST_INDIV_S_1F
    AdjustIndiv_whole_IWSN_S_1F(individual);
#endif
    //if (!checkLimits_IWSN_S_1F(individual, nx)) {
    //    printf("checkLimits_IWSN_1F FAIL, exiting...\n");
    //    exit(-1);
    //}
    posCountBad = 0;

    for(int i = 0; i < N_DIREC_S_1F; i++) {
        posFlag_DIREC[i] = TransPos((int)(individual[i * D_DIREC_S_1F + X]),
                                    (int)(individual[i * D_DIREC_S_1F + Y]),
                                    &pos3D_DIREC[i][0]);
        if(!posFlag_DIREC[i]) posCountBad++;
        pan_angle[i] = individual[i * D_DIREC_S_1F + PAN];
        tilt_angle[i] = individual[i * D_DIREC_S_1F + TILT];
    }
    int tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        posFlag_RELAY[i] = TransPos((int)(individual[tmp_offset + i * D_RELAY_S_1F + X]),
                                    (int)(individual[tmp_offset + i * D_RELAY_S_1F + Y]),
                                    &pos3D_RELAY[i][0]);
        //printf("%d %d %d\n", pos3D_RELAY[i][X], pos3D_RELAY[i][Y], pos3D_RELAY[i][Z]);
        if(!posFlag_RELAY[i]) posCountBad++;
    }
    //tmp_offset += N_RELAY_S_1F * D_RELAY_S_1F;
    //for(int i = 0; i < N_DIREC_S_1F; i++) {
    //    for(int j = 0; j < NUM_MAX_ROUTE; j++) {
    //        for(int k = 0; k < LEN_MAX_ROUTE; k++) {
    //            route_topo_coding[i][j][k] = individual[tmp_offset++];
    //        }
    //    }
    //}

    total_penalty = posCountBad * penaltyVal;

    Get_topology();
    double fit_covg = 1.0 - Coverage();
    double fit_secu = Security();
    double fit_dist = Dist_relays();
    Update_topology_djrp();
    double fit_life = Lifetime_rn_djrp();
    //fitness[2] = QoS_delay();
    //fitness[3] = Lifetime_rn();
    //fitness[4] = Lifetime_sn();
    Constraints_djrp();

    ////
    //double w_path = 0.5;
    //double w_rn_d = 0.5;
    //double final_fit;
    //final_fit = w_path * fit_secu + w_rn_d * fit_dist +
    //            w_path * w_rn_d * fit_secu * fit_dist;
    ////printf("\t\t\t\t\t%lf\n", fit_rn_d);
    ////final_fit = fit_sum_path * fit_rn_d;

    fitness[0] = fit_secu;
    fitness[1] = fit_dist;
    fitness[2] = fit_life;

    if(total_penalty > cur_penalty_val) {
        for(int i = 0; i < DIM_OBJ_IWSN_S_1F; i++) {
            fitness[i] += total_penalty;
        }
    }
    if(fit_covg > 0.1) {
        for(int i = 0; i < DIM_OBJ_IWSN_S_1F; i++) {
            fitness[i] += (fit_covg - 0.1) * penaltyVal * 100 + penaltyVal;
        }
    }

    //if(total_penalty > 0.0) {
    //    int mpi_rnk;
    //    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rnk);
    //    if(max_penalty_val < total_penalty) {
    //        max_penalty_val = total_penalty;
    //        printf("%d_total_penalty = %d ", mpi_rnk, (int)(max_penalty_val / 1e6));
    //    }
    //}

    return;
}

void SetLimits_IWSN_S_1F(double* minLimit, double* maxLimit, int nx)
{
    //printf("SUCCESS ");
    max_penalty_val = 50 * penaltyVal;

    seed_rand = 237;
    rnd_uni_init_val = -(long)seed_rand;

    int i, j, k;
    for(k = 0; k < N_DIREC_S_1F; k++) {
        minLimit[k * D_DIREC_S_1F + X] = 0;
        minLimit[k * D_DIREC_S_1F + Y] = 0;
        minLimit[k * D_DIREC_S_1F + PAN] = 0;
        minLimit[k * D_DIREC_S_1F + TILT] = -pi / 2.0;
        maxLimit[k * D_DIREC_S_1F + X] = wid + 2 * (hig - minH) - 1e-6;
        maxLimit[k * D_DIREC_S_1F + Y] = lon + 2 * (hig - minH) - 1e-6;
        maxLimit[k * D_DIREC_S_1F + PAN] = 2 * pi - 1e-6;
        maxLimit[k * D_DIREC_S_1F + TILT] = pi / 2.0;

        radiusRs_DIREC[k] = rnd_uni_gen(&rnd_uni_init_val, r_direc_min, r_direc_max);
        radiusRf_DIREC[k] = radiusRs_DIREC[k] * r_direc_ratio;
        angle_range[k] = rnd_uni_gen(&rnd_uni_init_val, angle_min, angle_max);
        //printf("%lf ", radiusRs_DIREC[k]);
        //printf("%lf\n", angle_range[k]);
    }
    occupVol = 0;
    for(i = 0; i < wid; i++) {
        for(j = 0; j < lon; j++) {
            for(k = 0; k < hig; k++) {
                if(k + 0.5 < map[i][j])
                    occupVol += 1;
            }
        }
    }
    int tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(k = 0; k < N_RELAY_S_1F; k++) {
        minLimit[tmp_offset + k * D_RELAY_S_1F + X] = 0;
        minLimit[tmp_offset + k * D_RELAY_S_1F + Y] = 0;
        maxLimit[tmp_offset + k * D_RELAY_S_1F + X] = wid + 2 * (hig - minH) - 1e-6;
        maxLimit[tmp_offset + k * D_RELAY_S_1F + Y] = lon + 2 * (hig - minH) - 1e-6;
    }
    //tmp_offset += N_RELAY_S_1F * D_RELAY_S_1F;
    //for(k = tmp_offset; k < DIM_IWSN_S_1F; k++) {
    //    minLimit[k] = 0.0;
    //    maxLimit[k] = 1.0 - 1e-6;
    //}

    return;
}

int CheckLimits_IWSN_S_1F(double* x, int nx)
{
    int k;

    for(k = 0; k < N_DIREC_S_1F; k++) {
        if(x[k * D_DIREC_S_1F + X] < 0 ||
           x[k * D_DIREC_S_1F + X] > wid + 2 * (hig - minH) - 1e-6) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   k * D_DIREC_S_1F + X, x[k * D_DIREC_S_1F + X], 0.0, (double)(wid + 2 * (hig - minH) - 1e-6));
            return false;
        }
        if(x[k * D_DIREC_S_1F + Y] < 0 ||
           x[k * D_DIREC_S_1F + Y] > lon + 2 * (hig - minH) - 1e-6) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   k * D_DIREC_S_1F + Y, x[k * D_DIREC_S_1F + Y], 0.0, (double)(lon + 2 * (hig - minH) - 1e-6));
            return false;
        }
        if(x[k * D_DIREC_S_1F + PAN] < 0 ||
           x[k * D_DIREC_S_1F + PAN] > 2 * pi - 1e-6) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   k * D_DIREC_S_1F + PAN, x[k * D_DIREC_S_1F + PAN], 0.0, 2 * pi - 1e-6);
            return false;
        }
        if(x[k * D_DIREC_S_1F + TILT] < -pi / 2 ||
           x[k * D_DIREC_S_1F + TILT] > pi / 2) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   k * D_DIREC_S_1F + TILT, x[k * D_DIREC_S_1F + TILT], -pi / 2, pi / 2);
            return false;
        }
    }
    int tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(k = 0; k < N_RELAY_S_1F; k++) {
        if(x[tmp_offset + k * D_RELAY_S_1F + X] < 0 ||
           x[tmp_offset + k * D_RELAY_S_1F + X] > wid + 2 * (hig - minH) - 1e-6) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   tmp_offset + k * D_RELAY_S_1F + X, x[tmp_offset + k * D_RELAY_S_1F + X],
                   0.0, (wid + 2 * (hig - minH) - 1e-6));
            return false;
        }
        if(x[tmp_offset + k * D_RELAY_S_1F + Y] < 0 ||
           x[tmp_offset + k * D_RELAY_S_1F + Y] > lon + 2 * (hig - minH) - 1e-6) {
            printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
                   tmp_offset + k * D_RELAY_S_1F + Y, x[tmp_offset + k * D_RELAY_S_1F + Y],
                   0.0, (lon + 2 * (hig - minH) - 1e-6));
            return false;
        }
    }
    //tmp_offset += N_RELAY_S_1F * D_RELAY_S_1F;
    //for(k = tmp_offset; k < DIM_IWSN_S_1F; k++) {
    //    if(x[k] < 0 || x[k] > 1.0 - 1e-6) {
    //        printf("Check limits FAIL - IWSN Security 1F: %d: %lf not in [%lf,%lf]\n",
    //               k, x[k], 0.0, 1.0 - 1e-6);
    //        return false;
    //    }
    //}

    return true;
}

static bool TransPos(int a, int b, int* _3D)
{
    int bound_left = hig - minH - 1;
    int bound_right = lon + hig - minH - 1;
    int bound_up = hig - minH - 1;
    int bound_down = wid + hig - minH - 1;

    if((a <= bound_up && b <= bound_left) ||
       (a <= bound_up && b > bound_right) ||
       (a > bound_down && b <= bound_left) ||
       (a > bound_down && b > bound_right)) {
        //printf("(%d,%d)!!!",a,b);
        return false;
    }
    if(!_3D)
        return true;

    if(a > bound_up && a <= bound_down && b <= bound_left) {        //left
        _3D[X] = a - (hig - minH);
        _3D[Y] = 0;
        _3D[Z] = b + minH;
    } else if(a > bound_up && a <= bound_down && b > bound_right) { //right
        _3D[X] = a - (hig - minH);
        _3D[Y] = lon - 1;
        _3D[Z] = lon + 2 * (hig - minH) - 1 - b + minH;
    } else if(b > bound_left && b <= bound_right && a <= bound_up) { //upp
        _3D[X] = 0;
        _3D[Y] = b - (hig - minH);
        _3D[Z] = a + minH;
    } else if(b > bound_left && b <= bound_right && a > bound_down) { //down
        _3D[X] = wid - 1;
        _3D[Y] = b - (hig - minH);
        _3D[Z] = wid + 2 * (hig - minH) - 1 - a + minH;
    } else { //center
        _3D[X] = a - (hig - minH);
        _3D[Y] = b - (hig - minH);
        _3D[Z] = hig;
    }

    //printf("(%d,%d)->(%d,%d,%d)",a,b,_3D[X],_3D[Y],_3D[Z]);
    return true;
}

static int AdjustPos_IWSN_S_1F(double* a, double* b)
{
    int bound_left = hig - minH - 1;
    int bound_right = lon + hig - minH - 1;
    int bound_up = hig - minH - 1;
    int bound_down = wid + hig - minH - 1;
    //
    int tmp_a = (int)(*a);
    int tmp_b = (int)(*b);
    //
    while(((int)(*a) <= bound_up && (int)(*b) <= bound_left) ||
          ((int)(*a) <= bound_up && (int)(*b) > bound_right) ||
          ((int)(*a) > bound_down && (int)(*b) <= bound_left) ||
          ((int)(*a) > bound_down && (int)(*b) > bound_right)) {
        if(rnd_uni_gen(&rnd_uni_init_val, 0.0, 1.0) < 0.5) {
            *a = rnd_uni_gen(&rnd_uni_init_val, bound_up + 1.0, bound_down + 1.0 - 1e-6);
        } else {
            *b = rnd_uni_gen(&rnd_uni_init_val, bound_left + 1.0, bound_right + 1.0 - 1e-6);
        }
    }
    //
    if(tmp_a <= bound_up && tmp_b <= bound_left) {
        return 1;
    }
    if(tmp_a <= bound_up && tmp_b > bound_right) {
        return 2;
    }
    if(tmp_a > bound_down && tmp_b <= bound_left) {
        return 3;
    }
    if(tmp_a > bound_down && tmp_b > bound_right) {
        return 4;
    }
    //
    return 0;
}

static int AdjustAngle_IWSN_S_1F(int indx, double x, double y, double* ang_pan, double* ang_tilt)
{
    //  pan 0 --- down, anti-clock
    // tilt 0 --- horizontal, up - positive, down - negative
    int bound_left = hig - minH - 1;
    int bound_right = lon + hig - minH - 1;
    int bound_up = hig - minH - 1;
    int bound_down = wid + hig - minH - 1;
    //
    int a = (int)x;
    int b = (int)y;
    //
    if(a > bound_up && a <= bound_down && b <= bound_left) {                //left
        *ang_pan = rnd_uni_gen(&rnd_uni_init_val, 0.0, pi);
        return 1;
    } else if(a > bound_up && a <= bound_down && b > bound_right) { //right
        *ang_pan = rnd_uni_gen(&rnd_uni_init_val, pi, 2 * pi - 1e-6);
        return 2;
    } else if(b > bound_left && b <= bound_right && a <= bound_up) { //upp
        if(rnd_uni_gen(&rnd_uni_init_val, 0.0, 1.0) < 0.5) {
            *ang_pan = rnd_uni_gen(&rnd_uni_init_val, 0, 0.5 * pi);
        } else {
            *ang_pan = rnd_uni_gen(&rnd_uni_init_val, 1.5 * pi, 2 * pi - 1e-6);
        }
        return 3;
    } else if(b > bound_left && b <= bound_right && a > bound_down) { //down
        *ang_pan = rnd_uni_gen(&rnd_uni_init_val, 0.5 * pi, 1.5 * pi);
        return 4;
    } else {                                           //center
        *ang_tilt = rnd_uni_gen(&rnd_uni_init_val, -0.5 * pi, 0);
        return 0;
    }
}

void AdjustIndiv_whole_IWSN_S_1F(double* individual)
{
    //
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        AdjustPos_IWSN_S_1F(&individual[i * D_DIREC_S_1F + X],
                            &individual[i * D_DIREC_S_1F + Y]);
        //AdjustAngle_S_1F(i,
        //                 individual[i * D_DIREC_S_1F + X],
        //                 individual[i * D_DIREC_S_1F + Y],
        //                 &individual[i * D_DIREC_S_1F + PAN],
        //                 &individual[i * D_DIREC_S_1F + TILT]);
    }
    int tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        AdjustPos_IWSN_S_1F(&individual[tmp_offset + i * D_RELAY_S_1F + X],
                            &individual[tmp_offset + i * D_RELAY_S_1F + Y]);
    }
    //
    return;
}

void check_and_repair_IWSN_S_1F(double* individual)
{
    //
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        AdjustPos_IWSN_S_1F(&individual[i * D_DIREC_S_1F + X],
                            &individual[i * D_DIREC_S_1F + Y]);
        //AdjustAngle_S_1F(i,
        //                 individual[i * D_DIREC_S_1F + X],
        //                 individual[i * D_DIREC_S_1F + Y],
        //                 &individual[i * D_DIREC_S_1F + PAN],
        //                 &individual[i * D_DIREC_S_1F + TILT]);
    }
    int tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        AdjustPos_IWSN_S_1F(&individual[tmp_offset + i * D_RELAY_S_1F + X],
                            &individual[tmp_offset + i * D_RELAY_S_1F + Y]);
    }
    //
    posCountBad = 0;
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        posFlag_DIREC[i] = TransPos((int)(individual[i * D_DIREC_S_1F + X]),
                                    (int)(individual[i * D_DIREC_S_1F + Y]),
                                    &pos3D_DIREC[i][0]);
        if(!posFlag_DIREC[i]) posCountBad++;
        pan_angle[i] = individual[i * D_DIREC_S_1F + PAN];
        tilt_angle[i] = individual[i * D_DIREC_S_1F + TILT];
    }
    tmp_offset = N_DIREC_S_1F * D_DIREC_S_1F;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        posFlag_RELAY[i] = TransPos((int)(individual[tmp_offset + i * D_RELAY_S_1F + X]),
                                    (int)(individual[tmp_offset + i * D_RELAY_S_1F + Y]),
                                    &pos3D_RELAY[i][0]);
        //printf("%d %d %d\n", pos3D_RELAY[i][X], pos3D_RELAY[i][Y], pos3D_RELAY[i][Z]);
        if(!posFlag_RELAY[i]) posCountBad++;
    }
    if(posCountBad > 0) {
        printf("%s(%d): Adjust pos error --- %d\n", __FILE__, __LINE__, posCountBad);
    }
    //
    Get_topology();
    //
    return;
}

void adjust_constraints_IWSN_S_1F(double cur_iter, double max_iter)
{
    if(cur_iter < max_iter / 2) {
        cur_penalty_val = (1.0 - cur_iter / max_iter / 2) * max_penalty_val;
    } else {
        cur_penalty_val = 0.0;
    }

    return;
}

//the random generator in [0,1)
double rnd_uni_gen(long* idum, double vmin, double vmax)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_IWSN_S_1F];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_IWSN_S_1F + 7; j >= 0; j--) {
            k = (*idum) / IQ1_IWSN_S_1F;
            *idum = IA1_IWSN_S_1F * (*idum - k * IQ1_IWSN_S_1F) - k * IR1_IWSN_S_1F;
            if(*idum < 0) *idum += IM1_IWSN_S_1F;
            if(j < NTAB_IWSN_S_1F) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_IWSN_S_1F;
    *idum = IA1_IWSN_S_1F * (*idum - k * IQ1_IWSN_S_1F) - k * IR1_IWSN_S_1F;
    if(*idum < 0) *idum += IM1_IWSN_S_1F;
    k = idum2 / IQ2_IWSN_S_1F;
    idum2 = IA2_IWSN_S_1F * (idum2 - k * IQ2_IWSN_S_1F) - k * IR2_IWSN_S_1F;
    if(idum2 < 0) idum2 += IM2_IWSN_S_1F;
    j = iy / NDIV_IWSN_S_1F;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_IWSN_S_1F;        //printf("%lf\n", AM_CLASS*iy);
    if((temp = AM_IWSN_S_1F * iy) > RNMX_IWSN_S_1F) return (vmin + RNMX_IWSN_S_1F * (vmax - vmin));
    else return (vmin + temp * (vmax - vmin));
}/*------End of rnd_uni_CLASS()--------------------------*/

static double Coverage()//cover
{
    Qoc();
    int i, j, k;
    double m = 0.0;
    for(i = 0; i < wid; i++)
        for(j = 0; j < lon; j++)
            for(k = 0; k < hig; k++) {
                m += qoc[i][j][k];
            }
    return (m / (N - occupVol));
}

static void Qoc()
{
    int i, j, k, l;
    double m;
    double h;
    for(i = 0; i < wid; i++) {
        for(j = 0; j < lon; j++) {
            for(k = 0; k < hig; k++) {
                m = 1.0;
                h = k + 0.5;
                for(l = 0; l < N_DIREC_S_1F; l++) {        //
                    if(!posFlag_DIREC[l]) continue;
                    m *= (1 + ga * Oq_DIREC(i, j, h, l));
                }
                m = (m - 1) / ga;
                if(m > oq_beta) {
                    m = 1;
                } else {
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

double Oq_DIREC(int i, int j, double h, int l)
{
    double r;        //range (s,p)
    if(h < map[i][j])        //点在设备中
        return 0.0;
    if(pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j && pos3D_DIREC[l][Z] == h)        //
        return 1.0;
    if(LOS(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, h)) {
        r = range(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, h);
        //printf("%f\t",r);
        double pan;
        if(pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j)
            pan = 0;
        else
            pan = pan_angle[l] - ArcPan(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, h); //anÎª(-2pi,2pi)
        if(pan < 0)
            pan = -pan; //[0,2pi)
        if(pan > pi)
            pan = 2 * pi - pan; //an为[0,pi]，像素点相对传感器的偏转角
        //if (pan < 0)
        //	printf("%f\t", pan);
        double tilt;
        tilt = tilt_angle[l] - ArcTilt(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, h);
        if(tilt < 0)
            tilt = -tilt;
        if(tilt > pi)
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
        double tmp1, tmp2;
        if(r <= radiusRs_DIREC[l])
            prob_d = 1.0;
        else if(r < radiusRs_DIREC[l] + radiusRf_DIREC[l]) {
            tmp1 = cos(pi * (r - radiusRs_DIREC[l]) / radiusRf_DIREC[l]) - 1.0;
            tmp2 = tan(pi / 2 * (r - radiusRs_DIREC[l]) / radiusRf_DIREC[l]);
            prob_d = exp(pow(tmp1, mu_D) * pow(tmp2, nu_D));
        } else
            return 0.0;
        if(angle_pan <= angle_range[l])
            prob_pan = 1.0;
        else if(angle_pan < angle_ratio_upp * angle_range[l]) {
            tmp1 = cos(pi * (angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])) - 1.0;
            tmp2 = tan(pi / 2 * (angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]));
            prob_pan = exp(pow(tmp1, mu_A) * pow(tmp2, nu_A));
            //printf("%lf->%lf\n",(angle_pan - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),prob_pan);
        } else
            return 0.0;
        if(angle_tilt <= angle_range[l])
            prob_tilt = 1.0;
        else if(angle_tilt < angle_ratio_upp * angle_range[l]) {
            tmp1 = cos(pi * (angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l])) - 1.0;
            tmp2 = tan(pi / 2 * (angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]));
            prob_tilt = exp(pow(tmp1, mu_A) * pow(tmp2, nu_A));
            //printf("%lf->%lf\n",(angle_tilt - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),prob_tilt);
        } else
            return 0.0;
        //printf("%lf->%lf->%lf->%lf\n",(r - radiusRs_DIREC[l]) / radiusRf_DIREC[l],radiusRs_DIREC[l],radiusRs_DIREC[l] + radiusRf_DIREC[l],prob_d);
        //printf("%lf->%lf->%lf->%lf\n",(angle_3D - angle_range[l]) / ((angle_ratio_upp - 1.0) * angle_range[l]),angle_range[l],angle_ratio_upp * angle_range[l],prob_ang);
        return (prob_d * prob_pan * prob_tilt);
    } else {
        return 0.0;
    }
}

static int LOS(int i, int j, double h, int a, int b, double c)
{
    double hh = h;
    double cc = c;
    int m;
    int x, y, x1, y1;
    double z1, z2, z;  //
    double k, k1;          //Á½µã³ÉÏßµÄÐ±ÂÊ
    double x2;            //ÓÃÓÚ´æ´¢ËùÇóµã×ø±ê
    double y2;
    int x3;
    int y3;             //½«y2ÕûÐÎ»¯
    x = abs(a - i);    //Á©µã¼äµÄ×ø±ê²î
    y = abs(b - j);
    if(x >= y) {
        if(i != a) {
            x1 = (a - i) / x;         //´æÕý¸ºÊý
            k = ((double)(b - j)) / ((double)(a - i));
            k1 = ((double)(cc - hh)) / (a - i);
            for(m = 1; m < x; m++) {
                x2 = i + x1 * m;
                y2 = k * (x2 - i) + j;
                z = k1 * (x2 - i) + hh; ////x¡¢y¡¢z×ø±ê£¨Ö±Ïß£©
                if(y2 == (int)y2) {
                    x3 = (int)x2;
                    y3 = (int)y2;
                    z1 = map[x3][y3];
                    if(z < z1) {
                        return 0;
                    }
                } else {
                    x3 = (int)x2;
                    y3 = (int)y2;
                    z1 = map[x3][y3];
                    y3 = y3 + 1;//不会越界
                    z2 = map[x3][y3];
                    if(z < z1 || z < z2) {
                        return 0;
                    }
                }
            }
        }
    } else {
        if(j != b) {
            y1 = (b - j) / y;         //´æÕý¸ºÊý
            k = ((double)(a - i)) / ((double)(b - j));
            k1 = ((double)(cc - hh)) / (b - j);
            for(m = 1; m < y; m++) {
                y2 = j + y1 * m;
                x2 = k * (y2 - j) + i;
                z = k1 * (y2 - j) + hh;
                if(x2 == (int)x2) {
                    x3 = (int)x2;
                    y3 = (int)y2;
                    z1 = map[x3][y3];
                    if(z < z1) {
                        return 0;
                    }
                } else {
                    x3 = (int)x2;
                    y3 = (int)y2;
                    z1 = map[x3][y3];
                    x3 = x3 + 1;//不会越界
                    z2 = map[x3][y3];
                    if(z < z1 || z < z2) {
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}

double range(int i, int j, double k, int a, int b, double c)
{
    double r;
    r = sqrt((double)((i - a) * (i - a) + (j - b) * (j - b) + (k - c) * (k - c)));
    return r;
}

double ArcPan(int i, int j, double k, int a, int b, double c)
{
    double temp;
    if(j == b) {
        if(a > i)
            return 0.0;
        else if(a < i)
            return pi;
    }
    temp = (a - i) / sqrt((double)(a - i) * (a - i) + (b - j) * (b - j));
    if(j > b)
        return 2 * pi - acos(temp);
    else
        return acos(temp);
}

double ArcTilt(int i, int j, double k, int a, int b, double c)
{
    double temp;
    if(k == c)        //·µ»Ø½Ç¶È0
        return 0.0;
    temp = sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)) /
           sqrt((double)(a - i) * (a - i) + (b - j) * (b - j) + (c - k) * (c - k)); //ÇóÓàÏÒ
    if(k > c)
        return -acos(temp);
    else
        return acos(temp);
}

static void Get_topology()
{
    // initialization
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        n_sn_rn_com[i] = 0;
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            sn_hop_table[i][j] = 0;
            dist_sn_rn[i][j] = INF_DOUBLE_S_1F;
        }
    }
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        n_rn_rn_com[i] = 0;
        for(int j = 0; j < N_RELAY_S_1F + 1; j++) {
            rn_hop_table[i][j] = 0;
            dist_rn_rn[i][j] = INF_DOUBLE_S_1F;
            dist_rn_all[i][j] = INF_DOUBLE_S_1F;
        }
    }
    // sensor nodes
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        double tmpD;
        if(!posFlag_DIREC[i]) {
            continue;
        }
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(!posFlag_RELAY[j]) {
                continue;
            }
            tmpD = range(pos3D_DIREC[i][X], pos3D_DIREC[i][Y], pos3D_DIREC[i][Z],
                         pos3D_RELAY[j][X], pos3D_RELAY[j][Y], pos3D_RELAY[j][Z]) *
                   resIWSN_S_1F;
            if(tmpD <= d_th_sn) {
                n_sn_rn_com[i]++;
                sn_hop_table[i][j] = 1;
                dist_sn_rn[i][j] = tmpD;
            }
        }
    }
    // relay nodes
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        ID_RELAY[i] = i;
        if(!posFlag_RELAY[i])
            dist2sink[i] = INF_DOUBLE_S_1F;
        else
            dist2sink[i] = range((int)pos3D_RELAY[i][X], (int)pos3D_RELAY[i][Y], (int)pos3D_RELAY[i][Z],
                                 (int)SINK_X, (int)SINK_Y, (int)SINK_Z) *
                           resIWSN_S_1F;
    }
    for(int i = N_RELAY_S_1F - 1; i > 0; i--) {
        for(int j = 0; j < i; j++) {
            if(dist2sink[j] < dist2sink[j + 1]) {
                double tmpD = dist2sink[j];
                dist2sink[j] = dist2sink[j + 1];
                dist2sink[j + 1] = tmpD;
                int tmpID = ID_RELAY[j];
                ID_RELAY[j] = ID_RELAY[j + 1];
                ID_RELAY[j + 1] = tmpID;
            }
        }
    }
    //
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        int ID = ID_RELAY[i];
        if(!posFlag_RELAY[ID]) {
            continue;
        }
        for(int j = i + 1; j < N_RELAY_S_1F; j++) {
            int id = ID_RELAY[j];
            if(!posFlag_RELAY[id]) {
                continue;
            }
            double tmpD = range(pos3D_RELAY[ID][X], pos3D_RELAY[ID][Y], pos3D_RELAY[ID][Z],
                                pos3D_RELAY[id][X], pos3D_RELAY[id][Y], pos3D_RELAY[id][Z]) *
                          resIWSN_S_1F;
            if(tmpD <= d_th_rn &&
               tmpD > 0.0 &&
               dist2sink[i] > dist2sink[j]) {
                n_rn_rn_com[ID]++;
                rn_hop_table[ID][id] = 1;
                dist_rn_rn[ID][id] = tmpD;
            }
            dist_rn_all[ID][id] = dist_rn_all[id][ID] = tmpD;
        }
        if(dist2sink[i] <= d_th_rn) {
            for(int j = 0; j < N_RELAY_S_1F; j++) {
                rn_hop_table[ID][j] = 0;
                dist_rn_rn[ID][j] = INF_DOUBLE_S_1F;
            }
            n_rn_rn_com[ID] = 1;
            rn_hop_table[ID][N_RELAY_S_1F] = 1;
            dist_rn_rn[ID][N_RELAY_S_1F] = dist2sink[i];
        }
        dist_rn_all[ID][N_RELAY_S_1F] = dist2sink[i];
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    //for(int i = 0; i < N_DIREC_S_1F; i++) {
    //    for(int j = 0; j < NUM_MAX_ROUTE; j++) {
    //        route_len_all[i][j] = 0;
    //        for(int k = 0; k < LEN_MAX_ROUTE; k++) {
    //            route_topo_all[i][j][k] = -1;
    //        }
    //    }
    //}

    //for(int i = 0; i < N_DIREC_S_1F; i++) {
    //    for(int j = 0; j < NUM_MAX_ROUTE; j++) {
    //        if(n_sn_rn_com[i] == 0) {
    //            route_len_all[i][j] = 0;
    //            continue;
    //        }
    //        int len_route = 0;
    //        int next_hop_id = route_topo_coding[i][j][0] * n_sn_rn_com[i];
    //        int next_hop = -1;
    //        for(int k = 0; k < N_RELAY_S_1F; k++) {
    //            if(sn_hop_table[i][k]) {
    //                next_hop = k;
    //                next_hop_id--;
    //            }
    //            if(next_hop_id < 0) {
    //                break;
    //            }
    //        }
    //        route_topo_all[i][j][len_route] = next_hop;
    //        len_route++;
    //        while(1) {
    //            int cur_rn_id = next_hop;
    //            if(n_rn_rn_com[cur_rn_id] == 0) {
    //                if(dist2sink[ID_RELAY[cur_rn_id]] * resIWSN_S_1F <= d_th_rn) {
    //                    route_len_all[i][j] = len_route;
    //                    break;
    //                } else {
    //                    route_len_all[i][j] = 0;
    //                    break;
    //                }
    //            }
    //            next_hop_id = route_topo_coding[i][j][len_route] * n_rn_rn_com[cur_rn_id];
    //            for(int k = 0; k < N_RELAY_S_1F; k++) {
    //                if(rn_hop_table[cur_rn_id][k]) {
    //                    next_hop = k;
    //                    next_hop_id--;
    //                }
    //                if(next_hop_id < 0) {
    //                    break;
    //                }
    //            }
    //            route_topo_all[i][j][len_route] = next_hop;
    //            len_route++;
    //        }
    //    }
    //}

    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        for(int j = 0; j < N_DIREC_S_1F; j++) {
            if(sn_hop_table[j][i]/* && n_sn_rn_com[j]*/) {
                prob_relay_rn_sn[i][j] = 1.0 / n_sn_rn_com[j];
            } else {
                prob_relay_rn_sn[i][j] = 0.0;
            }
        }
    }
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        int ID = ID_RELAY[i];
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(rn_hop_table[ID][j]) {
                for(int k = 0; k < N_DIREC_S_1F; k++) {
                    prob_relay_rn_sn[j][k] += prob_relay_rn_sn[ID][k] / n_rn_rn_com[ID];
                }
            }
        }
    }
    //
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        if(rn_hop_table[i][N_RELAY_S_1F]) {
            for(int k = 0; k < N_DIREC_S_1F; k++) {
                prob_relay_rn_sn[N_RELAY_S_1F][k] += prob_relay_rn_sn[i][k] / n_rn_rn_com[i];
            }
        }
    }

    return;
}

static double Security()
{
    int ans = 0, vs, vt;
    double fit_sum_path = 0;
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        dinic_tot = 0;
        ans = 0;
        vs = 0;
        vt = 2 * N_RELAY_S_1F + 1;
        memset(dinic_head, 255, sizeof(dinic_head));
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            dinic_add_edge(j + 1, j + 1 + N_RELAY_S_1F, 1);
        }
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(sn_hop_table[i][j]) {
                dinic_add_edge(vs, j + 1, dinic_max_cap);
            }
        }
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            for(int k = 0; k < N_RELAY_S_1F; k++) {
                if(rn_hop_table[j][k]) {
                    dinic_add_edge(j + 1 + N_RELAY_S_1F, k + 1, dinic_max_cap);
                }
            }
            if(rn_hop_table[j][N_RELAY_S_1F]) {
                dinic_add_edge(j + 1 + N_RELAY_S_1F, vt, dinic_max_cap);
            }
        }
        while(dinic_bfs(vs, vt)) {
            ans += dinic_dfs(vs, vt, dinic_INF);
        }
        n_sn_routing_path[i] = ans;
        fit_sum_path += ans;
        //////////////////////////////////////////////////////////////////////////
        djrp_n_sn_rn_com[i] = 0;
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            djrp_sn_hop_table[i][j] = 0;
        }
        //
        int tmp = dinic_head[vs];
        while(tmp != -1) {
            if(tmp % 2 == 1) {
                int tmp_y = dinic_a[tmp].y;
                int tmp_j = tmp_y - 1;
                if(tmp_j < 0 || tmp_j >= N_RELAY_S_1F) {
                    printf("%s(%d): index error, ...\n", __FILE__, __LINE__);
                }
                if(dinic_a[tmp].r < dinic_max_cap) {
                    if(dinic_a[tmp].r < dinic_max_cap - 1) {
                        printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                    }
                    if(djrp_sn_hop_table[i][tmp_j] == 0) {
                        djrp_n_sn_rn_com[i]++;
                        djrp_sn_hop_table[i][tmp_j] = 1;
                    }
                }
            }
            tmp = dinic_a[tmp].next;
        }
        //
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            djrp_n_rn_rn_com[i][j] = 0;
            for(int k = 0; k <= N_RELAY_S_1F; k++) {
                djrp_rn_hop_table[i][j][k] = 0;
            }
        }
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            tmp = dinic_head[j + 1 + N_RELAY_S_1F];
            while(tmp != -1) {
                if(tmp % 2 == 1) {
                    int tmp_y = dinic_a[tmp].y;
                    int tmp_j = tmp_y - 1;
                    if(tmp_j >= 0 && tmp_j < N_RELAY_S_1F) {
                        if(dinic_a[tmp].r < dinic_max_cap) {
                            if(dinic_a[tmp].r < dinic_max_cap - 1) {
                                printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                            }
                            if(djrp_rn_hop_table[i][j][tmp_j] == 0) {
                                djrp_n_rn_rn_com[i][j]++;
                                djrp_rn_hop_table[i][j][tmp_j] = 1;
                            }
                        }
                    } else if(tmp_j == vt - 1) {
                        if(dinic_a[tmp].r < dinic_max_cap) {
                            if(dinic_a[tmp].r < dinic_max_cap - 1) {
                                printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                            }
                            if(djrp_rn_hop_table[i][j][N_RELAY_S_1F] == 0) {
                                djrp_n_rn_rn_com[i][j]++;
                                djrp_rn_hop_table[i][j][N_RELAY_S_1F] = 1;
                            }
                        }
                    } else {
                        printf("%s(%d): index error, ...\n", __FILE__, __LINE__);
                    }
                }
                tmp = dinic_a[tmp].next;
            }
        }
        //////////////////////////////////////////////////////////////////////////
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            tmp = dinic_head[j + 1];
            while(tmp != -1) {
                if(tmp % 2 == 1) {
                    int tmp_y = dinic_a[tmp].y;
                    if(tmp_y != j + 1 + N_RELAY_S_1F) {
                        printf("%s(%d): index error, ...\n", __FILE__, __LINE__);
                    }
                    if(dinic_a[tmp].r < 1) {
                        if(dinic_a[tmp].r < 0) {
                            printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                        }
                        if(djrp_n_rn_rn_com[i][j] == 0) {
                            printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                        }
                    } else {
                        if(djrp_n_rn_rn_com[i][j]) {
                            printf("%s(%d): capacity error, ...\n", __FILE__, __LINE__);
                        }
                    }
                }
                tmp = dinic_a[tmp].next;
            }
        }
        //
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(djrp_n_rn_rn_com[i][j] > 1) {
                printf("%s(%d): djrp_n_rn_rn_com error, ...\n", __FILE__, __LINE__);
            }
        }
    }
    fit_sum_path /= N_DIREC_S_1F;
    fit_sum_path /= N_RELAY_S_1F;
    fit_sum_path = 1.0 - fit_sum_path;
    //
    return fit_sum_path;
}

static void Update_topology_djrp()
{
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(djrp_sn_hop_table[i][j]/* && n_sn_rn_com[i]*/) {
                djrp_prob_relay_rn_sn[i][j] = 1.0 / djrp_n_sn_rn_com[i];
            } else {
                djrp_prob_relay_rn_sn[i][j] = 0.0;
            }
        }
    }
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            int ID = ID_RELAY[j];
            for(int k = 0; k <= N_RELAY_S_1F; k++) {
                if(djrp_rn_hop_table[i][ID][k]) {
                    djrp_prob_relay_rn_sn[i][k] += djrp_prob_relay_rn_sn[i][ID] / djrp_n_rn_rn_com[i][ID];
                }
            }
        }
    }
    //
    return;
}

static double Dist_relays()
{
    //
    double fit_rn_d = 0.0;
    int tmp_count = 0;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        double tmp_min = INF_DOUBLE_S_1F;
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(tmp_min > dist_rn_all[i][j]) {
                tmp_min = dist_rn_all[i][j];
            }
        }
        if(tmp_min < INF_DOUBLE_S_1F) {
            fit_rn_d += (d_th_rn - tmp_min) / d_th_rn;
            tmp_count++;
        }
    }
    fit_rn_d /= tmp_count;

    return fit_rn_d;
}

//static double Security()
//{
//    double fit_sum_sn = 0.0;
//    double fit_sum_rn = 0.0;
//    double fit_rn_d = INF_DOUBLE_S_1F;
//    for(int i = 0; i < N_DIREC_S_1F; i++) {
//        fit_sum_sn += N_RELAY_S_1F - n_sn_rn_com[i];
//    }
//    for(int i = 0; i < N_RELAY_S_1F; i++) {
//        fit_sum_rn += N_RELAY_S_1F - n_rn_rn_com[i];
//        for(int j = 0; j < N_RELAY_S_1F; j++) {
//            if(fit_rn_d > dist_rn_all[i][j]) {
//                fit_rn_d = dist_rn_all[i][j];
//            }
//        }
//    }
//    fit_sum_sn /= N_DIREC_S_1F * N_RELAY_S_1F;
//    fit_sum_rn /= N_RELAY_S_1F * N_RELAY_S_1F;
//    fit_rn_d /= d_th_rn;
//    if(fit_rn_d <= 1e-6)
//        fit_rn_d = 1e-6;
//    fit_rn_d = 1.0 - fit_rn_d;
//    //
//    double w_sn = 1.0 / 3.0;
//    double w_rn = 1.0 / 3.0;
//    double w_rn_d = 1.0 / 3.0;
//    double final_fit = 0.0;
//    final_fit = w_sn * fit_sum_sn + w_rn * fit_sum_rn + w_rn_d * fit_rn_d +
//                w_sn * w_rn * fit_sum_sn * fit_sum_rn +
//                w_sn * w_rn_d * fit_sum_sn * fit_rn_d +
//                w_rn * w_rn_d * fit_sum_rn * fit_rn_d +
//                w_sn * w_rn * w_rn_d * fit_sum_sn * fit_sum_rn * fit_rn_d;
//    //final_fit = fit_sum_sn * fit_sum_rn * fit_rn_d;
//    //
//    return (final_fit);
//}

//static double Security()
//{
//    double prob_sum = 0.0;
//    for(int i = 0; i < N_RELAY_S_1F; i++) {
//        double prob_max = 0.0;
//        for(int j = 0; j < N_DIREC_S_1F; j++) {
//            if(prob_max < prob_relay_rn_sn[i][j])
//                prob_max = prob_relay_rn_sn[i][j];
//        }
//        prob_sum += prob_max;
//    }
//    return (prob_sum / N_RELAY_S_1F);
//}

//static double Security()
//{
//    int i, j, k, l;
//    double h;
//    //
//    for(i = 0; i < wid; i++) {
//        for(j = 0; j < lon; j++) {
//            for(k = 0; k < hig; k++) {
//                for(l = 0; l < N_DIREC_S_1F; l++) {    //
//                    prob_p_sn[i][j][k][l] = 0.0;
//                }
//            }
//        }
//    }
//    //
//    int a, b;
//    double m[N_DIREC_S_1F];
//    for(i = 0; i < wid; i++) {
//        for(j = 0; j < lon; j++) {
//            for(k = 0; k < hig; k++) {
//                h = k + 0.5;
//                if(h < map[i][j]) continue;
//                //
//                for(l = 0; l < N_DIREC_S_1F; l++) m[l] = 1.0;
//                //
//                for(l = 0; l < N_DIREC_S_1F; l++) {    //
//                    if(!posFlag_DIREC[l]) continue;
//                    double cur_dist = range(i, j, h,
//                                            pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z]) *
//                                      resIWSN_S_1F;
//                    for(a = 0; a < N_RELAY_S_1F; a++) {
//                        if(sn_hop_table[l][a] &&
//                           cur_dist <= dist_sn_rn[l][a]) {
//                            //prob_p_sn[i][j][k][l] += 1.0 / n_sn_rn_com[l];
//                            m[l] *= 1.0 - 1.0 / n_sn_rn_com[l];
//                        }
//                    }
//                }
//                //
//                //for(l = 0; l < N_DIREC_S_1F; l++) m[l] = 1.0 - m[l];
//                //
//                for(l = 0; l < N_RELAY_S_1F; l++) {
//                    if(!posFlag_RELAY[l]) continue;
//                    double cur_dist = range(i, j, h,
//                                            pos3D_RELAY[l][X], pos3D_RELAY[l][Y], pos3D_RELAY[l][Z]) *
//                                      resIWSN_S_1F;
//                    for(a = 0; a <= N_RELAY_S_1F; a++) {
//                        if(rn_hop_table[l][a] &&
//                           cur_dist <= dist_rn_rn[l][a]) {
//                            for(b = 0; b < N_DIREC_S_1F; b++) {
//                                //prob_p_sn[i][j][k][b] += prob_relay_rn_sn[l][b] / n_rn_rn_com[l];
//                                m[b] *= (1.0 - prob_relay_rn_sn[l][b] / n_rn_rn_com[l]);
//                            }
//                        }
//                    }
//                }
//                //
//                for(l = 0; l < N_DIREC_S_1F; l++) {
//                    prob_p_sn[i][j][k][l] = 1.0 - m[l];
//                }
//            }
//        }
//    }
//    //
//    double prob_sum = 0.0;
//    for(i = 0; i < wid; i++) {
//        for(j = 0; j < lon; j++) {
//            for(k = 0; k < hig; k++) {
//                h = k + 0.5;
//                if(h < map[i][j]) continue;
//                double prob_tmp = 0.0;
//                for(l = 0; l < N_DIREC_S_1F; l++) {     //
//                    //if(prob_tmp < prob_p_sn[i][j][k][l]) {
//                    //    prob_tmp = prob_p_sn[i][j][k][l];
//                    //}
//                    prob_tmp += prob_p_sn[i][j][k][l];
//                }
//                prob_sum += prob_tmp / N_DIREC_S_1F;
//            }
//        }
//    }
//    //
//    return (prob_sum / (N - occupVol));
//}

//static double Security()
//{
//    double prob_sum = 0.0;
//    for(int i = 0; i < N_RELAY_S_1F; i++) {
//        double prob_max = 0.0;
//        double dist_min = INF_DOUBLE_S_1F;
//        for(int j = 0; j < N_DIREC_S_1F; j++) {
//            if(prob_max < prob_relay_rn_sn[i][j])
//                prob_max = prob_relay_rn_sn[i][j];
//        }
//        for(int j = 0; j < N_RELAY_S_1F; j++) {
//            if(dist_min > dist_rn_all[i][j])
//                dist_min = dist_rn_all[i][j];
//        }
//        if(dist_min <= 0.0)
//            dist_min = 1e-6;
//        prob_sum += prob_max / dist_min;
//    }
//    return (prob_sum / N_RELAY_S_1F);
//}

//static double QoS_delay()
//{
//    double max_mean_hops = 0.0;
//    for(int i = 0; i < N_DIREC_S_1F; i++) {
//        mean_hops_sn[i] = 0.0;
//    }
//    for(int i = 0; i < N_RELAY_S_1F; i++) {
//        mean_hops_rn[i] = 0.0;
//    }
//
//    for(int i = N_RELAY_S_1F - 1; i >= 0; i--) {
//        int ID = ID_RELAY[i];
//        mean_hops_rn[ID] += 1.0;
//        for(int j = 0; j < N_RELAY_S_1F; j++) {
//            if(rn_hop_table[ID][j]) {
//                mean_hops_rn[ID] += mean_hops_rn[j] / n_rn_rn_com[ID];
//            }
//        }
//    }
//
//    for(int i = 0; i < N_DIREC_S_1F; i++) {
//        double tmp_hops = 1.0;
//        for(int j = 0; j < N_RELAY_S_1F; j++) {
//            if(sn_hop_table[i][j]) {
//                tmp_hops += mean_hops_rn[j] / n_sn_rn_com[i];
//            }
//        }
//        mean_hops_sn[i] = tmp_hops;
//        if(tmp_hops > max_mean_hops) {
//            max_mean_hops = tmp_hops;
//        }
//    }
//
//    return (max_mean_hops / 11.0);
//}

static double Lifetime_rn()
{
    double energy_max = -1.0;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        double tmp_data_amount = 0.0;
        for(int j = 0; j < N_DIREC_S_1F; j++) {
            tmp_data_amount += prob_relay_rn_sn[i][j];
        }
        tmp_data_amount *= l0;
        energy_consumed_RELAY[i] = E_elec * tmp_data_amount;
        for(int j = 0; j <= N_RELAY_S_1F; j++) {
            double cur_dist;
            if(rn_hop_table[i][j]) {
                cur_dist = dist_rn_rn[i][j];
                if(cur_dist < d_th) {
                    energy_consumed_RELAY[i] += (tmp_data_amount / n_rn_rn_com[i] * E_elec +
                                                 tmp_data_amount / n_rn_rn_com[i] * e_fs *
                                                 cur_dist * cur_dist);
                } else {
                    energy_consumed_RELAY[i] += (tmp_data_amount / n_rn_rn_com[i] * E_elec +
                                                 tmp_data_amount / n_rn_rn_com[i] * e_mp *
                                                 cur_dist * cur_dist * cur_dist * cur_dist);
                }
            }
        }
        if(energy_consumed_RELAY[i] > energy_max) {
            energy_max = energy_consumed_RELAY[i];
        }
    }
    //
    double fit;
    double Con_E = 1.0 / 100;
    //
    if(energy_max <= 0.0)
        fit = 1e-6;
    else
        fit = energy_max / Con_E;
    //
    return fit;
}

static double Lifetime_rn_djrp()
{
    double energy_max = -1.0;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        energy_consumed_RELAY[i] = 0.0;
        for(int j = 0; j < N_DIREC_S_1F; j++) {
            double tmp_data_amount = 0.0;
            tmp_data_amount = djrp_prob_relay_rn_sn[j][i];
            tmp_data_amount *= l0;
            energy_consumed_RELAY[i] += E_elec * tmp_data_amount;
            for(int k = 0; k <= N_RELAY_S_1F; k++) {
                double cur_dist;
                if(djrp_rn_hop_table[j][i][k]) {
                    cur_dist = dist_rn_rn[i][k];
                    if(cur_dist < d_th) {
                        energy_consumed_RELAY[i] += (tmp_data_amount / djrp_n_rn_rn_com[j][i] * E_elec +
                                                     tmp_data_amount / djrp_n_rn_rn_com[j][i] * e_fs *
                                                     cur_dist * cur_dist);
                    } else {
                        energy_consumed_RELAY[i] += (tmp_data_amount / djrp_n_rn_rn_com[j][i] * E_elec +
                                                     tmp_data_amount / djrp_n_rn_rn_com[j][i] * e_mp *
                                                     cur_dist * cur_dist * cur_dist * cur_dist);
                    }
                }
            }
        }
        if(energy_consumed_RELAY[i] > energy_max) {
            energy_max = energy_consumed_RELAY[i];
        }
    }
    //
    double fit;
    double Con_E = 1.0 / 100;
    //
    if(energy_max <= 0.0)
        fit = 1e-6;
    else
        fit = energy_max / Con_E;
    //
    return fit;
}

//static double Lifetime_sn()
//{
//    double max_avg_dist_SENSOR_S_1F = 0.0;
//    for(int i = 0; i < N_DIREC_S_1F; i++) {
//        double meanD = 0.0;
//        for(int j = 0; j < N_RELAY_S_1F; j++) {
//            if(sn_hop_table[i][j]) {
//                meanD += dist_sn_rn[i][j];
//            }
//        }
//        meanD /= n_sn_rn_com[i];
//        if(meanD > max_avg_dist_SENSOR_S_1F) {
//            max_avg_dist_SENSOR_S_1F = meanD;
//        }
//    }
//
//    return (max_avg_dist_SENSOR_S_1F / d_th_sn);
//}

static void Constraints()
{
    //////////////////////////////////////////////////////////////////////////
    // link to sink
    // Sensor Nodes
    double tmp_prob_sum[N_DIREC_S_1F];
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        tmp_prob_sum[i] = 0.0;
    }
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        if(rn_hop_table[i][N_RELAY_S_1F]) {
            for(int j = 0; j < N_DIREC_S_1F; j++) {
                tmp_prob_sum[j] += prob_relay_rn_sn[i][j];
            }
        }
    }
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        if(tmp_prob_sum[i] <= 0.0) {
            total_penalty += penaltyVal;
        }
    }
    // Relay Nodes
    int tmp_link_flag[N_RELAY_S_1F];
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        if(rn_hop_table[i][N_RELAY_S_1F])
            tmp_link_flag[i] = 1;
        else
            tmp_link_flag[i] = 0;
    }
    for(int i = N_RELAY_S_1F - 1; i >= 0; i--) {
        int ID = ID_RELAY[i];
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(rn_hop_table[j][ID]) {
                tmp_link_flag[j] += tmp_link_flag[ID];
            }
        }
    }
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        if(!tmp_link_flag[i]) {
            total_penalty += penaltyVal;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // reliability
    int min_rn;
    // Sensor Nodes
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        min_rn = n_rn_min;
        if(n_sn_rn_com[i] < min_rn) {
            total_penalty += (min_rn - n_sn_rn_com[i]) * penaltyVal;
        }
    }
    // Relay Nodes
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        min_rn = n_rn_min;
        if(n_rn_rn_com[i] < min_rn &&
           !rn_hop_table[i][N_RELAY_S_1F]) {
            total_penalty += (min_rn - n_rn_rn_com[i]) * penaltyVal;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // dist
    int count_min = 0;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(dist_rn_all[i][j] == 0) {
                count_min++;
            }
        }
    }
    total_penalty += count_min / 2 * penaltyVal;
    //
    return;
}

static void Constraints_djrp()
{
    //////////////////////////////////////////////////////////////////////////
    // link to sink
    // Sensor Nodes
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        if(djrp_prob_relay_rn_sn[i][N_RELAY_S_1F] <= 0.0) {
            total_penalty += penaltyVal;
        }
    }
    // Relay Nodes
    double tmp_prob_sum[N_RELAY_S_1F];
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        tmp_prob_sum[i] = 0.0;
        for(int j = 0; j < N_DIREC_S_1F; j++) {
            tmp_prob_sum[i] += djrp_prob_relay_rn_sn[j][i];
        }
        if(tmp_prob_sum[i] <= 0.0) {
            total_penalty += penaltyVal;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // reliability
    int min_rn;
    // Sensor Nodes
    for(int i = 0; i < N_DIREC_S_1F; i++) {
        min_rn = n_rn_min;
        if(n_sn_rn_com[i] < min_rn) {
            total_penalty += (min_rn - n_sn_rn_com[i]) * penaltyVal;
        }
    }
    // Relay Nodes
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        min_rn = n_rn_min;
        if(n_rn_rn_com[i] < min_rn &&
           !rn_hop_table[i][N_RELAY_S_1F]) {
            total_penalty += (min_rn - n_rn_rn_com[i]) * penaltyVal;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // dist
    int count_min = 0;
    for(int i = 0; i < N_RELAY_S_1F; i++) {
        for(int j = 0; j < N_RELAY_S_1F; j++) {
            if(dist_rn_all[i][j] == 0) {
                count_min++;
            }
        }
    }
    total_penalty += count_min / 2 * penaltyVal;
    //
    return;
}