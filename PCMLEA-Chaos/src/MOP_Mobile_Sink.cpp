#include "MOP_Mobile_Sink.h"
#include <float.h>
#include <math.h>
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#include <mkl_lapacke.h>
#endif

//////////////////////////////////////////////////////////////////////////
//#define SAVE_FILE_ENERGY_STATUS
//#define SAVE_FILE_MOVING_PATH

//////////////////////////////////////////////////////////////////////////
#define FLAG_OFF_MOP_Mob_Sink 0
#define FLAG_ON_MOP_Mob_Sink 1
#define STATUS_OUT_INDICES_MOP_Mob_Sink FLAG_OFF_MOP_Mob_Sink

//#define OUTPUT_PREDICTION_GROUNDTRUTH_MOP_Mob_Sink
#define PRINT_ERROR_PARA_MOP_Mob_Sink

//////////////////////////////////////////////////////////////////////////
#define MAX_STR_LEN_MOP_Mob_Sink 1024
#define MAX_IN_NUM_MOP_Mob_Sink 1024
#define MAX_OUT_NUM_MOP_Mob_Sink 1024
#define VIOLATION_PENALTY_MOP_Mob_Sink 1e6
#define MAX_NUM_CANDID_SINK_SITES 1024

//////////////////////////////////////////////////////////////////////////
#define THRESHOLD_NUM_ROUGH_NODES_MOP_Mob_Sink 3
#define HOP_DIST_INIT (NUM_SENSOR_MOP_Mob_Sink + MAX_NUM_SINK_MOP_Mob_Sink + 1)

//////////////////////////////////////////////////////////////////////////
static double pos_sensor_MOP_Mob_Sink[NUM_SENSOR_MOP_Mob_Sink][3];

//////////////////////////////////////////////////////////////////////////
int NDIM_MOP_Mob_Sink = 0;
int NOBJ_MOP_Mob_Sink = 0;
char prob_name_MOP_Mob_Sink[128];

int  repNum_MOP_Mob_Sink;
int  repNo_MOP_Mob_Sink;

static MY_FLT_TYPE total_penalty_MOP_Mob_Sink = 0.0;
static MY_FLT_TYPE penaltyVal_MOP_Mob_Sink = 1e6;

frnn_MOP_Mob_Sink* frnn_mop_mobile_sink = NULL;

double fitness_max[MAX_OUT_NUM_MOP_Mob_Sink];

//////////////////////////////////////////////////////////////////////////
static void   ff_MOP_Mob_Sink_c(double* individual, double* results);
static double simplicity_MOP_Mob_Sink();
static double generality_MOP_Mob_Sink();

//////////////////////////////////////////////////////////////////////////
#define N_I_MAX_MOP_Mob_Sink 20
#define N_O_MAX_MOP_Mob_Sink 10

#define NUM_EXP_MAX_MOP_Mob_Sink 5

#define INF_DOUBLE_S_1F (9.9E299)
#define resIWSN_S_1F (1.0)
#define d_th_sn (25) // (40)

#define energy_ini (0.5)
#define E_elec (50.0e-9)
#define e_fs (10.0e-12)
#define e_mp (0.0013e-12)
#define d_th (87.7)
#define l0 (4000)
#define l0_fld (256)
#define l0_attr (256)

#define pi 3.141592653589793238462643383279

//
static int    sn_hop_table[MAX_NUM_SINK_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink + 1];
static int    sn_hop_min_dist[MAX_NUM_SINK_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink];
static int    sn_hop_min_dist_overall[NUM_SENSOR_MOP_Mob_Sink];
static double pos3D_SINK[MAX_NUM_SINK_MOP_Mob_Sink][3];
static double dist_sn_all[NUM_SENSOR_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink + MAX_NUM_SINK_MOP_Mob_Sink];
static double data_amount_sn[NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_tmp_sn[NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_1fld_sn[NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_1fld_sep_sn[MAX_NUM_SINK_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_1rn_sn[NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_1rn_sep_sn[MAX_NUM_SINK_MOP_Mob_Sink][NUM_SENSOR_MOP_Mob_Sink];
static double energy_consumed_sum_sn[NUM_SENSOR_MOP_Mob_Sink];
static int    flag_grid = 0;
static int    num_grid = 10;
static double len_grid_x = REGION_W_MOP_Mob_Sink / num_grid;
static double len_grid_y = REGION_L_MOP_Mob_Sink / num_grid;
static double d_MOP_Mob_Sink_min = 10;
static double d_MOP_Mob_Sink_max = 50;
int           NUM_SINK_MOP_Mob_Sink = 1;
static int    THrnMin = 5;
static int    THrnMax = 50;
static int    fAdaptTHrn = 0;
int           POS_TYPE_CUR_MOP_Mob_Sink;
static int    DepthR = 0;
static int    flagT2 = 1;

static int    flagTEST = 0;

//
static void   init_pos_sn_sink(int n, int flag_refesh);
static double range(double *pos1, double *pos2);
static void   Get_topology_sep(int id_sink);
static void   Get_topology_ad_hoc();
static void   Get_topology();
//static void Check_topology();
static double msg_frwrd_cmb(double* energy_vec, double* data_vec, int f_ini, int d_hop, int n_msg, double l_msg);
static double msg_frwrd_sep(double* energy_vec, double* data_vec, int f_ini, int id_sink, int d_hop, int n_msg, double l_msg);
static double msg_flood_sep(double* energy_vec, double* data_vec, int f_ini, int id_sink, int d_hop, int n_msg, double l_msg);
static double msg_flood_ad_hoc();
static double Lifetime_sn();
static void   Lifetime_info_init(double& energy_min, double& energy_max, double& energy_avg);
static void   energy_flood_ini();
static double Get_attributes_sep(MY_FLT_TYPE* valAttr, int id_sink);
static double energy_accum_refresh(int flag_tmp, int flag_fld, int* num_rounds, int id_sink_fld, int id_sink_rn);
static double Update_sink(int id_sink, MY_FLT_TYPE* valOut, int* flag_moved_all);
static double Update_sink_direct(int id_sink, MY_FLT_TYPE* new_pos, int* flag_moved_all);
static void   RandomMovement(MY_FLT_TYPE* valOut);
static double trans_to_0_1(double val);
static int    Get_n_fwr(double val);
static double msg_one_sn(double* energy_vec, double* data_vec, int id_sn, int n_msg, double l_msg);
static void   GreedyMaximumResidualEnergy_sep_continuous(MY_FLT_TYPE* new_pos, int id_sink);
static void   GreedyMaximumResidualEnergy_sep(MY_FLT_TYPE* new_pos, int id_sink);
static void   GreedyMaximumResidualEnergy(MY_FLT_TYPE* new_pos, int id_sink);
static void   adjust_sink_2_grid(int id);

//
static int    seed_MOP_Mob_Sink = 237;
static long   rnd_uni_init_MOP_Mob_Sink = -(long)seed_MOP_Mob_Sink;
static double rnd_uni_MOP_Mob_Sink(long* idum);
static int    rnd_MOP_Mob_Sink(int low, int high);
static double rndreal_MOP_Mob_Sink(double low, double high);
static double gaussrand_MOP_Mob_Sink(double a = 0.0, double b = 1.0);
static void   trimLine_MOP_Mob_Sink(char line[]);
static int    get_setting_MOP_Mob_Sink(char* wholestr, const char* candidstr, int& val);

//////////////////////////////////////////////////////////////////////////
void Initialize_MOP_Mob_Sink(char* pro, int curN, int numN, int my_rank)
{
    //
    sprintf(prob_name_MOP_Mob_Sink, "%s", pro);
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    seed_MOP_Mob_Sink = 237;// +my_rank;
    seed_MOP_Mob_Sink = seed_MOP_Mob_Sink % 1235;
    rnd_uni_init_MOP_Mob_Sink = -(long)seed_MOP_Mob_Sink;
    for(int i = 0; i < curN; i++) {
        seed_MOP_Mob_Sink = (seed_MOP_Mob_Sink + 111) % 1235;
        rnd_uni_init_MOP_Mob_Sink = -(long)seed_MOP_Mob_Sink;
    }
    //
    repNo_MOP_Mob_Sink = curN;
    repNum_MOP_Mob_Sink = numN;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    d_MOP_Mob_Sink_max = 50;
    int tmp_d = 0;
    if(get_setting_MOP_Mob_Sink(pro, "dMove", tmp_d))
        d_MOP_Mob_Sink_max = tmp_d;
    if(d_MOP_Mob_Sink_min >= d_MOP_Mob_Sink_max) d_MOP_Mob_Sink_min = d_MOP_Mob_Sink_max / 3;
    flag_grid = get_setting_MOP_Mob_Sink(pro, "GRID", num_grid);
    if(flag_grid) {
        if(num_grid <= 0) {
            printf("\n%s(%d): Invalid value of num_grid -- %d, exiting...\n",
                   __FILE__, __LINE__, num_grid);
            exit(-9769127);
        }
        len_grid_x = REGION_W_MOP_Mob_Sink / num_grid;
        len_grid_y = REGION_L_MOP_Mob_Sink / num_grid;
        if(d_MOP_Mob_Sink_max < len_grid_x) d_MOP_Mob_Sink_max = len_grid_x;
        if(d_MOP_Mob_Sink_max < len_grid_y) d_MOP_Mob_Sink_max = len_grid_y;
        if(d_MOP_Mob_Sink_min > len_grid_x / 3) d_MOP_Mob_Sink_min = len_grid_x / 3;
        if(d_MOP_Mob_Sink_min > len_grid_y / 3) d_MOP_Mob_Sink_min = len_grid_y / 3;
    }
    NUM_SINK_MOP_Mob_Sink = 1;
    get_setting_MOP_Mob_Sink(pro, "NumSINK", NUM_SINK_MOP_Mob_Sink);
    //if(strstr(pro, "GMRE") && !flag_grid) {
    //    printf("\n%s(%d): The GMRE algorithm requires the GRID setting, exiting...\n",
    //           __FILE__, __LINE__);
    //    exit(-9764745);
    //}
    if(NUM_SINK_MOP_Mob_Sink > MAX_NUM_SINK_MOP_Mob_Sink || NUM_SINK_MOP_Mob_Sink <= 0) {
        printf("\n%s(%d): The number of sinks (%d) should within [1, %d], exiting...\n",
               __FILE__, __LINE__, NUM_SINK_MOP_Mob_Sink, MAX_NUM_SINK_MOP_Mob_Sink);
        exit(-976858);
    }
    fAdaptTHrn = 0;
    get_setting_MOP_Mob_Sink(pro, "fAdaptTHrn", fAdaptTHrn);
    if(fAdaptTHrn && strstr(pro, "GMRE")) {
        printf("\n%s(%d): GMRE and fAdaptTHrn=1 cannot coexist, exiting...\n",
               __FILE__, __LINE__);
        exit(-9766458);
    }
    POS_TYPE_CUR_MOP_Mob_Sink = POS_TYPE_UNIF_MOP_Mob_Sink;
    get_setting_MOP_Mob_Sink(pro, "PosType", POS_TYPE_CUR_MOP_Mob_Sink);
    if(POS_TYPE_CUR_MOP_Mob_Sink == POS_TYPE_UNIF_MOP_Mob_Sink) {
        //printf("POS_TYPE_UNIF_MOP_Mob_Sink\n");
    } else if(POS_TYPE_CUR_MOP_Mob_Sink == POS_TYPE_GAUSSIAN_MOP_Mob_Sink) {
        //printf("POS_TYPE_GAUSSIAN_MOP_Mob_Sink\n");
    } else if(POS_TYPE_CUR_MOP_Mob_Sink == POS_TYPE_HYBRID_MOP_Mob_Sink) {
        //printf("POS_TYPE_HYBRID_MOP_Mob_Sink\n");
    } else {
        printf("\n%s(%d): PosType setting wrong, which should be 0 (UNIF), 1 (GAUSS) or 2 (HYBRID), exiting...\n",
               __FILE__, __LINE__);
        exit(-9768758);
    }
    if(strstr(pro, "STATIC") && fAdaptTHrn) {
        printf("\n%s(%d): STATIC and fAdaptTHrn=1 cannot coexist, exiting...\n",
               __FILE__, __LINE__);
        exit(-9766418);
    }
    DepthR = 0;
    get_setting_MOP_Mob_Sink(pro, "DepthR", DepthR);
    flagT2 = 1;
    get_setting_MOP_Mob_Sink(pro, "flagT2", flagT2);
    flagTEST = 0;
    //
    fitness_max[0] = 300 * NUM_SINK_MOP_Mob_Sink;
    //
    frnn_mop_mobile_sink = (frnn_MOP_Mob_Sink*)calloc(1, sizeof(frnn_MOP_Mob_Sink));
    frnn_mop_mobile_sink->GEP0 = NULL;
    frnn_mop_mobile_sink->M1 = NULL;
    frnn_mop_mobile_sink->F2 = NULL;
    frnn_mop_mobile_sink->R3 = NULL;
    frnn_mop_mobile_sink->M4 = NULL;
    frnn_mop_mobile_sink->R5 = NULL;
    frnn_mop_mobile_sink->OL = NULL;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_sum_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1fld_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1rn_sn[i] = 0;

    if(strstr(pro, "GEP_only")) {
        frnn_mop_mobile_sink->tag_GEP = FLAG_STATUS_ON;
        frnn_mop_mobile_sink->tag_DIF = FLAG_STATUS_OFF;
        frnn_mop_mobile_sink->tag_GEPr = FLAG_STATUS_OFF;
        frnn_mop_mobile_sink->tag_multiKindInput = FLAG_STATUS_OFF;

        frnn_MOP_Mob_Sink_setup_GEP_only(frnn_mop_mobile_sink);
        //
        NDIM_MOP_Mob_Sink = frnn_mop_mobile_sink->numParaLocal;
        NOBJ_MOP_Mob_Sink = 3;
    } else if(strstr(pro, "FRNN")) {
        frnn_mop_mobile_sink->tag_GEP = FLAG_STATUS_OFF;
        frnn_mop_mobile_sink->tag_DIF = FLAG_STATUS_OFF;
        frnn_mop_mobile_sink->tag_GEPr = FLAG_STATUS_OFF;
        frnn_mop_mobile_sink->tag_multiKindInput = FLAG_STATUS_OFF;

        frnn_MOP_Mob_Sink_setup(frnn_mop_mobile_sink);
        //
        NDIM_MOP_Mob_Sink = frnn_mop_mobile_sink->numParaLocal;
        NOBJ_MOP_Mob_Sink = 3;
    } else if(strstr(pro, "RM") ||
              strstr(pro, "GMRE") ||
              strstr(pro, "STATIC")) {
        //
        NDIM_MOP_Mob_Sink = 1;
        NOBJ_MOP_Mob_Sink = 2;
    } else {
        printf("\n%s(%d): Unknown problem name ~ %s, the dataset cannot be found, exiting...\n",
               __FILE__, __LINE__, pro);
        exit(-9127);
    }
#if (defined SAVE_FILE_ENERGY_STATUS) || (defined SAVE_FILE_MOVING_PATH)
    system("mkdir -p OUT_Mob_Sink");
#endif
    //
    return;
}

void SetLimits_MOP_Mob_Sink(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
        if(frnn_mop_mobile_sink->tag_GEP == FLAG_STATUS_ON) {
            for(int n = 0; n < frnn_mop_mobile_sink->num_GEP; n++) {
                for(int i = 0; i < frnn_mop_mobile_sink->GEP0[n]->numParaLocal; i++) {
                    minLimit[count] = frnn_mop_mobile_sink->GEP0[n]->xMin[i];
                    maxLimit[count] = frnn_mop_mobile_sink->GEP0[n]->xMax[i];
                    count++;
                }
            }
        }
    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        if(frnn_mop_mobile_sink->tag_GEP == FLAG_STATUS_ON) {
            for(int n = 0; n < frnn_mop_mobile_sink->num_GEP; n++) {
                for(int i = 0; i < frnn_mop_mobile_sink->GEP0[n]->numParaLocal; i++) {
                    minLimit[count] = frnn_mop_mobile_sink->GEP0[n]->xMin[i];
                    maxLimit[count] = frnn_mop_mobile_sink->GEP0[n]->xMax[i];
                    count++;
                }
            }
        }
        for(int i = 0; i < frnn_mop_mobile_sink->M1->numParaLocal; i++) {
            minLimit[count] = frnn_mop_mobile_sink->M1->xMin[i];
            maxLimit[count] = frnn_mop_mobile_sink->M1->xMax[i];
            count++;
        }
        for(int i = 0; i < frnn_mop_mobile_sink->F2->numParaLocal; i++) {
            minLimit[count] = frnn_mop_mobile_sink->F2->xMin[i];
            maxLimit[count] = frnn_mop_mobile_sink->F2->xMax[i];
            count++;
        }
        for(int i = 0; i < frnn_mop_mobile_sink->R3->numParaLocal; i++) {
            minLimit[count] = frnn_mop_mobile_sink->R3->xMin[i];
            maxLimit[count] = frnn_mop_mobile_sink->R3->xMax[i];
            count++;
        }
        if(DepthR) {
            if(DepthR == 2)
                for(int i = 0; i < frnn_mop_mobile_sink->M4->numParaLocal; i++) {
                    minLimit[count] = frnn_mop_mobile_sink->M4->xMin[i];
                    maxLimit[count] = frnn_mop_mobile_sink->M4->xMax[i];
                    count++;
                }
            for(int i = 0; i < frnn_mop_mobile_sink->R5->numParaLocal; i++) {
                minLimit[count] = frnn_mop_mobile_sink->R5->xMin[i];
                maxLimit[count] = frnn_mop_mobile_sink->R5->xMax[i];
                count++;
            }
        }
        for(int i = 0; i < frnn_mop_mobile_sink->OL->numParaLocal; i++) {
            minLimit[count] = frnn_mop_mobile_sink->OL->xMin[i];
            maxLimit[count] = frnn_mop_mobile_sink->OL->xMax[i];
            count++;
        }
    } else {
        for(int i = 0; i < nx; i++) {
            minLimit[i] = 0;
            maxLimit[i] = 1;
        }
    }
    //
    return;
}

int CheckLimits_MOP_Mob_Sink(double* x, int nx)
{
    int count = 0;
    //
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
        if(frnn_mop_mobile_sink->tag_GEP == FLAG_STATUS_ON) {
            for(int n = 0; n < frnn_mop_mobile_sink->num_GEP; n++) {
                for(int i = 0; i < frnn_mop_mobile_sink->GEP0[n]->numParaLocal; i++) {
                    if(x[count] < frnn_mop_mobile_sink->GEP0[n]->xMin[i] ||
                       x[count] > frnn_mop_mobile_sink->GEP0[n]->xMax[i]) {
                        printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->GEP0[%d] %d, %.16e not in [%.16e, %.16e]\n",
                               __FILE__, __LINE__, n, i, x[count], frnn_mop_mobile_sink->GEP0[n]->xMin[i],
                               frnn_mop_mobile_sink->GEP0[n]->xMax[i]);
                        return 0;
                    }
                    count++;
                }
            }
        }
    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        if(frnn_mop_mobile_sink->tag_GEP == FLAG_STATUS_ON) {
            for(int n = 0; n < frnn_mop_mobile_sink->num_GEP; n++) {
                for(int i = 0; i < frnn_mop_mobile_sink->GEP0[n]->numParaLocal; i++) {
                    if(x[count] < frnn_mop_mobile_sink->GEP0[n]->xMin[i] ||
                       x[count] > frnn_mop_mobile_sink->GEP0[n]->xMax[i]) {
                        printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->GEP0[%d] %d, %.16e not in [%.16e, %.16e]\n",
                               __FILE__, __LINE__, n, i, x[count], frnn_mop_mobile_sink->GEP0[n]->xMin[i], frnn_mop_mobile_sink->GEP0[n]->xMax[i]);
                        return 0;
                    }
                    count++;
                }
            }
        }
        for(int i = 0; i < frnn_mop_mobile_sink->M1->numParaLocal; i++) {
            if(x[count] < frnn_mop_mobile_sink->M1->xMin[i] ||
               x[count] > frnn_mop_mobile_sink->M1->xMax[i]) {
                printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->M1 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->M1->xMin[i], frnn_mop_mobile_sink->M1->xMax[i]);
                return 0;
            }
            count++;
        }
        for(int i = 0; i < frnn_mop_mobile_sink->F2->numParaLocal; i++) {
            if(x[count] < frnn_mop_mobile_sink->F2->xMin[i] ||
               x[count] > frnn_mop_mobile_sink->F2->xMax[i]) {
                printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->F2 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->F2->xMin[i], frnn_mop_mobile_sink->F2->xMax[i]);
                return 0;
            }
            count++;
        }
        for(int i = 0; i < frnn_mop_mobile_sink->R3->numParaLocal; i++) {
            if(x[count] < frnn_mop_mobile_sink->R3->xMin[i] ||
               x[count] > frnn_mop_mobile_sink->R3->xMax[i]) {
                printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->R3 %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->R3->xMin[i], frnn_mop_mobile_sink->R3->xMax[i]);
                return 0;
            }
            count++;
        }
        if(DepthR) {
            if(DepthR == 2)
                for(int i = 0; i < frnn_mop_mobile_sink->M4->numParaLocal; i++) {
                    if(x[count] < frnn_mop_mobile_sink->M4->xMin[i] ||
                       x[count] > frnn_mop_mobile_sink->M4->xMax[i]) {
                        printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->M4 %d, %.16e not in [%.16e, %.16e]\n",
                               __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->M4->xMin[i], frnn_mop_mobile_sink->M4->xMax[i]);
                        return 0;
                    }
                    count++;
                }
            for(int i = 0; i < frnn_mop_mobile_sink->R5->numParaLocal; i++) {
                if(x[count] < frnn_mop_mobile_sink->R5->xMin[i] ||
                   x[count] > frnn_mop_mobile_sink->R5->xMax[i]) {
                    printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->R5 %d, %.16e not in [%.16e, %.16e]\n",
                           __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->R5->xMin[i], frnn_mop_mobile_sink->R5->xMax[i]);
                    return 0;
                }
                count++;
            }
        }
#ifndef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
        for(int i = 0; i < frnn_mop_mobile_sink->OL->numParaLocal; i++) {
            if(x[count] < frnn_mop_mobile_sink->OL->xMin[i] ||
               x[count] > frnn_mop_mobile_sink->OL->xMax[i]) {
                printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->OL %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->OL->xMin[i], frnn_mop_mobile_sink->OL->xMax[i]);
                return 0;
            }
            count++;
        }
#else
        if(frnn_mop_mobile_sink->flagConnectStatus != FLAG_STATUS_OFF ||
           frnn_mop_mobile_sink->flagConnectWeight != FLAG_STATUS_ON ||
           frnn_mop_mobile_sink->typeCoding != PARA_CODING_DIRECT) {
            printf("%s(%d): Parameter setting error of flagConnectStatus (%d) or flagConnectWeight (%d) or typeCoding (%d) with UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY, exiting...\n",
                   __FILE__, __LINE__, frnn_mop_mobile_sink->flagConnectStatus, frnn_mop_mobile_sink->flagConnectWeight,
                   frnn_mop_mobile_sink->typeCoding);
            exit(-275082);
        }
        int tmp_offset = frnn_mop_mobile_sink->OL->numOutput * frnn_mop_mobile_sink->OL->numInput;
        count += tmp_offset;
        for(int i = tmp_offset; i < frnn_mop_mobile_sink->OL->numParaLocal; i++) {
            if(x[count] < frnn_mop_mobile_sink->OL->xMin[i] ||
               x[count] > frnn_mop_mobile_sink->OL->xMax[i]) {
                printf("%s(%d): Check limits FAIL - frnn_mop_mobile_sink: frnn_mop_mobile_sink->OL %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[count], frnn_mop_mobile_sink->OL->xMin[i], frnn_mop_mobile_sink->OL->xMax[i]);
                return 0;
            }
            count++;
        }
#endif
    }
    //
    return 1;
}

void Fitness_MOP_Mob_Sink(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_sum_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1fld_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1rn_sn[i] = 0;
    total_penalty_MOP_Mob_Sink = 0.0;
    //
    flagTEST = 0;
    double results[3];
    ff_MOP_Mob_Sink_c(individual, results);
    double f_simp = simplicity_MOP_Mob_Sink();
    //
    fitness[0] = -results[0] + total_penalty_MOP_Mob_Sink;
    fitness[1] = results[1] + total_penalty_MOP_Mob_Sink;
    //fitness[2] = results[2] + total_penalty_MOP_Mob_Sink;
    //
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only") ||
       strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        fitness[2] = f_simp + total_penalty_MOP_Mob_Sink;
    }
    //
    //fitness[2] = generality_MOP_Mob_Sink() + total_penalty_MOP_Mob_Sink;
    //
    return;
}

void Fitness_MOP_Mob_Sink_test(double* individual, double* fitness)
{
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_sum_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1fld_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1rn_sn[i] = 0;
    total_penalty_MOP_Mob_Sink = 0.0;
    //
    flagTEST = 1;
    double results[3];
    ff_MOP_Mob_Sink_c(individual, results);
    double f_simp = simplicity_MOP_Mob_Sink();
    //
    fitness[0] = -results[0] + total_penalty_MOP_Mob_Sink;
    fitness[1] = results[1] + total_penalty_MOP_Mob_Sink;
    //fitness[2] = results[2] + total_penalty_MOP_Mob_Sink;
    //
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only") ||
       strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        fitness[2] = f_simp + total_penalty_MOP_Mob_Sink;
    }
    //
    //fitness[2] = generality_MOP_Mob_Sink() + total_penalty_MOP_Mob_Sink;
    //
    return;
}

static void ff_MOP_Mob_Sink_c(double* individual, double* results)
{
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
        frnn_MOP_Mob_Sink_init_GEP_only(frnn_mop_mobile_sink, individual, ASSIGN_MODE_FRNN);
    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        frnn_MOP_Mob_Sink_init(frnn_mop_mobile_sink, individual, ASSIGN_MODE_FRNN);
    }
    //
    MY_FLT_TYPE valIn[MAX_NUM_SINK_MOP_Mob_Sink][N_I_MAX_MOP_Mob_Sink];
    MY_FLT_TYPE valOut[MAX_NUM_SINK_MOP_Mob_Sink][N_O_MAX_MOP_Mob_Sink];
    double avg_n_round = 0;
    double avg_dist = 0;
    double avg_moved = 0;
    for(int i_exp = 0; i_exp < NUM_EXP_MAX_MOP_Mob_Sink; i_exp++) {
#ifdef SAVE_FILE_ENERGY_STATUS
        FILE* fpt_energy = NULL;
        char fnm_energy[MAX_STR_LEN_MOP_Mob_Sink];
        sprintf(fnm_energy, "OUT_Mob_Sink/%s_energy_status_run%02d_Exp%d",
                prob_name_MOP_Mob_Sink, repNo_MOP_Mob_Sink + 1, i_exp + 1);
        fpt_energy = fopen(fnm_energy, "w");
#endif
#ifdef SAVE_FILE_MOVING_PATH
        FILE* fpt_moving_path[MAX_NUM_SINK_MOP_Mob_Sink];
        for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
            char fnm_moving_path[MAX_STR_LEN_MOP_Mob_Sink];
            sprintf(fnm_moving_path, "OUT_Mob_Sink/%s_moving_path_status_run%02d_Exp%d_Sink%d",
                    prob_name_MOP_Mob_Sink, repNo_MOP_Mob_Sink, i_exp + 1, n);
            fpt_moving_path[n] = fopen(fnm_moving_path, "w");
        }
#endif
        init_pos_sn_sink(i_exp + 1, i_exp == 0);
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_sum_sn[i] = 0;
        double energy_max = 0;
        int n_round = 0;
        int flag_moved = NUM_SINK_MOP_Mob_Sink;
        int flag_moved_all[MAX_NUM_SINK_MOP_Mob_Sink];
        for(int n = 0; n < MAX_NUM_SINK_MOP_Mob_Sink; n++) flag_moved_all[n] = 1;
        int n_fwr_all[MAX_NUM_SINK_MOP_Mob_Sink];
        for(int n = 0; n < MAX_NUM_SINK_MOP_Mob_Sink; n++) n_fwr_all[n] = THrnMin;
        int count_moved = 0;
        double dist_moved = 0.0;
        double d_m_l = 0.0;
        do {
            for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
            for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                if(flag_moved_all[n]) Get_topology_sep(n);
                if(flag_moved_all[n]) msg_flood_sep(energy_consumed_1fld_sep_sn[n], data_amount_sn, 1, n, 0, 1, l0_fld);
                if(flag_moved_all[n]) energy_accum_refresh(0, 0, 0, n, -1);
                if(flag_moved_all[n]) msg_frwrd_sep(energy_consumed_1rn_sep_sn[n], data_amount_sn, 1, n, 0, 1, l0);
                flag_moved_all[n] = 0;
            }
            if(flag_moved) Get_topology_ad_hoc();
            if(flag_moved) msg_flood_ad_hoc();
            if(flag_moved) msg_frwrd_cmb(energy_consumed_1rn_sn, data_amount_sn, 1, 0, 1, l0);
            //Check_topology();
            int tmp_rn = THrnMin;
            if(fAdaptTHrn) {
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                    if(n == 0 || tmp_rn > n_fwr_all[n])
                        tmp_rn = n_fwr_all[n];
                }
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) n_fwr_all[n] -= tmp_rn;
            }
            energy_max = energy_accum_refresh(0, 0, &tmp_rn, -1, -1);
            n_round += tmp_rn;
            if(energy_max > energy_ini) break;
            d_m_l = 0;
            flag_moved = 0;
            if(strstr(prob_name_MOP_Mob_Sink, "GEP_only") ||
               strstr(prob_name_MOP_Mob_Sink, "FRNN") ||
               strstr(prob_name_MOP_Mob_Sink, "RM")) {
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                    if(fAdaptTHrn && n_fwr_all[n] > 0) continue;
                    energy_accum_refresh(0, 0, 0, n, -1);
                    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
                        Get_attributes_sep(valIn[n], n);
                        ff_MOP_Mob_Sink_FRNN_GRP_only(frnn_mop_mobile_sink, valIn[n], valOut[n], NULL);
                    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
                        Get_attributes_sep(valIn[n], n);
                        ff_MOP_Mob_Sink_FRNN(frnn_mop_mobile_sink, valIn[n], valOut[n], NULL);
                    } else if(strstr(prob_name_MOP_Mob_Sink, "RM")) {
                        RandomMovement(valOut[n]);
                    }
                    d_m_l += Update_sink(n, valOut[n], flag_moved_all);
                    flag_moved += flag_moved_all[n];
                    if(fAdaptTHrn) n_fwr_all[n] = Get_n_fwr(valOut[n][3]);
                }
            } else if(strstr(prob_name_MOP_Mob_Sink, "GMRE")) {
                MY_FLT_TYPE new_pos[MAX_NUM_SINK_MOP_Mob_Sink][N_O_MAX_MOP_Mob_Sink];
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                    GreedyMaximumResidualEnergy(new_pos[n], n);
                    d_m_l += Update_sink_direct(n, new_pos[n], flag_moved_all);
                    flag_moved += flag_moved_all[n];
                }
            }
            dist_moved += d_m_l;
            count_moved += flag_moved;
            energy_max = energy_accum_refresh(0, 0, 0, -1, -1);
#ifdef SAVE_FILE_MOVING_PATH
            for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                if(flag_moved_all[n]) {
                    if(fAdaptTHrn)
                        fprintf(fpt_moving_path[n], "%d %lf %lf %d\n",
                                n_round, pos3D_SINK[n][IND_X_MOP_Mob_Sink], pos3D_SINK[n][IND_Y_MOP_Mob_Sink], n_fwr_all[n]);
                    else
                        fprintf(fpt_moving_path[n], "%d %lf %lf %d\n",
                                n_round, pos3D_SINK[n][IND_X_MOP_Mob_Sink], pos3D_SINK[n][IND_Y_MOP_Mob_Sink], THrnMin);
                }
            }
#endif
        } while(energy_max <= energy_ini);
        avg_n_round += n_round;
        avg_moved += count_moved;
        avg_dist += dist_moved;
#ifdef SAVE_FILE_ENERGY_STATUS
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
            fprintf(fpt_energy, "%lf\n", energy_consumed_sum_sn[i]);
        fclose(fpt_energy);
#endif
#ifdef SAVE_FILE_MOVING_PATH
        for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
            fclose(fpt_moving_path[n]);
        }
#endif
    }
    avg_n_round /= NUM_EXP_MAX_MOP_Mob_Sink;
    avg_moved /= NUM_EXP_MAX_MOP_Mob_Sink;
    avg_dist /= NUM_EXP_MAX_MOP_Mob_Sink;
    //
    results[0] = avg_n_round;
    results[1] = avg_dist;
    results[2] = avg_moved;
    //
    return;
}

void Finalize_MOP_Mob_Sink()
{
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
        frnn_MOP_Mob_Sink_free_GEP_only(frnn_mop_mobile_sink);
    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        frnn_MOP_Mob_Sink_free(frnn_mop_mobile_sink);
    }
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
void frnn_MOP_Mob_Sink_setup_GEP_only(frnn_MOP_Mob_Sink* frnn)
{
    frnn->num_multiKindInput = 1;
    frnn->numInput = 12;
    frnn->num_multiKindOutput = 1;
    int numOutput = 3;
    if(fAdaptTHrn) {
        numOutput = 4;
    }
    frnn->numOutput = numOutput;
    //
    int GEP_head_len = 8;
    frnn->GEP_head_len = GEP_head_len;
    //
    frnn->layerNum = 8;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;
    //
    MY_FLT_TYPE energy_min, energy_max, energy_avg;
    Lifetime_info_init(energy_min, energy_max, energy_avg);
    MY_FLT_TYPE inputMin[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1 };
    MY_FLT_TYPE inputMax[] = {
        energy_ini, /*energy_ini / energy_avg*/10, energy_ini, /*energy_max*/0.1,
        energy_ini, /*energy_ini / energy_avg*/10, energy_ini, /*energy_max*/0.1,
        1, 1, 1, 1
    };
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        frnn->num_GEP = frnn->numOutput * frnn->num_multiKindOutput;
        frnn->GEP0 = (codingGEP**)malloc(frnn->num_GEP * sizeof(codingGEP*));
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->GEP0[n] = setupCodingGEP(frnn->numInput, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                           frnn->GEP_head_len,
                                           FLAG_STATUS_OFF,
                                           PARA_MIN_VAL_GEP_CFRNN_MODEL,
                                           PARA_MAX_VAL_GEP_CFRNN_MODEL);
        }
    }
    //
    frnn->numParaLocal = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal +=
                frnn->GEP0[n]->numParaLocal;
        }
    }
    //
    frnn->numParaLocal_disc = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal_disc +=
                frnn->GEP0[n]->numParaLocal_disc;
        }
    }
    frnn->layerNum = 1;
    //
    frnn->xType = (int*)malloc(frnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            memcpy(&frnn->xType[tmp_cnt_p], frnn->GEP0[n]->xType, frnn->GEP0[n]->numParaLocal * sizeof(int));
            tmp_cnt_p += frnn->GEP0[n]->numParaLocal;
        }
    }
    //
    return;
}

void frnn_MOP_Mob_Sink_setup(frnn_MOP_Mob_Sink* frnn)
{
    frnn->num_multiKindInput = 12;
    frnn->numInput = 1;
    frnn->num_multiKindOutput = 3;
    if(fAdaptTHrn) {
        frnn->num_multiKindOutput = 4;
    }
    int numOutput = 1;
    frnn->numOutput = numOutput;
    //
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    if(!flagT2) typeFuzzySet = FUZZY_SET_I;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE; // MUL_EACH_IN_TO_ONE_RULE; //
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;// FIXED_ROUGH_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
    int flagConnectStatus = FLAG_STATUS_OFF;
    int flagConnectWeight = FLAG_STATUS_OFF;
    if(frnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    if(frnn->num_multiKindOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
        //flagConnectWeight = FLAG_STATUS_ON;
    }
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    //
    frnn->typeFuzzySet = typeFuzzySet;
    frnn->typeRules = typeRules;
    frnn->typeInRuleCorNum = typeInRuleCorNum;
    frnn->typeTypeReducer = typeTypeReducer;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
    frnn->centroid_num_tag = centroid_num_tag;
    frnn->flagConnectStatus = flagConnectStatus;
    frnn->flagConnectWeight = flagConnectWeight;
    //
#if MF_RULE_NUM_MOP_Mob_Sink_CUR == MF_RULE_NUM_MOP_Mob_Sink_LESS
    int numFuzzyRules = 20;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#else
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = 10;// (int)sqrt(numFuzzyRules);
#endif
    if(numFuzzyRules > DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL)
        numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    //numRoughSets = 2 * frnn->numOutput * frnn->num_multiKindOutput;
    //numRoughSets = numFuzzyRules / 2;
    //if(numRoughSets < 3)
    numRoughSets = 10;
    //
    int GEP_head_len = 8;
    frnn->GEP_head_len = GEP_head_len;
    //
    frnn->layerNum = 8;
    frnn->numRules = numFuzzyRules;
    frnn->numRoughs = numRoughSets;
    //
    int tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;
    //
    int numInputConsequenceNode = 0;
#if FRNN_CONSEQUENCE_MOP_Mob_Sink_CUR == FRNN_CONSEQUENCE_MOP_Mob_Sink_FIXED
    numInputConsequenceNode = 0;
    consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
#elif FRNN_CONSEQUENCE_MOP_Mob_Sink_CUR == FRNN_CONSEQUENCE_MOP_Mob_Sink_ADAPT
    numInputConsequenceNode = frnn->numInput * frnn->num_multiKindInput;
    consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
#endif
    //
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
    flagConnectStatus = FLAG_STATUS_OFF;
    frnn->flagConnectStatus = flagConnectStatus;
    flagConnectWeight = FLAG_STATUS_ON;
    frnn->flagConnectWeight = flagConnectWeight;
    tmp_typeCoding = PARA_CODING_DIRECT;
    frnn->typeCoding = tmp_typeCoding;
    //numInputConsequenceNode = 0;
    //consequenceNodeStatus = FIXED_CONSEQUENCE_CENTROID;
    //frnn->consequenceNodeStatus = consequenceNodeStatus;
    centroid_num_tag = CENTROID_ALL_ONESET;
    if(frnn->num_multiKindOutput > 1 || frnn->numOutput > 1) {
        centroid_num_tag = CENTROID_ONESET_EACH;
    }
    frnn->centroid_num_tag = centroid_num_tag;
    if(numRoughSets < numOutput) {
        numRoughSets = numOutput;
        frnn->numRoughs = numRoughSets;
    }
#endif
    //
    MY_FLT_TYPE energy_min, energy_max, energy_avg;
    Lifetime_info_init(energy_min, energy_max, energy_avg);
    MY_FLT_TYPE inputMin[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1 };
    MY_FLT_TYPE inputMax[] = {
        energy_ini, /*energy_ini / energy_avg / 10*/1, energy_ini, /*energy_max * 10*/1,
        energy_ini, /*energy_ini / energy_avg / 10*/1, energy_ini, /*energy_max * 10*/1,
        1, 1, 1, 1
    };
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        frnn->num_GEP = frnn->numInput * frnn->num_multiKindInput;
        frnn->GEP0 = (codingGEP**)malloc(frnn->num_GEP * sizeof(codingGEP*));
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->GEP0[n] = setupCodingGEP(frnn->numInput * frnn->num_multiKindInput, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                           frnn->GEP_head_len,
                                           FLAG_STATUS_OFF,
                                           PARA_MIN_VAL_GEP_CFRNN_MODEL,
                                           PARA_MAX_VAL_GEP_CFRNN_MODEL);
        }
        for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
            inputMin[i] = -10;
            inputMax[i] = 10;
        }
    }
    if(frnn->tag_DIF == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_multiKindInput; n++) {
            int tmp_ind_os = n * frnn->numInput;
            inputMin[tmp_ind_os + 0] = -10;
            inputMax[tmp_ind_os + 0] = 10;
            for(int i = 1; i < frnn->numInput; i++) {
                inputMin[tmp_ind_os + i] = -10;
                inputMax[tmp_ind_os + i] = 10;
            }
        }
    }
    int* numMemship = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
#if MF_RULE_NUM_MOP_Mob_Sink_CUR == MF_RULE_NUM_MOP_Mob_Sink_LESS
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#else
        numMemship[i] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
#endif
    }
    int* flagAdapMemship = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        flagAdapMemship[i] = FLAG_STATUS_ON;
    }
    frnn->M1 = setupMemberLayer(frnn->numInput * frnn->num_multiKindInput, inputMin, inputMax,
                                numMemship, flagAdapMemship, frnn->typeFuzzySet,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    frnn->F2 = setupFuzzyLayer(frnn->numInput * frnn->num_multiKindInput, frnn->M1->numMembershipFun, frnn->numRules,
                               frnn->typeFuzzySet, frnn->typeRules,
                               frnn->typeInRuleCorNum, frnn->tag_GEPr,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, FLAG_STATUS_OFF);
    frnn->R3 = setupRoughLayer(frnn->numRules, frnn->numRoughs, frnn->typeFuzzySet,
                               FLAG_STATUS_ON,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    if(DepthR) {
        if(DepthR == 2) {
            MY_FLT_TYPE* inputMin2 = (MY_FLT_TYPE*)malloc(frnn->R3->numRoughSets * sizeof(MY_FLT_TYPE));
            MY_FLT_TYPE* inputMax2 = (MY_FLT_TYPE*)malloc(frnn->R3->numRoughSets * sizeof(MY_FLT_TYPE));
            int* numMemship2 = (int*)malloc(frnn->R3->numRoughSets * sizeof(int));
            int* flagAdapMemship2 = (int*)malloc(frnn->R3->numRoughSets * sizeof(int));
            for(int i = 0; i < frnn->R3->numRoughSets; i++) inputMin2[i] = 0;
            for(int i = 0; i < frnn->R3->numRoughSets; i++) inputMax2[i] = 1;
            for(int i = 0; i < frnn->R3->numRoughSets; i++) numMemship2[i] = 1;
            for(int i = 0; i < frnn->R3->numRoughSets; i++) flagAdapMemship2[i] = 1;
            frnn->M4 = setupMemberLayer(frnn->R3->numRoughSets, inputMin2, inputMax2,
                                        numMemship2, flagAdapMemship2, FUZZY_SET_I,
                                        tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
            free(inputMin2);
            free(inputMax2);
            free(numMemship2);
            free(flagAdapMemship2);
        }
        frnn->R5 = setupRoughLayer(frnn->R3->numRoughSets, frnn->numRoughs, frnn->typeFuzzySet,
                                   FLAG_STATUS_ON,
                                   tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    }
    MY_FLT_TYPE outputMin[MAX_OUT_NUM_MOP_Mob_Sink];
    MY_FLT_TYPE outputMax[MAX_OUT_NUM_MOP_Mob_Sink];
    for(int i = 0; i < frnn->numOutput * frnn->num_multiKindOutput; i++) {
        outputMin[i] = -1;
        outputMax[i] = 1;
    }
    MY_FLT_TYPE inputMin_cnsq[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1 };
    MY_FLT_TYPE inputMax_cnsq[] = {
        energy_ini, /*energy_ini / energy_avg / 10*/1, energy_ini, /*energy_max * 10*/1,
        energy_ini, /*energy_ini / energy_avg / 10*/1, energy_ini, /*energy_max * 10*/1,
        1, 1, 1, 1
    };
    frnn->OL = setupOutReduceLayer(frnn->numRoughs, frnn->numOutput * frnn->num_multiKindOutput, outputMin, outputMax,
                                   frnn->typeFuzzySet, frnn->typeTypeReducer,
                                   frnn->consequenceNodeStatus, frnn->centroid_num_tag,
                                   numInputConsequenceNode, inputMin_cnsq, inputMax_cnsq,
                                   frnn->flagConnectStatus, frnn->flagConnectWeight,
                                   tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, frnn->GEP_head_len, 1);
    //
    free(numMemship);
    free(flagAdapMemship);
    //
    frnn->numParaLocal = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal +=
                frnn->GEP0[n]->numParaLocal;
        }
    }
    frnn->numParaLocal += frnn->M1->numParaLocal;
    frnn->numParaLocal += frnn->F2->numParaLocal;
    frnn->numParaLocal += frnn->R3->numParaLocal;
    if(DepthR) {
        if(DepthR == 2)
            frnn->numParaLocal += frnn->M4->numParaLocal;
        frnn->numParaLocal += frnn->R5->numParaLocal;
    }
    frnn->numParaLocal += frnn->OL->numParaLocal;
    //
    frnn->numParaLocal_disc = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            frnn->numParaLocal_disc +=
                frnn->GEP0[n]->numParaLocal_disc;
        }
    }
    frnn->numParaLocal_disc += frnn->M1->numParaLocal_disc;
    frnn->numParaLocal_disc += frnn->F2->numParaLocal_disc;
    frnn->numParaLocal_disc += frnn->R3->numParaLocal_disc;
    if(DepthR) {
        if(DepthR == 2)
            frnn->numParaLocal_disc += frnn->M4->numParaLocal_disc;
        frnn->numParaLocal_disc += frnn->R5->numParaLocal_disc;
    }
    frnn->numParaLocal_disc += frnn->OL->numParaLocal_disc;
    frnn->layerNum = 4;
    //
    frnn->xType = (int*)malloc(frnn->numParaLocal * sizeof(int));
    int tmp_cnt_p = 0;
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            memcpy(&frnn->xType[tmp_cnt_p], frnn->GEP0[n]->xType, frnn->GEP0[n]->numParaLocal * sizeof(int));
            tmp_cnt_p += frnn->GEP0[n]->numParaLocal;
        }
    }
    memcpy(&frnn->xType[tmp_cnt_p], frnn->M1->xType, frnn->M1->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->M1->numParaLocal;
    memcpy(&frnn->xType[tmp_cnt_p], frnn->F2->xType, frnn->F2->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->F2->numParaLocal;
    memcpy(&frnn->xType[tmp_cnt_p], frnn->R3->xType, frnn->R3->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->R3->numParaLocal;
    if(DepthR) {
        if(DepthR == 2) {
            memcpy(&frnn->xType[tmp_cnt_p], frnn->M4->xType, frnn->M4->numParaLocal * sizeof(int));
            tmp_cnt_p += frnn->M4->numParaLocal;
        }
        memcpy(&frnn->xType[tmp_cnt_p], frnn->R5->xType, frnn->R5->numParaLocal * sizeof(int));
        tmp_cnt_p += frnn->R5->numParaLocal;
    }
    memcpy(&frnn->xType[tmp_cnt_p], frnn->OL->xType, frnn->OL->numParaLocal * sizeof(int));
    tmp_cnt_p += frnn->OL->numParaLocal;
    //tmp_cnt_p = 0;
    //for(int i = 0; i < cfrnn->numParaLocal; i++) {
    //    if(cfrnn->xType[i] != VAR_TYPE_CONTINUOUS)
    //        tmp_cnt_p++;
    //}
    //printf("%d ~ %d \n", tmp_cnt_p, cfrnn->numParaLocal_disc);

    frnn->e = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_sum = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_wrong = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->e_sum = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_TP = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_TN = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FP = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FN = (MY_FLT_TYPE*)calloc(frnn->OL->numOutput, sizeof(MY_FLT_TYPE));

    frnn->featureMapTagInitial = (int*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(int));
    frnn->dataflowInitial = (MY_FLT_TYPE*)calloc(frnn->numInput * frnn->num_multiKindInput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->numInput * frnn->num_multiKindInput; i++) {
        frnn->featureMapTagInitial[i] = 1;
        frnn->dataflowInitial[i] = 1;
    }

    if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->numInput * frnn->num_multiKindInput * frnn->numRules * frnn->numRoughs *
                                          frnn->numOutput * frnn->num_multiKindOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->numInput * frnn->num_multiKindInput * frnn->numRules +
                                            frnn->numRules * frnn->numRoughs);
    } else {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->numRules * frnn->numRoughs * frnn->numOutput *
                                          frnn->num_multiKindOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->numRules +
                                            frnn->numRules * frnn->numRoughs);
    }
    //
    return;
}

void frnn_MOP_Mob_Sink_free_GEP_only(frnn_MOP_Mob_Sink* frnn)
{
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int i = 0; i < frnn->num_GEP; i++) {
            freeCodingGEP(frnn->GEP0[i]);
        }
        free(frnn->GEP0);
    }

    free(frnn->xType);

    free(frnn);

    return;
}

void frnn_MOP_Mob_Sink_free(frnn_MOP_Mob_Sink* frnn)
{
    freeOutReduceLayer(frnn->OL);
    if(DepthR) freeRoughLayer(frnn->R5);
    if(DepthR == 2) freeMemberLayer(frnn->M4);
    freeRoughLayer(frnn->R3);
    freeFuzzyLayer(frnn->F2);
    freeMemberLayer(frnn->M1);
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int i = 0; i < frnn->num_GEP; i++) {
            freeCodingGEP(frnn->GEP0[i]);
        }
        free(frnn->GEP0);
    }

    free(frnn->xType);

    free(frnn->e);

    free(frnn->N_sum);
    free(frnn->N_wrong);
    free(frnn->e_sum);

    free(frnn->N_TP);
    free(frnn->N_TN);
    free(frnn->N_FP);
    free(frnn->N_FN);

    free(frnn->featureMapTagInitial);
    free(frnn->dataflowInitial);

    free(frnn);

    return;
}

void frnn_MOP_Mob_Sink_init_GEP_only(frnn_MOP_Mob_Sink* frnn, double* x, int mode)
{
    int count = 0;
    switch(mode) {
    case INIT_MODE_FRNN:
    case ASSIGN_MODE_FRNN:
    case OUTPUT_ALL_MODE_FRNN:
    case OUTPUT_CONTINUOUS_MODE_FRNN:
    case OUTPUT_DISCRETE_MODE_FRNN:
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            assignCodingGEP(frnn->GEP0[n], &x[count], mode);
            count += frnn->GEP0[n]->numParaLocal;
        }
    }
    //
    return;
}

void frnn_MOP_Mob_Sink_init(frnn_MOP_Mob_Sink* frnn, double* x, int mode)
{
    int count = 0;
    switch(mode) {
    case INIT_MODE_FRNN:
    case ASSIGN_MODE_FRNN:
    case OUTPUT_ALL_MODE_FRNN:
    case OUTPUT_CONTINUOUS_MODE_FRNN:
    case OUTPUT_DISCRETE_MODE_FRNN:
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            assignCodingGEP(frnn->GEP0[n], &x[count], mode);
            count += frnn->GEP0[n]->numParaLocal;
        }
    }
    assignMemberLayer(frnn->M1, &x[count], mode);
    count += frnn->M1->numParaLocal;
    assignFuzzyLayer(frnn->F2, &x[count], mode);
    count += frnn->F2->numParaLocal;
    assignRoughLayer(frnn->R3, &x[count], mode);
    count += frnn->R3->numParaLocal;
    if(DepthR) {
        if(DepthR == 2) {
            assignMemberLayer(frnn->M4, &x[count], mode);
            count += frnn->M4->numParaLocal;
        }
        assignRoughLayer(frnn->R5, &x[count], mode);
        count += frnn->R5->numParaLocal;
    }
    assignOutReduceLayer(frnn->OL, &x[count], mode);
    count += frnn->OL->numParaLocal;
    //
    return;
}

void ff_MOP_Mob_Sink_FRNN_GRP_only(frnn_MOP_Mob_Sink* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                                   MY_FLT_TYPE** inputConsequenceNode)
{
    int len_valIn = frnn->numInput * frnn->num_multiKindInput;
    MY_FLT_TYPE* tmpIn = (MY_FLT_TYPE*)malloc(len_valIn * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmpOut = (MY_FLT_TYPE*)malloc(frnn->num_GEP * sizeof(MY_FLT_TYPE));
    memcpy(tmpIn, valIn, frnn->numInput * frnn->num_multiKindInput * sizeof(MY_FLT_TYPE));
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            int tmp_ind = n / frnn->numInput;
            decodingGEP(frnn->GEP0[n], tmpIn, &tmpOut[n]);
            //printf("%lf ", tmpOut[n]);
        }
    }
    memcpy(valOut, tmpOut, frnn->num_GEP * sizeof(MY_FLT_TYPE));
    //
    free(tmpIn);
    free(tmpOut);
    //
    return;
}

void ff_MOP_Mob_Sink_FRNN(frnn_MOP_Mob_Sink* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut,
                          MY_FLT_TYPE** inputConsequenceNode)
{
    int len_valIn = frnn->numInput * frnn->num_multiKindInput;
    MY_FLT_TYPE* tmpIn = (MY_FLT_TYPE*)malloc(len_valIn * sizeof(MY_FLT_TYPE));
    MY_FLT_TYPE* tmpOut = (MY_FLT_TYPE*)malloc(frnn->num_GEP * sizeof(MY_FLT_TYPE));
    memcpy(tmpIn, valIn, frnn->numInput * frnn->num_multiKindInput * sizeof(MY_FLT_TYPE));
    if(frnn->tag_GEP == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_GEP; n++) {
            int tmp_ind = n / frnn->numInput;
            decodingGEP(frnn->GEP0[n], tmpIn, &tmpOut[n]);
            //printf("%lf ", tmpOut[n]);
        }
        ff_memberLayer(frnn->M1, tmpOut, frnn->dataflowInitial);
    } else if(frnn->tag_DIF == FLAG_STATUS_ON) {
        for(int n = 0; n < frnn->num_multiKindInput; n++) {
            int tmp_ind_os = n * frnn->numInput;
            for(int i = 1; i < frnn->numInput; i++) {
                tmpIn[tmp_ind_os + i] = valIn[tmp_ind_os + i - 1] - valIn[tmp_ind_os + i];
            }
        }
        ff_memberLayer(frnn->M1, tmpIn, frnn->dataflowInitial);
    } else {
        ff_memberLayer(frnn->M1, valIn, frnn->dataflowInitial);
    }
    free(tmpIn);
    free(tmpOut);
    //
    ff_fuzzyLayer(frnn->F2, frnn->M1->degreeMembership, frnn->M1->dataflowStatus);
    ff_roughLayer(frnn->R3, frnn->F2->degreeRules, frnn->F2->dataflowStatus);
    if(DepthR) {
        if(DepthR == 2) {
            MY_FLT_TYPE** degreeIn = (MY_FLT_TYPE**)malloc(frnn->R3->dim_degree * sizeof(MY_FLT_TYPE*));
            for(int i = 0; i < frnn->R3->dim_degree; i++) {
                degreeIn[i] = (MY_FLT_TYPE*)malloc(frnn->R3->numRoughSets * sizeof(MY_FLT_TYPE));
                for(int j = 0; j < frnn->R3->numRoughSets; j++)
                    degreeIn[i][j] = frnn->R3->degreeRough[j][i];
            }
            MY_FLT_TYPE** degreeOut = (MY_FLT_TYPE**)malloc(frnn->R3->numRoughSets * sizeof(MY_FLT_TYPE*));
            for(int i = 0; i < frnn->R3->numRoughSets; i++) {
                degreeOut[i] = (MY_FLT_TYPE*)malloc(frnn->R3->dim_degree * sizeof(MY_FLT_TYPE));
            }
            for(int i = 0; i < frnn->R3->dim_degree; i++) {
                ff_memberLayer(frnn->M4, degreeIn[i], frnn->R3->dataflowStatus);
                for(int j = 0; j < frnn->R3->numRoughSets; j++) {
                    degreeOut[j][i] = frnn->M4->degreeMembership[j][0][0];
                }
            }
            if(frnn->R3->dim_degree == 2) {
                for(int i = 0; i < frnn->M4->numInput; i++) {
                    if(degreeOut[i][0] > degreeOut[i][1]) {
                        MY_FLT_TYPE temp = degreeOut[i][0];
                        degreeOut[i][0] = degreeOut[i][1];
                        degreeOut[i][1] = temp;
                    }
                }
            }
            ff_roughLayer(frnn->R5, degreeOut, frnn->R3->dataflowStatus);
            for(int i = 0; i < frnn->R3->dim_degree; i++) free(degreeIn[i]);
            free(degreeIn);
            for(int i = 0; i < frnn->M4->numInput; i++) free(degreeOut[i]);
            free(degreeOut);
        } else {
            ff_roughLayer(frnn->R5, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
        }
    }
    //
#if FRNN_CONSEQUENCE_MOP_Mob_Sink_CUR == FRNN_CONSEQUENCE_MOP_Mob_Sink_ADAPT
    if(frnn->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int n = 0; n < frnn->num_multiKindOutput; n++) {
            for(int i = 0; i < frnn->numOutput; i++) {
                memcpy(frnn->OL->inputConsequenceNode[n * frnn->numOutput + i],
                       valIn,
                       frnn->OL->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
            }
        }
    }
#endif
    if(DepthR)
        ff_outReduceLayer(frnn->OL, frnn->R5->degreeRough, frnn->R5->dataflowStatus);
    else
        ff_outReduceLayer(frnn->OL, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
    //for(int i = 0; i < frnn->OL->numOutput; i++) {
    //    if(CHECK_INVALID(frnn->OL->valOutputFinal[i])) {
    //        printf("%s(%d): Invalid output %d ~ %lf, exiting...\n",
    //               __FILE__, __LINE__, i, frnn->OL->valOutputFinal[i]);
    //        print_para_memberLayer(frnn->M1);
    //        print_data_memberLayer(frnn->M1);
    //        print_para_fuzzyLayer(frnn->F2);
    //        print_data_fuzzyLayer(frnn->F2);
    //        print_para_roughLayer(frnn->R3);
    //        print_data_roughLayer(frnn->R3);
    //        print_para_outReduceLayer(frnn->OL);
    //        print_data_outReduceLayer(frnn->OL);
    //        exit(-94628);
    //    }
    //}
    //
    memcpy(valOut, frnn->OL->valOutputFinal, frnn->OL->numOutput * sizeof(MY_FLT_TYPE));
    //
    return;
}

static double simplicity_MOP_Mob_Sink()
{
    //
    double f_simpl = 0.0;
    if(strstr(prob_name_MOP_Mob_Sink, "GEP_only")) {
        double f_simpl_gl = 0.0;
        //
        if(frnn_mop_mobile_sink->tag_GEP) {
            for(int i = 0; i < frnn_mop_mobile_sink->num_GEP; i++) {
                int tmp_g = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->GEP0[i]->check_head; j++) {
                    if(frnn_mop_mobile_sink->GEP0[i]->check_op[j] >= 0) {
                        tmp_g++;
                    }
                }
                f_simpl_gl += (double)tmp_g / frnn_mop_mobile_sink->GEP0[i]->GEP_head_length;
            }
            f_simpl = f_simpl_gl / frnn_mop_mobile_sink->num_GEP;
        }
    } else if(strstr(prob_name_MOP_Mob_Sink, "FRNN")) {
        double f_simpl_gl = 0.0;
        double f_simpl_fl = 0.0;
        double f_simpl_rl = 0.0;
        double f_simpl_rl2 = 0.0;
        double f_simpl_ol = 0.0;
        //
        int *tmp_rule, *tmp_rough, *tmp_rough2, *tmp_out, **tmp_mem;
        tmp_rule = (int*)calloc(frnn_mop_mobile_sink->F2->numRules, sizeof(int));
        tmp_rough = (int*)calloc(frnn_mop_mobile_sink->R3->numRoughSets, sizeof(int));
        tmp_rough2 = (int*)calloc(frnn_mop_mobile_sink->R3->numRoughSets, sizeof(int));
        tmp_out = (int*)calloc(frnn_mop_mobile_sink->OL->numOutput, sizeof(int));
        tmp_mem = (int**)malloc(frnn_mop_mobile_sink->M1->numInput * sizeof(int*));
        for(int i = 0; i < frnn_mop_mobile_sink->M1->numInput; i++) {
            tmp_mem[i] = (int*)calloc(frnn_mop_mobile_sink->M1->numMembershipFun[i], sizeof(int));
        }

        if(frnn_mop_mobile_sink->tag_GEP) {
            for(int i = 0; i < frnn_mop_mobile_sink->num_GEP; i++) {
                int tmp_g = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->GEP0[i]->check_head; j++) {
                    if(frnn_mop_mobile_sink->GEP0[i]->check_op[j] >= 0) {
                        tmp_g++;
                    }
                }
                f_simpl_gl += (double)tmp_g / frnn_mop_mobile_sink->GEP0[i]->GEP_head_length;
            }
        }
        if(frnn_mop_mobile_sink->tag_GEPr == FLAG_STATUS_OFF) {
            for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
                tmp_rule[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->M1->numInput; j++) {
                    int tmp_count = 0;
                    for(int k = 0; k < frnn_mop_mobile_sink->M1->numMembershipFun[j]; k++) {
                        if(frnn_mop_mobile_sink->F2->connectStatusAll[i][j][k]) {
                            tmp_count++;
                            tmp_mem[j][k]++;
                        }
                    }
                    if(tmp_count) {
                        tmp_rule[i]++;
                    }
                }
                f_simpl_fl += (double)tmp_rule[i] / frnn_mop_mobile_sink->M1->numInput;
            }
            for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
                tmp_rough[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->F2->numRules; j++) {
                    if(tmp_rule[j] && frnn_mop_mobile_sink->R3->connectStatus[i][j]) {
                        tmp_rough[i]++;
                    }
                }
                f_simpl_rl += (double)tmp_rough[i] / frnn_mop_mobile_sink->F2->numRules;
            }
        } else {
            for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
                for(int j = 0; j < frnn_mop_mobile_sink->M1->numInput; j++) {
                    for(int k = 0; k < frnn_mop_mobile_sink->M1->numMembershipFun[j]; k++) {
                        if(frnn_mop_mobile_sink->F2->connectStatusAll[i][j][k]) {
                            tmp_mem[j][k]++;
                        }
                    }
                }
                //
                tmp_rule[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->F2->ruleGEP[i]->check_head; j++) {
                    if(frnn_mop_mobile_sink->F2->ruleGEP[i]->check_op[j] >= 0) {
                        tmp_rule[i]++;
                    }
                }
                f_simpl_fl += (double)tmp_rule[i] / frnn_mop_mobile_sink->F2->ruleGEP[i]->GEP_head_length;
            }
            //
            for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
                tmp_rough[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->F2->numRules; j++) {
                    if(frnn_mop_mobile_sink->R3->connectStatus[i][j]) {
                        tmp_rough[i]++;
                    }
                }
                f_simpl_rl += (double)tmp_rough[i] / frnn_mop_mobile_sink->F2->numRules;
            }
        }
        if(DepthR) {
            for(int i = 0; i < frnn_mop_mobile_sink->R5->numRoughSets; i++) {
                tmp_rough2[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->R5->numInput; j++) {
                    if(tmp_rough[j] && frnn_mop_mobile_sink->R5->connectStatus[i][j]) {
                        tmp_rough2[i]++;
                    }
                }
                f_simpl_rl2 += (double)tmp_rough2[i] / frnn_mop_mobile_sink->R5->numInput;
            }
        }
        if(frnn_mop_mobile_sink->OL->flagConnectStatusAdap) {
            for(int i = 0; i < frnn_mop_mobile_sink->OL->numOutput; i++) {
                tmp_out[i] = 0;
                for(int j = 0; j < frnn_mop_mobile_sink->OL->numInput; j++) {
                    if((!DepthR && tmp_rough[j]) || (DepthR && tmp_rough2[j]) &&
                       frnn_mop_mobile_sink->OL->connectStatus[i][j]) {
                        tmp_out[i]++;
                    }
                }
                f_simpl_ol += (double)tmp_out[i] / frnn_mop_mobile_sink->OL->numInput;
            }
        }
        //if(!DepthR) {
        //    if(frnn_mop_mobile_sink->tag_GEP)
        //        f_simpl = (f_simpl_gl + f_simpl_fl + f_simpl_rl) /
        //                  (frnn_mop_mobile_sink->num_GEP + frnn_mop_mobile_sink->F2->numRules + frnn_mop_mobile_sink->R3->numRoughSets);
        //    else
        //        f_simpl = (f_simpl_fl + f_simpl_rl) /
        //                  (frnn_mop_mobile_sink->F2->numRules + frnn_mop_mobile_sink->R3->numRoughSets);
        //} else {
        //    if(frnn_mop_mobile_sink->tag_GEP)
        //        f_simpl = (f_simpl_gl + f_simpl_fl + f_simpl_rl + f_simpl_rl2) /
        //                  (frnn_mop_mobile_sink->num_GEP + frnn_mop_mobile_sink->F2->numRules + frnn_mop_mobile_sink->R3->numRoughSets +
        //                   frnn_mop_mobile_sink->R5->numRoughSets);
        //    else
        //        f_simpl = (f_simpl_fl + f_simpl_rl + f_simpl_rl2) /
        //                  (frnn_mop_mobile_sink->F2->numRules + frnn_mop_mobile_sink->R3->numRoughSets +
        //                   frnn_mop_mobile_sink->R5->numRoughSets);
        //}
        f_simpl = f_simpl_fl + f_simpl_rl;
        double tmp_fenmu = frnn_mop_mobile_sink->F2->numRules + frnn_mop_mobile_sink->R3->numRoughSets;
        if(frnn_mop_mobile_sink->tag_GEP) {
            f_simpl += f_simpl_gl;
            tmp_fenmu += frnn_mop_mobile_sink->num_GEP;
        }
        if(DepthR) {
            f_simpl += f_simpl_rl2;
            tmp_fenmu += frnn_mop_mobile_sink->R5->numRoughSets;
        }
        if(frnn_mop_mobile_sink->OL->flagConnectStatusAdap) {
            f_simpl += f_simpl_ol;
            tmp_fenmu += frnn_mop_mobile_sink->OL->numOutput;
        }
        f_simpl /= tmp_fenmu;

        //
        //if (flag_no_fuzzy_rule) {
        //  f_prcsn += 1e6;
        //  f_simpl += 1e6;
        //  f_normp += 1e6;
        //}
        int tmp_sum = 0;
        for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
            tmp_sum += tmp_rule[i];
        }
        if(tmp_sum == 0) {
            total_penalty_MOP_Mob_Sink += penaltyVal_MOP_Mob_Sink;
        }
        //tmp_sum = 0;
        //for(int i = 0; i < NUM_CLASS_MOP_Predict_FRNN; i++) {
        //    tmp_sum += tmp2[i];
        //}
        //if(tmp_sum == 0.0) {
        //    f_prcsn += 1e6;
        //    f_simpl += 1e6;
        //}
        tmp_sum = 0;
        for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
            if(tmp_rough[i])
                tmp_sum++;
            //if(tmp_rough[i] == 0)
            //    total_penalty_MOP_Predict_FRNN += penaltyVal_MOP_Predict_FRNN;
        }
        if(tmp_sum < THRESHOLD_NUM_ROUGH_NODES_MOP_Mob_Sink) {
            total_penalty_MOP_Mob_Sink += penaltyVal_MOP_Mob_Sink * (THRESHOLD_NUM_ROUGH_NODES_MOP_Mob_Sink - tmp_sum);
        }
        if(DepthR) {
            tmp_sum = 0;
            for(int i = 0; i < frnn_mop_mobile_sink->R5->numRoughSets; i++) {
                if(tmp_rough2[i])
                    tmp_sum++;
                //if(tmp_rough[i] == 0)
                //    total_penalty_MOP_Predict_FRNN += penaltyVal_MOP_Predict_FRNN;
            }
            if(tmp_sum < THRESHOLD_NUM_ROUGH_NODES_MOP_Mob_Sink) {
                total_penalty_MOP_Mob_Sink += penaltyVal_MOP_Mob_Sink * (THRESHOLD_NUM_ROUGH_NODES_MOP_Mob_Sink - tmp_sum);
            }
        }
        if(frnn_mop_mobile_sink->OL->flagConnectStatusAdap) {
            for(int i = 0; i < frnn_mop_mobile_sink->OL->numOutput; i++) {
                if(tmp_out[i] < 2) {
                    total_penalty_MOP_Mob_Sink += penaltyVal_MOP_Mob_Sink * (2 - tmp_out[i]);
                }
            }
        }
        //
        free(tmp_rule);
        free(tmp_rough);
        free(tmp_rough2);
        free(tmp_out);
        for(int i = 0; i < frnn_mop_mobile_sink->M1->numInput; i++) {
            free(tmp_mem[i]);
        }
        free(tmp_mem);
    }
    //
    return f_simpl;
}

static double generality_MOP_Mob_Sink()
{
    double tmp_sum = 0.0;
    int tmp_cnt = 0;
    //
    if(frnn_mop_mobile_sink->OL->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < frnn_mop_mobile_sink->OL->numOutput; i++) {
            if(frnn_mop_mobile_sink->OL->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                continue;
            }
            for(int j = 0; j < frnn_mop_mobile_sink->OL->numInput; j++) {
                for(int k = 0; k < frnn_mop_mobile_sink->OL->dim_degree; k++) {
                    if(frnn_mop_mobile_sink->OL->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                        continue;
                    }
                    for(int m = 0; m < frnn_mop_mobile_sink->OL->numInputConsequenceNode; m++) {     // without bias
                        tmp_sum += fabs(frnn_mop_mobile_sink->OL->paraConsequenceNode[i][j][k][m]);
                        tmp_cnt++;
                    }
                }
            }
        }
        //printf("Tag 1\n");
    }
    if(frnn_mop_mobile_sink->OL->flagConnectWeightAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < frnn_mop_mobile_sink->OL->numOutput; i++) {
            for(int j = 0; j < frnn_mop_mobile_sink->OL->numInput; j++) {
                tmp_sum += fabs(frnn_mop_mobile_sink->OL->connectWeight[i][j]);
                tmp_cnt++;
            }
        }
        //printf("Tag 2\n");
    }
    if(tmp_cnt)
        tmp_sum /= tmp_cnt;
    //
    return tmp_sum;
}

void statistics_MOP_Mob_Sink()
{
    //
    print_para_memberLayer(frnn_mop_mobile_sink->M1);
    print_data_memberLayer(frnn_mop_mobile_sink->M1);
    print_para_fuzzyLayer(frnn_mop_mobile_sink->F2);
    print_data_fuzzyLayer(frnn_mop_mobile_sink->F2);
    print_para_roughLayer(frnn_mop_mobile_sink->R3);
    print_data_roughLayer(frnn_mop_mobile_sink->R3);
    print_para_outReduceLayer(frnn_mop_mobile_sink->OL);
    print_data_outReduceLayer(frnn_mop_mobile_sink->OL);
    //
    int *tmp_rule, **tmp_mem;
    tmp_rule = (int*)calloc(frnn_mop_mobile_sink->F2->numRules, sizeof(int));
    tmp_mem = (int**)malloc(frnn_mop_mobile_sink->M1->numInput * sizeof(int*));
    for(int i = 0; i < frnn_mop_mobile_sink->M1->numInput; i++) {
        tmp_mem[i] = (int*)calloc(frnn_mop_mobile_sink->M1->numMembershipFun[i], sizeof(int));
    }
    int *tmp_rough, *tmp_rough_op, *tmp_rough_in;
    tmp_rough = (int*)calloc(frnn_mop_mobile_sink->R3->numRoughSets, sizeof(int));
    tmp_rough_op = (int*)calloc(frnn_mop_mobile_sink->R3->numRoughSets, sizeof(int));
    tmp_rough_in = (int*)calloc(frnn_mop_mobile_sink->R3->numRoughSets, sizeof(int));
    int *tmp_rule_op, *tmp_rule_in;
    tmp_rule_op = (int*)calloc(frnn_mop_mobile_sink->F2->numRules, sizeof(int));
    tmp_rule_in = (int*)calloc(frnn_mop_mobile_sink->F2->numRules, sizeof(int));
    //
    for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_mop_mobile_sink->F2->numRules; j++) {
            if(frnn_mop_mobile_sink->R3->connectStatus[i][j]) {
                tmp_rule[j]++;
            }
        }
    }
    //
    if(frnn_mop_mobile_sink->tag_GEPr == FLAG_STATUS_OFF) {
        for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
            for(int j = 0; j < frnn_mop_mobile_sink->M1->numInput; j++) {
                for(int k = 0; k < frnn_mop_mobile_sink->M1->numMembershipFun[j]; k++) {
                    if(frnn_mop_mobile_sink->F2->connectStatusAll[i][j][k]) {
                        tmp_mem[j][k] += tmp_rule[i];
                        tmp_rule_in[i]++;
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
            for(int j = 0; j < frnn_mop_mobile_sink->F2->ruleGEP[i]->check_tail; j++) {
                if(frnn_mop_mobile_sink->F2->ruleGEP[i]->check_vInd[j] >= 0) {
                    tmp_rule_in[i]++;
                    int cur_in = frnn_mop_mobile_sink->F2->ruleGEP[i]->check_vInd[j];
                    for(int k = 0; k < frnn_mop_mobile_sink->M1->numMembershipFun[cur_in]; k++) {
                        if(frnn_mop_mobile_sink->F2->connectStatusAll[i][cur_in][k]) {
                            tmp_mem[cur_in][k] += tmp_rule[i];
                            break;
                        }
                    }
                }
                if(frnn_mop_mobile_sink->F2->ruleGEP[i]->check_op[j] >= 0) {
                    tmp_rule_op[i]++;
                }
            }
        }
    }
    //
    for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_mop_mobile_sink->F2->numRules; j++) {
            if(frnn_mop_mobile_sink->R3->connectStatus[i][j]) {
                tmp_rough[i]++;
                tmp_rough_op[i] += tmp_rule_op[j];
                tmp_rough_in[i] += tmp_rule_in[j];
            }
        }
    }
    //
    for(int j = 0; j < frnn_mop_mobile_sink->M1->numInput; j++) {
        for(int k = 0; k < frnn_mop_mobile_sink->M1->numMembershipFun[j]; k++) {
            printf("%d,", tmp_mem[j][k]);
        }
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
        printf("%d,", tmp_rule[i]);
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
        printf("%d,", tmp_rule_op[i]);
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->F2->numRules; i++) {
        printf("%d,", tmp_rule_in[i]);
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough[i]);
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough_op[i]);
    }
    printf("\n");
    for(int i = 0; i < frnn_mop_mobile_sink->R3->numRoughSets; i++) {
        printf("%d,", tmp_rough_in[i]);
    }
    printf("\n");
    //
    if(frnn_mop_mobile_sink->tag_GEP) {
    }
    //
    free(tmp_rule);
    for(int i = 0; i < frnn_mop_mobile_sink->M1->numInput; i++) {
        free(tmp_mem[i]);
    }
    free(tmp_mem);
    free(tmp_rough);
    free(tmp_rough_op);
    free(tmp_rough_in);
    //
    free(tmp_rule_op);
    free(tmp_rule_in);
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// WSN
static void init_pos_sn_sink(int n, int flag_refesh)
{
    if(flag_refesh) {
        seed_MOP_Mob_Sink = 237 + n; // +my_rank;
        seed_MOP_Mob_Sink = seed_MOP_Mob_Sink % 1235;
        rnd_uni_init_MOP_Mob_Sink = -(long)seed_MOP_Mob_Sink;
        int tmp_n = repNo_MOP_Mob_Sink;
        if(flagTEST) tmp_n += repNum_MOP_Mob_Sink;
        for(int i = 0; i < tmp_n; i++) {
            seed_MOP_Mob_Sink = (seed_MOP_Mob_Sink + 111) % 1235;
            rnd_uni_init_MOP_Mob_Sink = -(long)seed_MOP_Mob_Sink;
        }
    }
    //
    int flag_again = 0;
    do {
        flag_again = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
            if(POS_TYPE_CUR_MOP_Mob_Sink == POS_TYPE_UNIF_MOP_Mob_Sink) {
                pos_sensor_MOP_Mob_Sink[i][IND_X_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_W_MOP_Mob_Sink);
                pos_sensor_MOP_Mob_Sink[i][IND_Y_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_L_MOP_Mob_Sink);
            } else if(POS_TYPE_CUR_MOP_Mob_Sink == POS_TYPE_GAUSSIAN_MOP_Mob_Sink) {
                MY_FLT_TYPE tmp_rnd = 0.0;
                do {
                    tmp_rnd = gaussrand_MOP_Mob_Sink();
                } while(tmp_rnd < 0 || tmp_rnd > 1);
                pos_sensor_MOP_Mob_Sink[i][IND_X_MOP_Mob_Sink] = tmp_rnd * REGION_W_MOP_Mob_Sink;
                do {
                    tmp_rnd = gaussrand_MOP_Mob_Sink();
                } while(tmp_rnd < 0 || tmp_rnd > 1);
                pos_sensor_MOP_Mob_Sink[i][IND_Y_MOP_Mob_Sink] = tmp_rnd * REGION_L_MOP_Mob_Sink;
            } else {
                if(i < NUM_SENSOR_MOP_Mob_Sink / 2) {
                    pos_sensor_MOP_Mob_Sink[i][IND_X_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_W_MOP_Mob_Sink);
                    pos_sensor_MOP_Mob_Sink[i][IND_Y_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_L_MOP_Mob_Sink);
                } else {
                    MY_FLT_TYPE tmp_rnd = 0.0;
                    do {
                        tmp_rnd = gaussrand_MOP_Mob_Sink();
                    } while(tmp_rnd < 0 || tmp_rnd > 1);
                    pos_sensor_MOP_Mob_Sink[i][IND_X_MOP_Mob_Sink] = tmp_rnd * REGION_W_MOP_Mob_Sink;
                    do {
                        tmp_rnd = gaussrand_MOP_Mob_Sink();
                    } while(tmp_rnd < 0 || tmp_rnd > 1);
                    pos_sensor_MOP_Mob_Sink[i][IND_Y_MOP_Mob_Sink] = tmp_rnd * REGION_L_MOP_Mob_Sink;
                }
            }
            pos_sensor_MOP_Mob_Sink[i][IND_Z_MOP_Mob_Sink] = 0;
        }
        if(strstr(prob_name_MOP_Mob_Sink, "STATIC")) {
            if(NUM_SINK_MOP_Mob_Sink == 1) {
                int n = 0;
                pos3D_SINK[n][IND_X_MOP_Mob_Sink] = 0.5 * REGION_W_MOP_Mob_Sink;
                pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = 0.5 * REGION_L_MOP_Mob_Sink;
                pos3D_SINK[n][IND_Z_MOP_Mob_Sink] = 0;
                adjust_sink_2_grid(n);
            } else {
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                    pos3D_SINK[n][IND_X_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_W_MOP_Mob_Sink);
                    pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_L_MOP_Mob_Sink);
                    pos3D_SINK[n][IND_Z_MOP_Mob_Sink] = 0;
                    adjust_sink_2_grid(n);
                }
            }
        } else {
            for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
                pos3D_SINK[n][IND_X_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_W_MOP_Mob_Sink);
                pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = rndreal_MOP_Mob_Sink(0, REGION_L_MOP_Mob_Sink);
                pos3D_SINK[n][IND_Z_MOP_Mob_Sink] = 0;
                adjust_sink_2_grid(n);
            }
        }
        //
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
            for(int j = i + 1; j < NUM_SENSOR_MOP_Mob_Sink; j++) {
                dist_sn_all[i][j] = range(pos_sensor_MOP_Mob_Sink[i], pos_sensor_MOP_Mob_Sink[j]) * resIWSN_S_1F;
                dist_sn_all[j][i] = dist_sn_all[i][j];
            }
            dist_sn_all[i][i] = 0;
            double tmp_min;
            for(int j = 0; j < NUM_SENSOR_MOP_Mob_Sink; j++) {
                if(j == 0 || tmp_min > dist_sn_all[i][j])
                    tmp_min = dist_sn_all[i][j];
            }
            if(tmp_min > d_th_sn) {
                flag_again++;
                printf("%s(%d): Sensor %d (%d) is isolated, regeneratring sensor positions, or increase the sensor number...\n",
                       __FILE__, __LINE__, i, flag_again);
            }
        }
    } while(flag_again);
    //
    return;
}

static double range(double *pos1, double *pos2)
{
    double r = 0;
    for(int i = 0; i < 3; i++) r += (pos1[i] - pos2[i]) * (pos1[i] - pos2[i]);
    return sqrt(r);
}

static void Get_topology_sep(int id_sink)
{
    int n = id_sink;
    // initialization
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        for(int j = 0; j <= NUM_SENSOR_MOP_Mob_Sink; j++) {
            sn_hop_table[n][i][j] = HOP_DIST_INIT;
        }
    }
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
        sn_hop_min_dist[n][i] = HOP_DIST_INIT;
    //
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        dist_sn_all[i][NUM_SENSOR_MOP_Mob_Sink + n] = range(pos3D_SINK[n], pos_sensor_MOP_Mob_Sink[i]) * resIWSN_S_1F;
        if(dist_sn_all[i][NUM_SENSOR_MOP_Mob_Sink + n] <= d_th_sn) {
            sn_hop_table[n][i][NUM_SENSOR_MOP_Mob_Sink] = 1;
            sn_hop_min_dist[n][i] = 1;
        }
    }
    int cur_hop_dist = 1;
    int flag_chn = 1;
    while(flag_chn) {
        flag_chn = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
            if(sn_hop_min_dist[n][i] != cur_hop_dist) continue;
            for(int j = 0; j < NUM_SENSOR_MOP_Mob_Sink; j++) {
                if(sn_hop_min_dist[n][i] - sn_hop_min_dist[n][j] <= -1 &&
                   dist_sn_all[i][j] <= d_th_sn) {
                    sn_hop_table[n][j][i] = sn_hop_min_dist[n][i] + 1;
                    sn_hop_min_dist[n][j] = sn_hop_min_dist[n][i] + 1;
                    flag_chn++;
                }
            }
        }
        cur_hop_dist++;
    }
    //
    return;
}

static void Get_topology_ad_hoc()
{
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        int tmp_dist = HOP_DIST_INIT;
        int tmp_id = -1;
        for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
            if(sn_hop_min_dist[n][i] != HOP_DIST_INIT &&
               (n == 0 || tmp_dist > sn_hop_min_dist[n][i])) {
                tmp_dist = sn_hop_min_dist[n][i];
                tmp_id = n;
            }
        }
        if(tmp_id == -1) {
            printf("%s(%d): Sensor %d is disconnected, exiting...\n",
                   __FILE__, __LINE__, i);
            exit(-618661);
        }
        sn_hop_min_dist_overall[i] = tmp_dist;
    }
    //
    return;
}

static void Get_topology()
{
    for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
        Get_topology_sep(n);
    }
    //
    Get_topology_ad_hoc();
    //
    return;
}
/*
static double Lifetime_sn()
{
	double energy_max = -1.0;
	for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
		data_amount_sn[i] = 0;
	for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
		for(int t_i = 0; t_i < sink_sn_num[n]; t_i++) {
			int i = sink_sn_id[n][t_i];
			if(data_amount_sn[i] != 0) {
				printf("%s(%d): The data amount (%lf) of sensor %d is wrong, which should be zero, exiting...\n",
					   __FILE__, __LINE__, data_amount_sn[i], i);
				exit(-65156476145);
			}
		}
		for(int d = sink_hop_max_dist[n]; d > 0; d--) {
			for(int t_i = 0; t_i < sink_sn_num[n]; t_i++) {
				int i = sink_sn_id[n][t_i];
				if(sn_hop_min_dist[n][i] != d) continue;
				int id_src = i;
				data_amount_sn[id_src] += 1;
				int count_relay = 0;
				for(int j = 0; j < sink_sn_num[n]; j++) {
					int id_dst = sink_sn_id[n][j];
					if(sn_hop_table[n][id_src][id_dst] == d)
						count_relay++;
				}
				if(count_relay == 0 && d != 1) {
					printf("%s(%d): Sensor %d has no other node for relaying and not reacherable to the sink, exiting...\n",
						   __FILE__, __LINE__, id_src);
					exit(-641919);
				}
				double tmp_data_amount = data_amount_sn[id_src] * l0;
				energy_consumed_sn[id_src] = E_elec * (tmp_data_amount - l0);
				if(count_relay) {
					for(int j = 0; j < sink_sn_num[n]; j++) {
						int id_dst = sink_sn_id[n][j];
						if(sn_hop_table[n][id_src][id_dst] == d) {
							data_amount_sn[id_dst] += data_amount_sn[id_src] / count_relay;
							double cur_dist = dist_sn_all[id_src][id_dst];
							if(cur_dist < d_th) {
								energy_consumed_sn[id_src] += (tmp_data_amount / count_relay * E_elec +
															   tmp_data_amount / count_relay * e_fs *
															   cur_dist * cur_dist);
							} else {
								energy_consumed_sn[id_src] += (tmp_data_amount / count_relay * E_elec +
															   tmp_data_amount / count_relay * e_fs *
															   cur_dist * cur_dist * cur_dist * cur_dist);
							}
						}
					}
				} else {
					if(sn_hop_table[n][id_src][NUM_SENSOR_MOP_Mob_Sink] != 1) {
						printf("%s(%d): Sensor %d should connect to the sink, exiting...\n",
							   __FILE__, __LINE__, id_src);
						exit(-6141687);
					}
					double cur_dist = dist_sn_all[id_src][NUM_SENSOR_MOP_Mob_Sink + n];
					if(cur_dist < d_th) {
						energy_consumed_sn[id_src] += (tmp_data_amount * E_elec +
													   tmp_data_amount * e_fs *
													   cur_dist * cur_dist);
					} else {
						energy_consumed_sn[id_src] += (tmp_data_amount * E_elec +
													   tmp_data_amount * e_fs *
													   cur_dist * cur_dist * cur_dist * cur_dist);
					}
				}
				if(energy_consumed_sn[id_src] > energy_max) {
					energy_max = energy_consumed_sn[id_src];
				}
			}
		}
	}
	flag_lt_fn_called = 1;
	//
	return energy_max;
}
*/

static double msg_frwrd_cmb(double* energy_vec, double* data_vec, int f_ini, int d_hop, int n_msg, double l_msg)
{
    int hop_max_dist = d_hop;
    if(f_ini) {
        for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++) energy_vec[id_src] = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) data_vec[i] = 0;
        hop_max_dist = 0;
        for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
            for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
                if((n == 0 && i == 0) || hop_max_dist < sn_hop_min_dist[n][i])
                    hop_max_dist = sn_hop_min_dist[n][i];
    }
    for(int d = hop_max_dist; d > 0; d--) {
        for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++) {
            if(sn_hop_min_dist_overall[id_src] != d) continue;
            data_vec[id_src] += n_msg;
            int count_relay = 0;
            int vec_sn_tmp[NUM_SENSOR_MOP_Mob_Sink];
            for(int id_dst = 0; id_dst < NUM_SENSOR_MOP_Mob_Sink; id_dst++) {
                int tmp_flag = 0;
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
                    if(sn_hop_table[n][id_src][id_dst] == d)
                        tmp_flag++;
                if(tmp_flag) {
                    vec_sn_tmp[count_relay] = id_dst;
                    count_relay++;
                }
            }
            if(count_relay == 0 && d != 1) {
                printf("%s(%d): Sensor %d has no other node for relaying and not reacherable to the sink, exiting...\n",
                       __FILE__, __LINE__, id_src);
                exit(-641919);
            }
            double tmp_data_amount = data_vec[id_src] * l_msg;
            energy_vec[id_src] += E_elec * (tmp_data_amount - n_msg * l_msg);
            if(count_relay) {
                /*
                for(int id_dst = 0; id_dst < NUM_SENSOR_MOP_Mob_Sink; id_dst++) {
                	int tmp_flag = 0;
                	for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
                		if(sn_hop_table[n][id_src][id_dst] == d)
                			tmp_flag++;
                	if(tmp_flag) {
                		data_vec[id_dst] += data_vec[id_src] / count_relay;
                		double cur_dist = dist_sn_all[id_src][id_dst];
                		if(cur_dist < d_th) {
                			energy_vec[id_src] += (tmp_data_amount / count_relay * E_elec +
                								   tmp_data_amount / count_relay * e_fs *
                								   cur_dist * cur_dist);
                		} else {
                			energy_vec[id_src] += (tmp_data_amount / count_relay * E_elec +
                								   tmp_data_amount / count_relay * e_fs *
                								   cur_dist * cur_dist * cur_dist * cur_dist);
                		}
                	}
                }
                */
                shuffle_FRNN_MODEL(vec_sn_tmp, count_relay);
                int id_dst = vec_sn_tmp[0];
                data_vec[id_dst] += data_vec[id_src];
                double cur_dist = dist_sn_all[id_src][id_dst];
                if(cur_dist < d_th) {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_fs *
                                           cur_dist * cur_dist);
                } else {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_mp *
                                           cur_dist * cur_dist * cur_dist * cur_dist);
                }
            } else {
                int tmp_flag = 0;
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
                    if(sn_hop_table[n][id_src][NUM_SENSOR_MOP_Mob_Sink] == 1)
                        tmp_flag++;
                if(!tmp_flag) {
                    printf("%s(%d): Sensor %d should connect to the sink, exiting...\n",
                           __FILE__, __LINE__, id_src);
                    exit(-6141687);
                }
                double cur_dist = REGION_W_MOP_Mob_Sink + REGION_L_MOP_Mob_Sink;
                for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
                    if(sn_hop_table[n][id_src][NUM_SENSOR_MOP_Mob_Sink] == 1 &&
                       cur_dist > dist_sn_all[id_src][NUM_SENSOR_MOP_Mob_Sink + n])
                        cur_dist = dist_sn_all[id_src][NUM_SENSOR_MOP_Mob_Sink + n];
                if(cur_dist < d_th) {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_fs *
                                           cur_dist * cur_dist);
                } else {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_mp *
                                           cur_dist * cur_dist * cur_dist * cur_dist);
                }
            }
        }
    }
    double energy_max = -1.0;
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++)
        if(energy_vec[id_src] > energy_max)
            energy_max = energy_vec[id_src];
    //
    return energy_max;
}

static double msg_frwrd_sep(double* energy_vec, double* data_vec, int f_ini, int id_sink, int d_hop, int n_msg, double l_msg)
{
    int n = id_sink;
    int hop_max_dist = d_hop;
    if(f_ini) {
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_vec[i] = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) data_vec[i] = 0;
        hop_max_dist = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
            if(i == 0 || hop_max_dist < sn_hop_min_dist[n][i])
                hop_max_dist = sn_hop_min_dist[n][i];
    }
    for(int d = hop_max_dist; d > 0; d--) {
        for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++) {
            if(sn_hop_min_dist[n][id_src] != d) continue;
            data_vec[id_src] += n_msg;
            int count_relay = 0;
            int vec_sn_tmp[NUM_SENSOR_MOP_Mob_Sink];
            for(int id_dst = 0; id_dst < NUM_SENSOR_MOP_Mob_Sink; id_dst++) {
                if(sn_hop_table[n][id_src][id_dst] == d) {
                    vec_sn_tmp[count_relay] = id_dst;
                    count_relay++;
                }
            }
            if(count_relay == 0 && d != 1) {
                printf("%s(%d): Sensor %d has no other node for relaying and not reacherable to the sink, exiting...\n",
                       __FILE__, __LINE__, id_src);
                exit(-641919);
            }
            double tmp_data_amount = data_vec[id_src] * l_msg;
            energy_vec[id_src] += E_elec * (tmp_data_amount - n_msg * l_msg);
            if(count_relay) {
                /*
                for(int id_dst = 0; id_dst < NUM_SENSOR_MOP_Mob_Sink; id_dst++) {
                	if(sn_hop_table[n][id_src][id_dst] == d) {
                		data_vec[id_dst] += data_vec[id_src] / count_relay;
                		double cur_dist = dist_sn_all[id_src][id_dst];
                		if(cur_dist < d_th) {
                			energy_vec[id_src] += (tmp_data_amount / count_relay * E_elec +
                								   tmp_data_amount / count_relay * e_fs *
                								   cur_dist * cur_dist);
                		} else {
                			energy_vec[id_src] += (tmp_data_amount / count_relay * E_elec +
                								   tmp_data_amount / count_relay * e_fs *
                								   cur_dist * cur_dist * cur_dist * cur_dist);
                		}
                	}
                }
                */
                shuffle_FRNN_MODEL(vec_sn_tmp,
                                   count_relay); ////////////////////////////////////////////////////////////////////////// kehuanwei rnd_FRNN_MODEL
                int id_dst = vec_sn_tmp[0];
                data_vec[id_dst] += data_vec[id_src];
                double cur_dist = dist_sn_all[id_src][id_dst];
                if(cur_dist < d_th) {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_fs *
                                           cur_dist * cur_dist);
                } else {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_mp *
                                           cur_dist * cur_dist * cur_dist * cur_dist);
                }
            } else {
                if(sn_hop_table[n][id_src][NUM_SENSOR_MOP_Mob_Sink] != 1) {
                    printf("%s(%d): Sensor %d should connect to the sink, exiting...\n",
                           __FILE__, __LINE__, id_src);
                    exit(-6141688);
                }
                double cur_dist = dist_sn_all[id_src][NUM_SENSOR_MOP_Mob_Sink + n];
                if(cur_dist < d_th) {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_fs *
                                           cur_dist * cur_dist);
                } else {
                    energy_vec[id_src] += (tmp_data_amount * E_elec +
                                           tmp_data_amount * e_mp *
                                           cur_dist * cur_dist * cur_dist * cur_dist);
                }
            }
        }
    }
    double energy_max = -1.0;
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++)
        if(energy_vec[id_src] > energy_max)
            energy_max = energy_vec[id_src];
    //
    return energy_max;
}

static double Lifetime_sn()
{
    double energy_max = msg_frwrd_cmb(energy_consumed_1rn_sn, data_amount_sn, 1, 0, 1, l0);
    //////////////////////////////////////////////////////////////////////////
    for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
        msg_frwrd_sep(energy_consumed_1rn_sep_sn[n], data_amount_sn, 1, n, 0, 1, l0);
    }
    //
    return energy_max;
}

//static void Check_topology()
//{
//    if(flag_lt_fn_called == 0) Lifetime_sn();
//    for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
//        for(int t_i = 0; t_i < sink_sn_num[n]; t_i++) {
//            int id_src = sink_sn_id[n][t_i];
//            if(sn_hop_table[n][i] < 0)
//                total_penalty_MOP_Mob_Sink += penaltyVal_MOP_Mob_Sink * data_amount_sn[ID_SN[n][i]];
//        }
//    }
//    //
//    return;
//}

static void Lifetime_info_init(double& energy_min, double& energy_max, double& energy_avg)
{
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_sum_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1fld_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_1rn_sn[i] = 0;
    //
    init_pos_sn_sink(0, 1);
    for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
        pos3D_SINK[n][IND_X_MOP_Mob_Sink] = 0;
        pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = 0;
        pos3D_SINK[n][IND_Z_MOP_Mob_Sink] = 0;
    }
    //
    Get_topology();
    energy_max = Lifetime_sn();
    energy_min = energy_max;
    energy_avg = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        energy_avg += energy_consumed_1rn_sn[i];
        if(energy_min > energy_consumed_1rn_sn[i])
            energy_min = energy_consumed_1rn_sn[i];
    }
    energy_avg /= NUM_SENSOR_MOP_Mob_Sink;
    //
    return;
}

static double msg_flood_sep(double* energy_vec, double* data_vec, int f_ini, int id_sink, int d_hop, int n_msg, double l_msg)
{
    int n = id_sink;
    int hop_max_dist = d_hop;
    if(f_ini) {
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_vec[i] = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) data_vec[i] = 0;
        hop_max_dist = 0;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++)
            if(i == 0 || hop_max_dist < sn_hop_min_dist[n][i])
                hop_max_dist = sn_hop_min_dist[n][i];
    }
    for(int d = 1; d <= hop_max_dist; d++) {
        for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++) {
            if(sn_hop_min_dist[n][id_src] != d) continue;
            data_vec[id_src] = n_msg;
            int count_relay = 0;
            for(int id_dst = 0; id_dst < NUM_SENSOR_MOP_Mob_Sink; id_dst++) {
                if(sn_hop_table[n][id_src][id_dst] == d)
                    count_relay++;
            }
            if(count_relay == 0 && d != 1) {
                printf("%s(%d): Sensor %d has no other node for relaying and not reacherable to the sink, exiting...\n",
                       __FILE__, __LINE__, id_src);
                exit(-641919);
            }
            if(d == 1) count_relay = 1;
            double tmp_data_amount = data_vec[id_src] * l_msg;
            energy_vec[id_src] += E_elec * count_relay * l_msg;
            double cur_dist = d_th_sn;
            if(cur_dist < d_th) {
                energy_vec[id_src] += (tmp_data_amount * E_elec +
                                       tmp_data_amount * e_fs *
                                       cur_dist * cur_dist);
            } else {
                energy_vec[id_src] += (tmp_data_amount * E_elec +
                                       tmp_data_amount * e_mp *
                                       cur_dist * cur_dist * cur_dist * cur_dist);
            }
        }
    }
    double energy_max = -1.0;
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++)
        if(energy_vec[id_src] > energy_max)
            energy_max = energy_vec[id_src];
    //
    return energy_max;
}

static double msg_flood_ad_hoc()
{
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++) {
        energy_consumed_1fld_sn[id_src] = 0;
        for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++)
            energy_consumed_1fld_sn[id_src] += energy_consumed_1fld_sep_sn[n][id_src];
    }
    double energy_max = -1.0;
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++)
        if(energy_consumed_1fld_sn[id_src] > energy_max)
            energy_max = energy_consumed_1fld_sn[id_src];
    //
    return energy_max;
}

static void energy_flood_ini()
{
    for(int n = 0; n < NUM_SINK_MOP_Mob_Sink; n++) {
        msg_flood_sep(energy_consumed_1fld_sep_sn[n], data_amount_sn, 1, n, 0, 1, l0_fld);
    }
    msg_flood_ad_hoc();
    //
    //flag_lt_fn_called = 1;
    //
    return; // energy_max;
}

static double energy_accum_refresh(int flag_tmp, int flag_fld, int* num_rounds, int id_sink_fld, int id_sink_rn)
{
    double energy_sum_max2 = 0.0;
    int min_rm_rn = 0;
    for(int id_sn = 0; id_sn < NUM_SENSOR_MOP_Mob_Sink; id_sn++) {
        if(flag_tmp) energy_consumed_sum_sn[id_sn] += energy_consumed_tmp_sn[id_sn];
        if(flag_fld) energy_consumed_sum_sn[id_sn] += energy_consumed_1fld_sn[id_sn];
        if(id_sink_fld >= 0 && id_sink_fld < NUM_SINK_MOP_Mob_Sink)
            energy_consumed_sum_sn[id_sn] += energy_consumed_1fld_sep_sn[id_sink_fld][id_sn];
        if(id_sink_rn >= 0 && id_sink_rn < NUM_SINK_MOP_Mob_Sink)
            energy_consumed_sum_sn[id_sn] += energy_consumed_1rn_sep_sn[id_sink_rn][id_sn];
        //
        int n_rm_rn = (int)((energy_ini - energy_consumed_sum_sn[id_sn]) / energy_consumed_1rn_sn[id_sn]);
        if(id_sn == 0 || min_rm_rn > n_rm_rn) min_rm_rn = n_rm_rn;
        //
        if(num_rounds) energy_consumed_sum_sn[id_sn] += num_rounds[0] * energy_consumed_1rn_sn[id_sn];
        if(energy_sum_max2 < energy_consumed_sum_sn[id_sn])
            energy_sum_max2 = energy_consumed_sum_sn[id_sn];
        if(flag_tmp) energy_consumed_tmp_sn[id_sn] = 0;
    }
    if(min_rm_rn < 0) min_rm_rn = 0;
    if(num_rounds && min_rm_rn < num_rounds[0]) num_rounds[0] = min_rm_rn;
    if(energy_sum_max2 < 0) printf("%lf\n", energy_sum_max2);
    //
    return energy_sum_max2;
}

static double Get_attributes_sep(MY_FLT_TYPE* valAttr, int id_sink)
{
    int n = id_sink;
    double energy_sum_max2 = 0.0;
    double energy_sum_max = energy_ini;
    double lifetime_min = 0;
    int    ind_min = -1;
    double energy_rem_avg = 0.0;
    double energy_avg = 0.0;
    int flag_cv = 0;
    //
    msg_frwrd_sep(energy_consumed_tmp_sn, data_amount_sn, 1, n, 0, 1, l0_attr);
    //
    for(int rg = 0; rg < 2; rg++) {
        energy_sum_max = energy_ini;
        lifetime_min = 0;
        energy_rem_avg = 0.0;
        energy_avg = 0.0;
        flag_cv = 0;
        for(int id_sn = 0; id_sn < NUM_SENSOR_MOP_Mob_Sink; id_sn++) {
            if(energy_sum_max2 < energy_consumed_sum_sn[id_sn])
                energy_sum_max2 = energy_consumed_sum_sn[id_sn];
            if(rg == 0 ||
               (rg == 1 && dist_sn_all[id_sn][NUM_SENSOR_MOP_Mob_Sink + n] <= d_th_sn)) {
                energy_rem_avg += energy_ini - energy_consumed_sum_sn[id_sn];
                energy_avg += energy_consumed_1rn_sn[id_sn];
                //
                if(flag_cv == 0 || energy_sum_max < energy_consumed_sum_sn[id_sn])
                    energy_sum_max = energy_consumed_sum_sn[id_sn];
                double lifetime_tmp = (energy_ini - energy_consumed_sum_sn[id_sn]) / energy_consumed_1rn_sn[id_sn];
                if(flag_cv == 0 || lifetime_min > lifetime_tmp) {
                    lifetime_min = lifetime_tmp;
                    if(rg == 0) ind_min = id_sn;
                }
                flag_cv = 1;
            }
        }
        //
        valAttr[rg * 4 + 0] = energy_ini - energy_sum_max;
        valAttr[rg * 4 + 1] = lifetime_min / 100;
        valAttr[rg * 4 + 2] = energy_rem_avg / NUM_SENSOR_MOP_Mob_Sink;
        valAttr[rg * 4 + 3] = energy_avg / NUM_SENSOR_MOP_Mob_Sink * 10;
    }
    //
    valAttr[8] = pos3D_SINK[n][IND_X_MOP_Mob_Sink] / REGION_W_MOP_Mob_Sink;
    valAttr[9] = pos3D_SINK[n][IND_Y_MOP_Mob_Sink] / REGION_L_MOP_Mob_Sink;
    if(ind_min == -1) {
        printf("%s(%d): Index error, which should not appear, exiting...\n",
               __FILE__, __LINE__);
        exit(-64165919);
    }
    valAttr[10] = (pos_sensor_MOP_Mob_Sink[ind_min][IND_X_MOP_Mob_Sink] - pos3D_SINK[n][IND_X_MOP_Mob_Sink]) /
                  REGION_W_MOP_Mob_Sink;
    valAttr[11] = (pos_sensor_MOP_Mob_Sink[ind_min][IND_Y_MOP_Mob_Sink] - pos3D_SINK[n][IND_Y_MOP_Mob_Sink]) /
                  REGION_L_MOP_Mob_Sink;
    //
    return energy_sum_max2;
}

static double Update_sink(int id_sink, MY_FLT_TYPE* valOut, int* flag_moved_all)
{
    double d_tmp = 0.0;
    int n = id_sink;
    flag_moved_all[n] = 0;
    if(valOut[0] > 0) {
        double pos_old[] = {
            pos3D_SINK[n][IND_X_MOP_Mob_Sink],
            pos3D_SINK[n][IND_Y_MOP_Mob_Sink],
            pos3D_SINK[n][IND_Z_MOP_Mob_Sink]
        };
        double dist = d_MOP_Mob_Sink_min + trans_to_0_1(valOut[1]) * (d_MOP_Mob_Sink_max - d_MOP_Mob_Sink_min);
        if(dist > d_MOP_Mob_Sink_max) dist = d_MOP_Mob_Sink_max;
        if(dist < 0) dist = 0;
        double dirc = trans_to_0_1(valOut[2]) * 2 * pi;
        double dist_x = dist * cos(dirc);
        double dist_y = dist * sin(dirc);
        pos3D_SINK[n][IND_X_MOP_Mob_Sink] += dist_x;
        pos3D_SINK[n][IND_Y_MOP_Mob_Sink] += dist_y;
        if(pos3D_SINK[n][IND_X_MOP_Mob_Sink] < 0) pos3D_SINK[n][IND_X_MOP_Mob_Sink] = 0;
        if(pos3D_SINK[n][IND_X_MOP_Mob_Sink] > REGION_W_MOP_Mob_Sink) pos3D_SINK[n][IND_X_MOP_Mob_Sink] = REGION_W_MOP_Mob_Sink;
        if(pos3D_SINK[n][IND_Y_MOP_Mob_Sink] < 0) pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = 0;
        if(pos3D_SINK[n][IND_Y_MOP_Mob_Sink] > REGION_L_MOP_Mob_Sink) pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = REGION_L_MOP_Mob_Sink;
        adjust_sink_2_grid(n);
        //printf("(%lf %lf %lf) ", dist, dist_x, dist_y);
        //printf("(%lf, %lf) ", pos3D_SINK[0], pos3D_SINK[1]);
        d_tmp = range(pos_old, pos3D_SINK[n]) * resIWSN_S_1F;
        if(d_tmp > 0) flag_moved_all[n] = 1;
    }
    //
    return d_tmp;
}

static double Update_sink_direct(int id_sink, MY_FLT_TYPE* new_pos, int* flag_moved_all)
{
    double d_tmp = 0.0;
    int n = id_sink;
    flag_moved_all[n] = 0;
    if(new_pos[0] > 0) {
        double pos_old[] = {
            pos3D_SINK[n][IND_X_MOP_Mob_Sink],
            pos3D_SINK[n][IND_Y_MOP_Mob_Sink],
            pos3D_SINK[n][IND_Z_MOP_Mob_Sink]
        };
        pos3D_SINK[n][IND_X_MOP_Mob_Sink] = new_pos[1];
        pos3D_SINK[n][IND_Y_MOP_Mob_Sink] = new_pos[2];
        pos3D_SINK[n][IND_Z_MOP_Mob_Sink] = new_pos[3];
        adjust_sink_2_grid(n);
        //printf("(%lf %lf %lf) ", dist, dist_x, dist_y);
        //printf("(%lf, %lf) ", pos3D_SINK[0], pos3D_SINK[1]);
        d_tmp = range(pos_old, pos3D_SINK[n]) * resIWSN_S_1F;
        if(d_tmp > 0) flag_moved_all[n] = 1;
    }
    //
    return d_tmp;
}

static void RandomMovement(MY_FLT_TYPE* valOut)
{
    valOut[0] = rndreal_FRNN_MODEL(0, 1);
    valOut[1] = rndreal_FRNN_MODEL(0, 1);
    valOut[2] = rndreal_FRNN_MODEL(0, 1);
    if(fAdaptTHrn) valOut[3] = rndreal_FRNN_MODEL(0, 1);
    //
    return;
}

static double trans_to_0_1(double val)
{
    //double v = val > 0 ? val : -val;
    //return (1.0 / (v + 1));
    if(val > 1) return 1;
    if(val < -1) return 0;
    return (val + 1) / 2;
}

static int Get_n_fwr(double val)
{
    return (int)((THrnMax - THrnMin) * trans_to_0_1(val) + THrnMin);
}

static double msg_one_sn(double* energy_vec, double* data_vec, int id_sn, int n_msg, double l_msg)
{
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        if(dist_sn_all[id_sn][i] <= d_th_sn) {
            data_vec[i] = n_msg;
            double tmp_data_amount = data_vec[i] * l_msg;
            double cur_dist = dist_sn_all[id_sn][i];
            if(cur_dist < d_th) {
                energy_vec[i] += (tmp_data_amount * E_elec +
                                  tmp_data_amount * e_fs *
                                  cur_dist * cur_dist);
            } else {
                energy_vec[i] += (tmp_data_amount * E_elec +
                                  tmp_data_amount * e_mp *
                                  cur_dist * cur_dist * cur_dist * cur_dist);
            }
            energy_vec[id_sn] += tmp_data_amount * E_elec;
        }
    }
    double energy_max = -1.0;
    for(int id_src = 0; id_src < NUM_SENSOR_MOP_Mob_Sink; id_src++)
        if(energy_vec[id_src] > energy_max)
            energy_max = energy_vec[id_src];
    //
    return energy_max;
}

static void GreedyMaximumResidualEnergy_sep_continuous(MY_FLT_TYPE* new_pos, int id_sink)
{
    int n = id_sink;
    int step_max_x = (int)(d_MOP_Mob_Sink_max / len_grid_x);
    int step_max_y = (int)(d_MOP_Mob_Sink_max / len_grid_y);
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    energy_accum_refresh(0, 0, 0, n, -1);
    int    flag_sn[NUM_SENSOR_MOP_Mob_Sink];
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) flag_sn[i] = 0;
    // transmission vicinity
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        double cur_dist = range(pos3D_SINK[n], pos_sensor_MOP_Mob_Sink[i]) * resIWSN_S_1F;
        if(cur_dist <= d_MOP_Mob_Sink_max * resIWSN_S_1F + d_th_sn) {
            flag_sn[i]++;
        }
    }
    // energy
    int cur_hop_dist = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        data_amount_sn[i] = 0;
        if(flag_sn[i] == 0) continue;
        data_amount_sn[i] = 1;
        if(cur_hop_dist < sn_hop_min_dist[n][i])
            cur_hop_dist = sn_hop_min_dist[n][i];
    }
    msg_frwrd_sep(energy_consumed_tmp_sn, data_amount_sn, 0, n, cur_hop_dist, 0, l0_fld);
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) if(flag_sn[i]) energy_consumed_tmp_sn[i] -= E_elec * l0_fld;
    energy_accum_refresh(1, 0, 0, -1, -1);
    //
    double max_energy = 0;
    int tmp_ind = -1;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        if(flag_sn[i] == 0) continue;
        if(max_energy < energy_consumed_sum_sn[i]) {
            max_energy = energy_consumed_sum_sn[i];
            tmp_ind = i;
        }
    }
    if(tmp_ind != -1) {
        double dist_tmp = range(pos_sensor_MOP_Mob_Sink[tmp_ind], pos3D_SINK[n]);
        double dist_min = d_MOP_Mob_Sink_max < dist_tmp - d_th_sn / resIWSN_S_1F ? d_MOP_Mob_Sink_max : dist_tmp - d_th_sn /
                          resIWSN_S_1F;
        double dist_max = d_MOP_Mob_Sink_max < dist_tmp + d_th_sn / resIWSN_S_1F ? d_MOP_Mob_Sink_max : dist_tmp + d_th_sn /
                          resIWSN_S_1F;
        double alpha = atan((pos_sensor_MOP_Mob_Sink[tmp_ind][IND_Y_MOP_Mob_Sink] - pos3D_SINK[n][IND_Y_MOP_Mob_Sink]) /
                            (pos_sensor_MOP_Mob_Sink[tmp_ind][IND_X_MOP_Mob_Sink] - pos3D_SINK[n][IND_X_MOP_Mob_Sink] + 1e-6));
        new_pos[0] = 1;
        new_pos[1] = pos3D_SINK[n][IND_X_MOP_Mob_Sink] + rndreal_FRNN_MODEL(dist_min, dist_max) * cos(alpha);
        new_pos[2] = pos3D_SINK[n][IND_Y_MOP_Mob_Sink] + rndreal_FRNN_MODEL(dist_min, dist_max) * sin(alpha);
        if(new_pos[1] < 0) new_pos[1] = 0;
        if(new_pos[1] > REGION_W_MOP_Mob_Sink) new_pos[1] = REGION_W_MOP_Mob_Sink;
        if(new_pos[2] < 0) new_pos[2] = 0;
        if(new_pos[2] > REGION_L_MOP_Mob_Sink) new_pos[2] = REGION_L_MOP_Mob_Sink;
        new_pos[3] = pos3D_SINK[n][IND_Z_MOP_Mob_Sink];
    } else {
        new_pos[0] = -1;
    }
    //
    return;
}

static void GreedyMaximumResidualEnergy_sep(MY_FLT_TYPE* new_pos, int id_sink)
{
    int n = id_sink;
    int step_max_x = (int)(d_MOP_Mob_Sink_max / len_grid_x);
    int step_max_y = (int)(d_MOP_Mob_Sink_max / len_grid_y);
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) energy_consumed_tmp_sn[i] = 0;
    energy_accum_refresh(0, 0, 0, n, -1);
    int    vec_candid_sn[MAX_NUM_CANDID_SINK_SITES][NUM_SENSOR_MOP_Mob_Sink];
    double** vec_dist = (double**)malloc(MAX_NUM_CANDID_SINK_SITES * sizeof(double*));
    for(int n = 0; n < MAX_NUM_CANDID_SINK_SITES; n++)
        vec_dist[n] = (double*)malloc(NUM_SENSOR_MOP_Mob_Sink * sizeof(double));
    int    n_candid_sn[MAX_NUM_CANDID_SINK_SITES];
    int    vec_sentinel_sn_id[MAX_NUM_CANDID_SINK_SITES];
    int    flag_sn[NUM_SENSOR_MOP_Mob_Sink];
    for(int i = 0; i < MAX_NUM_CANDID_SINK_SITES; i++) n_candid_sn[i] = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) flag_sn[i] = 0;
    int num_candid_sites = 0;
    // transmission vicinity
    int tmp_cnt = 0;
    int cur_site_ind = 0;
    for(int x = -step_max_x; x <= step_max_x; x++) {
        for(int y = -step_max_y; y <= step_max_y; y++) {
            if(x == 0 && y == 0) cur_site_ind = tmp_cnt;
            double cur_site_pos[] = {
                pos3D_SINK[n][IND_X_MOP_Mob_Sink] + x * len_grid_x,
                pos3D_SINK[n][IND_Y_MOP_Mob_Sink] + y * len_grid_y,
                pos3D_SINK[n][IND_Z_MOP_Mob_Sink]
            };
            if(cur_site_pos[IND_X_MOP_Mob_Sink] >= 0 && cur_site_pos[IND_X_MOP_Mob_Sink] <= REGION_W_MOP_Mob_Sink &&
               cur_site_pos[IND_Y_MOP_Mob_Sink] >= 0 && cur_site_pos[IND_Y_MOP_Mob_Sink] <= REGION_L_MOP_Mob_Sink) {
                double dist_min_t = REGION_W_MOP_Mob_Sink + REGION_L_MOP_Mob_Sink;
                for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
                    double cur_dist = range(cur_site_pos, pos_sensor_MOP_Mob_Sink[i]) * resIWSN_S_1F;
                    if(cur_dist <= d_th_sn) {
                        flag_sn[i]++;
                        vec_candid_sn[tmp_cnt][n_candid_sn[tmp_cnt]] = i;
                        vec_dist[tmp_cnt][n_candid_sn[tmp_cnt]] = cur_dist;
                        n_candid_sn[tmp_cnt]++;
                    }
                }
            }
            tmp_cnt++;
        }
    }
    num_candid_sites = tmp_cnt;
    // energy
    int cur_hop_dist = 0;
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
        data_amount_sn[i] = 0;
        if(flag_sn[i] == 0) continue;
        data_amount_sn[i] = 1;
        if(cur_hop_dist < sn_hop_min_dist[n][i])
            cur_hop_dist = sn_hop_min_dist[n][i];
    }
    msg_frwrd_sep(energy_consumed_tmp_sn, data_amount_sn, 0, n, cur_hop_dist, 0, l0_fld);
    for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) if(flag_sn[i]) energy_consumed_tmp_sn[i] -= E_elec * l0_fld;
    energy_accum_refresh(1, 0, 0, -1, -1);
    // choose sentinel sensor node
    for(int i_site = 0; i_site < num_candid_sites; i_site++) {
        if(n_candid_sn[i_site] == 0) continue;
        double mean_energy = 0;
        double mean_dist = 0;
        for(int i = 0; i < n_candid_sn[i_site]; i++) {
            int cur_sn = vec_candid_sn[i_site][i];
            mean_energy += energy_ini - energy_consumed_sum_sn[cur_sn];
            mean_dist += vec_dist[i_site][i];
        }
        if(n_candid_sn[i_site] > 0) {
            double tmp_max = 0;
            int    tmp_ind = -1;
            mean_energy /= n_candid_sn[i_site];
            mean_dist /= n_candid_sn[i_site];
            int tmp_vec[NUM_SENSOR_MOP_Mob_Sink];
            for(int i = 0; i < n_candid_sn[i_site]; i++) tmp_vec[i] = i;
            shuffle_FRNN_MODEL(tmp_vec, n_candid_sn[i_site]);
            int tmp_flag = 0;
            for(int i = 0; i < n_candid_sn[i_site]; i++) {
                double tmp_ene = energy_ini - energy_consumed_sum_sn[vec_candid_sn[i_site][tmp_vec[i]]];
                if(tmp_ene >= mean_energy) {
                    if(vec_dist[i_site][tmp_vec[i]] <= mean_dist) {
                        vec_sentinel_sn_id[i_site] = vec_candid_sn[i_site][tmp_vec[i]];
                        tmp_flag = 1;
                        break;
                    }
                }
            }
            if(tmp_flag == 0) {
                for(int i = 0; i < n_candid_sn[i_site]; i++) {
                    double tmp_ene = energy_ini - energy_consumed_sum_sn[vec_candid_sn[i_site][tmp_vec[i]]];
                    if(tmp_ene >= mean_energy) {
                        vec_sentinel_sn_id[i_site] = vec_candid_sn[i_site][tmp_vec[i]];
                        tmp_flag = 1;
                    }
                    if(tmp_max < tmp_ene) {
                        tmp_max = tmp_ene;
                        tmp_ind = tmp_vec[i];
                    }
                }
            }
            //vec_sentinel_sn_id[i_site] = tmp_ind;
        }
    }
    // get min energy
    double max_energy = 0;
    int tmp_ind = -1;
    for(int i_site = 0; i_site < num_candid_sites; i_site++) {
        if(n_candid_sn[i_site] == 0) continue;
        int cur_sn = vec_sentinel_sn_id[i_site];
        for(int i_sn = 0; i_sn < NUM_SENSOR_MOP_Mob_Sink; i_sn++) data_amount_sn[i_sn] = 0;
        msg_one_sn(energy_consumed_tmp_sn, data_amount_sn, cur_sn, 1, l0_attr);
        energy_accum_refresh(1, 0, 0, -1, -1);
        //
        double min_energy = energy_ini;
        for(int i = 0; i < NUM_SENSOR_MOP_Mob_Sink; i++) {
            if(dist_sn_all[cur_sn][i] <= d_th_sn) {
                if(min_energy > energy_ini - energy_consumed_sum_sn[i])
                    min_energy = energy_ini - energy_consumed_sum_sn[i];
            }
        }
        //
        if(max_energy < min_energy) {
            max_energy = min_energy;
            tmp_ind = i_site;
        }
    }
    if(tmp_ind != -1 && tmp_ind != cur_site_ind) {
        int x = tmp_ind / (2 * step_max_y + 1) - step_max_x;
        int y = tmp_ind % (2 * step_max_y + 1) - step_max_y;
        new_pos[0] = 1;
        new_pos[1] = pos3D_SINK[n][IND_X_MOP_Mob_Sink] + x * len_grid_x;
        new_pos[2] = pos3D_SINK[n][IND_Y_MOP_Mob_Sink] + y * len_grid_y;
        new_pos[3] = pos3D_SINK[n][IND_Z_MOP_Mob_Sink];
        if(new_pos[1] < 0 || new_pos[1] > REGION_W_MOP_Mob_Sink ||
           new_pos[2] < 0 || new_pos[2] > REGION_L_MOP_Mob_Sink) {
            printf("%s(%d): Sink has moved to an infisible position, exiting...\n",
                   __FILE__, __LINE__);
            exit(-6471919);
        }
    } else {
        new_pos[0] = -1;
    }
    //
    for(int n = 0; n < MAX_NUM_CANDID_SINK_SITES; n++)
        free(vec_dist[n]);
    free(vec_dist);
    //
    return;
}

static void GreedyMaximumResidualEnergy(MY_FLT_TYPE* new_pos, int id_sink)
{
    int n = id_sink;
    if(flag_grid)
        GreedyMaximumResidualEnergy_sep(new_pos, n);
    else
        GreedyMaximumResidualEnergy_sep_continuous(new_pos, n);
    //
    return;
}

static void adjust_sink_2_grid(int id)
{
    if(!flag_grid) return;
    pos3D_SINK[id][IND_X_MOP_Mob_Sink] = (int)(pos3D_SINK[id][IND_X_MOP_Mob_Sink] / len_grid_x + 0.5) * len_grid_x;
    pos3D_SINK[id][IND_Y_MOP_Mob_Sink] = (int)(pos3D_SINK[id][IND_Y_MOP_Mob_Sink] / len_grid_y + 0.5) * len_grid_y;
    pos3D_SINK[id][IND_Z_MOP_Mob_Sink] = 0;
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
#define IM1_Predict_FRNN 2147483563
#define IM2_Predict_FRNN 2147483399
#define AM_Predict_FRNN (1.0/IM1_Predict_FRNN)
#define IMM1_Predict_FRNN (IM1_Predict_FRNN-1)
#define IA1_Predict_FRNN 40014
#define IA2_Predict_FRNN 40692
#define IQ1_Predict_FRNN 53668
#define IQ2_Predict_FRNN 52774
#define IR1_Predict_FRNN 12211
#define IR2_Predict_FRNN 3791
#define NTAB_Predict_FRNN 32
#define NDIV_Predict_FRNN (1+IMM1_Predict_FRNN/NTAB_Predict_FRNN)
#define EPS_Predict_FRNN 1.2e-7
#define RNMX_Predict_FRNN (1.0-EPS_Predict_FRNN)

//the random generator in [0,1)
static double rnd_uni_MOP_Mob_Sink(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_Predict_FRNN];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_Predict_FRNN + 7; j >= 0; j--) {
            k = (*idum) / IQ1_Predict_FRNN;
            *idum = IA1_Predict_FRNN * (*idum - k * IQ1_Predict_FRNN) - k * IR1_Predict_FRNN;
            if(*idum < 0) *idum += IM1_Predict_FRNN;
            if(j < NTAB_Predict_FRNN) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_Predict_FRNN;
    *idum = IA1_Predict_FRNN * (*idum - k * IQ1_Predict_FRNN) - k * IR1_Predict_FRNN;
    if(*idum < 0) *idum += IM1_Predict_FRNN;
    k = idum2 / IQ2_Predict_FRNN;
    idum2 = IA2_Predict_FRNN * (idum2 - k * IQ2_Predict_FRNN) - k * IR2_Predict_FRNN;
    if(idum2 < 0) idum2 += IM2_Predict_FRNN;
    j = iy / NDIV_Predict_FRNN;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_Predict_FRNN;         //printf("%lf\n", AM_Predict_FRNN*iy);
    if((temp = AM_Predict_FRNN * iy) > RNMX_Predict_FRNN) return RNMX_Predict_FRNN;
    else return temp;
}/*------End of rnd_uni_Classify_CNN()--------------------------*/

static int rnd_MOP_Mob_Sink(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_MOP_Mob_Sink(&rnd_uni_init_MOP_Mob_Sink) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}

static double rndreal_MOP_Mob_Sink(double low, double high)
{
    return (low + (high - low) * rnd_uni_MOP_Mob_Sink(&rnd_uni_init_MOP_Mob_Sink));
}

static double gaussrand_MOP_Mob_Sink(double a, double b)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = rnd_uni_MOP_Mob_Sink(&rnd_uni_init_MOP_Mob_Sink);
            double U2 = rnd_uni_MOP_Mob_Sink(&rnd_uni_init_MOP_Mob_Sink);

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    //return X;
    return (X * b + a);
}

static void trimLine_MOP_Mob_Sink(char line[])
{
    int i = 0;

    while(line[i] != '\0') {
        if(line[i] == '\r' || line[i] == '\n') {
            line[i] = '\0';
            break;
        }
        i++;
    }
}

static int get_setting_MOP_Mob_Sink(char* wholestr, const char* candidstr, int& val)
{
    //
    char tmp_str[MAX_STR_LEN_MOP_Mob_Sink];
    sprintf(tmp_str, "%s", wholestr);
    char tmp_delim[] = "_";
    char* p;
    int elem_int;
    int flag_found = 0;
    for(p = strtok(tmp_str, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
        if(flag_found == 1) {
            if(sscanf(p, "%d", &elem_int) != 1) {
                printf("\n%s(%d): setting value not found error...\n", __FILE__, __LINE__);
                exit(65871001);
            }
            val = elem_int;
            flag_found = 2;
            break;
        }
        if(!strcmp(p, candidstr)) flag_found = 1;
    }
    //
    return (flag_found == 2);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
