//  [1/26/2016 John]
#include <mpi.h>
#include "utility_list.h"
#include "utility_rand.h"
#include "MOP_test_suite.h"
#include <string.h>
#include <stdlib.h>
#include <iostream>
//#include "referencePoint.h"
#include <float.h>/*for
#define DBL_MAX         1.7976931348623158e+308 // max value
#define DBL_MIN         2.2250738585072014e-308 // min positive value
*/
//using namespace std;
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
#define INF_DOUBLE    1.0e299
#define INIT_POP_SIZE 10
#define MAX_UPDATE_NUM_PER_GEN 10
//
//#define CHECK_GAP_CC     5
//#define CHECK_GAP_UPDT   5
//#define CHECK_GAP_SYNC   10
//
//#define CHECK_GAP_EXCH   30
//
#define etax 20
#define etam 20
//
#define th_group 1e-6
//
#define VAL_MAX 1.0e+100
#define VAL_MIN -1.0e+100
//
#define NTRACE 25
#define NRUN 24
//
//#define PI 3.1415926535897932384626433832795
//
#define MAX_CHAR_ARR_SIZE 1024
//
//#define NUPDT 10
//#define NUPDT 50
//
//#define DEBUG_TAG_TMP
//
//#define CHECK_LOWER_BOUND(val_off, val_par, val_min) (val_off < val_min ? (val_par + val_min) / 2.0 : val_off)
//#define CHECK_UPPER_BOUND(val_off, val_par, val_max) (val_off > val_max ? (val_par + val_max) / 2.0 : val_off)
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
typedef enum {
    FLAG_OFF,
    FLAG_ON
} ENUM_GLOBAL_SWITCH_TAG;
typedef enum {
    SCALE_NONE,
    SCALE_QUANTUM,
    SCALE_LEVY,
    SCALE_CAUCHY,
    SCALE_GAUSS
} ENUM_SCALE_PARA;
typedef enum {
    LOCALIZATION,
    DECOMPOSITION,
    NONDOMINANCE
} ENUM_Algorithm_mech_type;
// execution mode controlling
typedef enum {
    EC_DE_CUR_1,
    EC_DE_CUR_2,
    EC_DE_ARCHIVE,
    EC_DE_RAND_1,
    EC_DE_RAND_2,
    EC_DE_ARCHIVE_RAND,
    EC_DE_2SELECTED,
    EC_SBX_CUR,
    EC_SBX_RAND,
    EC_MIX_DE_R_SBX_R,
    EC_MIX_DE_C_SBX_R,
    EC_MIX_DE_C_SBX_C,
    EC_MIX_SBX_C_R,
    EC_SI_MIX_DE_C_PSO,
    EC_MIX_DE_C_R,
    EC_MIX_DE_C_1_2,
    EC_MIX_DE_R_1_2,
    SI_PSO,
    SI_QPSO,
    OPTIMIZER_BLEND,
    OPTIMIZER_ENSEMBLE
} ENUM_optimizer;
typedef enum {
    DE_F_FIXED,
    DE_F_JADE,
    //DE_F_JADE_UNLIMITED,
    DE_F_SaNSDE,
    DE_F_SaNSDE_a,
    DE_F_NSDE,
    DE_F_SHADE,
    DE_F_jDE,
    DE_F_DISC
} ENUM_DE_F_type;
typedef enum {
    DE_CR_FIXED,
    DE_CR_LINEAR,
    DE_CR_JADE,
    DE_CR_SaNSDE,
    DE_CR_NSDE,
    DE_CR_SHADE,
    DE_CR_jDE,
    DE_CR_DISC
} ENUM_DE_CR_type;
typedef enum {
    PSO_PARA_FIXED,
    PSO_PARA_ADAP
} ENUM_PSO_PARA_type;
typedef enum {
    UPDATE_POP_MOEAD,
    UPDATE_POP_1TO1
} ENUM_update_pop_type;
typedef enum {
    CLONE_SLCT_ND1,
    CLONE_SLCT_ND2,
    CLONE_SLCT_ND_TOUR,
    CLONE_SLCT_UTILITY_TOUR,
    CLONE_SLCT_AGGFIT_G,
    CLONE_SLCT_AGGFIT_L,
    CLONE_SLCT_PREFER
} ENUM_Clone_selection_type;
typedef enum {
    SELECT_OBJ_BEST_TOP_N,
    SELECT_OBJ_BEST_HISTORY
} ENUM_select_obj_best_type;
typedef enum {
    PREFER_FIRST_OBJ = 0,
    PREFER_SECOND_OBJ = 1,
    PREFER_THIRD_OBJ = 2,
    PREFER_NONE_OBJ = -1
} ENUM_PREFER_WHICH_OBJ_TAG;
typedef enum {
    CLONE_EVO_LOCAL,
    CLONE_EVO_GLOBAL,
    CLONE_EVO_NONE
} ENUM_Clone_evo_type;
typedef enum {
    COMP_GREATER,
    COMP_LESS
} ENUM_COMP_OPERATOR_type;
typedef enum {
    MAIN_POP_ONLY,
    MAIN_POP_SUB_POPS_EQUAL,
    MAIN_POP_SUB_POPS_ND,
    //SUB_POPS_ONLY,
    UPDATE_MPI_STRUCTURE,
    UPDATE_MPI_STRUCTURE_ND,
    MP_ENSEMBEL
} ENUM_MPI_structure_type;
typedef enum {
    LOOP_NONE,
    LOOP_GRP,
    LOOP_POP
} ENUM_POP_GRP_LOOP_Type;
typedef enum {
    WEIGHT_BASED,
    DIVER_VAR_BASED
} ENUM_Neighbourhood_type;
typedef enum {// Not used.
    PARENT_LOCAL,
    PARENT_GLOBAL
} ENUM_Parent_type;
typedef enum {
    INIT_TAG,
    UPDATE_TAG,
    UPDATE_TAG_BRIEF,
    FINAL_TAG
} ENUM_MPI_structure_init_tag;
typedef enum {
    UPDATE_INIT,
    UPDATE_GIVEN
} ENUM_Update_tag;
typedef enum {
    COLLECT_WEIGHTED,
    COLLECT_NONDOMINATED
} ENUM_Collect_type;
typedef enum {
    OPTIMIZE_CONVER_VARS,
    OPTIMIZE_DIVER_VARS
} ENUM_Optimization_tag;
typedef enum {
    XOR_REMVARS_COPY,
    XOR_REMVARS_INHERIT,
    XOR_REMVARS_XOR_MIXED,
    XOR_REMVARS_XOR_POP,
    XOR_REMVARS_XOR_SAME_REGION
} ENUM_XOR_RemVars;
typedef enum {
    JOIN_XOR_RAND,
    JOIN_XOR_UTILITY,
    JOIN_XOR_AGGFIT
} ENUM_JOIN_VAR_XOR_type;
typedef enum {
    MUT_GENERAL_POLYNOMIAL,
    MUT_GENERAL_RAND
} ENUM_MUT_GENERAL;
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
//For selecting various kinds of algorithms
typedef enum {
    MP_0,    //DPCCMOEA: only one main population, simultaneously optimizing all objectives.
    MP_I,    //DPCCMOEA-MP-I: in the prophase, various populations have nearly equal importances; while during the anaphase, more computation resources are consumed by the main population.
    MP_II,   //the situation is opposite to that of DPCCMOEA-MP-I.
    MP_III,  //during the whole evolution process, various populations have equal importance.
    MP_ADAP
} ENUM_MultiPop_type;
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//facilitating the classification problems
typedef enum {
    MY_TYPE_NORMAL,
    MY_TYPE_FS_CLASSIFY,
    MY_TYPE_FS_CLASSIFY_TREE,
    MY_TYPE_LeNet,
    MY_TYPE_LeNet_ENSEMBLE,
    MY_TYPE_LeNet_CLASSIFY_Indus,
    MY_TYPE_NN_CLASSIFY_Indus,
    MY_TYPE_CFRNN_CLASSIFY,
    MY_TYPE_EVO1_FRNN,
    MY_TYPE_EVO2_FRNN,
    MY_TYPE_EVO3_FRNN,
    MY_TYPE_EVO4_FRNN,
    MY_TYPE_EVO5_FRNN,
    MY_TYPE_EVO_FRNN_PREDICT,
    MY_TYPE_EVO_DFRNN_PREDICT,
    MY_TYPE_INTRUSION_DETECTION_CLASSIFY,
    MY_TYPE_ACTIVITY_DETECTION_CLASSIFY,
    MY_TYPE_RecSys_SmartCity,
    MY_TYPE_EVO_CNN,
    MY_TYPE_EVO_CFRNN,
    MY_TYPE_EVO_MOBILE_SINK
} ENUM_Problem_type;
typedef enum {
    GROUPING_TYPE_CLASSIFY_NORMAL,
    GROUPING_TYPE_CLASSIFY_RANDOM,
    GROUPING_TYPE_CLASSIFY_ANALYS,
    GROUPING_TYPE_SPECTRAL_CLUSTERING
} ENUM_Grouping_type;
//#define GROUPING_TYPE_CLASSIFY GROUPING_TYPE_CLASSIFY_RANDOM
//#define GROUPING_TYPE_CLASSIFY GROUPING_TYPE_CLASSIFY_ANALYS
typedef enum {
    EVOLUTION_TYPE_NORMAL,
    EVOLUTION_TYPE_CLASSIFY_RAND,
    EVOLUTION_TYPE_CLASSIFY_ANAL,
    EVOLUTION_TYPE_CLASSIFY_ANAL_MU
} ENUM_Execution_type;
//#define EVOLUTION_TYPE EVOLUTION_TYPE_NORMAL
//#define EVOLUTION_TYPE EVOLUTION_TYPE_CLASSIFY_RAND
//#define EVOLUTION_TYPE EVOLUTION_TYPE_CLASSIFY_ANAL
typedef enum {
    TEST_TYPE_FS_NORMAL,
    TEST_TYPE_FS_BASELINE,
    TEST_TYPE_FS_ANALYSIS,
    TEST_TYPE_FS_ANALYSIS_MU
} ENUM_Test_type_FS;
//#define TEST_TYPE TEST_TYPE_NORMAL
//#define TEST_TYPE TEST_TYPE_CLASSIFY_BASELINE
//#define TEST_TYPE TEST_TYPE_CLASSIFY_ANALYSIS
//#define TEST_TYPE TEST_TYPE_CLASSIFY_ANALYSIS_MU
//////////////////////////////////////////////////////////////////////////
typedef enum {
    VAR_DOUBLE,
    VAR_BINARY,
    VAR_DISCRETE
} ENUM_VAR_ENCODING;
typedef enum {
    FEATURE_ADJUST_RAND,
    FEATURE_ADJUST_FILTER_MARKOV
} ENUM_FEATURE_ADJUST_TYPE;
typedef enum {
    MUTATION_FEATURE_NORMAL,
    MUTATION_FEATURE_MARKOV
} ENUM_FEATURE_MUTATION_TYPE;
typedef enum {
    XOR_EVO_MUT_FIX,
    XOR_EVO_MUT_ADAP
} ENUM_XOR_EVO_MUT;
//#define EVOLUTION_FEATURE_MARKOV
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
typedef enum {
    XOR_CNN_NORMAL,
    XOR_CNN_LeNet
} ENUM_XOR_CNN_TYPE;
typedef enum {
    DEL_NORMAL,
    DEL_LeNet
} ENUM_DEL_TYPE;
typedef enum {
    DIM_NORMAL,
    DIM_CONVERT_CNN
} ENUM_CNN_MAP_CONV;
typedef enum {
    LIMIT_ADJUST,
    LIMIT_TRUNCATION
} ENUM_LIMIT_EXCEEDING_PROCESSING_MECH;
typedef enum {
    DIRECT_CNN_MAP_OMNI,
    DIRECT_CNN_MAP_HORIZONTAL,
    DIRECT_CNN_MAP_VERTICAL,
    NUM_ENUM_DIRECT_CNN_MAP
} ENUM_DIRECT_CNN_MAP;
///////////////////////////////////////////////////////////
typedef enum {
    INDICATOR_IGD,
    INDICATOR_HV,
    INDICATOR_IGD_HV
} ENUM_INDICATOR_TAG;
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//Error codes
typedef enum {
    MY_ERROR_NO_ERROR,
    MY_ERROR_NO_SUCH_ALGO_MECH,
    MY_ERROR_NO_ADDRESS,
    MY_ERROR_NO_MEMORY,
    MY_ERROR_WEIGHT_READING,
    MY_ERROR_WEIGHT_NOT_ENOUGH,
    MY_ERROR_UNKNOWN_PROBLEM_INSTANCE,
    MY_ERROR_NO_CONVER_VAR,
    MY_ERROR_SAMPLE_POINT_READING,
    MY_ERROR_SAMPLE_POINT_NOT_ENOUGH,
    MY_ERROR_FILE_READING,
    MY_ERROR_OBJ_TOOMANY,
    MY_ERROR_UNKNOWN_UPDATE_TAG,
    MY_ERROR_LIMIT_PROC_MECH,
    MY_ERROR_ENUM_DIRECT_CNN_MAP,
    MY_ERROR_CHECK_LIMIT_WRONG,
    MY_ERROR_GROUPING,
    MY_ERROR_GROUP_NUM_INVALID,
    MY_ERROR_PROBLEM_TYPE,
    MY_ERROR_PROBLEM_PARA,
    MY_ERROR_OPTIMIZER_TYPE,
    MY_ERROR_TAG_SaNSDE_F,
    MY_ERROR_F_CR_SHADE,
    MY_ERROR_NAN,
    MY_ERROR_UNKNOWN_OPTIMIZER_TYPE,
    MY_ERROR_FILE_PARA,
    MY_ERROR_FILE_LINE,
    MY_ERROR_NO_SUCH_CLONE_SELECTION_TYPE,
    MY_ERROR_XOR_REMVARS_TYPE,
    MY_ERROR_LESS_INPUT,
    MY_ERROR_POP_WRONG,
    MY_ERROR_INDEX_INVALID,
    //MPI
    MY_ERROR_MPI_NO_ERROR,
    MY_ERROR_MULTI_POP_MODE,
    MY_ERROR_NO_MPI,
    MY_ERROR_MPI_STRUCTURE,
    MY_ERROR_NO_ENOUGH_MPI_IN_GROUP,
    MY_ERROR_NO_ENOUGH_MPI_IN_POP,
    MY_ERROR_MPI_TOOMANY,
    MY_ERROR_MPI_LESS,
    MY_ERROR_MPI_UPDATE_WRONG
} ENUM_Error_codes;
#define  STRINGIFY(x) #x
#define  TOSTRING(x) STRINGIFY(x)
#define  AT __FILE__ ":" TOSTRING(__LINE__)
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//global variables and functions
//---------------------------------------------------------
//variables
//
typedef struct STRCT_CONTROL_PARA {
    //
    int MFI_update_tag;
    int CLONALG_tag = FLAG_ON;
    int ScalePara_tag;
    typedef struct STRCT_SCALEPARA_PARA {
        double levy_c = 0.1;
        double levy_a = 0.1;
        double cauchy_a = 0;
        double cauchy_b = 1;
        double gauss_a = 0;
        double gauss_b = 1;
    } STRCT_scalepara_para;
    STRCT_scalepara_para st_scale_para;
    int Qubits_angle_opt_tag;
    int F_para_limit_tag;
    int weight_evo_tag;
    int mixed_var_types_tag;
    int Qubits_transform_tag;
    int commonality_xor_remvar_tag;
    int opt_binVar_as_realVar_tag;
    int opt_diverVar_separately;
    int type_select_obj_best;
    //
    int algo_mech_type = DECOMPOSITION;
    int type_grp_loop = LOOP_NONE;
    int type_pop_loop = LOOP_NONE;
    int type_xor_rem_vars;
    int type_join_xor;
    int type_mut_general;
    int optimizer_type = EC_DE_CUR_1;
    int collect_pop_type;
    int DE_F_type;
    int DE_CR_type;
    int PSO_para_type;
    int updatePop_type = UPDATE_POP_MOEAD;
    int type_clone_selection;
    int tag_prefer_which_obj;
    int type_clone_evo;
    int optimization_tag = OPTIMIZE_CONVER_VARS;
    int multiPop_mode = MP_0;
    int popSize_sub_max = 20;
    int count_multiPop = 0;
    int flag_mainPop = 0;
    int flag_multiPop = 0;
    int type_test;
    int type_grouping;
    int type_evolution;
    int type_test_type_FS;
    int type_var_encoding;
    int type_feature_adjust;
    int type_xor_evo_mut;
    int tag_gather_after_evaluate;
    int flag_check_more_update_DECOM;
    //
    int type_xor_CNN;
    int type_del_var;
    int type_dim_convert;
    int type_limit_exceed_proc;
    int indicator_tag;
    int flag_save_trace_PS;
    //
    int* types_var_all;
    int* tag_selection;
    //
    int cur_trace;
    int cur_run;
    long int global_time;
    int cur_MPI_cnt;
} STRCT_control_para;
extern STRCT_control_para st_ctrl_p;
//
typedef struct STRCT_INDICATOR_VALS {
    double mat_IGD_all[128][128];
    double mat_HV_all_TRAIN[128][128];
    double mat_minPrc_all_TRAIN[128][128];
    double mat_HV_all_VALIDATION[128][128];
    double mat_minPrc_all_VALIDATION[128][128];
    double mat_HV_all_TEST[128][128];
    double mat_HV_all[128][128];
    int    vec_NTRACE_all_VALIDATION[128];
    double vec_TIME_all[128];
    double vec_TIME_grouping[128];
    double vec_TIME_indicator[128];
    int    threshold_VALIDATION;
    int    count_VALIDATION;
} STRCT_indicator_vals;
extern STRCT_indicator_vals st_indicator_p;
//
typedef struct STRCT_GLOBAL_PARAS {
    char objsal[MAX_CHAR_ARR_SIZE];
    char varsal[MAX_CHAR_ARR_SIZE];
    //
    double start_time, end_time;
    double duration;
    //
    char algorithmName[MAX_CHAR_ARR_SIZE];
    char testInstance[MAX_CHAR_ARR_SIZE];
    char strFunctionType[MAX_CHAR_ARR_SIZE];
    int CHECK_GAP_CC;
    int CHECK_GAP_UPDT;
    int CHECK_GAP_SYNC;
    int CHECK_GAP_EXCH;
    int CHECK_GAP_BEST;
    int CHECK_GAP_REF;
    int NUPDT;
    int num_trail_per_gen = 10;
    int num_exploit_per_gen = 5;
    int num_trail_whole_per_gen = 5;
    int num_trail_per_gen_conve;
    int num_trail_per_gen_diver;
    int num_selected; // for cloning
    int* selectedIndv;
    double* selectedProb;
    int* cloneNum;
    int PF_size;
    int trans_size;
    int nObj, nDim, nPop;
    int nDim_MAP;//dimension of variable
    int nPop_mine;
    int nPop_exchange;
    int nonDominateSize;
    //
    int nInd_1pop;
    int num_subpops;
    int nInd_max;
    int maxNgrp;
    int the_size_OBJ;
    int the_size_VAR;
    int the_size_IND;
    int the_size_VAR_obj;
    int the_size_OBJ_obj;
    int the_size_VAR_1pop;
    int the_size_OBJ_1pop;
    int nInd_max_repo;
    //
    double* minLimit;
    double* maxLimit;
    //
    int iter;
    int maxIter;
    int iter_per_gen;
    int usedIter_init_grp;
    int usedIter_initPop;
    int usedIter_init;
    int remIter;
    int iter_each;
    int iter_sum;
    int generation;
    int generatMax;
    int tag_strct_updated;
    //
    FILE* fptobj;
    FILE* fptvar;
    FILE* fpttime;
    FILE* fptnum;
    FILE* debugFpt;
} STRCT_global_paras;
extern STRCT_global_paras st_global_p;
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//----Parameters in grouping-------//
typedef struct STRCT_GROUPING_ANALYSES_VALS {
    char strUpdateSolutionMode[MAX_CHAR_ARR_SIZE];
    int numDiverIndexes;
    int numConverIndexes;
    int* Dependent;
    int* Effect;
    int* Control;
    double* Control_Dist;
    double* Control_Mean;
    double* Control_Dist_Mean;
    double* Interdependence_Weight;
    double* weight_min;
    double* weight_max;
    double* var_current_grp;
    double* obj_current_grp;
    double* var_repository_grp;
    double* obj_repository_grp;
    int NumDependentAnalysis1;
    int NumDependentAnalysis;
    int NumControlAnalysis;
    int NumRepControlAnalysis;
    double div_ratio;
    double weight_thresh;
} STRCT_grouping_analyses_vals;
extern STRCT_grouping_analyses_vals st_grp_ana_p;
//
typedef struct STRCT_GROUPING_INFO_VALS {
    int* DiversityIndexs;
    int* ConvergenceIndexs;
    //
    double* diver_var_store_all;
    double* diver_var_store_mine;
    //
    int* Groups;
    int* Groups_sizes;
    int* Groups_sub_sizes;
    int* Groups_sub_disps;
    int* Groups_raw;
    int* Groups_raw_flags;
    int* Groups_raw_sizes;
    int* Groups_raw_sub_sizes;
    int* Groups_raw_sub_disps;
    int* table_mine;
    int* table_remain;
    int  table_mine_size;
    int* table_mine_flag;
    int* group_mine_flag;
    //
    int nGroup_refine;
    int numGROUP;
    int* vec_sizeGroups;
    //
    int minGroupSize = 111;
    int maxGroupSize;
    int limitDiverIndex;
    //
    int flag_predefined;
} STRCT_grouping_info_vals;
extern STRCT_grouping_info_vals st_grp_info_p;
//------------------------------------------------------------//
typedef struct STRCT_UTILITY_INFO {
    double* utility;
    double* utility_cur;
    double  utility_min;
    double  utility_max;
    double  utility_mid1;
    double  utility_mid2;
    double  utility_mean;
    double  utility_threshold;
} STRCT_utility_info;
extern STRCT_utility_info st_utility_p;

/////////////////////////////////
//evolution
typedef struct STRCT_POPULATION_EVO_INFO {
    double* var;
    double* obj;
    double* var_saved;
    double* obj_saved;
    int     curSize_inferior;
    double* var_inferior;
} STRCT_population_evo_info;
extern STRCT_population_evo_info st_pop_evo_cur;
typedef struct STRCT_POPULATION_EVO_OFFSPRING {
    double* var;
    double* obj;
    //
    int*    var_feature;
} STRCT_population_evo_offspring;
extern STRCT_population_evo_offspring st_pop_evo_offspring;
typedef struct STRCT_POPULATION_COMM_INFO {
    double* var_send;
    double* rot_angle_send;
    double* obj_send;
    double* var_recv;
    double* rot_angle_recv;
    double* obj_recv;
    double* var_left;
    double* obj_left;
    double* rot_angle_left;
    double* var_right;
    double* obj_right;
    double* rot_angle_right;
    double* var_exchange;
    double* obj_exchange;
    double* rot_angle_exchange;
    //
    double* posFactor_left;
    double* posFactor_right;
    double* posFactor_mine;
    //
    int* slctIndx;
    int* updtIndx;
    int  iUpdt;
    int* updtIndx_recv_left;
    int  iUpdt_recv_left;
    int* updtIndx_recv_right;
    int  iUpdt_recv_right;
    //
    int n_neighbor_left;
    int n_neighbor_right;
    int n_weights_left;
    int n_weights_right;
    int n_weights_mine;
} STRCT_population_comm_info;
extern STRCT_population_comm_info st_pop_comm_p;
typedef struct STRCT_POPULATION_BEST_INFO {
    double* var_best;
    double* obj_best;
    //double* rot_angle_best;
    double* var_best_exchange;
    double* obj_best_exchange;
    double* var_best_history;
    double* obj_best_history;
    //double* rot_angle_best_history;
    double* var_best_subObjs_all;
    double* obj_best_subObjs_all;
    //double* rot_angle_best_subObjs_all;
    int n_best_history;
    int cn_best_history;
    int i_best_history;
} STRCT_population_best_info;
extern STRCT_population_best_info st_pop_best_p;

typedef struct STRCT_QUANTUM_PARAS {
    double* minLimit;
    double* maxLimit;
    double* var_offspring;
    double* rot_angle_offspring;
    double* rot_angle_cur;
    double* rot_angle_cur_inferior;
    double* minLimit_rot_angle;
    double* maxLimit_rot_angle;
    //
    double* rot_angle_best;
    double* rot_angle_best_history;
    double* rot_angle_best_subObjs_all;
} STRCT_Quantum_paras;
extern STRCT_Quantum_paras st_qu_p;

typedef struct STRCT_DECOMPOSITION_PARAS {
    double* fitCur;
    double* fitImprove;
    int* countFitImprove;
    //
    double* weights_unit;
    double* weights_all;
    double* weights_left;
    double* weights_right;
    double* weights_mine;
    int*    weight_prefer_tag;
    int     prefer_intensity;
    //
    double* fun_max;
    double* fun_min;
    double* idealpoint;
    double* nadirpoint;
    //
    int*    tableNeighbor;
    int*    tableNeighbor_local;
    int     niche;
    int     niche_local;
    int     niche_neighb;
    int     limit = 2;
    double  th_select;
    int*    parent_type;
} STRCT_decomposition_paras;
extern STRCT_decomposition_paras st_decomp_p;
// PSO
typedef struct STRCT_PSO_PARA_ALL {
    double* velocity;
    double* vMax;
    double* vMin;
    double* w__cur, *w__archive;
    double* c1_cur, *c1_archive;
    double* c2_cur, *c2_archive;
    double  tt1, tt2;
    int*    indNeighbor;
    double  w_mu = 0.729;
    double  c1_mu = 1.49445;
    double  c2_mu = 1.49445;
    double  w_max = 0.9;
    double  c1_max = 2.0;
    double  c2_max = 2.0;
    double  w_min = 0.4;
    double  c1_min = 0.0;
    double  c2_min = 0.0;
    double  w_fixed = 0.729;
    double  c1_fixed = 1.49445;
    double  c2_fixed = 1.49445;
    double  alpha_begin_Qu = 1.0;
    double  alpha_final_Qu = 0.9;
} STRCT_PSO_para_all;
extern STRCT_PSO_para_all st_PSO_p;
//
//archive
typedef struct STRCT_ARCHIVE_INFO {
    double* var;
    double* obj;
    int*    rank;
    double* dens;
    double* var_exchange;
    double* obj_exchange;
    double* var_inferior;
    int     curSize_inferior;
    double* dens_exchange;
    double* var_Ex;
    double* obj_Ex;
    int*    rank_Ex;
    double* dens_Ex;
    int* indx;
    //
    int cnArchEx;
    int cnArchOld;
    int nArch, cnArch;
    int nArch_sub;
    int nArch_sub_before;
    int nArch_sub_max;
    int cnArch_exchange;
    //
    int* var_feature;
} STRCT_archive_info;
extern STRCT_archive_info st_archive_p;
//collection for selection
typedef struct STRCT_REPOSITORY_INFO {
    double* var;
    double* obj;
    double* dens;
    double* F;
    double* CR;
    double* CR_evo;
    double* w;
    double* c1;
    double* c2;
    int* tag;
    int* flag;
    //
    int nRep;
} STRCT_repository_info;
extern STRCT_repository_info st_repo_p;
//////////////////////////////////////////////////////////////////////////
//adaptive DE
typedef struct STRCT_ADAPTIVE_DE_PARA {
    double* F__cur;
    double* CR_cur, *CR_evo_cur;
    double* t1_best, *t2_best;
    int ns_JADE_F, nf_JADE_F;
    int ns_JADE_CR, nf_JADE_CR;
    int ns_SHADE, nf_SHADE;
    int ns1_SaNSDE_F, nf1_SaNSDE_F;
    int ns2_SaNSDE_F, nf2_SaNSDE_F;
    int nGen_th_accum_ada_para;
    int nGen_accum_ada_para;
    int* tag_SaNSDE_F;
    double slctProb_JADE_F;
    double slctProb_JADE_CR;
    double slctProb_SHADE;
    double slctProb_SaNSDE_F;
    double F, CR, CR_rem;
    int nHistSHADE = 50;
    int iHistSHADE;
    double* F_hist, *CR_hist;
    int Fcount, CRcount;
    int* Sflag;
    double F_mu, CR_mu, CR_evo_mu;
    double t1_best_mu, t2_best_mu;
    double F_mu_arch, CR_mu_arch, CR_evo_mu_arch;
    double c_para;
    double* F__archive, *CR_archive, *CR_evo_arc;
    int candid_num;
    double* candid_F;
    double* candid_CR;
    double* prob_F;
    double* prob_CR;
    double* disc_F;
    double* disc_CR;
    int* indx_disc_F;
    int* indx_disc_CR;
} STRCT_adaptive_DE_para;
extern STRCT_adaptive_DE_para st_DE_p;
//

typedef struct STRCT_ALL_OPTIMIZER_PARAS {
    int ns_optimizer_1;
    int nf_optimizer_1;
    int ns_optimizer_2;
    int nf_optimizer_2;
    int ns_optimizer_PSO;
    int nf_optimizer_PSO;
    int nGen_th_accum_ada_opti;
    int nGen_accum_ada_opti;
    double slctProb_opt_1;
    double slctProb_opt_2;
    double slctProb_PSO;
    double p_best_ratio;
    double* optimizer_prob;
    int     num_optimizer;
    int* optimizer_candid;
    int* optimizer_types_all;
    int* DE_F_types_all;
    int* DE_CR_types_all;
    int* PSO_para_types_all;
    double  ratio_mut;
    double* rate_Commonality;
} STRCT_all_optimizer_paras;
extern STRCT_all_optimizer_paras st_optimizer_p;
//
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
//  MPI
typedef struct STRCT_MPI_ENV_INFO {
    int   mpi_size;
    int   mpi_rank;
    char  my_name[MAX_CHAR_ARR_SIZE];
    int   name_len;
    //
    int     num_MPI_master;
    int*    vec_num_MPI_master;
    int*    vec_num_MPI_slave;
    double* vec_importance;
    double* vec_MPI_ratio;
    int*    nPop_all;
    //
    int* recv_size;
    int* disp_size;
    int* each_size;
    int* recv_size_subPop;
    int* disp_size_subPop;
    int* each_size_subPop;
    //
    int rank_test = 66;
    int color_pop;
    MPI_Comm comm_pop;
    int mpi_rank_pop;
    int mpi_size_pop;
    int* globalRank_master_pop;
    int color_subPop;
    MPI_Comm comm_subPop;
    int mpi_rank_subPop;
    int mpi_size_subPop;
    int color_obj;
    MPI_Comm comm_obj;
    int mpi_rank_obj;
    int mpi_size_obj;
    int color_master_subPop;
    MPI_Comm comm_master_subPop_globalScope;
    int mpi_rank_master_subPop_globalScope;
    int mpi_size_master_subPop_globalScope;
    int root_master_subPop_globalScope;
    MPI_Comm comm_master_subPop_popScope;
    int mpi_rank_master_subPop_popScope;
    int mpi_size_master_subPop_popScope;
    MPI_Comm comm_node;
    int mpi_rank_node;
    int mpi_size_node;
    int color_node;
    int color_master_pop;
    MPI_Comm comm_master_pop;
    int mpi_rank_master_pop;
    int mpi_size_master_pop;
    //
    int  cur_grp_index;
    int  cur_pop_index;
    //
    int* ns_pops;
    int* nf_pops;
} STRCT_MPI_env_info;
extern STRCT_MPI_env_info st_MPI_p;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//functions
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  control_main.cpp
void set_para(int npop, int ndim, int nobj, int narch, int maxIter, char* func_name, int iRun);
void run_DPCC();
void save_during_operation();
void updateStructure();
void save_during_operation_ND();
void updateStructure_ND();
//------------------------------------------------------------------------------------------------------------
//  control_mem.cpp
void allocateMemory_grouping_ana();
void allocateMemory_grouping_info();
void allocateMemory_MPI();
void allocateMemory();
double* allocDouble(int size);
int* allocInt(int size);
void freeMemory_grouping_ana();
void freeMemory_grouping_info();
void freeMemory_MPI();
void freeMemory();
//------------------------------------------------------------------------------------------------------------
//  control_init.cpp
void get_alg_mech_func_id_to_test(const char* filename, int& the_alg_type, int& fun_start_num,
                                  int& fun_end_num);
void modify_num_run(const char* prob, int& num_run);
void modify_hyper_paras(const char* prob, int nobj, int ndim, int& NP, int& N_arch, int& maxIter,
                        int& the_type_test);
void readParaFromFile();
void setParaDefault();
void checkParas();
void initializePara();
void initializePopulation();
void initializePopulation_Latin_hyperCube();
void setLimits_transformed();
void initializeGenNum();
void updateGenNum();
//------------------------------------------------------------------------------------------------------------
//  control_util.cpp
void updateNeighborTable(int type, int algoMechType);
double dist_vector(double* vec1, double* vec2, int len);
void minfastsort(double* val, int* ind, int size, int num);
void transformPop(int algoMechType);
void refinePop_ND(int ref_tag, int algoMechType);
void setVarTypes();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  grouping--------------------------------------------------------------------------------------------------
//  grouping_main.cpp
void exec_DVA();
void VarGrouping();
//------------------------------------------------------------------------------------------------------------
//  grouping_CVA.cpp
void ControlVarAnalysis();
void ControlVarAnalysis_weight();
void ControlVarAnalysis_serial();
//------------------------------------------------------------------------------------------------------------
//  grouping_dependence.cpp
void InterdependenceAnalysis();
void InterdependenceAnalysis_gDG2();
void InterdependenceAnalysis_gDG2_serial();
void InterdependenceAnalysis_master_slave();
void DependentVarAnalysis();
//------------------------------------------------------------------------------------------------------------
//  grouping_utils.cpp
void initializePopulation_grouping();
int dominate_main(double* Data);
int dominate_judge(double* pf1, double* pf2);
bool IsDistanceVariable(int j);
bool IsDiversityVariable(int j);
bool UpdateSolution(double* Parent, double* Offspring);
double CalcDistance(double* Data);
void myQuickSort(double Data[], int indexArr[], int left, int right);
//------------------------------------------------------------------------------------------------------------
//  grouping_join_assign.cpp
void localAssignGroup(int iPop, int iGrp);
void grouping_variables();
void grouping_variables_WDCN();
void grouping_variables_ARRANGE2D();
void grouping_variables_HDSN_URBAN();
void grouping_variables_RS_SC();
void grouping_variables_IWSN_S_1F();
void grouping_variables_IWSN_S_1F_6();
void grouping_variables_IWSN_S_1F_with_nG(int numGROUP);
void grouping_variables_EdgeComputation();
void grouping_variables_LeNet();
void grouping_variables_LeNet_less();
void grouping_variables_NN_indus();
void grouping_variables_CFRNN();
void grouping_variables_EVO1_FRNN();
void grouping_variables_EVO2_FRNN();
void grouping_variables_EVO3_FRNN();
void grouping_variables_EVO4_FRNN();
void grouping_variables_EVO5_FRNN();
void grouping_variables_EVO_FRNN(int* xType);
void grouping_variables_IntrusionDetection_Classify();
void grouping_variables_ActivityDetection_Classify();
void grouping_variables_evoCNN();
void grouping_variables_evoCFRNN();
void grouping_variables_unif(int numGrps, int flag_rand);
void grouping_variables_classify_random();
void grouping_variables_classify_cluster_kmeans();
void grouping_variables_classify_cluster_spectral();
//void refineGroup();
//void initializeGroup_refine();
//void localAssignGroup_refine();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  cooperativeCoevolution------------------------------------------------------------------------------------
//  CC_main.cpp
//----------decomposition-----------------------------------------------
void cooperativeCoevolution(int algoMechType);
void mainLoop_CC();
void mainLoop_CC_CLONALG();
void rand_selection(int* inputIndex, int inputNum, int* outputIndex, int outputNum);
void tour_selection(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth);
void tour_selection_repetitive(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth);
void tour_selection_ND(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth);
void tour_selection_aggFit_greater(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth);
void tour_selection_aggFit_less(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth);
void tour_selection_sub(int* outputIndex, int outputNum, int depth);
void greedy_selection(int* outputIndex, int outputNum);
void greedy_selection_DECOM(int* outputIndex, int& outputNum);
void clone_DECOM();
//----------non-dominance-----------------------------------------------
void cooperativeCoevolution_ND();
void mainLoop_allObjs_ND();
void mainLoop_subObj_ND();
void greedy_selection_ND(int* outputIndex, int& outputNum);
void clone_ND();
void qSortGeneral(double* data, int arrayFx[], int left, int right);
void qSortBase(int* data, int left, int right);
//------------------------------------------------------------------------------------------------------------
//  CC_joinVar.cpp
void joinVar(int iS, int iD, int var_prop_tag);
void joinVar_allObjs_local_ND(int iP, int jP);
void joinVar_allObjs_global_ND(int iP, int jP);
void joinVar_subObj_ND(int iP, int jP);
//------------------------------------------------------------------------------------------------------------
//  CC_update.cpp
void update_population_1to1(int iP);
void update_population_DECOM(int iP, int iC, int nPop, double* osp_obj, double* osp_var,
                             double* weights_all, int useSflag, int* Sflag,
                             double* rot_angle_offspring,
                             int niche, int niche_local, int* tableNeighbor, int* tableNeighbor_local, int maxNneighb, int parent_type);
void update_population_DECOM_from_transNeigh();
void update_population_and_weights(double* var_parents, double* obj_parents, int n_parents,
                                   double* var_offsprings, double* obj_offsprings, int n_offsprings,
                                   double* old_weights, int n_old_w, double* new_weights, int n_new_w,
                                   int n_var, int n_obj);
//int linear_dominate(double* compt1, double* compt2, double* w_comb, int n_w, int n_obj);
void update_xBest_history(int update_tag, int nPop, int* vec_indx, double* addrX, double* addrY, double* addrZ);
void update_xBest(int tag, int nPop, int* vec_indx, double* addrX, double* addrY, double* addrZ);
void OldSoluUpdate(int iReplaced, int depth, int nPop, double* weights_all, int niche, int* tableNeighbor, int maxNneighb);
void checkUpdatedIndx(int ind);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  optimizer_selection.cpp
void gen_offspring_selected(int* tmp_indx);
void gen_offspring_selected_one(int iS, int iD, int* tmp_indx);
void gen_offspring_allObjs_local_ND();
void gen_offspring_allObjs_global_ND();
void gen_offspring_subObj_ND();
//------------------------------------------------------------------------------------------------------------
//  optimizer_evolution.cpp
void DE_1_bin(double* pbase, double* p1, double* p2, double* parent, double* child, int iP, int iPara);
void DE_1_exp(double* pbase, double* p1, double* p2, double* parent, double* child, int iP, int iPara);
void DE_2_exp(double* pbase, double* p1, double* p2, double* p3, double* p4, double* parent, double* child, int iP, int iPara);
void DE_selected1_1_exp(double* pbase, double* p1, double* p2, double* pb, double* parent, double* child, int iP, int iPara);
void DE_selected1_2_exp(double* pbase, double* p1, double* p2, double* p3, double* p4, double* pb,
                        double* parent, double* child, int iP, int iPara);
void DE_selected2_1_exp(double* pbase, double* p1, double* p2, double* pb1, double* pb2, double* parent, double* child, int iP,
                        int iPara);
void EA_pure_xor(double* p1, double* p2, double* parent, double* child, int iP, int iPara);
void evo_bin_commonality(double* p0, double* p1, double* parent, double* child, int iP);
void SBX_classic(double* p1, double* p2, double* parent, double* child, int iP, int iPara);
void PSO_classic(double* p1, double* parent, double* child, double* vel, int iP, int iPara);
void QPSO_classic_2(double* p1, double* p_center, double* parent, double* child, double* vel, int iP, int iPara);
void Quantum_transform_update(double* parent, double* child, double* rot, int iP);
//------------------------------------------------------------------------------------------------------------
//  optimizer_crossover.cpp
void xor_remVars_switch(int iS, int iD);
void xor_remVars_inherit_block(int iS, int iD);
//------------------------------------------------------------------------------------------------------------
//  optimizer_mutation.cpp
void realmutation(double* trail, double rate);
void realmutation_whole_bin(double* indiv, double rate);
void binarymutation_whole_bin_Markov(double* indiv, double rate);
void randmutation_whole(double* indiv, double rate);
void realmutation_whole(double* indiv, double rate);
void realmutation_whole_fixed(double* indiv, int* inds, int n);
void localSearch(int iP);
void refinement(int iP);
void adjustFeatureNum_hybrid(double* indiv);
void adjustFeatureNum_filter(double* indiv);
void adjustFeatureNum_Markov(double* indiv);
void adjustFeatureNum_rand(double* indiv);
void adjustFeatureNum_rank_corr_tour(double* indiv);
void adjustFeatureNum_rank_replace(double* indiv);
void adjustFeatureNum_corr_replace(double* indiv);
void adjustFeatureNum_rank_corr_memetic(double* indiv);
void adjustFeatureNum_cluster(double* indiv);
//------------------------------------------------------------------------------------------------------------
//  optimizer_updatePara.cpp
void generate_para_all();
void generate_F_current();
void generate_CR_current();
void update_para_statistics();
void update_F_p_SaNSDE();
void update_F_CR_hist_SHADE_simple();
void update_F_CR_hist_SHADE();
void update_F_mu_JADE();
void update_CR_mu_JADE();
void update_CR_mu_evo();
void update_t_best_mu();
void generate_CRall_evo();
void generate_t_best();
void update_th_select_mu();
void generate_th_select_all();
void generate_para_PSO();
void update_para_mu_PSO();
void generate_optimizer_types();
void update_optimizer_prob();
void generate_F_disc();
void generate_CR_disc();
void update_F_disc_prob();
void update_CR_disc_prob();
//------------------------------------------------------------------------------------------------------------
//  optimizer_utils.cpp
int exponentialRankingSelection(int length, double pressure);
int linearRankingSelection(int length, double pressure);
void boundaryExceedingFixing(double x_parent, double& x_child, double x_min, double x_max);
inline void boundaryExceedingFixing(double x_parent, double& x_child, double x_min, double x_max)
{
    if(x_child < x_min) {
        if(st_ctrl_p.type_limit_exceed_proc == LIMIT_ADJUST)
            x_child = rndreal(x_min, x_parent);
        else if(st_ctrl_p.type_limit_exceed_proc == LIMIT_TRUNCATION)
            x_child = x_min;
        else {
            printf("strct_ctrl_para.type_limit_exceed_proc assignment wrong, exiting...\n");
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_LIMIT_PROC_MECH);
        }
    }
    if(x_child > x_max) {
        if(st_ctrl_p.type_limit_exceed_proc == LIMIT_ADJUST)
            x_child = rndreal(x_parent, x_max);
        else if(st_ctrl_p.type_limit_exceed_proc == LIMIT_TRUNCATION)
            x_child = x_max;
        else {
            printf("strct_ctrl_para.type_limit_exceed_proc assignment wrong, exiting...\n");
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_LIMIT_PROC_MECH);
        }
    }
    if(x_child < x_min || x_child > x_max)
        printf("%s: %e %e %e %e\n", AT, x_parent, x_child, x_min, x_max);
}
void transform_to_Qbits(double* source, double* destination, int nIndiv);
void transform_fr_Qbits(double* source, double* destination, int nIndiv);
void LeNet_delete(double* pTrail);
void LeNet_xor(double* pCurrent, double* p1, double* p2, double* pTrail, float t_CR);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  ND_nondominance.cpp
void refineRepository_generateArchive();
void refineRepository_generateArchive_sub();
void refineRepository_generateND(double* _dest, double* _destFit, double* _destDens,
                                 int* _destClass, double* _dest_one_group, int& _n_d, int _n);
void refineRepository_generateArchive_SDE();
void refineRepository_deleteTheSame(double* obj_store, double* var_store, int& n_store, int thresh);
void refineRepository_markTheSame(double* obj_store, int* tag_store, int n_store);
void K_Neighbor_Nearest_SDE(int count, int frontSize, list* elite);
void K_Neighbor_Nearest_SDE(int count, int frontSize, list* elite, double* _dest, double* _destFit,
                            double* _destDens, double* _dest_one_group, int _n);
double* generateDistMatrix(double* f, int non_size, int* non_indexes);
double Euclid_Dist(double* vec1, double* vec2);
void sort_dist_index(double* a, int* b, int left, int right);
bool cmp_crowd(double* a, double* b);
//------------------------------------------------------------------------------------------------------------
//  ND_dominanceComparator.cpp
int isTheSame(double* obj1, double* obj2);
int dominanceComparator(double* obj1, double* obj2);
int check_dominance(double* obj1, double* obj2);
int L_dominance(double* obj1, double* obj2);
//------------------------------------------------------------------------------------------------------------
//  ND_distance.cpp
void fillCrowdingDistance(int count, int frontSize, list* elite);
void fillCrowdingDistance(int count, int frontSize, list* elite, double* _dest, double* _destFit,
                          double* _destDens, double* _dest_one_group, int _n);
void assignCrowdingDistanceList(list* lst, int frontSize);
void assignCrowdingDistanceIndexes(int c1, int c2);
void assignCrowdingDistance(int* distance, int** arrayFx, int frontSize);
//------------------------------------------------------------------------------------------------------------
//  ND_sort.cpp
void quickSortDistance(int* distance, int frontSize);
void qSortDistance(int* distance, int left, int right);
void quickSortFrontObj(int objcount, int arrayFx[], int sizeArrayFx);
void qSortFrontObj(int objcount, int arrayFx[], int left, int right);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  utility.cpp
void copyToArchiveFromRepository(int iA, int iR);
void copyFromRepository(int iR, int iDest, double* dest, double* destFit, double* destDens, double* dest_one_group);
void selectSamples(int pp, int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3, int* r4, int* r5);
void selectSamples_niche(int* niche_table, int niche_size,
                         int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3, int* r4, int* r5);
void selectSamples_RR(int pp, double* prob, int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3);
void selectSamples_clone_RR(int pp, int* prob, int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3);
void tourSelectSamples_sub(int pp, int depth, int iObj, double* data,
                           int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3);
void tourSelectSamples_rank_dist(int pp, int depth, int* rank, double* dist,
                                 int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3);
bool isDuplicate(double* s, int i, int j);
void get_nonDominateSize();
void transform_var_feature(double* individual, int* var_transformed, int numIndiv);
void my_error_fun(const char* location, const char* msg, int error_code);
void convertVar_CNN(double* indiv_ori, double* indiv_cvt);
//------------------------------------------------------------------------------------------------------------
//  utility_display.cpp
void show_uTrail_one_group();
void show_uTrail();
void showPopulation();
void showArchive();
void showArchiveEx();
void showArchive_exchange();
void showRepository();
void show_uArchive();
void showLimits();
void showGlobalBest();
void showGlobalBestEx();
void showGroup();
void show_F_CR_mu();
void show_F_CR();
void showGroup_raw();
void show_mpi_info();
void show_indicator_vars_simp(int finalTag);
void show_para_A();
void show_para_B();
void show_para_Prob();
void show_DeBug_info();
//------------------------------------------------------------------------------------------------------------
//  utility_save.cpp
void save_double(FILE* fpt, double* pTarget, int num, int dim, int tag);
void save_double_as_int(FILE* fpt, double* pTarget, int num, int dim, int tag);
void save_int(FILE* fpt, int* pTarget, int num, int dim, int tag);
void output_group_info();
void output_group_info_brief();
void output_group_raw_info_brief();
void output_csv_matrix_recorded(const char* filename, double mat_data[][128], int nRun, int nTrace);
void output_csv_mean_std(const char* fnm_mean, const char* fnm_std, double mat_data[][128],
                         int nRun, int nTrace, const char* prob, int nobj, int ndim);
void output_csv_vec_int(const char* filename, int mat_data[],
                        int nRun, int nTrace, const char* prob, int nobj, int ndim);
void output_csv_vec_double_with_mean(const char* filename, double mat_data[],
                                     int nRun, int nTrace, const char* prob, int nobj, int ndim);
//------------------------------------------------------------------------------------------------------------
//  utility_fitness.cpp
double fitnessFunction(double* solution, double* lamda);
double norm(double* vec, int len);
double innerproduct(double* vec1, double* vec2, int len);
void update_idealpoint(double* candidate);
void update_nadirpoint(double* solutionAddress, int pop_size, int nObj);
void load_samplePoints();
void load_weights();
void generate_new_weights(double* new_weights, int nWT, int nOBJ);
void normalize_weights();
void calculate_utility();
void total_utility(int nPop, double* weights);
void one_utility(double obj_before, double obj_current, int iP);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------------
//  MPI-------------------------------------------------------------------------------------------------------
//  MPI_construction.cpp
void setMPI();
void build_MPI_structure(int structure_type, int init_tag);
//------------------------------------------------------------------------------------------------------------
//  MPI_localIni.cpp
void localInitialization();
void localInitialization_ND();
//------------------------------------------------------------------------------------------------------------
//  MPI_exchange.cpp
//void exchangePopWithinGroup();
//void exchangeArchWithinGroup();
//void exchangePopWithinObj();
//void exchangeArchWithinObj();
//void exchangePopGlobal();
void exchangePopGlobal_vonNeumann();
void exchangePopBestGlobal_vonNeumann();
//void exchangeArchGlobal();
void get_xBestWithinObj();
void exchange_xBestWithinGroup();
void exchange_xBestWithinObj();
void exchangeInfo();
void exchangeInfo_DPCCMOEA();
void exchangeInfo_ND();
//------------------------------------------------------------------------------------------------------------
//  MPI_collect.cpp
void collectNDArchive();
void collectNDArchiveEx();
void collectViaBinTree(int the_rank, int the_size, MPI_Comm the_comm, int flag_tag);
void collectDecompositionArchiveWithinPop(int algoMechType);
void collectDecompositionArchiveBeyondPop(int algoMechType);
void collectDecompositionArchive(int algoMechType);
void gatherInfoBeforeUpdateStructure(int algoMechType);
void scatterInfoAfterUpdateStructure(int algoMechType);
//------------------------------------------------------------------------------------------------------------
//  MPI_synchronize.cpp
void population_synchronize(int algoMechType);
void population_synchronize_random();
void population_synchronize_ND();
void population_synchronize_random_ND();
void synchronizeObjectiveBests(int algoMechType);
void synchronizeReferencePoint(int algoMechType);
//------------------------------------------------------------------------------------------------------------
//  MPI_combine.cpp
//void generateBestCombinations();
//void generateBestPopulations();
//void generateBestPopulationOne();
//------------------------------------------------------------------------------------------------------------
//  MPI_utility.cpp
void update_recv_disp_simp(int* num, int n, int l);
void update_recv_disp(int* _each, int _n, int _l, int* _recv, int* _disp);
void transfer_x_neighbor();
void transfer_x_neighbor_updated();
void scatter_evaluation_gather();
