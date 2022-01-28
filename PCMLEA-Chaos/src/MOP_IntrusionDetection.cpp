#include "MOP_IntrusionDetection.h"
#include <float.h>

//////////////////////////////////////////////////////////////////////////
#define FUZZY_RULE_OBJ_OFF_MOP_INTRUSIONDETECTION 0
#define FUZZY_RULE_OBJ_ON_MOP_INTRUSIONDETECTION  1
#define FUZZY_RULE_OBJ_STATUS_MOP_INTRUSIONDETECTION  FUZZY_RULE_OBJ_OFF_MOP_INTRUSIONDETECTION

#define LINEAR_CONVERSION_INC_DISC_MOP_INTRUSIONDETECTION 0
#define LINEAR_CONVERSION_EXC_DISC_MOP_INTRUSIONDETECTION 1
#define LINEAR_CONVERSION_DISC_MOP_INTRUSIONDETECTION LINEAR_CONVERSION_EXC_DISC_MOP_INTRUSIONDETECTION

#define NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION 0
#define NETWORK_SIMPLICITY_CONNECTION_MOP_INTRUSIONDETECTION 1
//#define NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION
#define NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION NETWORK_SIMPLICITY_CONNECTION_MOP_INTRUSIONDETECTION

//////////////////////////////////////////////////////////////////////////
#define NUM_label_KDD99 5
#define IND_label_KDD99 41
int LEN_label_class[5] = {
    1, 6, 10, 8, 15
};
char label_KDD99[5][16][32] = {
    {
        "normal."
    }, //NORMAL
    {
        "ipsweep.", "mscan.", "nmap.", "portsweep.", "saint.", "satan."
    }, //PROBE
    {
        "apache2.", "back.", "land.", "mailbomb.", "neptune.", "pod.", "processtable.", "smurf.", "teardrop.", "udpstorm."
    }, //DOS
    {
        "buffer_overflow.", "httptunnel.", "loadmodule.", "perl.", "ps.", "rootkit.", "sqlattack.", "xterm."
    }, //U2R
    {
        "ftp_write.", "guess_passwd.", "imap.", "multihop.", "named.", "phf.", "sendmail.", "snmpgetattack.", "snmpguess.",
        "spy.", "warezclient.", "warezmaster.", "worm.", "xlock.", "xsnoop."
    }  //R2L
};
int count_label_KDD99[NUM_label_KDD99][16] = { 0 };
#define NUM_protocol_type_KDD99 3
#define IND_protocol_type_KDD99 1
char protocol_type_KDD99[NUM_protocol_type_KDD99][8] = {
    "tcp", "udp", "icmp"
};
int count_protocol_type_KDD99[NUM_protocol_type_KDD99] = { 0 };
#define NUM_service_KDD99 71
#define IND_service_KDD99 2
char service_KDD99[NUM_service_KDD99][16] = {
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "domain_u", "echo", "eco_i", "ecr_i",
    "efs", "exec", "finger", "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001",
    "imap4", "IRC", "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns",
    "netbios_ssn", "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump", "pop_2", "pop_3", "printer", "private", "red_i",
    "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat", "telnet", "tftp_u", "tim_i", "time",
    "urh_i", "urp_i", "uucp", "uucp_path", "vmnet", "whois", "X11", "Z39_50",
    "icmp" // This only appears in test data, which is a protocol.
};
int count_service_KDD99[NUM_service_KDD99] = { 0 };
#define NUM_flag_KDD99 11
#define IND_flag_KDD99 3
char flag_KDD99[NUM_flag_KDD99][8] = {
    "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"
};
int count_flag_KDD99[NUM_flag_KDD99] = { 0 };

//////////////////////////////////////////////////////////////////////////
int NDIM_IntrusionDetection_FRNN_Classify = 0;
int NOBJ_IntrusionDetection_FRNN_Classify = 0;

#define FULL_TRAIN_DATA_MOP_INTRUSIONDETECTION 0
#define LESS_TRAIN_DATA_MOP_INTRUSIONDETECTION 1
#define TRAIN_DATA_NUM_MOP_INTRUSIONDETECTION LESS_TRAIN_DATA_MOP_INTRUSIONDETECTION

#define nCol_KDD99 42
MY_FLT_TYPE** dataKDD99_train = NULL;
#if TRAIN_DATA_NUM_MOP_INTRUSIONDETECTION == FULL_TRAIN_DATA_MOP_INTRUSIONDETECTION
#define nRow_KDD99_train 4898431
#else
#define nRow_KDD99_train 494021
#endif
//float** dataKDD99_valid = NULL;
//#define nRow_KDD99_valid 494021
MY_FLT_TYPE** dataKDD99_test = NULL;
#define nRow_KDD99_test 311029

#define FLAG_OFF_MOP_INTRUSIONDETECTION 0
#define FLAG_ON_MOP_INTRUSIONDETECTION  1
#define STATUS_OUTPUT_MOP_INTRUSIONDETECTION FLAG_OFF_MOP_INTRUSIONDETECTION

#define STATUS_OUT_INDEICES_MOP_INTRUSIONDETECTION FLAG_OFF_MOP_INTRUSIONDETECTION

#define STATUS_OUT_ASSOCIATION_MEM_MOP_INTRUSIONDETECTION FLAG_ON_MOP_INTRUSIONDETECTION

//////////////////////////////////////////////////////////////////////////
int flag_continuous_KDD99[nCol_KDD99] = {
    1, //duration: continuous. 1
    0, //protocol_type : symbolic. 2
    0, //service : symbolic. 3
    0, //flag : symbolic. 4
    1, //src_bytes : continuous. 5
    1, //dst_bytes : continuous. 6
    0, //land : symbolic. 7
    1, //wrong_fragment : continuous. 8
    1, //urgent : continuous. 9
    1, //hot : continuous. 10
    1, //num_failed_logins : continuous. 11
    0, //logged_in : symbolic. 12
    1, //num_compromised : continuous. 13
    0, //root_shell : symbolic. 14 ???????????????????????
    0, //su_attempted : symbolic. 15
    1, //num_root : continuous. 16
    1, //num_file_creations : continuous. 17
    1, //num_shells : continuous. 18
    1, //num_access_files : continuous. 19
    1, //num_outbound_cmds : continuous. 20 all are zero
    0, //is_host_login : symbolic. 21
    0, //is_guest_login : symbolic. 22
    1, //count : continuous. 23
    1, //srv_count : continuous. 24
    1, //serror_rate : continuous. 25
    1, //srv_serror_rate : continuous. 26
    1, //rerror_rate : continuous. 27
    1, //srv_rerror_rate : continuous. 28
    1, //same_srv_rate : continuous. 29
    1, //diff_srv_rate : continuous. 30
    1, //srv_diff_host_rate : continuous. 31
    1, //dst_host_count : continuous. 32
    1, //dst_host_srv_count : continuous. 33
    1, //dst_host_same_srv_rate : continuous. 34
    1, //dst_host_diff_srv_rate : continuous. 35
    1, //dst_host_same_src_port_rate : continuous. 36
    1, //dst_host_srv_diff_host_rate : continuous. 37
    1, //dst_host_serror_rate : continuous. 38
    1, //dst_host_srv_serror_rate : continuous. 39
    1, //dst_host_rerror_rate : continuous. 40
    1, //dst_host_srv_rerror_rate : continuous. 41
    0  //label 42
};
MY_FLT_TYPE max_feature_KDD99[nCol_KDD99];
MY_FLT_TYPE min_feature_KDD99[nCol_KDD99];
MY_FLT_TYPE max_feature_KDD99_train[nCol_KDD99];
MY_FLT_TYPE min_feature_KDD99_train[nCol_KDD99];
MY_FLT_TYPE max_feature_KDD99_valid[nCol_KDD99];
MY_FLT_TYPE min_feature_KDD99_valid[nCol_KDD99];
MY_FLT_TYPE max_feature_KDD99_test[nCol_KDD99];
MY_FLT_TYPE min_feature_KDD99_test[nCol_KDD99];
int flag_use_this_feature[nCol_KDD99];
int num_input_KDD99 = nCol_KDD99;
int num_output_KDD99 = NUM_label_KDD99;

int** ind_all_class_train = NULL;
//int** ind_all_class_valid = NULL;
int** ind_all_class_test = NULL;
int   size_all_class_train[NUM_label_KDD99];
//int   size_all_class_valid[NUM_label_KDD99];
int   size_all_class_test[NUM_label_KDD99];

int num_samples_KDD99_selected_train = 1000;
//int num_samples_selected_valid = 1000;
int num_samples_KDD99_selected_test = 1000;

#define VIOLATION_PENALTY_ID_C 1e6

int count_disc = 0;
int flagDisc[nCol_KDD99];

//////////////////////////////////////////////////////////////////////////
#define MAX_BUF_SIZE 10000 //
#define MAX_STR_LEN  256 //

static int** allocINT(int nrow, int ncol);
static MY_FLT_TYPE** allocFLOAT(int nrow, int ncol);
static void readData_KDD99(MY_FLT_TYPE** pDATA, char fname[], int nrow, int ncol);

//////////////////////////////////////////////////////////////////////////
FRNN_ID_C* frnn_id_c = NULL;
static void ff_IntrusionDetection_FRNN_Classify(double* individual,
        int num_samples_selected, int size_all_class[], int** ind_all_class, MY_FLT_TYPE** dataKDD99_cur,
        int& count_no_act, MY_FLT_TYPE& tmp_all_err, MY_FLT_TYPE& fire_lv_fules);

#define STATUS_USING_ALL_SAMPLES_MOP_INTRUSIONDETECTION FLAG_ON_MOP_INTRUSIONDETECTION

//////////////////////////////////////////////////////////////////////////
void Initialize_IntrusionDetection_FRNN_Classify(int curN, int numN)
{
    //
    seed_FRNN_MODEL = 237;
    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    for(int i = 0; i < curN; i++) {
        seed_FRNN_MODEL = (seed_FRNN_MODEL + 111) % 1235;
        rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;
    }
    // initialize
    num_input_KDD99 = 0;
    num_output_KDD99 = NUM_label_KDD99;
    // pre-process
    for(int i = 0; i < nCol_KDD99; i++) {
        flag_use_this_feature[i] = 1;
    }
    //
    char fname[MAX_STR_LEN];
    for(int i = 0; i < nCol_KDD99; i++) {
        max_feature_KDD99[i] = -FLT_MAX;
        min_feature_KDD99[i] = FLT_MAX;
    }
#if TRAIN_DATA_NUM_MOP_INTRUSIONDETECTION == FULL_TRAIN_DATA_MOP_INTRUSIONDETECTION
    sprintf(fname, "../Data_all/Data_IntrusionDetection/KDDCup99/kddcup.data.corrected");
#else
    sprintf(fname, "../Data_all/Data_IntrusionDetection/KDDCup99/kddcup.data_10_percent_corrected");
#endif
    dataKDD99_train = allocFLOAT(nRow_KDD99_train, nCol_KDD99);
    readData_KDD99(dataKDD99_train, fname, nRow_KDD99_train, nCol_KDD99);
    memcpy(min_feature_KDD99_train, min_feature_KDD99, nCol_KDD99 * sizeof(MY_FLT_TYPE));
    memcpy(max_feature_KDD99_train, max_feature_KDD99, nCol_KDD99 * sizeof(MY_FLT_TYPE));
    for(int i = 0; i < nCol_KDD99; i++) {
        max_feature_KDD99[i] = -FLT_MAX;
        min_feature_KDD99[i] = FLT_MAX;
    }
    //sprintf(fname, "../Data_all/Data_IntrusionDetection/KDDCup99/kddcup.data_10_percent_corrected");
    //dataKDD99_valid = allocFLOAT(nRow_KDD99_valid, nCol_KDD99);
    //readData_KDD99(dataKDD99_valid, fname, nRow_KDD99_valid, nCol_KDD99);
    //memcpy(min_feature_KDD99_valid, min_feature_KDD99, nCol_KDD99 * sizeof(float));
    //memcpy(max_feature_KDD99_valid, max_feature_KDD99, nCol_KDD99 * sizeof(float));
    //for(int i = 0; i < nCol_KDD99; i++) {
    //    max_feature_KDD99[i] = -FLT_MAX;
    //    min_feature_KDD99[i] = FLT_MAX;
    //}
    sprintf(fname, "../Data_all/Data_IntrusionDetection/KDDCup99/corrected");
    dataKDD99_test = allocFLOAT(nRow_KDD99_test, nCol_KDD99);
    readData_KDD99(dataKDD99_test, fname, nRow_KDD99_test, nCol_KDD99);
    memcpy(min_feature_KDD99_test, min_feature_KDD99, nCol_KDD99 * sizeof(MY_FLT_TYPE));
    memcpy(max_feature_KDD99_test, max_feature_KDD99, nCol_KDD99 * sizeof(MY_FLT_TYPE));
    //
    ind_all_class_train = allocINT(NUM_label_KDD99, nRow_KDD99_train);
    //ind_all_class_valid = allocINT(NUM_label_KDD99, nRow_KDD99_valid);
    ind_all_class_test = allocINT(NUM_label_KDD99, nRow_KDD99_test);
    for(int i = 0; i < NUM_label_KDD99; i++) {
        size_all_class_train[i] = 0;
        //size_all_class_valid[i] = 0;
        size_all_class_test[i] = 0;
    }
    for(int i = 0; i < nRow_KDD99_train; i++) {
        int cur_label = (int)(dataKDD99_train[i][IND_label_KDD99]);
        ind_all_class_train[cur_label][size_all_class_train[cur_label]] = i;
        size_all_class_train[cur_label]++;
    }
    //for(int i = 0; i < nRow_KDD99_valid; i++) {
    //    int cur_label = dataKDD99_valid[i][IND_label_KDD99];
    //    ind_all_class_valid[cur_label][size_all_class_valid[cur_label]] = i;
    //    size_all_class_valid[cur_label]++;
    //}
    for(int i = 0; i < nRow_KDD99_test; i++) {
        int cur_label = (int)(dataKDD99_test[i][IND_label_KDD99]);
        ind_all_class_test[cur_label][size_all_class_test[cur_label]] = i;
        size_all_class_test[cur_label]++;
    }
    //
    flag_use_this_feature[2] = 0; // the feature, service, is not used.
    flag_use_this_feature[IND_label_KDD99] = 0;
    flag_use_this_feature[19] = 0;
    flag_use_this_feature[20] = 0;
    for(int i = 0; i < nCol_KDD99; i++) {
        if(max_feature_KDD99_train[i] == min_feature_KDD99_train[i]) {  // only one value, not used.
            flag_use_this_feature[i] = 0;
        }
        if(flag_use_this_feature[i]) num_input_KDD99++;
    }
    //
#if STATUS_OUTPUT_MOP_INTRUSIONDETECTION == FLAG_ON_MOP_INTRUSIONDETECTION
    char tmp_out_fnm[1024];
    FILE* fpt_out = NULL;
    int out_flag;
    sprintf(tmp_out_fnm, "KDD99_train_data.csv");
    fpt_out = fopen(tmp_out_fnm, "w");
    for(int i = 0; i < nRow_KDD99_train; i++) {
        out_flag = 0;
        for(int j = 0; j < nCol_KDD99; j++) {
            if(flag_use_this_feature[j]) {
                if(out_flag) {
                    fprintf(fpt_out, ",");
                }
                fprintf(fpt_out, "%f", dataKDD99_train[i][j]);
                out_flag++;
            }
            if(j == nCol_KDD99 - 1) {
                fprintf(fpt_out, "\n");
            }
        }
    }
    fclose(fpt_out);
    sprintf(tmp_out_fnm, "KDD99_train_label.csv");
    fpt_out = fopen(tmp_out_fnm, "w");
    out_flag = 0;
    for(int i = 0; i < nRow_KDD99_train; i++) {
        fprintf(fpt_out, "%d\n", (int)dataKDD99_train[i][nCol_KDD99 - 1]);
    }
    fclose(fpt_out);
    //
    sprintf(tmp_out_fnm, "KDD99_test_data.csv");
    fpt_out = fopen(tmp_out_fnm, "w");
    for(int i = 0; i < nRow_KDD99_test; i++) {
        out_flag = 0;
        for(int j = 0; j < nCol_KDD99; j++) {
            if(flag_use_this_feature[j]) {
                if(out_flag) {
                    fprintf(fpt_out, ",");
                }
                fprintf(fpt_out, "%f", dataKDD99_test[i][j]);
                out_flag++;
            }
            if(j == nCol_KDD99 - 1) {
                fprintf(fpt_out, "\n");
            }
        }
    }
    fclose(fpt_out);
    sprintf(tmp_out_fnm, "KDD99_test_label.csv");
    fpt_out = fopen(tmp_out_fnm, "w");
    out_flag = 0;
    for(int i = 0; i < nRow_KDD99_test; i++) {
        fprintf(fpt_out, "%d\n", (int)dataKDD99_test[i][nCol_KDD99 - 1]);
    }
    fclose(fpt_out);
#endif
    //
    //for(int i = 0; i < nCol_KDD99_train; i++) {
    //    if(min_feature_KDD99_train[i] != min_feature_KDD99_test[i] ||
    //       max_feature_KDD99_train[i] != max_feature_KDD99_test[i]) {
    //        printf("Value range is not consistent for feature %d, continuous %d, used %d, train [%f, %f] - test [%f, %f]\n",
    //               i, flag_continuous_KDD99[i], flag_use_this_feature[i],
    //               min_feature_KDD99_train[i], max_feature_KDD99_train[i],
    //               min_feature_KDD99_test[i], max_feature_KDD99_test[i]);
    //    }
    //}
    //for(int i = 0; i < nCol_KDD99; i++) {
    //    printf("feature %d, continuous %d, used %d, train [%f, %f] - test [%f, %f]\n",
    //           i + 1, flag_continuous_KDD99[i], flag_use_this_feature[i],
    //           min_feature_KDD99_train[i], max_feature_KDD99_train[i],
    //           min_feature_KDD99_test[i], max_feature_KDD99_test[i]);
    //}
    //printf("%f %f %f %f\n", dataKDD99_train[0][14], dataKDD99_train[1][14], dataKDD99_train[2][14], dataKDD99_train[3][14]);
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    frnn_id_c = (FRNN_ID_C*)calloc(1, sizeof(FRNN_ID_C));
    int numInput = num_input_KDD99;
    MY_FLT_TYPE inputMin[nCol_KDD99];
    MY_FLT_TYPE inputMax[nCol_KDD99];
    for(int i = 0, tmp_c = 0; i < nCol_KDD99; i++) {
        if(flag_use_this_feature[i]) {
            if(flag_continuous_KDD99[i]) {
                inputMin[tmp_c] = min_feature_KDD99_train[i];
                inputMax[tmp_c] = max_feature_KDD99_train[i];
            } else {
                inputMin[tmp_c] = min_feature_KDD99_train[i];
                inputMax[tmp_c] = (MY_FLT_TYPE)(max_feature_KDD99_train[i] + 1 - 1e-6);
            }
            tmp_c++;
        }
    }
    int numMemship[nCol_KDD99];
    for(int i = 0, tmp_c = 0; i < nCol_KDD99; i++) {
        if(flag_use_this_feature[i]) {
            if(flag_continuous_KDD99[i])
                numMemship[tmp_c++] = DEFAULT_MEMFUNC_NUM_FRNN_MODEL; // DEFAULT_MEMFUNC_NUM_FRNN_MODEL;
            else
                numMemship[tmp_c++] = (int)max_feature_KDD99_train[i] + 1;
        }
    }
    count_disc = 0;
    int flagAdapMemship[nCol_KDD99];
    for(int i = 0, tmp_c = 0; i < nCol_KDD99; i++) {
        if(flag_use_this_feature[i]) {
            if(flag_continuous_KDD99[i]) {
                flagAdapMemship[tmp_c] = 1;
                flagDisc[tmp_c] = 0;
                tmp_c++;
            } else {
                flagAdapMemship[tmp_c] = 0;
                flagDisc[tmp_c] = 1;
                tmp_c++;
                count_disc++;
            }
        }
    }
    int numOutput = num_output_KDD99;
    MY_FLT_TYPE outputMin[NUM_label_KDD99];
    MY_FLT_TYPE outputMax[NUM_label_KDD99];
    for(int i = 0; i < numOutput; i++) {
        outputMin[i] = 0;
        outputMax[i] = 1;
    }
    int typeFuzzySet = FUZZY_INTERVAL_TYPE_II;
    int typeRules = PRODUCT_INFERENCE_ENGINE;
    int typeInRuleCorNum = ONE_EACH_IN_TO_ONE_RULE;
    int typeTypeReducer = NIE_TAN_TYPE_REDUCER;// CENTER_OF_SETS_TYPE_REDUCER;
    int numFuzzyRules = DEFAULT_FUZZY_RULE_NUM_FRNN_MODEL;
    int numRoughSets = (int)sqrt(numFuzzyRules);
    int consequenceNodeStatus = ADAPTIVE_CONSEQUENCE_CENTROID;
    int centroid_num_tag = CENTROID_ALL_ONESET;
#if LINEAR_CONVERSION_DISC_MOP_INTRUSIONDETECTION == LINEAR_CONVERSION_INC_DISC_MOP_INTRUSIONDETECTION
    int numInputConsequenceNode = numInput;
#else
    int numInputConsequenceNode = numInput - count_disc;
#endif
    int flagConnectStatus = FLAG_STATUS_OFF;
    int flagConnectWeight = FLAG_STATUS_ON;
    frnn_id_c_setup(frnn_id_c, numInput, inputMin, inputMax, numMemship, flagAdapMemship,
                    numOutput, outputMin, outputMax, typeFuzzySet, typeRules,
                    typeInRuleCorNum, typeTypeReducer, numFuzzyRules, numRoughSets,
                    consequenceNodeStatus, centroid_num_tag, numInputConsequenceNode,
                    flagConnectStatus, flagConnectWeight);
    //
    NDIM_IntrusionDetection_FRNN_Classify =
        frnn_id_c->M1->numParaLocal +
        frnn_id_c->F2->numParaLocal +
        frnn_id_c->R3->numParaLocal +
        frnn_id_c->O4->numParaLocal;
#if FUZZY_RULE_OBJ_STATUS_MOP_INTRUSIONDETECTION == FUZZY_RULE_OBJ_ON_MOP_INTRUSIONDETECTION
    NOBJ_IntrusionDetection_FRNN_Classify = 3;
#else
    NOBJ_IntrusionDetection_FRNN_Classify = 2;
#endif
    //
    return;
}

void SetLimits_IntrusionDetection_FRNN_Classify(double* minLimit, double* maxLimit, int nx)
{
    int count = 0;
    for(int i = 0; i < frnn_id_c->M1->numParaLocal; i++) {
        minLimit[count] = frnn_id_c->M1->xMin[i];
        maxLimit[count] = frnn_id_c->M1->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_id_c->F2->numParaLocal; i++) {
        minLimit[count] = frnn_id_c->F2->xMin[i];
        maxLimit[count] = frnn_id_c->F2->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_id_c->R3->numParaLocal; i++) {
        minLimit[count] = frnn_id_c->R3->xMin[i];
        maxLimit[count] = frnn_id_c->R3->xMax[i];
        count++;
    }
    for(int i = 0; i < frnn_id_c->O4->numParaLocal; i++) {
        minLimit[count] = frnn_id_c->O4->xMin[i];
        maxLimit[count] = frnn_id_c->O4->xMax[i];
        count++;
    }
    return;
}

int CheckLimits_IntrusionDetection_FRNN_Classify(double* x, int nx)
{
    int count = 0;
    for(int i = 0; i < frnn_id_c->M1->numParaLocal; i++) {
        if(x[count] < frnn_id_c->M1->xMin[i] ||
           x[count] > frnn_id_c->M1->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->M1 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_id_c->M1->xMin[i], frnn_id_c->M1->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_id_c->F2->numParaLocal; i++) {
        if(x[count] < frnn_id_c->F2->xMin[i] ||
           x[count] > frnn_id_c->F2->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->F2 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_id_c->F2->xMin[i], frnn_id_c->F2->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_id_c->R3->numParaLocal; i++) {
        if(x[count] < frnn_id_c->R3->xMin[i] ||
           x[count] > frnn_id_c->R3->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->R3 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_id_c->R3->xMin[i], frnn_id_c->R3->xMax[i]);
            return 0;
        }
        count++;
    }
    for(int i = 0; i < frnn_id_c->O4->numParaLocal; i++) {
        if(x[count] < frnn_id_c->O4->xMin[i] ||
           x[count] > frnn_id_c->O4->xMax[i]) {
            printf("%s(%d): Check limits FAIL - IntrusionDetection_FRNN_Classify: frnn_id_c->O4 %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[count], frnn_id_c->O4->xMin[i], frnn_id_c->O4->xMax[i]);
            return 0;
        }
        count++;
    }
    return 1;
}

void Fitness_IntrusionDetection_FRNN_Classify(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    int count_no_act;
    MY_FLT_TYPE tmp_all_err;
    MY_FLT_TYPE fire_lv_rules;
    ff_IntrusionDetection_FRNN_Classify(individual,
                                        num_samples_KDD99_selected_train, size_all_class_train, ind_all_class_train, dataKDD99_train,
                                        count_no_act, tmp_all_err, fire_lv_rules);
    //
    MY_FLT_TYPE sum_precision = frnn_id_c->sum_wrong / frnn_id_c->sum_all;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE tmp_precision[NUM_label_KDD99];
    MY_FLT_TYPE tmp_recall[NUM_label_KDD99];
    MY_FLT_TYPE tmp_Fvalue[NUM_label_KDD99];
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < NUM_label_KDD99; i++) {
        if(frnn_id_c->N_TP[i] > 0) {
            tmp_precision[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FP[i]);
            tmp_recall[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FN[i]);
            tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
                            (tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
        } else {
            tmp_precision[i] = 0;
            tmp_recall[i] = 0;
            tmp_Fvalue[i] = 0;
        }
        mean_precision += tmp_precision[i];
        mean_recall += tmp_recall[i];
        mean_Fvalue += tmp_Fvalue[i];
#if STATUS_OUT_INDEICES_MOP_INTRUSIONDETECTION == FLAG_ON_MOP_INTRUSIONDETECTION
        printf("%f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i]);
#endif
    }
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < NUM_label_KDD99; i++) {
        cur_dataflow += frnn_id_c->O4->dataflowStatus[i];
        if(frnn_id_c->O4->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE rule_Complexity = 0;
    for(int i = 0; i < frnn_id_c->F2->numRules; i++) {
        MY_FLT_TYPE tmp_rc = 0;
        for(int j = 0; j < frnn_id_c->F2->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < frnn_id_c->F2->numMembershipFun[j]; k++) {
                if(frnn_id_c->F2->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp_rc += ac_flag;
        }
        rule_Complexity += tmp_rc / frnn_id_c->F2->numInput;
    }
    rule_Complexity /= frnn_id_c->F2->numRules;
    //
    MY_FLT_TYPE count_connections = 0;
    for(int i = 0; i < frnn_id_c->F2->numRules; i++) {
        for(int j = 0; j < frnn_id_c->F2->numInput; j++) {
            for(int k = 0; k < frnn_id_c->F2->numMembershipFun[j]; k++) {
                if(frnn_id_c->F2->connectStatusAll[i][j][k])
                    count_connections++;
            }
        }
    }
    for(int i = 0; i < frnn_id_c->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_id_c->R3->numInput; j++) {
            if(frnn_id_c->R3->connectStatus[i][j])
                count_connections++;
        }
    }
    for(int i = 0; i < frnn_id_c->O4->numOutput; i++) {
        for(int j = 0; j < frnn_id_c->O4->numInput; j++) {
            if(frnn_id_c->O4->connectStatus[i][j])
                count_connections++;
        }
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_ID_C + count_no_act * 100);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
#if FUZZY_RULE_OBJ_STATUS_MOP_INTRUSIONDETECTION == FUZZY_RULE_OBJ_ON_MOP_INTRUSIONDETECTION
    fitness[0] = tmp_all_err + val_violation;
    fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION
    fitness[2] = cur_dataflow / frnn_id_c->dataflowMax + val_violation;
#else
    fitness[2] = count_connections / frnn_id_c->connectionMax + val_violation;
#endif
#else
    fitness[0] = tmp_all_err + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION
    fitness[1] = cur_dataflow / frnn_id_c->dataflowMax + val_violation;
#else
    fitness[1] = count_connections / frnn_id_c->connectionMax + val_violation;
#endif
#endif
    return;
}

//void Fitness_IntrusionDetection_FRNN_Classify_valid(double * individual, double * fitness)
//{
//    ff_IntrusionDetection_FRNN_Classify(individual,
//                                        num_samples_selected_valid, size_all_class_valid, ind_all_class_valid, dataKDD99_valid);
//    //
//    float sum_precision = frnn_id_c->sum_wrong / frnn_id_c->sum_all;
//    float mean_precision = 0;
//    float mean_recall = 0;
//    float tmp_precision[NUM_label_KDD99];
//    float tmp_recall[NUM_label_KDD99];
//    for(int i = 0; i < NUM_label_KDD99; i++) {
//        if(frnn_id_c->N_TP[i] > 0) {
//            tmp_precision[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FP[i]);
//            tmp_recall[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FN[i]);
//        } else {
//            tmp_precision[i] = 0;
//            tmp_recall[i] = 0;
//        }
//        mean_precision += tmp_precision[i];
//        mean_recall += tmp_recall[i];
//    }
//    //
//    int count_violation = 0;
//    int cur_dataflow = 0;
//    for(int i = 0; i < NUM_label_KDD99; i++) {
//        cur_dataflow += frnn_id_c->O4->dataflowStatus[i];
//        if(frnn_id_c->O4->dataflowStatus[i] == 0) count_violation++;
//    }
//    //
//    fitness[0] = 1 - mean_precision / NUM_label_KDD99 + count_violation * VIOLATION_PENALTY_ID_C;
//    fitness[1] = 1 - mean_recall / NUM_label_KDD99 + count_violation * VIOLATION_PENALTY_ID_C;
//    fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + count_violation * VIOLATION_PENALTY_ID_C;
//    return;
//}

void Fitness_IntrusionDetection_FRNN_Classify_test(double* individual, double* fitness)
{
    int count_no_act;
    MY_FLT_TYPE tmp_all_err;
    MY_FLT_TYPE fire_lv_rules;
    ff_IntrusionDetection_FRNN_Classify(individual,
                                        num_samples_KDD99_selected_test, size_all_class_test, ind_all_class_test, dataKDD99_test,
                                        count_no_act, tmp_all_err, fire_lv_rules);
    //
    MY_FLT_TYPE sum_precision = frnn_id_c->sum_wrong / frnn_id_c->sum_all;
    MY_FLT_TYPE mean_precision = 0;
    MY_FLT_TYPE mean_recall = 0;
    MY_FLT_TYPE mean_Fvalue = 0;
    MY_FLT_TYPE tmp_precision[NUM_label_KDD99];
    MY_FLT_TYPE tmp_recall[NUM_label_KDD99];
    MY_FLT_TYPE tmp_Fvalue[NUM_label_KDD99];
    MY_FLT_TYPE tmp_beta = 1;
    for(int i = 0; i < NUM_label_KDD99; i++) {
        if(frnn_id_c->N_TP[i] > 0) {
            tmp_precision[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FP[i]);
            tmp_recall[i] = frnn_id_c->N_TP[i] / (frnn_id_c->N_TP[i] + frnn_id_c->N_FN[i]);
            tmp_Fvalue[i] = (1 + tmp_beta * tmp_beta) * tmp_recall[i] * tmp_precision[i] /
                            (tmp_beta * tmp_beta * (tmp_recall[i] + tmp_precision[i]));
        } else {
            tmp_precision[i] = 0;
            tmp_recall[i] = 0;
            tmp_Fvalue[i] = 0;
        }
        mean_precision += tmp_precision[i];
        mean_recall += tmp_recall[i];
        mean_Fvalue += tmp_Fvalue[i];
#if STATUS_OUT_INDEICES_MOP_INTRUSIONDETECTION == FLAG_ON_MOP_INTRUSIONDETECTION
        printf("%f %f %f\n", tmp_precision[i], tmp_recall[i], tmp_Fvalue[i]);
#endif
    }
    //
    MY_FLT_TYPE count_violation = 0;
    MY_FLT_TYPE cur_dataflow = 0;
    for(int i = 0; i < NUM_label_KDD99; i++) {
        cur_dataflow += frnn_id_c->O4->dataflowStatus[i];
        if(frnn_id_c->O4->dataflowStatus[i] == 0) count_violation++;
    }
    //
    MY_FLT_TYPE rule_Complexity = 0;
    for(int i = 0; i < frnn_id_c->F2->numRules; i++) {
        MY_FLT_TYPE tmp_rc = 0;
        for(int j = 0; j < frnn_id_c->F2->numInput; j++) {
            int ac_flag = 0;
            for(int k = 0; k < frnn_id_c->F2->numMembershipFun[j]; k++) {
                if(frnn_id_c->F2->connectStatusAll[i][j][k]) {
                    ac_flag = 1;
                }
            }
            tmp_rc += ac_flag;
        }
        rule_Complexity += tmp_rc / frnn_id_c->F2->numInput;
    }
    rule_Complexity /= frnn_id_c->F2->numRules;
    //
    MY_FLT_TYPE count_connections = 0;
    for(int i = 0; i < frnn_id_c->F2->numRules; i++) {
        for(int j = 0; j < frnn_id_c->F2->numInput; j++) {
            for(int k = 0; k < frnn_id_c->F2->numMembershipFun[j]; k++) {
                if(frnn_id_c->F2->connectStatusAll[i][j][k])
                    count_connections++;
            }
        }
    }
    for(int i = 0; i < frnn_id_c->R3->numRoughSets; i++) {
        for(int j = 0; j < frnn_id_c->R3->numInput; j++) {
            if(frnn_id_c->R3->connectStatus[i][j])
                count_connections++;
        }
    }
    for(int i = 0; i < frnn_id_c->O4->numOutput; i++) {
        for(int j = 0; j < frnn_id_c->O4->numInput; j++) {
            if(frnn_id_c->O4->connectStatus[i][j])
                count_connections++;
        }
    }
    //
    MY_FLT_TYPE val_violation = (MY_FLT_TYPE)(count_violation * VIOLATION_PENALTY_ID_C + count_no_act * 100);
    //fitness[0] = 1 - mean_precision / NUM_label_KDD99 + val_violation;
    //fitness[1] = 1 - mean_recall / NUM_label_KDD99 + val_violation;
    //fitness[2] = cur_dataflow / (frnn_id_c->dataflowMax + 0.0) + val_violation;
#if FUZZY_RULE_OBJ_STATUS_MOP_INTRUSIONDETECTION == FUZZY_RULE_OBJ_ON_MOP_INTRUSIONDETECTION
    fitness[0] = tmp_all_err + val_violation;
    fitness[1] = rule_Complexity; //1 - fire_lv_rules + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION
    fitness[2] = cur_dataflow / frnn_id_c->dataflowMax + val_violation;
#else
    fitness[2] = count_connections / frnn_id_c->connectionMax + val_violation;
#endif
#else
    fitness[0] = tmp_all_err + val_violation;
#if NETWORK_SIMPLICITY_OBJ_CHOICE_MOP_INTRUSIONDETECTION == NETWORK_SIMPLICITY_DATAFLOW_MOP_INTRUSIONDETECTION
    fitness[1] = cur_dataflow / frnn_id_c->dataflowMax + val_violation;
#else
    fitness[1] = count_connections / frnn_id_c->connectionMax + val_violation;
#endif
#endif
    return;
}

static void ff_IntrusionDetection_FRNN_Classify(double* individual,
        int num_samples_selected, int size_all_class[], int** ind_all_class, MY_FLT_TYPE** dataKDD99_cur,
        int& count_no_act, MY_FLT_TYPE& tmp_all_err, MY_FLT_TYPE& fire_lv_fules)
{
    count_no_act = 0;
    tmp_all_err = 0;
    fire_lv_fules = 0;
    frnn_id_c->sum_all = 0;
    frnn_id_c->sum_wrong = 0;
    for(int i = 0; i < frnn_id_c->numOutput; i++) {
        frnn_id_c->N_sum[i] = 0;
        frnn_id_c->N_wrong[i] = 0;
        frnn_id_c->e_sum[i] = 0;
        frnn_id_c->N_TP[i] = 0;
        frnn_id_c->N_TN[i] = 0;
        frnn_id_c->N_FP[i] = 0;
        frnn_id_c->N_FN[i] = 0;
    }
    frnn_id_c_init(frnn_id_c, individual, ASSIGN_MODE_FRNN);
#if STATUS_OUT_ASSOCIATION_MEM_MOP_INTRUSIONDETECTION == FLAG_ON_MOP_INTRUSIONDETECTION

#endif
    MY_FLT_TYPE valIn[nCol_KDD99];
    MY_FLT_TYPE valOut[NUM_label_KDD99];
    MY_FLT_TYPE** paraIn = allocFLOAT(frnn_id_c->numOutput, frnn_id_c->O4->numInputConsequenceNode);
    //for(int i = 0; i < nRow_KDD99_test; i++) {
    for(int m = 0; m < NUM_label_KDD99; m++) {
#if STATUS_USING_ALL_SAMPLES_MOP_INTRUSIONDETECTION == FLAG_OFF_MOP_INTRUSIONDETECTION
        for(int n = 0; n < num_samples_selected && n < size_all_class[m]; n++) {
#else
        for(int n = 0; n < size_all_class[m]; n++) {
#endif
            int tmp_ind = (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (size_all_class[m] - n - 1e-6));
            int i = ind_all_class[m][tmp_ind];
            ind_all_class[m][tmp_ind] = ind_all_class[m][size_all_class[m] - n - 1];
            ind_all_class[m][size_all_class[m] - n - 1] = i;
            for(int j = 0, cur_c = 0; j < nCol_KDD99; j++) {
                if(flag_use_this_feature[j]) {
                    valIn[cur_c++] = dataKDD99_cur[i][j];
                }
            }
            for(int j = 0; j < frnn_id_c->numOutput; j++) {
                for(int k = 0, tmp_c1 = 0, tmp_c2 = 0; k < nCol_KDD99; k++) {
                    if(flag_use_this_feature[k]) {
#if LINEAR_CONVERSION_DISC_MOP_INTRUSIONDETECTION == LINEAR_CONVERSION_EXC_DISC_MOP_INTRUSIONDETECTION
                        if(!flagDisc[tmp_c1])
#endif
                        {
                            paraIn[j][tmp_c2] = valIn[tmp_c1];
                            //paraIn[j][tmp_c2] = (valIn[tmp_c1] - frnn_id_c->M1->inputMin[tmp_c1]) /
                            //                    (frnn_id_c->M1->inputMax[tmp_c1] - frnn_id_c->M1->inputMin[tmp_c1]);
                            //if(/*paraIn[j][tmp_c] < 0 || paraIn[j][tmp_c] > 1*/1) {
                            //    printf("%d %f (%f %f %f)\n", k, paraIn[j][tmp_c],
                            //           frnn_id_c->M1->inputMin[k], valIn[k], frnn_id_c->M1->inputMax[k]);
                            //}
                            tmp_c2++;
                        }
                        tmp_c1++;
                    }
                }
            }
            ff_frnn_id_c(frnn_id_c, valIn, valOut, paraIn);
            /*float tmpSum = 0;
            for(int j = 0; j < NUM_label_KDD99; j++) tmpSum += valOut[j];
            if(tmpSum > 0) {
            	printf("No. sample - %d - ", i);
            	for(int j = 0; j < NUM_label_KDD99; j++) printf("%e ", valOut[j]);
            	printf("\n");
            }*/
            int tmp_sum_ina = 0;
            int cur_label = 0;
            MY_FLT_TYPE cur_out = valOut[0];
            for(int j = 0; j < frnn_id_c->numOutput; j++) {
                if(cur_out < valOut[j]) {
                    cur_out = valOut[j];
                    cur_label = j;
                }
                if(valOut[j] == 0) {
                    tmp_sum_ina++;
                }
            }
            if(cur_label == -1) count_no_act++;
            int true_label = (int)dataKDD99_cur[i][IND_label_KDD99];
            for(int j = 0; j < frnn_id_c->numOutput; j++) {
                if(j == cur_label && j == true_label) frnn_id_c->N_TP[j]++;
                if(j == cur_label && j != true_label) frnn_id_c->N_FP[j]++;
                if(j != cur_label && j == true_label) frnn_id_c->N_FN[j]++;
                if(j != cur_label && j != true_label) frnn_id_c->N_TN[j]++;
            }
            frnn_id_c->sum_all++;
            frnn_id_c->N_sum[true_label]++;
            if(cur_label != true_label) {
                frnn_id_c->sum_wrong++;
                frnn_id_c->N_wrong[true_label]++;
            }
            //
            MY_FLT_TYPE softmax_outs[NUM_label_KDD99];
            MY_FLT_TYPE softmax_sum = 0;
            MY_FLT_TYPE softmax_degr[NUM_label_KDD99];
            for(int j = 0; j < frnn_id_c->numOutput; j++) {
                softmax_outs[j] = (MY_FLT_TYPE)(exp(valOut[j]));
                softmax_sum += softmax_outs[j];
            }
            if(softmax_sum > 0) {
                for(int j = 0; j < frnn_id_c->numOutput; j++) {
                    softmax_degr[j] = softmax_outs[j] / softmax_sum;
                }
                for(int j = 0; j < frnn_id_c->numOutput; j++) {
                    if(true_label == j) {
                        tmp_all_err += (1 - softmax_degr[j]) * (1 - softmax_degr[j]);
                    } else {
                        tmp_all_err += softmax_degr[j] * softmax_degr[j];
                    }
                }
            } else {
                for(int j = 0; j < frnn_id_c->numOutput; j++) {
                    tmp_all_err += 1;
                }
            }
            //
            MY_FLT_TYPE tmp_fire = 0;
            int tempN;
            if(frnn_id_c->typeFuzzySet == FUZZY_SET_I) tempN = 1;
            else tempN = 2;
            for(int j = 0; j < frnn_id_c->F2->numRules; j++) {
                for(int k = 0; k < tempN; k++) {
                    tmp_fire += frnn_id_c->F2->degreeRules[j][k];
                }
            }
            tmp_fire /= frnn_id_c->F2->numRules * tempN;
            fire_lv_fules += tmp_fire;
        }
    }
    //
    tmp_all_err /= frnn_id_c->sum_all;
    tmp_all_err /= frnn_id_c->numOutput;
    //
    fire_lv_fules /= frnn_id_c->sum_all;
    //}
    for(int i = 0; i < frnn_id_c->numOutput; i++) {
        free(paraIn[i]);
    }
    free(paraIn);
    //
    return;
}

void Finalize_IntrusionDetection_FRNN_Classify()
{
    for(int i = 0; i < nRow_KDD99_train; i++) free(dataKDD99_train[i]);
    for(int i = 0; i < nRow_KDD99_test; i++) free(dataKDD99_test[i]);
    free(dataKDD99_train);
    free(dataKDD99_test);
    for(int i = 0; i < NUM_label_KDD99; i++) free(ind_all_class_train[i]);
    for(int i = 0; i < NUM_label_KDD99; i++) free(ind_all_class_test[i]);
    free(ind_all_class_train);
    free(ind_all_class_test);
    frnn_id_c_free(frnn_id_c);
    return;
}

//////////////////////////////////////////////////////////////////////////
void frnn_id_c_setup(FRNN_ID_C* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                     int* flagAdapMemship,
                     int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
                     int typeFuzzySet, int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
                     int consequenceNodeStatus, int centroid_num_tag, int numInputConsequenceNode,
                     int flagConnectStatus, int flagConnectWeight)
{
    frnn->typeFuzzySet = typeFuzzySet;
    frnn->typeRules = typeRules;
    frnn->typeInRuleCorNum = typeInRuleCorNum;
    frnn->typeTypeReducer = typeTypeReducer;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
    frnn->flagConnectStatus = flagConnectStatus;
    frnn->flagConnectWeight = flagConnectWeight;

    frnn->layerNum = 4;
    frnn->numOutput = numOutput;

    int tmp_typeCoding = PARA_CODING_CANDECOMP_PARAFAC;

    frnn->M1 = setupMemberLayer(numInput, inputMin, inputMax, numMemship, flagAdapMemship, typeFuzzySet,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->F2 = setupFuzzyLayer(numInput, frnn->M1->numMembershipFun, numFuzzyRules, typeFuzzySet, typeRules, typeInRuleCorNum,
                               FLAG_STATUS_OFF,
                               PARA_CODING_DIRECT, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->R3 = setupRoughLayer(frnn->F2->numRules, numRoughSets, typeFuzzySet,
                               1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->O4 = setupOutReduceLayer(frnn->R3->numRoughSets, numOutput, outputMin, outputMax,
                                   typeFuzzySet, typeTypeReducer,
                                   consequenceNodeStatus, centroid_num_tag, numInputConsequenceNode, inputMin, inputMax, flagConnectStatus,
                                   flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);

    frnn->e = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_wrong = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->e_sum = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    frnn->N_TP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_TN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FP = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));
    frnn->N_FN = (MY_FLT_TYPE*)calloc(numOutput, sizeof(MY_FLT_TYPE));

    if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->numInput * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->O4->numOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->numInput * frnn->F2->numRules +
                                            frnn->F2->numRules * frnn->R3->numRoughSets +
                                            frnn->R3->numRoughSets * frnn->O4->numOutput);
    } else {
        frnn->dataflowMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->O4->numOutput);
        frnn->connectionMax = (MY_FLT_TYPE)(frnn->M1->outputSize * frnn->F2->numRules +
                                            frnn->F2->numRules * frnn->R3->numRoughSets +
                                            frnn->R3->numRoughSets * frnn->O4->numOutput);
    }

    return;
}

void frnn_id_c_free(FRNN_ID_C * frnn)
{
    freeMemberLayer(frnn->M1);
    freeFuzzyLayer(frnn->F2);
    freeRoughLayer(frnn->R3);
    freeOutReduceLayer(frnn->O4);

    free(frnn->e);

    free(frnn->N_sum);
    free(frnn->N_wrong);
    free(frnn->e_sum);

    free(frnn->N_TP);
    free(frnn->N_TN);
    free(frnn->N_FP);
    free(frnn->N_FN);

    free(frnn);

    return;
}

void frnn_id_c_init(FRNN_ID_C * frnn, double* x, int mode)
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

    assignMemberLayer(frnn->M1, &x[count], mode);
    count += frnn->M1->numParaLocal;
    memcpy(frnn->F2->numMembershipFunCur, frnn->M1->numMembershipFunCur, frnn->M1->numInput * sizeof(int));
    assignFuzzyLayer(frnn->F2, &x[count], mode);
    count += frnn->F2->numParaLocal;
    assignRoughLayer(frnn->R3, &x[count], mode);
    count += frnn->R3->numParaLocal;
    assignOutReduceLayer(frnn->O4, &x[count], mode);

    return;
}

void ff_frnn_id_c(FRNN_ID_C * frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode)
{
    MY_FLT_TYPE* dataflowStatus = (MY_FLT_TYPE*)calloc(frnn->M1->numInput, sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->M1->numInput; i++) dataflowStatus[i] = 1;
    ff_memberLayer(frnn->M1, valIn, dataflowStatus);
    free(dataflowStatus);
    ff_fuzzyLayer(frnn->F2, frnn->M1->degreeMembership, frnn->M1->dataflowStatus);
    ff_roughLayer(frnn->R3, frnn->F2->degreeRules, frnn->F2->dataflowStatus);
    if(frnn->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < frnn->O4->numOutput; i++)
            memcpy(frnn->O4->inputConsequenceNode[i], inputConsequenceNode[i], frnn->O4->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
    }
    ff_outReduceLayer(frnn->O4, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
    memcpy(valOut, frnn->O4->valOutputFinal, frnn->O4->numOutput * sizeof(MY_FLT_TYPE));

    return;
}

//////////////////////////////////////////////////////////////////////////
static int** allocINT(int nrow, int ncol)
{
    int** tmp = NULL;
    if((tmp = (int**)malloc(nrow * sizeof(int*))) == NULL) {
        printf("%s(%d): ERROR!! --> malloc: no memory for matrix*\n", __FILE__, __LINE__);
        exit(-123320);
    } else {
        for(int i = 0; i < nrow; i++) {
            if((tmp[i] = (int*)malloc(ncol * sizeof(int))) == NULL) {
                printf("%s(%d): ERROR!! --> malloc: no memory for vector\n", __FILE__, __LINE__);
                exit(-123323);
            }
        }
    }
    return tmp;
}

static MY_FLT_TYPE** allocFLOAT(int nrow, int ncol)
{
    MY_FLT_TYPE** tmp = NULL;
    if((tmp = (MY_FLT_TYPE**)malloc(nrow * sizeof(MY_FLT_TYPE*))) == NULL) {
        printf("%s(%d): ERROR!! --> malloc: no memory for matrix*\n", __FILE__, __LINE__);
        exit(-123321);
    } else {
        for(int i = 0; i < nrow; i++) {
            if((tmp[i] = (MY_FLT_TYPE*)malloc(ncol * sizeof(MY_FLT_TYPE))) == NULL) {
                printf("%s(%d): ERROR!! --> malloc: no memory for vector\n", __FILE__, __LINE__);
                exit(-123322);
            }
        }
    }
    return tmp;
}

static void readData_KDD99(MY_FLT_TYPE** pDATA, char fname[], int nrow, int ncol)
{
    FILE* fpt = fopen(fname, "r");
    if(fpt) {
        for(int i = 0; i < NUM_label_KDD99; i++) {
            for(int j = 0; j < LEN_label_class[i]; j++) {
                count_label_KDD99[i][j] = 0;
            }
        }
        for(int i = 0; i < NUM_protocol_type_KDD99; i++) {
            count_protocol_type_KDD99[i] = 0;
        }
        for(int i = 0; i < NUM_service_KDD99; i++) {
            count_service_KDD99[i] = 0;
        }
        for(int i = 0; i < NUM_flag_KDD99; i++) {
            count_flag_KDD99[i] = 0;
        }
        char buf[MAX_BUF_SIZE];
        char* p;
        for(int i = 0; i < nrow; i++) {
            int cur_c = 0;
            if(fgets(buf, MAX_BUF_SIZE, fpt)) {
                for(p = strtok(buf, " ,\t\r\n"); p; p = strtok(NULL, " ,\t\r\n")) {
                    int tmp_flag = 0;
                    double tmp_val;
                    if(cur_c >= ncol) {
                        printf("%s(%d): More data items for this row - %d (%d) - (%s), exiting...\n",
                               __FILE__, __LINE__, i, cur_c, p);
                        exit(-111000);
                    }
                    if(cur_c == IND_protocol_type_KDD99) {
                        for(int j = 0; j < NUM_protocol_type_KDD99; j++) {
                            if(!strcmp(p, protocol_type_KDD99[j])) {
                                count_protocol_type_KDD99[j]++;
                                tmp_flag = 1;
                                tmp_val = (MY_FLT_TYPE)j;
                                break;
                            }
                        }
                        if(!tmp_flag) {
                            printf("%s(%d): Unknown protocol_type_KDD99 - %s, exiting...\n", __FILE__, __LINE__, p);
                            exit(-111001);
                        }
                    } else if(cur_c == IND_service_KDD99) { // not used
                        int tmp_flag = 0;
                        for(int j = 0; j < NUM_service_KDD99; j++) {
                            if(!strcmp(p, service_KDD99[j])) {
                                count_service_KDD99[j]++;
                                tmp_flag = 1;
                                tmp_val = (MY_FLT_TYPE)j;
                                break;
                            }
                        }
                        if(!tmp_flag) {
                            printf("%s(%d): Unknown service_KDD99 - %s, exiting...\n", __FILE__, __LINE__, p);
                            exit(-111002);
                        }
                    } else if(cur_c == IND_flag_KDD99) {
                        int tmp_flag = 0;
                        for(int j = 0; j < NUM_flag_KDD99; j++) {
                            if(!strcmp(p, flag_KDD99[j])) {
                                count_flag_KDD99[j]++;
                                tmp_flag = 1;
                                tmp_val = (MY_FLT_TYPE)j;
                                break;
                            }
                        }
                        if(!tmp_flag) {
                            printf("%s(%d): Unknown flag_KDD99 - %s, exiting...\n", __FILE__, __LINE__, p);
                            exit(-111003);
                        }
                    } else if(cur_c == IND_label_KDD99) {
                        int tmp_flag = 0;
                        for(int j = 0; j < NUM_label_KDD99; j++) {
                            for(int k = 0; k < LEN_label_class[j]; k++) {
                                if(!strcmp(p, label_KDD99[j][k])) {
                                    count_label_KDD99[j][k]++;
                                    tmp_flag = 1;
                                    tmp_val = (MY_FLT_TYPE)j;
                                    break;
                                }
                            }
                            if(tmp_flag) break;
                        }
                        if(!tmp_flag) {
                            printf("%s(%d): Unknown label_KDD99 - %s, exiting...\n", __FILE__, __LINE__, p);
                            exit(-111004);
                        }
                    } else {
                        sscanf(p, "%lf", &tmp_val);
                    }
                    if(tmp_val > max_feature_KDD99[cur_c]) max_feature_KDD99[cur_c] = tmp_val;
                    if(tmp_val < min_feature_KDD99[cur_c]) min_feature_KDD99[cur_c] = tmp_val;
                    //if((int)tmp_val != tmp_val) flag_continuous_KDD99[cur_c] = 1;
                    pDATA[i][cur_c] = tmp_val;
                    cur_c++;
                }
                if(cur_c != ncol) {
                    printf("%s(%d): Number of data items is not consistent for this row - %d - (%d), exiting...\n",
                           __FILE__, __LINE__, i, cur_c);
                    exit(-111005);
                }
            } else {
                printf("%s(%d): No more data, exiting...\n", __FILE__, __LINE__);
                exit(-111006);
            }
        }
        fclose(fpt);
    } else {
        printf("Open file %s error, exiting...\n", fname);
        exit(-111007);
    }
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////