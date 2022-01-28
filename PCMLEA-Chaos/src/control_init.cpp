#include "global.h"

#define PI 3.1415926535897932384626433832795

void get_alg_mech_func_id_to_test(const char* filename, int& the_alg_type, int& fun_start_num,
                                  int& fun_end_num)
{
    FILE* fp;
    fp = fopen(filename, "r");
    if(fp == NULL) {
        if(0 == st_MPI_p.mpi_rank)
            printf("%s:  could not open %s\n", AT, filename);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_READING);
    }
    char tmp_s[1024];
    while(fgets(tmp_s, sizeof(tmp_s), fp) != NULL) {
        /* skip comments  */
        if(tmp_s[0] == '#' || tmp_s[0] == '%')
            continue;
        char tmp_str_nm[1024];
        if(sscanf(tmp_s, "%s", tmp_str_nm) != 1) {
            if(0 == st_MPI_p.mpi_rank)
                printf("%s: read para error\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
        }
        if(!strcmp(tmp_str_nm, "algo_mech_type")) {
            char tmp_alg_m_t[128];
            if(sscanf(tmp_s, "%*s %s", tmp_alg_m_t) != 1) {
                if(0 == st_MPI_p.mpi_rank)
                    printf("%s: read para error\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
            }
            if(!strcmp(tmp_alg_m_t, "LOCALIZATION")) {
                the_alg_type = LOCALIZATION;
            } else if(!strcmp(tmp_alg_m_t, "DECOMPOSITION")) {
                the_alg_type = DECOMPOSITION;
            } else if(!strcmp(tmp_alg_m_t, "NONDOMINANCE")) {
                the_alg_type = NONDOMINANCE;
            } else {
                if(0 == st_MPI_p.mpi_rank)
                    printf("%s: No such algorithm mechanism type\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
            }
        }
        if(!strcmp(tmp_str_nm, "Nos_instances")) {
            if(sscanf(tmp_s, "%*s %d %d", &fun_start_num, &fun_end_num) != 2) {
                if(0 == st_MPI_p.mpi_rank)
                    printf("%s: read para error\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
            }
        }
    }
    fclose(fp);
}

void modify_num_run(const char* prob, int& num_run)
{
    if(!strncmp(prob, "FINANCE", 7)) {
        num_run = 5;
    }
    if(!strcmp(prob, "Classify_CNN_Indus") ||
       !strcmp(prob, "Classify_NN_Indus") ||
       !strcmp(prob, "Classify_CFRNN_Indus") ||
       !strcmp(prob, "Classify_CFRNN_MNIST") ||
       !strcmp(prob, "Classify_CFRNN_FashionMNIST") ||
       !strcmp(prob, "Classify_CFRNN_MNIST_FM") ||
       !strcmp(prob, "Classify_CFRNN_FashionMNIST_FM")) {
        num_run = 10;
    }
    if(!strcmp(prob, "ARRANGE2D")) {
        num_run = 10;
    }
    if(!strcmp(prob, "FeatureSelection_25") ||
       !strcmp(prob, "FeatureSelectionTREE_25")) {
        num_run = 10;
    }
    if(!strcmp(prob, "ActivityDetection_FRNN_Classify")) {
        num_run = 10;
    }
    if(!strcmp(prob, "RecSys_SmartCity")) {
        num_run = 5;
    }
    if(!strncmp(prob, "evoFRNN_Predict_", 16) ||
       !strncmp(prob, "evoGFRNN_Predict_", 17) ||
       !strncmp(prob, "evoDFRNN_Predict_", 17) ||
       !strncmp(prob, "evoFGRNN_Predict_", 17) ||
       !strncmp(prob, "evoDFGRNN_Predict_", 18) ||
       !strncmp(prob, "evoGFGRNN_Predict_", 18) ||
       !strncmp(prob, "evoBFRNN_Predict_", 17) ||
       !strncmp(prob, "evoBGFRNN_Predict_", 18) ||
       !strncmp(prob, "evoBDFRNN_Predict_", 18) ||
       !strncmp(prob, "evoBFGRNN_Predict_", 18) ||
       !strncmp(prob, "evoBDFGRNN_Predict_", 19) ||
       !strncmp(prob, "evoBGFGRNN_Predict_", 19)) {
        if(strstr(prob, "Classify_"))
            num_run = 10;
    }
    if(!strncmp(prob, "evoDeepFRNN_Predict_", 20) ||
       !strncmp(prob, "evoDeepGFRNN_Predict_", 21) ||
       !strncmp(prob, "evoDeepDFRNN_Predict_", 21) ||
       !strncmp(prob, "evoDeepFGRNN_Predict_", 21) ||
       !strncmp(prob, "evoDeepDFGRNN_Predict_", 22) ||
       !strncmp(prob, "evoDeepGFGRNN_Predict_", 22) ||
       !strncmp(prob, "evoDeepBFRNN_Predict_", 21) ||
       !strncmp(prob, "evoDeepBGFRNN_Predict_", 22) ||
       !strncmp(prob, "evoDeepBDFRNN_Predict_", 22) ||
       !strncmp(prob, "evoDeepBFGRNN_Predict_", 22) ||
       !strncmp(prob, "evoDeepBDFGRNN_Predict_", 23) ||
       !strncmp(prob, "evoDeepBGFGRNN_Predict_", 23)) {
        num_run = 10;
    }
    //
    return;
}

void modify_hyper_paras(const char* prob, int nobj, int ndim, int& NP, int& N_arch, int& maxIter,
                        int& the_type_test)
{
    switch(nobj) {
    case 2:
        N_arch = 100;
        NP = 100;
        break;
    case 3:
    case 4:
        N_arch = 120;
        NP = 120;
        break;
    case 5:
        N_arch = 126;//(5,0)
        NP = 126;
        break;
    case 6:
        N_arch = 126;//(4,0)
        NP = 126;
        break;
    case 10:
        N_arch = 275;//(3,2)
        NP = 275;
        break;
    default:
        if(st_MPI_p.mpi_rank == 0)
            printf("%s:Undefined number of objectives (%d is not 2, 3, 4, 5, or 10), exiting...\n", AT, nobj);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OBJ_TOOMANY);
        break;
    }
    //////////////////////////////////////////////////////////////////////////
    if(!strncmp(prob, "LSMOP", 5)) {
        //maxIter = nobj * 1000 * NP;
    } else if(!strcmp(prob, "RS")) {
        maxIter = 10000 * NP;
    } else if(!strcmp(prob, "HDSN_URBAN")) {
        //maxIter = (int)(ndim * 2000);
    } else if(!strcmp(prob, "IWSN_S_1F")) {
        //maxIter = (int)(ndim * 2e3);
    } else if(!strcmp(prob, "EdgeComputation")) {
        maxIter = (int)(1e5);
    } else if(!strncmp(prob, "FeatureSelection_", 17)) {
        maxIter = (int)6e4;
        if(!strcmp(prob, "FeatureSelection_25")) {
            maxIter = (int)2e3;
            N_arch = 51;
            NP = 51;
        }
        the_type_test = MY_TYPE_FS_CLASSIFY;
    } else if(!strncmp(prob, "FeatureSelectionTREE_", 21)) {
        maxIter = (int)6e4;
        if(!strcmp(prob, "FeatureSelectionTREE_25")) {
            maxIter = (int)2e3;
            N_arch = 51;
            NP = 51;
        }
        the_type_test = MY_TYPE_FS_CLASSIFY_TREE;
    } else if(!strncmp(prob, "FINANCE", 7)) {
        maxIter = (int)(ndim * 2e1);
        the_type_test = MY_TYPE_LeNet;
    } else if(!strncmp(prob, "ENSEMBLE_FINANCE", 16)) {
        maxIter = (int)(ndim * 1e4);
        the_type_test = MY_TYPE_LeNet_ENSEMBLE;
    } else if(!strcmp(prob, "Classify_CNN_Indus")) {
        maxIter = (int)1e4;// (int)(ndim * 2e1);
        the_type_test = MY_TYPE_LeNet_CLASSIFY_Indus;
    } else if(!strcmp(prob, "Classify_NN_Indus")) {
        maxIter = (int)1e4;// (int)(ndim * 2e1);
        the_type_test = MY_TYPE_NN_CLASSIFY_Indus;
    } else if(!strcmp(prob, "Classify_CFRNN_Indus") ||
              !strcmp(prob, "Classify_CFRNN_MNIST") ||
              !strcmp(prob, "Classify_CFRNN_FashionMNIST") ||
              !strcmp(prob, "Classify_CFRNN_MNIST_FM") ||
              !strcmp(prob, "Classify_CFRNN_FashionMNIST_FM")) {
        maxIter = (int)2e3;// (int)(ndim * 2e1);
        N_arch = 10;
        NP = 10;
        the_type_test = MY_TYPE_CFRNN_CLASSIFY;
    } else if(!strncmp(prob, "EVO1_FRNN", 9)) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO1_FRNN;
    } else if(!strncmp(prob, "EVO2_FRNN", 9)) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO2_FRNN;
    } else if(!strncmp(prob, "EVO3_FRNN", 9)) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO3_FRNN;
    } else if(!strncmp(prob, "EVO4_FRNN", 9)) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO4_FRNN;
    } else if(!strncmp(prob, "EVO5_FRNN", 9)) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO5_FRNN;
    } else if(!strncmp(prob, "evoFRNN_Predict_", 16) ||
              !strncmp(prob, "evoGFRNN_Predict_", 17) ||
              !strncmp(prob, "evoDFRNN_Predict_", 17) ||
              !strncmp(prob, "evoFGRNN_Predict_", 17) ||
              !strncmp(prob, "evoDFGRNN_Predict_", 18) ||
              !strncmp(prob, "evoGFGRNN_Predict_", 18) ||
              !strncmp(prob, "evoBFRNN_Predict_", 17) ||
              !strncmp(prob, "evoBGFRNN_Predict_", 18) ||
              !strncmp(prob, "evoBDFRNN_Predict_", 18) ||
              !strncmp(prob, "evoBFGRNN_Predict_", 18) ||
              !strncmp(prob, "evoBDFGRNN_Predict_", 19) ||
              !strncmp(prob, "evoBGFGRNN_Predict_", 19)) {
        maxIter = (int)(2e3);
        N_arch = 12;
        NP = 12;
        the_type_test = MY_TYPE_EVO_FRNN_PREDICT;
    } else if(!strncmp(prob, "evoDeepFRNN_Predict_", 20) ||
              !strncmp(prob, "evoDeepGFRNN_Predict_", 21) ||
              !strncmp(prob, "evoDeepDFRNN_Predict_", 21) ||
              !strncmp(prob, "evoDeepFGRNN_Predict_", 21) ||
              !strncmp(prob, "evoDeepDFGRNN_Predict_", 22) ||
              !strncmp(prob, "evoDeepGFGRNN_Predict_", 22) ||
              !strncmp(prob, "evoDeepBFRNN_Predict_", 21) ||
              !strncmp(prob, "evoDeepBGFRNN_Predict_", 22) ||
              !strncmp(prob, "evoDeepBDFRNN_Predict_", 22) ||
              !strncmp(prob, "evoDeepBFGRNN_Predict_", 22) ||
              !strncmp(prob, "evoDeepBDFGRNN_Predict_", 23) ||
              !strncmp(prob, "evoDeepBGFGRNN_Predict_", 23)) {
        maxIter = (int)(2e3);
        N_arch = 12;
        NP = 12;
        the_type_test = MY_TYPE_EVO_DFRNN_PREDICT;
    } else if(!strcmp(prob, "IntrusionDetection_FRNN_Classify")) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_INTRUSION_DETECTION_CLASSIFY;
    } else if(!strcmp(prob, "ActivityDetection_FRNN_Classify")) {
        maxIter = (int)(1e3);
        N_arch = 10;
        NP = 10;
        the_type_test = MY_TYPE_ACTIVITY_DETECTION_CLASSIFY;
    } else if(!strcmp(prob, "RecSys_SmartCity")) {
        maxIter = (int)(1e5);
        the_type_test = MY_TYPE_RecSys_SmartCity;
    } else if(!strcmp(prob, "evoCNN_MNIST_Classify")) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO_CNN;
    } else if(!strcmp(prob, "evoCFRNN_Classify")) {
        maxIter = (int)(1e6);
        the_type_test = MY_TYPE_EVO_CFRNN;
    } else if(strstr(prob, "evoMobileSink")) {
        maxIter = NP * 1000;
        if(strstr(prob, "GEP_only"))
            maxIter = NP * 1000;
        the_type_test = MY_TYPE_EVO_MOBILE_SINK;
    }
    //
    return;
}

void readParaFromFile()
{
    char tmp_s[1024];
    char filename[] = "./DATA_alg/para_file";
    FILE* fp;
    fp = fopen(filename, "r");
    if(fp == NULL) {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s:  could not open %s\n", AT, filename);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_READING);
        return;
    }
    while(fgets(tmp_s, sizeof(tmp_s), fp) != NULL) {
        /* skip comments  */
        if(tmp_s[0] == '#' || tmp_s[0] == '%')
            continue;
        char tmp_str_nm[1024];
        char tmp_str_val[1024];
        if(sscanf(tmp_s, "%s %s", tmp_str_nm, tmp_str_val) != 2) {
            continue;
        }
        if(!strcmp(tmp_str_nm, "optimizer_type")) {
            if(!strcmp(tmp_str_val, "EC_DE_CUR_1")) {
                st_ctrl_p.optimizer_type = EC_DE_CUR_1;
            } else if(!strcmp(tmp_str_val, "EC_DE_CUR_2")) {
                st_ctrl_p.optimizer_type = EC_DE_CUR_2;
            } else if(!strcmp(tmp_str_val, "EC_DE_RAND_1")) {
                st_ctrl_p.optimizer_type = EC_DE_RAND_1;
            } else if(!strcmp(tmp_str_val, "EC_DE_RAND_2")) {
                st_ctrl_p.optimizer_type = EC_DE_RAND_2;
            } else if(!strcmp(tmp_str_val, "EC_DE_ARCHIVE")) {
                st_ctrl_p.optimizer_type = EC_DE_ARCHIVE;
            } else if(!strcmp(tmp_str_val, "EC_DE_ARCHIVE_RAND")) {
                st_ctrl_p.optimizer_type = EC_DE_ARCHIVE_RAND;
            } else if(!strcmp(tmp_str_val, "EC_DE_2SELECTED")) {
                st_ctrl_p.optimizer_type = EC_DE_2SELECTED;
            } else if(!strcmp(tmp_str_val, "EC_SBX_CUR")) {
                st_ctrl_p.optimizer_type = EC_SBX_CUR;
            } else if(!strcmp(tmp_str_val, "EC_SBX_RAND")) {
                st_ctrl_p.optimizer_type = EC_SBX_RAND;
            } else if(!strcmp(tmp_str_val, "SI_PSO")) {
                st_ctrl_p.optimizer_type = SI_PSO;
            } else if(!strcmp(tmp_str_val, "SI_QPSO")) {
                st_ctrl_p.optimizer_type = SI_QPSO;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_R_SBX_R")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_R_SBX_R;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_C_SBX_R")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_C_SBX_R;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_C_SBX_C")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_C_SBX_C;
            } else if(!strcmp(tmp_str_val, "EC_MIX_SBX_C_R")) {
                st_ctrl_p.optimizer_type = EC_MIX_SBX_C_R;
            } else if(!strcmp(tmp_str_val, "EC_SI_MIX_DE_C_PSO")) {
                st_ctrl_p.optimizer_type = EC_SI_MIX_DE_C_PSO;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_C_R")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_C_R;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_C_1_2")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_C_1_2;
            } else if(!strcmp(tmp_str_val, "EC_MIX_DE_R_1_2")) {
                st_ctrl_p.optimizer_type = EC_MIX_DE_R_1_2;
            } else if(!strcmp(tmp_str_val, "OPTIMIZER_BLEND")) {
                st_ctrl_p.optimizer_type = OPTIMIZER_BLEND;
            } else if(!strcmp(tmp_str_val, "OPTIMIZER_ENSEMBLE")) {
                st_ctrl_p.optimizer_type = OPTIMIZER_ENSEMBLE;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "DE_F_type")) {
            if(!strcmp(tmp_str_val, "DE_F_DISC")) {
                st_ctrl_p.DE_F_type = DE_F_DISC;
            } else if(!strcmp(tmp_str_val, "DE_F_FIXED")) {
                st_ctrl_p.DE_F_type = DE_F_FIXED;
            } else if(!strcmp(tmp_str_val, "DE_F_JADE")) {
                st_ctrl_p.DE_F_type = DE_F_JADE;
            } else if(!strcmp(tmp_str_val, "DE_F_jDE")) {
                st_ctrl_p.DE_F_type = DE_F_jDE;
            } else if(!strcmp(tmp_str_val, "DE_F_NSDE")) {
                st_ctrl_p.DE_F_type = DE_F_NSDE;
            } else if(!strcmp(tmp_str_val, "DE_F_SaNSDE")) {
                st_ctrl_p.DE_F_type = DE_F_SaNSDE;
            } else if(!strcmp(tmp_str_val, "DE_F_SaNSDE_a")) {
                st_ctrl_p.DE_F_type = DE_F_SaNSDE_a;
            } else if(!strcmp(tmp_str_val, "DE_F_SHADE")) {
                st_ctrl_p.DE_F_type = DE_F_SHADE;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
            //////////////////////////////////////////////////////////////////////////
            //st_ctrl_p.F_para_limit_tag =
            //    FLAG_ON;//(0,1]
            //FLAG_OFF;//(0,2]
            if(st_ctrl_p.DE_F_type == DE_F_NSDE || st_ctrl_p.DE_F_type == DE_F_SaNSDE)
                st_ctrl_p.F_para_limit_tag = FLAG_OFF;
        } else if(!strcmp(tmp_str_nm, "DE_CR_type")) {
            if(!strcmp(tmp_str_val, "DE_CR_FIXED")) {
                st_ctrl_p.DE_CR_type = DE_CR_FIXED;
            } else if(!strcmp(tmp_str_val, "DE_CR_JADE")) {
                st_ctrl_p.DE_CR_type = DE_CR_JADE;
            } else if(!strcmp(tmp_str_val, "DE_CR_jDE")) {
                st_ctrl_p.DE_CR_type = DE_CR_jDE;
            } else if(!strcmp(tmp_str_val, "DE_CR_LINEAR")) {
                st_ctrl_p.DE_CR_type = DE_CR_LINEAR;
            } else if(!strcmp(tmp_str_val, "DE_CR_SHADE")) {
                st_ctrl_p.DE_CR_type = DE_CR_SHADE;
            } else if(!strcmp(tmp_str_val, "DE_CR_DISC")) {
                st_ctrl_p.DE_CR_type = DE_CR_DISC;
            } else if(!strcmp(tmp_str_val, "DE_CR_SaNSDE")) {
                st_ctrl_p.DE_CR_type = DE_CR_SaNSDE;
            } else if(!strcmp(tmp_str_val, "DE_CR_NSDE")) {
                st_ctrl_p.DE_CR_type = DE_CR_NSDE;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "PSO_para_type")) {
            if(!strcmp(tmp_str_val, "PSO_PARA_FIXED")) {
                st_ctrl_p.PSO_para_type = PSO_PARA_FIXED;
            } else if(!strcmp(tmp_str_val, "PSO_PARA_ADAP")) {
                st_ctrl_p.PSO_para_type = PSO_PARA_ADAP;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "ScalePara_tag")) {
            if(!strcmp(tmp_str_val, "SCALE_NONE")) {
                st_ctrl_p.ScalePara_tag = SCALE_NONE;
            } else if(!strcmp(tmp_str_val, "SCALE_QUANTUM")) {
                st_ctrl_p.ScalePara_tag = SCALE_QUANTUM;
            } else if(!strcmp(tmp_str_val, "SCALE_LEVY")) {
                st_ctrl_p.ScalePara_tag = SCALE_LEVY;
            } else if(!strcmp(tmp_str_val, "SCALE_CAUCHY")) {
                st_ctrl_p.ScalePara_tag = SCALE_CAUCHY;
            } else if(!strcmp(tmp_str_val, "SCALE_GAUSS")) {
                st_ctrl_p.ScalePara_tag = SCALE_GAUSS;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "multiPop_mode")) {
            if(!strcmp(tmp_str_val, "MP_0")) {
                st_ctrl_p.multiPop_mode = MP_0;
            } else if(!strcmp(tmp_str_val, "MP_I")) {
                st_ctrl_p.multiPop_mode = MP_I;
            } else if(!strcmp(tmp_str_val, "MP_II")) {
                st_ctrl_p.multiPop_mode = MP_II;
            } else if(!strcmp(tmp_str_val, "MP_III")) {
                st_ctrl_p.multiPop_mode = MP_III;
            } else if(!strcmp(tmp_str_val, "MP_ADAP")) {
                st_ctrl_p.multiPop_mode = MP_ADAP;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "collect_pop_type")) {
            if(!strcmp(tmp_str_val, "COLLECT_NONDOMINATED")) {
                st_ctrl_p.collect_pop_type = COLLECT_NONDOMINATED;
            } else if(!strcmp(tmp_str_val, "COLLECT_WEIGHTED")) {
                st_ctrl_p.collect_pop_type = COLLECT_WEIGHTED;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "type_xor_rem_vars")) {
            if(!strcmp(tmp_str_val, "XOR_REMVARS_XOR_MIXED")) {
                st_ctrl_p.type_xor_rem_vars = XOR_REMVARS_XOR_MIXED;
            } else if(!strcmp(tmp_str_val, "XOR_REMVARS_COPY")) {
                st_ctrl_p.type_xor_rem_vars = XOR_REMVARS_COPY;
            } else if(!strcmp(tmp_str_val, "XOR_REMVARS_INHERIT")) {
                st_ctrl_p.type_xor_rem_vars = XOR_REMVARS_INHERIT;
            } else if(!strcmp(tmp_str_val, "XOR_REMVARS_XOR_POP")) {
                st_ctrl_p.type_xor_rem_vars = XOR_REMVARS_XOR_POP;
            } else if(!strcmp(tmp_str_val, "XOR_REMVARS_XOR_SAME_REGION")) {
                st_ctrl_p.type_xor_rem_vars = XOR_REMVARS_XOR_SAME_REGION;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "rand_type")) {
            if(!strcmp(tmp_str_val, "RAND_UNIF")) {
                rand_type = RAND_UNIF;
            } else if(!strcmp(tmp_str_val, "RAND_CHEBYSHEV")) {
                rand_type = RAND_CHEBYSHEV;
            } else if(!strcmp(tmp_str_val, "RAND_PIECEWISE_LINEAR")) {
                rand_type = RAND_PIECEWISE_LINEAR;
            } else if(!strcmp(tmp_str_val, "RAND_SINUS")) {
                rand_type = RAND_SINUS;
            } else if(!strcmp(tmp_str_val, "RAND_LOGISTIC")) {
                rand_type = RAND_LOGISTIC;
            } else if(!strcmp(tmp_str_val, "RAND_CIRCLE")) {
                rand_type = RAND_CIRCLE;
            } else if(!strcmp(tmp_str_val, "RAND_GAUSS")) {
                rand_type = RAND_GAUSS;
            } else if(!strcmp(tmp_str_val, "RAND_TENT")) {
                rand_type = RAND_TENT;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "flag_save_trace_PS")) {
            if(!strcmp(tmp_str_val, "FLAG_OFF")) {
                st_ctrl_p.flag_save_trace_PS = FLAG_OFF;
            } else if(!strcmp(tmp_str_val, "FLAG_ON")) {
                st_ctrl_p.flag_save_trace_PS = FLAG_ON;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "flag_check_more_update_DECOM")) {
            if(!strcmp(tmp_str_val, "FLAG_OFF")) {
                st_ctrl_p.flag_check_more_update_DECOM = FLAG_OFF;
            } else if(!strcmp(tmp_str_val, "FLAG_ON")) {
                st_ctrl_p.flag_check_more_update_DECOM = FLAG_ON;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "type_xor_evo_mut")) {
            if(!strcmp(tmp_str_val, "XOR_EVO_MUT_FIX")) {
                st_ctrl_p.type_xor_evo_mut = XOR_EVO_MUT_FIX;
            } else if(!strcmp(tmp_str_val, "XOR_EVO_MUT_ADAP")) {
                st_ctrl_p.type_xor_evo_mut = XOR_EVO_MUT_ADAP;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "type_select_obj_best")) {
            if(!strcmp(tmp_str_val, "SELECT_OBJ_BEST_TOP_N")) {
                st_ctrl_p.type_select_obj_best = SELECT_OBJ_BEST_TOP_N;
            } else if(!strcmp(tmp_str_val, "SELECT_OBJ_BEST_HISTORY")) {
                st_ctrl_p.type_select_obj_best = SELECT_OBJ_BEST_HISTORY;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "type_clone_evo")) {
            if(!strcmp(tmp_str_val, "CLONE_EVO_LOCAL")) {
                st_ctrl_p.type_clone_evo = CLONE_EVO_LOCAL;
            } else if(!strcmp(tmp_str_val, "CLONE_EVO_GLOBAL")) {
                st_ctrl_p.type_clone_evo = CLONE_EVO_GLOBAL;
            } else if(!strcmp(tmp_str_val, "CLONE_EVO_NONE")) {
                st_ctrl_p.type_clone_evo = CLONE_EVO_NONE;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        } else if(!strcmp(tmp_str_nm, "Qubits_angle_opt_tag")) {
            if(!strcmp(tmp_str_val, "FLAG_OFF")) {
                st_ctrl_p.Qubits_angle_opt_tag = FLAG_OFF;
            } else if(!strcmp(tmp_str_val, "FLAG_ON")) {
                st_ctrl_p.Qubits_angle_opt_tag = FLAG_ON;
            } else {
                if(0 == st_MPI_p.mpi_rank) {
                    printf("%s: invalid para val for %s\n", AT, tmp_str_nm);
                }
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
                return;
            }
        }
    }
    fclose(fp);
    //////////////////////////////////////////////////////////////////////////
    switch(st_ctrl_p.multiPop_mode) {
    case MP_0:
        break;
    case MP_I:
        switch(st_ctrl_p.algo_mech_type) {
        case LOCALIZATION:
            sprintf(st_global_p.algorithmName, "DPCCMOEA_MP_I");
            break;
        case DECOMPOSITION:
            sprintf(st_global_p.algorithmName, "DPCCMOLSEA_MP_I");
            break;
        case NONDOMINANCE:
            sprintf(st_global_p.algorithmName, "DPCCMOLSIA_MP_I");
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s: No such algorithm mechanism type\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
            break;
        }
        break;
    case MP_II:
        switch(st_ctrl_p.algo_mech_type) {
        case LOCALIZATION:
            sprintf(st_global_p.algorithmName, "DPCCMOEA_MP_II");
            break;
        case DECOMPOSITION:
            sprintf(st_global_p.algorithmName, "DPCCMOLSEA_MP_II");
            break;
        case NONDOMINANCE:
            sprintf(st_global_p.algorithmName, "DPCCMOLSIA_MP_II");
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s: No such algorithm mechanism type\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
            break;
        }
        break;
    case MP_III:
        switch(st_ctrl_p.algo_mech_type) {
        case LOCALIZATION:
            sprintf(st_global_p.algorithmName, "DPCCMOEA_MP_III");
            break;
        case DECOMPOSITION:
            sprintf(st_global_p.algorithmName, "DPCCMOLSEA_MP_III");
            break;
        case NONDOMINANCE:
            sprintf(st_global_p.algorithmName, "DPCCMOLSIA_MP_III");
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s: No such algorithm mechanism type\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
            break;
        }
        break;
    case MP_ADAP:
        switch(st_ctrl_p.algo_mech_type) {
        case LOCALIZATION:
            sprintf(st_global_p.algorithmName, "DPCCMOEA_MP_ADAP");
            break;
        case DECOMPOSITION:
            sprintf(st_global_p.algorithmName, "DPCCMOLSEA_MP_ADAP");
            break;
        case NONDOMINANCE:
            sprintf(st_global_p.algorithmName, "DPCCMOLSIA_MP_ADAP");
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s: No such algorithm mechanism type\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
            break;
        }
        break;
    default:
        if(st_MPI_p.mpi_rank == 0) {
            printf("%s:MP_MODE selection is wrong, no other algorithm available.\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        break;
    }
    //////////////////////////////////////////////////////////////////////////
    return;
}

void setParaDefault()
{
    st_ctrl_p.optimizer_type =
        EC_DE_CUR_1;
    SI_QPSO;
    SI_PSO;
    OPTIMIZER_ENSEMBLE;
    EC_DE_RAND_1;
    EC_SBX_CUR;
    EC_SBX_RAND;
    EC_MIX_DE_C_SBX_R;
    EC_DE_2SELECTED;
    EC_MIX_DE_C_SBX_C;
    EC_MIX_DE_R_SBX_R;
    EC_MIX_SBX_C_R;
    EC_SI_MIX_DE_C_PSO;
    EC_MIX_DE_C_1_2;
    EC_DE_CUR_2;
    EC_MIX_DE_R_1_2;
    EC_DE_RAND_2;
    OPTIMIZER_BLEND;
    //////////////////////////////////////////////////////////////////////////
    st_optimizer_p.num_optimizer = 2;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.updatePop_type = UPDATE_POP_MOEAD;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.DE_F_type = //
        DE_F_JADE;
    DE_F_FIXED;
    DE_F_jDE;
    DE_F_SHADE;
    DE_F_SaNSDE_a;
    DE_F_SaNSDE;
    st_ctrl_p.F_para_limit_tag =
        FLAG_ON;//(0,1]
    FLAG_OFF;//(0,2]
    if(st_ctrl_p.DE_F_type == DE_F_NSDE ||
       st_ctrl_p.DE_F_type == DE_F_SaNSDE)
        st_ctrl_p.F_para_limit_tag = FLAG_OFF;
    st_ctrl_p.DE_CR_type = //
        DE_CR_JADE;
    DE_CR_FIXED;
    DE_CR_jDE;
    DE_CR_SHADE;
    DE_CR_LINEAR;
    st_ctrl_p.PSO_para_type =
        PSO_PARA_FIXED;
    PSO_PARA_ADAP;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.type_clone_selection =
        CLONE_SLCT_AGGFIT_G;// tour selection based on fitness
    CLONE_SLCT_ND2;// only from the first rank
    CLONE_SLCT_ND1;// all ranks
    CLONE_SLCT_AGGFIT_L;// tour selection based on fitness
    CLONE_SLCT_PREFER;
    CLONE_SLCT_UTILITY_TOUR;// tour selection
    CLONE_SLCT_ND_TOUR;// tour selection based on nondominance
    st_ctrl_p.tag_prefer_which_obj =
        PREFER_NONE_OBJ;
    PREFER_FIRST_OBJ;
    PREFER_THIRD_OBJ;
    PREFER_SECOND_OBJ;
    st_ctrl_p.type_clone_evo =
        CLONE_EVO_LOCAL;
    CLONE_EVO_GLOBAL;
    CLONE_EVO_NONE;
    st_ctrl_p.type_join_xor =
        JOIN_XOR_RAND;
    JOIN_XOR_AGGFIT;
    JOIN_XOR_UTILITY;
    st_ctrl_p.CLONALG_tag =
        FLAG_OFF;
    FLAG_ON;
    st_decomp_p.prefer_intensity = 0; //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.MFI_update_tag =
        FLAG_OFF;
    FLAG_ON;
    st_ctrl_p.ScalePara_tag =
        SCALE_NONE;
    SCALE_QUANTUM;
    SCALE_LEVY;
    SCALE_CAUCHY;
    SCALE_GAUSS;
    st_ctrl_p.st_scale_para.levy_c = 0.1;
    st_ctrl_p.st_scale_para.levy_a = 1;
    st_ctrl_p.st_scale_para.cauchy_a = 0;
    st_ctrl_p.st_scale_para.cauchy_b = 1;
    st_ctrl_p.st_scale_para.gauss_a = 0;
    st_ctrl_p.st_scale_para.gauss_b = 1;
    st_ctrl_p.Qubits_angle_opt_tag =
        FLAG_OFF;
    FLAG_ON;
    st_ctrl_p.Qubits_transform_tag =
        FLAG_OFF;
    FLAG_ON;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.weight_evo_tag =
        FLAG_OFF;
    FLAG_ON;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.multiPop_mode =
        MP_0;
    MP_ADAP;
    MP_I;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.collect_pop_type =
        COLLECT_NONDOMINATED;
    COLLECT_WEIGHTED;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.type_var_encoding = VAR_DOUBLE;
    st_ctrl_p.commonality_xor_remvar_tag = FLAG_OFF;
    st_ctrl_p.opt_binVar_as_realVar_tag = FLAG_ON;
#ifdef WEIGHT_ENCODING
    st_ctrl_p.type_var_encoding = VAR_DOUBLE;
    st_ctrl_p.commonality_xor_remvar_tag = FLAG_OFF;
#else
    st_ctrl_p.type_var_encoding = VAR_BINARY;
    st_ctrl_p.commonality_xor_remvar_tag = FLAG_ON;
#endif
    st_ctrl_p.opt_diverVar_separately = FLAG_OFF;
    //
    st_ctrl_p.type_select_obj_best = SELECT_OBJ_BEST_HISTORY;
    SELECT_OBJ_BEST_TOP_N;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.type_xor_rem_vars =
        XOR_REMVARS_XOR_MIXED;
    XOR_REMVARS_COPY;
    XOR_REMVARS_INHERIT;
    XOR_REMVARS_XOR_POP;
    XOR_REMVARS_XOR_SAME_REGION;
    //////////////////////////////////////////////////////////////////////////
    st_grp_ana_p.NumDependentAnalysis1 = 1;
    st_grp_ana_p.NumDependentAnalysis = 5;
    st_grp_ana_p.NumControlAnalysis = 20;
    st_grp_ana_p.NumRepControlAnalysis = 1;
    st_grp_ana_p.div_ratio = 111.1;
    st_grp_ana_p.weight_thresh = 1e-6;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.type_grouping =
        GROUPING_TYPE_CLASSIFY_RANDOM;
    GROUPING_TYPE_SPECTRAL_CLUSTERING;
    st_ctrl_p.type_feature_adjust = FEATURE_ADJUST_FILTER_MARKOV;
    st_ctrl_p.type_xor_evo_mut = XOR_EVO_MUT_FIX;//
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.type_xor_CNN = XOR_CNN_NORMAL;
    XOR_CNN_LeNet;
    st_ctrl_p.type_del_var = DEL_NORMAL;
    DEL_LeNet;
    st_ctrl_p.type_dim_convert = DIM_NORMAL;
    DIM_CONVERT_CNN;
    st_ctrl_p.type_limit_exceed_proc = LIMIT_ADJUST;
    LIMIT_TRUNCATION;
    st_ctrl_p.flag_save_trace_PS = FLAG_OFF;
    FLAG_ON;
    st_ctrl_p.indicator_tag = INDICATOR_IGD_HV;
    INDICATOR_HV;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.tag_gather_after_evaluate = FLAG_OFF;
    if(st_ctrl_p.type_test == MY_TYPE_ACTIVITY_DETECTION_CLASSIFY ||
       st_ctrl_p.type_test == MY_TYPE_CFRNN_CLASSIFY ||
       st_ctrl_p.type_test == MY_TYPE_EVO_CFRNN ||
       st_ctrl_p.type_test == MY_TYPE_EVO_FRNN_PREDICT ||
       st_ctrl_p.type_test == MY_TYPE_EVO_DFRNN_PREDICT) {
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
        st_ctrl_p.tag_gather_after_evaluate = FLAG_ON;
#endif
    }
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.flag_check_more_update_DECOM = FLAG_ON;
    //////////////////////////////////////////////////////////////////////////
    st_DE_p.F = 0.5;
    st_DE_p.CR = 1.0;
    st_DE_p.CR_rem = 0.5;
    st_optimizer_p.ratio_mut = 1.0 / st_global_p.nDim;
    st_ctrl_p.type_mut_general = MUT_GENERAL_POLYNOMIAL;
    if(st_ctrl_p.type_test == MY_TYPE_ACTIVITY_DETECTION_CLASSIFY) {
        //strct_all_optimizer_paras.ratio_mut = 1.0;
        //strct_ctrl_para.type_mut_general = MUT_GENERAL_RAND;
    }
    //////////////////////////////////////////////////////////////////////////
    st_global_p.nDim_MAP = 3;
    if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
       st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
        st_global_p.nDim =
            NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B +
            NUM_PARA_C3_MAPS * st_global_p.nDim_MAP + NUM_PARA_C3_B +
            NUM_PARA_O5;
#if OPTIMIZE_STRUCTURE_CNN == 1
        st_global_p.nDim += DIM_ALL_STRU_CNN;
#endif
    }
    //////////////////////////////////////////////////////////////////////////
    return;
}

void checkParas()
{
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    //
    if(algo_mech_type == LOCALIZATION) {
        if(st_ctrl_p.multiPop_mode != MP_0) {
            printf("%s: If algo_mech_type is LOCALIZATION (DPCCMOEA), objective decomposition cannot be utilized,\n", AT);
            printf("\t\t therefore, multiPop_mode is set to MP_0\n");
            st_ctrl_p.multiPop_mode = MP_0;
        }
        if(st_ctrl_p.CLONALG_tag != FLAG_OFF) {
            printf("%s: If algo_mech_type is LOCALIZATION (DPCCMOEA), CLONALG_tag cannot be FLAG_ON,\n", AT);
            printf("\t\t therefore, CLONALG_tag is set to FLAG_OFF\n");
            st_ctrl_p.CLONALG_tag = FLAG_OFF;
        }
        st_ctrl_p.opt_diverVar_separately = FLAG_ON;
        st_ctrl_p.type_join_xor = JOIN_XOR_UTILITY;
    }
    //
    return;
}

void initializePara()
{
    setLimits_transformed();

    int i, j;

    if(st_ctrl_p.cur_run == 1) {
        for(i = 0; i < 128; i++) {
            for(j = 0; j < 128; j++) {
                st_indicator_p.mat_IGD_all[i][j] = 0;
                st_indicator_p.mat_HV_all_TRAIN[i][j] = 0;
                st_indicator_p.mat_minPrc_all_TRAIN[i][j] = 0;
                st_indicator_p.mat_HV_all_VALIDATION[i][j] = 0;
                st_indicator_p.mat_minPrc_all_VALIDATION[i][j] = 0;
                st_indicator_p.mat_HV_all_TEST[i][j] = 0;
                st_indicator_p.mat_HV_all[i][j] = 0;
            }
            st_indicator_p.vec_NTRACE_all_VALIDATION[i] = NTRACE;
            st_indicator_p.vec_TIME_all[i] = 0;
            st_indicator_p.vec_TIME_grouping[i] = 0;
            st_indicator_p.vec_TIME_indicator[i] = 0;
        }
    }
    st_indicator_p.threshold_VALIDATION = (int)(ceil(NTRACE * 0.2));
    st_indicator_p.count_VALIDATION = 0;

    for(i = 0; i < st_optimizer_p.num_optimizer; i++) {
        st_optimizer_p.optimizer_prob[i] = 1.0 / st_optimizer_p.num_optimizer;
    }

    for(i = 0; i < st_global_p.nInd_1pop; i++) {
        st_optimizer_p.optimizer_types_all[i] = st_ctrl_p.optimizer_type;
        st_optimizer_p.DE_F_types_all[i] = st_ctrl_p.DE_F_type;
        st_optimizer_p.DE_CR_types_all[i] = st_ctrl_p.DE_CR_type;
        st_optimizer_p.PSO_para_types_all[i] = st_ctrl_p.PSO_para_type;
    }

    //PSO
    for(i = 0; i < st_global_p.nDim; i++) {
        st_PSO_p.vMax[i] = 0.035 * (st_global_p.maxLimit[i] - st_global_p.minLimit[i]);
        st_PSO_p.vMin[i] = -st_PSO_p.vMax[i];
    }
    for(i = 0; i < st_global_p.nInd_max; i++) {
        for(j = 0; j < st_global_p.nDim; j++) {
            st_PSO_p.velocity[i * st_global_p.nDim + j] = rndreal(st_PSO_p.vMin[j], st_PSO_p.vMax[j]);
        }
    }
    st_PSO_p.w_fixed = 0.729;
    st_PSO_p.c1_fixed = 1.49445;
    st_PSO_p.c2_fixed = 1.49445;
    st_PSO_p.w_min = 0.4;
    st_PSO_p.c1_min = 0.0;
    st_PSO_p.c2_min = 0.0;
    st_PSO_p.w_max = 0.9;
    st_PSO_p.c1_max = 2.0;
    st_PSO_p.c2_max = 2.0;
    st_PSO_p.w_mu = st_PSO_p.w_min + 0.5 * (st_PSO_p.w_max - st_PSO_p.w_min);
    st_PSO_p.c1_mu = st_PSO_p.c1_min + 0.5 * (st_PSO_p.c1_max - st_PSO_p.c1_min);
    st_PSO_p.c2_mu = st_PSO_p.c2_min + 0.5 * (st_PSO_p.c2_max - st_PSO_p.c2_min);
    for(i = 0; i < st_global_p.the_size_IND; i++) {
        st_PSO_p.w__cur[i] = st_PSO_p.w_fixed;
        st_PSO_p.c1_cur[i] = st_PSO_p.c1_fixed;
        st_PSO_p.c2_cur[i] = st_PSO_p.c2_fixed;
    }
    st_PSO_p.alpha_begin_Qu = 1.0;
    st_PSO_p.alpha_final_Qu = 0.9;

    for(i = 0; i < st_archive_p.nArch; i++) {
        st_PSO_p.w__archive[i] = st_PSO_p.w_fixed;
        st_PSO_p.c1_archive[i] = st_PSO_p.c1_fixed;
        st_PSO_p.c2_archive[i] = st_PSO_p.c2_fixed;
    }

    //
    for(i = 0; i < st_global_p.nObj; i++) {
        st_pop_best_p.obj_best[i] = INF_DOUBLE;
    }
    for(i = 0; i < st_global_p.nObj; i++) {
        for(j = 0; j < st_global_p.nObj; j++) {
            st_pop_best_p.obj_best_subObjs_all[i * st_global_p.nObj + j] = INF_DOUBLE;
        }
    }

    for(i = 0; i < st_global_p.the_size_IND; i++) st_utility_p.utility[i] = 1.0;
    for(i = 0; i < st_global_p.the_size_IND; i++) st_utility_p.utility_cur[i] = 1.0;

    st_optimizer_p.p_best_ratio = 0.05;

    //
    for(i = 0; i <= st_global_p.nObj; i++) {
        for(j = 0; j < st_global_p.nDim; j++) {
            st_MPI_p.ns_pops[i * st_global_p.nDim + j] = 0;
            st_MPI_p.nf_pops[i * st_global_p.nDim + j] = 0;
        }
    }

    st_optimizer_p.ns_optimizer_1 = 0;
    st_optimizer_p.nf_optimizer_1 = 0;
    st_optimizer_p.ns_optimizer_2 = 0;
    st_optimizer_p.nf_optimizer_2 = 0;
    st_optimizer_p.ns_optimizer_PSO = 0;
    st_optimizer_p.nf_optimizer_PSO = 0;
    st_optimizer_p.nGen_accum_ada_opti = 0;
    st_optimizer_p.nGen_th_accum_ada_opti = 50;
    st_optimizer_p.slctProb_opt_1 = 0.5;
    st_optimizer_p.slctProb_opt_2 = 0.5;
    st_optimizer_p.slctProb_PSO = 0.0;
    switch(st_ctrl_p.optimizer_type) {
    case EC_MIX_DE_R_SBX_R:
        st_optimizer_p.optimizer_candid[0] = EC_DE_RAND_1;
        st_optimizer_p.optimizer_candid[1] = EC_SBX_RAND;
        break;
    case EC_MIX_DE_C_SBX_R:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = EC_SBX_RAND;
        break;
    case EC_MIX_DE_C_SBX_C:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = EC_SBX_CUR;
        break;
    case EC_MIX_SBX_C_R:
        st_optimizer_p.optimizer_candid[0] = EC_SBX_CUR;
        st_optimizer_p.optimizer_candid[1] = EC_SBX_RAND;
        break;
    case EC_SI_MIX_DE_C_PSO:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = SI_PSO;
        break;
    case EC_MIX_DE_C_R:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = EC_DE_RAND_1;
        break;
    case EC_MIX_DE_C_1_2:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = EC_DE_CUR_2;
        break;
    case EC_MIX_DE_R_1_2:
        st_optimizer_p.optimizer_candid[0] = EC_DE_RAND_1;
        st_optimizer_p.optimizer_candid[1] = EC_DE_RAND_2;
        break;
    case OPTIMIZER_BLEND:
        st_optimizer_p.optimizer_candid[0] = EC_SBX_RAND;
        EC_SBX_CUR;
        st_optimizer_p.optimizer_candid[1] = EC_DE_CUR_1;
        break;
    case OPTIMIZER_ENSEMBLE:
        st_optimizer_p.optimizer_candid[0] = EC_DE_CUR_1;
        st_optimizer_p.optimizer_candid[1] = SI_PSO;
        EC_SBX_RAND;
        EC_SBX_CUR;
        break;
    default:
        break;
    }
    st_DE_p.ns_JADE_F = 0;
    st_DE_p.nf_JADE_F = 0;
    st_DE_p.ns_JADE_CR = 0;
    st_DE_p.nf_JADE_CR = 0;
    st_DE_p.ns_SHADE = 0;
    st_DE_p.nf_SHADE = 0;
    st_DE_p.ns1_SaNSDE_F = 0;
    st_DE_p.nf1_SaNSDE_F = 0;
    st_DE_p.ns2_SaNSDE_F = 0;
    st_DE_p.nf2_SaNSDE_F = 0;
    st_DE_p.nGen_accum_ada_para = 0;
    st_DE_p.nGen_th_accum_ada_para = 50;

    st_DE_p.slctProb_JADE_F = 0.0;
    st_DE_p.slctProb_JADE_CR = 0.0;
    st_DE_p.slctProb_SHADE = 0.0;
    st_DE_p.slctProb_SaNSDE_F = 0.5;

    st_DE_p.F_mu = 0.5;
    st_DE_p.CR_mu = 0.5;
    st_DE_p.CR_evo_mu = 0.5;
    st_DE_p.t1_best_mu = 0.5;
    st_DE_p.t2_best_mu = 0.5;
    st_DE_p.F_mu_arch = 0.5;
    st_DE_p.CR_mu_arch = 0.5;
    st_DE_p.CR_evo_mu_arch = 0.5;
    st_decomp_p.th_select = 0.9;

    st_DE_p.Fcount = 0;
    st_DE_p.CRcount = 0;
    st_DE_p.c_para = 0.1;

    for(i = 0; i < st_global_p.nInd_1pop; i++) st_optimizer_p.rate_Commonality[i] = 1;

    for(i = 0; i < st_global_p.nInd_1pop; i++) {
        st_DE_p.Sflag[i] = 0;
        st_DE_p.CR_evo_cur[i] = st_DE_p.CR_evo_mu;
        st_DE_p.t1_best[i] = st_DE_p.t1_best_mu;
        st_DE_p.t2_best[i] = st_DE_p.t2_best_mu;
        st_DE_p.F__cur[i] = 0.5;
        st_DE_p.CR_cur[i] = 0.5;
    }

    for(i = 0; i < st_archive_p.nArch; i++) {
        st_DE_p.F__archive[i] = st_DE_p.F_mu_arch;
        st_DE_p.CR_archive[i] = st_DE_p.CR_mu_arch;
        st_DE_p.CR_evo_arc[i] = st_DE_p.CR_evo_mu_arch;
    }

    for(i = 0; i < st_global_p.num_subpops * st_DE_p.nHistSHADE; i++) {
        st_DE_p.F_hist[i] = 0.5;
        st_DE_p.CR_hist[i] = 0.5;
    }

    for(i = 0; i < st_DE_p.candid_num; i++) {
        st_DE_p.candid_F[i] = (i + 1) / st_DE_p.candid_num;
        st_DE_p.candid_CR[i] = (i + 1) / st_DE_p.candid_num;
        st_DE_p.prob_F[i] = 1.0 / st_DE_p.candid_num;
        st_DE_p.prob_CR[i] = 1.0 / st_DE_p.candid_num;
    }

    st_decomp_p.limit = (int)(0.01 * st_global_p.nPop);
    st_decomp_p.niche = (int)(0.1 * st_global_p.nPop);
    if(st_global_p.nObj == 2) {
        st_decomp_p.limit = 2;
        st_decomp_p.niche = 10;
    }
    if(st_global_p.nObj == 3) {
        st_decomp_p.limit = 2;
        st_decomp_p.niche = 12;
    }
    if(st_global_p.nObj == 10) {
        st_decomp_p.limit = 3;
    }
    if(st_decomp_p.limit < 2)
        st_decomp_p.limit = 2;
    if(st_decomp_p.niche > st_global_p.nPop)
        st_decomp_p.niche = st_global_p.nPop;

    for(j = 0; j < st_global_p.nObj; j++) {
        st_decomp_p.idealpoint[j] = 1.0e30;
        st_decomp_p.nadirpoint[j] = -1.0e30;
    }

    st_global_p.nPop_exchange = 0;
    st_archive_p.cnArch_exchange = 0;

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    setVarTypes();

    strcpy(st_global_p.strFunctionType, "_MTCH2");
    // 	return;

    if(!strcmp("UF1", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF2", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF3", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF4", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF5", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF6", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF7", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF8", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF9", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("UF10", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ1", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ2", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ3", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ4", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ5", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ6", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH1");
        return;
    }
    if(!strcmp("DTLZ7", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG1", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG2", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG3", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG4", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG5", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG6", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG7", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG8", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    if(!strcmp("WFG9", st_global_p.testInstance)) {
        strcpy(st_global_p.strFunctionType, "_MTCH2");
        return;
    }
    return;
}

void initializePopulation()
{
    load_samplePoints();
    //
    int i, j;
    //
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE) {
        int tmp_i, tmp_j;
        for(i = 0; i < st_global_p.nPop; i++) {
            for(j = 0; j < st_global_p.nDim; j++) {
                tmp_i = j % st_global_p.nPop;
                tmp_j = j / st_global_p.nPop;
                st_pop_evo_cur.var[i * st_global_p.nDim + j] =
                    1.0 - st_decomp_p.weights_all[tmp_i * st_global_p.nObj + tmp_j];
                if(st_pop_evo_cur.var[i * st_global_p.nDim + j] < 0.0)
                    st_pop_evo_cur.var[i * st_global_p.nDim + j] = 0.0;
                if(st_pop_evo_cur.var[i * st_global_p.nDim + j] > 1.0)
                    st_pop_evo_cur.var[i * st_global_p.nDim + j] = 1.0;
            }
        }
    } else if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
        int tmp_nGroup = 0;
        int* var_clusterIDs = (int*)calloc(st_global_p.nDim, sizeof(int));
        int* classSizes = (int*)calloc(st_global_p.nDim, sizeof(int));
        int* clusterIDs = (int*)calloc(st_global_p.nDim, sizeof(int));
        for(int i = 0; i < st_global_p.nDim; i++) {
            classSizes[i] = 0;
        }
        //////////////////////////////////////////////////////////////////////////
        char file[1024];
        FILE* ifs;
        int max_buf_size = 1000 * 20 + 1;
        char tmp_delim[] = " ,\t\r\n";
        char* buff = (char*)calloc(max_buf_size, sizeof(char));
        char* p;
        sprintf(file, "../Data_all/Data_EdgeComputation/%dall.csv", RSU_SET_CARDINALITY_CUR);
        ifs = fopen(file, "r");
        if(!ifs) {
            printf("%s(%d): Open file %s error, exiting...\n",
                   __FILE__, __LINE__, file);
            exit(-111007);
        }
        if(!fgets(buff, max_buf_size, ifs)) {
            printf("%s(%d): Not enough data for file %s, exiting...\n",
                   __FILE__, __LINE__, file);
            exit(-111006);
        }
        int tmp_cnt = 0;
        while(fgets(buff, max_buf_size, ifs)) {
            int k = 0;
            for(p = strtok(buff, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                int tmp_val;
                if(sscanf(p, "%d", &tmp_val) != 1) {
                    printf("%s(%d): No more data file %s, exiting...\n",
                           __FILE__, __LINE__, file);
                    exit(-111006);
                }
                if(k == 3) {
                    var_clusterIDs[tmp_cnt++] = tmp_val - 1;
                    if(tmp_val > st_global_p.nDim || tmp_val < 1) {
                        printf("%s(%d): Invalid cluster ID for row %d (%d not in %d~%d) file %s, exiting...\n",
                               __FILE__, __LINE__, tmp_cnt, tmp_val, 1, st_global_p.nDim, file);
                        exit(-1110055);
                    }
                    classSizes[tmp_val - 1]++;
                    //printf("var_c_ID - %d ", tmp_val - 1);
                }
                k++;
            }
        }
        //printf("\n");
        if(tmp_cnt != st_global_p.nDim) {
            printf("%s(%d): The number of IDs is not consistent withe the setting (%d != %d), exiting...\n",
                   __FILE__, __LINE__, tmp_cnt, st_global_p.nDim);
            exit(-111006);
        }
        fclose(ifs);
        free(buff);
        //////////////////////////////////////////////////////////////////////////
        for(int i = 0; i < st_global_p.nDim; i++) {
            if(classSizes[i]) {
                //printf("size - %d\n", classSizes[i]);
                clusterIDs[tmp_nGroup++] = i;
            }
        }
        //
        for(int i = 0; i < st_global_p.nPop; i++) {
            //if(i >= 0) {
            //    for(int j = 0; j < strct_global_paras.nDim; j++) {
            //        if(flip_r((1.0 + i) / strct_global_paras.nPop / tmp_nGroup))
            //            strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] = rndreal(VAR_THRESHOLD_EdgeComputation, strct_global_paras.maxLimit[j]);
            //        else
            //            strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] = rndreal(strct_global_paras.minLimit[j], VAR_THRESHOLD_EdgeComputation - 1e-6);
            //        //printf("%d ", tagVar[j]);
            //    }
            //    continue;
            //}
            int* tagVar = (int*)calloc(st_global_p.nDim, sizeof(int));
            for(int iGrp = 0; iGrp < tmp_nGroup; iGrp++) {
                int cur_tag = clusterIDs[iGrp];
                int cur_size = classSizes[cur_tag];
                if(i && flip_r((float)(1.0 / tmp_nGroup)))continue;
                int cur_selection = rnd(0, cur_size - 1);
                int tmp_ind = 0;
                for(int iInd = 0; iInd < st_global_p.nDim; iInd++) {
                    if(i && flip_r((float)(1.0 / st_global_p.nDim / tmp_nGroup)))
                        tagVar[iInd] = 1;
                    if(var_clusterIDs[iInd] == cur_tag) {
                        if(tmp_ind == cur_selection) {
                            tagVar[iInd] = 1;
                            if(i && flip_r((float)(1.0 / st_global_p.nDim / tmp_nGroup)))
                                tagVar[iInd] = 0;
                        }
                        tmp_ind++;
                    }
                }
            }
            int tmpVar = rnd(0, st_global_p.nDim - 1);
            for(int j = 0; j < st_global_p.nDim; j++) {
                if(tagVar[j]) {
                    st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(VAR_THRESHOLD_EdgeComputation, st_global_p.maxLimit[j]);
                    if(i && j == tmpVar)
                        st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(st_global_p.minLimit[j],
                                VAR_THRESHOLD_EdgeComputation - 1e-6);
                } else {
                    st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(st_global_p.minLimit[j],
                            VAR_THRESHOLD_EdgeComputation - 1e-6);
                    if(i && j == tmpVar)
                        st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(VAR_THRESHOLD_EdgeComputation, st_global_p.maxLimit[j]);
                }
                //printf("%d ", tagVar[j]);
            }
            //printf("\n");
            free(tagVar);
        }
        //
        free(var_clusterIDs);
        free(classSizes);
        free(clusterIDs);
    } else {
        for(i = 0; i < st_global_p.nPop; i++) {
            for(j = 0; j < st_global_p.nDim; j++) {
                st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(st_global_p.minLimit[j], st_global_p.maxLimit[j]);
            }
            for(j = 0; j < st_grp_ana_p.numDiverIndexes; j++) {
                st_pop_evo_cur.var[i * st_global_p.nDim + st_grp_info_p.DiversityIndexs[j]] =
                    st_grp_info_p.diver_var_store_all[i * st_global_p.nDim + j];
            }
            //if (strct_ctrl_para.type_test == MY_TYPE_LeNet && strct_ctrl_para.type_dim_convert == DIM_CONVERT_CNN){
            //	double tmp[DIM_LeNet];
            //	convertVar_CNN(&strct_pop_evo_info.var_current[i * strct_global_paras.nDim], tmp);
            //	evaluate_problems(strct_global_paras.testInstance, tmp,
            //		&strct_pop_evo_info.obj_current[i * strct_global_paras.nObj], strct_global_paras.nDim, 1, strct_global_paras.nObj);
            //}
            //else{
            //	evaluate_problems(strct_global_paras.testInstance, &strct_pop_evo_info.var_current[i * strct_global_paras.nDim],
            //		&strct_pop_evo_info.obj_current[i * strct_global_paras.nObj], strct_global_paras.nDim, 1, strct_global_paras.nObj);
            //}

            if((st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) &&
               st_ctrl_p.type_var_encoding == VAR_BINARY) {
                //for(j = 0; j < strct_global_paras.nDim; j++) {
                //    if(strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] > 0.5)
                //        strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] = 1;
                //    else
                //        strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] = 0;
                //}
                /////////////////////////////shrink to at most 50 features are selected
                if(st_ctrl_p.type_feature_adjust == FEATURE_ADJUST_FILTER_MARKOV)
                    adjustFeatureNum_Markov(&st_pop_evo_cur.var[i * st_global_p.nDim]);
                else
                    adjustFeatureNum_rand(&st_pop_evo_cur.var[i * st_global_p.nDim]);

                //evaluate_problems(strct_global_paras.testInstance, &strct_pop_evo_info.var_current[i * strct_global_paras.nDim],
                //	&strct_pop_evo_info.obj_current[i * strct_global_paras.nObj], strct_global_paras.nDim, 1, strct_global_paras.nObj);
            }
        }
    }
    //
    if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_ON) {
        for(i = 0; i < st_global_p.nPop; i++) {
            for(j = 0; j < st_global_p.nDim; j++) {
                double yl = st_qu_p.minLimit_rot_angle[j];
                double yu = st_qu_p.maxLimit_rot_angle[j];
                st_qu_p.rot_angle_cur[i * st_global_p.nDim + j] = rndreal(yl, yu);
            }
        }
    }

    // 	int rank;
    // 	int size;
    // 	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // 	MPI_Comm_size(MPI_COMM_WORLD,&size);
    // 	if(!rank)
    // 	{
    // 		int nDiv=strct_grp_ana_vals.numDiverIndexes;
    // 		printf("Diversity position...\n");
    // 		{
    // 			{
    // 				for(i=0;i<strct_global_paras.nPop;i++)
    // 				{
    // 					printf("ID: %d\n",i+1);
    // // 					for(j=0;j<nDiv;j++)
    // // 					{
    // // 						printf("%d: %lf\t",strct_grp_info_vals.DiversityIndexs[j],diver_pos_store_all[i*strct_global_paras.nDim+j]);
    // // 					}
    // // 					printf("\n");
    // 					for(j=0;j<strct_global_paras.nObj;j++)
    // 					{
    // 						printf("%lf\t",fCurrent[i*strct_global_paras.nObj+j]);
    // 					}
    // 					printf("\n");
    // 				}
    // 			}
    // 		}
    // 	}
    // 	MPI_Barrier(MPI_COMM_WORLD);

    // 	for(i=0;i<INIT_POP_SIZE;i++)
    // 	{
    // 		for(j=0;j<strct_global_paras.nDim;j++)
    // 		{
    // 			xCurrent[i*strct_global_paras.nDim+j]=rndreal(strct_global_paras.minLimit[j], strct_global_paras.maxLimit[j]);
    // 		}
    // 	}
    //if(strct_ctrl_para.optimizer_type == GA_QUANTUM) {
    //    for(int i = 0; i < strct_global_paras.nPop; i++) {
    //        if(i % 2) {
    //            continue;
    //        }
    //        for(int j = 0; j < strct_global_paras.nDim; j++) {
    //            double val_quantum = (2.0 * strct_pop_evo_info.var_current[i * strct_global_paras.nDim + j] - strct_global_paras.minLimit[j] - strct_global_paras.maxLimit[j]) /
    //                                 (strct_global_paras.maxLimit[j] - strct_global_paras.minLimit[j]);
    //            double new_quantum = sqrt(1.0 - val_quantum * val_quantum);
    //            if(flip_r((float)0.5)) new_quantum = -new_quantum;
    //            strct_pop_evo_info.var_current[(i + 1)*strct_global_paras.nDim + j] = 0.5 * (strct_global_paras.maxLimit[j] * (1.0 + new_quantum) +
    //                                                   strct_global_paras.minLimit[j] * (1.0 - new_quantum));
    //        }
    //    }
    //}
    int the_rank, the_size;
    MPI_Comm the_comm;
    int the_root;
    int the_NP;
    switch(st_ctrl_p.algo_mech_type) {
    case LOCALIZATION:
    case DECOMPOSITION:
        the_rank = st_MPI_p.mpi_rank;
        the_size = st_MPI_p.mpi_size;
        the_comm = MPI_COMM_WORLD;
        the_root = 0;
        the_NP = st_global_p.nPop;
        break;
    case NONDOMINANCE:
        //for(int i = 0; i < strct_global_paras.nPop; i++)
        //    EMO_evaluate_problems(strct_global_paras.testInstance, &strct_pop_evo_info.var_current[i * strct_global_paras.nDim],
        //                          &strct_pop_evo_info.obj_current[i * strct_global_paras.nObj], strct_global_paras.nDim, 1, strct_global_paras.nObj);
        //memcpy(strct_archive_info.var_archive, strct_pop_evo_info.var_current, strct_global_paras.nPop * strct_global_paras.nDim * sizeof(double));
        //memcpy(strct_archive_info.obj_archive, strct_pop_evo_info.obj_current, strct_global_paras.nPop * strct_global_paras.nObj * sizeof(double));
        //strct_archive_info.cnArch = strct_global_paras.nPop;
        the_rank = st_MPI_p.mpi_rank_subPop;
        the_size = st_MPI_p.mpi_size_subPop;
        the_comm = st_MPI_p.comm_subPop;
        the_root = 0;
        the_NP = st_MPI_p.nPop_all[st_MPI_p.color_pop];
        break;
    default:
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
        break;
    }
    int* tmp_recv_size;
    int* tmp_disp_size;
    int* tmp_each_size;
    tmp_recv_size = (int*)calloc(the_size, sizeof(int));
    tmp_disp_size = (int*)calloc(the_size, sizeof(int));
    tmp_each_size = (int*)calloc(the_size, sizeof(int));
    //
    int quo;
    int rem;
    quo = the_NP / the_size;
    rem = the_NP % the_size;
    for(i = 0; i < the_size; i++) {
        tmp_each_size[i] = quo;
        if(i < rem) tmp_each_size[i]++;
    }
    update_recv_disp(tmp_each_size, st_global_p.nDim, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Scatterv(st_pop_evo_cur.var, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                 st_repo_p.var, tmp_recv_size[the_rank], MPI_DOUBLE,
                 the_root, the_comm);
    for(i = 0; i < tmp_each_size[the_rank]; i++) {
        if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
           st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
            double tmp[DIM_LeNet];
            convertVar_CNN(&st_repo_p.var[i * st_global_p.nDim], tmp);
            EMO_evaluate_problems(st_global_p.testInstance, tmp,
                                  &st_repo_p.obj[i * st_global_p.nObj], DIM_LeNet, 1, st_global_p.nObj);
        } else {
            EMO_evaluate_problems(st_global_p.testInstance, &st_repo_p.var[i * st_global_p.nDim],
                                  &st_repo_p.obj[i * st_global_p.nObj], st_global_p.nDim, 1, st_global_p.nObj);
        }
    }
    update_recv_disp(tmp_each_size, st_global_p.nDim, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Allgatherv(st_repo_p.var, tmp_recv_size[the_rank], MPI_DOUBLE,
                   st_pop_evo_cur.var, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                   the_comm);
    update_recv_disp(tmp_each_size, st_global_p.nObj, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Allgatherv(st_repo_p.obj, tmp_recv_size[the_rank], MPI_DOUBLE,
                   st_pop_evo_cur.obj, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                   the_comm);
    //update_recv_disp(strct_MPI_info.each_size, strct_global_paras.nObj, strct_MPI_info.mpi_size);
    //MPI_Gatherv(strct_repo_info.obj, strct_MPI_info.recv_size[strct_MPI_info.mpi_rank], MPI_DOUBLE,
    //	strct_pop_evo_info.obj_current, strct_MPI_info.recv_size, strct_MPI_info.disp_size, MPI_DOUBLE,
    //	0, MPI_COMM_WORLD);

    //MPI_Bcast(strct_pop_evo_info.var_current, strct_global_paras.nPop * strct_global_paras.nDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(strct_pop_evo_info.obj_current, strct_global_paras.nPop * strct_global_paras.nObj, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(st_archive_p.var, st_pop_evo_cur.var, the_NP * st_global_p.nDim * sizeof(double));
    memcpy(st_archive_p.obj, st_pop_evo_cur.obj, the_NP * st_global_p.nObj * sizeof(double));
    //strct_archive_info.cnArch = the_NP;
    //
    free(tmp_recv_size);
    free(tmp_disp_size);
    free(tmp_each_size);
    //
    return;
}

void initializePopulation_Latin_hyperCube()
{
    load_samplePoints();
    //
    int i, j;

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int* tmp_ind = (int*)calloc(st_global_p.nPop, sizeof(int));
    for(i = 0; i < st_global_p.nPop; i++) tmp_ind[i] = i;
    for(j = 0; j < st_global_p.nDim; j++) {
        shuffle(tmp_ind, st_global_p.nPop);
        double tmp_min, tmp_max;
        double tmp_step = (st_global_p.maxLimit[j] - st_global_p.minLimit[j]) / st_global_p.nPop;
        for(i = 0; i < st_global_p.nPop; i++) {
            tmp_min = st_global_p.minLimit[j] + tmp_step * tmp_ind[i];
            tmp_max = tmp_min + tmp_step;
            st_pop_evo_cur.var[i * st_global_p.nDim + j] = rndreal(tmp_min, tmp_max);
        }
    }
    free(tmp_ind);
    //
    int the_rank, the_size;
    MPI_Comm the_comm;
    int the_root;
    int the_NP;
    switch(st_ctrl_p.algo_mech_type) {
    case LOCALIZATION:
    case DECOMPOSITION:
        the_rank = st_MPI_p.mpi_rank;
        the_size = st_MPI_p.mpi_size;
        the_comm = MPI_COMM_WORLD;
        the_root = 0;
        the_NP = st_global_p.nPop;
        break;
    case NONDOMINANCE:
        //for(int i = 0; i < strct_global_paras.nPop; i++)
        //    EMO_evaluate_problems(strct_global_paras.testInstance, &strct_pop_evo_info.var_current[i * strct_global_paras.nDim],
        //                          &strct_pop_evo_info.obj_current[i * strct_global_paras.nObj], strct_global_paras.nDim, 1, strct_global_paras.nObj);
        //memcpy(strct_archive_info.var_archive, strct_pop_evo_info.var_current, strct_global_paras.nPop * strct_global_paras.nDim * sizeof(double));
        //memcpy(strct_archive_info.obj_archive, strct_pop_evo_info.obj_current, strct_global_paras.nPop * strct_global_paras.nObj * sizeof(double));
        //strct_archive_info.cnArch = strct_global_paras.nPop;
        the_rank = st_MPI_p.mpi_rank_subPop;
        the_size = st_MPI_p.mpi_size_subPop;
        the_comm = st_MPI_p.comm_subPop;
        the_root = 0;
        the_NP = st_MPI_p.nPop_all[st_MPI_p.color_pop];
        break;
    default:
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
        break;
    }
    int* tmp_recv_size;
    int* tmp_disp_size;
    int* tmp_each_size;
    tmp_recv_size = (int*)calloc(the_size, sizeof(int));
    tmp_disp_size = (int*)calloc(the_size, sizeof(int));
    tmp_each_size = (int*)calloc(the_size, sizeof(int));
    //
    int quo;
    int rem;
    quo = the_NP / the_size;
    rem = the_NP % the_size;
    for(i = 0; i < the_size; i++) {
        tmp_each_size[i] = quo;
        if(i < rem) tmp_each_size[i]++;
    }
    update_recv_disp(tmp_each_size, st_global_p.nDim, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Scatterv(st_pop_evo_cur.var, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                 st_repo_p.var, tmp_recv_size[the_rank], MPI_DOUBLE,
                 the_root, the_comm);
    for(i = 0; i < tmp_each_size[the_rank]; i++) {
        if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
           st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
            double tmp[DIM_LeNet];
            convertVar_CNN(&st_repo_p.var[i * st_global_p.nDim], tmp);
            EMO_evaluate_problems(st_global_p.testInstance, tmp,
                                  &st_repo_p.obj[i * st_global_p.nObj], DIM_LeNet, 1, st_global_p.nObj);
        } else {
            EMO_evaluate_problems(st_global_p.testInstance, &st_repo_p.var[i * st_global_p.nDim],
                                  &st_repo_p.obj[i * st_global_p.nObj], st_global_p.nDim, 1, st_global_p.nObj);
        }
    }
    update_recv_disp(tmp_each_size, st_global_p.nDim, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Allgatherv(st_repo_p.var, tmp_recv_size[the_rank], MPI_DOUBLE,
                   st_pop_evo_cur.var, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                   the_comm);
    update_recv_disp(tmp_each_size, st_global_p.nObj, the_size, tmp_recv_size, tmp_disp_size);
    MPI_Allgatherv(st_repo_p.obj, tmp_recv_size[the_rank], MPI_DOUBLE,
                   st_pop_evo_cur.obj, tmp_recv_size, tmp_disp_size, MPI_DOUBLE,
                   the_comm);
    //update_recv_disp(strct_MPI_info.each_size, strct_global_paras.nObj, strct_MPI_info.mpi_size);
    //MPI_Gatherv(strct_repo_info.obj, strct_MPI_info.recv_size[strct_MPI_info.mpi_rank], MPI_DOUBLE,
    //	strct_pop_evo_info.obj_current, strct_MPI_info.recv_size, strct_MPI_info.disp_size, MPI_DOUBLE,
    //	0, MPI_COMM_WORLD);

    //MPI_Bcast(strct_pop_evo_info.var_current, strct_global_paras.nPop * strct_global_paras.nDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(strct_pop_evo_info.obj_current, strct_global_paras.nPop * strct_global_paras.nObj, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(st_archive_p.var, st_pop_evo_cur.var, the_NP * st_global_p.nDim * sizeof(double));
    memcpy(st_archive_p.obj, st_pop_evo_cur.obj, the_NP * st_global_p.nObj * sizeof(double));
    //strct_archive_info.cnArch = the_NP;
    //
    free(tmp_recv_size);
    free(tmp_disp_size);
    free(tmp_each_size);
    //
    return;
}

void setLimits_transformed()
{
    if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_ON) {
        for(int i = 0; i < st_global_p.nDim; i++) {
            st_qu_p.minLimit_rot_angle[i] = -0.05 * PI;
            st_qu_p.maxLimit_rot_angle[i] = 0.05 * PI;
        }
    }

    if(st_ctrl_p.Qubits_transform_tag == FLAG_ON) {
        for(int i = 0; i < st_global_p.nDim; i++) {
            st_qu_p.minLimit[i] = 0.0;
            st_qu_p.maxLimit[i] = 1.0;
        }
    }

    if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
       st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
        int thresh_ori1 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP;
        int thresh_ori2 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B;
        int thresh_ori3 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                          st_global_p.nDim_MAP;
        int thresh_ori4 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                          st_global_p.nDim_MAP + NUM_PARA_C3_B;
        int thresh_ori5 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                          st_global_p.nDim_MAP + NUM_PARA_C3_B + NUM_PARA_O5;
        for(int i = 0; i < st_global_p.nDim;) {
            if(i < thresh_ori1) {
                st_global_p.minLimit[i] = 0.0;
                st_global_p.minLimit[i + 1] = 0.0;
                st_global_p.minLimit[i + 2] = 0.0;
                st_global_p.maxLimit[i] = NUM_ENUM_DIRECT_CNN_MAP - 1e-6;
                st_global_p.maxLimit[i + 1] = 2 * PI - 1e-6;
                st_global_p.maxLimit[i + 2] = 2 * PI - 1e-6;
                i += st_global_p.nDim_MAP;
            } else if(i < thresh_ori2) {
                st_global_p.minLimit[i] = -MAX_WEIGHT_BIAS_CNN;
                st_global_p.maxLimit[i] = MAX_WEIGHT_BIAS_CNN;
                i++;
            } else if(i < thresh_ori3) {
                st_global_p.minLimit[i] = 0.0;
                st_global_p.minLimit[i + 1] = 0.0;
                st_global_p.minLimit[i + 2] = 0.0;
                st_global_p.maxLimit[i] = NUM_ENUM_DIRECT_CNN_MAP - 1e-6;
                st_global_p.maxLimit[i + 1] = 2 * PI - 1e-6;
                st_global_p.maxLimit[i + 2] = 2 * PI - 1e-6;
                i += st_global_p.nDim_MAP;
            } else if(i < thresh_ori4) {
                st_global_p.minLimit[i] = -MAX_WEIGHT_BIAS_CNN;
                st_global_p.maxLimit[i] = MAX_WEIGHT_BIAS_CNN;
                i++;
            } else if(i < thresh_ori5) {
                st_global_p.minLimit[i] = -MAX_WEIGHT_BIAS_CNN;
                st_global_p.maxLimit[i] = MAX_WEIGHT_BIAS_CNN;
                i++;
            } else {
                st_global_p.minLimit[i] = 0.0;
                st_global_p.maxLimit[i] = 2.0 - 1e-6;
                i++;
            }
        }
        //if (0 == strct_MPI_info.mpi_rank){
        //	for (int i = 0; i < strct_global_paras.nDim; i++){
        //		printf("[%lf, %lf]\n", strct_global_paras.minLimit[i], strct_global_paras.maxLimit[i]);
        //	}
        //}
    }
    return;
}

void initializeGenNum()
{
    int nPop = st_global_p.nPop;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int* nPop_all = st_MPI_p.nPop_all;
    int* Groups_sizes = st_grp_info_p.vec_sizeGroups;
    int nPop_mine = st_global_p.nPop_mine;
    int* num_trail_per_gen = &st_global_p.num_trail_per_gen;
    int* num_selected = &st_global_p.num_selected;
    int* num_exploit_per_gen = &st_global_p.num_exploit_per_gen;
    int* num_trail_per_gen_conve = &st_global_p.num_trail_per_gen_conve;
    int* num_trail_per_gen_diver = &st_global_p.num_trail_per_gen_diver;
    int* CHECK_GAP_CC = &st_global_p.CHECK_GAP_CC;
    int* CHECK_GAP_UPDT = &st_global_p.CHECK_GAP_UPDT;
    int* CHECK_GAP_SYNC = &st_global_p.CHECK_GAP_SYNC;
    int* CHECK_GAP_EXCH = &st_global_p.CHECK_GAP_EXCH;
    int* NUPDT = &st_global_p.NUPDT;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int flag_grp_predefined = st_grp_info_p.flag_predefined;
    char* testInstance = st_global_p.testInstance;
    int* usedIter_init_grp = &st_global_p.usedIter_init_grp;
    int NumControlAnalysis = st_grp_ana_p.NumControlAnalysis;
    int NumRepControlAnalysis = st_grp_ana_p.NumRepControlAnalysis;
    int* usedIter_initPop = &st_global_p.usedIter_initPop;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size = st_MPI_p.mpi_size;
    int* usedIter_init = &st_global_p.usedIter_init;
    int* iter = &st_global_p.iter;
    int* remIter = &st_global_p.remIter;
    int maxIter = st_global_p.maxIter;
    int* iter_per_gen = &st_global_p.iter_per_gen;
    int* generatMax = &st_global_p.generatMax;
    int* generation = &st_global_p.generation;
    //
    int min_num;
    MPI_Allreduce(&nPop_mine, &min_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    (*num_trail_per_gen) = min_num > MAX_UPDATE_NUM_PER_GEN ? MAX_UPDATE_NUM_PER_GEN : min_num;
    //
    (*num_selected) = nPop / 10;
    MPI_Barrier(MPI_COMM_WORLD);
    (*num_exploit_per_gen) = 1;
    if((*num_exploit_per_gen) == 0)(*num_exploit_per_gen)++;
    (*num_trail_per_gen_conve) = min_num / nObj;
    if((*num_trail_per_gen_conve) == 0)(*num_trail_per_gen_conve)++;
    (*num_trail_per_gen_diver) = min_num / nObj;
    if((*num_trail_per_gen_diver) == 0)(*num_trail_per_gen_diver)++;
    //
    (*CHECK_GAP_CC) = 5;
    (*CHECK_GAP_UPDT) = 5;
    (*CHECK_GAP_SYNC) = 10;
    (*CHECK_GAP_EXCH) = 30;
    (*NUPDT) = 50;
    if(algo_mech_type == LOCALIZATION) {
        (*CHECK_GAP_CC) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_UPDT) = nPop / 3 / 2 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_SYNC) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_EXCH) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*NUPDT) = 10;
    }

    //////////////////////////////////////////////////////////////////////////
    if(flag_grp_predefined) {
        (*usedIter_init_grp) = 0;
    } else {
        (*usedIter_init_grp) = 0 + //initialize pop for grouping
                               nDim * NumControlAnalysis * NumRepControlAnalysis + //ControlVariableAnalysis
                               (nDim) * (nDim + 1);//InterdependenceAnalysis
    }
    //
    (*usedIter_initPop) = 0;
    if(algo_mech_type == LOCALIZATION ||
       algo_mech_type == DECOMPOSITION) {
        (*usedIter_initPop) = nPop;
    } else if(algo_mech_type == NONDOMINANCE) {
        for(int i = 0; i <= nObj; i++)
            (*usedIter_initPop) += Groups_sizes[i] * nPop_all[i];
    } else {
        if(0 == mpi_rank)
            printf("%s: No such algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    //
    (*usedIter_init) = (*usedIter_init_grp) + (*usedIter_initPop);
    (*iter) = (*usedIter_init);
    (*remIter) = maxIter - (*iter);
    if(algo_mech_type == LOCALIZATION) {
        (*iter_per_gen) = mpi_size * (*num_trail_per_gen);
    } else if(algo_mech_type == DECOMPOSITION ||
              algo_mech_type == NONDOMINANCE) {
        (*iter_per_gen) = 0;
        for(int i = 0; i <= nObj; i++)
            (*iter_per_gen) += Groups_sizes[i] * nPop_all[i];
    } else {
        if(0 == mpi_rank)
            printf("%s: No such algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpi_rank == 0)
        printf("st_global_p.iter_per_gen / st_global_p.remIter / st_global_p.iter / st_global_p.maxIter : %d/%d/%d/%d.\n",
               (*iter_per_gen), (*remIter), (*iter), maxIter);
    (*generatMax) = (int)((double)(*remIter) / (*iter_per_gen));
    (*generation) = 0;
}

void updateGenNum()
{
    int nPop = st_global_p.nPop;
    int nObj = st_global_p.nObj;
    int* nPop_all = st_MPI_p.nPop_all;
    int* Groups_sizes = st_grp_info_p.vec_sizeGroups;
    int nPop_mine = st_global_p.nPop_mine;
    int* num_trail_per_gen = &st_global_p.num_trail_per_gen;
    int* num_selected = &st_global_p.num_selected;
    int* num_exploit_per_gen = &st_global_p.num_exploit_per_gen;
    int* num_trail_per_gen_conve = &st_global_p.num_trail_per_gen_conve;
    int* num_trail_per_gen_diver = &st_global_p.num_trail_per_gen_diver;
    int* CHECK_GAP_CC = &st_global_p.CHECK_GAP_CC;
    int* CHECK_GAP_UPDT = &st_global_p.CHECK_GAP_UPDT;
    int* CHECK_GAP_SYNC = &st_global_p.CHECK_GAP_SYNC;
    int* CHECK_GAP_EXCH = &st_global_p.CHECK_GAP_EXCH;
    int* NUPDT = &st_global_p.NUPDT;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size = st_MPI_p.mpi_size;
    int* iter = &st_global_p.iter;
    int* remIter = &st_global_p.remIter;
    int maxIter = st_global_p.maxIter;
    int* iter_per_gen = &st_global_p.iter_per_gen;
    int* generatMax = &st_global_p.generatMax;
    int* generation = &st_global_p.generation;
    //
    int min_num;
    MPI_Allreduce(&nPop_mine, &min_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    (*num_trail_per_gen) = min_num > MAX_UPDATE_NUM_PER_GEN ? MAX_UPDATE_NUM_PER_GEN : min_num;
    //
    (*num_selected) = nPop / 10;
    MPI_Barrier(MPI_COMM_WORLD);
    (*num_exploit_per_gen) = 1;
    if((*num_exploit_per_gen) == 0)(*num_exploit_per_gen)++;
    (*num_trail_per_gen_conve) = min_num / nObj;
    if((*num_trail_per_gen_conve) == 0)(*num_trail_per_gen_conve)++;
    (*num_trail_per_gen_diver) = min_num / nObj;
    if((*num_trail_per_gen_diver) == 0)(*num_trail_per_gen_diver)++;
    //
    (*CHECK_GAP_CC) = 5;
    (*CHECK_GAP_UPDT) = 5;
    (*CHECK_GAP_SYNC) = 10;
    (*CHECK_GAP_EXCH) = 30;
    (*NUPDT) = 50;
    if(algo_mech_type == LOCALIZATION) {
        (*CHECK_GAP_CC) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_UPDT) = nPop / 3 / 2 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_SYNC) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*CHECK_GAP_EXCH) = nPop / 3 / (*num_trail_per_gen) + 1;
        (*NUPDT) = 10;
    }

    //////////////////////////////////////////////////////////////////////////
    (*remIter) = maxIter - (*iter);
    if(algo_mech_type == LOCALIZATION) {
        (*iter_per_gen) = mpi_size * (*num_trail_per_gen);
    } else if(algo_mech_type == DECOMPOSITION ||
              algo_mech_type == NONDOMINANCE) {
        (*iter_per_gen) = 0;
        for(int i = 0; i <= nObj; i++) {
            (*iter_per_gen) += Groups_sizes[i] * nPop_all[i];
        }
    } else {
        if(0 == mpi_rank)
            printf("%s: No such algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    (*generatMax) = (int)((double)(*remIter) / (*iter_per_gen)) + (*generation);
    //
    return;
}
