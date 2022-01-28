#include "global.h"
#include "time.h"

void set_para(int npop, int ndim, int nobj, int narch, int maxIter, char* func_name, int iRun)
{
    //////////////////////////////////////////////////////////////////////////
    strcpy(st_global_p.testInstance, func_name);
    st_global_p.nObj = nobj;
    st_global_p.nDim = ndim;
    st_global_p.nPop = npop;
    st_archive_p.nArch = narch;
    st_archive_p.nArch_sub = st_archive_p.nArch;//20;
    st_global_p.maxIter = maxIter;
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.popSize_sub_max = npop;
    st_archive_p.nArch_sub_max = narch;
    20;
    narch;
    st_archive_p.nArch_sub_before = 20; // narch;
    st_ctrl_p.flag_multiPop = 0;
    st_pop_best_p.n_best_history = 7;
    if(st_global_p.nObj == 2) st_utility_p.utility_threshold = 0.01;
    else st_utility_p.utility_threshold = 0.03;
    st_ctrl_p.optimization_tag = OPTIMIZE_CONVER_VARS;
    //////////////////////////////////////////////////////////////////////////
    st_global_p.tag_strct_updated = FLAG_OFF;
    st_global_p.iter = 0;
    st_global_p.iter_each = 0;
    st_archive_p.cnArch = 0;
    st_archive_p.cnArch_exchange = 0;
    st_archive_p.cnArchEx = 0;
    st_archive_p.cnArchOld = 0;
    st_repo_p.nRep = 0;
    st_global_p.nonDominateSize = 0;
    st_global_p.nPop_mine = 0;
    st_global_p.nPop_exchange = 0;
    st_pop_best_p.cn_best_history = 0;
    st_pop_best_p.i_best_history = 0;
    st_pop_comm_p.n_neighbor_left = 0;
    st_pop_comm_p.n_neighbor_right = 0;
    st_pop_comm_p.n_weights_left = 0;
    st_pop_comm_p.n_weights_right = 0;
    st_pop_comm_p.n_weights_mine = 0;
    //////////////////////////////////////////////////////////////////////////
    st_grp_info_p.numGROUP = st_global_p.nDim / 100;
    st_MPI_p.cur_grp_index = 0;
    st_MPI_p.cur_pop_index = 0;
    st_ctrl_p.type_grp_loop = LOOP_NONE;
    st_ctrl_p.type_pop_loop = LOOP_NONE;
    //////////////////////////////////////////////////////////////////////////
    st_grp_info_p.minGroupSize = 111; //30;// 50;
    //if(strct_ctrl_para.algo_mech_type == DECOMPOSITION)
    //    strct_grp_info_vals.minGroupSize = 30;// 50;
    //else if(strct_ctrl_para.algo_mech_type == NONDOMINANCE)
    //    strct_grp_info_vals.minGroupSize = 100;
    st_grp_info_p.maxGroupSize = (int)(1.2 * st_grp_info_p.minGroupSize);
    st_grp_info_p.limitDiverIndex = st_grp_info_p.minGroupSize / 2;
    st_ctrl_p.cur_trace = 0;
    st_ctrl_p.cur_run = iRun;
    st_ctrl_p.cur_MPI_cnt = 0;
    st_indicator_p.vec_TIME_indicator[st_ctrl_p.cur_run] = 0;
    rand_type =
        RAND_UNIF;
    RAND_TENT;
    st_DE_p.candid_num = 10;
    st_DE_p.nHistSHADE = 50;
    st_DE_p.iHistSHADE = 0;
    st_pop_evo_cur.curSize_inferior = 0;
    st_archive_p.curSize_inferior = 0;
    //////////////////////////////////////////////////////////////////////////
    setParaDefault();
    readParaFromFile();
    checkParas();
    set_init_rand_type();
    for(int i = 0; i < st_MPI_p.mpi_rank; i++) pointer_gen_rand();
    show_para_A();
    show_para_Prob();
    return;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_DPCC()
{
    MPI_Comm_rank(MPI_COMM_WORLD, &st_MPI_p.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &st_MPI_p.mpi_size);
    //////////////////////////////////////////////////////////////////////////
    allocateMemory_grouping_info();
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("allocateMemory_grouping_info(); \n");
#endif
    EMO_setLimits(st_global_p.testInstance, st_global_p.minLimit, st_global_p.maxLimit, st_global_p.nDim);
    //EMO_adjust_constraint_penalty(0.0, strct_global_paras.maxIter);
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("EMO_setLimits(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    /*grouping
    property, interdependence
    population initialization*/
    st_grp_info_p.flag_predefined = 1;
    if(st_ctrl_p.type_test == MY_TYPE_NORMAL) {
        if(!strcmp(st_global_p.testInstance, "WDCN")) {
            //grouping_variables_rand_unif(4);
            grouping_variables_WDCN();
            //grouping_variables();
        } else if(!strcmp(st_global_p.testInstance, "ARRANGE2D")) {
            grouping_variables_ARRANGE2D();
        } else if(!strcmp(st_global_p.testInstance, "HDSN_URBAN")) {
            grouping_variables_HDSN_URBAN();
        } else if(!strcmp(st_global_p.testInstance, "IWSN_S_1F")) {
            //grouping_variables_IWSN_S_1F();
            //grouping_variables_IWSN_S_1F_24();
            //grouping_variables_rand_unif(6);
            grouping_variables_IWSN_S_1F_with_nG(5);
            //grouping_variables_rand_unif(5);
        } else if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
            //grouping_variables_EdgeComputation();
            grouping_variables_unif(4, 1);
        } else {
            allocateMemory_grouping_ana();
            grouping_variables();
            freeMemory_grouping_ana();
            st_grp_info_p.flag_predefined = 0;
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
        if(st_ctrl_p.type_grouping == GROUPING_TYPE_SPECTRAL_CLUSTERING)
            grouping_variables_classify_cluster_spectral();
        else
            grouping_variables_classify_random();
    } else if(st_ctrl_p.type_test == MY_TYPE_LeNet ||
              st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE ||
              st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        //grouping_variables_LeNet();
        grouping_variables_LeNet_less();
    } else if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        grouping_variables_NN_indus();
    } else if(st_ctrl_p.type_test == MY_TYPE_CFRNN_CLASSIFY) {
        grouping_variables_CFRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO1_FRNN) {
        grouping_variables_EVO1_FRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO2_FRNN) {
        grouping_variables_EVO2_FRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO3_FRNN) {
        grouping_variables_EVO3_FRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO4_FRNN) {
        grouping_variables_EVO4_FRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO5_FRNN) {
        grouping_variables_EVO5_FRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO_FRNN_PREDICT) {
        grouping_variables_EVO_FRNN(frnn_MOP_Predict->xType);
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO_DFRNN_PREDICT) {
        grouping_variables_EVO_FRNN(frnn_MOP_Predict_dfrnn->xType);
    } else if(st_ctrl_p.type_test == MY_TYPE_INTRUSION_DETECTION_CLASSIFY) {
        grouping_variables_IntrusionDetection_Classify();
    } else if(st_ctrl_p.type_test == MY_TYPE_ACTIVITY_DETECTION_CLASSIFY) {
        grouping_variables_ActivityDetection_Classify();
    } else if(st_ctrl_p.type_test == MY_TYPE_RecSys_SmartCity) {
        grouping_variables_RS_SC();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO_CNN) {
        grouping_variables_unif(5, 1);
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO_CFRNN) {
        grouping_variables_evoCFRNN();
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO_MOBILE_SINK) {
        if(strstr(st_global_p.testInstance, "evoMobileSink_FRNN")) {
            grouping_variables_EVO_FRNN(frnn_mop_mobile_sink->xType);
        } else if(strstr(st_global_p.testInstance, "evoMobileSink_GEP_only")) {
            grouping_variables_unif(2, 0);
            // st_ctrl_p.type_xor_evo_mut = XOR_EVO_MUT_ADAP;
        } else {
            grouping_variables_unif(2, 0);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s:Problem type error, exiting...\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_PROBLEM_TYPE);
    }
    st_indicator_p.vec_TIME_grouping[st_ctrl_p.cur_run] = (clock() - st_global_p.start_time) / CLOCKS_PER_SEC;
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("grouping_variables(); \n");
#endif
    /*output grouping results*/
    if(st_MPI_p.mpi_rank == 0) {
        char debugName[MAX_CHAR_ARR_SIZE];
        sprintf(debugName, "GROUP/GROUP_%s_%s_OBJ%d_VAR%d_MPI%d_RUN%d_%ld",
                st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.global_time);
        st_global_p.debugFpt = fopen(debugName, "w");
        output_group_info_brief();
        fclose(st_global_p.debugFpt);
    }
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("output_group_info_brief(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    if(st_grp_ana_p.numDiverIndexes == 0)
        st_ctrl_p.opt_diverVar_separately = FLAG_OFF;
    //////////////////////////////////////////////////////////////////////////
    /*MPI parallel structure*/
    allocateMemory_MPI();
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("allocateMemory_MPI(); \n");
#endif
    setMPI();
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("setMPI(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    allocateMemory();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("allocateMemory(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    sprintf(st_global_p.objsal, "PF/%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d",
            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
            st_MPI_p.mpi_size, st_ctrl_p.cur_run);
    sprintf(st_global_p.varsal, "PS/%s_VAR_%s_OBJ%d_VAR%d_MPI%d_RUN%d",
            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
            st_MPI_p.mpi_size, st_ctrl_p.cur_run);
    //////////////////////////////////////////////////////////////////////////
    /*time serial, one main process*/
    if(st_MPI_p.mpi_rank == 0) st_global_p.start_time = clock();
    //////////////////////////////////////////////////////////////////////////
    /*parameters*/
    initializePara();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("initializePara(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    show_para_B();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("show_para_B(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    /*MPI process local initialization*/
    if(algo_mech_type == LOCALIZATION ||
       algo_mech_type == DECOMPOSITION)
        localInitialization();
    else if(algo_mech_type == NONDOMINANCE)
        localInitialization_ND();
    else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("localInitialization \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    initializePopulation();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == st_MPI_p.mpi_size - 1) printf("initializePopulation(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    /*number of generation*/
    initializeGenNum();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("initializeGenNum();. MAX=%d ", st_global_p.generatMax);
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(algo_mech_type == LOCALIZATION) { //DPCCMOEA
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        update_xBest(UPDATE_INIT, 0, NULL, NULL, NULL, NULL);
        update_xBest(UPDATE_GIVEN, st_global_p.nPop, NULL, st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
        update_xBest_history(UPDATE_INIT, 0, NULL, NULL, NULL, NULL);
        update_xBest_history(UPDATE_GIVEN, st_global_p.nPop, NULL, st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("Update_xBest(); done.\n");
#endif
        transformPop(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("transformPop(); \n");
#endif
        updateNeighborTable(WEIGHT_BASED, st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("updateNeighborTable(); ~ niche = %d \n", st_decomp_p.niche);
#endif
        //////////////////////////////////////////////////////////////////////////
        /*Main Loop*/
        while(st_global_p.generation < st_global_p.generatMax) {
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0)
                printf("while(st_global_p.generation < st_global_p.generatMax) - %d~%d.\n",
                       st_global_p.generation, st_global_p.generatMax);
#endif
            //EMO_adjust_constraint_penalty(strct_global_paras.iter - strct_global_paras.usedIter_init_grp, strct_global_paras.maxIter - strct_global_paras.usedIter_init_grp);
            save_during_operation();
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation().\n");
#endif
            //exchange
            exchangeInfo_DPCCMOEA();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("exchangeInfo_DPCCMOEA(); \n");
#endif
            //evolution
            cooperativeCoevolution(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("cooperativeCoevolution().\n");
#endif
            if(st_global_p.generation % st_global_p.CHECK_GAP_SYNC == st_global_p.CHECK_GAP_SYNC - 1) {
                population_synchronize(st_ctrl_p.algo_mech_type);
                transfer_x_neighbor();
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("population_synchronize().\n");
#endif
                update_xBest_history(UPDATE_GIVEN, st_global_p.nPop_mine, NULL,
                                     st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("update_xBest_history().\n");
#endif
            }
            //////////////////////////////////////////////////////////////////////////
            //if(strct_global_paras.generation % CHECK_GAP_EXCH == CHECK_GAP_EXCH - 1) {
            //    exchangeInformation();
            //    update_xBest_history(UPDATE_WHOLE, NULL, NULL, NULL);
            //}
            //update_xBest_history(UPDATE_WHOLE, NULL, NULL);
            st_global_p.iter += st_global_p.iter_per_gen;
            st_global_p.generation++;
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("GEN: %d~%d.\n", st_global_p.generation, st_global_p.generatMax);
#endif
            save_during_operation();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation(); \n");
#endif
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
            updateStructure();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("updateStructure().\n");
#endif
        }
        collectDecompositionArchive(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("collectDecompositionArchive().\n");
#endif
        if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
            transform_var_feature(st_archive_p.var, st_archive_p.var_feature, st_global_p.nPop);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("transform_var_feature().\n");
#endif
        }
        //	collectNDArchive();
        if(st_MPI_p.mpi_rank == 0) {
            st_global_p.end_time = clock();
            st_global_p.duration = (st_global_p.end_time - st_global_p.start_time) / CLOCKS_PER_SEC;
            st_indicator_p.vec_TIME_all[st_ctrl_p.cur_run] = st_global_p.duration;
            st_global_p.nonDominateSize = st_archive_p.cnArch;
            st_global_p.fptvar = fopen(st_global_p.varsal, "w");
            if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
                save_int(st_global_p.fptvar, st_archive_p.var_feature, st_archive_p.cnArch, TH_N_FEATURE, 0);
            } else {
                if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
                    save_double_as_int(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                } else if(st_global_p.nDim <= 10000) {
                    save_double(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                }
            }
            fclose(st_global_p.fptvar);
        }
        show_indicator_vars_simp(FINAL_TAG);
    } else if(algo_mech_type == DECOMPOSITION) { //DPCCMOLSEA
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        update_xBest(UPDATE_INIT, 0, NULL, NULL, NULL, NULL);
        update_xBest(UPDATE_GIVEN, st_global_p.nPop, NULL, st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
        update_xBest_history(UPDATE_INIT, 0, NULL, NULL, NULL, NULL);
        update_xBest_history(UPDATE_GIVEN, st_global_p.nPop, NULL, st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("Update_xBest(); done.\n");
#endif
        transformPop(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("transformPop(); \n");
#endif
        updateNeighborTable(WEIGHT_BASED, st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("updateNeighborTable(); \n");
#endif
        //////////////////////////////////////////////////////////////////////////
        /*Main Loop*/
        while(st_global_p.generation < st_global_p.generatMax) {
            //EMO_adjust_constraint_penalty(strct_global_paras.iter - strct_global_paras.usedIter_init_grp, strct_global_paras.maxIter - strct_global_paras.usedIter_init_grp);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0)
                printf("while(strct_global_paras.generation < strct_global_paras.generatMax) - %d~%d.\n",
                       st_global_p.generation, st_global_p.generatMax);
#endif
            save_during_operation();
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation().\n");
#endif
            //evolution
            cooperativeCoevolution(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("cooperativeCoevolution().\n");
#endif
            if(st_global_p.generation % st_global_p.CHECK_GAP_SYNC == st_global_p.CHECK_GAP_SYNC - 1) {
                population_synchronize(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("population_synchronize().\n");
#endif
                update_xBest_history(UPDATE_GIVEN, st_global_p.nPop, NULL,
                                     st_pop_evo_cur.var, st_pop_evo_cur.obj, st_qu_p.rot_angle_cur);
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("update_xBest_history().\n");
#endif
            }
            //////////////////////////////////////////////////////////////////////////
            //if(strct_global_paras.generation % CHECK_GAP_EXCH == CHECK_GAP_EXCH - 1) {
            //    exchangeInformation();
            //    update_xBest_history(UPDATE_WHOLE, NULL, NULL, NULL);
            //}
            //update_xBest_history(UPDATE_WHOLE, NULL, NULL);
            st_global_p.iter += st_global_p.iter_per_gen;
            st_global_p.generation++;
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("GEN: %d~%d.\n", st_global_p.generation, st_global_p.generatMax);
#endif
            save_during_operation();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation(); \n");
#endif
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
            updateStructure();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("updateStructure().\n");
#endif
        }
        //////////////////////////////////////////////////////////////////////////
        collectDecompositionArchive(st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("collectDecompositionArchive().\n");
#endif
        if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
            transform_var_feature(st_archive_p.var, st_archive_p.var_feature, st_global_p.nPop);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("transform_var_feature().\n");
#endif
        }
        //	collectNDArchive();
        if(st_MPI_p.mpi_rank == 0) {
            st_global_p.end_time = clock();
            st_global_p.duration = (st_global_p.end_time - st_global_p.start_time) / CLOCKS_PER_SEC;
            st_indicator_p.vec_TIME_all[st_ctrl_p.cur_run] = st_global_p.duration;
            st_global_p.nonDominateSize = st_archive_p.cnArch;
            st_global_p.fptvar = fopen(st_global_p.varsal, "w");
            if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
                save_int(st_global_p.fptvar, st_archive_p.var_feature, st_archive_p.cnArch, TH_N_FEATURE, 0);
            } else {
                if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
                    save_double_as_int(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                } else if(st_global_p.nDim <= 10000) {
                    save_double(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                }
            }
            fclose(st_global_p.fptvar);
        }
        show_indicator_vars_simp(FINAL_TAG);
    } else if(algo_mech_type == NONDOMINANCE) { //DPCCMOLSIA
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        refinePop_ND(INIT_TAG, st_ctrl_p.algo_mech_type);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("refinePop_ND(INIT_TAG); \n");
#endif
        //////////////////////////////////////////////////////////////////////////
        /*Main Loop*/
        while(st_global_p.generation < st_global_p.generatMax) {
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0)
                printf("while(strct_global_paras.generation < strct_global_paras.generatMax) - %d~%d.\n",
                       st_global_p.generation, st_global_p.generatMax);
#endif
            save_during_operation_ND();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation_ND(); \n");
#endif
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
            //evolution
            cooperativeCoevolution_ND();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("cooperativeCoevolution_ND(); \n");
#endif
            //exchange
            if(st_global_p.generation % st_global_p.CHECK_GAP_EXCH == 0 && st_global_p.generation) {
                exchangeInfo_ND();
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("exchangeInformation_ND(); \n");
#endif
            } else { //if(strct_global_paras.generation%CHECK_GAP_SYNC==CHECK_GAP_SYNC-1)
                population_synchronize_ND();
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("population_synchronize_ND(); \n");
#endif
            }
            //
            st_global_p.iter += st_global_p.iter_per_gen;
            st_global_p.generation++;
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("GEN: %d~%d.\n", st_global_p.generation, st_global_p.generatMax);
#endif
            //
            save_during_operation_ND();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("save_during_operation_ND(); \n");
#endif
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && st_ctrl_p.cur_trace > 1) {
                if(st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] <
                   st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] * 1.1 &&
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 1] >=
                   st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace - 2]) {
                    st_indicator_p.count_VALIDATION++;
                } else {
                    st_indicator_p.count_VALIDATION = 0;
                }
                if(st_indicator_p.count_VALIDATION >= st_indicator_p.threshold_VALIDATION) {
                    st_indicator_p.vec_NTRACE_all_VALIDATION[st_ctrl_p.cur_run] = st_ctrl_p.cur_trace - 2;
                    //break;
                }
            }
            updateStructure_ND();
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("updateStructure_ND(); \n");
#endif
        }
        //////////////////////////////////////////////////////////////////////////
        collectNDArchive();
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("collectNDArchive(); \n");
#endif
        //////////////////////////////////////////////////////////////////////////
        //
        if(st_MPI_p.mpi_rank == 0) {
            st_global_p.end_time = clock();
            st_global_p.duration = (st_global_p.end_time - st_global_p.start_time) / CLOCKS_PER_SEC;
            st_indicator_p.vec_TIME_all[st_ctrl_p.cur_run] = st_global_p.duration;
            st_global_p.nonDominateSize = st_archive_p.cnArch;
            st_global_p.fptvar = fopen(st_global_p.varsal, "w");
            if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
                save_int(st_global_p.fptvar, st_archive_p.var_feature, st_archive_p.cnArch, TH_N_FEATURE, 0);
            } else {
                if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
                    save_double_as_int(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                } else if(st_global_p.nDim <= 10000) {
                    save_double(st_global_p.fptvar, st_archive_p.var, st_archive_p.cnArch, st_global_p.nDim, 0);
                }
            }
            fclose(st_global_p.fptvar);
        }
        memcpy(st_archive_p.var_Ex, st_archive_p.var, st_archive_p.cnArch * st_global_p.nDim * sizeof(double));
        memcpy(st_archive_p.obj_Ex, st_archive_p.obj, st_archive_p.cnArch * st_global_p.nObj * sizeof(double));
        st_archive_p.cnArchEx = st_archive_p.cnArch;
        show_indicator_vars_simp(FINAL_TAG);
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    //////////////////////////////////////////////////////////////////////////
    freeMemory();
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void save_during_operation()
{
    if((st_global_p.iter - st_global_p.usedIter_init) >=
       st_ctrl_p.cur_trace * (st_global_p.maxIter - st_global_p.usedIter_init) / NTRACE &&
       st_ctrl_p.cur_trace <= NTRACE) {
        collectDecompositionArchive(st_ctrl_p.algo_mech_type);
        show_indicator_vars_simp(UPDATE_TAG);
        st_ctrl_p.cur_trace++;
    }
    return;
}

void updateStructure()
{
    if((st_global_p.iter - st_global_p.usedIter_init) >=
       st_ctrl_p.cur_MPI_cnt * (st_global_p.maxIter - st_global_p.usedIter_init) / st_global_p.NUPDT) {
        gatherInfoBeforeUpdateStructure(st_ctrl_p.algo_mech_type);
        if(st_ctrl_p.multiPop_mode == MP_I || st_ctrl_p.multiPop_mode == MP_II
           || st_ctrl_p.multiPop_mode == MP_ADAP) {
            build_MPI_structure(UPDATE_MPI_STRUCTURE, UPDATE_TAG);
            localInitialization();
            updateNeighborTable(WEIGHT_BASED, st_ctrl_p.algo_mech_type);
        } else if(st_ctrl_p.multiPop_mode == MP_0 || st_ctrl_p.multiPop_mode == MP_III) {
        } else {
            if(st_MPI_p.mpi_rank == 0) {
                printf("%s:MP_MODE selection is wrong, no other algorithm available.\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        }
        scatterInfoAfterUpdateStructure(st_ctrl_p.algo_mech_type);
        //if (strct_MPI_info.color_master_species)
        //	update_xBest(0, NULL, NULL);
        //exchangeReferencePoint();
        updateGenNum();
        //for (int i = 0; i < strct_global_paras.nPop; i++) strct_utility_info.utility[i] = 1.0;
        st_ctrl_p.cur_MPI_cnt++;
        st_global_p.tag_strct_updated = FLAG_ON;
    }
    return;
}

void save_during_operation_ND()
{
    if((st_global_p.iter - st_global_p.usedIter_init) >=
       st_ctrl_p.cur_trace * (st_global_p.maxIter - st_global_p.usedIter_init) / NTRACE &&
       st_ctrl_p.cur_trace <= NTRACE) {
        collectNDArchiveEx();
        show_indicator_vars_simp(UPDATE_TAG);
        st_ctrl_p.cur_trace++;
    }
    return;
}

void updateStructure_ND()
{
    if((st_global_p.iter - st_global_p.usedIter_init) >=
       st_ctrl_p.cur_MPI_cnt * (st_global_p.maxIter - st_global_p.usedIter_init) / st_global_p.NUPDT) {
        //gatherInfoBeforeUpdateStructure();
        if(st_ctrl_p.multiPop_mode == MP_I) {
            build_MPI_structure(UPDATE_MPI_STRUCTURE_ND, UPDATE_TAG);
            localInitialization_ND();
            //refinePop_ND(UPDATE_TAG_BRIEF);
        } else if(st_ctrl_p.multiPop_mode == MP_II) {
            build_MPI_structure(UPDATE_MPI_STRUCTURE_ND, UPDATE_TAG);
            localInitialization_ND();
            //refinePop_ND(UPDATE_TAG);
        } else if(st_ctrl_p.multiPop_mode == MP_ADAP) {
            build_MPI_structure(UPDATE_MPI_STRUCTURE_ND, UPDATE_TAG);
            localInitialization_ND();
            //refinePop_ND(UPDATE_TAG);
        } else if(st_ctrl_p.multiPop_mode == MP_0 || st_ctrl_p.multiPop_mode == MP_III) {
        } else {
            if(st_MPI_p.mpi_rank == 0) {
                printf("%s:MP_MODE selection is wrong, no other algorithm available.\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        }
        //scatterInfoAfterUpdateStructure();
        //if (strct_MPI_info.color_master_species)
        //	update_xBest(0, NULL, NULL);
        //exchangeReferencePoint();
        updateGenNum();
        //for (int i = 0; i < strct_global_paras.nPop; i++) strct_utility_info.utility[i] = 1.0;
        st_ctrl_p.cur_MPI_cnt++;
    }
    return;
}
