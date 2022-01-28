# include "global.h"
# include <math.h>
//
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// DECOMPOSITION
void cooperativeCoevolution(int algoMechType)
{
    int mpi_rank = st_MPI_p.mpi_rank;
    int color_master_subPop = st_MPI_p.color_master_subPop;
    int generation = st_global_p.generation;
    int CHECK_GAP_CC = st_global_p.CHECK_GAP_CC;
    int CHECK_GAP_UPDT = st_global_p.CHECK_GAP_UPDT;
    int nDim = st_global_p.nDim;
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nPop_cur = nPop;
    double* weights_mine = st_decomp_p.weights_mine;
    double* weights_all = st_decomp_p.weights_all;
    double* weights_cur = weights_all;
    if(algoMechType == LOCALIZATION) {
        nPop_cur = nPop_mine;
        weights_cur = weights_mine;
    } else if(algoMechType == DECOMPOSITION) {
        nPop_cur = nPop;
        weights_cur = weights_all;
    } else {
        if(0 == mpi_rank)
            printf("%s: Improper algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    int* optimization_tag = &st_ctrl_p.optimization_tag;
    int numDiverIndexes = st_grp_ana_p.numDiverIndexes;
    int numConverIndexes = st_grp_ana_p.numConverIndexes;
    int* iUpdt = &st_pop_comm_p.iUpdt;
    int* Sflag = st_DE_p.Sflag;
    double* fitImprove = st_decomp_p.fitImprove;
    int* countFitImprove = st_decomp_p.countFitImprove;
    char* testInstance = st_global_p.testInstance;
    int* tag_selection = st_ctrl_p.tag_selection;
    double* cur_var = st_pop_evo_cur.var;
    double  th_select = st_decomp_p.th_select;
    int* parent_type = st_decomp_p.parent_type;
    int type_grp_loop = st_ctrl_p.type_grp_loop;
    int color_pop = st_MPI_p.color_pop;
    int* cur_grp_index = &st_MPI_p.cur_grp_index;
    int* Groups_sizes = st_grp_info_p.vec_sizeGroups;
    int CLONALG_tag = st_ctrl_p.CLONALG_tag;
    int color_subPop = st_MPI_p.color_subPop;
    int* ns_pops = st_MPI_p.ns_pops;
    int* nf_pops = st_MPI_p.nf_pops;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    if(algoMechType == LOCALIZATION ||
       (algoMechType == DECOMPOSITION && color_master_subPop)) {
        if(generation % CHECK_GAP_CC == CHECK_GAP_CC - 1) {
            total_utility(nPop_cur, weights_cur);
            double utility_mean = st_utility_p.utility_mean;
            double utility_threshold = st_utility_p.utility_threshold;
            if(utility_mean < utility_threshold) {
                if((*optimization_tag) == OPTIMIZE_CONVER_VARS)
                    (*optimization_tag) = OPTIMIZE_DIVER_VARS;
                else
                    (*optimization_tag) = OPTIMIZE_CONVER_VARS;
            }
        }
    }
    if(numDiverIndexes == 0)(*optimization_tag) = OPTIMIZE_CONVER_VARS;
    if(numConverIndexes == 0)(*optimization_tag) = OPTIMIZE_DIVER_VARS;
    if(algoMechType == LOCALIZATION) {}
    else if(algoMechType == DECOMPOSITION)(*optimization_tag) = OPTIMIZE_CONVER_VARS;
    //////////////////////////////////////////////////////////////////////////
    (*iUpdt) = 0;
    if(algoMechType == LOCALIZATION &&
       generation % CHECK_GAP_UPDT == CHECK_GAP_UPDT - 1) {
        update_population_DECOM_from_transNeigh();
    }
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < nPop; i++) {
        Sflag[i] = 0;
        fitImprove[i] = 0.0;
        countFitImprove[i] = 0;
    }
    //////////////////////////////////////////////////////////////////////////
    if(!strncmp(testInstance, "FeatureSelection_", 17) ||
       !strncmp(testInstance, "FeatureSelectionTREE_", 21)) {
        for(int i = 0; i < nDim; i++) {
            tag_selection[i] = 0;
        }
        for(int i = 0; i < nPop; i++) {
            for(int j = 0; j < nDim; j++) {
                if((int)cur_var[i * nDim + j]) {
                    tag_selection[j]++;
                }
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < nPop; i++) {
        if(flip_r((float)th_select)) {
            parent_type[i] = PARENT_LOCAL;
        } else {
            parent_type[i] = PARENT_GLOBAL;
        }
    }
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("cooperativeCoevolution() initialized. \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    generate_para_all();
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("generate_para_all(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    localAssignGroup(color_pop, (*cur_grp_index));
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("localAssignGroup(); \n");
#endif
    if(type_grp_loop == LOOP_GRP)(*cur_grp_index) = ((*cur_grp_index) + 1) % Groups_sizes[color_pop];
    //////////////////////////////////////////////////////////////////////////
    if(CLONALG_tag == FLAG_OFF) {
        mainLoop_CC();
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("mainLoop_CC(); \n");
#endif
    } else {
        mainLoop_CC_CLONALG();
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("mainLoop_CC_CLONALG(); \n");
#endif
    }
    //////////////////////////////////////////////////////////////////////////
    int cur_indx_ns_nf = color_pop * nDim + color_subPop;
    for(int i = 0; i < nPop; i++) {
        if(Sflag[i]) {
            ns_pops[cur_indx_ns_nf]++;
        } else {
            nf_pops[cur_indx_ns_nf]++;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    update_para_statistics();
    //////////////////////////////////////////////////////////////////////////
    synchronizeObjectiveBests(algoMechType);
    synchronizeReferencePoint(algoMechType);
    //
    return;
}

void mainLoop_CC()
{
    int* num_selected = &st_global_p.num_selected;
    int* selectedIndv = st_global_p.selectedIndv;
    int num_trail_per_gen = st_global_p.num_trail_per_gen;
    int color_master_subPop = st_MPI_p.color_master_subPop;
    int MFI_update_tag = st_ctrl_p.MFI_update_tag;
    int optimization_tag = st_ctrl_p.optimization_tag;
    int updatePop_type = st_ctrl_p.updatePop_type;
    int type_test = st_ctrl_p.type_test;
    int type_dim_convert = st_ctrl_p.type_dim_convert;
    char* testInstance = st_global_p.testInstance;
    int tag_gather_after_evaluate = st_ctrl_p.tag_gather_after_evaluate;
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* repo_obj = st_repo_p.obj;
    double* weights_all = st_decomp_p.weights_all;//
    int  useSflag = 1; //
    int* Sflag = st_DE_p.Sflag;//
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;//
    int niche = st_decomp_p.niche;//
    int niche_local = st_decomp_p.niche_local;//
    int* tableNeighbor = st_decomp_p.tableNeighbor;//
    int* tableNeighbor_local = st_decomp_p.tableNeighbor;//
    int maxNneighb = nPop; //
    int* parent_type = st_decomp_p.parent_type;//
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int nPop_mine = st_global_p.nPop_mine;
    int* nPop_all = st_MPI_p.nPop_all;
    int color_pop = st_MPI_p.color_pop;
    int nPop_cur = nPop;
    if(algo_mech_type == LOCALIZATION) {
        nPop_cur = nPop_mine;
        weights_all = st_decomp_p.weights_mine;
        tableNeighbor_local = st_decomp_p.tableNeighbor_local;
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int ind_prob = color_pop - 1;
    int* ind_sub = (int*)calloc(nPop_cur, sizeof(int));
    if(color_pop) {
        for(int i = 0; i < nPop_cur; i++) ind_sub[i] = i;
        memcpy(repo_obj, cur_obj, nPop_cur * nObj * sizeof(double));
        qSortFrontObj(ind_prob, ind_sub, 0, nPop_cur - 1);
    }
    int i, j;
    if(algo_mech_type == LOCALIZATION) {
        (*num_selected) = num_trail_per_gen;
        int depth = 10 / (nPop / nPop_mine);
        depth = depth > 5 ? depth : 5;
        tour_selection(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
        for(i = 0; i < (*num_selected); i++) {
            j = selectedIndv[i];
            gen_offspring_selected_one(j, i, ind_sub);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("gen_offspring_selected_one(); \n");
#endif
            joinVar(j, i, optimization_tag);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("joinVar(); \n");
#endif
            if((type_test == MY_TYPE_LeNet || type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
               type_dim_convert == DIM_CONVERT_CNN) {
                double tmp[DIM_LeNet];
                convertVar_CNN(&osp_var[i * nDim], tmp);
                EMO_evaluate_problems(testInstance, tmp, &osp_obj[i * nObj], DIM_LeNet, 1, nObj);
            } else {
                EMO_evaluate_problems(testInstance, &osp_var[i * nDim], &osp_obj[i * nObj], nDim, 1, nObj);
            }
            update_idealpoint(&osp_obj[i * nObj]);
            update_population_DECOM(j, i, nPop_cur, osp_obj, osp_var,
                                    weights_all, useSflag, Sflag,
                                    rot_angle_offspring,
                                    niche, niche_local, tableNeighbor, tableNeighbor_local, maxNneighb, parent_type[i]);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("update_population_DECOM(); \n");
#endif
        }
        update_xBest(UPDATE_GIVEN, (*num_selected), NULL, osp_var, osp_obj, rot_angle_offspring);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("update_xBest(); \n");
#endif
        update_xBest_history(UPDATE_GIVEN, (*num_selected), NULL, osp_var, osp_obj, rot_angle_offspring);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("update_xBest_history(); \n");
#endif
    } else if(algo_mech_type == DECOMPOSITION) {
        (*num_selected) = nPop_all[color_pop];
        if((*num_selected) < nPop_cur) {
            int depth = 10;
            if(color_pop)
                tour_selection_sub(selectedIndv, (*num_selected), depth);
            else
                tour_selection(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
        } else
            for(int i = 0; i < (*num_selected); i++) selectedIndv[i] = i;
        if(color_master_subPop) {
            for(i = 0; i < (*num_selected); i++) {
                j = selectedIndv[i];
                gen_offspring_selected_one(j, i, ind_sub);
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
                if(st_MPI_p.mpi_rank == 0) printf("gen_offspring_selected_one(%d, %d, ind_sub); \n", j, i);
#endif
                joinVar(j, i, optimization_tag);
#ifdef DEBUG_TAG_TMP
                show_DeBug_info();
                MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
                if(st_MPI_p.mpi_rank == 0) printf("joinVar(%d, %d, optimization_tag); \n", j, i);
#endif
            }
        }
        scatter_evaluation_gather();
        if(color_master_subPop) {
            for(i = 0; i < (*num_selected); i++) {
                update_idealpoint(&osp_obj[i * nObj]);
            }
            update_xBest(UPDATE_GIVEN, (*num_selected), NULL, osp_var, osp_obj, rot_angle_offspring);
            update_xBest_history(UPDATE_GIVEN, (*num_selected), NULL, osp_var, osp_obj, rot_angle_offspring);
            for(i = 0; i < (*num_selected); i++) {
                j = selectedIndv[i];
                update_population_DECOM(j, i, nPop_cur, osp_obj, osp_var,
                                        weights_all, useSflag, Sflag,
                                        rot_angle_offspring,
                                        niche, niche_local, tableNeighbor, tableNeighbor_local, maxNneighb, parent_type[i]);
            }
        }
    }
    free(ind_sub);
    //
    return;
}

void mainLoop_CC_CLONALG()
{
    int* num_selected = &st_global_p.num_selected;
    int* selectedIndv = st_global_p.selectedIndv;
    int color_master_subPop = st_MPI_p.color_master_subPop;
    int MFI_update_tag = st_ctrl_p.MFI_update_tag;
    int optimization_tag = st_ctrl_p.optimization_tag;
    int updatePop_type = st_ctrl_p.updatePop_type;
    int type_test = st_ctrl_p.type_test;
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    int* nRep = &st_repo_p.nRep;
    double* weights_all = st_decomp_p.weights_all;//
    int  useSflag = 1; //
    int* Sflag = st_DE_p.Sflag;//
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;//
    int niche = st_decomp_p.niche;//
    int niche_local = st_decomp_p.niche_local;//
    int* tableNeighbor = st_decomp_p.tableNeighbor;//
    int* tableNeighbor_local = st_decomp_p.tableNeighbor;//
    int maxNneighb = nPop; //
    int* parent_type = st_decomp_p.parent_type;//
    int type_clone_selection = st_ctrl_p.type_clone_selection;
    int* indx = st_archive_p.indx;
    int mpi_rank = st_MPI_p.mpi_rank;
    int* cloneNum = st_global_p.cloneNum;
    double* fitCur = st_decomp_p.fitCur;
    int* weight_prefer_tag = st_decomp_p.weight_prefer_tag;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int nPop_cur = nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int* nPop_all = st_MPI_p.nPop_all;
    int color_pop = st_MPI_p.color_pop;
    if(algo_mech_type == LOCALIZATION) {
        nPop_cur = nPop_mine;
        weights_all = st_decomp_p.weights_mine;
        tableNeighbor_local = st_decomp_p.tableNeighbor_local;
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int ind_prob = color_pop - 1;
    int* ind_sub = (int*)calloc(nPop_cur, sizeof(int));
    if(color_pop) {
        for(int i = 0; i < nPop_cur; i++) ind_sub[i] = i;
        memcpy(repo_obj, cur_obj, nPop_cur * nObj * sizeof(double));
        qSortFrontObj(ind_prob, ind_sub, 0, nPop_cur - 1);
    }
    int i, j;
    int depth = 10;
    (*num_selected) = nPop_cur / 5;
    //////////////////////////////////////////////////////////////////////////
    if((algo_mech_type == DECOMPOSITION && color_master_subPop) ||
       algo_mech_type == LOCALIZATION) {
        if(type_clone_selection == CLONE_SLCT_ND1) {
            memcpy(repo_var, cur_var, nPop_cur * nDim * sizeof(double));
            memcpy(repo_obj, cur_obj, nPop_cur * nObj * sizeof(double));
            (*nRep) = nPop_cur;
            refineRepository_generateArchive();
            greedy_selection_DECOM(selectedIndv, (*num_selected));
            clone_DECOM();
            for(int i = 0; i < (*num_selected); i++) selectedIndv[i] = indx[selectedIndv[i]];
        } else if(type_clone_selection == CLONE_SLCT_ND2) {
            memcpy(repo_var, cur_var, nPop_cur * nDim * sizeof(double));
            memcpy(repo_obj, cur_obj, nPop_cur * nObj * sizeof(double));
            (*nRep) = nPop_cur;
            refineRepository_generateArchive();
            greedy_selection_ND(selectedIndv, (*num_selected));
            clone_ND();
            for(int i = 0; i < (*num_selected); i++) selectedIndv[i] = indx[selectedIndv[i]];
        } else if(type_clone_selection == CLONE_SLCT_ND_TOUR) {
            memcpy(repo_var, cur_var, nPop_cur * nDim * sizeof(double));
            memcpy(repo_obj, cur_obj, nPop_cur * nObj * sizeof(double));
            (*nRep) = nPop_cur;
            refineRepository_generateArchive();
            (*num_selected) = nPop_cur;
            depth = 10;
            tour_selection_ND(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
            for(int i = 0; i < (*num_selected); i++) cloneNum[i] = 1;
            for(int i = 0; i < (*num_selected); i++) selectedIndv[i] = indx[selectedIndv[i]];
        } else if(type_clone_selection == CLONE_SLCT_UTILITY_TOUR) {
            (*num_selected) = nPop_cur;
            depth = 10;
            tour_selection_repetitive(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
            for(int i = 0; i < (*num_selected); i++) cloneNum[i] = 1;
        } else if(type_clone_selection == CLONE_SLCT_AGGFIT_G) {
            (*num_selected) = nPop_cur;
            depth = 3;
            for(int i = 0; i < nPop_cur; i++) {
                fitCur[i] = fitnessFunction(&cur_obj[i * nObj], &weights_all[i * nObj]);
            }
            tour_selection_aggFit_greater(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
            for(int i = 0; i < (*num_selected); i++) cloneNum[i] = 1;
        } else if(type_clone_selection == CLONE_SLCT_AGGFIT_L) {
            (*num_selected) = nPop_cur;
            depth = 3;
            for(int i = 0; i < nPop_cur; i++) {
                fitCur[i] = fitnessFunction(&cur_obj[i * nObj], &weights_all[i * nObj]);
            }
            tour_selection_aggFit_less(NULL, nPop_cur, selectedIndv, (*num_selected), depth);
            for(int i = 0; i < (*num_selected); i++) st_global_p.cloneNum[i] = 1;
        } else if(type_clone_selection == CLONE_SLCT_PREFER) {
            (*num_selected) = nPop_cur;
            for(int i = 0; i < nPop_cur; i++) selectedIndv[i] = i;
            for(int i = 0; i < (*num_selected); i++) cloneNum[i] = 1;
        } else {
            if(0 == mpi_rank) {
                printf("No such st_ctrl_p.type_clone_selection\n");
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_CLONE_SELECTION_TYPE);
        }
        //
        if(type_clone_selection == CLONE_SLCT_ND1 ||
           type_clone_selection == CLONE_SLCT_ND2) {
            int* tmp_indx = (int*)calloc(nPop_cur, sizeof(int));
            for(int i = 0; i < nPop_cur; i++) tmp_indx[i] = i;
            shuffle(tmp_indx, (*num_selected));
            int count = 0;
            int count2 = 0;
            for(int i = 0; i < nPop_cur; i++) {
                for(int j = 0; j < weight_prefer_tag[i]; j++) {
                    while(cloneNum[tmp_indx[count]] <= 0) {
                        count++;
                        count = count % (*num_selected);
                    }
                    cloneNum[tmp_indx[count]]--;
                    count++;
                    count = count % (*num_selected);
                }
                if(weight_prefer_tag[i]) {
                    selectedIndv[(*num_selected) + count2] = i;
                    cloneNum[(*num_selected) + count2] = weight_prefer_tag[i];
                    count2++;
                }
            }
            (*num_selected) += count2;
            free(tmp_indx);
        } else {
            int* tmp_indx = (int*)calloc(nPop_cur, sizeof(int));
            for(int i = 0; i < nPop_cur; i++) tmp_indx[i] = i;
            shuffle(tmp_indx, nPop_cur);
            int count = 0;
            for(int i = 0; i < nPop_cur; i++) {
                for(int j = 0; j < weight_prefer_tag[i]; j++) {
                    selectedIndv[tmp_indx[count++]] = i;
                }
            }
            free(tmp_indx);
        }
        //////////////////////////////////////////////////////////////////////////
        int count = 0;
        for(i = 0; i < (*num_selected); i++) {
            for(j = 0; j < cloneNum[i]; j++) {
                int cur_ind = selectedIndv[i];
                gen_offspring_selected_one(cur_ind, count, ind_sub);
                joinVar(cur_ind, count, optimization_tag);
                count++;
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    scatter_evaluation_gather();
    //////////////////////////////////////////////////////////////////////////
    if((algo_mech_type == DECOMPOSITION && color_master_subPop) ||
       algo_mech_type == LOCALIZATION) {
        for(i = 0; i < nPop_cur; i++) {
            j = i;
            update_idealpoint(&osp_obj[j * nObj]);
        }
        update_xBest(UPDATE_GIVEN, nPop_cur, NULL, osp_var, osp_obj, rot_angle_offspring);
        update_xBest_history(UPDATE_GIVEN, nPop_cur, NULL, osp_var, osp_obj, rot_angle_offspring);
        int count = 0;
        for(i = 0; i < (*num_selected); i++) {
            for(j = 0; j < cloneNum[i]; j++) {
                int cur_ind = selectedIndv[i];
                update_population_DECOM(cur_ind, count, nPop_cur, osp_obj, osp_var,
                                        weights_all, useSflag, Sflag,
                                        rot_angle_offspring,
                                        niche, niche_local, tableNeighbor, tableNeighbor_local, maxNneighb, parent_type[count]);
                count++;
            }
        }
    }
    free(ind_sub);
    //
    return;
}
void rand_selection(int* inputIndex, int inputNum, int* outputIndex, int outputNum)
{
    if(inputNum < outputNum) {
        printf("%s(%d): The number input elements (%d) is less than the output (%d).\n",
               __FILE__, __LINE__, inputNum, outputNum);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_LESS_INPUT);
    }
    //
    int* tmp_ind = (int*)malloc(inputNum * sizeof(int));
    memcpy(tmp_ind, inputIndex, inputNum * sizeof(int));
    for(int i = 0; i < outputNum; i++) {
        int curInd = rnd(0, inputNum - i - 1);
        outputIndex[i] = tmp_ind[curInd];
        tmp_ind[curInd] = tmp_ind[inputNum - i - 1];
    }
    //
    free(tmp_ind);
    return;
}

void tour_selection(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(inputNum, sizeof(int));
    if(inputIndex)
        memcpy(remIndex, inputIndex, inputNum * sizeof(int));
    else
        for(int i = 0; i < inputNum; i++) remIndex[i] = i;
    int remNum = inputNum;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_utility_p.utility[remIndex[candInd]] > st_utility_p.utility[remIndex[bestInd]]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void tour_selection_repetitive(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(inputNum, sizeof(int));
    if(inputIndex)
        memcpy(remIndex, inputIndex, inputNum * sizeof(int));
    else
        for(int i = 0; i < inputNum; i++) remIndex[i] = i;
    int remNum = inputNum;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_utility_p.utility[remIndex[candInd]] > st_utility_p.utility[remIndex[bestInd]]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        //remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void tour_selection_ND(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(inputNum, sizeof(int));
    if(inputIndex)
        memcpy(remIndex, inputIndex, inputNum * sizeof(int));
    else
        for(int i = 0; i < inputNum; i++) remIndex[i] = i;
    int remNum = inputNum;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_archive_p.rank[remIndex[candInd]] < st_archive_p.rank[remIndex[bestInd]] ||
               (st_archive_p.rank[remIndex[candInd]] == st_archive_p.rank[remIndex[bestInd]] &&
                st_archive_p.dens[remIndex[candInd]] > st_archive_p.dens[remIndex[bestInd]])) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        //remIndex[bestInd] = remIndex[--remNum];
        //printf("%d ", remIndex[bestInd]);
    }
    free(remIndex);
    return;
}

void tour_selection_aggFit_greater(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(inputNum, sizeof(int));
    if(inputIndex)
        memcpy(remIndex, inputIndex, inputNum * sizeof(int));
    else
        for(int i = 0; i < inputNum; i++) remIndex[i] = i;
    int remNum = inputNum;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_decomp_p.fitCur[remIndex[candInd]] > st_decomp_p.fitCur[remIndex[bestInd]]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        //remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void tour_selection_aggFit_less(int* inputIndex, int inputNum, int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(inputNum, sizeof(int));
    if(inputIndex)
        memcpy(remIndex, inputIndex, inputNum * sizeof(int));
    else
        for(int i = 0; i < inputNum; i++) remIndex[i] = i;
    int remNum = inputNum;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_decomp_p.fitCur[remIndex[candInd]] < st_decomp_p.fitCur[remIndex[bestInd]]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        //remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void tour_selection_sub(int* outputIndex, int outputNum, int depth)
{
    int* remIndex = (int*)calloc(st_global_p.nPop, sizeof(int));
    for(int i = 0; i < st_global_p.nPop; i++)
        remIndex[i] = i;
    int remNum = st_global_p.nPop;
    int curNum = 0;
    int bestInd;
    int candInd;
    int probInd = st_MPI_p.color_pop - 1;
    while(curNum < outputNum) {
        bestInd = rnd(0, remNum - 1);
        for(int i = 1; i < depth; i++) {
            candInd = rnd(0, remNum - 1);
            if(st_pop_evo_cur.obj[remIndex[candInd] * st_global_p.nObj + probInd] <
               st_pop_evo_cur.obj[remIndex[bestInd] * st_global_p.nObj + probInd]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void greedy_selection(int* outputIndex, int outputNum)
{
    int* remIndex = (int*)calloc(st_global_p.nPop, sizeof(int));
    for(int i = 0; i < st_global_p.nPop; i++)
        remIndex[i] = i;
    int remNum = st_global_p.nPop;
    int curNum = 0;
    int bestInd;
    int candInd;
    while(curNum < outputNum) {
        bestInd = 0;
        for(int i = 1; i < remNum; i++) {
            candInd = i;
            if(st_utility_p.utility_cur[remIndex[candInd]] > st_utility_p.utility_cur[remIndex[bestInd]]) {
                bestInd = candInd;
            }
        }
        outputIndex[curNum++] = remIndex[bestInd];
        remIndex[bestInd] = remIndex[--remNum];
    }
    free(remIndex);
    return;
}

void greedy_selection_DECOM(int* outputIndex, int& outputNum)
{
    int cur_rank = 1;
    int* remIndex = (int*)calloc(st_global_p.nPop, sizeof(int));
    int remNum = 0;
    int curNum = 0;
    while(curNum < outputNum) {
        remNum = 0;
        for(int i = 0; i < st_archive_p.nArch; i++) {
            if(st_archive_p.rank[i] == cur_rank) {
                remIndex[remNum++] = i;
            }
        }
        if(curNum + remNum <= outputNum) {
            for(int i = 0; i < remNum; i++)
                outputIndex[curNum++] = remIndex[i];
        } else {
            qSortGeneral(st_archive_p.dens, remIndex, 0, remNum - 1);
            /*if(strct_MPI_info.mpi_rank==0)
            for(int iii=0;iii<remNum;iii++)
            printf("%lf ",archDens[remIndex[iii]]);
            if(strct_MPI_info.mpi_rank==0)
            printf("\n");*/
            for(int i = 0; i < outputNum - curNum; i++) {
                outputIndex[curNum + i] = remIndex[i];
            }
            curNum = outputNum;
        }
        cur_rank++;
    }
    outputNum = curNum;
    free(remIndex);
    return;
}

void clone_DECOM()
{
    //printf("%d ",strct_MPI_info.mpi_rank);
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.0\n");
    int i;
    double max_dist = -INF_DOUBLE;
    double min_dist = INF_DOUBLE;
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_archive_p.dens[st_global_p.selectedIndv[i]] < INF_DOUBLE &&
           st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                   st_archive_p.rank[st_global_p.selectedIndv[i]]) > max_dist) {
            max_dist = st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                       st_archive_p.rank[st_global_p.selectedIndv[i]]);
        }
        if(  //strct_archive_info.rank_archive[strct_global_paras.selectedIndv[i]] == 1 &&
            st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                    st_archive_p.rank[st_global_p.selectedIndv[i]]) < min_dist &&
            st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                    st_archive_p.rank[st_global_p.selectedIndv[i]]) > 0.0) {
            min_dist = st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                       st_archive_p.rank[st_global_p.selectedIndv[i]]);
        }
    }
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_archive_p.dens[st_global_p.selectedIndv[i]] >= INF_DOUBLE) {
            st_global_p.selectedProb[i] = 2 * max_dist;
        } else {
            st_global_p.selectedProb[i] = st_archive_p.dens[st_global_p.selectedIndv[i]] / pow(2,
                                          st_archive_p.rank[st_global_p.selectedIndv[i]]);
        }
        //if (strct_archive_info.rank_archive[strct_global_paras.selectedIndv[i]] > 1)
        //	strct_global_paras.selectedProb[i] *= min_dist;
    }
    //min_dist = INF_DOUBLE;
    //for (i = 0; i < strct_global_paras.num_selected; i++) {
    //	if (strct_global_paras.selectedProb[i] > 0.0 && strct_global_paras.selectedProb[i] < min_dist)
    //		min_dist = strct_global_paras.selectedProb[i];
    //}
    double sumD = 0.0;
    for(i = 0; i < st_global_p.num_selected; i++) {
        //if (strct_global_paras.selectedProb[i] < min_dist)
        //	strct_global_paras.selectedProb[i] = min_dist;
        sumD += st_global_p.selectedProb[i];
    }
    int sum = 0;
    for(i = 0; i < st_global_p.num_selected; i++) {
        st_global_p.selectedProb[i] = st_global_p.selectedProb[i] / sumD;
        st_global_p.cloneNum[i] = (int)(st_global_p.selectedProb[i] * st_global_p.nPop);
        if(st_global_p.cloneNum[i] == 0) st_global_p.cloneNum[i] = 1;
        sum += st_global_p.cloneNum[i];
    }//printf("%d ",sum);
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.1\n");
    i = 0;
    while(sum < st_global_p.nPop) {
        st_global_p.cloneNum[i]++;
        i++;
        if(i >= st_global_p.num_selected) {
            i = 0;
        }
        sum++;
    }
    while(sum > st_global_p.nPop) {
        int maxN = st_global_p.cloneNum[0];
        int maxI = 0;
        for(i = 1; i < st_global_p.num_selected; i++) {
            if(st_global_p.cloneNum[i] > maxN) {
                maxN = st_global_p.cloneNum[i];
                maxI = i;
            }
        }
        st_global_p.cloneNum[maxI]--;
        sum--;
    }
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.2\n");
    /*if(1)
    {
    for(i=0;i<strct_global_paras.num_selected;i++)
    printf("%e ",strct_global_paras.selectedProb[i]);
    }*/
    return;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// Pareto Dominance
//
void cooperativeCoevolution_ND()
{
    //if(strct_MPI_info.color_master_species && strct_global_paras.generation % CHECK_GAP_CC == CHECK_GAP_CC - 1) {
    //    total_utility();
    //    if(strct_utility_info.utility_mean < strct_utility_info.utility_threshold) {
    //        if(strct_ctrl_para.optimization_tag == 1)
    //            strct_ctrl_para.optimization_tag = 2;
    //        else
    //            strct_ctrl_para.optimization_tag = 1;
    //        printf("optimization_tag_CHANGED_%s_%s_RANK%d_GEN%d_GEN_MAX%d\n", strct_global_paras.algorithmName, strct_global_paras.testInstance, strct_MPI_info.mpi_rank, strct_global_paras.generation,
    //               strct_global_paras.generatMax);
    //    }
    //}
    //MPI_Barrier(MPI_COMM_WORLD);if(strct_MPI_info.mpi_rank==0)printf("line111.\n");
    //MPI_Barrier(MPI_COMM_WORLD); if (strct_MPI_info.mpi_rank == 0)printf("line113.\n");
    st_ctrl_p.optimization_tag =
        OPTIMIZE_CONVER_VARS;//
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < st_archive_p.nArch; i++) {
        st_DE_p.Sflag[i] = 0;
        st_decomp_p.fitImprove[i] = 0.0;
        st_decomp_p.countFitImprove[i] = 0;
        st_repo_p.flag[i] = -1;
    }
    //////////////////////////////////////////////////////////////////////////
    generate_para_all();
    //////////////////////////////////////////////////////////////////////////
    if(st_MPI_p.color_obj) {
        mainLoop_subObj_ND();//
    } else {
        mainLoop_allObjs_ND();//
    }
    //////////////////////////////////////////////////////////////////////////
    if(st_MPI_p.color_master_subPop) {
        int tmp_nPop;
        if(st_MPI_p.color_obj) {
            tmp_nPop = st_archive_p.nArch_sub;
        } else {
            tmp_nPop = st_archive_p.nArch;
        }
        for(int i = 0; i < tmp_nPop; i++) {
            if(st_archive_p.indx[i] >= tmp_nPop) {
                st_DE_p.Sflag[i] = 1;
            }
        }
        int tmp_ind;
        for(int i = 0; i < tmp_nPop; i++) {
            if(st_repo_p.flag[i] < 0 || st_repo_p.flag[i] >= tmp_nPop) {
                if(st_archive_p.curSize_inferior < tmp_nPop) {
                    tmp_ind = st_archive_p.curSize_inferior;
                    st_archive_p.curSize_inferior++;
                } else {
                    st_archive_p.curSize_inferior = tmp_nPop;
                    tmp_ind = rnd(0, tmp_nPop - 1);
                }
                memcpy(&st_pop_evo_cur.var_inferior[tmp_ind * st_global_p.nDim], &st_repo_p.var[i * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
            }
        }
        //////////////////////////////////////////////////////////////////////////
        int cur_indx_ns_nf = st_MPI_p.color_pop * st_global_p.nDim + st_MPI_p.color_subPop;
        for(int i = 0; i < tmp_nPop; i++) {
            if(st_DE_p.Sflag[i]) {
                st_MPI_p.ns_pops[cur_indx_ns_nf]++;
            } else {
                st_MPI_p.nf_pops[cur_indx_ns_nf]++;
            }
        }
    }
    //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 0) && 5 == strct_MPI_info.mpi_rank) {
    //    for(int i = 0; i < strct_repo_info.nRep; i++) {
    //        printf("(%d~%lf-%lf) ", i, strct_repo_info.obj[i * strct_global_paras.nObj + 0], strct_repo_info.obj[i * strct_global_paras.nObj + 1]);
    //    }
    //    printf("\n");
    //    for(int i = 0; i < strct_global_paras.nPop; i++) {
    //        printf("(%d-%d-%d~%d) ", i, strct_apap_DE_para.Sflag[i], strct_archive_info.indx_archive[i], strct_repo_info.flag[i]);
    //    }
    //    printf("\n");
    //}
    //

    //if(strct_ctrl_para.cur_run == 1 && strct_ctrl_para.cur_trace == 23 && strct_MPI_info.color_master_species) {
    //    char debugName[MAX_CHAR_ARR_SIZE];
    //    sprintf(debugName, "Debug_File_RUN_%d_TRACE_%d_MPI_%d",
    //            strct_ctrl_para.cur_run, strct_ctrl_para.cur_trace, strct_MPI_info.mpi_rank);
    //    strct_global_paras.debugFpt = fopen(debugName, "a");
    //    //fprintf(strct_global_paras.debugFpt, "strct_repo_info.var with Addr %d gen -> %d.2.1 rank -> %d\n", strct_repo_info.var, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_repo_info.var, 10, strct_global_paras.nDim, 1);
    //    fprintf(strct_global_paras.debugFpt, "strct_repo_info.obj with Addr %d gen -> %d.2.1 rank -> %d\n", strct_repo_info.obj, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    save_double(strct_global_paras.debugFpt, strct_repo_info.obj, strct_repo_info.nRep, strct_global_paras.nObj, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_archive_info.var_archive with Addr %d gen -> %d.2.1 rank -> %d\n", strct_archive_info.var_archive, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_archive_info.var_archive, 10, strct_global_paras.nDim, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_archive_info.obj_archive with Addr %d gen -> %d.2.1 rank -> %d\n", strct_archive_info.obj_archive, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_archive_info.obj_archive, strct_global_paras.nPop, strct_global_paras.nObj, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_repo_info.F with Addr %d gen -> %d.2.1 rank -> %d\n", strct_repo_info.F, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_repo_info.F, strct_repo_info.nRep, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_repo_info.CR with Addr %d gen -> %d.2.1 rank -> %d\n", strct_repo_info.CR, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_repo_info.CR, strct_repo_info.nRep, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.F__archive with Addr %d gen -> %d.2.1 rank -> %d\n", strct_apap_DE_para.F__archive, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_apap_DE_para.F__archive, strct_archive_info.nArch, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.CR_archive with Addr %d gen -> %d.2.1 rank -> %d\n", strct_apap_DE_para.CR_archive, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_apap_DE_para.CR_archive, strct_archive_info.nArch, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.F_cur with Addr %d gen -> %d.2.1 rank -> %d\n", strct_apap_DE_para.F_cur, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_apap_DE_para.F_cur, strct_global_paras.nPop, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.CR_cur with Addr %d gen -> %d.2.1 rank -> %d\n", strct_apap_DE_para.CR_cur, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, strct_apap_DE_para.CR_cur, strct_global_paras.nPop, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.F_mu_JADE with Addr %d gen -> %d.2.1 rank -> %d\n", &strct_apap_DE_para.F_mu_JADE, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, &strct_apap_DE_para.F_mu_JADE, 1, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.CR_mu with Addr %d gen -> %d.2.1 rank -> %d\n", &strct_apap_DE_para.CR_mu, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_double(strct_global_paras.debugFpt, &strct_apap_DE_para.CR_mu, 1, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_apap_DE_para.Sflag with Addr %d gen -> %d.2.1 rank -> %d\n", strct_apap_DE_para.Sflag, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_int(strct_global_paras.debugFpt, strct_apap_DE_para.Sflag, strct_global_paras.nPop, 1, 1);
    //    //fprintf(strct_global_paras.debugFpt, "strct_all_optimizer_paras.DE_F_types_all with Addr %d gen -> %d.2.1 rank -> %d\n", strct_all_optimizer_paras.DE_F_types_all, strct_global_paras.generation, strct_MPI_info.mpi_rank);
    //    //save_int(strct_global_paras.debugFpt, strct_all_optimizer_paras.DE_F_types_all, strct_global_paras.nPop, 1, 1);
    //    fclose(strct_global_paras.debugFpt);
    //}

    update_para_statistics();

    //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 0) && 5 == strct_MPI_info.mpi_rank)
    //    printf("+++++++++++++++++++%ld\n", get_rnd_uni_init());

    //MPI_Barrier(MPI_COMM_WORLD); if (strct_MPI_info.mpi_rank == 0)printf("line114.\n");

    return;
}

//
void mainLoop_subObj_ND()
{
    int i;

    if(st_MPI_p.color_master_subPop) {
        //////////////////////////////////////////////////////////////////////////
        gen_offspring_subObj_ND();
        //MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
        //if (strct_MPI_info.mpi_rank_master_species_populationScope == 0)printf("LINE ND sep 1.1\n");
        //////////////////////////////////////////////////////////////////////////
        int count = 0;
        for(i = 0; i < st_archive_p.nArch_sub; i++) {
            int iP = i;
            joinVar_subObj_ND(iP, count);
            count++;
        }
        //MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
        //if (strct_MPI_info.mpi_rank_master_species_populationScope == 0)printf("LINE ND sep 1.3\n");
    }
    //////////////////////////////////////////////////////////////////////////
    scatter_evaluation_gather();//
    //////////////////////////////////////////////////////////////////////////
    if(st_MPI_p.color_master_subPop) {
        memcpy(st_repo_p.var, st_archive_p.var, st_archive_p.nArch_sub * st_global_p.nDim * sizeof(double));
        memcpy(st_repo_p.obj, st_archive_p.obj, st_archive_p.nArch_sub * st_global_p.nObj * sizeof(double));
        memcpy(st_repo_p.F, st_DE_p.F__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.CR, st_DE_p.CR_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.CR_evo, st_DE_p.CR_evo_arc, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.w, st_PSO_p.w__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.c1, st_PSO_p.c1_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.c2, st_PSO_p.c2_archive, st_archive_p.nArch_sub * sizeof(double));
        st_repo_p.nRep = st_archive_p.nArch_sub;
        memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_pop_evo_offspring.var,
               st_archive_p.nArch_sub * st_global_p.nDim * sizeof(double));
        memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_pop_evo_offspring.obj,
               st_archive_p.nArch_sub * st_global_p.nObj * sizeof(double));
        memcpy(&st_repo_p.F[st_repo_p.nRep], st_DE_p.F__cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.CR[st_repo_p.nRep], st_DE_p.CR_cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.CR_evo[st_repo_p.nRep], st_DE_p.CR_evo_cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.w[st_repo_p.nRep], st_PSO_p.w__cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.c1[st_repo_p.nRep], st_PSO_p.c1_cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.c2[st_repo_p.nRep], st_PSO_p.c2_cur, st_archive_p.nArch_sub * sizeof(double));
        st_repo_p.nRep += st_archive_p.nArch_sub;
        refineRepository_generateArchive_sub();
        //
        memcpy(st_DE_p.F__cur, st_DE_p.F__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_DE_p.CR_cur, st_DE_p.CR_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_DE_p.CR_evo_cur, st_DE_p.CR_evo_arc, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.w__cur, st_PSO_p.w__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.c1_cur, st_PSO_p.c1_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.c2_cur, st_PSO_p.c2_archive, st_archive_p.nArch_sub * sizeof(double));

        //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 0) && 5 == strct_MPI_info.mpi_rank) {
        //    for(i = 0; i < strct_global_paras.nPop; i++)
        //        printf("(FF%d-%.16lf-%ld) ", i, strct_apap_DE_para.F_cur[i], get_rnd_uni_init());
        //}
    }

    return;
}

//
void mainLoop_allObjs_ND()
{
    int i, j;
    st_global_p.num_selected = st_global_p.nPop / 10;

    if(st_MPI_p.color_master_subPop) {
        greedy_selection_ND(st_global_p.selectedIndv, st_global_p.num_selected);
        //MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
        //if (strct_MPI_info.mpi_rank_master_species_populationScope == 0)printf("LINE ND 1.0\n");
        //printf("%d ",strct_global_paras.num_selected);
        clone_ND();
        //////////////////////////////////////////////////////////////////////////
        if(st_ctrl_p.type_clone_evo == CLONE_EVO_LOCAL) {
            gen_offspring_allObjs_local_ND();
        } else {
            gen_offspring_allObjs_global_ND();
        }
        //MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
        //if (strct_MPI_info.mpi_rank_master_species_populationScope == 0)printf("LINE ND 1.1\n");
        //////////////////////////////////////////////////////////////////////////
        int count;
        if(st_ctrl_p.type_clone_evo == CLONE_EVO_LOCAL) {
            count = 0;
            for(i = 0; i < st_global_p.num_selected; i++) {
                for(j = 0; j < st_global_p.cloneNum[i]; j++) {
                    int iP = i;
                    joinVar_allObjs_local_ND(iP, count);
                    count++;
                }
            }
        } else {
            for(i = 0; i < st_archive_p.nArch; i++) {
                joinVar_allObjs_global_ND(i, i);
            }
        }
        //MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
        //if (strct_MPI_info.mpi_rank_master_species_populationScope == 0)printf("LINE ND 1.3\n");
    }
    //////////////////////////////////////////////////////////////////////////
    scatter_evaluation_gather();
    //////////////////////////////////////////////////////////////////////////
    if(st_MPI_p.color_master_subPop) {
        memcpy(st_repo_p.var, st_archive_p.var, st_archive_p.nArch * st_global_p.nDim * sizeof(double));
        memcpy(st_repo_p.obj, st_archive_p.obj, st_archive_p.nArch * st_global_p.nObj * sizeof(double));
        memcpy(st_repo_p.F, st_DE_p.F__archive, st_archive_p.nArch * sizeof(double));
        memcpy(st_repo_p.CR, st_DE_p.CR_archive, st_archive_p.nArch * sizeof(double));
        memcpy(st_repo_p.CR_evo, st_DE_p.CR_evo_arc, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.w, st_PSO_p.w__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.c1, st_PSO_p.c1_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_repo_p.c2, st_PSO_p.c2_archive, st_archive_p.nArch_sub * sizeof(double));
        st_repo_p.nRep = st_archive_p.nArch;
        memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_pop_evo_offspring.var,
               st_archive_p.nArch * st_global_p.nDim * sizeof(double));
        memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_pop_evo_offspring.obj,
               st_archive_p.nArch * st_global_p.nObj * sizeof(double));
        memcpy(&st_repo_p.F[st_repo_p.nRep], st_DE_p.F__cur, st_archive_p.nArch * sizeof(double));
        memcpy(&st_repo_p.CR[st_repo_p.nRep], st_DE_p.CR_cur, st_archive_p.nArch * sizeof(double));
        memcpy(&st_repo_p.CR_evo[st_repo_p.nRep], st_DE_p.CR_evo_cur, st_archive_p.nArch * sizeof(double));
        memcpy(&st_repo_p.w[st_repo_p.nRep], st_PSO_p.w__cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.c1[st_repo_p.nRep], st_PSO_p.c1_cur, st_archive_p.nArch_sub * sizeof(double));
        memcpy(&st_repo_p.c2[st_repo_p.nRep], st_PSO_p.c2_cur, st_archive_p.nArch_sub * sizeof(double));
        st_repo_p.nRep += st_archive_p.nArch;
        refineRepository_generateArchive();
        //
        memcpy(st_DE_p.F__cur, st_DE_p.F__archive, st_archive_p.nArch * sizeof(double));
        memcpy(st_DE_p.CR_cur, st_DE_p.CR_archive, st_archive_p.nArch * sizeof(double));
        memcpy(st_DE_p.CR_evo_cur, st_DE_p.CR_evo_arc, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.w__cur, st_PSO_p.w__archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.c1_cur, st_PSO_p.c1_archive, st_archive_p.nArch_sub * sizeof(double));
        memcpy(st_PSO_p.c2_cur, st_PSO_p.c2_archive, st_archive_p.nArch_sub * sizeof(double));
    }

    return;
}

void greedy_selection_ND(int* outputIndex, int& outputNum)
{
    int cur_rank = 1;
    int* remIndex = (int*)calloc(st_global_p.nPop, sizeof(int));
    int remNum = 0;
    int curNum = 0;
    //	while(curNum<outputNum)
    {
        remNum = 0;
        for(int i = 0; i < st_archive_p.nArch; i++) {
            if(st_archive_p.rank[i] == cur_rank) {
                remIndex[remNum++] = i;
            }
        }
        if(curNum + remNum <= outputNum) {
            for(int i = 0; i < remNum; i++) {
                outputIndex[curNum++] = remIndex[i];
            }
        } else {
            qSortGeneral(st_archive_p.dens, remIndex, 0, remNum - 1);
            /*if(strct_MPI_info.mpi_rank==0)
            for(int iii=0;iii<remNum;iii++)
            printf("%lf ",archDens[remIndex[iii]]);
            if(strct_MPI_info.mpi_rank==0)
            printf("\n");*/
            for(int i = 0; i < outputNum - curNum; i++) {
                outputIndex[curNum + i] = remIndex[i];
            }
            curNum = outputNum;
        }
        cur_rank++;
    }
    outputNum = curNum;
    free(remIndex);
    return;
}

void clone_ND()
{
    //printf("%d ",strct_MPI_info.mpi_rank);
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.0\n");
    int i;
    double max_dist = -INF_DOUBLE;
    double min_dist = INF_DOUBLE;
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_archive_p.dens[st_global_p.selectedIndv[i]] < INF_DOUBLE &&
           st_archive_p.dens[st_global_p.selectedIndv[i]] > max_dist) {
            max_dist = st_archive_p.dens[st_global_p.selectedIndv[i]];
        }
        if(st_archive_p.rank[st_global_p.selectedIndv[i]] == 1 &&
           st_archive_p.dens[st_global_p.selectedIndv[i]] < max_dist &&
           st_archive_p.dens[st_global_p.selectedIndv[i]] > 0.0) {
            min_dist = st_archive_p.dens[st_global_p.selectedIndv[i]];
        }
    }
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_archive_p.dens[st_global_p.selectedIndv[i]] >= INF_DOUBLE) {
            st_global_p.selectedProb[i] = 2 * max_dist;
        } else {
            st_global_p.selectedProb[i] = st_archive_p.dens[st_global_p.selectedIndv[i]];
        }
        if(st_archive_p.rank[st_global_p.selectedIndv[i]] > 1) {
            st_global_p.selectedProb[i] *= min_dist;
        }
    }
    min_dist = INF_DOUBLE;
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_global_p.selectedProb[i] > 0.0 && st_global_p.selectedProb[i] < min_dist) {
            min_dist = st_global_p.selectedProb[i];
        }
    }
    double sumD = 0.0;
    for(i = 0; i < st_global_p.num_selected; i++) {
        if(st_global_p.selectedProb[i] < min_dist) {
            st_global_p.selectedProb[i] = min_dist;
        }
        sumD += st_global_p.selectedProb[i];
    }
    int sum = 0;
    for(i = 0; i < st_global_p.num_selected; i++) {
        st_global_p.selectedProb[i] = st_global_p.selectedProb[i] / sumD;
        st_global_p.cloneNum[i] = (int)(st_global_p.selectedProb[i] * st_global_p.nPop);
        sum += st_global_p.cloneNum[i];
    }//printf("%d ",sum);
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.1\n");
    i = 0;
    while(sum < st_global_p.nPop) {
        st_global_p.cloneNum[i]++;
        i++;
        if(i >= st_global_p.num_selected) {
            i = 0;
        }
        sum++;
    }
    while(sum > st_global_p.nPop) {
        int maxN = st_global_p.cloneNum[0];
        int maxI = 0;
        for(i = 1; i < st_global_p.num_selected; i++) {
            if(st_global_p.cloneNum[i] > maxN) {
                maxN = st_global_p.cloneNum[i];
                maxI = i;
            }
        }
        st_global_p.cloneNum[maxI]--;
        sum--;
    }
    //	MPI_Barrier(strct_MPI_info.comm_master_species_populationScope);
    //	if(strct_MPI_info.mpi_rank_master_species_populationScope==0)printf("LINE ND 2.2\n");
    /*if(1)
    {
    for(i=0;i<strct_global_paras.num_selected;i++)
    printf("%e ",strct_global_paras.selectedProb[i]);
    }*/
    return;
}

// descending
void qSortGeneral(double* data, int arrayFx[], int left, int right)
{
    int index;
    int temp;
    int i, j;
    double pivot;
    if(left < right) {
        index = rnd(left, right);
        temp = arrayFx[right];
        arrayFx[right] = arrayFx[index];
        arrayFx[index] = temp;
        pivot = data[arrayFx[right]];
        i = left - 1;
        for(j = left; j < right; j++) {
            if(data[arrayFx[j]] >= pivot) {
                i += 1;
                temp = arrayFx[j];
                arrayFx[j] = arrayFx[i];
                arrayFx[i] = temp;
            }
        }
        index = i + 1;
        temp = arrayFx[index];
        arrayFx[index] = arrayFx[right];
        arrayFx[right] = temp;
        qSortGeneral(data, arrayFx, left, index - 1);
        qSortGeneral(data, arrayFx, index + 1, right);
    }
    return;
}

void qSortBase(int* data, int left, int right)
{
    int index;
    int temp;
    int i, j;
    double pivot;
    if(left < right) {
        index = rnd(left, right);
        temp = right;
        right = index;
        index = temp;
        pivot = data[right];
        i = left - 1;
        for(j = left; j < right; j++) {
            if(data[j] >= pivot) {
                i += 1;
                temp = j;
                j = i;
                i = temp;
            }
        }
        index = i + 1;
        temp = index;
        index = right;
        right = temp;
        qSortBase(data, left, index - 1);
        qSortBase(data, index + 1, right);
    }
    return;
}