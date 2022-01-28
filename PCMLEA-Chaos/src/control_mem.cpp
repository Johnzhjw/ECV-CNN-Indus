# include "global.h"

void allocateMemory_grouping_ana()
{
    // grouping
    st_grp_ana_p.Dependent = allocInt(st_global_p.nObj * st_global_p.nDim * st_global_p.nDim);
    st_grp_ana_p.Control = allocInt(st_global_p.nDim * st_global_p.nObj);
    st_grp_ana_p.Control_Dist = allocDouble(st_global_p.nDim);
    st_grp_ana_p.Control_Mean = allocDouble(st_global_p.nDim * st_global_p.nObj);
    st_grp_ana_p.Control_Dist_Mean = allocDouble(st_global_p.nDim);
    st_grp_ana_p.Interdependence_Weight = allocDouble(st_global_p.nObj * st_global_p.nDim *
                                          st_global_p.nDim);
    st_grp_ana_p.weight_min = allocDouble(st_global_p.nObj);
    st_grp_ana_p.weight_max = allocDouble(st_global_p.nObj);
    st_grp_ana_p.Effect = allocInt(st_global_p.nObj * st_global_p.nDim);
    //
    st_grp_ana_p.var_current_grp = allocDouble(st_global_p.nPop * st_global_p.nDim);
    st_grp_ana_p.obj_current_grp = allocDouble(st_global_p.nPop * st_global_p.nDim);
    st_grp_ana_p.var_repository_grp = allocDouble(st_global_p.nPop * st_global_p.nDim);
    st_grp_ana_p.obj_repository_grp = allocDouble(st_global_p.nPop * st_global_p.nDim);
    //
    return;
}

void allocateMemory_grouping_info()
{
    //
    st_global_p.minLimit = allocDouble(st_global_p.nDim);
    st_global_p.maxLimit = allocDouble(st_global_p.nDim);
    //
    st_grp_info_p.DiversityIndexs = allocInt(st_global_p.nDim);
    st_grp_info_p.ConvergenceIndexs = allocInt(st_global_p.nDim);
    st_grp_info_p.Groups = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_sizes = allocInt(st_global_p.nObj + 1);
    st_grp_info_p.Groups_sub_sizes = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_sub_disps = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_raw = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_raw_flags = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_raw_sizes = allocInt(st_global_p.nObj + 1);
    st_grp_info_p.Groups_raw_sub_sizes = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.Groups_raw_sub_disps = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_grp_info_p.table_mine = allocInt(st_global_p.nDim);
    st_grp_info_p.table_remain = allocInt(st_global_p.nDim);
    st_grp_info_p.table_mine_flag = allocInt(st_global_p.nDim);
    st_grp_info_p.group_mine_flag = allocInt(st_global_p.nDim);
    //
    return;
}

void freeMemory_grouping_ana()
{
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("FREE_MEM_BEFORE...\n");
    //}
    free(st_grp_ana_p.Dependent);
    free(st_grp_ana_p.Control);
    free(st_grp_ana_p.Control_Dist);
    free(st_grp_ana_p.Control_Mean);
    free(st_grp_ana_p.Control_Dist_Mean);
    free(st_grp_ana_p.Interdependence_Weight);
    free(st_grp_ana_p.weight_min);
    free(st_grp_ana_p.weight_max);
    free(st_grp_ana_p.Effect);
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == st_MPI_p.mpi_rank) {
    //    printf("FREE_MEM_MID...\n");
    //}
    //
    free(st_grp_ana_p.var_current_grp);
    free(st_grp_ana_p.obj_current_grp);
    free(st_grp_ana_p.var_repository_grp);
    free(st_grp_ana_p.obj_repository_grp);
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("FREE_MEM_AFTER...\n");
    //}
    //
    return;
}

void freeMemory_grouping_info()
{
    //
    free(st_grp_info_p.DiversityIndexs);
    free(st_grp_info_p.ConvergenceIndexs);
    free(st_grp_info_p.Groups);
    free(st_grp_info_p.Groups_sizes);
    free(st_grp_info_p.Groups_sub_sizes);
    free(st_grp_info_p.Groups_sub_disps);
    free(st_grp_info_p.Groups_raw);
    free(st_grp_info_p.Groups_raw_flags);
    free(st_grp_info_p.Groups_raw_sizes);
    free(st_grp_info_p.Groups_raw_sub_sizes);
    free(st_grp_info_p.Groups_raw_sub_disps);
    free(st_grp_info_p.table_mine);
    free(st_grp_info_p.table_remain);
    free(st_grp_info_p.table_mine_flag);
    free(st_grp_info_p.group_mine_flag);
    //
    return;
}

void allocateMemory_MPI()
{
    int the_size = (st_MPI_p.mpi_size > (st_global_p.nObj + 1) ? st_MPI_p.mpi_size :
                    (st_global_p.nObj + 1));

    st_grp_info_p.vec_sizeGroups = allocInt(st_global_p.nObj + 1);
    st_MPI_p.nPop_all = allocInt(st_global_p.nObj + 1);

    st_MPI_p.recv_size = (int*)calloc(the_size, sizeof(int));
    st_MPI_p.disp_size = (int*)calloc(the_size, sizeof(int));
    st_MPI_p.each_size = (int*)calloc(the_size, sizeof(int));

    st_MPI_p.recv_size_subPop = (int*)calloc(the_size, sizeof(int));
    st_MPI_p.disp_size_subPop = (int*)calloc(the_size, sizeof(int));
    st_MPI_p.each_size_subPop = (int*)calloc(the_size, sizeof(int));

    st_MPI_p.globalRank_master_pop = (int*)calloc(st_global_p.nObj + 1, sizeof(int));

    st_MPI_p.vec_num_MPI_master = (int*)calloc(st_global_p.nObj + 1, sizeof(int));
    st_MPI_p.vec_num_MPI_slave = (int*)calloc(st_global_p.nObj + 1, sizeof(int));
    st_MPI_p.vec_importance = (double*)calloc(st_global_p.nObj + 1, sizeof(double));
    st_MPI_p.vec_MPI_ratio = (double*)calloc(st_global_p.nObj + 1, sizeof(double));
    //
    return;
}

void freeMemory_MPI()
{
    free(st_grp_info_p.vec_sizeGroups);
    free(st_MPI_p.nPop_all);

    free(st_MPI_p.recv_size);
    free(st_MPI_p.disp_size);
    free(st_MPI_p.each_size);

    free(st_MPI_p.recv_size_subPop);
    free(st_MPI_p.disp_size_subPop);
    free(st_MPI_p.each_size_subPop);

    free(st_MPI_p.globalRank_master_pop);

    free(st_MPI_p.vec_num_MPI_master);
    free(st_MPI_p.vec_num_MPI_slave);
    free(st_MPI_p.vec_importance);
    free(st_MPI_p.vec_MPI_ratio);

    MPI_Comm_free(&st_MPI_p.comm_obj);
    MPI_Comm_free(&st_MPI_p.comm_pop);
    MPI_Comm_free(&st_MPI_p.comm_subPop);
    MPI_Comm_free(&st_MPI_p.comm_master_subPop_globalScope);
    MPI_Comm_free(&st_MPI_p.comm_master_subPop_popScope);
    MPI_Comm_free(&st_MPI_p.comm_master_pop);
    //
    return;
}

void allocateMemory()
{
    st_global_p.nInd_1pop = st_global_p.nPop > st_archive_p.nArch ? st_global_p.nPop : st_archive_p.nArch;
    st_global_p.num_subpops = 1;
    st_global_p.nInd_max = st_global_p.nInd_1pop;
    if(st_ctrl_p.type_grp_loop == LOOP_GRP || st_ctrl_p.type_pop_loop == LOOP_POP) {
        st_global_p.maxNgrp = st_grp_info_p.vec_sizeGroups[0] * st_ctrl_p.flag_mainPop;
        for(int i = 1; i <= st_global_p.nObj; i++)
            if(st_global_p.maxNgrp < st_grp_info_p.vec_sizeGroups[i] * st_ctrl_p.flag_multiPop)
                st_global_p.maxNgrp = st_grp_info_p.vec_sizeGroups[i] * st_ctrl_p.flag_multiPop;
        st_global_p.num_subpops = st_global_p.maxNgrp;
        if(st_ctrl_p.type_pop_loop == LOOP_POP)
            st_global_p.num_subpops *= st_global_p.nObj + 1;
    }
    st_global_p.nInd_max = st_global_p.nInd_1pop * st_global_p.num_subpops;
    st_global_p.the_size_OBJ = st_global_p.nInd_max * st_global_p.nObj;
    st_global_p.the_size_VAR = st_global_p.nInd_max * st_global_p.nDim;
    st_global_p.the_size_IND = st_global_p.nInd_max;
    st_global_p.the_size_VAR_obj = st_global_p.nObj * st_global_p.nDim;
    st_global_p.the_size_OBJ_obj = st_global_p.nObj * st_global_p.nObj;
    st_global_p.the_size_VAR_1pop = st_global_p.nInd_1pop * st_global_p.nDim;
    st_global_p.the_size_OBJ_1pop = st_global_p.nInd_1pop * st_global_p.nObj;
    st_global_p.nInd_max_repo = 2 * st_global_p.nInd_1pop;
    //
    st_ctrl_p.types_var_all = allocInt(st_global_p.nDim);
    st_ctrl_p.tag_selection = allocInt(st_global_p.the_size_VAR);
    st_utility_p.utility = allocDouble(st_global_p.the_size_IND);
    st_utility_p.utility_cur = allocDouble(st_global_p.the_size_IND);
    st_global_p.selectedIndv = allocInt(st_global_p.nInd_max);
    st_global_p.selectedProb = allocDouble(st_global_p.nInd_max);
    st_global_p.cloneNum = allocInt(st_global_p.nInd_max);
    st_decomp_p.parent_type = allocInt(st_global_p.the_size_IND);
    st_pop_comm_p.slctIndx = allocInt(st_global_p.the_size_IND);
    st_pop_comm_p.updtIndx = allocInt(st_global_p.the_size_IND);
    st_pop_comm_p.updtIndx_recv_left = allocInt(st_global_p.the_size_IND);
    st_pop_comm_p.updtIndx_recv_right = allocInt(st_global_p.the_size_IND);
    st_pop_evo_offspring.var_feature = allocInt(st_global_p.the_size_IND * TH_N_FEATURE);
    st_archive_p.var_feature = allocInt(st_global_p.the_size_IND * TH_N_FEATURE);
    st_pop_evo_cur.var = allocDouble(2 * st_global_p.the_size_VAR);
    st_qu_p.minLimit = allocDouble(st_global_p.nDim);
    st_qu_p.maxLimit = allocDouble(st_global_p.nDim);
    st_pop_comm_p.var_send = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.var_recv = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.rot_angle_send = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.rot_angle_recv = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.var_left = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.var_right = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.var_exchange = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.rot_angle_left = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.rot_angle_right = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.rot_angle_exchange = allocDouble(st_global_p.the_size_VAR);
    st_pop_evo_cur.var_saved = allocDouble(st_global_p.the_size_VAR);
    st_pop_evo_cur.var_inferior = allocDouble(st_global_p.the_size_VAR);
    st_pop_evo_offspring.var = allocDouble(st_global_p.the_size_VAR);
    st_pop_evo_offspring.obj = allocDouble(st_global_p.the_size_OBJ);
    st_qu_p.var_offspring = allocDouble(2 * st_global_p.the_size_VAR);
    st_qu_p.rot_angle_offspring = allocDouble(st_global_p.the_size_VAR);
    st_qu_p.rot_angle_cur = allocDouble(st_global_p.the_size_VAR);
    st_qu_p.rot_angle_cur_inferior = allocDouble(st_global_p.the_size_VAR);
    st_qu_p.minLimit_rot_angle = allocDouble(st_global_p.nDim);
    st_qu_p.maxLimit_rot_angle = allocDouble(st_global_p.nDim);
    st_pop_evo_cur.obj = allocDouble(2 * st_global_p.the_size_OBJ);
    st_pop_comm_p.obj_send = allocDouble(st_global_p.the_size_OBJ);
    st_pop_comm_p.obj_recv = allocDouble(st_global_p.the_size_OBJ);
    st_pop_comm_p.obj_left = allocDouble(st_global_p.the_size_OBJ);
    st_pop_comm_p.obj_right = allocDouble(st_global_p.the_size_OBJ);
    st_pop_comm_p.obj_exchange = allocDouble(st_global_p.the_size_OBJ);
    st_pop_evo_cur.obj_saved = allocDouble(st_global_p.the_size_OBJ);
    st_decomp_p.fitCur = allocDouble(st_global_p.the_size_IND);
    st_decomp_p.fitImprove = allocDouble(st_global_p.the_size_IND);
    st_decomp_p.countFitImprove = allocInt(st_global_p.the_size_IND);
    st_pop_best_p.var_best = allocDouble(st_global_p.nDim);
    st_pop_best_p.obj_best = allocDouble(st_global_p.nObj);
    st_qu_p.rot_angle_best = allocDouble(st_global_p.nDim);
    st_pop_best_p.var_best_exchange = allocDouble(st_global_p.nDim);
    st_pop_best_p.obj_best_exchange = allocDouble(st_global_p.nObj);
    st_pop_best_p.var_best_history = allocDouble((st_pop_best_p.n_best_history + 1) * st_global_p.nDim);
    st_pop_best_p.obj_best_history = allocDouble((st_pop_best_p.n_best_history + 1) * st_global_p.nObj);
    st_qu_p.rot_angle_best_history = allocDouble((st_pop_best_p.n_best_history + 1) * st_global_p.nDim);
    st_pop_best_p.var_best_subObjs_all = allocDouble(st_global_p.the_size_VAR_obj);
    st_pop_best_p.obj_best_subObjs_all = allocDouble(st_global_p.the_size_OBJ_obj);
    st_qu_p.rot_angle_best_subObjs_all = allocDouble(st_global_p.the_size_VAR_obj);
    st_DE_p.F__cur = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.CR_cur = allocDouble(st_global_p.nInd_1pop);
    st_decomp_p.weights_unit = allocDouble(st_global_p.the_size_OBJ);
    st_decomp_p.weights_all = allocDouble(st_global_p.the_size_OBJ);
    st_decomp_p.weights_left = allocDouble(st_global_p.the_size_OBJ);
    st_decomp_p.weights_right = allocDouble(st_global_p.the_size_OBJ);
    st_decomp_p.weights_mine = allocDouble(st_global_p.the_size_OBJ);
    st_pop_comm_p.posFactor_left = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.posFactor_right = allocDouble(st_global_p.the_size_VAR);
    st_pop_comm_p.posFactor_mine = allocDouble(st_global_p.the_size_VAR);
    st_decomp_p.weight_prefer_tag = allocInt(st_global_p.nInd_max);
    st_decomp_p.tableNeighbor = allocInt(4 * st_global_p.the_size_IND * st_global_p.nInd_1pop);
    st_decomp_p.tableNeighbor_local = allocInt(4 * st_global_p.the_size_IND * st_global_p.nInd_1pop);
    st_PSO_p.velocity = allocDouble(st_global_p.the_size_VAR);
    st_PSO_p.vMax = allocDouble(st_global_p.nDim);
    st_PSO_p.vMin = allocDouble(st_global_p.nDim);
    st_PSO_p.w__cur = allocDouble(st_global_p.the_size_IND);
    st_PSO_p.c1_cur = allocDouble(st_global_p.the_size_IND);
    st_PSO_p.c2_cur = allocDouble(st_global_p.the_size_IND);
    st_PSO_p.w__archive = allocDouble(st_global_p.nInd_1pop);
    st_PSO_p.c1_archive = allocDouble(st_global_p.nInd_1pop);
    st_PSO_p.c2_archive = allocDouble(st_global_p.nInd_1pop);
    st_PSO_p.indNeighbor = allocInt(st_global_p.the_size_IND);
    st_archive_p.var = allocDouble(st_global_p.the_size_VAR);
    st_archive_p.var_exchange = allocDouble(st_global_p.the_size_VAR);
    st_archive_p.var_inferior = allocDouble(st_global_p.the_size_VAR);
    st_archive_p.obj = allocDouble(st_global_p.the_size_OBJ);
    st_archive_p.obj_exchange = allocDouble(st_global_p.the_size_OBJ);
    st_archive_p.dens = allocDouble(st_global_p.the_size_IND);
    st_archive_p.dens_exchange = allocDouble(st_global_p.the_size_IND);
    st_archive_p.var_Ex = allocDouble(st_global_p.the_size_VAR);
    st_archive_p.obj_Ex = allocDouble(st_global_p.the_size_OBJ);
    st_archive_p.dens_Ex = allocDouble(st_global_p.the_size_IND);
    st_archive_p.indx = allocInt(st_global_p.the_size_IND);
    st_archive_p.rank = allocInt(st_global_p.the_size_IND);
    st_archive_p.rank_Ex = allocInt(st_global_p.the_size_IND);
    st_grp_info_p.diver_var_store_all = allocDouble(st_global_p.the_size_VAR_1pop);
    st_grp_info_p.diver_var_store_mine = allocDouble(st_global_p.the_size_VAR);
    st_repo_p.var = allocDouble(st_global_p.nInd_max_repo * st_global_p.nDim);
    st_repo_p.obj = allocDouble(st_global_p.nInd_max_repo * st_global_p.nObj);
    st_repo_p.dens = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.F = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.CR = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.CR_evo = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.w = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.c1 = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.c2 = allocDouble(st_global_p.nInd_max_repo);
    st_repo_p.tag = allocInt(st_global_p.nInd_max_repo);
    st_repo_p.flag = allocInt(st_global_p.nInd_max_repo);
    st_decomp_p.fun_max = allocDouble(st_global_p.nObj);
    st_decomp_p.fun_min = allocDouble(st_global_p.nObj);
    st_decomp_p.idealpoint = allocDouble(st_global_p.nObj);
    st_decomp_p.nadirpoint = allocDouble(st_global_p.nObj);
    //
    st_MPI_p.ns_pops = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    st_MPI_p.nf_pops = allocInt((st_global_p.nObj + 1) * st_global_p.nDim);
    //
    st_DE_p.tag_SaNSDE_F = allocInt(st_global_p.nInd_1pop);
    st_DE_p.F_hist = allocDouble(st_global_p.num_subpops * st_DE_p.nHistSHADE);
    st_DE_p.CR_hist = allocDouble(st_global_p.num_subpops * st_DE_p.nHistSHADE);
    st_DE_p.CR_evo_cur = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.t1_best = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.t2_best = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.Sflag = allocInt(st_global_p.nInd_1pop);
    st_DE_p.F__archive = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.CR_archive = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.CR_evo_arc = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.candid_F = allocDouble(st_DE_p.candid_num);
    st_DE_p.candid_CR = allocDouble(st_DE_p.candid_num);
    st_DE_p.prob_F = allocDouble(st_DE_p.candid_num);
    st_DE_p.prob_CR = allocDouble(st_DE_p.candid_num);
    st_DE_p.disc_F = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.disc_CR = allocDouble(st_global_p.nInd_1pop);
    st_DE_p.indx_disc_F = allocInt(st_global_p.nInd_1pop);
    st_DE_p.indx_disc_CR = allocInt(st_global_p.nInd_1pop);

    st_optimizer_p.rate_Commonality = allocDouble(st_global_p.nInd_1pop);
    st_optimizer_p.optimizer_prob = allocDouble(st_optimizer_p.num_optimizer);
    st_optimizer_p.optimizer_candid = allocInt(st_optimizer_p.num_optimizer);
    st_optimizer_p.optimizer_types_all = allocInt(st_global_p.nInd_1pop);
    st_optimizer_p.DE_F_types_all = allocInt(st_global_p.nInd_1pop);
    st_optimizer_p.DE_CR_types_all = allocInt(st_global_p.nInd_1pop);
    st_optimizer_p.PSO_para_types_all = allocInt(st_global_p.nInd_1pop);

    return;
}

double* allocDouble(int size)
{
    double* tmp;
    if((tmp = (double*)calloc(size, sizeof(double))) == NULL) {
        printf("%s:ERROR!! --> calloc: no memory for vector (%d, %d)\n", AT, size, (int)sizeof(double));
        exit(MY_ERROR_NO_MEMORY);
    }
    return tmp;
}

int* allocInt(int size)
{
    int* tmp;
    if((tmp = (int*)calloc(size, sizeof(int))) == NULL) {
        printf("%s:ERROR!! --> calloc: no memory for vector (%d, %d)\n", AT, size, (int)sizeof(double));
        exit(MY_ERROR_NO_MEMORY);
    }
    return tmp;
}

void freeMemory()
{
    //
    free(st_global_p.minLimit);
    free(st_global_p.maxLimit);
    //
    freeMemory_grouping_info();
    //
    freeMemory_MPI();
    //////////////////////////////////////////////////////////////////////////
    free(st_ctrl_p.types_var_all);
    free(st_ctrl_p.tag_selection);
    free(st_utility_p.utility);
    free(st_utility_p.utility_cur);
    free(st_global_p.selectedIndv);
    free(st_global_p.selectedProb);
    free(st_global_p.cloneNum);
    free(st_decomp_p.parent_type);
    free(st_pop_comm_p.slctIndx);
    free(st_pop_comm_p.updtIndx);
    free(st_pop_comm_p.updtIndx_recv_left);
    free(st_pop_comm_p.updtIndx_recv_right);
    free(st_pop_evo_offspring.var_feature);
    free(st_archive_p.var_feature);
    free(st_pop_evo_cur.var);
    free(st_qu_p.minLimit);
    free(st_qu_p.maxLimit);
    free(st_pop_comm_p.var_send);
    free(st_pop_comm_p.var_recv);
    free(st_pop_comm_p.rot_angle_send);
    free(st_pop_comm_p.rot_angle_recv);
    free(st_pop_comm_p.var_left);
    free(st_pop_comm_p.var_right);
    free(st_pop_comm_p.var_exchange);
    free(st_pop_comm_p.rot_angle_left);
    free(st_pop_comm_p.rot_angle_right);
    free(st_pop_comm_p.rot_angle_exchange);
    free(st_pop_evo_cur.var_saved);
    free(st_pop_evo_cur.var_inferior);
    free(st_pop_evo_offspring.var);
    free(st_pop_evo_offspring.obj);
    free(st_qu_p.var_offspring);
    free(st_qu_p.rot_angle_offspring);
    free(st_qu_p.rot_angle_cur);
    free(st_qu_p.rot_angle_cur_inferior);
    free(st_qu_p.minLimit_rot_angle);
    free(st_qu_p.maxLimit_rot_angle);
    free(st_pop_evo_cur.obj);
    free(st_pop_comm_p.obj_send);
    free(st_pop_comm_p.obj_recv);
    free(st_pop_comm_p.obj_left);
    free(st_pop_comm_p.obj_right);
    free(st_pop_comm_p.obj_exchange);
    free(st_pop_evo_cur.obj_saved);
    free(st_decomp_p.fitCur);
    free(st_decomp_p.fitImprove);
    free(st_decomp_p.countFitImprove);
    free(st_pop_best_p.var_best);
    free(st_pop_best_p.obj_best);
    free(st_qu_p.rot_angle_best);
    free(st_pop_best_p.var_best_exchange);
    free(st_pop_best_p.obj_best_exchange);
    free(st_pop_best_p.var_best_history);
    free(st_pop_best_p.obj_best_history);
    free(st_qu_p.rot_angle_best_history);
    free(st_pop_best_p.var_best_subObjs_all);
    free(st_pop_best_p.obj_best_subObjs_all);
    free(st_qu_p.rot_angle_best_subObjs_all);
    free(st_DE_p.F__cur);
    free(st_DE_p.CR_cur);
    free(st_decomp_p.weights_unit);
    free(st_decomp_p.weights_all);
    free(st_decomp_p.weights_left);
    free(st_decomp_p.weights_right);
    free(st_decomp_p.weights_mine);
    free(st_pop_comm_p.posFactor_left);
    free(st_pop_comm_p.posFactor_right);
    free(st_pop_comm_p.posFactor_mine);
    free(st_decomp_p.weight_prefer_tag);
    free(st_decomp_p.tableNeighbor);
    free(st_decomp_p.tableNeighbor_local);
    free(st_PSO_p.velocity);
    free(st_PSO_p.vMax);
    free(st_PSO_p.vMin);
    free(st_PSO_p.w__cur);
    free(st_PSO_p.c1_cur);
    free(st_PSO_p.c2_cur);
    free(st_PSO_p.w__archive);
    free(st_PSO_p.c1_archive);
    free(st_PSO_p.c2_archive);
    free(st_PSO_p.indNeighbor);
    free(st_archive_p.var);
    free(st_archive_p.var_exchange);
    free(st_archive_p.var_inferior);
    free(st_archive_p.obj);
    free(st_archive_p.obj_exchange);
    free(st_archive_p.dens);
    free(st_archive_p.dens_exchange);
    free(st_archive_p.var_Ex);
    free(st_archive_p.obj_Ex);
    free(st_archive_p.dens_Ex);
    free(st_archive_p.indx);
    free(st_archive_p.rank);
    free(st_archive_p.rank_Ex);
    free(st_grp_info_p.diver_var_store_all);
    free(st_grp_info_p.diver_var_store_mine);
    free(st_repo_p.var);
    free(st_repo_p.obj);
    free(st_repo_p.dens);
    free(st_repo_p.F);
    free(st_repo_p.CR);
    free(st_repo_p.CR_evo);
    free(st_repo_p.w);
    free(st_repo_p.c1);
    free(st_repo_p.c2);
    free(st_repo_p.tag);
    free(st_repo_p.flag);
    free(st_decomp_p.fun_max);
    free(st_decomp_p.fun_min);
    free(st_decomp_p.idealpoint);
    free(st_decomp_p.nadirpoint);
    //////////////////////////////////////////////////////////////////////////
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == st_MPI_p.mpi_rank) {
    //    printf("FREE_MEM_BEFORE...\n");
    //}
    free(st_MPI_p.ns_pops);
    free(st_MPI_p.nf_pops);
    //////////////////////////////////////////////////////////////////////////
    free(st_DE_p.tag_SaNSDE_F);
    free(st_DE_p.F_hist);
    free(st_DE_p.CR_hist);
    free(st_DE_p.CR_evo_cur);
    free(st_DE_p.Sflag);
    free(st_DE_p.F__archive);
    free(st_DE_p.CR_archive);
    free(st_DE_p.CR_evo_arc);
    free(st_DE_p.t1_best);
    free(st_DE_p.t2_best);
    free(st_DE_p.candid_F);
    free(st_DE_p.candid_CR);
    free(st_DE_p.prob_F);
    free(st_DE_p.prob_CR);
    free(st_DE_p.disc_F);
    free(st_DE_p.disc_CR);
    free(st_DE_p.indx_disc_F);
    free(st_DE_p.indx_disc_CR);
    free(st_optimizer_p.rate_Commonality);
    free(st_optimizer_p.optimizer_prob);
    free(st_optimizer_p.optimizer_types_all);
    free(st_optimizer_p.DE_F_types_all);
    free(st_optimizer_p.DE_CR_types_all);
    free(st_optimizer_p.PSO_para_types_all);
    //
    return;
}