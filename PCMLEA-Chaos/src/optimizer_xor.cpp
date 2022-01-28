#include "global.h"

void xor_remVars_switch(int iS, int iD)
{
#ifdef DEBUG_TAG_TMP
    show_DeBug_info();
    MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
    if(st_MPI_p.mpi_rank == 0) printf("xor_remVars_switch(%d, %d, ind_sub); \n", iS, iD);
#endif
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int type_xor_rem_vars = st_ctrl_p.type_xor_rem_vars;
    int* parent_type = st_decomp_p.parent_type;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop; //
    int niche = st_decomp_p.niche;
    int type_join_xor = st_ctrl_p.type_join_xor;
    int color_pop = st_MPI_p.color_pop;
    double* cur_obj = st_pop_evo_cur.obj;
    double* cur_var = st_pop_evo_cur.var;
    int commonality_xor_remvar_tag = st_ctrl_p.commonality_xor_remvar_tag;
    int* table_mine_flag = st_grp_info_p.table_mine_flag;
    int* types_var_all = st_ctrl_p.types_var_all;
    double* rate_Commonality = st_optimizer_p.rate_Commonality;
    int type_test = st_ctrl_p.type_test;
    int type_xor_CNN = st_ctrl_p.type_xor_CNN;
    int nPop_cur = nPop;
    int nPop_candid_all = nPop;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    if(algo_mech_type == LOCALIZATION) {
        nPop_cur = nPop_mine;
        nPop_candid_all = nPop_mine +
                          st_pop_comm_p.n_neighbor_left +
                          st_pop_comm_p.n_neighbor_right +
                          st_global_p.nPop_exchange;
    }
    //
    int i;
    int r1, r2;
    double* pCurrent, * pTrail, * p1, * p2;
    float tmp_CR = (float)st_DE_p.CR_cur[iD];
    //
    pCurrent = &st_pop_evo_cur.var[iS * nDim];
    pTrail = &st_pop_evo_offspring.var[iD * nDim];
    //
    int depth = 10;
    if(algo_mech_type == LOCALIZATION) {
        depth = 10 / (nPop / nPop_mine) > 5 ? 10 / (nPop / nPop_mine) : 5;
    }
    switch(type_xor_rem_vars) {
    case XOR_REMVARS_XOR_MIXED:
        //selectSamples_niche(&strct_decomp_paras.tableNeighbor[iS * strct_global_paras.nPop], strct_decomp_paras.niche, iS, -1, -1, &r1, NULL, NULL, NULL, NULL);
        if(PARENT_LOCAL == parent_type[iD]) {
            selectSamples_niche(&tableNeighbor[iS * maxNneighb], niche, iS, -1, -1, &r1, NULL, NULL, NULL, NULL);
        } else {
            selectSamples(nPop_candid_all, iS, -1, -1, &r1, NULL, NULL, NULL, NULL);
        }
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
        if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
        //
        do {
            if(type_join_xor == JOIN_XOR_UTILITY) {
                if(nPop_cur > 2)
                    tour_selection(NULL, nPop_cur, &r2, 1, depth);
                else
                    selectSamples(nPop_candid_all, iS, r1, -1, &r2, NULL, NULL, NULL, NULL);
            } else if(type_join_xor == JOIN_XOR_AGGFIT) {
                tour_selection_aggFit_less(NULL, nPop_cur, &r2, 1, depth);
            } else {
                selectSamples(nPop_candid_all, iS, r1, -1, &r2, NULL, NULL, NULL, NULL);
            }
        } while(r2 == iS || r2 == r1);
#ifdef DEBUG_TAG_TMP
        show_DeBug_info();
        MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
        if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
        break;
    case XOR_REMVARS_XOR_POP:
        selectSamples(nPop_candid_all, iS, -1, -1, &r1, &r2, NULL, NULL, NULL);
        break;
    case XOR_REMVARS_XOR_SAME_REGION:
        if(PARENT_LOCAL == parent_type[iD]) {
            selectSamples_niche(&tableNeighbor[iS * maxNneighb], niche, iS, -1, -1, &r1, &r2, NULL, NULL, NULL);
        } else {
            selectSamples(nPop_candid_all, iS, -1, -1, &r1, &r2, NULL, NULL, NULL);
        }
        break;
    default:
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such xor type for generating remaining variables\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_XOR_REMVARS_TYPE);
        break;
    }
    //
    p1 = &cur_var[r1 * nDim];
    p2 = &cur_var[r2 * nDim];
    //////////////////////////////////////////////////////////////////////////
    if(color_pop) {
        int type_select_obj_best = st_ctrl_p.type_select_obj_best;
        if(type_select_obj_best == SELECT_OBJ_BEST_TOP_N) {
            int depth = 10;
            tourSelectSamples_sub(nPop_candid_all, depth, color_pop - 1, cur_obj, iS, r1, -1, &r2, NULL, NULL);
            p2 = &cur_var[r2 * nDim];
        } else {
            double* t2_best = st_DE_p.t2_best;
            double* var_best_subObjs_all = st_pop_best_p.var_best_subObjs_all;
            int  cn_best_history = st_pop_best_p.cn_best_history;
            double* var_best_history = st_pop_best_p.var_best_history;
            if(flip_r((float)t2_best[iD]) || !cn_best_history) {
                p2 = &var_best_subObjs_all[(color_pop - 1) * nDim];
            } else {
                selectSamples(cn_best_history, -1, -1, -1, &r2, NULL, NULL, NULL, NULL);
                p2 = &var_best_history[r2 * nDim];
            }
        }
        //do {
        //    if(strct_ctrl_para.type_join_xor == JOIN_XOR_UTILITY) {
        //        tour_selection(&r2, 1, depth); //////////////////////////////////////////////////////////////////////////
        //    } else if(strct_ctrl_para.type_join_xor == JOIN_XOR_AGGFIT) {
        //        tour_selection_aggFit_less(&r2, 1, depth);
        //    } else {
        //        selectSamples(strct_global_paras.nPop, iS, r1, -1, &r2, NULL, NULL);
        //    }
        //} while(r2 == iS || r2 == r1);
    }
    //////////////////////////////////////////////////////////////////////////

    int n_common = 0;
    int n_feat_p0 = 0;
    int n_feat_p1 = 0;
    double rate;
    if(commonality_xor_remvar_tag == FLAG_ON) {
        for(i = 0; i < nDim; i++) {
            if(table_mine_flag[i]) continue;
            if(types_var_all[i] != VAR_BINARY) {
                continue;
            }
            if((int)pCurrent[i] && (int)p1[i]) n_common++;
            if((int)pCurrent[i])               n_feat_p0++;
            if((int)p1[i])                     n_feat_p1++;
        }
        if((n_feat_p0 + n_feat_p1) == 2 * n_common) {
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
            rate = -1;
        } else {
            rate = (double)(n_feat_p0 - n_common) / (n_feat_p0 + n_feat_p1 - 2 * n_common);
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
        }
        rate_Commonality[iD] = rate;
    }
    //////////////////////////////////////////////////////////////////////////
    if((type_test == MY_TYPE_LeNet || type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
       type_xor_CNN == XOR_CNN_LeNet && flip_r((float)0.9)) {
        LeNet_xor(pCurrent, p1, p2, pTrail, tmp_CR);
    } else {
        if(commonality_xor_remvar_tag == FLAG_OFF) {
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
            for(i = 0; i < nDim; i++) {
                if(table_mine_flag[i]) continue;
                if(flip_r(tmp_CR)) {
                    pTrail[i] = pCurrent[i];
                } else if(flip_r(0.5)) {
                    pTrail[i] = p1[i];
                } else {
                    pTrail[i] = p2[i];
                }
            }
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
        } else {
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
            for(i = 0; i < nDim; i++) {
                if(table_mine_flag[i]) continue;
                if(types_var_all[i] != VAR_BINARY) {
                    if(flip_r(tmp_CR)) {
                        pTrail[i] = pCurrent[i];
                    } else if(flip_r(0.5)) {
                        pTrail[i] = p1[i];
                    } else {
                        pTrail[i] = p2[i];
                    }
                } else {
                    if(((int)pCurrent[i] && (int)p1[i]) ||
                       (!(int)pCurrent[i] && !(int)p1[i])) {
                        pTrail[i] = pCurrent[i];
                    } else if(flip_r((float)rate)) {
                        if((int)pCurrent[i])
                            pTrail[i] = pCurrent[i];
                        else
                            pTrail[i] = p1[i];
                    } else {
                        if(!(int)pCurrent[i])
                            pTrail[i] = pCurrent[i];
                        else
                            pTrail[i] = p1[i];
                    }
                }
            }
#ifdef DEBUG_TAG_TMP
            show_DeBug_info();
            MPI_Barrier(st_MPI_p.comm_master_subPop_popScope);
            if(st_MPI_p.mpi_rank == 0) printf("%s: xor_remVars_switch(%d, %d, ind_sub); \n", AT, iS, iD);
#endif
        }
    }

    return;
}

void xor_remVars_inherit_block(int iS, int iD)
{
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int type_xor_rem_vars = st_ctrl_p.type_xor_rem_vars;
    int* parent_type = st_decomp_p.parent_type;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop; //
    int niche = st_decomp_p.niche;
    int color_pop = st_MPI_p.color_pop;
    int color_subPop = st_MPI_p.color_subPop;
    double* cur_var = st_pop_evo_cur.var;
    double* osp_var = st_pop_evo_offspring.var;
    int* table_mine_flag = st_grp_info_p.table_mine_flag;
    int* Groups = st_grp_info_p.Groups;
    int* Groups_sizes = st_grp_info_p.Groups_sizes;
    int* Groups_sub_sizes = st_grp_info_p.Groups_sub_sizes;
    int* Groups_sub_disps = st_grp_info_p.Groups_sub_disps;
    int nPop_cur = nPop;
    int nPop_candid_all = nPop;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    if(algo_mech_type == LOCALIZATION) {
        nPop_cur = nPop_mine;
        nPop_candid_all = nPop_mine +
                          st_pop_comm_p.n_neighbor_left +
                          st_pop_comm_p.n_neighbor_right +
                          st_global_p.nPop_exchange;
    }
    //
    double* pCurrent, * pTrail;

    switch(type_xor_rem_vars) {
    case XOR_REMVARS_COPY:
        pCurrent = &cur_var[iS * nDim];
        pTrail = &osp_var[iD * nDim];
        for(int i = 0; i < nDim; i++) {
            if(table_mine_flag[i]) continue;
            pTrail[i] = pCurrent[i];
        }
        break;
    case XOR_REMVARS_INHERIT:
        int jP;
        int iGroup;
        int nGroup;
        nGroup = Groups_sizes[color_pop];
        pTrail = &osp_var[iD * nDim];
        for(iGroup = 0; iGroup < nGroup; iGroup++) {
            if(!st_grp_info_p.group_mine_flag[iGroup]) {
                if(PARENT_LOCAL == parent_type[iD]) {
                    selectSamples_niche(&tableNeighbor[iS * maxNneighb], niche, -1, -1, -1, &jP, NULL, NULL, NULL, NULL);
                } else {
                    selectSamples(nPop_candid_all, -1, -1, -1, &jP, NULL, NULL, NULL, NULL);
                }
                pCurrent = &cur_var[jP * nDim];
                int realInd;
                int tmpInd;
                tmpInd = color_pop * nDim + iGroup;
                for(int k = 0; k < Groups_sub_sizes[tmpInd]; k++) {
                    realInd = Groups[color_pop * nDim + Groups_sub_disps[tmpInd] + k];
                    pTrail[realInd] = pCurrent[realInd];
                }
            }
        }
        break;
    default:
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such xor type for generating remaining variables\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_XOR_REMVARS_TYPE);
        break;
    }

    return;
}
