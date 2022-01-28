#include "global.h"
#include <math.h>

void gen_offspring_selected_one(int iS, int iD, int* tmp_indx)
{
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;
    int opt_binVar_as_realVar_tag = st_ctrl_p.opt_binVar_as_realVar_tag;
    int mixed_var_types_tag = st_ctrl_p.mixed_var_types_tag;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    int type_var_encoding = st_ctrl_p.type_var_encoding;
    int color_pop = st_MPI_p.color_pop;
    int* optimizer_types_all = st_optimizer_p.optimizer_types_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    double p_best_ratio = st_optimizer_p.p_best_ratio;
    int* parent_type = st_decomp_p.parent_type;
    int niche = st_decomp_p.niche;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop;
    int curSize_inferior = st_pop_evo_cur.curSize_inferior;
    double* velocity = st_PSO_p.velocity;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int nPop_cur = nPop;
    int nPop_candid_all = nPop;
    int nPop_mine = st_global_p.nPop_mine;
    if(algo_mech_type == LOCALIZATION) {
        nPop_cur = nPop_mine;
        nPop_candid_all = nPop_mine +
                          st_pop_comm_p.n_neighbor_left +
                          st_pop_comm_p.n_neighbor_right +
                          st_global_p.nPop_exchange;
    }
    int type_select_obj_best = st_ctrl_p.type_select_obj_best;
    double* t1_best = st_DE_p.t1_best;
    double* var_best_subObjs_all = st_pop_best_p.var_best_subObjs_all;
    double* rot_angle_best_subObjs_all = st_qu_p.rot_angle_best_subObjs_all;
    int  cn_best_history = st_pop_best_p.cn_best_history;
    int  n_best_history = st_pop_best_p.n_best_history;
    double* var_best_history = st_pop_best_p.var_best_history;
    double* rot_angle_best_history = st_qu_p.rot_angle_best_history;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int tmp_type = optimizer_types_all[iD];
    int iP = iS, iRef = -1, iCur = -1;
    int idx_vec[5] = { -1 };
    int flag_inferior;
    double* p_vec[5] = { NULL };
    double* pCurrent = NULL;
    double* pTrail = NULL;
    double* pRef = NULL;
    int count_selection = 0;
    //////////////////////////////////////////////////////////////////////////
    if(opt_binVar_as_realVar_tag == FLAG_ON ||
       mixed_var_types_tag == FLAG_ON ||
       (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
        if(color_pop && tmp_indx) {
            if(tmp_type == EC_SBX_CUR) count_selection = 0;
            else if(tmp_type == EC_SBX_RAND) count_selection = 1;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection = 2;
            else if(tmp_type == EC_DE_CUR_2) count_selection = 4;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection = 3;
            else if(tmp_type == EC_DE_RAND_2) count_selection = 5;
            else if(tmp_type == EC_DE_2SELECTED) count_selection = 3;
            else if(tmp_type == SI_PSO) count_selection = 0;
            else if(tmp_type == SI_QPSO) count_selection = 1;
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        } else {
            if(tmp_type == EC_SBX_CUR) count_selection = 1;
            else if(tmp_type == EC_SBX_RAND) count_selection = 2;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection = 2;
            else if(tmp_type == EC_DE_CUR_2) count_selection = 4;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection = 3;
            else if(tmp_type == EC_DE_RAND_2) count_selection = 5;
            else if(tmp_type == EC_DE_2SELECTED) count_selection = 3;
            else if(tmp_type == SI_PSO) count_selection = 1;
            else if(tmp_type == SI_QPSO) count_selection = 2;
            else {
                printf("%s: Rank-%d: Unknown optimizer type - optimizer_types_all[%d] = %d, exiting...\n",
                       AT, mpi_rank, iD, tmp_type);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        }
    }
    if(opt_binVar_as_realVar_tag == FLAG_OFF &&
       (mixed_var_types_tag == FLAG_ON ||
        (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
        if(count_selection < 1) count_selection = 1;
    }
    //
    iCur = iP;
    //
    if(color_pop) {
        if(type_select_obj_best == SELECT_OBJ_BEST_TOP_N) {
            if(!tmp_indx) {
                printf("%s: Rank-%d: The indices of top-N individuals are not provided, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_ADDRESS);
            }
            iRef = rnd(0, (int)(nPop_cur * p_best_ratio) - 1);
            iRef = tmp_indx[iRef];
        } else {
            if(flip_r((float)t1_best[iD]) || !cn_best_history) {
                iRef = n_best_history + nPop;
            } else {
                selectSamples(cn_best_history, -1, -1, -1, &iRef, NULL, NULL, NULL, NULL);
                iRef += nPop;
            }
        }
    } else {
        iRef = -1;
    }
    for(int i = 0; i < 5; i++) idx_vec[i] = -1;
    for(int i = 0; i < 5; i++) p_vec[i] = NULL;
    int* tmp_ind_vec = (int*)malloc(nPop_candid_all * sizeof(int));
    int  tmp_count = 0;
    //
    if(PARENT_LOCAL == parent_type[iD]) {
        //selectSamples_niche(&strct_decomp_paras.tableNeighbor[iP * strct_global_paras.nPop], strct_decomp_paras.niche, iCur, iRef, -1, &idx0, &idx1, &idx2, &idx3, &idx4);
        tmp_count = 0;
        for(int i = 0; i < niche; i++) {
            int tmp_ind = tableNeighbor[iP * maxNneighb + i];
            if(tmp_ind == iCur || tmp_ind == iRef) continue;
            tmp_ind_vec[tmp_count++] = tmp_ind;
        }
    } else {
        //selectSamples(strct_global_paras.nPop, iCur, iRef, -1, &idx0, &idx1, &idx2, &idx3, &idx4);
        tmp_count = 0;
        for(int i = 0; i < nPop_candid_all; i++) {
            int tmp_ind = i;
            if(tmp_ind == iCur || tmp_ind == iRef) continue;
            tmp_ind_vec[tmp_count++] = tmp_ind;
        }
    }
    rand_selection(tmp_ind_vec, tmp_count, idx_vec, count_selection);
    flag_inferior = 0;
    if(tmp_type == EC_DE_ARCHIVE ||
       tmp_type == EC_DE_ARCHIVE_RAND) {
        if(count_selection > 0 && curSize_inferior > 0 && flip_r((float)0.5)) {
            flag_inferior = 1;
            idx_vec[0] = rnd(0, curSize_inferior - 1);
        } else {
            flag_inferior = 0;
        }
    }
    free(tmp_ind_vec);
    //////////////////////////////////////////////////////////////////////////
    if(Qubits_angle_opt_tag == FLAG_OFF) {
        pCurrent = &cur_var[iP * nDim];
        pTrail = &osp_var[iD * nDim];
        if(iRef >= 0) {
            if(type_select_obj_best == SELECT_OBJ_BEST_TOP_N)
                pRef = &cur_var[iRef * nDim];
            else {
                iRef -= nPop;
                if(iRef < n_best_history && iRef >= 0) {
                    pRef = &var_best_history[iRef * nDim];
                } else if(iRef == n_best_history) {
                    pRef = var_best_subObjs_all;
                } else {
                    printf("%s: Rank-%d: Wrong index - %d, exiting...\n",
                           AT, mpi_rank, iRef);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_INDEX_INVALID);
                }
            }
        }
        for(int i = 0; i < count_selection; i++) {
            if(i == 0 && flag_inferior) {
                p_vec[i] = &cur_var_inferior[idx_vec[i] * nDim];
            } else {
                p_vec[i] = &cur_var[idx_vec[i] * nDim];
            }
        }
    } else {
        pCurrent = &rot_angle_cur[iP * nDim];
        pTrail = &rot_angle_offspring[iD * nDim];
        if(iRef >= 0) {
            if(type_select_obj_best == SELECT_OBJ_BEST_TOP_N)
                pRef = &rot_angle_cur[iRef * nDim];
            else {
                iRef -= nPop;
                if(iRef < n_best_history && iRef >= 0) {
                    pRef = &rot_angle_best_history[iRef * nDim];
                } else if(iRef == n_best_history) {
                    pRef = rot_angle_best_subObjs_all;
                } else {
                    printf("%s: Rank-%d: Wrong index - %d, exiting...\n",
                           AT, mpi_rank, iRef);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_INDEX_INVALID);
                }
            }
        }
        for(int i = 0; i < count_selection; i++) {
            if(i == 0 && flag_inferior) {
                p_vec[i] = &rot_angle_cur_inferior[idx_vec[i] * nDim];
            } else {
                p_vec[i] = &rot_angle_cur[idx_vec[i] * nDim];
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    if(color_pop && tmp_indx) {
        //////////////////////////////////////////////////////////////////////////
        if(opt_binVar_as_realVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR)
                SBX_classic(pCurrent, pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_SBX_RAND)
                SBX_classic(p_vec[0], pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE)
                DE_selected1_1_exp(pCurrent, p_vec[1], p_vec[0], pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_CUR_2)
                DE_selected1_2_exp(pCurrent, p_vec[0], p_vec[1], p_vec[2], p_vec[3], pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND)
                DE_selected1_1_exp(p_vec[2], p_vec[1], p_vec[0], pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_RAND_2)
                DE_selected1_2_exp(p_vec[0], p_vec[1], p_vec[2], p_vec[3], p_vec[4], pRef, pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_2SELECTED)
                DE_selected1_1_exp(p_vec[0], p_vec[1], p_vec[2], pCurrent, pCurrent, pTrail, iP, iD);
            else if(tmp_type == SI_PSO) {
                double* pVel;
                pVel = &velocity[iD * nDim];
                PSO_classic(pRef, pCurrent, pTrail, pVel, iP, iD);
            } else if(tmp_type == SI_QPSO)
                QPSO_classic_2(p_vec[0], pRef, pCurrent, pTrail, NULL, iP, iD);
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        }
        //////////////////////////////////////////////////////////////////////////
        if(opt_binVar_as_realVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            evo_bin_commonality(pCurrent, p_vec[0], pCurrent, pTrail, iD);
        }
    } else {
        //////////////////////////////////////////////////////////////////////////
        if(opt_binVar_as_realVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR)
                SBX_classic(pCurrent, p_vec[0], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_SBX_RAND)
                SBX_classic(p_vec[0], p_vec[1], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE)
                DE_1_exp(pCurrent, p_vec[1], p_vec[0], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_CUR_2)
                DE_2_exp(pCurrent, p_vec[0], p_vec[1], p_vec[2], p_vec[3], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND)
                DE_1_exp(p_vec[2], p_vec[1], p_vec[0], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_RAND_2)
                DE_2_exp(p_vec[0], p_vec[1], p_vec[2], p_vec[3], p_vec[4], pCurrent, pTrail, iP, iD);
            else if(tmp_type == EC_DE_2SELECTED)
                DE_selected1_1_exp(p_vec[0], p_vec[1], p_vec[2], pCurrent, pCurrent, pTrail, iP, iD);
            else if(tmp_type == SI_PSO) {
                double* pVel;
                pVel = &velocity[iD * nDim];
                PSO_classic(p_vec[0], pCurrent, pTrail, pVel, iP, iD);
            } else if(tmp_type == SI_QPSO)
                QPSO_classic_2(p_vec[0], p_vec[1], pCurrent, pTrail, NULL, iP, iD);
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        }
        //////////////////////////////////////////////////////////////////////////
        if(opt_binVar_as_realVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            evo_bin_commonality(pCurrent, p_vec[0], pCurrent, pTrail, iD);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    if(Qubits_angle_opt_tag == FLAG_ON) {
        pCurrent = &cur_var[iP * nDim];
        pTrail = &osp_var[iD * nDim];
        double* p_rot = &rot_angle_offspring[iD * nDim];
        Quantum_transform_update(pCurrent, pTrail, p_rot, iP);
    }
    //
    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//
void gen_offspring_allObjs_local_ND()
{
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int nArch = st_archive_p.nArch;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* arc_var_inferior = st_archive_p.var_inferior;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;
    int optimize_binaryVar_as_doubleVar_tag = st_ctrl_p.opt_binVar_as_realVar_tag;
    int mixed_var_types_tag = st_ctrl_p.mixed_var_types_tag;
    int type_var_encoding = st_ctrl_p.type_var_encoding;
    int color_pop = st_MPI_p.color_pop;
    int* optimizer_types_all = st_optimizer_p.optimizer_types_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    int* parent_type = st_decomp_p.parent_type;
    int niche = st_decomp_p.niche;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop;
    int curSize_inferior = st_pop_evo_cur.curSize_inferior;
    double* velocity = st_PSO_p.velocity;
    int num_selected = st_global_p.num_selected;
    int* cloneNum = st_global_p.cloneNum;
    int* selectedIndv = st_global_p.selectedIndv;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int i = -1, j = -1, iNic = -1, iCur = -1;
    int iP = -1;
    int r_vec1[2] = { -1 };
    int r_vec2[2] = { -1 };
    double* p_vec1[2] = { NULL };
    double* p_vec2[2] = { NULL };
    double* pCurrent = NULL;
    double* pTrail = NULL;
    int count_selection1;
    int count_selection2;
    int count = 0;
    //////////////////////////////////////////////////////////////////////////
    for(i = 0; i < num_selected; i++) {
        for(j = 0; j < cloneNum[i]; j++) {
            int tmp_type = optimizer_types_all[count];
            count_selection1 = 0;
            count_selection2 = 0;
            if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
               mixed_var_types_tag == FLAG_ON ||
               (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
                if(tmp_type == EC_SBX_CUR) count_selection1 = 1;
                else if(tmp_type == EC_SBX_RAND) count_selection1 = 2;
                else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection1 = 0;
                else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection1 = 1;
                else if(tmp_type == EC_DE_2SELECTED) count_selection1 = 1;
                else if(tmp_type == SI_PSO) count_selection1 = 1;
                else if(tmp_type == SI_QPSO) count_selection1 = 2;
                else {
                    printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                           AT, mpi_rank);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
                }
                if(tmp_type == EC_SBX_CUR) count_selection2 = 0;
                else if(tmp_type == EC_SBX_RAND) count_selection2 = 0;
                else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection2 = 2;
                else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection2 = 2;
                else if(tmp_type == EC_DE_2SELECTED) count_selection2 = 2;
                else if(tmp_type == SI_PSO) count_selection2 = 0;
                else if(tmp_type == SI_QPSO) count_selection2 = 0;
            }
            if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
               (mixed_var_types_tag == FLAG_ON ||
                (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
                if(count_selection1 < 1) count_selection1 = 1;
                if(count_selection2 < 1) count_selection2 = 1;
            }
            //
            iP = selectedIndv[i];
            iNic = i;
            iCur = iP;
            pCurrent = &arc_var[iP * nDim];
            pTrail = &osp_var[count * nDim];
            //
            for(int i = 0; i < 2; i++) r_vec1[i] = r_vec2[i] = -1;
            for(int i = 0; i < 2; i++) p_vec1[i] = p_vec2[i] = NULL;
            int* tmp_ind_vec = (int*)malloc(nArch * 10 * sizeof(int));
            int tmp_count = 0;
            //
            if(num_selected >= 3) {
                //selectSamples(strct_global_paras.num_selected, iNic, -1, -1, &rb, &r0, NULL, NULL, NULL);
                for(int iInd = 0; iInd < num_selected; iInd++) {
                    if(selectedIndv[iInd] == iCur) continue;
                    tmp_ind_vec[tmp_count++] = selectedIndv[iInd];
                }
            } else {
                //selectSamples(strct_archive_info.nArch, iCur, -1, -1, &rb, &r0, NULL, NULL, NULL);
                for(int iInd = 0; iInd < nArch; iInd++) {
                    if(iInd == iCur) continue;
                    tmp_ind_vec[tmp_count++] = iInd;
                }
            }
            rand_selection(tmp_ind_vec, tmp_count, r_vec1, count_selection1);
            //selectSamples_RR(strct_global_paras.num_selected, strct_global_paras.selectedProb, i, &r1, &r2, NULL);
            //pb = &strct_archive_info.var_archive[rb * strct_global_paras.nDim];
            //p0 = &strct_archive_info.var_archive[r0 * strct_global_paras.nDim];
            for(int iInd = 0; iInd < count_selection1; iInd++) {
                p_vec1[iInd] = &arc_var[r_vec1[iInd] * nDim];
            }
            //selectSamples(strct_archive_info.nArch, iCur, rb, r0, &r1, NULL, NULL, NULL, NULL);
            tmp_count = 0;
            for(int iInd = 0; iInd < nArch; iInd++) {
                if(iInd == iCur) continue;
                int tmp_flag = 0;
                for(int jInd = 0; jInd < count_selection1; jInd++) {
                    if(r_vec1[jInd] == iInd)
                        tmp_flag = 1;
                }
                if(tmp_flag) continue;
                tmp_ind_vec[tmp_count++] = iInd;
            }
            rand_selection(tmp_ind_vec, tmp_count, r_vec2, count_selection2);
            //p1 = &strct_archive_info.var_archive[r1 * strct_global_paras.nDim];
            for(int iInd = 0; iInd < count_selection2; iInd++) {
                p_vec2[iInd] = &arc_var[r_vec2[iInd] * nDim];
            }
            if(tmp_type == EC_DE_ARCHIVE ||
               tmp_type == EC_DE_ARCHIVE_RAND) {
                if(count_selection2 > 0 && curSize_inferior && flip_r((float)0.5)) {
                    r_vec2[0] = rnd(0, curSize_inferior - 1);
                    p_vec2[0] = &arc_var_inferior[r_vec2[0] * nDim];
                }
            }
            free(tmp_ind_vec);
            //////////////////////////////////////////////////////////////////////////
            if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
               mixed_var_types_tag == FLAG_ON ||
               (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
                if(tmp_type == EC_SBX_CUR)
                    SBX_classic(pCurrent, p_vec1[0], pCurrent, pTrail, iP, count);
                else if(tmp_type == EC_SBX_RAND)
                    SBX_classic(p_vec1[0], p_vec1[1], pCurrent, pTrail, iP, count);
                else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE)
                    DE_1_exp(pCurrent, p_vec2[1], p_vec2[0], pCurrent, pTrail, iP, count);
                else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND)
                    DE_1_exp(p_vec1[0], p_vec2[1], p_vec2[0], pCurrent, pTrail, iP, count);
                else if(tmp_type == EC_DE_2SELECTED)
                    DE_selected1_1_exp(p_vec1[0], p_vec2[0], p_vec2[1], pCurrent, pCurrent, pTrail, iP, count);
                else if(tmp_type == SI_PSO) {
                    double* pVel;
                    pVel = &velocity[count * nDim];
                    PSO_classic(p_vec1[0], pCurrent, pTrail, pVel, iP, count);
                } else if(tmp_type == SI_QPSO)
                    QPSO_classic_2(p_vec1[0], p_vec1[1], pCurrent, pTrail, NULL, iP, count);
                else {
                    printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                           AT, mpi_rank);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
                }
            }
            //////////////////////////////////////////////////////////////////////////
            if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
               (mixed_var_types_tag == FLAG_ON ||
                (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
                evo_bin_commonality(p_vec1[0], p_vec2[0], pCurrent, pTrail, count);
            }
            count++;
        }
    }
    //
    return;
}

void gen_offspring_allObjs_global_ND()
{
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int nArch = st_archive_p.nArch;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* arc_var_inferior = st_archive_p.var_inferior;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;
    int optimize_binaryVar_as_doubleVar_tag = st_ctrl_p.opt_binVar_as_realVar_tag;
    int mixed_var_types_tag = st_ctrl_p.mixed_var_types_tag;
    int type_var_encoding = st_ctrl_p.type_var_encoding;
    int color_pop = st_MPI_p.color_pop;
    int* optimizer_types_all = st_optimizer_p.optimizer_types_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    int* parent_type = st_decomp_p.parent_type;
    int niche = st_decomp_p.niche;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop;
    int curSize_inferior = st_pop_evo_cur.curSize_inferior;
    double* velocity = st_PSO_p.velocity;
    int num_selected = st_global_p.num_selected;
    int* cloneNum = st_global_p.cloneNum;
    int* selectedIndv = st_global_p.selectedIndv;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int /*i, j, */iNic = -1, iCur = -1;
    int iP = -1;
    int r_vec1[2] = { -1 };
    int r_vec2[2] = { -1 };
    double* p_vec1[2] = { NULL };
    double* p_vec2[2] = { NULL };
    double* pCurrent = NULL;
    double* pTrail = NULL;
    int count_selection1;
    int count_selection2;
    //////////////////////////////////////////////////////////////////////////
    for(iP = 0; iP < nArch; iP++) {
        int tmp_type = optimizer_types_all[iP];
        count_selection1 = 0;
        count_selection2 = 0;
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR) count_selection1 = 1;
            else if(tmp_type == EC_SBX_RAND) count_selection1 = 2;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection1 = 0;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection1 = 1;
            else if(tmp_type == EC_DE_2SELECTED) count_selection1 = 1;
            else if(tmp_type == SI_PSO) count_selection1 = 1;
            else if(tmp_type == SI_QPSO) count_selection1 = 2;
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
            if(tmp_type == EC_SBX_CUR) count_selection2 = 0;
            else if(tmp_type == EC_SBX_RAND) count_selection2 = 0;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection2 = 2;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection2 = 2;
            else if(tmp_type == EC_DE_2SELECTED) count_selection2 = 2;
            else if(tmp_type == SI_PSO) count_selection2 = 0;
            else if(tmp_type == SI_QPSO) count_selection2 = 0;
        }
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            if(count_selection1 < 1) count_selection1 = 1;
            if(count_selection2 < 1) count_selection2 = 1;
        }
        //
        iNic = -1;
        for(int i = 0; i < num_selected; i++) {
            if(selectedIndv[i] == iP &&
               cloneNum[i] > 0) {
                iNic = i;
            }
        }
        iCur = iP;
        pCurrent = &arc_var[iP * nDim];
        pTrail = &osp_var[iP * nDim];
        //
        for(int i = 0; i < 2; i++) r_vec1[i] = r_vec2[i] = -1;
        for(int i = 0; i < 2; i++) p_vec1[i] = p_vec2[i] = NULL;
        int* tmp_ind_vec = (int*)malloc(nArch * 10 * sizeof(int));
        if(st_ctrl_p.type_clone_evo == CLONE_EVO_GLOBAL) {
            int tmp_count = 0;
            //
            if(num_selected >= 3) {
                //selectSamples_clone_RR(strct_global_paras.num_selected, strct_global_paras.cloneNum, iNic, -1, -1, &rb, &r0, NULL);
                for(int iInd = 0; iInd < num_selected; iInd++) {
                    if(iInd == iNic) continue;
                    if(selectedIndv[iInd] == iCur) continue;
                    tmp_ind_vec[tmp_count++] = selectedIndv[iInd];
                }
            } else {
                //selectSamples(strct_archive_info.nArch, iCur, -1, -1, &rb, &r0, NULL, NULL, NULL);
                for(int iInd = 0; iInd < nArch; iInd++) {
                    if(iInd == iCur) continue;
                    tmp_ind_vec[tmp_count++] = iInd;
                }
            }
            rand_selection(tmp_ind_vec, tmp_count, r_vec1, count_selection1);
            //pb = &strct_archive_info.var_archive[rb * strct_global_paras.nDim];
            //p0 = &strct_archive_info.var_archive[r0 * strct_global_paras.nDim];
            for(int iInd = 0; iInd < count_selection1; iInd++) {
                p_vec1[iInd] = &arc_var[r_vec1[iInd] * nDim];
            }
            //selectSamples(strct_archive_info.nArch, iCur, rb, r0, &r1, NULL, NULL, NULL, NULL);
            tmp_count = 0;
            for(int iInd = 0; iInd < nArch; iInd++) {
                if(iInd == iCur) continue;
                int tmp_flag = 0;
                for(int jInd = 0; jInd < count_selection1; jInd++) {
                    if(r_vec1[jInd] == iInd)
                        tmp_flag = 1;
                }
                if(tmp_flag) continue;
                tmp_ind_vec[tmp_count++] = iInd;
            }
            rand_selection(tmp_ind_vec, tmp_count, r_vec2, count_selection2);
            //p1 = &strct_archive_info.var_archive[r1 * strct_global_paras.nDim];
            for(int iInd = 0; iInd < count_selection2; iInd++) {
                p_vec2[iInd] = &arc_var[r_vec2[iInd] * nDim];
            }
        } else { // st_ctrl_p.type_clone_evo == CLONE_EVO_NONE
            int tmp_count = 0;
            //
            for(int iInd = 0; iInd < nArch; iInd++) {
                if(iInd == iCur) continue;
                tmp_ind_vec[tmp_count++] = iInd;
            }
            tour_selection_ND(tmp_ind_vec, tmp_count, r_vec1, count_selection1, 1);
            for(int iInd = 0; iInd < count_selection1; iInd++) {
                p_vec1[iInd] = &arc_var[r_vec1[iInd] * nDim];
            }
            tmp_count = 0;
            for(int iInd = 0; iInd < nArch; iInd++) {
                if(iInd == iCur) continue;
                int tmp_flag = 0;
                for(int jInd = 0; jInd < count_selection1; jInd++) {
                    if(r_vec1[jInd] == iInd)
                        tmp_flag = 1;
                }
                if(tmp_flag) continue;
                tmp_ind_vec[tmp_count++] = iInd;
            }
            tour_selection_ND(tmp_ind_vec, tmp_count, r_vec2, count_selection2, 1);
            for(int iInd = 0; iInd < count_selection2; iInd++) {
                p_vec2[iInd] = &arc_var[r_vec2[iInd] * nDim];
            }
        }
        if(tmp_type == EC_DE_ARCHIVE ||
           tmp_type == EC_DE_ARCHIVE_RAND) {
            if(count_selection2 > 0 && curSize_inferior && flip_r((float)0.5)) {
                r_vec2[0] = rnd(0, curSize_inferior - 1);
                p_vec2[0] = &arc_var_inferior[r_vec2[0] * nDim];
            }
        }
        free(tmp_ind_vec);
        //////////////////////////////////////////////////////////////////////////
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR)
                SBX_classic(pCurrent, p_vec1[0], pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_SBX_RAND)
                SBX_classic(p_vec1[0], p_vec1[1], pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE)
                DE_1_exp(pCurrent, p_vec2[1], p_vec2[0], pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND)
                DE_1_exp(p_vec1[0], p_vec2[1], p_vec2[0], pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_2SELECTED)
                DE_selected1_1_exp(p_vec1[0], p_vec2[0], p_vec2[1], pCurrent, pCurrent, pTrail, iP, iP);
            else if(tmp_type == SI_PSO) {
                double* pVel;
                pVel = &velocity[iP * nDim];
                PSO_classic(p_vec1[0], pCurrent, pTrail, pVel, iP, iP);
            } else if(tmp_type == SI_QPSO)
                QPSO_classic_2(p_vec1[0], p_vec1[1], pCurrent, pTrail, NULL, iP, iP);
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        }
        //////////////////////////////////////////////////////////////////////////
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            evo_bin_commonality(p_vec1[0], p_vec2[0], pCurrent, pTrail, iP);
        }
    }
    //
    return;
}

void gen_offspring_subObj_ND()
{
    int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int nArch = st_archive_p.nArch;
    int nArch_sub = st_archive_p.nArch_sub;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* arc_var_inferior = st_archive_p.var_inferior;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;
    int optimize_binaryVar_as_doubleVar_tag = st_ctrl_p.opt_binVar_as_realVar_tag;
    int mixed_var_types_tag = st_ctrl_p.mixed_var_types_tag;
    int type_var_encoding = st_ctrl_p.type_var_encoding;
    int color_pop = st_MPI_p.color_pop;
    int* optimizer_types_all = st_optimizer_p.optimizer_types_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    double p_best_ratio = st_optimizer_p.p_best_ratio;
    int* parent_type = st_decomp_p.parent_type;
    int niche = st_decomp_p.niche;
    int* tableNeighbor = st_decomp_p.tableNeighbor;
    int maxNneighb = nPop;
    int curSize_inferior = st_pop_evo_cur.curSize_inferior;
    double* velocity = st_PSO_p.velocity;
    int num_selected = st_global_p.num_selected;
    int* cloneNum = st_global_p.cloneNum;
    int* selectedIndv = st_global_p.selectedIndv;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    int iP = -1, iCur = -1;
    int rb = -1, r0 = -1, r_vec2[2] = { -1 };
    double* pb = NULL, * p0 = NULL, * p_vec2[2] = { NULL };
    double* pCurrent = NULL;
    double* pTrail = NULL;
    int count_selection1;
    int count_selection2;
    int depth = 5;
    //////////////////////////////////////////////////////////////////////////
    int ind_prob = color_pop - 1;
    int* ind_sep = (int*)calloc(nArch_sub, sizeof(int));
    for(int i = 0; i < nArch_sub; i++) ind_sep[i] = i;
    //shuffle(ind_sep, strct_archive_info.nArch_sub);
    memcpy(repo_obj, arc_obj, nArch_sub * nObj * sizeof(double));
    qSortFrontObj(ind_prob, ind_sep, 0, nArch_sub - 1);
    //////////////////////////////////////////////////////////////////////////
    for(iP = 0; iP < nArch_sub; iP++) {
        int tmp_type = optimizer_types_all[iP];
        count_selection1 = 0;
        count_selection2 = 0;
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR) count_selection1 = 1;
            else if(tmp_type == EC_SBX_RAND) count_selection1 = 2;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection1 = 1;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection1 = 2;
            else if(tmp_type == EC_DE_2SELECTED) count_selection1 = 1;
            else if(tmp_type == SI_PSO) count_selection1 = 1;
            else if(tmp_type == SI_QPSO) count_selection1 = 2;
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
            if(tmp_type == EC_SBX_CUR) count_selection2 = 0;
            else if(tmp_type == EC_SBX_RAND) count_selection2 = 0;
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE) count_selection2 = 2;
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND) count_selection2 = 2;
            else if(tmp_type == EC_DE_2SELECTED) count_selection2 = 2;
            else if(tmp_type == SI_PSO) count_selection2 = 0;
            else if(tmp_type == SI_QPSO) count_selection2 = 0;
        }
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            if(count_selection1 < 1) count_selection1 = 1;
            if(count_selection2 < 1) count_selection2 = 1;
        }
        //
        iCur = iP;
        pCurrent = &arc_var[iP * nDim];
        pTrail = &osp_var[iP * nDim];
        //
        for(int i = 0; i < 2; i++) r_vec2[i] = -1;
        for(int i = 0; i < 2; i++) p_vec2[i] = NULL;
        int* tmp_ind_vec = (int*)malloc(nArch_sub * 10 * sizeof(int));
        int tmp_count = 0;
        //
        if(tmp_type == EC_DE_ARCHIVE ||
           tmp_type == EC_DE_ARCHIVE_RAND) {
            int tmp_N = (int)(nArch_sub * p_best_ratio);
            if(tmp_N < 2) tmp_N = 2 < nArch_sub ? 2 : nArch_sub;
            do {
                rb = rnd(0, tmp_N - 1);
                rb = ind_sep[rb];
            } while(rb == iCur);
        } else {
            tourSelectSamples_sub(nArch_sub, depth, color_pop - 1, arc_obj, iCur, -1, -1, &rb, NULL, NULL);
        }
        pb = &arc_var[rb * nDim];
        //selectSamples(nArch_sep, iP, &r1, &r2, NULL);
        tourSelectSamples_sub(nArch_sub, depth, color_pop - 1, arc_obj, iCur, rb, -1, &r0, NULL, NULL);
        p0 = &arc_var[r0 * nDim];
        //selectSamples(strct_archive_info.nArch_sub, iCur, rb, r0, &r1, NULL, NULL, NULL, NULL);
        tmp_count = 0;
        for(int iInd = 0; iInd < nArch_sub; iInd++) {
            if(iInd == iCur || iInd == rb || iInd == r0) continue;
            tmp_ind_vec[tmp_count++] = iInd;
        }
        rand_selection(tmp_ind_vec, tmp_count, r_vec2, count_selection2);
        //p1 = &strct_archive_info.var_archive[r1 * strct_global_paras.nDim];
        for(int iInd = 0; iInd < count_selection2; iInd++) {
            p_vec2[iInd] = &arc_var[r_vec2[iInd] * nDim];
        }
        if(tmp_type == EC_DE_ARCHIVE ||
           tmp_type == EC_DE_ARCHIVE_RAND) {
            if(count_selection2 > 0 && curSize_inferior && flip_r((float)0.5)) {
                r_vec2[0] = rnd(0, curSize_inferior - 1);
                p_vec2[0] = &arc_var_inferior[r_vec2[0] * nDim];
            }
        }
        free(tmp_ind_vec);
        //////////////////////////////////////////////////////////////////////////
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_ON ||
           mixed_var_types_tag == FLAG_ON ||
           (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_DOUBLE)) {
            if(tmp_type == EC_SBX_CUR)
                SBX_classic(pCurrent, p0, pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_SBX_RAND)
                SBX_classic(pb, p0, pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_CUR_1 || tmp_type == EC_DE_ARCHIVE)
                DE_selected1_1_exp(pCurrent, p_vec2[1], p_vec2[0], pb, pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_RAND_1 || tmp_type == EC_DE_ARCHIVE_RAND)
                DE_selected1_1_exp(p0, p_vec2[1], p_vec2[0], pb, pCurrent, pTrail, iP, iP);
            else if(tmp_type == EC_DE_2SELECTED)
                DE_selected1_1_exp(p0, p_vec2[0], p_vec2[1], pCurrent, pCurrent, pTrail, iP, iP);
            else if(tmp_type == SI_PSO) {
                double* pVel;
                pVel = &velocity[iP * nDim];
                PSO_classic(pb, pCurrent, pTrail, pVel, iP, iP);
            } else if(tmp_type == SI_QPSO)
                QPSO_classic_2(p0, pb, pCurrent, pTrail, NULL, iP, iP);
            else {
                printf("%s: Rank-%d: Unknown optimizer type, exiting...\n",
                       AT, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
            }
        }
        //////////////////////////////////////////////////////////////////////////
        if(optimize_binaryVar_as_doubleVar_tag == FLAG_OFF &&
           (mixed_var_types_tag == FLAG_ON ||
            (mixed_var_types_tag == FLAG_OFF && type_var_encoding == VAR_BINARY))) {
            evo_bin_commonality(p0, p_vec2[0], pCurrent, pTrail, iP);
        }
    }
    //
    free(ind_sep);
    //
    return;
}
