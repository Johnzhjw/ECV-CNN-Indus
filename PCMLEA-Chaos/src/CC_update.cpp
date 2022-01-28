# include "global.h"
# include <math.h>

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// one to one
void update_population_1to1(int iP)
{
    double f1;
    double f2;

    f1 = fitnessFunction(&st_pop_evo_offspring.obj[iP * st_global_p.nObj],
                         &st_decomp_p.weights_all[iP * st_global_p.nObj]);
    f2 = fitnessFunction(&st_pop_evo_cur.obj[iP * st_global_p.nObj],
                         &st_decomp_p.weights_all[iP * st_global_p.nObj]);
    if(f1 < f2) {
        st_DE_p.Sflag[iP] = 1;
        int tmp_ind;
        if(st_pop_evo_cur.curSize_inferior < st_global_p.nPop) {
            tmp_ind = st_pop_evo_cur.curSize_inferior;
            st_pop_evo_cur.curSize_inferior++;
        } else {
            selectSamples(st_global_p.nPop, -1, -1, -1, &tmp_ind, NULL, NULL, NULL, NULL);
            st_pop_evo_cur.curSize_inferior = st_global_p.nPop;
        }
        memcpy(&st_pop_evo_cur.var_inferior[tmp_ind * st_global_p.nDim], &st_pop_evo_cur.var[iP * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        memcpy(&st_qu_p.rot_angle_cur_inferior[tmp_ind * st_global_p.nDim],
               &st_qu_p.rot_angle_cur[iP * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        memcpy(&st_pop_evo_cur.var[iP * st_global_p.nDim], &st_pop_evo_offspring.var[iP * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        memcpy(&st_pop_evo_cur.obj[iP * st_global_p.nObj], &st_pop_evo_offspring.obj[iP * st_global_p.nObj],
               st_global_p.nObj * sizeof(double));
        memcpy(&st_qu_p.rot_angle_cur[iP * st_global_p.nDim],
               &st_qu_p.rot_angle_offspring[iP * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        st_decomp_p.fitCur[iP] = f1;
        st_decomp_p.fitImprove[iP] += f2 - f1;
        st_decomp_p.countFitImprove[iP]++;
        one_utility(f2, f1, iP);
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// DECOMPOSITION
void update_population_DECOM(int iP, int iC, int nPop, double* osp_obj, double* osp_var,
                             double* weights_all, int useSflag, int* Sflag,
                             double* rot_angle_offspring,
                             int niche, int niche_local, int* tableNeighbor, int* tableNeighbor_local, int maxNneighb, int parent_type)
{
    int MFI_update_tag = st_ctrl_p.MFI_update_tag;
    //int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    //double* osp_obj = st_pop_evo_offspring.obj;//
    double* cur_obj = st_pop_evo_cur.obj;
    //double* osp_var = st_pop_evo_offspring.var;//
    double* cur_var = st_pop_evo_cur.var;
    //double* weights_all = st_decomp_p.weights_all;//
    int* curSize_inferior = &st_pop_evo_cur.curSize_inferior;
    //int  useSflag = 1; //
    //int* Sflag = st_DE_p.Sflag;//
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    //double* rot_angle_offspring = st_qu_p.rot_angle_offspring;//
    //int niche = st_decomp_p.niche;//
    int limit = st_decomp_p.limit;
    //int* tableNeighbor = st_decomp_p.tableNeighbor;//
    //int maxNneighb = nPop; //
    double* fitCur = st_decomp_p.fitCur;
    double* fitImprove = st_decomp_p.fitImprove;
    int* countFitImprove = st_decomp_p.countFitImprove;
    //int parent_type = st_decomp_p.parent_type[iC];//
    int flag_check_more_update_DECOM = st_ctrl_p.flag_check_more_update_DECOM;
    //
    int i, j;
    double f1;
    double f2;
    int* tIndex = (int*)calloc(nPop + niche, sizeof(int));
    int count = 0;
    if(MFI_update_tag == FLAG_ON) {
        int index = -1;
        double MaxFitImp = -1.0e10;
        for(int i_tmp = 0; i_tmp < nPop; i_tmp++) {
            f1 = fitnessFunction(&osp_obj[iC * nObj], &weights_all[i_tmp * nObj]);
            f2 = fitnessFunction(&cur_obj[i_tmp * nObj], &weights_all[i_tmp * nObj]);
            if(f2 - f1 > MaxFitImp) {
                MaxFitImp = f2 - f1;
                index = i_tmp;
            }
        }
        if(MaxFitImp > 0) {
            OldSoluUpdate(index, 5, nPop, weights_all, niche_local, tableNeighbor_local, maxNneighb);
            if(useSflag) Sflag[iC] = 1;
            f1 = fitnessFunction(&osp_obj[iC * nObj], &weights_all[index * nObj]);
            f2 = fitnessFunction(&cur_obj[index * nObj], &weights_all[index * nObj]);
            int tmp_ind;
            if((*curSize_inferior) < nPop) {
                tmp_ind = (*curSize_inferior);
                (*curSize_inferior)++;
            } else {
                selectSamples(nPop, -1, -1, -1, &tmp_ind, NULL, NULL, NULL, NULL);
                (*curSize_inferior) = nPop;
            }
            memcpy(&cur_var_inferior[tmp_ind * nDim], &cur_var[index * nDim], nDim * sizeof(double));
            memcpy(&cur_var[index * nDim], &osp_var[iC * nDim], nDim * sizeof(double));
            memcpy(&cur_obj[index * nObj], &osp_obj[iC * nObj], nObj * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON) {
                memcpy(&rot_angle_cur_inferior[tmp_ind * nDim], &rot_angle_cur[index * nDim], nDim * sizeof(double));
                memcpy(&rot_angle_cur[index * nDim], &rot_angle_offspring[iC * nDim], nDim * sizeof(double));
            }
            fitCur[index] = f1;
            fitImprove[iC] += f2 - f1;
            countFitImprove[iC]++;
            count++;
            one_utility(f2, f1, index);
            checkUpdatedIndx(index);
        }
    }
    //
    i = iP;
    if(PARENT_LOCAL == parent_type) {
        for(j = 0; j < niche; j++) tIndex[j] = j;
        shuffle(tIndex, niche);
        int realInd;
        for(j = 0; j < niche; j++) {
            if(count >= limit) break;
            realInd = tableNeighbor[i * maxNneighb + tIndex[j]];
            if(realInd < 0 || realInd >= nPop) continue;
            f1 = fitnessFunction(&osp_obj[iC * nObj], &weights_all[realInd * nObj]);
            f2 = fitnessFunction(&cur_obj[realInd * nObj], &weights_all[realInd * nObj]);
            if(f1 < f2) {
                if(useSflag) Sflag[iC] = 1;
                if(MFI_update_tag == FLAG_ON) {
                    OldSoluUpdate(realInd, 5, nPop, weights_all, niche_local, tableNeighbor_local, maxNneighb);
                }
                int tmp_ind;
                if((*curSize_inferior) < nPop) {
                    tmp_ind = (*curSize_inferior);
                    (*curSize_inferior)++;
                } else {
                    selectSamples(nPop, -1, -1, -1, &tmp_ind, NULL, NULL, NULL, NULL);
                    (*curSize_inferior) = nPop;
                }
                memcpy(&cur_var_inferior[tmp_ind * nDim], &cur_var[realInd * nDim], nDim * sizeof(double));
                memcpy(&cur_var[realInd * nDim], &osp_var[iC * nDim], nDim * sizeof(double));
                memcpy(&cur_obj[realInd * nObj], &osp_obj[iC * nObj], nObj * sizeof(double));
                if(Qubits_angle_opt_tag == FLAG_ON) {
                    memcpy(&rot_angle_cur_inferior[tmp_ind * nDim], &rot_angle_cur[realInd * nDim], nDim * sizeof(double));
                    memcpy(&rot_angle_cur[realInd * nDim], &rot_angle_offspring[iC * nDim], nDim * sizeof(double));
                }
                fitCur[realInd] = f1;
                fitImprove[iC] += f2 - f1;
                countFitImprove[iC]++;
                count++;
                one_utility(f2, f1, realInd);
                checkUpdatedIndx(realInd);
            }
        }
    }
    if(count < limit && (!(flag_check_more_update_DECOM == FLAG_OFF && parent_type == PARENT_LOCAL))) {
        for(j = 0; j < nPop; j++) tIndex[j] = j;
        shuffle(tIndex, nPop);
        for(j = 0; j < nPop; j++) {
            if(count >= limit) break;
            int realInd;
            realInd = tIndex[j];
            f1 = fitnessFunction(&osp_obj[iC * nObj], &weights_all[realInd * nObj]);
            f2 = fitnessFunction(&cur_obj[realInd * nObj], &weights_all[realInd * nObj]);
            if(f1 < f2) {
                if(useSflag) Sflag[iC] = 1;
                if(MFI_update_tag == FLAG_ON) {
                    OldSoluUpdate(realInd, 5, nPop, weights_all, niche_local, tableNeighbor_local, maxNneighb);
                }
                int tmp_ind;
                if((*curSize_inferior) < nPop) {
                    tmp_ind = (*curSize_inferior);
                    (*curSize_inferior)++;
                } else {
                    selectSamples(nPop, -1, -1, -1, &tmp_ind, NULL, NULL, NULL, NULL);
                    (*curSize_inferior) = nPop;
                }
                memcpy(&cur_var_inferior[tmp_ind * nDim], &cur_var[realInd * nDim], nDim * sizeof(double));
                memcpy(&cur_var[realInd * nDim], &osp_var[iC * nDim], nDim * sizeof(double));
                memcpy(&cur_obj[realInd * nObj], &osp_obj[iC * nObj], nObj * sizeof(double));
                if(Qubits_angle_opt_tag == FLAG_ON) {
                    memcpy(&rot_angle_cur_inferior[tmp_ind * nDim], &rot_angle_cur[realInd * nDim], nDim * sizeof(double));
                    memcpy(&rot_angle_cur[realInd * nDim], &rot_angle_offspring[iC * nDim], nDim * sizeof(double));
                }
                fitCur[realInd] = f1;
                fitImprove[iC] += f2 - f1;
                countFitImprove[iC]++;
                count++;
                one_utility(f2, f1, realInd);
                checkUpdatedIndx(realInd);
            }
        }
    }
    free(tIndex);
    //
    return;
}

void update_population_DECOM_from_transNeigh()
{
    int nPop = st_global_p.nPop;//
    int nPop_mine = st_global_p.nPop_mine;//
    double* osp_obj = st_pop_evo_offspring.obj;//
    double* osp_var = st_pop_evo_offspring.var;//
    double* weights_all = st_decomp_p.weights_all;//
    int  useSflag = 0; //
    int* Sflag = NULL;//
    double* rot_angle_offspring = st_qu_p.rot_angle_offspring;//
    int niche = st_decomp_p.niche;//
    int niche_local = st_decomp_p.niche_local;//
    int* tableNeighbor = st_decomp_p.tableNeighbor;//
    int* tableNeighbor_local = st_decomp_p.tableNeighbor_local;//
    int maxNneighb = nPop; //
    int num_trail_per_gen = st_global_p.num_trail_per_gen;
    int n_neighbor_left = st_pop_comm_p.n_neighbor_left;
    int n_neighbor_right = st_pop_comm_p.n_neighbor_right;
    int* parent_type = (int*)malloc(nPop * sizeof(int)); //
    double th_select = st_decomp_p.th_select;
    int* tmpInd = (int*)malloc(nPop * sizeof(int));
    // left
    osp_obj = st_pop_comm_p.obj_left;
    osp_var = st_pop_comm_p.var_left;
    weights_all = st_decomp_p.weights_mine;
    rot_angle_offspring = st_pop_comm_p.rot_angle_left;
    niche = st_decomp_p.niche_neighb;
    tableNeighbor = &st_decomp_p.tableNeighbor[nPop_mine * maxNneighb];
    for(int i = 0; i < nPop; i++) {
        if(flip_r((float)th_select)) parent_type[i] = PARENT_LOCAL;
        else parent_type[i] = PARENT_GLOBAL;
    }
    for(int i = 0; i < n_neighbor_left; i++) tmpInd[i] = i;
    shuffle(tmpInd, n_neighbor_left);
    for(int i = 0; i < num_trail_per_gen && i < n_neighbor_left; i++) {
        int j = tmpInd[i];
        update_population_DECOM(j, j, nPop_mine, osp_obj, osp_var,
                                weights_all, useSflag, Sflag,
                                rot_angle_offspring,
                                niche, niche_local, tableNeighbor, tableNeighbor_local, maxNneighb, parent_type[j]);
    }
    //
    update_xBest(UPDATE_GIVEN, n_neighbor_left, NULL, osp_var, osp_obj, rot_angle_offspring);
    update_xBest_history(UPDATE_GIVEN, n_neighbor_left, NULL, osp_var, osp_obj, rot_angle_offspring);
    // right
    osp_obj = st_pop_comm_p.obj_right;
    osp_var = st_pop_comm_p.var_right;
    weights_all = st_decomp_p.weights_mine;
    rot_angle_offspring = st_pop_comm_p.rot_angle_right;
    niche = st_decomp_p.niche_neighb;
    tableNeighbor = &st_decomp_p.tableNeighbor[(nPop_mine + n_neighbor_left) * maxNneighb];
    for(int i = 0; i < nPop; i++) {
        if(flip_r((float)th_select)) parent_type[i] = PARENT_LOCAL;
        else parent_type[i] = PARENT_GLOBAL;
    }
    for(int i = 0; i < n_neighbor_right; i++) tmpInd[i] = i;
    shuffle(tmpInd, n_neighbor_right);
    for(int i = 0; i < num_trail_per_gen && i < n_neighbor_right; i++) {
        int j = tmpInd[i];
        update_population_DECOM(j, j, nPop_mine, osp_obj, osp_var,
                                weights_all, useSflag, Sflag,
                                rot_angle_offspring,
                                niche, niche_local, tableNeighbor, tableNeighbor_local, maxNneighb, parent_type[j]);
    }
    //
    update_xBest(UPDATE_GIVEN, n_neighbor_right, NULL, osp_var, osp_obj, rot_angle_offspring);
    update_xBest_history(UPDATE_GIVEN, n_neighbor_right, NULL, osp_var, osp_obj, rot_angle_offspring);
    //
    free(parent_type);
    free(tmpInd);
    //
    return;
}

void update_population_and_weights(double* var_parents, double* obj_parents, int n_parents,
                                   double* var_offsprings, double* obj_offsprings, int n_offsprings,
                                   double* old_weights, int n_old_w, double* new_weights, int n_new_w,
                                   int n_var, int n_obj)
{
    //
    double* var_rep = allocDouble((n_parents + n_offsprings) * n_var);
    double* obj_rep = allocDouble((n_parents + n_offsprings) * n_obj);
    double* weights_comb = allocDouble((n_old_w + n_new_w) * n_obj);
    int* rank_ind = allocInt(n_parents + n_offsprings);
    int* flag_w = allocInt(n_old_w + n_new_w);
    //
    int n_rep = 0;
    int n_wts = 0;
    //
    memcpy(&var_rep[n_rep * n_var], var_parents, n_parents * n_var * sizeof(double));
    memcpy(&obj_rep[n_rep * n_obj], obj_parents, n_parents * n_obj * sizeof(double));
    n_rep += n_parents;
    memcpy(&var_rep[n_rep * n_var], var_offsprings, n_offsprings * n_var * sizeof(double));
    memcpy(&obj_rep[n_rep * n_obj], obj_offsprings, n_offsprings * n_obj * sizeof(double));
    n_rep += n_offsprings;
    //
    memcpy(&weights_comb[n_wts * n_obj], old_weights, n_old_w * n_obj * sizeof(double));
    n_wts += n_old_w;
    memcpy(&weights_comb[n_wts * n_obj], new_weights, n_new_w * n_obj * sizeof(double));
    n_wts += n_new_w;
    //
    int cur_n_pop = 0;
    int cur_n_w = 0;

    //
    free(var_rep);
    free(obj_rep);
    free(weights_comb);
    free(rank_ind);
    free(flag_w);
    //
    return;
}

//int linear_dominate(double* compt1, double* compt2, double* w_comb, int n_w, int n_obj)
//{
//    int i;
//    int tag1 = 0, tag2 = 0, tag0 = 0;
//    for(i = 0; i < n_w; i++) {
//        double* cur_w = &w_comb[i * n_obj];
//        double f1 = fitnessFunction(compt1, cur_w);
//        double f2 = fitnessFunction(compt2, cur_w);
//        if(f1 < f2) {
//            tag1++;
//        } else if(f2 < f1) {
//            tag2++;
//        } else {
//            tag0++;
//        }
//    }
//    //
//    if(tag0 == n_w) {
//        return 0;
//    }
//    if(tag1 + tag0 == n_w) {
//        return 1;
//    }
//    if(tag2 + tag0 == n_w) {
//        return -1;
//    }
//}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// xBest
void update_xBest_history(int update_tag, int nPop, int* vec_indx, double* addrX, double* addrY, double* addrZ)
{
    int color_pop = st_MPI_p.color_pop;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* var_best = st_pop_best_p.var_best;
    double* obj_best = st_pop_best_p.obj_best;
    int* i_best_history = &st_pop_best_p.i_best_history;
    int* cn_best_history = &st_pop_best_p.cn_best_history;
    int  n_best_history = st_pop_best_p.n_best_history;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    double* rot_angle_best = st_qu_p.rot_angle_best;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* var_best_history = st_pop_best_p.var_best_history;
    double* obj_best_history = st_pop_best_p.obj_best_history;
    double* rot_angle_best_history = st_qu_p.rot_angle_best_history;
    //
    int i;
    int best_ind;
    double* best_fit;
    int probIdx;

    probIdx = color_pop - 1;
    if(update_tag != UPDATE_INIT && (probIdx < 0 || probIdx > nObj - 1)) return;

    best_fit = (double*)malloc(nObj * sizeof(double));

    if(update_tag == UPDATE_INIT) {
        for(i = 0; i < nObj; i++) {
            obj_best[i] = INF_DOUBLE;
        }
        (*i_best_history) = 0;
        (*cn_best_history) = 0;
    } else if(update_tag == UPDATE_GIVEN) {
        if(!addrX || !addrY || (Qubits_angle_opt_tag == FLAG_ON && !addrZ)) {
            printf("%s:Calling void update_xBest_history(int update_tag, double* addrX, double* addrY, double* addrZ) error: addresses of addrX and/or addrY and/or addrZ not provided.\n",
                   AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_ADDRESS);
        }
        memcpy(best_fit, obj_best, nObj * sizeof(double));
        best_ind = -1;
        for(i = 0; i < nPop; i++) {
            int ii = i;
            if(vec_indx) ii = vec_indx[i];
            if(addrY[ii * nObj + probIdx] <= best_fit[probIdx] && 2 != dominanceComparator(&addrY[ii * nObj], best_fit)) {
                memcpy(best_fit, &addrY[ii * nObj], nObj * sizeof(double));
                best_ind = ii;
            }
        }
        if(best_ind != -1) {
            memcpy(var_best, &addrX[best_ind * nDim], nDim * sizeof(double));
            memcpy(obj_best, &addrY[best_ind * nObj], nObj * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON)
                memcpy(rot_angle_best, &addrZ[best_ind * nDim], nDim * sizeof(double));
            memcpy(&var_best_history[(*i_best_history) * nDim], var_best, nDim * sizeof(double));
            memcpy(&obj_best_history[(*i_best_history) * nObj], obj_best, nObj * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON)
                memcpy(&rot_angle_best_history[(*i_best_history) * nDim], rot_angle_best, nDim * sizeof(double));
            (*i_best_history) = ((*i_best_history) + 1) % n_best_history;
            if((*cn_best_history) < n_best_history)(*cn_best_history)++;
        }
    } else {
        if(0 == st_MPI_p.mpi_rank)
            printf("%s:Unkown update tag for xBest history, exiting.\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_UPDATE_TAG);
    }

    free(best_fit);
}

void update_xBest(int tag, int nPop, int* vec_indx, double* addrX, double* addrY, double* addrZ)
{
    int color_pop = st_MPI_p.color_pop;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    double* obj_best_subObjs_all = st_pop_best_p.obj_best_subObjs_all;
    double* var_best_subObjs_all = st_pop_best_p.var_best_subObjs_all;
    double* rot_angle_best_subObjs_all = st_qu_p.rot_angle_best_subObjs_all;
    //
    int i;
    int best_idx;
    double* best_fit;
    int probIdx;
    //
    //probIdx = strct_MPI_info.color_population - 1;
    if(tag == UPDATE_INIT) {
        for(probIdx = 0; probIdx < nObj; probIdx++) {
            for(i = 0; i < nObj; i++) {
                obj_best_subObjs_all[probIdx * nObj + i] = INF_DOUBLE;
            }
        }
    } else if(tag == UPDATE_GIVEN) {
        if(!addrX || !addrY || (Qubits_angle_opt_tag == FLAG_ON && !addrZ)) {
            printf("%s:Calling void update_xBest(int tag, double* addrX, double* addrY, double* addrZ) error: addresses of addrX and/or addrY and/or addrZ not provided.\n",
                   AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_ADDRESS);
        }
        for(probIdx = 0; probIdx < nObj; probIdx++) {
            best_fit = (double*)malloc(nObj * sizeof(double));
            memcpy(best_fit, &obj_best_subObjs_all[probIdx * nObj], nObj * sizeof(double));

            best_idx = -1;
            for(i = 0; i < nPop; i++) {
                int ii = i;
                if(vec_indx) ii = vec_indx[i];
                if(addrY[ii * nObj + probIdx] <= best_fit[probIdx] && 2 != dominanceComparator(&addrY[ii * nObj], best_fit)) {
                    memcpy(best_fit, &addrY[ii * nObj], nObj * sizeof(double));
                    best_idx = ii;
                }
            }
            if(best_idx != -1) {
                memcpy(&var_best_subObjs_all[probIdx * nDim], &addrX[best_idx * nDim], nDim * sizeof(double));
                memcpy(&obj_best_subObjs_all[probIdx * nObj], &addrY[best_idx * nObj], nObj * sizeof(double));
                if(Qubits_angle_opt_tag == FLAG_ON)
                    memcpy(&rot_angle_best_subObjs_all[probIdx * nDim], &addrZ[best_idx * nDim], nDim * sizeof(double));
            }
            free(best_fit);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank)
            printf("%s:Unkown update tag for xBest history, exiting.\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_UPDATE_TAG);
    }

    return;
}

void OldSoluUpdate(int iReplaced, int depth, int nPop, double* weights_all, int niche, int* tableNeighbor, int maxNneighb)
{
    int MFI_update_tag = st_ctrl_p.MFI_update_tag;
    //int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* cur_obj = st_pop_evo_cur.obj;
    double* cur_var = st_pop_evo_cur.var;
    //double* weights_all = st_decomp_p.weights_all;//
    int* curSize_inferior = &st_pop_evo_cur.curSize_inferior;
    int* Sflag = st_DE_p.Sflag;
    double* cur_var_inferior = st_pop_evo_cur.var_inferior;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    double* rot_angle_cur_inferior = st_qu_p.rot_angle_cur_inferior;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    //int niche = st_decomp_p.niche;
    //int* tableNeighbor = st_decomp_p.tableNeighbor;//
    //int maxNneighb = nPop; //
    double* fitCur = st_decomp_p.fitCur;
    //
    int i, j;
    if(depth <= 0) return;
    int realInd;
    i = iReplaced;
    for(j = 0; j < niche; j++) {
        realInd = tableNeighbor[i * maxNneighb + j];
        double f1, f2;
        f1 = fitnessFunction(&cur_obj[i * nObj], &weights_all[realInd * nObj]);
        f2 = fitnessFunction(&cur_obj[realInd * nObj], &weights_all[realInd * nObj]);
        fitCur[realInd] = f2;
        if(f1 < f2) {
            OldSoluUpdate(realInd, depth - 1, nPop, weights_all, niche, tableNeighbor, maxNneighb);
            int tmp_ind;
            if((*curSize_inferior) < nPop) {
                tmp_ind = (*curSize_inferior);
                (*curSize_inferior)++;
            } else {
                selectSamples(nPop, -1, -1, -1, &tmp_ind, NULL, NULL, NULL, NULL);
                (*curSize_inferior) = nPop;
            }
            memcpy(&cur_var_inferior[tmp_ind * nDim], &cur_var[realInd * nDim], nDim * sizeof(double));
            memcpy(&cur_var[realInd * nDim], &cur_var[i * nDim], nDim * sizeof(double));
            memcpy(&cur_obj[realInd * nObj], &cur_obj[i * nObj], nObj * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON) {
                memcpy(&rot_angle_cur_inferior[tmp_ind * nDim], &rot_angle_cur[realInd * nDim], nDim * sizeof(double));
                memcpy(&rot_angle_cur[realInd * nDim], &rot_angle_cur[i * nDim], nDim * sizeof(double));
            }
            fitCur[realInd] = f1;
            checkUpdatedIndx(realInd);
            //
            return;
        }
    }
}

void checkUpdatedIndx(int ind)
{
    int* iUpdt = &st_pop_comm_p.iUpdt;
    int* updtIndx = st_pop_comm_p.updtIndx;
    //
    int iii;
    for(iii = 0; iii < (*iUpdt) && updtIndx[iii] != ind; iii++) {}
    if(iii >= (*iUpdt)) {
        updtIndx[(*iUpdt)++] = ind;
    }
    //
    return;
}
