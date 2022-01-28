# include "global.h"
# include <math.h>

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// DECOMPOSITION
void joinVar(int iS, int iD, int var_prop_tag)
{
    int nDim = st_global_p.nDim;
    double ratio_mut = st_optimizer_p.ratio_mut;
    int type_xor_rem_vars = st_ctrl_p.type_xor_rem_vars;
    int type_test = st_ctrl_p.type_test;
    int commonality_xor_remvar_tag = st_ctrl_p.commonality_xor_remvar_tag;
    double* rate_Commonality = st_optimizer_p.rate_Commonality;
    int type_feature_adjust = st_ctrl_p.type_feature_adjust;
    int type_mut_general = st_ctrl_p.type_mut_general;
    int type_del_var = st_ctrl_p.type_del_var;
    //
    double* pCurrent, * pTrail;

    pCurrent = &st_pop_evo_cur.var[iS * nDim];
    pTrail = &st_pop_evo_offspring.var[iD * nDim];

    if(type_xor_rem_vars == XOR_REMVARS_INHERIT ||
       type_xor_rem_vars == XOR_REMVARS_COPY) {
        xor_remVars_inherit_block(iS, iD);
    } else {
        xor_remVars_switch(iS, iD);
    }

    if((type_test == MY_TYPE_FS_CLASSIFY || type_test == MY_TYPE_FS_CLASSIFY_TREE) &&
       commonality_xor_remvar_tag == FLAG_ON) {
        if(rate_Commonality[iD] <= 0) {
            binarymutation_whole_bin_Markov(pTrail, ratio_mut);
        } else {
            realmutation_whole_bin(pTrail, ratio_mut);
        }
        //
        if(type_feature_adjust == FEATURE_ADJUST_FILTER_MARKOV) {
            adjustFeatureNum_Markov(pTrail);
        } else {
            adjustFeatureNum_rand(pTrail);
        }
    } else {
        if(type_mut_general == MUT_GENERAL_POLYNOMIAL)
            realmutation_whole(pTrail, ratio_mut);
        else
            randmutation_whole(pTrail, ratio_mut);
    }

    //adjustFeatureNum_rand(&strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim]);
    //if (strct_global_paras.generation % 2){
    //	adjustFeatureNum_rank_replace(&uTrail[iP * strct_global_paras.nDim]);
    //}
    //else{
    //	adjustFeatureNum_corr_replace(&uTrail[iP * strct_global_paras.nDim]);
    //}
    //adjustFeatureNum_cluster(&strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim]);

    if(type_del_var == DEL_LeNet) {
        LeNet_delete(pTrail);
    }

    //if (var_prop_tag == OPTIMIZE_CONVER_VARS) {
    //    for (i = 0; i < strct_grp_ana_vals.numDiverIndexes; i++) {
    //        int realIdx = strct_grp_info_vals.DiversityIndexs[i];
    //        strct_pop_evo_info.var_offspring[iD * strct_global_paras.nDim + realIdx] = strct_pop_evo_info.var_current[iS * strct_global_paras.nDim + realIdx];
    //    }
    //}

    //if(strct_ctrl_para.type_var_encoding == VAR_BINARY) {
    //    int flag = 0;
    //    for(i = 0; i < strct_global_paras.nDim; i++) {
    //        if(((int)pTrail[i] && !(int)pCurrent[i]) ||
    //           (pTrail[i] <= 0.5 && pCurrent[i] > 0.5)) {
    //            flag = 1;
    //        }
    //    }
    //    if(!flag) {
    //        printf("strct_MPI_info.mpi_rank - %d, identical %d\n", strct_MPI_info.mpi_rank, iD);
    //    }
    //}

    return;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// Pareto Dominance
// 免疫选择克隆 + Pareto
//
void joinVar_allObjs_local_ND(int iP, int jP)
{
    int i;
    int r1, r2;
    double* p1, * p2, * pCurrent, * pTrail;

    float tmp_CR = (float)st_DE_p.CR_cur[jP];

    pCurrent = &st_archive_p.var[st_global_p.selectedIndv[iP] * st_global_p.nDim];
    pTrail = &st_pop_evo_offspring.var[jP * st_global_p.nDim];

    //	selectSamples_RR(strct_global_paras.num_selected,strct_global_paras.selectedProb,iP,&r1,&r2,NULL);
    if(st_global_p.num_selected >= 3) {
        selectSamples(st_global_p.num_selected, iP, -1, -1, &r1, &r2, NULL, NULL, NULL);
        r1 = st_global_p.selectedIndv[r1];
        r2 = st_global_p.selectedIndv[r2];
    } else {
        selectSamples(st_global_p.nPop, st_global_p.selectedIndv[iP], -1, -1, &r1, &r2, NULL, NULL, NULL);
    }

    p1 = &st_archive_p.var[r1 * st_global_p.nDim];
    p2 = &st_archive_p.var[r2 * st_global_p.nDim];

    for(i = 0; i < st_global_p.nDim; i++) {
        if(st_grp_info_p.table_mine_flag[i]) continue;
        if(flip_r(tmp_CR)) {
            pTrail[i] = pCurrent[i];
        } else if(flip_r(0.5)) {
            pTrail[i] = p1[i];
        } else {
            pTrail[i] = p2[i];
        }
    }

    if(st_ctrl_p.type_mut_general == MUT_GENERAL_POLYNOMIAL)
        realmutation_whole(pTrail, st_optimizer_p.ratio_mut);
    else
        randmutation_whole(pTrail, st_optimizer_p.ratio_mut);

    return;
}

void joinVar_allObjs_global_ND(int iP, int jP)
{
    int i;
    int r1, r2;
    double* p1, * p2, * pCurrent, * pTrail;

    float tmp_CR = (float)st_DE_p.CR_cur[jP];

    pCurrent = &st_archive_p.var[iP * st_global_p.nDim];
    pTrail = &st_pop_evo_offspring.var[jP * st_global_p.nDim];

    selectSamples(st_global_p.nPop, iP, -1, -1, &r1, &r2, NULL, NULL, NULL);

    p1 = &st_archive_p.var[r1 * st_global_p.nDim];
    p2 = &st_archive_p.var[r2 * st_global_p.nDim];

    for(i = 0; i < st_global_p.nDim; i++) {
        if(st_grp_info_p.table_mine_flag[i]) continue;
        if(flip_r(tmp_CR)) {
            pTrail[i] = pCurrent[i];
        } else if(flip_r(0.5)) {
            pTrail[i] = p1[i];
        } else {
            pTrail[i] = p2[i];
        }
    }

    if(st_ctrl_p.type_mut_general == MUT_GENERAL_POLYNOMIAL)
        realmutation_whole(pTrail, st_optimizer_p.ratio_mut);
    else
        randmutation_whole(pTrail, st_optimizer_p.ratio_mut);

    return;
}

void joinVar_subObj_ND(int iP, int jP)
{
    int i;
    int r1, r2;
    double* p1, * p2, * pCurrent;
    int depth = 5;

    float tmp_CR = (float)st_DE_p.CR_cur[jP];

    pCurrent = &st_archive_p.var[iP * st_global_p.nDim];

    //selectSamples_RR(strct_global_paras.num_selected,strct_global_paras.selectedProb,iP,&r1,&r2,NULL);
    //selectSamples(nArch_sep, iP, &r1, &r2, NULL);
    tourSelectSamples_sub(st_archive_p.nArch_sub, depth, st_MPI_p.color_pop - 1, st_archive_p.obj, iP, -1, -1, &r1, &r2, NULL);

    p1 = &st_archive_p.var[r1 * st_global_p.nDim];
    p2 = &st_archive_p.var[r2 * st_global_p.nDim];

    for(i = 0; i < st_global_p.nDim; i++) {
        if(st_grp_info_p.table_mine_flag[i]) continue;
        if(flip_r(tmp_CR)) {
            st_pop_evo_offspring.var[jP * st_global_p.nDim + i] = pCurrent[i];
        } else if(flip_r(0.5)) {
            st_pop_evo_offspring.var[jP * st_global_p.nDim + i] = p1[i];
        } else {
            st_pop_evo_offspring.var[jP * st_global_p.nDim + i] = p2[i];
        }
    }

    if(st_ctrl_p.type_mut_general == MUT_GENERAL_POLYNOMIAL)
        realmutation_whole(&st_pop_evo_offspring.var[jP * st_global_p.nDim], st_optimizer_p.ratio_mut);
    else
        randmutation_whole(&st_pop_evo_offspring.var[jP * st_global_p.nDim], st_optimizer_p.ratio_mut);

    return;
}
