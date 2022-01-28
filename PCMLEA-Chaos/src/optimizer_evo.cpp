# include "global.h"
# include <math.h>

#define EPS 1.2e-7

void DE_1_bin(double* pbase, double* p1, double* p2, double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, size_g - 1);
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.types_var_all[real_ind] != VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (p1[real_ind] - p2[real_ind]);
            if((int)child[real_ind]) {
                child[real_ind] = 1;
            } else {
                child[real_ind] = 0;
            }
        } else {
            child[real_ind] = parent[real_ind];
        }
    }

    return;
}

void DE_1_exp(double* pbase, double* p1, double* p2, double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, size_g - 1);
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (p1[real_ind] - p2[real_ind]);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(pbase[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(pbase[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void DE_2_exp(double* pbase, double* p1, double* p2, double* p3, double* p4, double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, size_g - 1);
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (p1[real_ind] - p2[real_ind]) +
                              F_dyn * (p3[real_ind] - p4[real_ind]);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(pbase[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(pbase[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void DE_selected1_1_exp(double* pbase, double* p1, double* p2, double* pb, double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int g_size = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }
    /*	if(strct_utility_info.utility[iP]<strct_utility_info.utility_mid1)
    {
    F_dyn=fabs(gaussrand(0.0,1.0));
    }
    else if(strct_utility_info.utility[iP]<strct_utility_info.utility_mid2)
    {
    F_dyn=fabs(cauchyrand(0.0,1.0));
    }
    else
    {
    F_dyn=fabs(LevyRand(0.1,0.1));
    }*/

    // rand selection
    // 0 -> uniform
    // 1 -> chebyshevMap
    // 2 -> piecewise_linearMap
    // 3 -> sinusMap
    // 4 -> logisticMap
    // 5 -> circleMap
    // 6 -> gaussMap
    // 7 -> tentMap

    //	F_dyn=tentMap();
    //	F_dyn=gaussMap();
    //	F_dyn=piecewise_linearMap();
    //	F_dyn=0.5*LevyRand(0.1,0.1)+0.5*gaussrand();

    k_ind = rnd(0, g_size - 1);
    int real_ind;

    for(k = 0; k < g_size; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            //			F_dyn=0.5*fabs(LevyRand(0.7,0.36))+0.5*gaussMap();
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (pb[real_ind] - pbase[real_ind]) +
                              F_dyn * (p1[real_ind] - p2[real_ind]);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(pbase[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(pbase[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }

        //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 0) && 5 == strct_MPI_info.mpi_rank && iP == 14 && real_ind == 53) {
        //    printf("DE_RAND~%d~%d~(%.16lf~%.16lf)~(%.16lf~%.16lf~%.16lf~%.16lf) (%.16lf==%.16lf~%.16lf) ", iP, real_ind,
        //           child[real_ind], parent[real_ind], pbase[real_ind], pb[real_ind], p1[real_ind], p2[real_ind], F_dyn, strct_apap_DE_para.F_cur[iPara], strct_apap_DE_para.CR);
        //    printf("\n");
        //}
    }
    return;
}

void DE_selected1_2_exp(double* pbase, double* p1, double* p2, double* p3, double* p4, double* pb,
                        double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int g_size = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, g_size - 1);
    int real_ind;

    for(k = 0; k < g_size; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (pb[real_ind] - pbase[real_ind]) +
                              F_dyn * (p1[real_ind] - p2[real_ind]) +
                              F_dyn * (p3[real_ind] - p4[real_ind]);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(pbase[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(pbase[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void DE_selected2_1_exp(double* pbase, double* p1, double* p2, double* pb1, double* pb2, double* parent, double* child, int iP,
                        int iPara)
{
    int k;
    int k_ind;
    int g_size = st_grp_info_p.table_mine_size;

    double F_dyn;
    double CR_dyn;
    F_dyn = st_DE_p.F__cur[iPara];
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            F_dyn *= log(1.0 / tmp_rnd);
        else
            F_dyn *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        F_dyn *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        F_dyn *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        F_dyn *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, g_size - 1);
    int real_ind;

    for(k = 0; k < g_size; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = pbase[real_ind] +
                              F_dyn * (pb1[real_ind] - pbase[real_ind]) +
                              F_dyn * (pb2[real_ind] - pbase[real_ind]) +
                              F_dyn * (p1[real_ind] - p2[real_ind]);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(pbase[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(pbase[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void EA_pure_xor(double* p1, double* p2, double* parent, double* child, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    //double F_dyn;
    double CR_dyn;

    //F_dyn = strct_apap_DE_para.F_cur[iPara];

    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    k_ind = rnd(0, size_g - 1);
    //strct_apap_DE_para.CRall_evo[iP] = strct_apap_DE_para.CR;
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.types_var_all[real_ind] != VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            child[real_ind] = p1[real_ind];
        } else {
            child[real_ind] = parent[real_ind];
        }
    }

    return;
}

void evo_bin_commonality(double* p0, double* p1, double* parent, double* child, int iP)
{
    int k;
    int size_g = st_grp_info_p.table_mine_size;
    int real_ind;

    int n_common = 0;
    int n_feat_p0 = 0;
    int n_feat_p1 = 0;
    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.types_var_all[real_ind] != VAR_BINARY) {
            continue;
        }
        if((int)p0[real_ind] && (int)p1[real_ind]) n_common++;
        if((int)p0[real_ind])                       n_feat_p0++;
        if((int)p1[real_ind])                       n_feat_p1++;
    }

    double rate;
    if((n_feat_p0 + n_feat_p1) == 2 * n_common) {
        rate = -1;
    } else {
        rate = (double)(n_feat_p0 - n_common) / (n_feat_p0 + n_feat_p1 - 2 * n_common);
    }
    st_optimizer_p.rate_Commonality[iP] = rate;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.types_var_all[real_ind] != VAR_BINARY) {
            continue;
        }
        if(((int)p0[real_ind] && (int)p1[real_ind]) ||
           (!(int)p0[real_ind] && !(int)p1[real_ind])) {
            child[real_ind] = p0[real_ind];
        } else if(flip_r((float)st_optimizer_p.rate_Commonality[iP])) {
            if((int)p0[real_ind])
                child[real_ind] = p0[real_ind];
            else
                child[real_ind] = p1[real_ind];
        } else {
            if(!(int)p0[real_ind])
                child[real_ind] = p0[real_ind];
            else
                child[real_ind] = p1[real_ind];
        }
    }

    return;
}

void SBX_classic(double* p1, double* p2, double* parent, double* child, int iP, int iPara)
{
    double rand;
    double y1, y2, yl, yu;
    double c11, c22;
    double alpha, beta, betaq;
    double eta_c = etax;
    int i_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double CR_dyn;
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    i_ind = rnd(0, size_g - 1);
    int real_ind;

    if(1) {
        for(int i = 0; i < size_g; i++) {
            real_ind = st_grp_info_p.table_mine[i];
            if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
               st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
                continue;
            }
            if(flip_r((float)CR_dyn) || i == i_ind) {
                if(fabs(p1[real_ind] - p2[real_ind]) > EPS) {
                    if(p1[real_ind] < p2[real_ind]) {
                        y1 = p1[real_ind];
                        y2 = p2[real_ind];
                    } else {
                        y1 = p2[real_ind];
                        y2 = p1[real_ind];
                    }
                    if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF) {
                        yl = st_global_p.minLimit[real_ind];
                        yu = st_global_p.maxLimit[real_ind];
                    } else {
                        yl = st_qu_p.minLimit_rot_angle[real_ind];
                        yu = st_qu_p.maxLimit_rot_angle[real_ind];
                    }
                    rand = pointer_gen_rand();
                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1));
                    alpha = 2.0 - pow(beta, -(eta_c + 1.0));
                    if(rand <= (1.0 / alpha)) {
                        betaq = pow((rand * alpha), (1.0 / (eta_c + 1.0)));
                    } else {
                        betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (eta_c + 1.0)));
                    }
                    c11 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
                    alpha = 2.0 - pow(beta, -(eta_c + 1.0));
                    if(rand <= (1.0 / alpha)) {
                        betaq = pow((rand * alpha), (1.0 / (eta_c + 1.0)));
                    } else {
                        betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (eta_c + 1.0)));
                    }
                    c22 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
                    boundaryExceedingFixing(y1, c11, yl, yu);
                    boundaryExceedingFixing(y2, c22, yl, yu);
                    if(flip_r((float)0.5)) {
                        child[real_ind] = c22;
                    } else {
                        child[real_ind] = c11;
                    }
                } else {
                    child[real_ind] = parent[real_ind];
                }
            } else {
                child[real_ind] = parent[real_ind];
            }
        }
    } else {
        for(int i = 0; i < size_g; i++) {
            real_ind = st_grp_info_p.table_mine[i];
            if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
               st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
                continue;
            }
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void PSO_classic(double* p1, double* parent, double* child, double* vel, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double CR_dyn;
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    double cur_w = st_PSO_p.w_fixed;
    double cur_c1 = st_PSO_p.c1_fixed;
    double cur_c2 = st_PSO_p.c2_fixed;
    if(st_optimizer_p.PSO_para_types_all[iP] == PSO_PARA_FIXED) {
        cur_w = st_PSO_p.w_fixed;
        cur_c1 = st_PSO_p.c1_fixed;
        cur_c2 = st_PSO_p.c2_fixed;
    } else if(st_optimizer_p.PSO_para_types_all[iP] == PSO_PARA_ADAP) {
        cur_w = st_PSO_p.w__cur[iPara];
        cur_c1 = st_PSO_p.c1_cur[iPara];
        cur_c2 = st_PSO_p.c2_cur[iPara];
    }

    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            cur_w *= log(1.0 / tmp_rnd);
        else
            cur_w *= -log(1.0 / tmp_rnd);
        tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            cur_c1 *= log(1.0 / tmp_rnd);
        else
            cur_c1 *= -log(1.0 / tmp_rnd);
        tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            cur_c2 *= log(1.0 / tmp_rnd);
        else
            cur_c2 *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        cur_w *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
        cur_c1 *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
        cur_c2 *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        cur_w *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
        cur_c1 *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
        cur_c2 *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        cur_w *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
        cur_c1 *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
        cur_c2 *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, size_g - 1);
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            vel[real_ind] = cur_w * vel[real_ind] +
                            cur_c1 * pointer_gen_rand() * (parent[real_ind] - child[real_ind]) +
                            cur_c2 * pointer_gen_rand() * (p1[real_ind] - child[real_ind]);
            if(vel[real_ind] < st_PSO_p.vMin[real_ind])
                vel[real_ind] = st_PSO_p.vMin[real_ind];
            if(vel[real_ind] > st_PSO_p.vMax[real_ind])
                vel[real_ind] = st_PSO_p.vMax[real_ind];
            child[real_ind] = child[real_ind] + vel[real_ind];
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(parent[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(parent[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void QPSO_classic_2(double* p1, double* p_center, double* parent, double* child, double* vel, int iP, int iPara)
{
    int k;
    int k_ind;
    int size_g = st_grp_info_p.table_mine_size;

    double CR_dyn;
    if(st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
        CR_dyn = st_DE_p.CR_evo_cur[iPara];
    else
        CR_dyn = st_DE_p.CR;

    //double cur_w = strct_PSO_para_all.w_fixed_PSO;
    //double cur_c1 = strct_PSO_para_all.c1_fixed_PSO;
    //double cur_c2 = strct_PSO_para_all.c2_fixed_PSO;
    //if(strct_all_optimizer_paras.PSO_para_types_all[iP] == PSO_PARA_FIXED) {
    //    cur_w = strct_PSO_para_all.w_fixed_PSO;
    //    cur_c1 = strct_PSO_para_all.c1_fixed_PSO;
    //    cur_c2 = strct_PSO_para_all.c2_fixed_PSO;
    //} else if(strct_all_optimizer_paras.PSO_para_types_all[iP] == PSO_PARA_ADAP) {
    //    cur_w = strct_PSO_para_all.w_PSO[iPara];
    //    cur_c1 = strct_PSO_para_all.c1_PSO[iPara];
    //    cur_c2 = strct_PSO_para_all.c2_PSO[iPara];
    //}

    //
    double cur_w = st_PSO_p.alpha_begin_Qu + (st_PSO_p.alpha_final_Qu - st_PSO_p.alpha_begin_Qu) *
                   (st_global_p.iter - st_global_p.usedIter_init) / (st_global_p.maxIter - st_global_p.usedIter_init +
                           1.0);

    //
    //
    if(st_ctrl_p.ScalePara_tag == SCALE_QUANTUM) {
        double tmp_rnd = pointer_gen_rand();
        while(tmp_rnd == 0.0) {
            tmp_rnd = pointer_gen_rand();
        }
        if(flip_r((float)0.5))
            cur_w *= log(1.0 / tmp_rnd);
        else
            cur_w *= -log(1.0 / tmp_rnd);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_LEVY) {
        cur_w *= LevyRand(st_ctrl_p.st_scale_para.levy_c, st_ctrl_p.st_scale_para.levy_a);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_CAUCHY) {
        cur_w *= cauchyrand(st_ctrl_p.st_scale_para.cauchy_a, st_ctrl_p.st_scale_para.cauchy_b);
    } else if(st_ctrl_p.ScalePara_tag == SCALE_GAUSS) {
        cur_w *= gaussrand(st_ctrl_p.st_scale_para.gauss_a, st_ctrl_p.st_scale_para.gauss_b);
    }

    k_ind = rnd(0, size_g - 1);
    int real_ind;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        if(st_ctrl_p.opt_binVar_as_realVar_tag == FLAG_OFF &&
           st_ctrl_p.types_var_all[real_ind] == VAR_BINARY) {
            continue;
        }
        if(flip_r((float)CR_dyn) || k == k_ind) {
            double tmp_rnd = pointer_gen_rand();
            double tmp_val = tmp_rnd * parent[real_ind] + (1.0 - tmp_rnd) * p1[real_ind];
            //double tmp_dif = child[real_ind] - p_center[real_ind];
            double tmp_dif = child[real_ind] - tmp_val;
            child[real_ind] = tmp_val + cur_w * fabs(tmp_dif);
            if(st_ctrl_p.Qubits_angle_opt_tag == FLAG_OFF)
                boundaryExceedingFixing(parent[real_ind], child[real_ind], st_global_p.minLimit[real_ind],
                                        st_global_p.maxLimit[real_ind]);
            else
                boundaryExceedingFixing(parent[real_ind], child[real_ind],
                                        st_qu_p.minLimit_rot_angle[real_ind], st_qu_p.maxLimit_rot_angle[real_ind]);
        } else {
            child[real_ind] = parent[real_ind];
        }
    }
    return;
}

void Quantum_transform_update(double* parent, double* child, double* rot, int iP)
{
    int k;
    int size_g = st_grp_info_p.table_mine_size;

    int real_ind;

    double old_cos;
    double old_sin;
    double new_cos;
    //double new_sin;
    double rot_cos;
    double rot_sin;

    for(k = 0; k < size_g; k++) {
        real_ind = st_grp_info_p.table_mine[k];
        old_cos = (2.0 * parent[real_ind] - st_global_p.minLimit[real_ind] - st_global_p.maxLimit[real_ind]) /
                  (st_global_p.maxLimit[real_ind] - st_global_p.minLimit[real_ind]);
        old_sin = sqrt(1.0 - old_cos * old_cos);
        rot_cos = cos(rot[real_ind]);
        rot_sin = sin(rot[real_ind]);
        new_cos = rot_cos * old_cos - rot_sin * old_sin;
        //new_sin = rot_sin * old_cos + rot_cos * old_sin;
        child[real_ind] = 0.5 * (st_global_p.maxLimit[real_ind] * (1.0 + new_cos) +
                                 st_global_p.minLimit[real_ind] * (1.0 - new_cos));
        //boundaryExceedingFixing(parent[real_ind], child[real_ind], strct_global_paras.minLimit[real_ind], strct_global_paras.maxLimit[real_ind]);
    }
    return;
}
