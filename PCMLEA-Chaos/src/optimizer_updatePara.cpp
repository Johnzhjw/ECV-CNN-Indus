# include "global.h"
# include <math.h>

#define EPS 1.2e-7

void generate_para_all()
{
    if(st_MPI_p.color_master_subPop) {
        switch(st_ctrl_p.optimizer_type) {
        case EC_MIX_DE_R_SBX_R:
        case EC_MIX_DE_C_SBX_R:
        case EC_MIX_DE_C_SBX_C:
        case EC_MIX_SBX_C_R:
        case EC_SI_MIX_DE_C_PSO:
        case EC_MIX_DE_C_R:
        case EC_MIX_DE_R_1_2:
        case EC_MIX_DE_C_1_2:
        case OPTIMIZER_BLEND:
        case OPTIMIZER_ENSEMBLE:
            generate_optimizer_types();
            generate_F_current();
            generate_CR_current();
            generate_para_PSO();
            generate_CRall_evo();
            break;
        case EC_DE_CUR_1:
        case EC_DE_CUR_2:
        case EC_DE_RAND_1:
        case EC_DE_RAND_2:
        case EC_DE_ARCHIVE:
        case EC_DE_ARCHIVE_RAND:
        case EC_DE_2SELECTED:
            generate_F_current();
            generate_CR_current();
            generate_CRall_evo();
            break;
        case EC_SBX_CUR:
        case EC_SBX_RAND:
            generate_CR_current();
            generate_CRall_evo();
            break;
        case SI_PSO:
        case SI_QPSO:
            generate_para_PSO();
            generate_CR_current();
            generate_CRall_evo();
            break;
        default:
            printf("%s: Unknown strct_ctrl_para.optimizer_type, exiting...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_OPTIMIZER_TYPE);
            break;
        }
        generate_t_best();
    }
    return;
}

void update_para_statistics()
{
    if(st_MPI_p.color_master_subPop) {
        switch(st_ctrl_p.optimizer_type) {
        case EC_MIX_DE_R_SBX_R:
        case EC_MIX_DE_C_SBX_R:
        case EC_MIX_DE_C_SBX_C:
        case EC_MIX_SBX_C_R:
        case EC_SI_MIX_DE_C_PSO:
        case EC_MIX_DE_C_R:
        case EC_MIX_DE_R_1_2:
        case EC_MIX_DE_C_1_2:
        case OPTIMIZER_BLEND:
        case OPTIMIZER_ENSEMBLE:
            update_optimizer_prob();
            //update_F_CR_hist_SHADE_simple();
            update_F_CR_hist_SHADE();
            update_CR_mu_JADE();
            update_F_mu_JADE();
            update_para_mu_PSO();
            update_CR_mu_evo();
            update_F_disc_prob();
            update_CR_disc_prob();
            update_F_p_SaNSDE();
            break;
        case EC_DE_CUR_1:
        case EC_DE_CUR_2:
        case EC_DE_RAND_1:
        case EC_DE_RAND_2:
        case EC_DE_ARCHIVE:
        case EC_DE_ARCHIVE_RAND:
        case EC_DE_2SELECTED:
            //update_F_CR_hist_SHADE_simple();
            update_F_CR_hist_SHADE();
            update_CR_mu_JADE();
            update_F_mu_JADE();
            update_CR_mu_evo();
            update_F_disc_prob();
            update_CR_disc_prob();
            update_F_p_SaNSDE();
            break;
        case EC_SBX_CUR:
        case EC_SBX_RAND:
            update_CR_mu_JADE();
            update_CR_mu_evo();
            break;
        case SI_PSO:
        case SI_QPSO:
            update_para_mu_PSO();
            update_CR_mu_JADE();
            update_CR_mu_evo();
            break;
        default:
            printf("%s: Unknown strct_ctrl_para.optimizer_type, exiting...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_OPTIMIZER_TYPE);
            break;
        }
        update_t_best_mu();
    }
    return;
}

void generate_optimizer_types()
{
    switch(st_ctrl_p.optimizer_type) {
    case EC_MIX_DE_R_SBX_R:
    case EC_MIX_DE_C_SBX_R:
    case EC_MIX_DE_C_SBX_C:
    case EC_MIX_SBX_C_R:
    case EC_SI_MIX_DE_C_PSO:
    case EC_MIX_DE_C_R:
    case EC_MIX_DE_C_1_2:
    case EC_MIX_DE_R_1_2:
        for(int i = 0; i < st_global_p.nPop; i++) {
            if(flip_r((float)st_optimizer_p.slctProb_opt_1)) {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[0];
            } else {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[1];
            }
        }
        break;
    case OPTIMIZER_BLEND:
        for(int i = 0; i < st_global_p.nPop; i++) {
            if((st_global_p.iter - st_global_p.usedIter_init) <
               0.5 * (st_global_p.maxIter - st_global_p.usedIter_init)) {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[0];
            } else {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[1];
            }
        }
        break;
    case OPTIMIZER_ENSEMBLE:
        float tmp_rate;
        if(st_optimizer_p.optimizer_prob[0] + st_optimizer_p.optimizer_prob[1] == 0)
            tmp_rate = 0.5;
        else
            tmp_rate = (float)(st_optimizer_p.optimizer_prob[0] / (st_optimizer_p.optimizer_prob[0] +
                               st_optimizer_p.optimizer_prob[1]));
        for(int i = 0; i < st_global_p.nPop; i++) {
            if(flip_r((float)tmp_rate)) {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[0];
            } else {
                st_optimizer_p.optimizer_types_all[i] = st_optimizer_p.optimizer_candid[1];
            }
        }
        break;
    default:
        printf("%s: Unknown strct_ctrl_para.optimizer_type, exiting...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_OPTIMIZER_TYPE);
        break;
    }

    return;
}

void update_optimizer_prob()
{
    switch(st_ctrl_p.optimizer_type) {
    case EC_MIX_SBX_C_R:
    case EC_MIX_DE_R_SBX_R:
    case EC_MIX_DE_R_1_2:
    case EC_MIX_DE_C_SBX_R:
    case EC_MIX_DE_C_SBX_C:
    case EC_SI_MIX_DE_C_PSO:
    case EC_MIX_DE_C_R:
    case EC_MIX_DE_C_1_2:
        if(st_optimizer_p.nGen_accum_ada_opti >= st_optimizer_p.nGen_th_accum_ada_opti) {
            double tmp1 = st_optimizer_p.ns_optimizer_1 * (st_optimizer_p.ns_optimizer_2 +
                          st_optimizer_p.nf_optimizer_2) + 0.01;
            double tmp2 = st_optimizer_p.ns_optimizer_2 * (st_optimizer_p.ns_optimizer_1 +
                          st_optimizer_p.nf_optimizer_1) + 0.01;
            st_optimizer_p.slctProb_opt_1 = tmp1 / (tmp1 + tmp2);
            st_optimizer_p.ns_optimizer_1 = 0;
            st_optimizer_p.nf_optimizer_1 = 0;
            st_optimizer_p.ns_optimizer_2 = 0;
            st_optimizer_p.nf_optimizer_2 = 0;
            st_optimizer_p.nGen_accum_ada_opti = 0;
        } else {
            for(int i = 0; i < st_global_p.nPop; i++) {
                if(st_optimizer_p.optimizer_types_all[i] == st_optimizer_p.optimizer_candid[0]) {
                    if(st_DE_p.Sflag[i])
                        st_optimizer_p.ns_optimizer_1++;
                    else
                        st_optimizer_p.nf_optimizer_1++;
                } else if(st_optimizer_p.optimizer_types_all[i] == st_optimizer_p.optimizer_candid[1]) {
                    if(st_DE_p.Sflag[i])
                        st_optimizer_p.ns_optimizer_2++;
                    else
                        st_optimizer_p.nf_optimizer_2++;
                } else {
                    printf("%s: EC_MIX_DE_SBX error - Unknown optimizer type, not EC_DE or EC_SBX/EC_SBX_CUR, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_OPTIMIZER_TYPE);
                }
            }
            st_optimizer_p.nGen_accum_ada_opti++;
        }
        break;
    case OPTIMIZER_BLEND:
        break;
    case OPTIMIZER_ENSEMBLE:
        int count;
        count = 0;
        int i;
        for(i = 0; i < st_global_p.nPop; i++)
            if(st_DE_p.Sflag[i])
                count++;

        if(count) {
            double c = 0.1;
            double m = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            for(i = 0; i < st_global_p.nPop; i++) {
                if(st_DE_p.Sflag[i]) {
                    if(st_optimizer_p.optimizer_types_all[i] == st_optimizer_p.optimizer_candid[0])
                        sum1++;
                    sum2++;
                }
            }
            m = sum1 / sum2;
            st_optimizer_p.optimizer_prob[0] = (1.0 - c) * st_optimizer_p.optimizer_prob[0] + c * m;
            st_optimizer_p.optimizer_prob[1] = (1.0 - c) * st_optimizer_p.optimizer_prob[1] + c * (1 - m);
            if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
                //printf("strct_apap_DE_para.CR_mu = %lf\n", strct_apap_DE_para.CR_mu);
            }
        } else {
            if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
                //printf("strct_apap_DE_para.CR_mu unchanged.\n");
            }
        }
        break;
    default:
        printf("%s: Unknown strct_ctrl_para.optimizer_type, exiting...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_OPTIMIZER_TYPE);
        break;
    }

    return;
}

void generate_F_current()
{
    //static int my_count = 1;
    //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 0) && 5 == strct_MPI_info.mpi_rank)
    //    printf("====================%ld\n", get_rnd_uni_init());
    //my_count++;
    int iP;
    for(iP = 0; iP < st_global_p.nPop; iP++) {
        switch(st_optimizer_p.DE_F_types_all[iP]) {
        case DE_F_SHADE:
            if(st_optimizer_p.DE_CR_types_all[iP] != DE_CR_SHADE) {
                printf("%s: For SHADE, F and CR should be the same, all with SHADE, exiting...\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_F_CR_SHADE);
            }
            break;
        case DE_F_NSDE:
            if(flip_r((float)0.5)) {
                do {
                    st_DE_p.F__cur[iP] = gaussrand(0.5, 0.5);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            } else {
                do {
                    st_DE_p.F__cur[iP] = cauchyrand(0.0, 1.0);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            }
            if(st_DE_p.F__cur[iP] > 2.0) st_DE_p.F__cur[iP] = 2.0;
            break;
        case DE_F_SaNSDE:
            if(flip_r((float)st_DE_p.slctProb_SaNSDE_F)) {
                st_DE_p.tag_SaNSDE_F[iP] = 1;
                do {
                    st_DE_p.F__cur[iP] = gaussrand(0.5, 0.3);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            } else {
                st_DE_p.tag_SaNSDE_F[iP] = 2;
                do {
                    st_DE_p.F__cur[iP] = cauchyrand(0.0, 1.0);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            }
            if(st_DE_p.F__cur[iP] >= 2.0) st_DE_p.F__cur[iP] = 2.0;
            break;
        case DE_F_SaNSDE_a:
            if(flip_r((float)st_DE_p.slctProb_SaNSDE_F)) {
                st_DE_p.tag_SaNSDE_F[iP] = 1;
                do {
                    st_DE_p.F__cur[iP] = gaussrand(st_DE_p.F_mu, 0.3);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            } else {
                st_DE_p.tag_SaNSDE_F[iP] = 2;
                do {
                    st_DE_p.F__cur[iP] = cauchyrand(st_DE_p.F_mu, 0.1);
                } while(st_DE_p.F__cur[iP] <= 0.0);
            }
            if(st_DE_p.F__cur[iP] >= 1.0) st_DE_p.F__cur[iP] = 1.0;
            break;
        case DE_F_JADE:
            do {
                st_DE_p.F__cur[iP] = cauchyrand(st_DE_p.F_mu, 0.1);
            } while(st_DE_p.F__cur[iP] <= 0.0);
            if(st_ctrl_p.F_para_limit_tag == FLAG_ON) {
                if(st_DE_p.F__cur[iP] > 1.0) st_DE_p.F__cur[iP] = 1.0;
            } else {
                if(st_DE_p.F__cur[iP] > 2.0) st_DE_p.F__cur[iP] = 1.0;
            }
            break;
        case DE_F_jDE:
            if(flip_r((float)0.1)) st_DE_p.F__cur[iP] = rndreal(0.1, 1.0);
            break;
        case DE_F_FIXED:
            st_DE_p.F__cur[iP] = st_DE_p.F;
            break;
        default:
            st_DE_p.F__cur[iP] = st_DE_p.F;
            break;
        }
    }
    //
    return;
}

void generate_CR_current()
{
    int iP;
    for(iP = 0; iP < st_global_p.nPop; iP++) {
        switch(st_optimizer_p.DE_CR_types_all[iP]) {
        case DE_CR_SHADE:
            if(st_optimizer_p.DE_F_types_all[iP] != DE_F_SHADE) {
                printf("%s: For SHADE, F and CR should be the same, all with SHADE, exiting...\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_F_CR_SHADE);
            }
            int tmp;
            selectSamples(st_DE_p.nHistSHADE, -1, -1, -1, &tmp, NULL, NULL, NULL, NULL);
            double tmpF_mu;
            double tmpCR_mu;
            tmpF_mu = st_DE_p.F_hist[tmp];
            tmpCR_mu = st_DE_p.CR_hist[tmp];
            do {
                st_DE_p.F__cur[iP] = cauchyrand(tmpF_mu, 0.1);
            } while(st_DE_p.F__cur[iP] <= 0);
            if(st_DE_p.F__cur[iP] > 1.0) st_DE_p.F__cur[iP] = 1.0;
            st_DE_p.CR_cur[iP] = gaussrand(tmpCR_mu, 0.1);
            if(st_DE_p.CR_cur[iP] < 0.0) st_DE_p.CR_cur[iP] = 0.0;
            if(st_DE_p.CR_cur[iP] > 1.0) st_DE_p.CR_cur[iP] = 1.0;
            break;
        case DE_CR_NSDE:
            st_DE_p.CR_cur[iP] = st_DE_p.CR_rem;
            break;
        case DE_CR_SaNSDE:
            st_DE_p.CR_cur[iP] = gaussrand(st_DE_p.CR_mu, 0.1);
            if(st_DE_p.CR_cur[iP] < 0.0) st_DE_p.CR_cur[iP] = 0.0;
            if(st_DE_p.CR_cur[iP] > 1.0) st_DE_p.CR_cur[iP] = 1.0;
            break;
        case DE_CR_JADE:
            st_DE_p.CR_cur[iP] = gaussrand(st_DE_p.CR_mu, 0.1);
            if(st_DE_p.CR_cur[iP] < 0.0) st_DE_p.CR_cur[iP] = 0.0;
            if(st_DE_p.CR_cur[iP] > 1.0) st_DE_p.CR_cur[iP] = 1.0;
            break;
        case DE_CR_jDE:
            if(flip_r((float)0.1)) st_DE_p.CR_cur[iP] = rndreal(0.0, 1.0);
            break;
        case DE_CR_FIXED:
            st_DE_p.CR_cur[iP] = st_DE_p.CR_rem;
            break;
        case DE_CR_LINEAR:
            st_DE_p.CR_cur[iP] = (float)(st_global_p.iter - st_global_p.usedIter_init) /
                                 (st_global_p.maxIter - st_global_p.usedIter_init);
            break;
        default:
            st_DE_p.CR_cur[iP] = gaussrand(st_DE_p.CR_mu, 0.1);
            if(st_DE_p.CR_cur[iP] < 0.0) st_DE_p.CR_cur[iP] = 0.0;
            if(st_DE_p.CR_cur[iP] > 1.0) st_DE_p.CR_cur[iP] = 1.0;
            break;
        }
    }
    return;
}

void update_F_mu_JADE()
{
    int count = 0;
    int i;
    for(i = 0; i < st_global_p.nPop; i++)
        if(st_DE_p.Sflag[i] &&
           (st_optimizer_p.DE_F_types_all[i] == DE_F_JADE ||
            st_optimizer_p.DE_F_types_all[i] == DE_F_SaNSDE_a))
            count++;

    if(count) {
        double c = 0.1;
        double m = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;

        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i] &&
               (st_optimizer_p.DE_F_types_all[i] == DE_F_JADE ||
                st_optimizer_p.DE_F_types_all[i] == DE_F_SaNSDE_a)) {
                sum1 += st_DE_p.F__cur[i] * st_DE_p.F__cur[i];
                sum2 += st_DE_p.F__cur[i];
            }
        }
        //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 2) && 3 == strct_MPI_info.mpi_rank) {
        //    fprintf(strct_global_paras.debugFpt, "sum1 with Addr %d gen -> %d.2.101 rank -> %d\n", &sum1, strct_global_paras.generation, strct_MPI_info.mpi_rank);
        //    save_double(strct_global_paras.debugFpt, &sum1, 1, 1, 1);
        //    fprintf(strct_global_paras.debugFpt, "sum2 with Addr %d gen -> %d.2.101 rank -> %d\n", &sum2, strct_global_paras.generation, strct_MPI_info.mpi_rank);
        //    save_double(strct_global_paras.debugFpt, &sum2, 1, 1, 1);
        //}
        m = sum1 / sum2;
        //double m_tmp = m;
        //if(fabs(m_tmp - m) > 0.0)
        //    printf("%e", m_tmp - m);
        st_DE_p.F_mu = (1.0 - c) * st_DE_p.F_mu + c * m;
        //if((strct_global_paras.generation >= 0 && strct_global_paras.generation <= 2) && 3 == strct_MPI_info.mpi_rank) {
        //    fprintf(strct_global_paras.debugFpt, "m with Addr %d gen -> %d.2.101 rank -> %d\n", &m, strct_global_paras.generation, strct_MPI_info.mpi_rank);
        //    save_double(strct_global_paras.debugFpt, &m, 1, 1, 1);
        //}
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("F_mu = %lf\n", F_mu);
        }
        if(isnan(st_DE_p.F_mu)) {
            printf("%s: update_F_mu_JADE error - isnan, maybe divided by ZERO, exiting...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NAN);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("F_mu unchanged.\n");
        }
    }

    return;
}

void update_CR_mu_JADE()
{
    int count = 0;
    int i;
    for(i = 0; i < st_global_p.nPop; i++)
        if(st_DE_p.Sflag[i])
            count++;

    if(count) {
        double c = 0.1;
        double m = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i]) {
                sum1 += st_DE_p.CR_cur[i];
                sum2++;
            }
        }
        m = sum1 / sum2;
        st_DE_p.CR_mu = (1.0 - c) * st_DE_p.CR_mu + c * m;
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu = %lf\n", strct_apap_DE_para.CR_mu);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu unchanged.\n");
        }
    }

    return;
}

void update_CR_mu_evo()
{
    int count = 0;
    int i;
    for(i = 0; i < st_global_p.nPop; i++)
        if(st_DE_p.Sflag[i] &&
           st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP)
            count++;

    if(count) {
        double c = 0.1;
        double m = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i] &&
               st_ctrl_p.type_xor_evo_mut == XOR_EVO_MUT_ADAP) {
                sum1 += st_DE_p.CR_evo_cur[i];
                sum2++;
            }
        }
        m = sum1 / sum2;
        st_DE_p.CR_evo_mu = (1.0 - c) * st_DE_p.CR_evo_mu + c * m;
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu_evo = %lf\n", strct_apap_DE_para.CR_mu_evo);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu_evo unchanged.\n");
        }
    }

    return;
}

void update_t_best_mu()
{
    int count = 0;
    int i;
    for(i = 0; i < st_global_p.nPop; i++)
        if(st_DE_p.Sflag[i])
            count++;

    if(count) {
        double c = 0.1;
        double sum1_1 = 0.0;
        double sum1_2 = 0.0;
        double sum2 = 0.0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i]) {
                sum1_1 += st_DE_p.t1_best[i];
                sum1_2 += st_DE_p.t2_best[i];
                sum2++;
            }
        }
        st_DE_p.t1_best_mu = (1.0 - c) * st_DE_p.t1_best_mu + c * sum1_1 / sum2;
        st_DE_p.t2_best_mu = (1.0 - c) * st_DE_p.t2_best_mu + c * sum1_2 / sum2;
        //
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu_evo = %lf\n", strct_apap_DE_para.CR_mu_evo);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("strct_apap_DE_para.CR_mu_evo unchanged.\n");
        }
    }

    return;
}

void generate_CRall_evo()
{
    int i;
    for(i = 0; i < st_global_p.nPop; i++) {
        st_DE_p.CR_evo_cur[i] = gaussrand(st_DE_p.CR_evo_mu, 0.1);
        if(st_DE_p.CR_evo_cur[i] < 0.0) st_DE_p.CR_evo_cur[i] = 0.0;
        if(st_DE_p.CR_evo_cur[i] > 1.0) st_DE_p.CR_evo_cur[i] = 1.0;
    }

    return;
}

void generate_t_best()
{
    int i;
    for(i = 0; i < st_global_p.nPop; i++) {
        st_DE_p.t1_best[i] = gaussrand(st_DE_p.t1_best_mu, 0.1);
        if(st_DE_p.t1_best[i] < 0.0) st_DE_p.t1_best[i] = 0.0;
        if(st_DE_p.t1_best[i] > 1.0) st_DE_p.t1_best[i] = 1.0;
        st_DE_p.t2_best[i] = gaussrand(st_DE_p.t2_best_mu, 0.1);
        if(st_DE_p.t2_best[i] < 0.0) st_DE_p.t2_best[i] = 0.0;
        if(st_DE_p.t2_best[i] > 1.0) st_DE_p.t2_best[i] = 1.0;
    }

    return;
}

void generate_para_PSO()
{
    int i;
    for(i = 0; i < st_global_p.nPop; i++) {
        do {
            st_PSO_p.w__cur[i] = cauchyrand(st_PSO_p.w_mu, 0.1);
        } while(st_PSO_p.w__cur[i] <= st_PSO_p.w_min);
        if(st_PSO_p.w__cur[i] > st_PSO_p.w_max) st_PSO_p.w__cur[i] = st_PSO_p.w_max;
    }
    for(i = 0; i < st_global_p.nPop; i++) {
        do {
            st_PSO_p.c1_cur[i] = cauchyrand(st_PSO_p.c1_mu, 0.1);
        } while(st_PSO_p.c1_cur[i] <= st_PSO_p.c1_min);
        if(st_PSO_p.c1_cur[i] > st_PSO_p.c1_max) st_PSO_p.c1_cur[i] = st_PSO_p.c1_max;
    }
    for(i = 0; i < st_global_p.nPop; i++) {
        do {
            st_PSO_p.c2_cur[i] = cauchyrand(st_PSO_p.c2_mu, 0.1);
        } while(st_PSO_p.c2_cur[i] <= st_PSO_p.c2_min);
        if(st_PSO_p.c2_cur[i] > st_PSO_p.c2_max) st_PSO_p.c2_cur[i] = st_PSO_p.c2_max;
    }

    return;
}

void update_para_mu_PSO()
{
    int count = 0;
    int i;
    for(i = 0; i < st_global_p.nPop; i++)
        if(st_DE_p.Sflag[i] &&
           (st_optimizer_p.optimizer_types_all[i] == SI_PSO ||
            st_optimizer_p.optimizer_types_all[i] == SI_QPSO) &&
           st_optimizer_p.PSO_para_types_all[i] == PSO_PARA_ADAP)
            count++;

    if(count) {
        double c = 0.1;
        double m = 0.0;
        double sum1_w = 0.0;
        double sum1_c1 = 0.0;
        double sum1_c2 = 0.0;
        double sum2_w = 0.0;
        double sum2_c1 = 0.0;
        double sum2_c2 = 0.0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i] &&
               (st_optimizer_p.optimizer_types_all[i] == SI_PSO ||
                st_optimizer_p.optimizer_types_all[i] == SI_QPSO) &&
               st_optimizer_p.PSO_para_types_all[i] == PSO_PARA_ADAP) {
                sum1_w += st_PSO_p.w__cur[i] * st_PSO_p.w__cur[i];
                sum2_w += st_PSO_p.w__cur[i];
                sum1_c1 += st_PSO_p.c1_cur[i] * st_PSO_p.c1_cur[i];
                sum2_c1 += st_PSO_p.c1_cur[i];
                sum1_c2 += st_PSO_p.c2_cur[i] * st_PSO_p.c2_cur[i];
                sum2_c2 += st_PSO_p.c2_cur[i];
            }
        }
        m = sum1_w / sum2_w;
        st_PSO_p.w_mu = (1.0 - c) * st_PSO_p.w_mu + c * m;
        m = sum1_c1 / sum2_c1;
        st_PSO_p.c1_mu = (1.0 - c) * st_PSO_p.c1_mu + c * m;
        m = sum1_c2 / sum2_c2;
        st_PSO_p.c2_mu = (1.0 - c) * st_PSO_p.c2_mu + c * m;
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("F_mu = %lf\n", F_mu);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank_master_subPop_globalScope) {
            //printf("F_mu unchanged.\n");
        }
    }

    return;
}

void generate_F_disc()
{
    int i, j;
    double tmp;
    for(i = 0; i < st_global_p.nPop; i++) {
        tmp = pointer_gen_rand();
        j = 0;
        while(st_DE_p.prob_F[j] < tmp && j < st_DE_p.candid_num) {
            tmp -= st_DE_p.prob_F[j];
            j++;
        }
        do {
            st_DE_p.F__cur[i] = cauchyrand(st_DE_p.candid_F[j], 0.1);
        } while(st_DE_p.F__cur[i] <= 0);
        if(st_DE_p.F__cur[i] > 1.0) st_DE_p.F__cur[i] = 1.0;
        //strct_apap_DE_para.disc_F[i] = strct_apap_DE_para.candid_F[j];
        st_DE_p.indx_disc_F[i] = j;
    }

    return;
}

void generate_CR_disc()
{
    int i, j;
    double tmp;
    for(i = 0; i < st_global_p.nPop; i++) {
        tmp = pointer_gen_rand();
        j = 0;
        while(st_DE_p.prob_CR[j] < tmp && j < st_DE_p.candid_num) {
            tmp -= st_DE_p.prob_CR[j];
            j++;
        }
        st_DE_p.CR_cur[i] = gaussrand(st_DE_p.candid_CR[j], 0.1);
        if(st_DE_p.CR_cur[i] < 0.0) st_DE_p.CR_cur[i] = 0.0;
        if(st_DE_p.CR_cur[i] > 1.0) st_DE_p.CR_cur[i] = 1.0;
        //strct_apap_DE_para.disc_CR[i] = strct_apap_DE_para.candid_CR[j];
        st_DE_p.indx_disc_CR[i] = j;
    }

    return;
}

void update_F_disc_prob()
{
    int i, j;
    double inc = 0.01;
    if(st_global_p.generation % 100 == 0) {
        for(i = 0; i < st_DE_p.candid_num; i++) {
            st_DE_p.prob_F[i] = 1.0 / st_DE_p.candid_num;
        }
    } else {
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i] &&
               (st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE_RAND ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_2SELECTED) &&
               st_optimizer_p.DE_F_types_all[i] == DE_F_DISC) {
                j = st_DE_p.indx_disc_F[i];
                st_DE_p.prob_F[j] += inc;
            }
        }
    }

    double sum = 0.0;
    for(i = 0; i < st_DE_p.candid_num; i++) {
        sum += st_DE_p.prob_F[i];
    }
    for(i = 0; i < st_DE_p.candid_num; i++) {
        st_DE_p.prob_F[i] /= sum;
    }
}

void update_CR_disc_prob()
{
    int i, j;
    double inc = 0.01;
    if(st_global_p.generation % 100 == 0) {
        for(i = 0; i < st_DE_p.candid_num; i++) {
            st_DE_p.prob_CR[i] = 1.0 / st_DE_p.candid_num;
        }
    } else {
        for(i = 0; i < st_global_p.nPop; i++) {
            if(st_DE_p.Sflag[i] &&
               (st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE_RAND ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_2SELECTED) &&
               st_optimizer_p.DE_CR_types_all[i] == DE_CR_DISC) {
                j = st_DE_p.indx_disc_CR[i];
                st_DE_p.prob_CR[j] += inc;
            }
        }
    }

    double sum = 0.0;
    for(i = 0; i < st_DE_p.candid_num; i++) {
        sum += st_DE_p.prob_CR[i];
    }
    for(i = 0; i < st_DE_p.candid_num; i++) {
        st_DE_p.prob_CR[i] /= sum;
    }
}

void update_F_CR_hist_SHADE_simple()
{
    int i;
    for(i = 0; i < st_global_p.nPop; i++) {
        if(st_DE_p.Sflag[i] &&
           (st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_1 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_2 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_1 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_2 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE_RAND ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_2SELECTED)) { //
            //int tmp = i * strct_apap_DE_para.H4Ind + strct_apap_DE_para.i_F_CR_hist[i];
            //strct_apap_DE_para.F_hist[tmp] = strct_apap_DE_para.Fall_SHADE[i];
            //strct_apap_DE_para.CR_hist[tmp] = strct_apap_DE_para.CRall_SHADE[i];
            //strct_apap_DE_para.i_F_CR_hist[i] = (strct_apap_DE_para.i_F_CR_hist[i] + 1) % strct_apap_DE_para.H4Ind;
            st_DE_p.F_hist[st_DE_p.iHistSHADE] = st_DE_p.F__cur[i];
            st_DE_p.CR_hist[st_DE_p.iHistSHADE] = st_DE_p.CR_cur[i];
            st_DE_p.iHistSHADE = (st_DE_p.iHistSHADE + 1) % st_DE_p.nHistSHADE;
        }
        //for(int j = 0; j < strct_apap_DE_para.H4Ind; j++) {
        //    printf("(%lf, %lf)\t", strct_apap_DE_para.F_hist[i * strct_apap_DE_para.H4Ind + j], strct_apap_DE_para.CR_hist[i * strct_apap_DE_para.H4Ind + j]);
        //}
    }
    //printf("\n");

    return;
}

void update_F_CR_hist_SHADE()
{
    double* w_fit = (double*)calloc(st_global_p.nPop, sizeof(double));
    double w_sum = 0.0;
    int count = 0;
    int i;
    double tmp_F_mu1 = 0.0;
    double tmp_F_mu2 = 0.0;
    double tmp_CR_mu = 0.0;
    for(i = 0; i < st_global_p.nPop; i++) {
        if(st_decomp_p.countFitImprove[i] &&
           (st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_1 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_2 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_1 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_2 ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE_RAND ||
            st_optimizer_p.optimizer_types_all[i] == EC_DE_2SELECTED)) {  //
            w_fit[i] = 1.0;// strct_decomp_paras.fitImprove[i] / strct_decomp_paras.countFitImprove[i];
            tmp_CR_mu += w_fit[i] * st_DE_p.CR_cur[i];
            tmp_F_mu1 += w_fit[i] * st_DE_p.F__cur[i] * st_DE_p.F__cur[i];
            tmp_F_mu2 += w_fit[i] * st_DE_p.F__cur[i];
            count++;
        } else {
            w_fit[i] = 0.0;
        }
        w_sum += w_fit[i];
    }

    if(count) {
        st_DE_p.F_hist[st_DE_p.iHistSHADE] = tmp_F_mu1 / tmp_F_mu2;
        st_DE_p.CR_hist[st_DE_p.iHistSHADE] = tmp_CR_mu / w_sum;
        //printf("(%lf, %lf, %lf, %lf)\t", tmp_F_mu1, w_sum, strct_apap_DE_para.F_hist[strct_apap_DE_para.iHistSHADE], strct_apap_DE_para.CR_hist[strct_apap_DE_para.iHistSHADE]);
        st_DE_p.iHistSHADE = (st_DE_p.iHistSHADE + 1) % st_DE_p.nHistSHADE;
    }

    free(w_fit);
    return;
}

void update_F_p_SaNSDE()
{
    if(st_DE_p.nGen_accum_ada_para >= st_DE_p.nGen_th_accum_ada_para) {
        double tmp1 = st_DE_p.ns1_SaNSDE_F * (st_DE_p.ns2_SaNSDE_F + st_DE_p.nf2_SaNSDE_F) + 0.01;
        double tmp2 = st_DE_p.ns2_SaNSDE_F * (st_DE_p.ns1_SaNSDE_F + st_DE_p.nf1_SaNSDE_F) + 0.01;
        st_DE_p.slctProb_SaNSDE_F = tmp1 / (tmp1 + tmp2);
        st_DE_p.ns1_SaNSDE_F = 0;
        st_DE_p.nf1_SaNSDE_F = 0;
        st_DE_p.ns2_SaNSDE_F = 0;
        st_DE_p.nf2_SaNSDE_F = 0;
        st_DE_p.nGen_accum_ada_para = 0;
    } else {
        int i;
        int tmp = 0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if((st_optimizer_p.DE_F_types_all[i] == DE_F_SaNSDE ||
                st_optimizer_p.DE_F_types_all[i] == DE_F_SaNSDE_a) &&
               (st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_CUR_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_1 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_RAND_2 ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_ARCHIVE_RAND ||
                st_optimizer_p.optimizer_types_all[i] == EC_DE_2SELECTED)) {
                tmp++;
                if(st_DE_p.tag_SaNSDE_F[i] == 1) {
                    if(st_DE_p.Sflag[i])
                        st_DE_p.ns1_SaNSDE_F++;
                    else
                        st_DE_p.nf1_SaNSDE_F++;
                } else if(st_DE_p.tag_SaNSDE_F[i] == 2) {
                    if(st_DE_p.Sflag[i])
                        st_DE_p.ns2_SaNSDE_F++;
                    else
                        st_DE_p.nf2_SaNSDE_F++;
                } else {
                    printf("%s: strct_apap_DE_para.tag_SaNSDE_F error - %d, exiting...\n",
                           AT, st_DE_p.tag_SaNSDE_F[i]);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_TAG_SaNSDE_F);
                }
            }
        }
        if(tmp) {
            st_DE_p.nGen_accum_ada_para++;
        }
    }
    return;
}