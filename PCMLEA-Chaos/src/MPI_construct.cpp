#include "global.h"
#include <math.h>
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

#define MAX_STR 1024

void setMPI()
{
    int cur_strct_type;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int multiPop_mode = st_ctrl_p.multiPop_mode;
    //
    if(algo_mech_type == LOCALIZATION ||
       algo_mech_type == DECOMPOSITION) {
        if(multiPop_mode == MP_0)
            cur_strct_type = MAIN_POP_ONLY;
        else if(multiPop_mode == MP_I ||
                multiPop_mode == MP_II ||
                multiPop_mode == MP_ADAP)
            cur_strct_type = UPDATE_MPI_STRUCTURE;
        else if(multiPop_mode == MP_III)
            cur_strct_type = MAIN_POP_SUB_POPS_EQUAL;
        else {
            if(st_MPI_p.mpi_rank == 0) {
                printf("%s: MP_MODE selection is wrong, no other algorithm available.\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        }
    } else if(algo_mech_type == NONDOMINANCE) {
        if(multiPop_mode == MP_0)
            cur_strct_type = MAIN_POP_ONLY;
        else if(multiPop_mode == MP_I ||
                multiPop_mode == MP_II ||
                multiPop_mode == MP_ADAP)
            cur_strct_type = UPDATE_MPI_STRUCTURE_ND;
        else if(multiPop_mode == MP_III)
            cur_strct_type = MAIN_POP_SUB_POPS_ND;
        else {
            if(st_MPI_p.mpi_rank == 0) {
                printf("%s: MP_MODE selection is wrong, no other algorithm available.\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: No such algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    //
    //	classifyProcPerNode();
    //	show_mpi_info();
    build_MPI_structure(cur_strct_type, INIT_TAG);
    //
    return;
}

void build_MPI_structure(int structure_type, int init_tag)
{
    int nArch = st_archive_p.nArch;
    int nPop = st_global_p.nPop;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int mpi_rank = st_MPI_p.mpi_rank;
    // init MPI
    if(init_tag == INIT_TAG) {
        st_MPI_p.comm_obj = NULL;
        st_MPI_p.comm_pop = NULL;
        st_MPI_p.comm_subPop = NULL;
        st_MPI_p.comm_master_subPop_globalScope = NULL;
        st_MPI_p.comm_master_subPop_popScope = NULL;
        st_MPI_p.comm_master_pop = NULL;
    }

    for(int i = 0; i <= nObj; i++) {
        int tmp_size = st_grp_info_p.Groups_sizes[i];
        if(st_ctrl_p.opt_diverVar_separately == FLAG_ON) tmp_size--;
        if(tmp_size <= 0) {
            if(0 == mpi_rank)
                printf("%s: The number of groups for pop %d is %d, invalid, exiting...\n",
                       AT, i, tmp_size);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUP_NUM_INVALID);
        }
        st_grp_info_p.vec_sizeGroups[i] = tmp_size;
    }

    if(structure_type == MAIN_POP_ONLY ||
       structure_type == MAIN_POP_SUB_POPS_EQUAL ||
       structure_type == MAIN_POP_SUB_POPS_ND) {
        if(init_tag != INIT_TAG) {
            if(0 == mpi_rank) {
                printf("%s:Func: void build_MPI_structure(int structure_type, int init_tag)\n", AT);
                printf("can only be called with structure_type == MAIN_POP_ONLY with the\n");
                printf("of INIT_TAG once for initialization");
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MPI_UPDATE_WRONG);
        }
    }
    st_ctrl_p.flag_mainPop = 0;
    st_ctrl_p.flag_multiPop = 0;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    double tmpProcs = (st_global_p.iter - st_global_p.usedIter_init) /
                      (st_global_p.maxIter - st_global_p.usedIter_init + 1.0);
    double tmpRatio = 0.0;
    if(structure_type == MAIN_POP_ONLY) {
        st_ctrl_p.flag_mainPop = 1;
        st_ctrl_p.flag_multiPop = 0;
        st_MPI_p.vec_importance[0] = 1.0;
        for(int i = 1; i <= nObj; i++) {
            st_MPI_p.vec_importance[i] = 0.0;
        }
    } else if(structure_type == MAIN_POP_SUB_POPS_EQUAL) {
        st_ctrl_p.flag_mainPop = 1;
        st_ctrl_p.flag_multiPop = 1;
        st_MPI_p.vec_importance[0] = 1.0 * nPop / st_ctrl_p.popSize_sub_max;
        for(int i = 1; i <= nObj; i++) {
            st_MPI_p.vec_importance[i] = 1.0;
        }
    } else if(structure_type == MAIN_POP_SUB_POPS_ND) {
        st_ctrl_p.flag_mainPop = 1;
        st_ctrl_p.flag_multiPop = 1;
        st_MPI_p.vec_importance[0] = (double)st_archive_p.nArch / st_archive_p.nArch_sub;
        for(int i = 1; i <= nObj; i++) {
            st_MPI_p.vec_importance[i] = 1.0;
        }
    } else if(structure_type == UPDATE_MPI_STRUCTURE ||
              structure_type == UPDATE_MPI_STRUCTURE_ND) {
        st_ctrl_p.flag_mainPop = 1;
        st_ctrl_p.flag_multiPop = 1;
        if(structure_type == UPDATE_MPI_STRUCTURE)
            st_MPI_p.vec_importance[0] = 1.0 * nPop / st_ctrl_p.popSize_sub_max;
        else
            st_MPI_p.vec_importance[0] = 1.0 * nArch / st_archive_p.nArch_sub_max;
        if(st_ctrl_p.multiPop_mode == MP_I) {
            for(int i = 1; i <= nObj; i++) {
                //tmpRatio = (exp(10.0 * tmpProcs) - 1.0) / (exp(10.0) - 1.0);
                //tmpRatio = (exp(10.0 * (1.0 - tmpProcs)) - 1.0) / (exp(10.0) - 1.0);
                //tmpRatio = (1.0 - tmpProcs);
                //tmpRatio = (1.0 - 1.0 / (1.0 + exp(-50 * (tmpProcs - 0.5))));
                //tmpRatio = (1.0 / (1.0 + exp(-50 * (tmpProcs - 0.1))));
                //tmpRatio = (0.01 + 0.99 * (exp(10 * tmpProcs) - 1.0) / (exp(10.0) - 1.0));
                //tmpRatio = (0.01 + 0.99 * (exp(10 * (1.0 - tmpProcs)) - 1.0) / (exp(10.0) - 1.0));
                tmpRatio = (1.0 / (1.0 + exp(-10 * ((1.0 - tmpProcs) - 0.5))));
                st_MPI_p.vec_importance[i] = tmpRatio;
            }
        } else if(st_ctrl_p.multiPop_mode == MP_II) {
            for(int i = 1; i <= nObj; i++) {
                //tmpRatio = (0.01 + 0.99 * (exp(10 * tmpProcs) - 1.0) / (exp(10.0) - 1.0));
                tmpRatio = (1.0 / (1.0 + exp(-10 * (tmpProcs - 0.5))));
                st_MPI_p.vec_importance[i] = tmpRatio;
            }
        } else if(st_ctrl_p.multiPop_mode == MP_ADAP) {
            if(init_tag == INIT_TAG) {
                for(int i = 1; i <= nObj; i++) {
                    st_MPI_p.vec_importance[i] = 1.0;
                }
            } else {
                int* ns_all_pops = (int*)calloc(nObj + 1, sizeof(int));
                int* nf_all_pops = (int*)calloc(nObj + 1, sizeof(int));
                int cur_indx_ns_nf = st_MPI_p.color_pop * nDim + st_MPI_p.color_subPop;
                int my_ns = st_MPI_p.ns_pops[cur_indx_ns_nf];
                int my_nf = st_MPI_p.nf_pops[cur_indx_ns_nf];
                int sum_ns = 0;
                int sum_nf = 0;
                if(!st_MPI_p.color_master_subPop) {
                    my_ns = 0;
                    my_nf = 0;
                } else {
                    MPI_Reduce(&my_ns, &sum_ns, 1, MPI_INT, MPI_SUM, 0, st_MPI_p.comm_master_subPop_popScope);
                    MPI_Reduce(&my_nf, &sum_nf, 1, MPI_INT, MPI_SUM, 0, st_MPI_p.comm_master_subPop_popScope);
                    my_ns = sum_ns;
                    my_nf = sum_nf;
                    if(st_MPI_p.color_master_pop) {
                        MPI_Gather(&my_ns, 1, MPI_INT, ns_all_pops, 1, MPI_INT, 0, st_MPI_p.comm_master_pop);
                        MPI_Gather(&my_nf, 1, MPI_INT, nf_all_pops, 1, MPI_INT, 0, st_MPI_p.comm_master_pop);
                    }
                }
                MPI_Bcast(ns_all_pops, nObj + 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(nf_all_pops, nObj + 1, MPI_INT, 0, MPI_COMM_WORLD);
                if(0 == st_MPI_p.mpi_rank) {
                    for(int i = 0; i <= nObj; i++) {
                        printf("ns-nf_all_pops %d: (%d %d) ", i, ns_all_pops[i], nf_all_pops[i]);
                    }
                    printf("\n");
                }
                double* sc_rate = (double*)calloc(nObj + 1, sizeof(double));
                double sum_all = 0.0;
                for(int i = 0; i <= nObj; i++) {
                    sum_all += ns_all_pops[i];
                    sum_all += nf_all_pops[i];
                }
                double sum_sc_rates = 0.0;
                for(int i = 0; i <= nObj; i++) {
                    //sc_rate[i] = ns_all_pops[i] / sum_all + 0.01;
                    sc_rate[i] = nf_all_pops[i] / (double)(ns_all_pops[i] + nf_all_pops[i]) + 0.01;
                    if(i && sc_rate[i] > sc_rate[0]) {
                        sc_rate[0] = sc_rate[i];
                    }
                }
                for(int i = 0; i <= nObj; i++) {
                    sum_sc_rates += sc_rate[i];
                }
                for(int i = 0; i <= nObj; i++) {
                    tmpRatio = sc_rate[i] / sum_sc_rates;
                    if(tmpRatio > 1.0) tmpRatio = 1.0;
                    st_MPI_p.vec_importance[i] = tmpRatio;
                }
                free(ns_all_pops);
                free(nf_all_pops);
                free(sc_rate);
                st_MPI_p.ns_pops[cur_indx_ns_nf] = 0;
                st_MPI_p.nf_pops[cur_indx_ns_nf] = 0;
            }
        } else {
            if(mpi_rank == 0)
                printf("%s:MP_MODE selection is wrong, no other algorithm available.\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MULTI_POP_MODE);
        }
    } else {
        if(0 == mpi_rank)
            printf("%s:INVALID MPI structure type, EXITING...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MPI_STRUCTURE);
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // master MPI
    st_MPI_p.num_MPI_master =
        st_MPI_p.vec_num_MPI_master[0] = st_grp_info_p.vec_sizeGroups[0] * st_ctrl_p.flag_mainPop;
    for(int i = 1; i <= nObj; i++) {
        st_MPI_p.vec_num_MPI_master[i] = st_grp_info_p.vec_sizeGroups[i] * st_ctrl_p.flag_multiPop;
        st_MPI_p.num_MPI_master += st_MPI_p.vec_num_MPI_master[i];
    }
    // check
    st_ctrl_p.type_grp_loop = LOOP_NONE;
    st_ctrl_p.type_pop_loop = LOOP_NONE;
    int tmp_np_min = 7;
    if(st_MPI_p.num_MPI_master > st_MPI_p.mpi_size) {
        if(0 == mpi_rank) {
            printf("%s: The number of resources is not enough, %d < %d, only for debugging, please provide %d or more CPU cores.\n",
                   AT, st_MPI_p.mpi_size, st_MPI_p.num_MPI_master, st_MPI_p.num_MPI_master);
        }
        if(st_ctrl_p.flag_multiPop) {
            st_ctrl_p.type_pop_loop = LOOP_POP;
            st_MPI_p.vec_num_MPI_master[0] = st_grp_info_p.vec_sizeGroups[0];
            st_MPI_p.num_MPI_master = st_MPI_p.vec_num_MPI_master[0];
        }
        if(st_MPI_p.num_MPI_master > st_MPI_p.mpi_size) {
            st_ctrl_p.type_grp_loop = LOOP_GRP;
            st_MPI_p.vec_num_MPI_master[0] = 1;
            st_MPI_p.num_MPI_master = st_MPI_p.vec_num_MPI_master[0];
        }
        st_MPI_p.vec_num_MPI_slave[0] = st_MPI_p.mpi_size - st_MPI_p.vec_num_MPI_master[0];
        for(int i = 1; i <= nObj; i++) {
            st_MPI_p.vec_num_MPI_master[i] = 0;
            st_MPI_p.vec_num_MPI_slave[i] = 0;
        }
        // main pop size
        if(st_ctrl_p.algo_mech_type == LOCALIZATION ||
           st_ctrl_p.algo_mech_type == DECOMPOSITION) {
            st_MPI_p.nPop_all[0] = st_global_p.nPop;
        } else if(st_ctrl_p.algo_mech_type == NONDOMINANCE) {
            st_MPI_p.nPop_all[0] = st_archive_p.nArch;
        }
        // sub pop sizes
        for(int i = 1; i <= nObj; i++) {
            st_MPI_p.nPop_all[i] = (int)(st_MPI_p.nPop_all[0] / st_MPI_p.vec_importance[0] *
                                         st_MPI_p.vec_importance[i]);
            if(st_MPI_p.nPop_all[i] < tmp_np_min && st_ctrl_p.flag_multiPop)
                st_MPI_p.nPop_all[i] = tmp_np_min;
        }
        //// ND info update
        //if(strct_ctrl_para.algo_mech_type == NONDOMINANCE/* && init_tag != INIT_TAG*/) {
        //    strct_archive_info.nArch_sub_before = strct_archive_info.nArch_sub;
        //    strct_archive_info.nArch_sub = strct_MPI_info.nPop_all[1];
        //}
    } else {
        int tmp_flag = 1;
        while(tmp_flag) {
            tmp_flag = 0;
            // also consider group number
            double tmp_sum_ratio = 0.0;
            for(int i = 0; i <= nObj; i++) {
                st_MPI_p.vec_MPI_ratio[i] = st_MPI_p.vec_importance[i] * st_grp_info_p.vec_sizeGroups[i];
                tmp_sum_ratio += st_MPI_p.vec_MPI_ratio[i];
            }
            for(int i = 0; i <= nObj; i++) {
                st_MPI_p.vec_MPI_ratio[i] /= tmp_sum_ratio;
            }
            // allocate MPI
            int tmp_sum_MPI = 0;
            for(int i = 0; i <= nObj; i++) {
                st_MPI_p.each_size[i] = (int)(st_MPI_p.mpi_size * st_MPI_p.vec_MPI_ratio[i] + 0.5);
                if(structure_type != MAIN_POP_SUB_POPS_ND &&
                   structure_type != UPDATE_MPI_STRUCTURE_ND) {
                    int tmp_sz = st_grp_info_p.vec_sizeGroups[i] - (st_MPI_p.each_size[i] % st_grp_info_p.vec_sizeGroups[i]);
                    st_MPI_p.each_size[i] += (tmp_sz % st_grp_info_p.vec_sizeGroups[i]);
                }
                if(i &&
                   st_ctrl_p.flag_multiPop &&
                   st_MPI_p.each_size[i] < st_grp_info_p.vec_sizeGroups[i]) {
                    st_MPI_p.each_size[i] = st_grp_info_p.vec_sizeGroups[i];
                }
                tmp_sum_MPI += st_MPI_p.each_size[i];
            }
            st_MPI_p.each_size[0] += (st_MPI_p.mpi_size - tmp_sum_MPI);
            // slave MPI number
            for(int i = 0; i <= nObj; i++) {
                st_MPI_p.vec_num_MPI_slave[i] = st_MPI_p.each_size[i] - st_MPI_p.vec_num_MPI_master[i];
            }
            ////pop size for sub pops
            // get back importance ratios
            for(int i = 0; i <= nObj; i++) {
                st_MPI_p.vec_importance[i] = (double)st_MPI_p.each_size[i] / st_grp_info_p.vec_sizeGroups[i];
            }
            // main pop size
            if(st_ctrl_p.algo_mech_type == LOCALIZATION ||
               st_ctrl_p.algo_mech_type == DECOMPOSITION) {
                st_MPI_p.nPop_all[0] = nPop;
            } else if(st_ctrl_p.algo_mech_type == NONDOMINANCE) {
                st_MPI_p.nPop_all[0] = nArch;
            }
            // sub pop sizes
            for(int i = 1; i <= nObj; i++) {
                st_MPI_p.nPop_all[i] = st_MPI_p.nPop_all[0] * st_MPI_p.vec_importance[i] / st_MPI_p.vec_importance[0] + 0.5;
                if(st_MPI_p.nPop_all[i] < tmp_np_min && st_ctrl_p.flag_multiPop) {
                    tmp_flag++;
                    st_MPI_p.nPop_all[i] = tmp_np_min;
                }
                if(st_MPI_p.nPop_all[i] > st_MPI_p.nPop_all[0]) {
                    tmp_flag++;
                    st_MPI_p.nPop_all[i] = st_MPI_p.nPop_all[0];
                }
            }
            if(tmp_flag) {
                for(int i = 1; i <= nObj; i++) {
                    st_MPI_p.vec_importance[i] = (double)st_MPI_p.nPop_all[i] / st_MPI_p.nPop_all[0] * st_MPI_p.vec_importance[0];
                }
            }
            if(0 == mpi_rank) {
                printf("strct_MPI_info.vec_MPI_ratio\n");
                for(int j = 0; j < nObj + 1; j++) {
                    printf("%lf ", st_MPI_p.vec_MPI_ratio[j]);
                }
                printf("\n");
                printf("strct_MPI_info.vec_importance\n");
                for(int j = 0; j < nObj + 1; j++) {
                    printf("%lf ", st_MPI_p.vec_importance[j]);
                }
                printf("\n");
                printf("pop_size\n");
                for(int j = 0; j < nObj + 1; j++) {
                    printf("%d ", st_MPI_p.nPop_all[j]);
                }
                printf("\n");
                printf("MPI_num\n");
                for(int j = 0; j < nObj + 1; j++) {
                    printf("%d ", st_MPI_p.each_size[j]);
                }
                printf("\n");
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //if(0 == mpi_rank) {
    //    printf("strct_MPI_info.vec_MPI_ratio\n");
    //    for(int j = 0; j < nObj + 1; j++) {
    //        printf("%lf ", st_MPI_p.vec_MPI_ratio[j]);
    //    }
    //    printf("\n");
    //    printf("strct_MPI_info.vec_importance\n");
    //    for(int j = 0; j < nObj + 1; j++) {
    //        printf("%lf ", st_MPI_p.vec_importance[j]);
    //    }
    //    printf("\n");
    //    printf("pop_size\n");
    //    for(int j = 0; j < nObj + 1; j++) {
    //        printf("%d ", st_MPI_p.nPop_all[j]);
    //    }
    //    printf("\n");
    //    printf("MPI_num\n");
    //    for(int j = 0; j < nObj + 1; j++) {
    //        printf("%d ", st_MPI_p.each_size[j]);
    //    }
    //    printf("\n");
    //}

    // free MPI before update
    if(init_tag == UPDATE_TAG) {
        MPI_Comm_free(&st_MPI_p.comm_obj);
        MPI_Comm_free(&st_MPI_p.comm_pop);
        MPI_Comm_free(&st_MPI_p.comm_subPop);
        MPI_Comm_free(&st_MPI_p.comm_master_subPop_globalScope);
        MPI_Comm_free(&st_MPI_p.comm_master_subPop_popScope);
        MPI_Comm_free(&st_MPI_p.comm_master_pop);
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //master tag assignment
    //num of MPI for each pop: main pop and sub pops
    //color to identify population
    if(st_MPI_p.mpi_rank < st_MPI_p.num_MPI_master) {
        st_MPI_p.color_master_subPop = 1;
        update_recv_disp_simp(st_MPI_p.vec_num_MPI_master, 1, st_global_p.nObj + 1);
        st_MPI_p.color_pop = 0;
        while(st_MPI_p.mpi_rank >= st_MPI_p.recv_size[st_MPI_p.color_pop] +
              st_MPI_p.disp_size[st_MPI_p.color_pop]) {
            st_MPI_p.color_pop++;
        }
    } else {
        st_MPI_p.color_master_subPop = 0;
        update_recv_disp_simp(st_MPI_p.vec_num_MPI_slave, 1, st_global_p.nObj + 1);
        st_MPI_p.color_pop = 0;
        while(st_MPI_p.mpi_rank - st_MPI_p.num_MPI_master >= st_MPI_p.recv_size[st_MPI_p.color_pop] +
              st_MPI_p.disp_size[st_MPI_p.color_pop]) {
            st_MPI_p.color_pop++;
        }
    }
    //
    //color to identify main pop or sub pops
    if(st_MPI_p.color_pop) {
        st_MPI_p.color_obj = 1;
    } else {
        st_MPI_p.color_obj = 0;
    }
    //split to 2 kinds of MPI_Comm, either M or 1
    MPI_Comm_split(MPI_COMM_WORLD, st_MPI_p.color_obj, st_MPI_p.mpi_rank, &st_MPI_p.comm_obj);
    MPI_Comm_size(st_MPI_p.comm_obj, &st_MPI_p.mpi_size_obj);
    MPI_Comm_rank(st_MPI_p.comm_obj, &st_MPI_p.mpi_rank_obj);
    //MPI processes are split to (M+1) MPI_Comm
    MPI_Comm_split(MPI_COMM_WORLD, st_MPI_p.color_pop, st_MPI_p.mpi_rank, &st_MPI_p.comm_pop);
    MPI_Comm_size(st_MPI_p.comm_pop, &st_MPI_p.mpi_size_pop);
    MPI_Comm_rank(st_MPI_p.comm_pop, &st_MPI_p.mpi_rank_pop);

    //MPI processes are split to 2 MPI_Comm,
    //if master_color==1, comm_masters contains the masters of each strct_MPI_info.comm_master_species_globalScope
    //else, strct_MPI_info.comm_master_species_globalScope contain the remain MPIs
    MPI_Comm_split(MPI_COMM_WORLD, st_MPI_p.color_master_subPop, st_MPI_p.mpi_rank,
                   &st_MPI_p.comm_master_subPop_globalScope);
    MPI_Comm_size(st_MPI_p.comm_master_subPop_globalScope, &st_MPI_p.mpi_size_master_subPop_globalScope);
    MPI_Comm_rank(st_MPI_p.comm_master_subPop_globalScope, &st_MPI_p.mpi_rank_master_subPop_globalScope);
    //MPI processes are split to M+1 MPI_Comm,
    //if master_flag==1, comm_masters contains the masters of each comm_master_species_pop
    //else, comm_master_species_pop contain the remain MPIs
    MPI_Comm_split(st_MPI_p.comm_pop, st_MPI_p.color_master_subPop, st_MPI_p.mpi_rank,
                   &st_MPI_p.comm_master_subPop_popScope);
    MPI_Comm_size(st_MPI_p.comm_master_subPop_popScope, &st_MPI_p.mpi_size_master_subPop_popScope);
    MPI_Comm_rank(st_MPI_p.comm_master_subPop_popScope, &st_MPI_p.mpi_rank_master_subPop_popScope);

    //color to identify group/species, get the population's grouping info
    int tmp_nGrp = st_grp_info_p.vec_sizeGroups[st_MPI_p.color_pop];
    if(st_MPI_p.color_master_subPop) {
        st_MPI_p.color_subPop = st_MPI_p.mpi_rank_pop;
    } else {
        if(st_MPI_p.mpi_size_pop < tmp_nGrp) {
            printf("pop_local_rank %d (%d) - More strct_grp_info_vals.Groups (%d) than MPI processes (%d) in population %d\n",
                   st_MPI_p.mpi_rank_pop, st_MPI_p.mpi_rank, tmp_nGrp, st_MPI_p.mpi_size_pop,
                   st_MPI_p.color_pop);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MPI_LESS);
        }
        int quo = st_MPI_p.mpi_size_pop / tmp_nGrp;
        int rem = st_MPI_p.mpi_size_pop % tmp_nGrp;
        for(int i = 0; i < tmp_nGrp; i++) {
            st_MPI_p.each_size[i] = quo;
            if(i < rem)
                st_MPI_p.each_size[i]++;
            //substract the master MPI, one for each
            st_MPI_p.each_size[i]--;
        }
        update_recv_disp_simp(st_MPI_p.each_size, 1, tmp_nGrp);
        st_MPI_p.color_subPop = 0;
        while(st_MPI_p.mpi_rank_pop - tmp_nGrp >= st_MPI_p.recv_size[st_MPI_p.color_subPop] +
              st_MPI_p.disp_size[st_MPI_p.color_subPop]) {
            st_MPI_p.color_subPop++;
        }
    }
    //split according to group/species
    MPI_Comm_split(st_MPI_p.comm_pop, st_MPI_p.color_subPop, st_MPI_p.mpi_rank_pop, &st_MPI_p.comm_subPop);
    MPI_Comm_size(st_MPI_p.comm_subPop, &st_MPI_p.mpi_size_subPop);
    MPI_Comm_rank(st_MPI_p.comm_subPop, &st_MPI_p.mpi_rank_subPop);

    //get the master rank in the global scope
    st_MPI_p.root_master_subPop_globalScope = st_MPI_p.mpi_rank;
    if(st_MPI_p.color_master_subPop) {
        MPI_Bcast(&st_MPI_p.root_master_subPop_globalScope, 1, MPI_INT, 0, st_MPI_p.comm_master_subPop_globalScope);
    }
    MPI_Bcast(&st_MPI_p.root_master_subPop_globalScope, 1, MPI_INT, 0, st_MPI_p.comm_subPop);

    st_ctrl_p.count_multiPop = 0;
    st_MPI_p.color_master_pop = 0;
    for(int i = 0; i < nObj + 1; i++)
        st_MPI_p.globalRank_master_pop[i] = -1;
    int iPopulation = 0;
    int cPop = -1;
    int cCMS = -1;
    int cRMSP = -1;
    for(int i = 0; i < st_MPI_p.mpi_size; i++) {
        cPop = st_MPI_p.color_pop;
        cCMS = st_MPI_p.color_master_subPop;
        cRMSP = st_MPI_p.mpi_rank_master_subPop_popScope;
        MPI_Bcast(&cPop, 1, MPI_INT, i, MPI_COMM_WORLD);
        MPI_Bcast(&cCMS, 1, MPI_INT, i, MPI_COMM_WORLD);
        MPI_Bcast(&cRMSP, 1, MPI_INT, i, MPI_COMM_WORLD);
        if(cPop == iPopulation && cCMS == 1 && cRMSP == 0) {
            st_MPI_p.globalRank_master_pop[iPopulation++] = i;
            if(mpi_rank == i) st_MPI_p.color_master_pop = 1;
            st_ctrl_p.count_multiPop++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if(mpi_rank == 0) {
        for(int i = 0; i <= nObj; i++) {
            printf("%d ", st_MPI_p.globalRank_master_pop[i]);
        }
        printf("\n\n");
    }

    MPI_Comm_split(MPI_COMM_WORLD, st_MPI_p.color_master_pop, st_MPI_p.mpi_rank,
                   &st_MPI_p.comm_master_pop);
    MPI_Comm_size(st_MPI_p.comm_master_pop, &st_MPI_p.mpi_size_master_pop);
    MPI_Comm_rank(st_MPI_p.comm_master_pop, &st_MPI_p.mpi_rank_master_pop);

    //get the group info for each species
    st_MPI_p.cur_pop_index = st_MPI_p.color_pop;
    st_MPI_p.cur_grp_index = st_MPI_p.color_subPop;
    localAssignGroup(st_MPI_p.color_pop, st_MPI_p.color_subPop);

    //////////////////////////////////////////////////////////////////////////
    // ND info update
    if(st_ctrl_p.algo_mech_type == NONDOMINANCE/* && init_tag != INIT_TAG*/) {
        st_archive_p.nArch_sub_before = st_archive_p.nArch_sub;
        st_archive_p.nArch_sub = st_MPI_p.nPop_all[st_MPI_p.color_pop];
    }

    //	classifyProcPerNode();
    return;
}
