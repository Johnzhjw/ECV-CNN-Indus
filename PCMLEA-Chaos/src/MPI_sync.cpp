#include "global.h"
#include <math.h>
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

void population_synchronize(int algoMechType)
{
    int nPop = st_global_p.nPop;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj = st_pop_evo_cur.obj;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    MPI_Comm comm_subPop = st_MPI_p.comm_subPop;
    int mpi_rank = st_MPI_p.mpi_rank;
    //
    if(algoMechType == LOCALIZATION) {
        //for each group, collect the current population to rank 0 process
        update_recv_disp(each_size_subPop, nDim, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Gatherv(cur_var, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                    arc_var, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                    0, comm_subPop);
        update_recv_disp(each_size_subPop, nObj, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Gatherv(cur_obj, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                    arc_obj, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                    0, comm_subPop);
    } else if(algoMechType == DECOMPOSITION) {
        memcpy(arc_var, cur_var, nPop * nDim * sizeof(double));
        memcpy(arc_obj, cur_obj, nPop * nObj * sizeof(double));
    } else {
        if(0 == mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    st_archive_p.cnArch = nPop;

    int the_rank, the_size;
    MPI_Comm the_comm;
    if(st_ctrl_p.flag_multiPop) {
        the_rank = st_MPI_p.mpi_rank_master_subPop_popScope;
        the_size = st_MPI_p.mpi_size_master_subPop_popScope;
        the_comm = st_MPI_p.comm_master_subPop_popScope;
    } else {
        the_rank = st_MPI_p.mpi_rank_master_subPop_globalScope;
        the_size = st_MPI_p.mpi_size_master_subPop_globalScope;
        the_comm = st_MPI_p.comm_master_subPop_globalScope;
    }

    //strct_MPI_info.comm_master_species_globalScope is the communication area of all group masters
    //refinement is done within this MPI_Comm
    int i, j;
    if(st_MPI_p.color_master_subPop) {
        int step = (int)sqrt((double)the_size);
        int ind_u = (the_rank - step + the_size) % the_size;
        int ind_d = (the_rank + step) % the_size;
        int ind_l = (the_rank - 1 + the_size) % the_size;
        int ind_r = (the_rank + 1) % the_size;
        double f1, f2;

        double rndt = pointer_gen_rand();
        MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, the_comm);

        int collect_to;
        int collect_from;

        //up->down
        if(rndt < 0.25) {
            collect_to = ind_d;
            collect_from = ind_u;
        }
        //down->up
        else if(rndt < 0.5) {
            collect_to = ind_u;
            collect_from = ind_d;
        }
        //left->right
        else if(rndt < 0.75) {
            collect_to = ind_r;
            collect_from = ind_l;
        }
        //right->left
        else {
            collect_to = ind_l;
            collect_from = ind_r;
        }
        MPI_Sendrecv(arc_var, nPop * nDim, MPI_DOUBLE, collect_to, 1,
                     repo_var, nPop * nDim, MPI_DOUBLE, collect_from, 1,
                     the_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(arc_obj, nPop * nObj, MPI_DOUBLE, collect_to, 2,
                     repo_obj, nPop * nObj, MPI_DOUBLE, collect_from, 2,
                     the_comm, MPI_STATUS_IGNORE);
        // reference points
        // ideal point
        for(i = 0; i < nPop; i++)
            for(j = 0; j < nObj; j++)
                if(st_decomp_p.idealpoint[j] > arc_obj[i * nObj + j]) {
                    st_decomp_p.idealpoint[j] = arc_obj[i * nObj + j];
                }
        for(i = 0; i < nPop; i++)
            for(j = 0; j < nObj; j++)
                if(st_decomp_p.idealpoint[j] > repo_obj[i * nObj + j]) {
                    st_decomp_p.idealpoint[j] = repo_obj[i * nObj + j];
                }
        // nadir point
        for(j = 0; j < nObj; j++) {
            st_decomp_p.nadirpoint[j] = -1e30;
        }
        for(i = 0; i < nPop; i++)
            for(j = 0; j < nObj; j++)
                if(st_decomp_p.nadirpoint[j] < arc_obj[i * nObj + j]) {
                    st_decomp_p.nadirpoint[j] = arc_obj[i * nObj + j];
                }
        for(i = 0; i < nPop; i++)
            for(j = 0; j < nObj; j++)
                if(st_decomp_p.nadirpoint[j] < repo_obj[i * nObj + j]) {
                    st_decomp_p.nadirpoint[j] = repo_obj[i * nObj + j];
                }

        for(i = 0; i < nPop; i++) {
            f1 = fitnessFunction(&repo_obj[i * nObj], &st_decomp_p.weights_all[i * nObj]);
            f2 = fitnessFunction(&arc_obj[i * nObj], &st_decomp_p.weights_all[i * nObj]);
            if(f1 < f2) {
                memcpy(&arc_var[i * nDim], &repo_var[i * nDim], nDim * sizeof(double));
                memcpy(&arc_obj[i * nObj], &repo_obj[i * nObj], nObj * sizeof(double));
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(algoMechType == LOCALIZATION) {
        //for each group, collect the current population to rank 0 process
        update_recv_disp(each_size_subPop, nDim, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Scatterv(arc_var, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                     cur_var, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                     0, comm_subPop);
        update_recv_disp(each_size_subPop, nObj, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Scatterv(arc_obj, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                     cur_obj, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                     0, comm_subPop);
    } else if(algoMechType == DECOMPOSITION) {
        memcpy(cur_var, arc_var, nPop * nDim * sizeof(double));
        memcpy(cur_obj, arc_obj, nPop * nObj * sizeof(double));
    }

    return;
}

void population_synchronize_random()
{
    /*	//for each group, collect the current population to rank 0 process
    update_recv_disp(each_size_group,strct_global_paras.nDim,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Gatherv(xCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    archive,recv_size_group,disp_size_group,MPI_DOUBLE,
    0,strct_MPI_info.comm_species);
    update_recv_disp(each_size_group,strct_global_paras.nObj,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Gatherv(fCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    archFit,recv_size_group,disp_size_group,MPI_DOUBLE,
    0,strct_MPI_info.comm_species);*/
    memcpy(st_archive_p.var, st_pop_evo_cur.var, st_global_p.nPop * st_global_p.nDim * sizeof(double));
    memcpy(st_archive_p.obj, st_pop_evo_cur.obj, st_global_p.nPop * st_global_p.nObj * sizeof(double));
    st_archive_p.cnArch = st_global_p.nPop;
    // 	MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("LINE 1.\n");

    int the_rank, the_size;
    MPI_Comm the_comm;
    if(st_ctrl_p.flag_multiPop) {
        double rndt = pointer_gen_rand();
        MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(rndt < 0.9) {
            the_rank = st_MPI_p.mpi_rank_master_subPop_popScope;
            the_size = st_MPI_p.mpi_size_master_subPop_popScope;
            the_comm = st_MPI_p.comm_master_subPop_popScope;
        } else {
            the_rank = st_MPI_p.mpi_rank_master_subPop_globalScope;
            the_size = st_MPI_p.mpi_size_master_subPop_globalScope;
            the_comm = st_MPI_p.comm_master_subPop_globalScope;
        }
    } else {
        the_rank = st_MPI_p.mpi_rank_master_subPop_globalScope;
        the_size = st_MPI_p.mpi_size_master_subPop_globalScope;
        the_comm = st_MPI_p.comm_master_subPop_globalScope;
    }

    //strct_MPI_info.comm_master_species_globalScope is the communication area of all group masters
    //refinement is done within this MPI_Comm
    int i, j;
    if(st_MPI_p.color_master_subPop && the_size > 1) {
        int step = rnd(1, the_size - 1);
        MPI_Bcast(&step, 1, MPI_INT, 0, the_comm);
        int ind_l = (the_rank - step + the_size) % the_size;
        int ind_r = (the_rank + step) % the_size;
        double f1, f2;

        double rndt = pointer_gen_rand();
        MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, the_comm);

        int collect_from;
        int collect_to;

        //left->right
        if(rndt < 0.5) {
            collect_to = ind_r;
            collect_from = ind_l;
        }
        //right->left
        else {
            collect_to = ind_l;
            collect_from = ind_r;
        }
        MPI_Sendrecv(st_archive_p.var, st_global_p.nPop * st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                     st_repo_p.var, st_global_p.nPop * st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                     the_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_archive_p.obj, st_global_p.nPop * st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                     st_repo_p.obj, st_global_p.nPop * st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                     the_comm, MPI_STATUS_IGNORE);
        // reference points
        // ideal point
        for(i = 0; i < st_global_p.nPop; i++) for(j = 0; j < st_global_p.nObj;
                    j++) if(st_decomp_p.idealpoint[j] > st_archive_p.obj[i * st_global_p.nObj + j]) {
                    st_decomp_p.idealpoint[j] = st_archive_p.obj[i * st_global_p.nObj + j];
                }
        for(i = 0; i < st_global_p.nPop; i++) for(j = 0; j < st_global_p.nObj;
                    j++) if(st_decomp_p.idealpoint[j] > st_repo_p.obj[i * st_global_p.nObj + j]) {
                    st_decomp_p.idealpoint[j] = st_repo_p.obj[i * st_global_p.nObj + j];
                }
        // nadir point
        for(j = 0; j < st_global_p.nObj; j++) {
            st_decomp_p.nadirpoint[j] = -1e30;
        }
        for(i = 0; i < st_global_p.nPop; i++) for(j = 0; j < st_global_p.nObj;
                    j++) if(st_decomp_p.nadirpoint[j] < st_archive_p.obj[i * st_global_p.nObj + j]) {
                    st_decomp_p.nadirpoint[j] = st_archive_p.obj[i * st_global_p.nObj + j];
                }
        for(i = 0; i < st_global_p.nPop; i++) for(j = 0; j < st_global_p.nObj;
                    j++) if(st_decomp_p.nadirpoint[j] < st_repo_p.obj[i * st_global_p.nObj + j]) {
                    st_decomp_p.nadirpoint[j] = st_repo_p.obj[i * st_global_p.nObj + j];
                }

        for(i = 0; i < st_global_p.nPop; i++) {
            f1 = fitnessFunction(&st_repo_p.obj[i * st_global_p.nObj],
                                 &st_decomp_p.weights_all[i * st_global_p.nObj]);
            f2 = fitnessFunction(&st_archive_p.obj[i * st_global_p.nObj],
                                 &st_decomp_p.weights_all[i * st_global_p.nObj]);
            if(f1 < f2) {
                memcpy(&st_archive_p.var[i * st_global_p.nDim], &st_repo_p.var[i * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
                memcpy(&st_archive_p.obj[i * st_global_p.nObj], &st_repo_p.obj[i * st_global_p.nObj],
                       st_global_p.nObj * sizeof(double));
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("line1003.\n");
    /*	//for each group, scatter the current population to processes
    update_recv_disp(each_size_group,strct_global_paras.nDim,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Scatterv(archive,recv_size_group,disp_size_group,MPI_DOUBLE,
    xCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    0,strct_MPI_info.comm_species);
    update_recv_disp(each_size_group,strct_global_paras.nObj,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Scatterv(archFit,recv_size_group,disp_size_group,MPI_DOUBLE,
    fCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    0,strct_MPI_info.comm_species);*/
    memcpy(st_pop_evo_cur.var, st_archive_p.var, st_global_p.nPop * st_global_p.nDim * sizeof(double));
    memcpy(st_pop_evo_cur.obj, st_archive_p.obj, st_global_p.nPop * st_global_p.nObj * sizeof(double));

    return;
}

void population_synchronize_ND()
{
    if(st_MPI_p.color_obj) {
        memcpy(st_archive_p.var_Ex, st_archive_p.var,
               st_archive_p.nArch_sub * st_global_p.nDim * sizeof(double));
        memcpy(st_archive_p.obj_Ex, st_archive_p.obj,
               st_archive_p.nArch_sub * st_global_p.nObj * sizeof(double));
        st_archive_p.cnArchEx = st_archive_p.nArch_sub;
    } else {
        memcpy(st_archive_p.var_Ex, st_archive_p.var, st_archive_p.nArch * st_global_p.nDim * sizeof(double));
        memcpy(st_archive_p.obj_Ex, st_archive_p.obj, st_archive_p.nArch * st_global_p.nObj * sizeof(double));
        st_archive_p.cnArchEx = st_archive_p.nArch;
    }
    MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("LINE 1.\n");

    //int i;
    //if(strct_global_paras.generation >= 14 && 0 == strct_MPI_info.mpi_rank)
    //    printf("xxxxxxxxxxxxxxxxxxxx%ld\n", get_rnd_uni_init());
    if(st_MPI_p.color_master_subPop) {
        int step = (int)sqrt((double)st_MPI_p.mpi_size_master_subPop_globalScope);
        int ind_u = (st_MPI_p.mpi_rank_master_subPop_globalScope - step + st_MPI_p.mpi_size_master_subPop_globalScope) %
                    st_MPI_p.mpi_size_master_subPop_globalScope;
        int ind_d = (st_MPI_p.mpi_rank_master_subPop_globalScope + step) % st_MPI_p.mpi_size_master_subPop_globalScope;
        int ind_l = (st_MPI_p.mpi_rank_master_subPop_globalScope - 1 + st_MPI_p.mpi_size_master_subPop_globalScope) %
                    st_MPI_p.mpi_size_master_subPop_globalScope;
        int ind_r = (st_MPI_p.mpi_rank_master_subPop_globalScope + 1) % st_MPI_p.mpi_size_master_subPop_globalScope;

        double rndt = pointer_gen_rand();
        MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, st_MPI_p.comm_master_subPop_globalScope);
        //if(strct_global_paras.generation >= 14 && 0 == strct_MPI_info.mpi_rank)
        //    printf("xxxxxxxxxxx------------%ld\n", get_rnd_uni_init());

        int collect_from;
        int collect_to;

        //up->down
        if(rndt < 0.25) {
            collect_to = ind_d;
            collect_from = ind_u;
        }
        //down->up
        else if(rndt < 0.5) {
            collect_to = ind_u;
            collect_from = ind_d;
        }
        //left->right
        else if(rndt < 0.75) {
            collect_to = ind_r;
            collect_from = ind_l;
        }
        //right->left
        else {
            collect_to = ind_l;
            collect_from = ind_r;
        }
        //if(strct_global_paras.generation == 0 && 5 == strct_MPI_info.mpi_rank)
        //    printf("xxxxxxxxxxx------------to_%d_fr_%d -> gen %d\n", collect_to, collect_from, strct_global_paras.generation);

        MPI_Sendrecv(&st_archive_p.cnArchEx, 1, MPI_INT, collect_to, 0,
                     &st_repo_p.nRep, 1, MPI_INT, collect_from, 0,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_archive_p.var_Ex, st_archive_p.cnArchEx * st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                     st_repo_p.var, st_repo_p.nRep * st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_archive_p.obj_Ex, st_archive_p.cnArchEx * st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                     st_repo_p.obj, st_repo_p.nRep * st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);

        //
        for(int i = 0; i < st_global_p.nInd_max_repo; i++) st_repo_p.flag[i] = -1;
        int tmp_size = st_repo_p.nRep;
        if(st_MPI_p.color_obj) {
            memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var,
                   st_archive_p.nArch_sub * st_global_p.nDim * sizeof(double));
            memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj,
                   st_archive_p.nArch_sub * st_global_p.nObj * sizeof(double));
            st_repo_p.nRep += st_archive_p.nArch_sub;
            refineRepository_generateArchive_sub();
        } else {
            memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var,
                   st_archive_p.nArch * st_global_p.nDim * sizeof(double));
            memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj,
                   st_archive_p.nArch * st_global_p.nObj * sizeof(double));
            st_repo_p.nRep += st_archive_p.nArch;
            refineRepository_generateArchive();
        }
        //////////////////////////////////////////////////////////////////////////
        //if(strct_global_paras.generation >= 14 && 0 == strct_MPI_info.mpi_rank)
        //    printf("xxxxxxxxxxx::::::::::::::::%ld\n", get_rnd_uni_init());
        int tmp_nPop;
        if(st_MPI_p.color_obj)
            tmp_nPop = st_archive_p.nArch_sub;
        else
            tmp_nPop = st_archive_p.nArch;
        int tmp_ind;
        for(int i = 0; i < tmp_nPop; i++) {
            int cur_ind = st_repo_p.nRep - 1 - i;
            if(st_repo_p.flag[cur_ind] < 0 || st_repo_p.flag[cur_ind] >= tmp_nPop) {
                if(st_archive_p.curSize_inferior < tmp_nPop) {
                    tmp_ind = st_archive_p.curSize_inferior;
                    st_archive_p.curSize_inferior++;
                } else {
                    st_archive_p.curSize_inferior = tmp_nPop;
                    tmp_ind = rnd(0, tmp_nPop - 1);
                }
                memcpy(&st_pop_evo_cur.var_inferior[tmp_ind * st_global_p.nDim],
                       &st_repo_p.var[cur_ind * st_global_p.nDim], st_global_p.nDim * sizeof(double));
            }
        }
    }
    //if(strct_global_paras.generation >= 14 && 0 == strct_MPI_info.mpi_rank)
    //    printf("xxxxxxxxxxx............%ld\n", get_rnd_uni_init());
    MPI_Barrier(MPI_COMM_WORLD); //if (strct_MPI_info.mpi_rank == 0)printf("line1003.\n");

    return;
}

void population_synchronize_random_ND()
{
    /*	//for each group, collect the current population to rank 0 process
    update_recv_disp(each_size_group,strct_global_paras.nDim,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Gatherv(xCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    archive,recv_size_group,disp_size_group,MPI_DOUBLE,
    0,strct_MPI_info.comm_species);
    update_recv_disp(each_size_group,strct_global_paras.nObj,strct_MPI_info.mpi_size_species,recv_size_group,disp_size_group);
    MPI_Gatherv(fCurrent,recv_size_group[strct_MPI_info.mpi_rank_species],MPI_DOUBLE,
    archFit,recv_size_group,disp_size_group,MPI_DOUBLE,
    0,strct_MPI_info.comm_species);*/
    memcpy(st_archive_p.var_Ex, st_archive_p.var, st_archive_p.cnArch * st_global_p.nDim * sizeof(double));
    memcpy(st_archive_p.obj_Ex, st_archive_p.obj, st_archive_p.cnArch * st_global_p.nObj * sizeof(double));
    st_archive_p.cnArchEx = st_archive_p.cnArch;
    // 	MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("LINE 1.\n");

    //strct_MPI_info.comm_master_species_globalScope is the communication area of all group masters
    //refinement is done within this MPI_Comm
    if(st_MPI_p.color_master_subPop) {
        int step = rnd(1, st_MPI_p.mpi_size_master_subPop_globalScope - 1);
        MPI_Bcast(&step, 1, MPI_INT, 0, st_MPI_p.comm_master_subPop_globalScope);
        int ind_l = (st_MPI_p.mpi_rank_master_subPop_globalScope - step + st_MPI_p.mpi_size_master_subPop_globalScope) %
                    st_MPI_p.mpi_size_master_subPop_globalScope;
        int ind_r = (st_MPI_p.mpi_rank_master_subPop_globalScope + step) % st_MPI_p.mpi_size_master_subPop_globalScope;

        double rndt = pointer_gen_rand();
        MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, st_MPI_p.comm_master_subPop_globalScope);

        int collect_from;
        int collect_to;

        //left->right
        if(rndt < 0.5) {
            collect_to = ind_r;
            collect_from = ind_l;
        }
        //right->left
        else {
            collect_to = ind_l;
            collect_from = ind_r;
        }
        MPI_Sendrecv(&st_archive_p.cnArchEx, 1, MPI_INT, collect_to, 0,
                     &st_repo_p.nRep, 1, MPI_INT, collect_from, 0,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_archive_p.var_Ex, st_archive_p.cnArchEx * st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                     st_repo_p.var, st_repo_p.nRep * st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_archive_p.obj_Ex, st_archive_p.cnArchEx * st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                     st_repo_p.obj, st_repo_p.nRep * st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                     st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);

        memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var,
               st_archive_p.cnArch * st_global_p.nDim * sizeof(double));
        memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj,
               st_archive_p.cnArch * st_global_p.nObj * sizeof(double));
        st_repo_p.nRep += st_archive_p.cnArch;

        if(st_MPI_p.color_obj) {
            refineRepository_generateArchive_sub();
        } else {
            refineRepository_generateArchive();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("line1003.\n");

    return;
}

void synchronizeObjectiveBests(int algoMechType)
{
    if(algoMechType == DECOMPOSITION && !st_MPI_p.color_master_subPop) return;

    int the_rank;
    //int the_size;
    MPI_Comm the_comm;

    if(algoMechType == LOCALIZATION) {
        the_rank = st_MPI_p.mpi_rank;
        the_comm = MPI_COMM_WORLD;
    } else if(algoMechType == DECOMPOSITION) {
        the_rank = st_MPI_p.mpi_rank_master_subPop_globalScope;
        //the_size = strct_MPI_info.mpi_size_master_species_globalScope;
        the_comm = st_MPI_p.comm_master_subPop_globalScope;
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }

    struct val_loc {
        double val;
        int    loc;
    };

    int i;
    val_loc strct1, strct2;

    for(i = 0; i < st_global_p.nObj; i++) {
        strct1.val = st_pop_best_p.obj_best_subObjs_all[i * st_global_p.nObj + i];
        strct1.loc = the_rank;
        MPI_Allreduce(&strct1, &strct2, 1, MPI_DOUBLE_INT, MPI_MINLOC, the_comm);
        MPI_Bcast(&st_pop_best_p.var_best_subObjs_all[i * st_global_p.nDim], st_global_p.nDim, MPI_DOUBLE,
                  strct2.loc, the_comm);
        MPI_Bcast(&st_pop_best_p.obj_best_subObjs_all[i * st_global_p.nObj], st_global_p.nObj, MPI_DOUBLE,
                  strct2.loc, the_comm);
    }

    return;
}

void synchronizeReferencePoint(int algoMechType)
{
    if(algoMechType == DECOMPOSITION && !st_MPI_p.color_master_subPop) return;

    int the_rank;
    //int the_size;
    MPI_Comm the_comm;
    int pop_size;

    if(algoMechType == LOCALIZATION) {
        the_rank = st_MPI_p.mpi_rank;
        the_comm = MPI_COMM_WORLD;
        pop_size = st_global_p.nPop_mine;
    } else if(algoMechType == DECOMPOSITION) {
        the_rank = st_MPI_p.mpi_rank_master_subPop_globalScope;
        //the_size = strct_MPI_info.mpi_size_master_species_globalScope;
        the_comm = st_MPI_p.comm_master_subPop_globalScope;
        pop_size = st_global_p.nPop;
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }

    int i, j;
    for(i = 0; i < pop_size; i++) {
        update_idealpoint(&st_pop_evo_cur.obj[i * st_global_p.nObj]);
    }
    // 	if (!strcmp("_TCH2",strct_global_paras.strFunctionType) ||
    // 		!strcmp("_MTCH2",strct_global_paras.strFunctionType) ||
    // 		!strcmp("_MPBI",strct_global_paras.strFunctionType) ||
    // 		!strcmp("_APD",strct_global_paras.strFunctionType))
    for(j = 0; j < st_global_p.nObj; j++) {
        st_decomp_p.nadirpoint[j] = -1e30;
    }
    update_nadirpoint(st_pop_evo_cur.obj, pop_size, st_global_p.nObj);

    struct D_I {
        double val;
        int id;
    };

    D_I info_in, info_out;
    // ideal point
    for(i = 0; i < st_global_p.nObj; i++) {
        /*for (int j = 0; j < strct_MPI_info.mpi_size; j++) {
        if (j == strct_MPI_info.mpi_rank_master_species_globalScope) {
        printf("rank: %04d obj: %d: %lf\t", strct_MPI_info.mpi_rank_master_species_globalScope, i + 1, strct_decomp_paras.idealpoint[i]);
        printf("\n");
        }
        MPI_Barrier(strct_MPI_info.comm_master_species_globalScope);
        }*/
        info_in.val = st_decomp_p.idealpoint[i];
        info_in.id = the_rank;
        MPI_Allreduce(&info_in, &info_out, 1, MPI_DOUBLE_INT, MPI_MINLOC, the_comm);
        //best_rank = info_out.id;
        //MPI_Bcast(&strct_decomp_paras.idealpoint[i], 1, MPI_DOUBLE, best_rank, strct_MPI_info.comm_master_species_globalScope);
        st_decomp_p.idealpoint[i] = info_out.val;
        /*for (int j = 0; j < strct_MPI_info.mpi_size; j++) {
        if (strct_MPI_info.mpi_rank_master_species_globalScope == j) {
        printf("ID: %d\n", best_rank);
        printf("rank: %03d obj: %d: %lf\t", strct_MPI_info.mpi_rank_master_species_globalScope, i + 1, strct_decomp_paras.idealpoint[i]);
        printf("\n");
        }
        MPI_Barrier(strct_MPI_info.comm_master_species_globalScope);
        }*/
    }
    // nad point
    for(i = 0; i < st_global_p.nObj; i++) {
        /*for (int j = 0; j < strct_MPI_info.mpi_size; j++) {
        if (j == strct_MPI_info.mpi_rank_master_species_globalScope) {
        printf("rank: %04d obj: %d: %lf\t", strct_MPI_info.mpi_rank_master_species_globalScope, i + 1, strct_decomp_paras.nadirpoint[i]);
        printf("\n");
        }
        MPI_Barrier(strct_MPI_info.comm_master_species_globalScope);
        }*/
        info_in.val = st_decomp_p.nadirpoint[i];
        info_in.id = the_rank;
        MPI_Allreduce(&info_in, &info_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, the_comm);
        //best_rank = info_out.id;
        //MPI_Bcast(&strct_decomp_paras.nadirpoint[i], 1, MPI_DOUBLE, best_rank, strct_MPI_info.comm_master_species_globalScope);
        st_decomp_p.nadirpoint[i] = info_out.val;
        /*for (int j = 0; j < strct_MPI_info.mpi_size; j++) {
        if (strct_MPI_info.mpi_rank_master_species_globalScope == j) {
        printf("ID: %d\n", best_rank);
        printf("rank: %03d obj: %d: %lf\t", strct_MPI_info.mpi_rank_master_species_globalScope, i + 1, strct_decomp_paras.nadirpoint[i]);
        printf("\n");
        }
        MPI_Barrier(strct_MPI_info.comm_master_species_globalScope);
        }*/
    }

    return;
}