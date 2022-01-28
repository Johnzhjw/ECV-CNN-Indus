#include "global.h"
#include <math.h>
#include <time.h>

void localInitialization()
{
    int color_pop = st_MPI_p.color_pop;
    int* nPop_all = st_MPI_p.nPop_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int nPop = st_global_p.nPop;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    int* nPop_mine = &st_global_p.nPop_mine;
    int* nPop_exchange = &st_global_p.nPop_exchange;
    int* n_weights_mine = &st_pop_comm_p.n_weights_mine;
    int* n_weights_left = &st_pop_comm_p.n_weights_left;
    int* n_weights_right = &st_pop_comm_p.n_weights_right;
    int* n_neighbor_left = &st_pop_comm_p.n_neighbor_left;
    int* n_neighbor_right = &st_pop_comm_p.n_neighbor_right;
    double* weights_mine = st_decomp_p.weights_mine;
    double* weights_all = st_decomp_p.weights_all;
    double* posFactor_mine = st_pop_comm_p.posFactor_mine;
    double* diver_var_store_all = st_grp_info_p.diver_var_store_all;
    double* weights_left = st_decomp_p.weights_left;
    double* weights_right = st_decomp_p.weights_right;
    double* posFactor_left = st_pop_comm_p.posFactor_left;
    double* posFactor_right = st_pop_comm_p.posFactor_right;
    //
    int i;
    int quo;
    int rem;
    quo = nPop_all[color_pop] / mpi_size_subPop;
    rem = nPop_all[color_pop] % mpi_size_subPop;
    for(i = 0; i < mpi_size_subPop; i++) {
        each_size_subPop[i] = quo;
        if(i < rem) each_size_subPop[i]++;
        recv_size_subPop[i] = each_size_subPop[i];
    }
    disp_size_subPop[0] = 0;
    for(i = 1; i < mpi_size_subPop; i++) {
        disp_size_subPop[i] = disp_size_subPop[i - 1] + recv_size_subPop[i - 1];
    }
    //
    (*nPop_mine) = each_size_subPop[mpi_rank_subPop];
    int min_num;
    MPI_Allreduce(nPop_mine, &min_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(min_num == 0) {
        if(0 == mpi_rank)
            printf("%s: There are too many CPUs, at least one rank has 0 individual, exiting...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MPI_TOOMANY);
    }
    (*nPop_exchange) = 0;
    (*n_weights_mine) = (*nPop_mine);
    if(mpi_rank_subPop)
        (*n_weights_left) = each_size_subPop[mpi_rank_subPop - 1];
    else
        (*n_weights_left) = 0;
    if(mpi_rank_subPop < mpi_size_subPop - 1)
        (*n_weights_right) = each_size_subPop[mpi_rank_subPop + 1];
    else
        (*n_weights_right) = 0;
    (*n_neighbor_left) = (*n_weights_left);
    (*n_neighbor_right) = (*n_weights_right);
    //
    load_weights();//////////////////////////////////////////////////////////////////////////
    //
    memcpy(weights_mine, &weights_all[disp_size_subPop[mpi_rank_subPop] * nObj], (*n_weights_mine) * nObj * sizeof(double));
    memcpy(posFactor_mine, &diver_var_store_all[disp_size_subPop[mpi_rank_subPop] * nDim],
           (*n_weights_mine) * nDim * sizeof(double));
    if((*n_weights_left)) {
        memcpy(weights_left, &weights_all[disp_size_subPop[mpi_rank_subPop - 1] * nObj], (*n_weights_left) * nObj * sizeof(double));
        memcpy(posFactor_left, &diver_var_store_all[disp_size_subPop[mpi_rank_subPop - 1] * nDim],
               (*n_weights_left) * nDim * sizeof(double));
    }
    if((*n_weights_right)) {
        memcpy(weights_right, &weights_all[disp_size_subPop[mpi_rank_subPop + 1] * nObj], (*n_weights_right) * nObj * sizeof(double));
        memcpy(posFactor_right, &diver_var_store_all[disp_size_subPop[mpi_rank_subPop + 1] * nDim],
               (*n_weights_right) * nDim * sizeof(double));
    }
    //
    return;
}

void localInitialization_ND()
{
    int color_pop = st_MPI_p.color_pop;
    int* nPop_all = st_MPI_p.nPop_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    int* nPop_mine = &st_global_p.nPop_mine;
    //
    int i;
    int quo;
    int rem;

    quo = nPop_all[color_pop] / mpi_size_subPop;
    rem = nPop_all[color_pop] % mpi_size_subPop;
    for(i = 0; i < mpi_size_subPop; i++) {
        each_size_subPop[i] = quo;
        if(i < rem) each_size_subPop[i]++;
        recv_size_subPop[i] = each_size_subPop[i];
    }
    disp_size_subPop[0] = 0;
    for(i = 1; i < mpi_size_subPop; i++) {
        disp_size_subPop[i] = disp_size_subPop[i - 1] + recv_size_subPop[i - 1];
    }
    //
    (*nPop_mine) = each_size_subPop[mpi_rank_subPop];
    int min_num;
    MPI_Allreduce(nPop_mine, &min_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(min_num == 0) {
        if(0 == mpi_rank)
            printf("%s: There are too many CPUs, at least one rank has 0 individual, exiting...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_MPI_TOOMANY);
    }

    return;
}