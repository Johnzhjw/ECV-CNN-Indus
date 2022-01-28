#include "global.h"
#include <math.h>
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

void update_recv_disp_simp(int* num, int n, int l)
{
    int size = n;
    int i;
    int len = l;
    for(i = 0; i < len; i++) {
        st_MPI_p.recv_size[i] = num[i] * size;
    }
    st_MPI_p.disp_size[0] = 0;
    for(i = 1; i < len; i++) {
        st_MPI_p.disp_size[i] = st_MPI_p.disp_size[i - 1] + st_MPI_p.recv_size[i - 1];
    }
    return;
}

void update_recv_disp(int* _each, int _n, int _l, int* _recv, int* _disp)
{
    int size = _n;
    int len = _l;
    int i;
    for(i = 0; i < len; i++) {
        _recv[i] = _each[i] * size;
    }
    _disp[0] = 0;
    for(i = 1; i < len; i++) {
        _disp[i] = _disp[i - 1] + _recv[i - 1];
    }
    return;
}

void transfer_x_neighbor()
{
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int n_neighbor_left = st_pop_comm_p.n_neighbor_left;
    int n_neighbor_right = st_pop_comm_p.n_neighbor_right;
    double* var = st_pop_evo_cur.var;
    double* rot = st_qu_p.rot_angle_cur;
    double* obj = st_pop_evo_cur.obj;
    double* var_left = st_pop_comm_p.var_left;
    double* rot_left = st_pop_comm_p.rot_angle_left;
    double* obj_left = st_pop_comm_p.obj_left;
    double* var_right = st_pop_comm_p.var_right;
    double* rot_right = st_pop_comm_p.rot_angle_right;
    double* obj_right = st_pop_comm_p.obj_right;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    //
    int index_l, index_r;
    if(st_MPI_p.mpi_rank_subPop > 0)
        index_l = st_MPI_p.mpi_rank_subPop - 1;
    else
        index_l = MPI_PROC_NULL;
    if(st_MPI_p.mpi_rank_subPop < st_MPI_p.mpi_size_subPop - 1)
        index_r = st_MPI_p.mpi_rank_subPop + 1;
    else
        index_r = MPI_PROC_NULL;
    //////////////////////////////////////////////////////////////////////////
    st_pop_comm_p.iUpdt_recv_left = 0;
    st_pop_comm_p.iUpdt_recv_right = 0;
    // left to right
    MPI_Sendrecv(var, nPop_mine * nDim, MPI_DOUBLE, index_r, 1,
                 var_left, n_neighbor_left * nDim, MPI_DOUBLE, index_l, 1,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(Qubits_angle_opt_tag == FLAG_ON)
        MPI_Sendrecv(rot, nPop_mine * nDim, MPI_DOUBLE, index_r, 1,
                     rot_left, n_neighbor_left * nDim, MPI_DOUBLE, index_l, 1,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(obj, nPop_mine * nObj, MPI_DOUBLE, index_r, 11,
                 obj_left, n_neighbor_left * nObj, MPI_DOUBLE, index_l, 11,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    // right to left
    MPI_Sendrecv(var, nPop_mine * nDim, MPI_DOUBLE, index_l, 2,
                 var_right, n_neighbor_right * nDim, MPI_DOUBLE, index_r, 2,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(Qubits_angle_opt_tag == FLAG_ON)
        MPI_Sendrecv(rot, nPop_mine * nDim, MPI_DOUBLE, index_l, 2,
                     rot_right, n_neighbor_right * nDim, MPI_DOUBLE, index_r, 2,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(obj, nPop_mine * nObj, MPI_DOUBLE, index_l, 22,
                 obj_right, n_neighbor_right * nObj, MPI_DOUBLE, index_r, 22,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    st_pop_comm_p.iUpdt = 0;
    //
    memcpy(&var[nPop_mine * nDim], var_left, n_neighbor_left * nDim * sizeof(double));
    if(Qubits_angle_opt_tag == FLAG_ON)
        memcpy(&rot[nPop_mine * nDim], rot_left, n_neighbor_left * nDim * sizeof(double));
    memcpy(&obj[nPop_mine * nObj], obj_left, n_neighbor_left * nObj * sizeof(double));
    memcpy(&var[(nPop_mine + n_neighbor_left) * nDim], var_right, n_neighbor_right * nDim * sizeof(double));
    if(Qubits_angle_opt_tag == FLAG_ON)
        memcpy(&rot[(nPop_mine + n_neighbor_left) * nDim], rot_right, n_neighbor_right * nDim * sizeof(double));
    memcpy(&obj[(nPop_mine + n_neighbor_left) * nObj], obj_right, n_neighbor_right * nObj * sizeof(double));
    //
    return;
}

void transfer_x_neighbor_updated()
{
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int n_neighbor_left = st_pop_comm_p.n_neighbor_left;
    int n_neighbor_right = st_pop_comm_p.n_neighbor_right;
    double* var = st_pop_evo_cur.var;
    double* rot = st_qu_p.rot_angle_cur;
    double* obj = st_pop_evo_cur.obj;
    double* var_send = st_pop_comm_p.var_send;
    double* rot_send = st_pop_comm_p.rot_angle_send;
    double* obj_send = st_pop_comm_p.obj_send;
    double* var_recv = st_pop_comm_p.var_recv;
    double* rot_recv = st_pop_comm_p.rot_angle_recv;
    double* obj_recv = st_pop_comm_p.obj_recv;
    double* var_left = st_pop_comm_p.var_left;
    double* rot_left = st_pop_comm_p.rot_angle_left;
    double* obj_left = st_pop_comm_p.obj_left;
    double* var_right = st_pop_comm_p.var_right;
    double* rot_right = st_pop_comm_p.rot_angle_right;
    double* obj_right = st_pop_comm_p.obj_right;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    //
    int index_l, index_r;
    if(st_MPI_p.mpi_rank_subPop > 0)
        index_l = st_MPI_p.mpi_rank_subPop - 1;
    else
        index_l = MPI_PROC_NULL;
    if(st_MPI_p.mpi_rank_subPop < st_MPI_p.mpi_size_subPop - 1)
        index_r = st_MPI_p.mpi_rank_subPop + 1;
    else
        index_r = MPI_PROC_NULL;
    //////////////////////////////////////////////////////////////////////////
    int i;
    for(i = 0; i < st_pop_comm_p.iUpdt; i++) {
        memcpy(&var_send[i * nDim], &var[st_pop_comm_p.updtIndx[i] * nDim], nDim * sizeof(double));
        if(Qubits_angle_opt_tag == FLAG_ON)
            memcpy(&rot_send[i * nDim], &rot[st_pop_comm_p.updtIndx[i] * nDim], nDim * sizeof(double));
        memcpy(&obj_send[i * nObj], &obj[st_pop_comm_p.updtIndx[i] * nObj], nObj * sizeof(double));
    }
    // left to right
    st_pop_comm_p.iUpdt_recv_left = 0;
    MPI_Sendrecv(&st_pop_comm_p.iUpdt, 1, MPI_INT, index_r, 1,
                 &st_pop_comm_p.iUpdt_recv_left, 1, MPI_INT, index_l, 1,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(st_pop_comm_p.updtIndx, st_pop_comm_p.iUpdt, MPI_INT, index_r, 11,
                 st_pop_comm_p.updtIndx_recv_left, st_pop_comm_p.iUpdt_recv_left, MPI_INT, index_l, 11,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(var_send, st_pop_comm_p.iUpdt * nDim, MPI_DOUBLE, index_r, 2,
                 var_recv, st_pop_comm_p.iUpdt_recv_left * nDim, MPI_DOUBLE, index_l, 2,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(Qubits_angle_opt_tag == FLAG_ON)
        MPI_Sendrecv(rot_send, st_pop_comm_p.iUpdt * nDim, MPI_DOUBLE, index_r, 2,
                     rot_recv, st_pop_comm_p.iUpdt_recv_left * nDim, MPI_DOUBLE, index_l, 2,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(obj_send, st_pop_comm_p.iUpdt * nObj, MPI_DOUBLE, index_r, 22,
                 obj_recv, st_pop_comm_p.iUpdt_recv_left * nObj, MPI_DOUBLE, index_l, 22,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(st_MPI_p.mpi_rank_subPop != 0)
        for(i = 0; i < st_pop_comm_p.iUpdt_recv_left; i++) {
            memcpy(&var_left[st_pop_comm_p.updtIndx_recv_left[i] * nDim], &var_recv[i * nDim], nDim * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON)
                memcpy(&rot_left[st_pop_comm_p.updtIndx_recv_left[i] * nDim], &rot_recv[i * nDim], nDim * sizeof(double));
            memcpy(&obj_left[st_pop_comm_p.updtIndx_recv_left[i] * nObj], &obj_recv[i * nObj], nObj * sizeof(double));
        }
    // right to left
    st_pop_comm_p.iUpdt_recv_right = 0;
    MPI_Sendrecv(&st_pop_comm_p.iUpdt, 1, MPI_INT, index_l, 3,
                 &st_pop_comm_p.iUpdt_recv_right, 1, MPI_INT, index_r, 3,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(st_pop_comm_p.updtIndx, st_pop_comm_p.iUpdt, MPI_INT, index_l, 33,
                 st_pop_comm_p.updtIndx_recv_right, st_pop_comm_p.iUpdt_recv_right, MPI_INT, index_r, 33,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(var_send, st_pop_comm_p.iUpdt * nDim, MPI_DOUBLE, index_l, 4,
                 var_recv, st_pop_comm_p.iUpdt_recv_right * nDim, MPI_DOUBLE, index_r, 4,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(Qubits_angle_opt_tag == FLAG_ON)
        MPI_Sendrecv(rot_send, st_pop_comm_p.iUpdt * nDim, MPI_DOUBLE, index_l, 4,
                     rot_recv, st_pop_comm_p.iUpdt_recv_right * nDim, MPI_DOUBLE, index_r, 4,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    MPI_Sendrecv(obj_send, st_pop_comm_p.iUpdt * nObj, MPI_DOUBLE, index_l, 44,
                 obj_recv, st_pop_comm_p.iUpdt_recv_right * nObj, MPI_DOUBLE, index_r, 44,
                 st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    if(st_MPI_p.mpi_rank_subPop != st_MPI_p.mpi_size_subPop - 1)
        for(i = 0; i < st_pop_comm_p.iUpdt_recv_right; i++) {
            memcpy(&var_right[st_pop_comm_p.updtIndx_recv_right[i] * nDim], &var_recv[i * nDim], nDim * sizeof(double));
            if(Qubits_angle_opt_tag == FLAG_ON)
                memcpy(&rot_right[st_pop_comm_p.updtIndx_recv_right[i] * nDim], &rot_recv[i * nDim], nDim * sizeof(double));
            memcpy(&obj_right[st_pop_comm_p.updtIndx_recv_right[i] * nObj], &obj_recv[i * nObj], nObj * sizeof(double));
        }
    st_pop_comm_p.iUpdt = 0;
    //
    memcpy(&var[nPop_mine * nDim], var_left, n_neighbor_left * nDim * sizeof(double));
    if(Qubits_angle_opt_tag == FLAG_ON)
        memcpy(&rot[nPop_mine * nDim], rot_left, n_neighbor_left * nDim * sizeof(double));
    memcpy(&obj[nPop_mine * nObj], obj_left, n_neighbor_left * nObj * sizeof(double));
    memcpy(&var[(nPop_mine + n_neighbor_left) * nDim], var_right, n_neighbor_right * nDim * sizeof(double));
    if(Qubits_angle_opt_tag == FLAG_ON)
        memcpy(&rot[(nPop_mine + n_neighbor_left) * nDim], rot_right, n_neighbor_right * nDim * sizeof(double));
    memcpy(&obj[(nPop_mine + n_neighbor_left) * nObj], obj_right, n_neighbor_right * nObj * sizeof(double));
    //
    return;
}

void scatter_evaluation_gather()
{
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int algo_mech_type = st_ctrl_p.algo_mech_type;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    MPI_Comm comm_subPop = st_MPI_p.comm_subPop;
    double* osp_var = st_pop_evo_offspring.var;
    double* osp_obj = st_pop_evo_offspring.obj;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    int type_test = st_ctrl_p.type_test;
    int type_dim_convert = st_ctrl_p.type_dim_convert;
    char* testInstance = st_global_p.testInstance;
    int tag_gather_after_evaluate = st_ctrl_p.tag_gather_after_evaluate;
    //
    if(algo_mech_type == DECOMPOSITION ||
       algo_mech_type == NONDOMINANCE) {
        //for each group, scatter the current population to processes
        update_recv_disp(each_size_subPop, nDim, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Scatterv(osp_var, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                     repo_var, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                     0, comm_subPop);
    }

    int i;
    for(i = 0; i < nPop_mine; i++) {
        if((type_test == MY_TYPE_LeNet || type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
           type_dim_convert == DIM_CONVERT_CNN) {
            double tmp[DIM_LeNet];
            convertVar_CNN(&repo_var[i * nDim], tmp);
            EMO_evaluate_problems(testInstance, tmp, &repo_obj[i * nObj], DIM_LeNet, 1, nObj);
        } else {
            EMO_evaluate_problems(testInstance, &repo_var[i * nDim], &repo_obj[i * nObj], nDim, 1, nObj);
        }
    }

    if(tag_gather_after_evaluate == FLAG_ON) {
        if(algo_mech_type == LOCALIZATION) {
            memcpy(osp_var, repo_var, recv_size_subPop[mpi_rank_subPop] * nDim * sizeof(double));
        } else if(algo_mech_type == DECOMPOSITION ||
                  algo_mech_type == NONDOMINANCE) {
            update_recv_disp(each_size_subPop, nDim, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
            MPI_Gatherv(repo_var, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                        osp_var, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                        0, comm_subPop);
        }
    }
    if(algo_mech_type == DECOMPOSITION ||
       algo_mech_type == NONDOMINANCE) {
        //for each group, collect the current population to rank 0 process
        update_recv_disp(each_size_subPop, nObj, mpi_size_subPop, recv_size_subPop, disp_size_subPop);
        MPI_Gatherv(repo_obj, recv_size_subPop[mpi_rank_subPop], MPI_DOUBLE,
                    osp_obj, recv_size_subPop, disp_size_subPop, MPI_DOUBLE,
                    0, comm_subPop);
    }

    return;
}
