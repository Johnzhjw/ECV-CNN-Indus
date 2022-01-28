#include "global.h"
#include <math.h>
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//MPI exchange
//com_step strategy
void exchangePopGlobal_vonNeumann()
{
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int n_neighbor_left = st_pop_comm_p.n_neighbor_left;
    int n_neighbor_right = st_pop_comm_p.n_neighbor_right;
    int num_trail_per_gen = st_global_p.num_trail_per_gen;
    double* xCurrent = st_pop_evo_cur.var;
    double* rCurrent = st_qu_p.rot_angle_cur;
    double* fCurrent = st_pop_evo_cur.obj;
    double* xExchange = st_pop_comm_p.var_exchange;
    double* rExchange = st_pop_comm_p.rot_angle_exchange;
    double* fExchange = st_pop_comm_p.obj_exchange;
    double* weights = st_decomp_p.weights_mine;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size = st_MPI_p.mpi_size;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    //
    double* tmp_fit = (double*)malloc(nPop_mine * sizeof(double));
    int*    tmp_ind = (int*)malloc(nPop_mine * sizeof(int));
    for(int i = 0; i < nPop_mine; i++) {
        tmp_fit[i] = fitnessFunction(&fCurrent[i * nObj], &weights[i * nObj]);
        tmp_ind[i] = i;
    }
    minfastsort(tmp_fit, tmp_ind, nPop_mine, num_trail_per_gen);
    double* tmp_xExchange = (double*)malloc(num_trail_per_gen * nDim * sizeof(double));
    double* tmp_rExchange = (double*)malloc(num_trail_per_gen * nDim * sizeof(double));
    double* tmp_fExchange = (double*)malloc(num_trail_per_gen * nObj * sizeof(double));
    for(int i = 0; i < num_trail_per_gen; i++) {
        memcpy(&tmp_xExchange[i * nDim], &xCurrent[tmp_ind[i] * nDim], nDim * sizeof(double));
        if(Qubits_angle_opt_tag == FLAG_ON)
            memcpy(&tmp_rExchange[i * nDim], &rCurrent[tmp_ind[i] * nDim], nDim * sizeof(double));
        memcpy(&tmp_fExchange[i * nObj], &fCurrent[tmp_ind[i] * nObj], nObj * sizeof(double));
    }
    //
    int step = (int)sqrt((double)mpi_size);
    int ind_u = (mpi_rank - step + mpi_size) % mpi_size;
    int ind_d = (mpi_rank + step) % mpi_size;
    int ind_l = (mpi_rank - 1 + mpi_size) % mpi_size;
    int ind_r = (mpi_rank + 1) % mpi_size;

    //	double rndt=rnd_uni(&rnd_uni_init);
    //	MPI_Bcast(&rndt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    int collect_from;
    int collect_to;
    //up->down
    if(st_global_p.generation % 40 < 10) {   // if(rndt<0.25)
        collect_to = ind_d;
        collect_from = ind_u;
    }
    //down->up
    else if(st_global_p.generation % 40 < 20) {   // if(rndt<0.5)
        collect_to = ind_u;
        collect_from = ind_d;
    }
    //left->right
    else if(st_global_p.generation % 40 < 30) {   // if(rndt<0.75)
        collect_to = ind_r;
        collect_from = ind_l;
    }
    //right->left
    else {
        collect_to = ind_l;
        collect_from = ind_r;
    }
    int tmp_n;
    MPI_Sendrecv(&num_trail_per_gen, 1, MPI_INT, collect_to, 0,
                 &tmp_n, 1, MPI_INT, collect_from, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(tmp_xExchange, num_trail_per_gen * nDim, MPI_DOUBLE, collect_to, 1,
                 &xExchange[st_global_p.nPop_exchange * nDim], tmp_n * nDim, MPI_DOUBLE, collect_from, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(Qubits_angle_opt_tag == FLAG_ON)
        MPI_Sendrecv(tmp_rExchange, num_trail_per_gen * nDim, MPI_DOUBLE, collect_to, 1,
                     &rExchange[st_global_p.nPop_exchange * nDim], tmp_n * nDim, MPI_DOUBLE, collect_from, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(tmp_fExchange, num_trail_per_gen * nObj, MPI_DOUBLE, collect_to, 2,
                 &fExchange[st_global_p.nPop_exchange * nObj], tmp_n * nObj, MPI_DOUBLE, collect_from, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    st_global_p.nPop_exchange += tmp_n;
    //
    free(tmp_fit);
    free(tmp_ind);
    free(tmp_xExchange);
    free(tmp_rExchange);
    free(tmp_fExchange);
    //
    memcpy(&xCurrent[(nPop_mine + n_neighbor_left + n_neighbor_right)*nDim], xExchange,
           st_global_p.nPop_exchange * nDim * sizeof(double));
    if(Qubits_angle_opt_tag == FLAG_ON)
        memcpy(&rCurrent[(nPop_mine + n_neighbor_left + n_neighbor_right)*nDim], rExchange,
               st_global_p.nPop_exchange * nDim * sizeof(double));
    memcpy(&fCurrent[(nPop_mine + n_neighbor_left + n_neighbor_right)*nObj], fExchange,
           st_global_p.nPop_exchange * nObj * sizeof(double));
    //
    return;
}

void exchangePopBestGlobal_vonNeumann()
{
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    double* xCurrent = st_pop_evo_cur.var;
    double* fCurrent = st_pop_evo_cur.obj;
    double* utility = st_utility_p.utility;
    double* weights = st_decomp_p.weights_mine;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_size = st_MPI_p.mpi_size;
    //
    double bestIndFit[2];
    int bestIndIndex[2];
    bestIndFit[0] = INF_DOUBLE;
    bestIndFit[1] = -INF_DOUBLE;
    bestIndIndex[0] = bestIndIndex[1] = -1;
    double f1, f2;
    double* tmp_xExchange = (double*)malloc(2 * nDim * sizeof(double));
    double* tmp_fExchange = (double*)malloc(2 * nObj * sizeof(double));
    //
    for(int i = 0; i < nPop_mine; i++) {
        f1 = fitnessFunction(&fCurrent[i * nObj], &weights[i * nObj]);
        if(f1 < bestIndFit[0]) {
            bestIndFit[0] = f1;
            bestIndIndex[0] = i;
        }
        if(utility[i] > bestIndFit[1]) {
            bestIndFit[1] = utility[i];
            bestIndIndex[1] = i;
        }
    }
    //
    int step = sqrt((double)mpi_size);
    int ind_u = (mpi_rank - step + mpi_size) % mpi_size;
    int ind_d = (mpi_rank + step) % mpi_size;
    int ind_l = (mpi_rank - 1 + mpi_size) % mpi_size;
    int ind_r = (mpi_rank + 1) % mpi_size;
    //
    double rndt = pointer_gen_rand();
    MPI_Bcast(&rndt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    //
    MPI_Sendrecv(&xCurrent[bestIndIndex[0] * nDim], nDim, MPI_DOUBLE, collect_to, 1,
                 &tmp_xExchange[0], nDim, MPI_DOUBLE, collect_from, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&fCurrent[bestIndIndex[0] * nObj], nObj, MPI_DOUBLE, collect_to, 2,
                 &tmp_fExchange[0], nObj, MPI_DOUBLE, collect_from, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&xCurrent[bestIndIndex[1] * nDim], nDim, MPI_DOUBLE, collect_to, 3,
                 &tmp_xExchange[nDim], nDim, MPI_DOUBLE, collect_from, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&fCurrent[bestIndIndex[1] * nObj], nObj, MPI_DOUBLE, collect_to, 4,
                 &tmp_fExchange[nObj], nObj, MPI_DOUBLE, collect_from, 4,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //update pop
    int* tIndex = (int*)malloc(nPop * sizeof(int));
    for(int j = 0; j < nPop_mine; j++) tIndex[j] = j;
    shuffle(tIndex, nPop_mine);
    int count = 0;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < nPop_mine; j++) {
            if(count >= 1) break;
            int realInd;
            realInd = tIndex[j];
            f1 = fitnessFunction(&tmp_fExchange[i * nObj], &weights[realInd * nObj]);
            f2 = fitnessFunction(&fCurrent[realInd * nObj], &weights[realInd * nObj]);
            if(f1 < f2) {
                memcpy(&xCurrent[realInd * nDim], &tmp_xExchange[i * nDim], nDim * sizeof(double));
                memcpy(&fCurrent[realInd * nObj], &tmp_fExchange[i * nObj], nObj * sizeof(double));
                count++;
                int iii;
                for(iii = 0; iii < st_pop_comm_p.iUpdt && st_pop_comm_p.updtIndx[iii] != realInd; iii++) {}
                if(iii >= st_pop_comm_p.iUpdt) {
                    st_pop_comm_p.updtIndx[st_pop_comm_p.iUpdt++] = realInd;
                }
            }
        }
    }
    //
    free(tmp_xExchange);
    free(tmp_fExchange);
    free(tIndex);
    //
    return;
}

void get_xBestWithinObj()
{
    if(0 == st_MPI_p.color_pop) return;

    struct D_I {
        double val;
        int id;
    };

    D_I info_in, info_out;

    int no_obj = st_MPI_p.color_pop - 1;
    info_in.val = st_pop_best_p.obj_best[no_obj];
    info_in.id = st_MPI_p.mpi_rank_pop;

    MPI_Allreduce(&info_in, &info_out, 1, MPI_DOUBLE_INT, MPI_MINLOC, st_MPI_p.comm_pop);

    int best_rank = info_out.id;
    memcpy(st_pop_best_p.var_best_exchange, st_pop_best_p.var_best, st_global_p.nDim * sizeof(double));
    memcpy(st_pop_best_p.obj_best_exchange, st_pop_best_p.obj_best, st_global_p.nObj * sizeof(double));
    MPI_Bcast(st_pop_best_p.var_best_exchange, st_global_p.nDim, MPI_DOUBLE, best_rank, st_MPI_p.comm_pop);
    MPI_Bcast(st_pop_best_p.obj_best_exchange, st_global_p.nObj, MPI_DOUBLE, best_rank, st_MPI_p.comm_pop);

    // 	if(strct_MPI_info.mpi_rank_population==0)
    // 	{
    // 		printf("ID: %d\n",best_rank);
    // 		for(int i=0;i<strct_global_paras.nDim;i++)
    // 			printf("%lf\t",xBest_exchange[i]);
    // 		printf("\n");
    // 		for(int i=0;i<strct_global_paras.nObj;i++)
    // 			printf("%lf\t",xBFitness[i]);
    // 		printf("\n");
    // 		for(int i=0;i<strct_global_paras.nObj;i++)
    // 			printf("%lf\t",xBFitness_exchange[i]);
    // 		printf("\n");
    // 	}
    return;
}

void exchange_xBestWithinGroup()
{
    double p_rnd = pointer_gen_rand();
    MPI_Bcast(&p_rnd, 1, MPI_DOUBLE, 0, st_MPI_p.comm_subPop);
    if(st_MPI_p.mpi_size_subPop > 1) {
        int tmp = (int)((double)st_global_p.generation / st_global_p.generatMax * (st_MPI_p.mpi_size_subPop - 1));
        int com_step = tmp % (st_MPI_p.mpi_size_subPop - 1) + 1;
        int ind_l = (st_MPI_p.mpi_rank_subPop - com_step + st_MPI_p.mpi_size_subPop) % st_MPI_p.mpi_size_subPop;
        int ind_r = (st_MPI_p.mpi_rank_subPop + com_step) % st_MPI_p.mpi_size_subPop;

        int collect_to;
        int collect_from;

        if(p_rnd < 0.5) {   //from left to right
            collect_to = ind_r;
            collect_from = ind_l;
        } else { //from right to left
            collect_to = ind_l;
            collect_from = ind_r;
        }
        MPI_Sendrecv(st_pop_best_p.var_best, st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                     st_pop_best_p.var_best_exchange, st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_pop_best_p.obj_best, st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                     st_pop_best_p.obj_best_exchange, st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                     st_MPI_p.comm_subPop, MPI_STATUS_IGNORE);
    } else {
        printf("%s:NOT ENOUGH PROCS IN GROUP (LESS THAN 2), EXITING...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_ENOUGH_MPI_IN_GROUP);
    }
    return;
}

void exchange_xBestWithinObj()
{
    double p_rnd = pointer_gen_rand();
    MPI_Bcast(&p_rnd, 1, MPI_DOUBLE, 0, st_MPI_p.comm_pop);
    if(st_MPI_p.mpi_size_pop > 1) {
        int tmp = (int)((double)st_global_p.generation / st_global_p.generatMax * (st_MPI_p.mpi_size_pop - 1));
        int com_step = tmp % (st_MPI_p.mpi_size_pop - 1) + 1;
        int ind_l = (st_MPI_p.mpi_rank_pop - com_step + st_MPI_p.mpi_size_pop) %
                    st_MPI_p.mpi_size_pop;
        int ind_r = (st_MPI_p.mpi_rank_pop + com_step) % st_MPI_p.mpi_size_pop;

        int collect_from;
        int collect_to;

        if(p_rnd < 0.5) {   //from left to right
            collect_to = ind_r;
            collect_from = ind_l;
        } else { //from right to left
            collect_to = ind_l;
            collect_from = ind_r;
        }
        MPI_Sendrecv(st_pop_best_p.var_best, st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                     st_pop_best_p.var_best_exchange, st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                     st_MPI_p.comm_pop, MPI_STATUS_IGNORE);
        MPI_Sendrecv(st_pop_best_p.obj_best, st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                     st_pop_best_p.obj_best_exchange, st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                     st_MPI_p.comm_pop, MPI_STATUS_IGNORE);
    } else {
        printf("%s:NOT ENOUGH PROCS IN OBJ (LESS THAN 2), EXITING...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_ENOUGH_MPI_IN_POP);
    }
    return;
}

void exchangeInfo_ND()
{
    MPI_Barrier(MPI_COMM_WORLD); //if (strct_MPI_info.mpi_rank == 0)printf("line10.75.\n");

    collectNDArchiveEx();

    if(st_MPI_p.color_master_subPop) {
        MPI_Bcast(&st_archive_p.cnArchEx, 1, MPI_INT, 0, st_MPI_p.comm_master_subPop_globalScope);
        MPI_Bcast(st_archive_p.var_Ex, st_archive_p.cnArchEx * st_global_p.nDim, MPI_DOUBLE, 0,
                  st_MPI_p.comm_master_subPop_globalScope);
        MPI_Bcast(st_archive_p.obj_Ex, st_archive_p.cnArchEx * st_global_p.nObj, MPI_DOUBLE, 0,
                  st_MPI_p.comm_master_subPop_globalScope);
        MPI_Bcast(st_archive_p.dens_Ex, st_archive_p.cnArchEx, MPI_DOUBLE, 0,
                  st_MPI_p.comm_master_subPop_globalScope);
        MPI_Bcast(st_archive_p.rank_Ex, st_archive_p.cnArchEx, MPI_INT, 0,
                  st_MPI_p.comm_master_subPop_globalScope);

        if(st_MPI_p.color_obj) {
            st_repo_p.nRep = 0;
            //memcpy(&repository[strct_repo_info.nRep * strct_global_paras.nDim], archive, nArch_sep * strct_global_paras.nDim * sizeof(double));
            //memcpy(&repositFit[strct_repo_info.nRep * strct_global_paras.nObj], archFit, nArch_sep * strct_global_paras.nObj * sizeof(double));
            //strct_repo_info.nRep = nArch_sep;
            memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var_Ex,
                   st_archive_p.nArch * st_global_p.nDim * sizeof(double));
            memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj_Ex,
                   st_archive_p.nArch * st_global_p.nObj * sizeof(double));
            st_repo_p.nRep += st_archive_p.nArch;
            refineRepository_generateArchive_sub();
        } else {
            memcpy(st_archive_p.var, st_archive_p.var_Ex, st_archive_p.nArch * st_global_p.nDim * sizeof(double));
            memcpy(st_archive_p.obj, st_archive_p.obj_Ex, st_archive_p.nArch * st_global_p.nObj * sizeof(double));
            memcpy(st_archive_p.dens, st_archive_p.dens_Ex, st_archive_p.nArch * sizeof(double));
            memcpy(st_archive_p.rank, st_archive_p.rank_Ex, st_archive_p.nArch * sizeof(int));
            st_archive_p.cnArch = st_archive_p.nArch;
        }
    }

    return;
}

void exchangeInfo()
{
    collectDecompositionArchive(st_ctrl_p.algo_mech_type);
    //
    //int the_rank;
    //int the_size;
    MPI_Comm the_comm;
    //
    //the_rank = strct_MPI_info.mpi_rank_master_species_globalScope;
    //the_size = strct_MPI_info.mpi_size_master_species_globalScope;
    the_comm = st_MPI_p.comm_master_subPop_globalScope;
    //
    if(st_MPI_p.color_master_subPop) {
        MPI_Bcast(st_archive_p.var, st_global_p.nPop * st_global_p.nDim, MPI_DOUBLE, 0, the_comm);
        MPI_Bcast(st_archive_p.obj, st_global_p.nPop * st_global_p.nObj, MPI_DOUBLE, 0, the_comm);

        memcpy(st_pop_evo_cur.var, st_archive_p.var, st_global_p.nPop * st_global_p.nDim * sizeof(double));
        memcpy(st_pop_evo_cur.obj, st_archive_p.obj, st_global_p.nPop * st_global_p.nObj * sizeof(double));
    }
    //
    return;
}

void exchangeInfo_DPCCMOEA()
{
    st_global_p.nPop_exchange = 0;
    if(st_global_p.generation % 1 == 0) {
        exchangePopGlobal_vonNeumann();
        exchangePopBestGlobal_vonNeumann();
    }

    if(st_global_p.generation == 0 || st_global_p.tag_strct_updated == FLAG_ON) {
        transfer_x_neighbor();
        st_global_p.tag_strct_updated = FLAG_OFF;
    } else {
        transfer_x_neighbor_updated();
    }

    if(st_MPI_p.color_obj) {
        //if (generation % CHECK_GAP_BEST == CHECK_GAP_BEST - 1)
        if(st_global_p.generation % 1 == 0) {
            get_xBestWithinObj();
            //update_xBest_history_after_exchange();
            update_xBest(UPDATE_GIVEN, 1, NULL, st_pop_best_p.var_best_exchange, st_pop_best_p.obj_best_exchange, NULL);
            update_xBest_history(UPDATE_GIVEN, 1, NULL, st_pop_best_p.var_best_exchange, st_pop_best_p.obj_best_exchange, NULL);
        }
    }

    return;
}

//void exchangeInformation()
//{
//	strct_global_paras.nPop_exchange = 0;
//
//	// 	if(strct_global_paras.generation<strct_global_paras.generatMax/3)
//	// 		exchangePopGlobal();
//	// 	else if(strct_global_paras.generation<2*strct_global_paras.generatMax/3)
//	// 		exchangePopWithinObj();
//	// 	else
//	// 		exchangePopWithinGroup();
//
//	//	exchangePopGlobal_vonNeumann();
//
//	//	exchangePopBestGlobal_vonNeumann();
//
//	//	transfer_x_neighbor();
//
//	if (strct_MPI_info.color_population_property_2 && strct_MPI_info.color_master_species) {
//
//		get_xBestWithinObj();
//		update_xBest_history_after_exchange();
//
//		// 		update_xBest_exchange();
//		// 		exchange_xBestWithinGroup();
//		// 		exchange_xBestWithinObj();
//	}
//
//	//output
//	// 	for(i=0;i<strct_MPI_info.mpi_size;i++)
//	// 	{
//	// 		MPI_Barrier(MPI_COMM_WORLD);
//	// 		if(strct_MPI_info.mpi_rank==i)
//	// 		{
//	// 			printf("strct_MPI_info.mpi_rank\t strct_global_paras.nPop_mine\t nPop_exchg\n");
//	// 			printf("%d\t\t %d\t\t %d\n",strct_MPI_info.mpi_rank,strct_global_paras.nPop_mine,strct_global_paras.nPop_exchange);
//	// 			printf("%lf\n",fCurrent_exchange[1]);
//	// 		}
//	// 		MPI_Barrier(MPI_COMM_WORLD);
//	// 	}
//	// 	MPI_Barrier(MPI_COMM_WORLD);
//
//	if (strct_global_paras.generation % CHECK_GAP_EXCH == 0) {
//		//		generateBestCombinations();
//		//		generateBestPopulationOne();
//	}
//	//	if(strct_MPI_info.mpi_rank==0) showBestCombinationAndPopulation();
//	//	generateBestPopulations();
//	/*	int countSHOW=0;
//	for(int i=0;i<strct_MPI_info.mpi_size;i++)
//	{
//	if(strct_MPI_info.mpi_rank==0)
//	{
//	showBestCombinationAndPopulation();
//	}
//	if(countSHOW++>7)
//	break;
//	MPI_Barrier(MPI_COMM_WORLD);
//	}*/
//
//	return;
//}