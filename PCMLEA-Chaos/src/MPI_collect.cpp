#include "global.h"
#include <math.h>
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

void collectNDArchive()
{
    if(st_MPI_p.color_master_subPop) {
        int my_tag; //printf("%d\n", strct_MPI_info.mpi_size_master_species_globalScope);
        int collect_step = 1;
        int collect_to;
        int collect_from;
        if(st_MPI_p.color_obj) {
            st_global_p.trans_size = st_archive_p.nArch_sub;
        } else {
            st_global_p.trans_size = st_archive_p.nArch;
        }
        while(collect_step < st_MPI_p.mpi_size_master_subPop_globalScope) {
            if(st_MPI_p.mpi_rank_master_subPop_globalScope % (2 * collect_step) == 0) {
                my_tag = 1;
            } else if(st_MPI_p.mpi_rank_master_subPop_globalScope % collect_step == 0) {
                my_tag = 0;
            } else {
                my_tag = -1;    //if(strct_MPI_info.mpi_rank==0)printf("line1001.\n");
            }

            if(my_tag != -1) {
                collect_to = st_MPI_p.mpi_rank_master_subPop_globalScope - st_MPI_p.mpi_rank_master_subPop_globalScope %
                             (2 * collect_step);
                collect_from = collect_to + collect_step;

                if(collect_from < st_MPI_p.mpi_size_master_subPop_globalScope) {
                    if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_from) {
                        MPI_Send(&st_global_p.trans_size, 1, MPI_INT, collect_to, 0, st_MPI_p.comm_master_subPop_globalScope);
                        MPI_Send(st_archive_p.var, st_global_p.trans_size * st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                                 st_MPI_p.comm_master_subPop_globalScope);
                        MPI_Send(st_archive_p.obj, st_global_p.trans_size * st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                                 st_MPI_p.comm_master_subPop_globalScope);
                    }
                    if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_to) {
                        MPI_Recv(&st_repo_p.nRep, 1, MPI_INT, collect_from, 0, st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
                        MPI_Recv(st_repo_p.var, st_repo_p.nRep * st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                                 st_MPI_p.comm_master_subPop_globalScope,
                                 MPI_STATUS_IGNORE);
                        MPI_Recv(st_repo_p.obj, st_repo_p.nRep * st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                                 st_MPI_p.comm_master_subPop_globalScope,
                                 MPI_STATUS_IGNORE);
                        memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var,
                               st_global_p.trans_size * st_global_p.nDim * sizeof(double));
                        memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj,
                               st_global_p.trans_size * st_global_p.nObj * sizeof(double));
                        st_repo_p.nRep += st_global_p.trans_size;
                        if(st_repo_p.nRep > st_archive_p.nArch) {
                            refineRepository_generateND(st_archive_p.var, st_archive_p.obj, st_archive_p.dens,
                                                        st_archive_p.rank, NULL, st_archive_p.cnArch, st_archive_p.nArch);
                            st_global_p.trans_size = st_archive_p.nArch;
                        } else {
                            refineRepository_generateND(st_archive_p.var, st_archive_p.obj, st_archive_p.dens,
                                                        st_archive_p.rank, NULL, st_archive_p.cnArch, st_repo_p.nRep);
                            st_global_p.trans_size = st_repo_p.nRep;
                        }
                    }
                }
            }
            MPI_Barrier(st_MPI_p.comm_master_subPop_globalScope);//if(strct_MPI_info.mpi_rank==0)printf("line1002.\n");
            collect_step *= 2;
        } //if(strct_MPI_info.mpi_rank==0)printf("line102\n");
    }

    //strct_MPI_info.mpi_rank==0 has the final ND

    /*	int dest_rank=0;

    int tmp_color=strct_MPI_info.color_population_property_2;
    MPI_Bcast(&tmp_color,1,MPI_INT,0,MPI_COMM_WORLD);
    if(tmp_color)
    {
    int i;
    for(i=0;i<strct_MPI_info.mpi_size;i++)
    {
    tmp_color=strct_MPI_info.color_population_property_2;
    MPI_Bcast(&tmp_color,1,MPI_INT,i,MPI_COMM_WORLD);
    if(!tmp_color)break;
    }
    dest_rank=i;
    if(strct_MPI_info.mpi_rank==0)
    {
    MPI_Send(&strct_archive_info.cnArch,1,MPI_INT,dest_rank,0,MPI_COMM_WORLD);
    MPI_Send(archive,strct_archive_info.cnArch*strct_global_paras.nDim,MPI_DOUBLE,dest_rank,1,MPI_COMM_WORLD);
    MPI_Send(archFit,strct_archive_info.cnArch*strct_global_paras.nObj,MPI_DOUBLE,dest_rank,2,MPI_COMM_WORLD);
    }
    if(strct_MPI_info.mpi_rank==dest_rank)
    {
    MPI_Recv(&strct_archive_info.cnArch,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(archive,strct_archive_info.cnArch*strct_global_paras.nDim,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(archFit,strct_archive_info.cnArch*strct_global_paras.nObj,MPI_DOUBLE,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    }
    */

    // 	//archive
    // 	int tmp_rank=strct_MPI_info.mpi_rank_population_2;
    // 	MPI_Bcast(&tmp_rank,1,MPI_INT,dest_rank,MPI_COMM_WORLD);
    // 	if(strct_MPI_info.color_population_property_2==0)
    // 	{
    // 		if(strct_MPI_info.mpi_rank==dest_rank)
    // 		{
    // 			memcpy(archive,archiveEx,strct_archive_info.cnArchEx*strct_global_paras.nDim*sizeof(double));
    // 			memcpy(archFit,archFitEx,strct_archive_info.cnArchEx*strct_global_paras.nObj*sizeof(double));
    // 			strct_archive_info.cnArch=strct_archive_info.cnArchEx;
    // 		}
    // 		MPI_Bcast(&strct_archive_info.cnArch,1,MPI_INT,tmp_rank,strct_MPI_info.comm_population_2);
    // 		MPI_Bcast(archive,strct_archive_info.cnArch*strct_global_paras.nDim,MPI_DOUBLE,tmp_rank,strct_MPI_info.comm_population_2);
    // 		MPI_Bcast(archFit,strct_archive_info.cnArch*strct_global_paras.nObj,MPI_DOUBLE,tmp_rank,strct_MPI_info.comm_population_2);
    //
    // 		memcpy(repository,archive,strct_archive_info.cnArch*strct_global_paras.nDim*sizeof(double));
    // 		memcpy(repositFit,archFit,strct_archive_info.cnArch*strct_global_paras.nObj*sizeof(double));
    // 		strct_repo_info.nRep=strct_archive_info.cnArch;
    // 		assignCrowdingDistanceIndexes(0,strct_repo_info.nRep-1);
    // 		memcpy(archDens,repositDens,strct_repo_info.nRep*sizeof(double));
    //
    // 		int i,j;
    // 		for(i=0;i<strct_archive_info.cnArch;i++)
    // 		{
    // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
    // 			{
    // 				archive_one_group[i*strct_global_paras.nDim+j]=
    // 					archive[i*strct_global_paras.nDim+strct_grp_info_vals.table_mine[j]];
    // 			}
    // 		}
    // 	}

    //global
    // 	if(strct_MPI_info.mpi_rank==dest_rank)
    // 	{
    // 		memcpy(archive,archiveEx,strct_archive_info.cnArchEx*strct_global_paras.nDim*sizeof(double));
    // 		memcpy(archFit,archFitEx,strct_archive_info.cnArchEx*strct_global_paras.nObj*sizeof(double));
    // 		strct_archive_info.cnArch=strct_archive_info.cnArchEx;
    // 	}

    ///////////
    /*	MPI_Bcast(&strct_archive_info.cnArch,1,MPI_INT,dest_rank,MPI_COMM_WORLD);
    MPI_Bcast(archive,strct_archive_info.cnArch*strct_global_paras.nDim,MPI_DOUBLE,dest_rank,MPI_COMM_WORLD);
    MPI_Bcast(archFit,strct_archive_info.cnArch*strct_global_paras.nObj,MPI_DOUBLE,dest_rank,MPI_COMM_WORLD);

    memcpy(repository,archiveEx,strct_archive_info.cnArchEx*strct_global_paras.nDim*sizeof(double));
    memcpy(repositFit,archFitEx,strct_archive_info.cnArchEx*strct_global_paras.nObj*sizeof(double));
    strct_repo_info.nRep=strct_archive_info.cnArchEx;
    memcpy(&repository[strct_repo_info.nRep*strct_global_paras.nDim],archiveOld,strct_archive_info.cnArchOld*strct_global_paras.nDim*sizeof(double));
    memcpy(&repositFit[strct_repo_info.nRep*strct_global_paras.nObj],archFitOld,strct_archive_info.cnArchOld*strct_global_paras.nObj*sizeof(double));
    strct_repo_info.nRep+=strct_archive_info.cnArchOld;
    refineRepository_generateND(archiveEx,archFitEx,NULL,strct_archive_info.cnArchEx,strct_archive_info.nArch);

    int i,j;
    // 	if(strct_MPI_info.color_population_property_2)
    {
    for(i=0;i<strct_archive_info.cnArchEx;i++)
    {
    for(j=0;j<(int)strct_grp_info_vals.table_mine_size;j++)
    {
    archiveEx_one_group[i*strct_global_paras.nDim+j]=
    archiveEx[i*strct_global_paras.nDim+strct_grp_info_vals.table_mine[j]];
    }
    }
    }
    // 	else
    // 	{
    // 		for(i=0;i<strct_archive_info.cnArchEx;i++)
    // 		{
    // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
    // 			{
    // 				archiveEx_one_group[i*strct_global_paras.nDim+j]=
    // 					archiveEx[i*strct_global_paras.nDim+strct_grp_info_vals.table_mine[j]];
    // 			}
    // 		}
    // 	}

    //backup
    //	memcpy(archiveOld,archiveEx,strct_archive_info.cnArchEx*strct_global_paras.nDim*sizeof(double));
    //	memcpy(archFitOld,archFitEx,strct_archive_info.cnArchEx*strct_global_paras.nObj*sizeof(double));
    //	strct_archive_info.cnArchOld=strct_archive_info.cnArchEx;
    */
    //MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void collectNDArchiveEx()
{
    if(st_MPI_p.color_obj) {
        memcpy(st_archive_p.var_Ex, st_archive_p.var,
               st_archive_p.nArch_sub * st_global_p.nDim * sizeof(double));
        memcpy(st_archive_p.obj_Ex, st_archive_p.obj,
               st_archive_p.nArch_sub * st_global_p.nObj * sizeof(double));
        st_global_p.trans_size = st_archive_p.nArch_sub; //if(strct_MPI_info.mpi_rank==0)printf("line101\n");
    } else {
        memcpy(st_archive_p.var_Ex, st_archive_p.var, st_archive_p.nArch * st_global_p.nDim * sizeof(double));
        memcpy(st_archive_p.obj_Ex, st_archive_p.obj, st_archive_p.nArch * st_global_p.nObj * sizeof(double));
        st_global_p.trans_size = st_archive_p.nArch; //if(strct_MPI_info.mpi_rank==0)printf("line101\n");
    }

    if(st_MPI_p.color_master_subPop) {
        int my_tag;
        int collect_step = 1;
        int collect_to;
        int collect_from;
        while(collect_step < st_MPI_p.mpi_size_master_subPop_globalScope) {
            if(st_MPI_p.mpi_rank_master_subPop_globalScope % (2 * collect_step) == 0) {
                my_tag = 1;
            } else if(st_MPI_p.mpi_rank_master_subPop_globalScope % collect_step == 0) {
                my_tag = 0;
            } else {
                my_tag = -1;    //if(strct_MPI_info.mpi_rank==0)printf("line1001.\n");
            }

            if(my_tag != -1) {
                collect_to = st_MPI_p.mpi_rank_master_subPop_globalScope - st_MPI_p.mpi_rank_master_subPop_globalScope %
                             (2 * collect_step);
                collect_from = collect_to + collect_step;

                if(collect_from < st_MPI_p.mpi_size_master_subPop_globalScope) {
                    if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_from) {
                        MPI_Send(&st_global_p.trans_size, 1, MPI_INT, collect_to, 0, st_MPI_p.comm_master_subPop_globalScope);
                        MPI_Send(st_archive_p.var_Ex, st_global_p.trans_size * st_global_p.nDim, MPI_DOUBLE, collect_to, 1,
                                 st_MPI_p.comm_master_subPop_globalScope);
                        MPI_Send(st_archive_p.obj_Ex, st_global_p.trans_size * st_global_p.nObj, MPI_DOUBLE, collect_to, 2,
                                 st_MPI_p.comm_master_subPop_globalScope);
                    }
                    if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_to) {
                        MPI_Recv(&st_repo_p.nRep, 1, MPI_INT, collect_from, 0, st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
                        MPI_Recv(st_repo_p.var, st_repo_p.nRep * st_global_p.nDim, MPI_DOUBLE, collect_from, 1,
                                 st_MPI_p.comm_master_subPop_globalScope,
                                 MPI_STATUS_IGNORE);
                        MPI_Recv(st_repo_p.obj, st_repo_p.nRep * st_global_p.nObj, MPI_DOUBLE, collect_from, 2,
                                 st_MPI_p.comm_master_subPop_globalScope,
                                 MPI_STATUS_IGNORE);
                        memcpy(&st_repo_p.var[st_repo_p.nRep * st_global_p.nDim], st_archive_p.var_Ex,
                               st_global_p.trans_size * st_global_p.nDim * sizeof(double));
                        memcpy(&st_repo_p.obj[st_repo_p.nRep * st_global_p.nObj], st_archive_p.obj_Ex,
                               st_global_p.trans_size * st_global_p.nObj * sizeof(double));
                        st_repo_p.nRep += st_global_p.trans_size;
                        if(st_repo_p.nRep > st_archive_p.nArch) {
                            refineRepository_generateND(st_archive_p.var_Ex, st_archive_p.obj_Ex,
                                                        st_archive_p.dens_Ex, st_archive_p.rank_Ex, NULL, st_archive_p.cnArchEx, st_archive_p.nArch);
                            st_global_p.trans_size = st_archive_p.cnArchEx;
                        } else {
                            refineRepository_generateND(st_archive_p.var_Ex, st_archive_p.obj_Ex,
                                                        st_archive_p.dens_Ex, st_archive_p.rank_Ex, NULL, st_archive_p.cnArchEx, st_repo_p.nRep);
                            st_global_p.trans_size = st_archive_p.cnArchEx;
                        }
                    }
                }
            }
            MPI_Barrier(st_MPI_p.comm_master_subPop_globalScope);//if(strct_MPI_info.mpi_rank==0)printf("line1002.\n");
            collect_step *= 2;
        } //if(strct_MPI_info.mpi_rank==0)printf("line102\n");
    }

    //MPI_Barrier(MPI_COMM_WORLD);
    return;
}

void collectViaBinTree(int the_rank, int the_size, MPI_Comm the_comm, int flag_tag)
{
    int nPop = st_global_p.nPop;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    //
    int i, j;
    if(flag_tag) {
        int my_tag;
        int collect_step = 1;
        int collect_to;
        int collect_from;
        while(collect_step < the_size) {
            if(the_rank % (2 * collect_step) == 0) {
                my_tag = 1;
            } else if(the_rank % collect_step == 0) {
                my_tag = 0;
            } else {
                my_tag = -1;
            }
            //
            if(my_tag != -1) {
                collect_to = the_rank - the_rank % (2 * collect_step);
                collect_from = collect_to + collect_step;

                if(collect_from < the_size) {
                    if(the_rank == collect_from) {
                        MPI_Send(&st_archive_p.cnArch, 1, MPI_INT, collect_to, 0, the_comm);
                        MPI_Send(arc_var, st_archive_p.cnArch * nDim, MPI_DOUBLE, collect_to, 1, the_comm);
                        MPI_Send(arc_obj, st_archive_p.cnArch * nObj, MPI_DOUBLE, collect_to, 2, the_comm);
                    }
                    if(the_rank == collect_to) {
                        MPI_Recv(&st_repo_p.nRep, 1, MPI_INT, collect_from, 0, the_comm, MPI_STATUS_IGNORE);
                        MPI_Recv(repo_var, st_repo_p.nRep * nDim, MPI_DOUBLE, collect_from, 1, the_comm, MPI_STATUS_IGNORE);
                        MPI_Recv(repo_obj, st_repo_p.nRep * nObj, MPI_DOUBLE, collect_from, 2, the_comm, MPI_STATUS_IGNORE);

                        if(st_ctrl_p.collect_pop_type == COLLECT_WEIGHTED) {
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

                            double f1, f2;
                            for(i = 0; i < nPop; i++) {
                                f1 = fitnessFunction(&repo_obj[i * nObj], &st_decomp_p.weights_all[i * nObj]);
                                f2 = fitnessFunction(&arc_obj[i * nObj], &st_decomp_p.weights_all[i * nObj]);
                                if(f1 < f2) {
                                    memcpy(&arc_var[i * nDim], &repo_var[i * nDim], nDim * sizeof(double));
                                    memcpy(&arc_obj[i * nObj], &repo_obj[i * nObj], nObj * sizeof(double));
                                }
                            }
                        } else {
                            memcpy(&repo_var[st_repo_p.nRep * nDim], arc_var, nPop * nDim * sizeof(double));
                            memcpy(&repo_obj[st_repo_p.nRep * nObj], arc_obj, nPop * nObj * sizeof(double));
                            st_repo_p.nRep += nPop;
                            refineRepository_generateND(arc_var, arc_obj, st_archive_p.dens, st_archive_p.rank, NULL,
                                                        st_archive_p.cnArch, nPop);
                        }
                    }
                }
            }
            MPI_Barrier(the_comm);//if(strct_MPI_info.mpi_rank==0)printf("line1002.\n");
            collect_step *= 2;
        } //if(strct_MPI_info.mpi_rank==0)printf("line102\n");
    }
}

void collectDecompositionArchiveWithinPop(int algoMechType)
{
    int nPop = st_global_p.nPop;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int color_pop = st_MPI_p.color_pop;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    MPI_Comm comm_subPop = st_MPI_p.comm_subPop;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj = st_pop_evo_cur.obj;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
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
        if(0 == mpi_rank)
            printf("%s: Improper algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    st_archive_p.cnArch = nPop;
    MPI_Barrier(MPI_COMM_WORLD);//if(strct_MPI_info.mpi_rank==0)printf("LINE 1.\n");

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
    collectViaBinTree(the_rank, the_size, the_comm, st_MPI_p.color_master_subPop);
    //
    return;
}

void collectDecompositionArchiveBeyondPop(int algoMechType)
{
    int nPop = st_global_p.nPop;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int color_pop = st_MPI_p.color_pop;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    MPI_Comm comm_subPop = st_MPI_p.comm_subPop;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj = st_pop_evo_cur.obj;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    //
    if(algoMechType == LOCALIZATION) {
    } else if(algoMechType == DECOMPOSITION) {
    } else {
        if(0 == mpi_rank)
            printf("%s: Improper algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    st_archive_p.cnArch = nPop;
    int the_rank, the_size;
    MPI_Comm the_comm;
    if(st_ctrl_p.flag_multiPop && st_MPI_p.color_master_pop) {
        the_rank = st_MPI_p.mpi_rank_master_pop;
        the_size = st_MPI_p.mpi_size_master_pop;
        the_comm = st_MPI_p.comm_master_pop;
        collectViaBinTree(the_rank, the_size, the_comm, st_MPI_p.color_master_pop);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //
    return;
}

void collectDecompositionArchive(int algoMechType)
{
    collectDecompositionArchiveWithinPop(algoMechType);
    collectDecompositionArchiveBeyondPop(algoMechType);
    //
    return;
}

void gatherInfoBeforeUpdateStructure(int algoMechType)
{
    if(LOCALIZATION != algoMechType) return;

    collectDecompositionArchiveWithinPop(algoMechType);

    return;
}

void scatterInfoAfterUpdateStructure(int algoMechType)
{
    if(LOCALIZATION != algoMechType) return;

    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int color_pop = st_MPI_p.color_pop;
    int mpi_rank = st_MPI_p.mpi_rank;
    int mpi_rank_subPop = st_MPI_p.mpi_rank_subPop;
    int mpi_size_subPop = st_MPI_p.mpi_size_subPop;
    MPI_Comm comm_subPop = st_MPI_p.comm_subPop;
    int* each_size_subPop = st_MPI_p.each_size_subPop;
    int* recv_size_subPop = st_MPI_p.recv_size_subPop;
    int* disp_size_subPop = st_MPI_p.disp_size_subPop;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj = st_pop_evo_cur.obj;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* cur_var_saved = st_pop_evo_cur.var_saved;
    double* cur_obj_saved = st_pop_evo_cur.obj_saved;
    double* arc_var = st_archive_p.var;
    double* arc_obj = st_archive_p.obj;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    //
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
    //
    if(st_MPI_p.color_master_subPop) {
        MPI_Bcast(arc_var, nPop * nDim, MPI_DOUBLE, 0, the_comm);
        MPI_Bcast(arc_obj, nPop * nObj, MPI_DOUBLE, 0, the_comm);
    }
    int nPop_cur = nPop;
    if(algoMechType == LOCALIZATION) {
        nPop_cur = nPop_mine;
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
        nPop_cur = nPop;
        memcpy(cur_var, arc_var, nPop * nDim * sizeof(double));
        memcpy(cur_obj, arc_obj, nPop * nObj * sizeof(double));
    } else {
        if(0 == mpi_rank)
            printf("%s: Improper algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }

    memcpy(cur_var, arc_var, nPop_cur * nDim * sizeof(double));
    memcpy(cur_obj, arc_obj, nPop_cur * nObj * sizeof(double));
    memcpy(cur_var_saved, cur_var, nPop_cur * nDim * sizeof(double));
    memcpy(cur_obj_saved, cur_obj, nPop_cur * nObj * sizeof(double));

    update_xBest(UPDATE_GIVEN, nPop_cur, NULL, cur_var, cur_obj, rot_angle_cur);
    update_xBest_history(UPDATE_GIVEN, nPop_cur, NULL, cur_var, cur_obj, rot_angle_cur);

    synchronizeReferencePoint(algoMechType);
    //
    return;
}
