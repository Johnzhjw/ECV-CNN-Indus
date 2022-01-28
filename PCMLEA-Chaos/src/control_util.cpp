#include "global.h"

void updateNeighborTable(int type, int algoMechType)
{
    //type = 0: weight vector
    //type = 1: diverse variable
    //
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int n_left = st_pop_comm_p.n_weights_left;
    int n_right = st_pop_comm_p.n_weights_right;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    int nDiv = st_grp_ana_p.numDiverIndexes;
    int* niche = &st_decomp_p.niche;
    int* niche_local = &st_decomp_p.niche_local;
    int* niche_neighb = &st_decomp_p.niche_neighb;
    int mpi_rank = st_MPI_p.mpi_rank;
    //
    double* pt_p_all = st_grp_info_p.diver_var_store_all;
    double* pt_p_m = st_pop_comm_p.posFactor_mine;
    double* pt_p_l = st_pop_comm_p.posFactor_left;
    double* pt_p_r = st_pop_comm_p.posFactor_right;
    double* pt_w_all = st_decomp_p.weights_all;
    double* pt_w_m = st_decomp_p.weights_mine;
    double* pt_w_l = st_decomp_p.weights_left;
    double* pt_w_r = st_decomp_p.weights_right;
    int* pt_tb_neighb = st_decomp_p.tableNeighbor;
    int* pt_tb_neighb_local = st_decomp_p.tableNeighbor_local;
    //
    (*niche_local) = (*niche);
    //
    if(algoMechType == LOCALIZATION) {
        (*niche_local) = ((*niche) < nPop_mine) ? (*niche) : nPop_mine;
        (*niche) = ((*niche) < (nPop_mine + n_left + n_right)) ? (*niche) : (nPop_mine + n_left + n_right);
        (*niche_neighb) = nPop_mine / 2;
        if((*niche_neighb) == 0)(*niche_neighb)++;
        //
        int i, j;
        int* tmpIndex = (int*)malloc(3 * nPop * sizeof(int));
        double* tmpDist = (double*)malloc(3 * nPop * sizeof(double));
        //
        for(i = 0; i < nPop_mine; i++) {
            for(j = 0; j < nPop_mine + n_left + n_right; j++) tmpIndex[j] = j;
            if(type != WEIGHT_BASED) {
                for(j = 0; j < nPop_mine; j++) tmpDist[j] = dist_vector(&pt_p_m[i * nDim], &pt_p_m[j * nDim], nDiv);
                for(j = 0; j < n_left; j++) tmpDist[nPop_mine + j] = dist_vector(&pt_p_m[i * nDim], &pt_p_l[j * nDim], nDiv);
                for(j = 0; j < n_right; j++) tmpDist[nPop_mine + n_left + j] = dist_vector(&pt_p_m[i * nDim], &pt_p_r[j * nDim], nDiv);
            } else {
                for(j = 0; j < nPop_mine; j++) tmpDist[j] = dist_vector(&pt_w_m[i * nObj], &pt_w_m[j * nObj], nObj);
                for(j = 0; j < n_left; j++) tmpDist[nPop_mine + j] = dist_vector(&pt_w_m[i * nObj], &pt_w_l[j * nObj], nObj);
                for(j = 0; j < n_right; j++) tmpDist[nPop_mine + n_left + j] = dist_vector(&pt_w_m[i * nObj], &pt_w_r[j * nObj], nObj);
            }
            minfastsort(tmpDist, tmpIndex, nPop_mine, nPop_mine);
            memcpy(&pt_tb_neighb_local[i * nPop], tmpIndex, nPop_mine * sizeof(int));
            minfastsort(tmpDist, tmpIndex, nPop_mine + n_left + n_right, nPop_mine + n_left + n_right);
            //for(j = 0; j < n_mine + n_left + n_right; j++) pt_tb_neighb[i * n_pop + j] = tmpIndex[j];
            memcpy(&pt_tb_neighb[i * nPop], tmpIndex, (*niche) * sizeof(int));
        }
        for(i = nPop_mine; i < nPop_mine + n_left + n_right; i++) {
            for(j = 0; j < nPop_mine; j++) tmpIndex[j] = j;
            if(type != WEIGHT_BASED) {
                if(i < nPop_mine + n_left)
                    for(j = 0; j < nPop_mine; j++)
                        tmpDist[j] = dist_vector(&pt_p_l[(i - nPop_mine) * nDim], &pt_p_m[j * nDim], nDiv);
                else
                    for(j = 0; j < nPop_mine; j++)
                        tmpDist[j] = dist_vector(&pt_p_r[(i - nPop_mine - n_left) * nDim], &pt_p_m[j * nDim], nDiv);
            } else {
                if(i < nPop_mine + n_left)
                    for(j = 0; j < nPop_mine; j++)
                        tmpDist[j] = dist_vector(&pt_w_l[(i - nPop_mine) * nObj], &pt_w_m[j * nObj], nObj);
                else
                    for(j = 0; j < nPop_mine; j++)
                        tmpDist[j] = dist_vector(&pt_w_r[(i - nPop_mine - n_left) * nObj], &pt_w_m[j * nObj], nObj);
            }
            minfastsort(tmpDist, tmpIndex, nPop_mine, nPop_mine);
            //for(j = 0; j < n_mine; j++) pt_tb_neighb[i * n_pop + j] = tmpIndex[j];
            memcpy(&pt_tb_neighb[i * nPop], tmpIndex, nPop_mine * sizeof(int));
        }
        //
        free(tmpIndex);
        free(tmpDist);
    } else if(algoMechType == DECOMPOSITION) {
        int i, j;
        int* tmpIndex = (int*)malloc(nPop * sizeof(int));
        double* tmpDist = (double*)malloc(nPop * sizeof(double));
        //
        for(i = 0; i < nPop; i++) {
            for(j = 0; j < nPop; j++) tmpIndex[j] = j;
            if(type != WEIGHT_BASED) {
                for(j = 0; j < nPop; j++) tmpDist[j] = dist_vector(&pt_p_all[i * nDim], &pt_p_all[j * nDim], nDiv);
            } else {
                for(j = 0; j < nPop; j++) tmpDist[j] = dist_vector(&pt_w_all[i * nObj], &pt_w_all[j * nObj], nObj);
            }
            minfastsort(tmpDist, tmpIndex, nPop, nPop);
            //for(j = 0; j < n_pop; j++) pt_tb_neighb[i * n_pop + j] = tmpIndex[j];
            memcpy(&pt_tb_neighb[i * nPop], tmpIndex, nPop * sizeof(int));
            memcpy(&pt_tb_neighb_local[i * nPop], tmpIndex, nPop * sizeof(int));
        }
        //
        free(tmpDist);
        free(tmpIndex);
    } else {
        if(0 == mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    //
    return;
}

double dist_vector(double* vec1, double* vec2, int len)
{
    int i;
    double sum = 0.0;
    int n = len;
    for(i = 0; i < n; i++)
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    return sum;
}

void minfastsort(double* val, int* ind, int size, int num)
{
    int n = size;
    int m = num;
    int i, j;

    for(i = 0; i < m; i++) {
        for(j = i + 1; j < n; j++) {
            if(val[i] > val[j]) {
                double temp = val[i];
                val[i] = val[j];
                val[j] = temp;
                int id = ind[i];
                ind[i] = ind[j];
                ind[j] = id;
            }
        }
    }
}

void transformPop(int algoMechType)
{
    int nPop = st_global_p.nPop;
    int nPop_mine = st_global_p.nPop_mine;
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* weights_mine = st_decomp_p.weights_mine;
    double* weights_all = st_decomp_p.weights_all;
    int mpi_rank = st_MPI_p.mpi_rank;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj = st_pop_evo_cur.obj;
    double* nadirpoint = st_decomp_p.nadirpoint;
    double* repo_var = st_repo_p.var;
    double* repo_obj = st_repo_p.obj;
    double* fitCur = st_decomp_p.fitCur;
    double* cur_var_saved = st_pop_evo_cur.var_saved;
    double* cur_obj_saved = st_pop_evo_cur.obj_saved;
    double* osp_var = st_pop_evo_offspring.var;
    double* osp_obj = st_pop_evo_offspring.obj;
    double* vMax = st_PSO_p.vMax;
    double* vMin = st_PSO_p.vMin;
    double* maxLimit = st_global_p.maxLimit;
    double* minLimit = st_global_p.minLimit;
    double* velocity = st_PSO_p.velocity;
    int Qubits_angle_opt_tag = st_ctrl_p.Qubits_angle_opt_tag;
    double* rot_angle_cur = st_qu_p.rot_angle_cur;
    double* minLimit_rot_angle = st_qu_p.minLimit_rot_angle;
    double* maxLimit_rot_angle = st_qu_p.maxLimit_rot_angle;
    //
    int theSize = nPop;
    int theSize_out = nPop;
    double* pt_weights = weights_all;
    if(algoMechType == LOCALIZATION) {
        theSize_out = nPop_mine;
        pt_weights = weights_mine;
    } else if(algoMechType == DECOMPOSITION) {
        theSize_out = nPop;
        pt_weights = weights_all;
    } else {
        if(0 == mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }

    int* count = (int*)calloc(theSize, sizeof(int));
    int* flag = (int*)calloc(theSize, sizeof(int));
    int* tIndex = (int*)calloc(theSize, sizeof(int));

    int i, j;
    for(i = 0; i < theSize; i++) {
        update_idealpoint(&cur_obj[i * nObj]);
    }
    for(j = 0; j < nObj; j++) nadirpoint[j] = -1e30;
    for(i = 0; i < theSize; i++)
        for(j = 0; j < nObj; j++)
            if(nadirpoint[j] < cur_obj[i * nObj + j]) nadirpoint[j] = cur_obj[i * nObj + j];
    //
    for(i = 0; i < theSize; i++) {
        count[i] = 1;
        flag[i] = 1;
    }
    double minVal;
    int minIdx;
    double fit;
    for(i = 0; i < theSize_out; i++) tIndex[i] = i;
    shuffle(tIndex, theSize_out);
    int realIdx;
    memcpy(repo_var, cur_var, theSize * nDim * sizeof(double));
    memcpy(repo_obj, cur_obj, theSize * nObj * sizeof(double));
    //
    for(i = 0; i < theSize_out; i++) {
        realIdx = tIndex[i];
        minVal = INF_DOUBLE;
        minIdx = -1;
        //
        for(j = 0; j < theSize; j++) {
            if(flag[j]) {
                fit = fitnessFunction(&repo_obj[j * nObj], &pt_weights[realIdx * nObj]);
                if(fit < minVal) {
                    minVal = fit;
                    minIdx = j;
                }
            }
        }
        if(minIdx >= 0 && minIdx < theSize) {
            memcpy(&cur_var[realIdx * nDim], &repo_var[minIdx * nDim], nDim * sizeof(double));
            memcpy(&cur_obj[realIdx * nObj], &repo_obj[minIdx * nObj], nObj * sizeof(double));
            count[minIdx]--;
            if(!count[minIdx]) flag[minIdx] = 0;
            fitCur[realIdx] = minVal;
        } else {
            if(0 == mpi_rank) {
                printf("%s: Index value is out of bound\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_INDEX_INVALID);
        }
    }
    //
    free(count);
    free(flag);
    free(tIndex);
    //
    memcpy(cur_var_saved, cur_var, theSize_out * nDim * sizeof(double));
    memcpy(cur_obj_saved, cur_obj, theSize_out * nObj * sizeof(double));
    // PSO
    memcpy(osp_var, cur_var, theSize_out * nDim * sizeof(double));
    memcpy(osp_obj, cur_obj, theSize_out * nObj * sizeof(double));
    //
    for(i = 0; i < nDim; i++) {
        vMax[i] = 0.035 * (maxLimit[i] - minLimit[i]);
        vMin[i] = -vMax[i];
    }
    for(i = 0; i < theSize_out; i++) {
        for(j = 0; j < nDim; j++) {
            velocity[i * nDim + j] = rndreal(vMin[j], vMax[j]);
        }
    }
    if(Qubits_angle_opt_tag == FLAG_ON) {
        for(int i = 0; i < theSize_out; i++) {
            for(int j = 0; j < nDim; j++) {
                rot_angle_cur[i * nDim + j] = rndreal(minLimit_rot_angle[j], maxLimit_rot_angle[j]);
            }
        }
    }
    //
    return;
}

void refinePop_ND(int ref_tag, int algoMechType)
{
    if(algoMechType != NONDOMINANCE) {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s: Improper algorithm mechanism type\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
    }
    //
    int nArch = st_archive_p.nArch;
    int nArch_sub = st_archive_p.nArch_sub;
    int nArch_sub_before = st_archive_p.nArch_sub_before;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    //
    switch(ref_tag) {
    case INIT_TAG:
        if(st_MPI_p.color_obj) {
            st_global_p.trans_size = nArch_sub;
        } else {
            st_global_p.trans_size = nArch;
        }
        if(st_MPI_p.color_master_subPop) {
            memcpy(st_archive_p.var, st_pop_evo_cur.var, st_global_p.trans_size * nDim * sizeof(double));
            memcpy(st_archive_p.obj, st_pop_evo_cur.obj, st_global_p.trans_size * nObj * sizeof(double));
            st_archive_p.cnArch = st_global_p.trans_size;
            //
            collectNDArchive();
            //
            //MPI_Bcast(&strct_archive_info.cnArch, 1, MPI_INT, strct_MPI_info.root_master_species_globalScope, strct_MPI_info.comm_master_species_globalScope);
            MPI_Bcast(st_archive_p.var, nArch * nDim, MPI_DOUBLE,
                      st_MPI_p.root_master_subPop_globalScope,
                      st_MPI_p.comm_master_subPop_globalScope);
            MPI_Bcast(st_archive_p.obj, nArch * nObj, MPI_DOUBLE,
                      st_MPI_p.root_master_subPop_globalScope,
                      st_MPI_p.comm_master_subPop_globalScope);
            MPI_Bcast(st_archive_p.rank, nArch, MPI_INT,
                      st_MPI_p.root_master_subPop_globalScope,
                      st_MPI_p.comm_master_subPop_globalScope);
            MPI_Bcast(st_archive_p.dens, nArch, MPI_DOUBLE,
                      st_MPI_p.root_master_subPop_globalScope,
                      st_MPI_p.comm_master_subPop_globalScope);

            if(st_MPI_p.color_obj) {
                memcpy(st_repo_p.var, st_archive_p.var, nArch * nDim * sizeof(double));
                memcpy(st_repo_p.obj, st_archive_p.obj, nArch * nObj * sizeof(double));
                st_repo_p.nRep = nArch;
                refineRepository_generateArchive_sub();
            } else {
            }
        }
        //MPI_Barrier(MPI_COMM_WORLD); if (strct_MPI_info.mpi_rank == strct_MPI_info.mpi_size - 1)printf("line10.\n");

        //		  char filename[MAX_STR];
        //	 		sprintf(filename,"PS/trace/DPCCMOEA_VAR_%s_OBJ%d_VAR%d_key%d_RUN%d", strct_global_paras.testInstance, strct_global_paras.nObj, strct_global_paras.nDim, cur_gen, cur_run);
        //	 		strct_global_paras.fptvar=fopen(filename,"w");
        //	 		save_var(strct_global_paras.fptvar);
        //	 		fclose(strct_global_paras.fptvar);

        //show_indicator_vars(0);
        ////		MPI_Barrier(MPI_COMM_WORLD);if(strct_MPI_info.mpi_rank==0)printf("line0000000.\n");
        //cur_gen++;

        //	MPI_Barrier(MPI_COMM_WORLD); if(strct_MPI_info.mpi_rank==0) printf("BEFORE TRANS\n");
        memcpy(st_pop_evo_cur.var, st_archive_p.var, st_global_p.trans_size * nDim * sizeof(double));
        memcpy(st_pop_evo_cur.obj, st_archive_p.obj, st_global_p.trans_size * nObj * sizeof(double));
        memcpy(st_repo_p.var, st_pop_evo_cur.var, st_global_p.trans_size * nDim * sizeof(double));
        memcpy(st_repo_p.obj, st_pop_evo_cur.obj, st_global_p.trans_size * nObj * sizeof(double));
        {
            memcpy(st_pop_evo_cur.var_saved, st_pop_evo_cur.var,
                   st_global_p.trans_size * nDim * sizeof(double));
            memcpy(st_pop_evo_cur.obj_saved, st_pop_evo_cur.obj,
                   st_global_p.trans_size * nObj * sizeof(double));
        }

        // PSO
        {
            memcpy(st_pop_evo_offspring.var, st_pop_evo_cur.var,
                   st_global_p.trans_size * nDim * sizeof(double));
            memcpy(st_pop_evo_offspring.obj, st_pop_evo_cur.obj,
                   st_global_p.trans_size * nObj * sizeof(double));
        }

        {
            int i, j;
            for(i = 0; i < nDim; i++) {
                st_PSO_p.vMax[i] = 0.035 * (st_global_p.maxLimit[i] - st_global_p.minLimit[i]);
                st_PSO_p.vMin[i] = -st_PSO_p.vMax[i];
            }
            for(i = 0; i < nArch; i++) {
                for(j = 0; j < nDim; j++) {
                    st_PSO_p.velocity[i * nDim + j] = rndreal(st_PSO_p.vMin[j], st_PSO_p.vMax[j]);
                }
            }
        }
        //	MPI_Barrier(MPI_COMM_WORLD); if(strct_MPI_info.mpi_rank==0) printf("AFTER TRANS\n");
        // 	MPI_Barrier(MPI_COMM_WORLD);
        break;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    case UPDATE_TAG:
        MPI_Barrier(MPI_COMM_WORLD); //if (strct_MPI_info.mpi_rank == 0)printf("line10.75.\n");

        if(st_MPI_p.color_obj) {
            memcpy(st_archive_p.var_Ex, st_archive_p.var, nArch_sub_before * nDim * sizeof(double));
            memcpy(st_archive_p.obj_Ex, st_archive_p.obj, nArch_sub_before * nObj * sizeof(double));
            st_global_p.trans_size = nArch_sub_before; //if(strct_MPI_info.mpi_rank==0)printf("line101\n");
        } else {
            memcpy(st_archive_p.var_Ex, st_archive_p.var, nArch * nDim * sizeof(double));
            memcpy(st_archive_p.obj_Ex, st_archive_p.obj, nArch * nObj * sizeof(double));
            st_global_p.trans_size = nArch; //if(strct_MPI_info.mpi_rank==0)printf("line101\n");
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
                //
                if(my_tag != -1) {
                    collect_to = st_MPI_p.mpi_rank_master_subPop_globalScope -
                                 st_MPI_p.mpi_rank_master_subPop_globalScope % (2 * collect_step);
                    collect_from = collect_to + collect_step;

                    if(collect_from < st_MPI_p.mpi_size_master_subPop_globalScope) {
                        if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_from) {
                            MPI_Send(&st_global_p.trans_size, 1, MPI_INT, collect_to, 0, st_MPI_p.comm_master_subPop_globalScope);
                            MPI_Send(st_archive_p.var_Ex, st_global_p.trans_size * nDim, MPI_DOUBLE, collect_to, 1,
                                     st_MPI_p.comm_master_subPop_globalScope);
                            MPI_Send(st_archive_p.obj_Ex, st_global_p.trans_size * nObj, MPI_DOUBLE, collect_to, 2,
                                     st_MPI_p.comm_master_subPop_globalScope);
                        }
                        if(st_MPI_p.mpi_rank_master_subPop_globalScope == collect_to) {
                            MPI_Recv(&st_repo_p.nRep, 1, MPI_INT, collect_from, 0, st_MPI_p.comm_master_subPop_globalScope, MPI_STATUS_IGNORE);
                            MPI_Recv(st_repo_p.var, st_repo_p.nRep * nDim, MPI_DOUBLE, collect_from, 1,
                                     st_MPI_p.comm_master_subPop_globalScope,
                                     MPI_STATUS_IGNORE);
                            MPI_Recv(st_repo_p.obj, st_repo_p.nRep * nObj, MPI_DOUBLE, collect_from, 2,
                                     st_MPI_p.comm_master_subPop_globalScope,
                                     MPI_STATUS_IGNORE);
                            memcpy(&st_repo_p.var[st_repo_p.nRep * nDim], st_archive_p.var_Ex,
                                   st_global_p.trans_size * nDim * sizeof(double));
                            memcpy(&st_repo_p.obj[st_repo_p.nRep * nObj], st_archive_p.obj_Ex,
                                   st_global_p.trans_size * nObj * sizeof(double));
                            st_repo_p.nRep += st_global_p.trans_size;
                            if(st_repo_p.nRep > nArch) {
                                refineRepository_generateND(st_archive_p.var_Ex, st_archive_p.obj_Ex,
                                                            st_archive_p.dens_Ex, st_archive_p.rank_Ex, NULL,
                                                            st_archive_p.cnArchEx, nArch);
                                st_global_p.trans_size = nArch;
                            } else {
                                refineRepository_generateND(st_archive_p.var_Ex, st_archive_p.obj_Ex,
                                                            st_archive_p.dens_Ex, st_archive_p.rank_Ex, NULL,
                                                            st_archive_p.cnArchEx, st_repo_p.nRep);
                                st_global_p.trans_size = st_repo_p.nRep;
                            }
                        }
                    }
                }
                MPI_Barrier(st_MPI_p.comm_master_subPop_globalScope);//if(strct_MPI_info.mpi_rank==0)printf("line1002.\n");
                collect_step *= 2;
            } //if(strct_MPI_info.mpi_rank==0)printf("line102\n");
        }

        if(st_MPI_p.color_master_subPop) {
            MPI_Bcast(&st_archive_p.cnArchEx, 1, MPI_INT, 0, st_MPI_p.comm_master_subPop_globalScope);
            MPI_Bcast(st_archive_p.var_Ex, st_archive_p.cnArchEx * nDim, MPI_DOUBLE, 0,
                      st_MPI_p.comm_master_subPop_globalScope);
            MPI_Bcast(st_archive_p.obj_Ex, st_archive_p.cnArchEx * nObj, MPI_DOUBLE, 0,
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
                memcpy(&st_repo_p.var[st_repo_p.nRep * nDim], st_archive_p.var_Ex,
                       st_archive_p.cnArchEx * nDim * sizeof(double));
                memcpy(&st_repo_p.obj[st_repo_p.nRep * nObj], st_archive_p.obj_Ex,
                       st_archive_p.cnArchEx * nObj * sizeof(double));
                st_repo_p.nRep += st_archive_p.cnArchEx;
                refineRepository_generateArchive_sub();
            } else {
                memcpy(st_archive_p.var, st_archive_p.var_Ex, nArch * nDim * sizeof(double));
                memcpy(st_archive_p.obj, st_archive_p.obj_Ex, nArch * nObj * sizeof(double));
                memcpy(st_archive_p.dens, st_archive_p.dens_Ex, nArch * sizeof(double));
                memcpy(st_archive_p.rank, st_archive_p.rank_Ex, nArch * sizeof(int));
                st_archive_p.cnArch = nArch;
            }
        }
        break;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    case UPDATE_TAG_BRIEF:
        if(st_MPI_p.color_obj) {
            st_repo_p.nRep = 0;
            memcpy(&st_repo_p.var[st_repo_p.nRep * nDim], st_archive_p.var,
                   st_archive_p.nArch_sub_before * nDim * sizeof(double));
            memcpy(&st_repo_p.obj[st_repo_p.nRep * nObj], st_archive_p.obj,
                   st_archive_p.nArch_sub_before * nObj * sizeof(double));
            st_repo_p.nRep = st_archive_p.nArch_sub_before;
            //memcpy(&strct_repo_info.var[strct_repo_info.nRep * strct_global_paras.nDim], strct_archive_info.var_archiveEx, strct_archive_info.nArch * strct_global_paras.nDim * sizeof(double));
            //memcpy(&strct_repo_info.obj[strct_repo_info.nRep * strct_global_paras.nObj], strct_archive_info.obj_archiveEx, strct_archive_info.nArch * strct_global_paras.nObj * sizeof(double));
            //strct_repo_info.nRep += strct_archive_info.nArch;
            refineRepository_generateArchive_sub();
        }
        break;
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    default:
        printf("%s:INVALID ref_tag for refinePop_ND, EXITING...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_UNKNOWN_UPDATE_TAG);
        break;
    }
    st_archive_p.cnArch = nArch;
    memcpy(st_archive_p.var_Ex, st_archive_p.var, st_archive_p.cnArch * nDim * sizeof(double));
    memcpy(st_archive_p.obj_Ex, st_archive_p.obj, st_archive_p.cnArch * nObj * sizeof(double));
    st_archive_p.cnArchEx = st_archive_p.cnArch;
    //
    return;
}

void setVarTypes()
{
    for(int i = 0; i < st_global_p.nDim; i++) {
        st_ctrl_p.types_var_all[i] = VAR_DOUBLE;
    }

    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.mixed_var_types_tag =
        FLAG_OFF;
    FLAG_ON;

    if(st_ctrl_p.type_test == MY_TYPE_NORMAL) {
        if(!strcmp(st_global_p.testInstance, "IWSN_S_1F")) {
            int cur_ind;
            for(int i = 0; i < N_DIREC_S_1F; i++) {
                cur_ind = i * D_DIREC_S_1F;
                st_ctrl_p.types_var_all[cur_ind++] = VAR_DISCRETE;
                st_ctrl_p.types_var_all[cur_ind++] = VAR_DISCRETE;
            }
            for(int i = 0; i < N_RELAY_S_1F; i++) {
                cur_ind = N_DIREC_S_1F * D_DIREC_S_1F + i * D_RELAY_S_1F;
                st_ctrl_p.types_var_all[cur_ind++] = VAR_DISCRETE;
                st_ctrl_p.types_var_all[cur_ind++] = VAR_DISCRETE;
            }
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
        if(st_ctrl_p.type_var_encoding == VAR_DOUBLE) {
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_ctrl_p.types_var_all[i] = VAR_DOUBLE;
            }
        } else {
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_ctrl_p.types_var_all[i] = VAR_BINARY;
            }
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO1_FRNN) {
        st_ctrl_p.mixed_var_types_tag =
            FLAG_ON;
        int offset0 = 0;
        int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO1_FRNN;
        int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO1_FRNN;
        int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO1_FRNN;
        int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO1_FRNN;
        int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO1_FRNN;
        int offset6 = offset5 + NUM_CLASS_EVO1_FRNN;
        //
        for(int i = offset0; i < offset1; i++) {
            st_ctrl_p.types_var_all[i] = VAR_DISCRETE;
        }
        for(int i = offset1; i < offset2; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
        for(int i = offset2; i < offset3; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO2_FRNN) {
        st_ctrl_p.mixed_var_types_tag =
            FLAG_ON;
        int offset0 = 0;
        int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO2_FRNN;
        int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO2_FRNN;
        int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO2_FRNN;
        int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO2_FRNN;
        int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO2_FRNN;
        int offset6 = offset5 + DIM_CONSEQUENCE_EVO2_FRNN;
        //
        for(int i = offset0; i < offset1; i++) {
            st_ctrl_p.types_var_all[i] = VAR_DISCRETE;
        }
        for(int i = offset1; i < offset2; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
        for(int i = offset2; i < offset3; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO3_FRNN) {
        st_ctrl_p.mixed_var_types_tag =
            FLAG_ON;
        int offset0 = 0;
        int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO3_FRNN;
        int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO3_FRNN;
        int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO3_FRNN;
        int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO3_FRNN;
        int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO3_FRNN;
        int offset6 = offset5 + NUM_CLASS_EVO3_FRNN;
        //
        for(int i = offset0; i < offset1; i++) {
            st_ctrl_p.types_var_all[i] = VAR_DISCRETE;
        }
        for(int i = offset1; i < offset2; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
        for(int i = offset2; i < offset3; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO4_FRNN) {
        st_ctrl_p.mixed_var_types_tag =
            FLAG_ON;
        int offset0 = 0;
        int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO4_FRNN;
        int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO4_FRNN;
        int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO4_FRNN;
        int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO4_FRNN;
        int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO4_FRNN;
        int offset6 = offset5 + DIM_CONSEQUENCE_EVO4_FRNN;
        //
        for(int i = offset0; i < offset1; i++) {
            st_ctrl_p.types_var_all[i] = VAR_DISCRETE;
        }
        for(int i = offset1; i < offset2; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
        for(int i = offset2; i < offset3; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_EVO5_FRNN) {
        st_ctrl_p.mixed_var_types_tag =
            FLAG_ON;
        int offset0 = 0;
        int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVO5_FRNN;
        int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVO5_FRNN;
        int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVO5_FRNN;
        int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVO5_FRNN;
        int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVO5_FRNN;
        int offset6 = offset5 + DIM_CONSEQUENCE_EVO5_FRNN;
        int offset7 = offset6 + DIM_CONSE_WEIGHT_EVO5_FRNN;
        //
        for(int i = offset0; i < offset1; i++) {
            st_ctrl_p.types_var_all[i] = VAR_DISCRETE;
        }
        for(int i = offset1; i < offset2; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
        for(int i = offset2; i < offset3; i++) {
            st_ctrl_p.types_var_all[i] = VAR_BINARY;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.mixed_var_types_tag =
        FLAG_OFF;
    FLAG_ON;

    return;
}
