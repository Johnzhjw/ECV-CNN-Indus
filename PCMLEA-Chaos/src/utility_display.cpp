# include "global.h"
#include "indicator_igd.h"
#include "indicator_hv.h"
#include <time.h>

void show_uTrail_one_group()
{
    int i, j;
    printf("\nuTrail_one_group...\n");
    for(i = 0; i < st_global_p.nPop_mine; i++) {
        printf("ID: %d\n", i + 1);
        if(st_MPI_p.color_obj) {
            for(j = 0; j < st_grp_info_p.table_mine_size; j++) {
                printf("%lf\t", st_pop_evo_offspring.var[i * st_global_p.nDim + st_grp_info_p.table_mine[j]]);
            }
        } else {
            for(j = 0; j < st_grp_info_p.table_mine_size; j++) {
                printf("%lf\t", st_pop_evo_offspring.var[i * st_global_p.nDim + st_grp_info_p.table_mine[j]]);
            }
        }
        printf("\n");
    }
}

void show_uTrail()
{
    int i, j;
    printf("\nuTrail...\n");
    for(i = 0; i < st_global_p.nPop_mine; i++) {
        printf("ID: %d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_pop_evo_offspring.var[i * st_global_p.nDim + j]);
        }
        printf("\nFitness...\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_pop_evo_offspring.obj[i * st_global_p.nObj + j]);
        }
        printf("\n");
    }
}

void showRepository()
{
    int i, j;
    printf("\n repository...\n");
    for(i = 0; i < st_repo_p.nRep; i++) {
        printf("ID:%d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_repo_p.var[i * st_global_p.nDim + j]);
        }
        printf("\nFitness:\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_repo_p.obj[i * st_global_p.nObj + j]);
        }
        printf("\n");
    }
}

void showLimits()
{
    int i, j;
    printf("\n Limits...\n");
    for(i = 0; i < st_global_p.nObj; i++) {
        printf("ID:%d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_global_p.minLimit[j]);
        }
        printf("\n");
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_global_p.maxLimit[j]);
        }
        printf("\n");
    }
}

void showArchive()
{
    int i, j;
    printf("\n Archive...%d\n", st_archive_p.cnArch);
    for(i = 0; i < st_archive_p.cnArch; i++) {
        printf("ID:%d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_archive_p.var[i * st_global_p.nDim + j]);
        }
        printf("\n");
        // 		if(strct_MPI_info.color_population_property_2)
        // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
        // 			{
        // 				printf("%lf\t",archive_one_group[i*strct_global_paras.nDim+j]);
        // 			}
        // 		else
        // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
        // 			{
        // 				printf("%lf\t",archive_one_group[i*strct_global_paras.nDim+j]);
        // 			}
        printf("\nFitness:\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_archive_p.obj[i * st_global_p.nObj + j]);
        }
        printf("\n");
        printf("rank:\t%d\n", st_archive_p.rank[i]);
        printf("Distance: %lf\n\n", st_archive_p.dens[i]);
    }
}

void showArchiveEx()
{
    int i, j;
    printf("\n ArchiveEx...%d\n", st_archive_p.cnArchEx);
    for(i = 0; i < st_archive_p.cnArchEx; i++) {
        printf("ID:%d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_archive_p.var_Ex[i * st_global_p.nDim + j]);
        }
        printf("\n");
        if(st_MPI_p.color_obj) {
            for(j = 0; j < st_grp_info_p.table_mine_size; j++) {
                printf("%lf\t", st_archive_p.var_Ex[i * st_global_p.nDim + st_grp_info_p.table_mine[j]]);
            }
        } else {
            for(j = 0; j < st_grp_info_p.table_mine_size; j++) {
                printf("%lf\t", st_archive_p.var_Ex[i * st_global_p.nDim + st_grp_info_p.table_mine[j]]);
            }
        }
        printf("\n");
        printf("Fitness:\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_archive_p.obj_Ex[i * st_global_p.nObj + j]);
        }
        printf("\n");
        printf("Distance: %lf\n\n", st_archive_p.dens_Ex[i]);
    }
}

void showArchive_exchange()
{
    int i, j;
    printf("\n Archive_exchg...%d\n", st_archive_p.cnArch_exchange);
    for(i = 0; i < st_archive_p.cnArch_exchange; i++) {
        printf("ID:%d\n", i + 1);
        for(j = 0; j < st_global_p.nDim; j++) {
            printf("%lf\t", st_archive_p.var_exchange[i * st_global_p.nDim + j]);
        }
        printf("\nFitness:\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_archive_p.obj_exchange[i * st_global_p.nObj + j]);
        }
        printf("\n");
        printf("Distance: %lf\n", st_archive_p.dens_exchange[i]);
    }
}

void showGlobalBest()
{
    if(st_MPI_p.color_obj) {
        int i, j;
        printf("xBest\t%d\n", st_MPI_p.mpi_rank);
        for(i = 0; i < st_global_p.nDim; i++) {
            printf("%lf\t", st_pop_best_p.var_best[i]);
        }
        printf("\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_pop_best_p.obj_best[j]);
        }
        printf("\n");
    }
}

void showGlobalBestEx()
{
    if(st_MPI_p.color_obj) {
        int i, j;
        printf("xBestEx\t%d\n", st_MPI_p.mpi_rank);
        for(i = 0; i < st_global_p.nDim; i++) {
            printf("%lf\t", st_pop_best_p.var_best_exchange[i]);
        }
        printf("\n");
        for(i = 0; i < st_grp_info_p.table_mine_size; i++) {
            printf("%lf\t", st_pop_best_p.var_best_exchange[st_grp_info_p.table_mine[i]]);
        }
        printf("\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_pop_best_p.obj_best_exchange[j]);
        }
        printf("\n");
    }
}

void showPopulation()
{
    int i, j;

    printf("Swarm:\t%d\n", st_MPI_p.mpi_rank);
    for(i = 0; i < st_global_p.nPop_mine; i++) {
        printf("ID:\t%d\n", i + 1);
        // 		for(j=0;j<strct_global_paras.nDim;j++)
        // 		{
        // 			printf("%lf\t",xCurrent[INDEX(0,i,j)]);
        // 		}
        // 		printf("\n");
        // 		if(strct_MPI_info.color_population_property_2)
        // 		{
        // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
        // 			{
        // 				printf("%lf\t",xCurrent_one_group[i*strct_global_paras.nDim+j]);
        // 			}
        // 		}
        // 		else
        // 		{
        // 			for(j=0;j<strct_grp_info_vals.table_mine_size;j++)
        // 			{
        // 				printf("%lf\t",xCurrent_one_group[i*strct_global_paras.nDim+j]);
        // 			}
        // 		}
        // 		printf("\n");
        for(j = 0; j < st_global_p.nObj; j++) {
            printf("%.16e\t", st_pop_evo_cur.obj[i * st_global_p.nObj + j]);
        }
        printf("\n");
    }
}

void showGroup()
{
    int l, i;
    printf("Grouping information...\n");
    for(int iObj = 0; iObj <= 0; iObj++) {
        if(iObj == 0)
            printf("\ngroup_table_allObjectives...%d...\n", st_grp_info_p.Groups_sizes[iObj]);
        else
            printf("\ngroup_table_separatedObjective...NO%d...%d...\n", iObj, st_grp_info_p.Groups_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_sizes[iObj]; l++) {
            printf("GROUP: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(i = 0; i < st_grp_info_p.Groups_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_i] + i;
                printf("%06d\t", st_grp_info_p.Groups[tmp_j]);
            }
            printf("\n");
        }
    }
}

void show_F_CR_mu()
{
    if(st_MPI_p.color_obj) {
        printf("rank: %d\n", st_MPI_p.mpi_rank);
        printf("F_mu = %lf\n", st_DE_p.F_mu);
        printf("CR_mu = %lf\n", st_DE_p.CR_mu);
    }
}

void show_F_CR()
{
    printf("rank: %d\n", st_MPI_p.mpi_rank);
    int i;
    if(st_MPI_p.color_obj) {
        printf("Obj F:\n");
        for(i = 0; i < st_global_p.nPop_mine; i++) {
            printf("ID: %d F %lf\n", i + 1, st_DE_p.F__cur[i]);
        }
        printf("Obj CR:\n");
        for(i = 0; i < st_global_p.nPop_mine; i++) {
            printf("ID: %d CR %lf\n", i + 1, st_DE_p.CR_cur[i]);
        }
    } else {
        printf("Arch F:\n");
        for(i = 0; i < st_global_p.nPop_mine; i++) {
            printf("ID: %d F %lf\n", i + 1, st_DE_p.F__archive[i]);
        }
        printf("Arch CR:\n");
        for(i = 0; i < st_global_p.nPop_mine; i++) {
            printf("ID: %d CR %lf\n", i + 1, st_DE_p.CR_archive[i]);
        }
    }
}

void showGroup_raw()
{
    int iObj, l, i;

    printf("\nnDiver=%d\n", st_grp_ana_p.numDiverIndexes);
    for(i = 0; i < st_grp_ana_p.numDiverIndexes; i++) {
        printf("%06d ", st_grp_info_p.DiversityIndexs[i]);
    }
    printf("\n");
    for(iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0)
            printf("\ngroup_raw_table_allObjectives...%d...\n", st_grp_info_p.Groups_raw_sizes[iObj]);
        else
            printf("\ngroup_raw_table_separatedObjective...NO%d...%d...\n", iObj, st_grp_info_p.Groups_raw_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_raw_sizes[iObj]; l++) {
            printf("GROUP_RAW: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(i = 0; i < st_grp_info_p.Groups_raw_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_raw_sub_disps[tmp_i] + i;
                printf("%06d\t", st_grp_info_p.Groups_raw[tmp_j]);
            }
            printf("\n");
        }
    }
}

void show_mpi_info()
{
    int i, j;
    //printf("Outputting...\n");
    for(i = 0; i < st_MPI_p.mpi_size; i++) {
        if(st_MPI_p.mpi_rank == i) {
            printf("\n//----------------------------------------------\n");
            printf("strct_MPI_info.mpi_rank\t strct_MPI_info.mpi_size\n");
            printf("%d\t %d\n", st_MPI_p.mpi_rank, st_MPI_p.mpi_size);
            printf("strct_MPI_info.color_population_property_2\t strct_MPI_info.mpi_rank_population_2\t strct_MPI_info.mpi_size_population_2\t strct_MPI_info.comm_population_2\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_obj, st_MPI_p.mpi_rank_obj,
                   st_MPI_p.mpi_size_obj, st_MPI_p.comm_obj);
            printf("strct_MPI_info.color_population\t strct_MPI_info.mpi_rank_population\t strct_MPI_info.mpi_size_population\t strct_MPI_info.comm_population\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_pop, st_MPI_p.mpi_rank_pop,
                   st_MPI_p.mpi_size_pop,
                   st_MPI_p.comm_pop);
            printf("strct_MPI_info.color_species\t strct_MPI_info.mpi_rank_species\t strct_MPI_info.mpi_size_species\t strct_MPI_info.comm_species\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_subPop, st_MPI_p.mpi_rank_subPop, st_MPI_p.mpi_size_subPop,
                   st_MPI_p.comm_subPop);
            printf("strct_MPI_info.color_master_species\t strct_MPI_info.mpi_rank_master_species_globalScope\t strct_MPI_info.mpi_size_master_species_globalScope\t strct_MPI_info.comm_master_species_globalScope\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_master_subPop, st_MPI_p.mpi_rank_master_subPop_globalScope,
                   st_MPI_p.mpi_size_master_subPop_globalScope, st_MPI_p.comm_master_subPop_globalScope);
            printf("strct_MPI_info.root_master_species_globalScope\n");
            printf("%d\n", st_MPI_p.root_master_subPop_globalScope);
            printf("strct_MPI_info.color_master_species\t strct_MPI_info.mpi_rank_master_species_populationScope\t strct_MPI_info.mpi_size_master_species_populationScope\t strct_MPI_info.comm_master_species_populationScope\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_master_subPop, st_MPI_p.mpi_rank_master_subPop_popScope,
                   st_MPI_p.mpi_size_master_subPop_popScope, st_MPI_p.comm_master_subPop_popScope);
            printf("strct_MPI_info.color_master_population\t strct_MPI_info.mpi_rank_master_population\t strct_MPI_info.mpi_size_master_population\t strct_MPI_info.comm_master_population\n");
            printf("%d\t %d\t %d\t %x\n", st_MPI_p.color_master_pop, st_MPI_p.mpi_rank_master_pop,
                   st_MPI_p.mpi_size_master_pop, st_MPI_p.comm_master_pop);
            printf("strct_MPI_info.globalRank_master_population\n");
            for(j = 0; j < st_global_p.nObj + 1; j++) {
                printf("%d ", st_MPI_p.globalRank_master_pop[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return;
}

void show_indicator_vars_simp(int finalTag)
{
    double tmp_time = clock();

    double* var_tmp = NULL;
    double* obj_tmp = NULL;
    int     size_tmp = 0;

    switch(st_ctrl_p.algo_mech_type) {
    case LOCALIZATION:
    case DECOMPOSITION:
        var_tmp = st_archive_p.var;
        obj_tmp = st_archive_p.obj;
        size_tmp = st_archive_p.cnArch;
        MPI_Bcast(&size_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        break;
    case NONDOMINANCE:
        var_tmp = st_archive_p.var_Ex;
        obj_tmp = st_archive_p.obj_Ex;
        size_tmp = st_archive_p.cnArchEx;
        MPI_Bcast(&size_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        break;
    default:
        if(0 == st_MPI_p.mpi_rank) {
            printf("No such algorithm mechanism type\n");
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
        break;
    }

    if(st_ctrl_p.type_test == MY_TYPE_NORMAL) {   //case 0
        if(st_MPI_p.mpi_rank == 0) {
            char filename[MAX_CHAR_ARR_SIZE];
            if(finalTag != FINAL_TAG) {
                if(st_ctrl_p.flag_save_trace_PS == FLAG_ON) {
                    sprintf(filename, "PS/trace/%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                            st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                    st_global_p.fptvar = fopen(filename, "w");
                    if(!strcmp(st_global_p.testInstance, "EdgeComputation")) {
                        save_double_as_int(st_global_p.fptvar, var_tmp, size_tmp, st_global_p.nDim, 0);
                    } else {
                        save_double(st_global_p.fptvar, var_tmp, size_tmp, st_global_p.nDim, 0);
                    }
                    fclose(st_global_p.fptvar);
                }
                sprintf(filename, "PF/trace/%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                st_global_p.fptobj = fopen(filename, "w");
            } else {
                st_global_p.fptobj = fopen(st_global_p.objsal, "w");
            }
            save_double(st_global_p.fptobj, obj_tmp, size_tmp, st_global_p.nObj, 0);
            fclose(st_global_p.fptobj);
            double tmp_min[128];
            for(int i = 0; i < st_global_p.nObj; i++) tmp_min[i] = 6e60;
            for(int i = 0; i < size_tmp; i++) {
                for(int j = 0; j < st_global_p.nObj; j++) {
                    if(tmp_min[j] > obj_tmp[i * st_global_p.nObj + j])
                        tmp_min[j] = obj_tmp[i * st_global_p.nObj + j];
                }
            }
            if(finalTag != FINAL_TAG) {
                if(st_ctrl_p.indicator_tag == INDICATOR_IGD || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    st_indicator_p.mat_IGD_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                        generateIGD(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                    st_global_p.nDim, &st_ctrl_p.cur_trace,
                                    st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-IGD: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100 * st_ctrl_p.cur_trace / NTRACE, st_indicator_p.mat_IGD_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
                    for(int i = 0; i < st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
                    printf("\n");
                }
                if(st_ctrl_p.indicator_tag == INDICATOR_HV || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, &st_ctrl_p.cur_trace,
                                   st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-HV:  %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100 * st_ctrl_p.cur_trace / NTRACE, st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
                    for(int i = 0; i < st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
                    printf("\n");
                }
            } else {
                if(st_ctrl_p.indicator_tag == INDICATOR_IGD || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    st_indicator_p.mat_IGD_all[st_ctrl_p.cur_run][NTRACE + 1] =
                        generateIGD(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                    st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-IGD: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100, st_indicator_p.mat_IGD_all[st_ctrl_p.cur_run][NTRACE + 1]);
                    for(int i = 0; i < st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
                    printf("\n");
                }
                if(st_ctrl_p.indicator_tag == INDICATOR_HV || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][NTRACE + 1] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-HV:  %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100, st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][NTRACE + 1]);
                    for(int i = 0; i < st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
                    printf("\n");
                }
            }
        }
    } else if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE ||
              st_ctrl_p.type_test == MY_TYPE_LeNet ||
              st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE ||
              st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus ||
              st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus ||
              st_ctrl_p.type_test == MY_TYPE_CFRNN_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_EVO1_FRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO2_FRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO3_FRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO4_FRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO5_FRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO_FRNN_PREDICT ||
              st_ctrl_p.type_test == MY_TYPE_EVO_DFRNN_PREDICT ||
              st_ctrl_p.type_test == MY_TYPE_INTRUSION_DETECTION_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_ACTIVITY_DETECTION_CLASSIFY ||
              st_ctrl_p.type_test == MY_TYPE_RecSys_SmartCity ||
              st_ctrl_p.type_test == MY_TYPE_EVO_CNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO_CFRNN ||
              st_ctrl_p.type_test == MY_TYPE_EVO_MOBILE_SINK) {  //case 1 and 2
        char filename[MAX_CHAR_ARR_SIZE];
        double* pFront_valid = (double*)calloc(size_tmp * st_global_p.nObj, sizeof(double));
        double* pFront = (double*)calloc(size_tmp * st_global_p.nObj, sizeof(double));
        if(st_MPI_p.mpi_rank == 0) {
            /////////TRAIN
            if(finalTag != FINAL_TAG) {
                if(st_ctrl_p.flag_save_trace_PS == FLAG_ON) {
                    sprintf(filename, "PS/trace/%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                            st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                    st_global_p.fptvar = fopen(filename, "w");
                    if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY || st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY_TREE) {
                        transform_var_feature(var_tmp, st_archive_p.var_feature, size_tmp);
                        save_int(st_global_p.fptvar, st_archive_p.var_feature, size_tmp, TH_N_FEATURE, 0);
                    } else
                        save_double(st_global_p.fptvar, var_tmp, size_tmp, st_global_p.nDim, 0);
                    fclose(st_global_p.fptvar);
                }
                sprintf(filename, "PF/trace/%s_FUN_%s_TRAIN_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                st_global_p.fptobj = fopen(filename, "w");
            } else {
                sprintf(filename, "PF/%s_FUN_%s_TRAIN_OBJ%d_VAR%d_MPI%d_RUN%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run);
                st_global_p.fptobj = fopen(filename, "w");
            }
            save_double(st_global_p.fptobj, obj_tmp, size_tmp, st_global_p.nObj, 0);
            fclose(st_global_p.fptobj);
            double tmp_min[128];
            for(int i = 0; i <= st_global_p.nObj; i++) tmp_min[i] = 6e60;
            for(int i = 0; i < size_tmp; i++) {
                for(int j = 0; j < st_global_p.nObj; j++) {
                    if(tmp_min[j] > obj_tmp[i * st_global_p.nObj + j])
                        tmp_min[j] = obj_tmp[i * st_global_p.nObj + j];
                }
                double tmp_beta = 1;
                double tmp_F;
                if(obj_tmp[i * st_global_p.nObj + 0] < 1 && obj_tmp[i * st_global_p.nObj + 1] < 1) {
                    tmp_F = (1 + tmp_beta * tmp_beta) * (1 - obj_tmp[i * st_global_p.nObj + 0]) * (1 - obj_tmp[i * st_global_p.nObj +
                            1]) /
                            (tmp_beta * tmp_beta * (1 - obj_tmp[i * st_global_p.nObj + 0] + 1 - obj_tmp[i * st_global_p.nObj + 1]));
                    tmp_F = 1 - tmp_F;
                } else {
                    tmp_F = 1;
                }
                if(tmp_min[st_global_p.nObj] > tmp_F)
                    tmp_min[st_global_p.nObj] = tmp_F;
            }
            if(finalTag != FINAL_TAG) {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, &st_ctrl_p.cur_trace,
                                   st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-TRAIN: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100 * st_ctrl_p.cur_trace / NTRACE,
                       st_indicator_p.mat_HV_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
            } else {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all_TRAIN[st_ctrl_p.cur_run][NTRACE + 1] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, obj_tmp, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-TRAIN: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100, st_indicator_p.mat_HV_all_TRAIN[st_ctrl_p.cur_run][NTRACE + 1]);
            }
            for(int i = 0; i < st_global_p.nObj + 1; i++) printf("%lf ", tmp_min[i]);
            printf("\n");
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////VALIDATION & TEST
        int quo = 0, rem = 0;
        //void(*testFuncPtINT)(int *, double *, MPI_Comm, int, int) = NULL;
        void(*validFuncPtFLT)(double*, double*) = NULL;
        void(*testFuncPtFLT)(double*, double*) = NULL;
        //
        switch(st_ctrl_p.type_test) {
        case MY_TYPE_FS_CLASSIFY:
        case MY_TYPE_FS_CLASSIFY_TREE:
            break;
        case MY_TYPE_LeNet:
            testFuncPtFLT = Fitness_LeNet_test;
            break;
        case MY_TYPE_LeNet_ENSEMBLE:
            testFuncPtFLT = Fitness_LeNet_test_ensemble;
            break;
        case MY_TYPE_LeNet_CLASSIFY_Indus:
            testFuncPtFLT = Fitness_Classify_CNN_test;
            validFuncPtFLT = Fitness_Classify_CNN_validation;
            break;
        case MY_TYPE_NN_CLASSIFY_Indus:
            testFuncPtFLT = Fitness_Classify_NN_test;
            validFuncPtFLT = Fitness_Classify_NN_validation;
            break;
        case MY_TYPE_CFRNN_CLASSIFY:
            testFuncPtFLT = Fitness_Classify_CFRNN_test;
            break;
        case MY_TYPE_EVO1_FRNN:
            testFuncPtFLT = Fitness_EVO1_FRNN_testSet;
            break;
        case MY_TYPE_EVO2_FRNN:
            testFuncPtFLT = Fitness_EVO2_FRNN_testSet;
            break;
        case MY_TYPE_EVO3_FRNN:
            testFuncPtFLT = Fitness_EVO3_FRNN_testSet;
            break;
        case MY_TYPE_EVO4_FRNN:
            testFuncPtFLT = Fitness_EVO4_FRNN_testSet;
            break;
        case MY_TYPE_EVO5_FRNN:
            testFuncPtFLT = Fitness_EVO5_FRNN_testSet;
            break;
        case MY_TYPE_EVO_FRNN_PREDICT:
            testFuncPtFLT = Fitness_MOP_Predict_FRNN_test;
            break;
        case MY_TYPE_EVO_DFRNN_PREDICT:
            testFuncPtFLT = Fitness_MOP_Predict_DFRNN_test;
            break;
        case MY_TYPE_INTRUSION_DETECTION_CLASSIFY:
            testFuncPtFLT = Fitness_IntrusionDetection_FRNN_Classify_test;
            break;
        case MY_TYPE_ACTIVITY_DETECTION_CLASSIFY:
            testFuncPtFLT = Fitness_ActivityDetection_FRNN_Classify_test;
            break;
        case MY_TYPE_RecSys_SmartCity:
            testFuncPtFLT = Fitness_RS_SC_testSet;
            break;
        case MY_TYPE_EVO_CNN:
            testFuncPtFLT = Fitness_evoCNN_Classify_test;
            break;
        case MY_TYPE_EVO_CFRNN:
            testFuncPtFLT = Fitness_evoCFRNN_Classify_test;
            break;
        case MY_TYPE_EVO_MOBILE_SINK:
            testFuncPtFLT = Fitness_MOP_Mob_Sink_test;
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s:Problem type error, exiting...\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_PROBLEM_TYPE);
            break;
        }
        //////////////////////////////////////////////////////////////////////////
        switch(st_ctrl_p.type_test) {
        case MY_TYPE_FS_CLASSIFY:
        case MY_TYPE_FS_CLASSIFY_TREE:
            transform_var_feature(var_tmp, st_archive_p.var_feature, st_global_p.nPop);
            for(int i = 0; i < size_tmp; i++) {
                if(st_ctrl_p.type_test == MY_TYPE_FS_CLASSIFY) {
                    //testAccuracy(&var_tmp[i * strct_global_paras.nDim], &pFront[i * strct_global_paras.nObj]);
                    //transform_var_feature(&var_tmp[i*strct_global_paras.nDim], &strct_repo_info.var_feature[i*TH_N_FEATURE], 1);
                    //testAccuracy(&strct_repo_info.var_feature[i*TH_N_FEATURE], &pFront[i*strct_global_paras.nObj]);
                    testAccuracy(&st_archive_p.var_feature[i * TH_N_FEATURE], &pFront[i * st_global_p.nObj], MPI_COMM_WORLD,
                                 st_MPI_p.mpi_rank,
                                 st_MPI_p.mpi_size);
                } else {
                    f_testAccuracy(&var_tmp[i * st_global_p.nDim], &pFront[i * st_global_p.nObj]);
                }
            }
            break;
        case MY_TYPE_LeNet:
        case MY_TYPE_LeNet_ENSEMBLE:
        case MY_TYPE_LeNet_CLASSIFY_Indus:
            if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE) {
                if(0 == st_MPI_p.mpi_rank) {
                    for(int i = 0; i < size_tmp; i++) {
                        testFuncPtFLT(&var_tmp[i * st_global_p.nDim], &pFront[i * st_global_p.nObj]);
                    }
                }
            } else {
                quo = size_tmp / st_MPI_p.mpi_size;
                rem = size_tmp % st_MPI_p.mpi_size;
                for(int i = 0; i < st_MPI_p.mpi_size; i++) {
                    st_MPI_p.each_size[i] = quo;
                    if(i < rem) st_MPI_p.each_size[i]++;
                }
                update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nDim, st_MPI_p.mpi_size);
                MPI_Scatterv(var_tmp, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                             st_repo_p.var, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                             0, MPI_COMM_WORLD);
                for(int i = 0; i < st_MPI_p.each_size[st_MPI_p.mpi_rank]; i++) {
                    if(st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
                        double tmp[DIM_LeNet];
                        convertVar_CNN(&st_repo_p.var[i * st_global_p.nDim], tmp);
                        testFuncPtFLT(tmp, &st_repo_p.obj[i * st_global_p.nObj]);
                    } else {
                        testFuncPtFLT(&st_repo_p.var[i * st_global_p.nDim], &st_repo_p.obj[i * st_global_p.nObj]);
                    }
                }
                update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nObj, st_MPI_p.mpi_size);
                MPI_Gatherv(st_repo_p.obj, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                            pFront, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
                ///////////////////////////////
                if(validFuncPtFLT) {
                    for(int i = 0; i < st_MPI_p.each_size[st_MPI_p.mpi_rank]; i++) {
                        if(st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
                            double tmp[DIM_LeNet];
                            convertVar_CNN(&st_repo_p.var[i * st_global_p.nDim], tmp);
                            validFuncPtFLT(tmp, &st_repo_p.obj[i * st_global_p.nObj]);
                        } else {
                            validFuncPtFLT(&st_repo_p.var[i * st_global_p.nDim], &st_repo_p.obj[i * st_global_p.nObj]);
                        }
                    }
                    update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nObj, st_MPI_p.mpi_size);
                    MPI_Gatherv(st_repo_p.obj, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                                pFront_valid, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                                0, MPI_COMM_WORLD);
                }
            }
            break;
        case MY_TYPE_EVO1_FRNN:
        case MY_TYPE_EVO2_FRNN:
        case MY_TYPE_EVO3_FRNN:
        case MY_TYPE_EVO4_FRNN:
        case MY_TYPE_EVO5_FRNN:
        case MY_TYPE_EVO_FRNN_PREDICT:
        case MY_TYPE_EVO_DFRNN_PREDICT:
        case MY_TYPE_INTRUSION_DETECTION_CLASSIFY:
        case MY_TYPE_ACTIVITY_DETECTION_CLASSIFY:
        case MY_TYPE_RecSys_SmartCity:
        case MY_TYPE_EVO_CNN:
        case MY_TYPE_EVO_CFRNN:
        case MY_TYPE_NN_CLASSIFY_Indus:
        case MY_TYPE_CFRNN_CLASSIFY:
        case MY_TYPE_EVO_MOBILE_SINK:
            quo = size_tmp / st_MPI_p.mpi_size;
            rem = size_tmp % st_MPI_p.mpi_size;
            for(int i = 0; i < st_MPI_p.mpi_size; i++) {
                st_MPI_p.each_size[i] = quo;
                if(i < rem) st_MPI_p.each_size[i]++;
            }
            update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nDim, st_MPI_p.mpi_size);
            //
            //char tmp_fnm[1024];
            //sprintf(tmp_fnm, "Sizes_MPI%d", strct_MPI_info.mpi_rank);
            //FILE* tmp_fpt = fopen(tmp_fnm, "w");
            //fprintf(tmp_fpt, "%d \n", size_tmp);
            //for (int i = 0; i < strct_MPI_info.mpi_size; i++) {
            //    fprintf(tmp_fpt, "%d ", strct_MPI_info.each_size[i]);
            //}
            //fprintf(tmp_fpt, "\n");
            //for (int i = 0; i < strct_MPI_info.mpi_size; i++) {
            //    fprintf(tmp_fpt, "%d ", strct_MPI_info.recv_size[i]);
            //}
            //fprintf(tmp_fpt, "\n");
            //for (int i = 0; i < strct_MPI_info.mpi_size; i++) {
            //    fprintf(tmp_fpt, "%d ", strct_MPI_info.disp_size[i]);
            //}
            //fprintf(tmp_fpt, "\n");
            //fclose(tmp_fpt);
            //
            MPI_Scatterv(var_tmp, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                         st_repo_p.var, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                         0, MPI_COMM_WORLD);
            ///////////////////////////////
            for(int i = 0; i < st_MPI_p.each_size[st_MPI_p.mpi_rank]; i++) {
                testFuncPtFLT(&st_repo_p.var[i * st_global_p.nDim], &st_repo_p.obj[i * st_global_p.nObj]);
            }
            update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nObj, st_MPI_p.mpi_size);
            MPI_Gatherv(st_repo_p.obj, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                        pFront, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            ///////////////////////////////
            if(validFuncPtFLT) {
                for(int i = 0; i < st_MPI_p.each_size[st_MPI_p.mpi_rank]; i++) {
                    validFuncPtFLT(&st_repo_p.var[i * st_global_p.nDim], &st_repo_p.obj[i * st_global_p.nObj]);
                }
                update_recv_disp_simp(st_MPI_p.each_size, st_global_p.nObj, st_MPI_p.mpi_size);
                MPI_Gatherv(st_repo_p.obj, st_MPI_p.recv_size[st_MPI_p.mpi_rank], MPI_DOUBLE,
                            pFront_valid, st_MPI_p.recv_size, st_MPI_p.disp_size, MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
            }
            break;
        default:
            if(0 == st_MPI_p.mpi_rank) {
                printf("%s:Problem type error, exiting...\n", AT);
            }
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_PROBLEM_TYPE);
            break;
        }
        //////////////////////////////////////////////////////////////////////////
        if(st_MPI_p.mpi_rank == 0) {
            //////////////////////////////////////////////////////////////////////////
            /////////VALIDATION
            if(validFuncPtFLT) {
                if(finalTag != FINAL_TAG) {
                    sprintf(filename, "PF/trace/%s_FUN_%s_VALIDATION_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                            st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                    st_global_p.fptobj = fopen(filename, "w");
                } else {
                    sprintf(filename, "PF/%s_FUN_%s_VALIDATION_OBJ%d_VAR%d_MPI%d_RUN%d",
                            st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                            st_MPI_p.mpi_size, st_ctrl_p.cur_run);
                    st_global_p.fptobj = fopen(filename, "w");
                }
                save_double(st_global_p.fptobj, pFront_valid, size_tmp, st_global_p.nObj, 0);
                fclose(st_global_p.fptobj);
                double tmp_min[128];
                for(int i = 0; i <= st_global_p.nObj; i++) tmp_min[i] = 6e60;
                for(int i = 0; i < size_tmp; i++) {
                    for(int j = 0; j < st_global_p.nObj; j++) {
                        if(tmp_min[j] > pFront_valid[i * st_global_p.nObj + j])
                            tmp_min[j] = pFront_valid[i * st_global_p.nObj + j];
                    }
                    double tmp_beta = 1;
                    double tmp_F;
                    if(pFront_valid[i * st_global_p.nObj + 0] < 1 && pFront_valid[i * st_global_p.nObj + 1] < 1) {
                        tmp_F = (1 + tmp_beta * tmp_beta) * (1 - pFront_valid[i * st_global_p.nObj + 0]) *
                                (1 - pFront_valid[i * st_global_p.nObj + 1]) /
                                (tmp_beta * tmp_beta * (1 - pFront_valid[i * st_global_p.nObj + 0] + 1 - pFront_valid[i * st_global_p.nObj + 1]));
                        tmp_F = 1 - tmp_F;
                    } else {
                        tmp_F = 1;
                    }
                    if(tmp_min[st_global_p.nObj] > tmp_F)
                        tmp_min[st_global_p.nObj] = tmp_F;
                }
                if(finalTag != FINAL_TAG) {
                    if(st_global_p.nObj < 4)
                        st_indicator_p.mat_HV_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                            generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront_valid, size_tmp, st_global_p.nObj,
                                       st_global_p.nDim, &st_ctrl_p.cur_trace,
                                       st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-VALIDATION: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100 * st_ctrl_p.cur_trace / NTRACE,
                           st_indicator_p.mat_HV_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
                    st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] = tmp_min[0];
                } else {
                    if(st_global_p.nObj < 4)
                        st_indicator_p.mat_HV_all_VALIDATION[st_ctrl_p.cur_run][NTRACE + 1] =
                            generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront_valid, size_tmp, st_global_p.nObj,
                                       st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                    printf("%s-OBJ_%d-DIM_%d-Run_%d-VALIDATION: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                           st_global_p.nDim, st_ctrl_p.cur_run,
                           100, st_indicator_p.mat_HV_all_VALIDATION[st_ctrl_p.cur_run][NTRACE + 1]);
                    st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][NTRACE + 1] = tmp_min[0];
                }
                for(int i = 0; i <= st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
                printf("\n");
            }
            //////////////////////////////////////////////////////////////////////////
            /////////TEST
            if(finalTag != FINAL_TAG) {
                sprintf(filename, "PF/trace/%s_FUN_%s_TEST_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                st_global_p.fptobj = fopen(filename, "w");
            } else {
                sprintf(filename, "PF/%s_FUN_%s_TEST_OBJ%d_VAR%d_MPI%d_RUN%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run);
                st_global_p.fptobj = fopen(filename, "w");
            }
            save_double(st_global_p.fptobj, pFront, size_tmp, st_global_p.nObj, 0);
            fclose(st_global_p.fptobj);
            double tmp_min[128];
            int    tmp_ind[128];
            for(int i = 0; i <= st_global_p.nObj; i++) tmp_min[i] = 6e60;
            for(int i = 0; i < size_tmp; i++) {
                for(int j = 0; j < st_global_p.nObj; j++) {
                    if(tmp_min[j] > pFront[i * st_global_p.nObj + j]) {
                        tmp_min[j] = pFront[i * st_global_p.nObj + j];
                        tmp_ind[j] = i;
                    }
                }
                double tmp_beta = 1;
                double tmp_F;
                if(pFront[i * st_global_p.nObj + 0] < 1 && pFront[i * st_global_p.nObj + 1] < 1) {
                    tmp_F = (1 + tmp_beta * tmp_beta) * (1 - pFront[i * st_global_p.nObj + 0]) * (1 - pFront[i * st_global_p.nObj +
                            1]) /
                            (tmp_beta * tmp_beta * (1 - pFront[i * st_global_p.nObj + 0] + 1 - pFront[i * st_global_p.nObj + 1]));
                    tmp_F = 1 - tmp_F;
                } else {
                    tmp_F = 1;
                }
                if(tmp_min[st_global_p.nObj] > tmp_F) {
                    tmp_min[st_global_p.nObj] = tmp_F;
                    tmp_ind[st_global_p.nObj] = i;
                }
            }
            if(finalTag != FINAL_TAG) {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all_TEST[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, &st_ctrl_p.cur_trace,
                                   st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-TEST:  %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100 * st_ctrl_p.cur_trace / NTRACE,
                       st_indicator_p.mat_HV_all_TEST[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
                st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] = tmp_min[0];
            } else {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all_TEST[st_ctrl_p.cur_run][NTRACE + 1] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-TEST:  %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100, st_indicator_p.mat_HV_all_TEST[st_ctrl_p.cur_run][NTRACE + 1]);
                st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][NTRACE + 1] = tmp_min[0];
            }
            for(int i = 0; i < st_global_p.nObj; i++) {
                printf("%lf (", tmp_min[i]);
                for(int j = 0; j < st_global_p.nObj; j++) {
                    printf("%lf ", pFront[tmp_ind[i] * st_global_p.nObj + j]);
                }
                printf(") ");
            }
            printf("%lf", tmp_min[st_global_p.nObj]);
            printf("\n");
            //////////////////////////////////////////////////////////////////////////
            /////////FINAL
            for(int i = 0; i < size_tmp; i++) {
                for(int j = 0; j < st_global_p.nObj; j++) {
                    pFront[i * st_global_p.nObj + j] = 0.5 * pFront[i * st_global_p.nObj + j] + 0.5 * obj_tmp[i *
                                                       st_global_p.nObj + j];
                }
            }
            if(finalTag != FINAL_TAG) {
                sprintf(filename, "PF/trace/%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d_key%d",
                        st_global_p.algorithmName, st_global_p.testInstance, st_global_p.nObj, st_global_p.nDim,
                        st_MPI_p.mpi_size, st_ctrl_p.cur_run, st_ctrl_p.cur_trace);
                st_global_p.fptobj = fopen(filename, "w");
            } else {
                st_global_p.fptobj = fopen(st_global_p.objsal, "w");
            }
            save_double(st_global_p.fptobj, pFront, size_tmp, st_global_p.nObj, 0);
            fclose(st_global_p.fptobj);
            for(int i = 0; i <= st_global_p.nObj; i++) tmp_min[i] = 6e60;
            for(int i = 0; i < size_tmp; i++) {
                for(int j = 0; j < st_global_p.nObj; j++) {
                    if(tmp_min[j] > pFront[i * st_global_p.nObj + j])
                        tmp_min[j] = pFront[i * st_global_p.nObj + j];
                }
                double tmp_beta = 1;
                double tmp_F;
                if(pFront[i * st_global_p.nObj + 0] < 1 && pFront[i * st_global_p.nObj + 1] < 1) {
                    tmp_F = (1 + tmp_beta * tmp_beta) * (1 - pFront[i * st_global_p.nObj + 0]) * (1 - pFront[i * st_global_p.nObj +
                            1]) /
                            (tmp_beta * tmp_beta * (1 - pFront[i * st_global_p.nObj + 0] + 1 - pFront[i * st_global_p.nObj + 1]));
                    tmp_F = 1 - tmp_F;
                } else {
                    tmp_F = 1;
                }
                if(tmp_min[st_global_p.nObj] > tmp_F)
                    tmp_min[st_global_p.nObj] = tmp_F;
            }
            if(finalTag != FINAL_TAG) {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, &st_ctrl_p.cur_trace,
                                   st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-FINAL: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100 * st_ctrl_p.cur_trace / NTRACE, st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][st_ctrl_p.cur_trace]);
            } else {
                if(st_global_p.nObj < 4)
                    st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][NTRACE + 1] =
                        generateHV(st_global_p.testInstance, st_global_p.algorithmName, pFront, size_tmp, st_global_p.nObj,
                                   st_global_p.nDim, NULL, st_ctrl_p.cur_run);
                printf("%s-OBJ_%d-DIM_%d-Run_%d-FINAL: %3d%% --- %lf - ", st_global_p.testInstance, st_global_p.nObj,
                       st_global_p.nDim, st_ctrl_p.cur_run,
                       100, st_indicator_p.mat_HV_all[st_ctrl_p.cur_run][NTRACE + 1]);
            }
            for(int i = 0; i <= st_global_p.nObj; i++) printf("%lf ", tmp_min[i]);
            printf("\n");
        }
        free(pFront_valid);
        free(pFront);
        //////////////////////////////////////////////////////////////////////////
        /////////VALIDATION
        if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus && validFuncPtFLT) {
            //MPI_Barrier(MPI_COMM_WORLD); if (strct_MPI_info.mpi_rank == 0)printf("mat_minPrc_all_TRAIN.\n");
            MPI_Bcast(&st_indicator_p.mat_minPrc_all_TRAIN[st_ctrl_p.cur_run][st_ctrl_p.cur_trace], 1, MPI_DOUBLE, 0,
                      MPI_COMM_WORLD);
            MPI_Bcast(&st_indicator_p.mat_minPrc_all_VALIDATION[st_ctrl_p.cur_run][st_ctrl_p.cur_trace], 1, MPI_DOUBLE, 0,
                      MPI_COMM_WORLD);
        }
    } else {
        if(0 == st_MPI_p.mpi_rank) {
            printf("%s:Problem type error, exiting...\n", AT);
        }
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_PROBLEM_TYPE);
    }

    st_indicator_p.vec_TIME_indicator[st_ctrl_p.cur_run] += (clock() - tmp_time) / CLOCKS_PER_SEC;
    //MPI_Barrier(MPI_COMM_WORLD); if (strct_MPI_info.mpi_rank == strct_MPI_info.mpi_size - 1) printf("show_indicator_vars_simp DONE.\n");

    return;
}

void show_para_A()
{
    if(0 == st_MPI_p.mpi_rank) {
        int fail_tag = 0;
        int count = 0;
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.algo_mech_type:\t");
        switch(st_ctrl_p.algo_mech_type) {
        case LOCALIZATION:
            printf("LOCALIZATION\n");
            break;
        case DECOMPOSITION:
            printf("DECOMPOSITION\n");
            break;
        case NONDOMINANCE:
            printf("NONDOMINANCE\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_test:\t\t");
        switch(st_ctrl_p.type_test) {
        case MY_TYPE_NORMAL:
            printf("MY_TYPE_NORMAL\n");
            break;
        case MY_TYPE_FS_CLASSIFY:
            printf("MY_TYPE_FS_CLASSIFY\n");
            break;
        case MY_TYPE_FS_CLASSIFY_TREE:
            printf("MY_TYPE_FS_CLASSIFY_TREE\n");
            break;
        case MY_TYPE_LeNet:
            printf("MY_TYPE_LeNet\n");
            break;
        case MY_TYPE_LeNet_ENSEMBLE:
            printf("MY_TYPE_LeNet_ENSEMBLE\n");
            break;
        case MY_TYPE_LeNet_CLASSIFY_Indus:
            printf("MY_TYPE_LeNet_CLASSIFY_Indus\n");
            break;
        case MY_TYPE_NN_CLASSIFY_Indus:
            printf("MY_TYPE_NN_CLASSIFY_Indus\n");
            break;
        case MY_TYPE_CFRNN_CLASSIFY:
            printf("MY_TYPE_CFRNN_CLASSIFY\n");
            break;
        case MY_TYPE_EVO1_FRNN:
            printf("MY_TYPE_EVO1_FRNN\n");
            break;
        case MY_TYPE_EVO2_FRNN:
            printf("MY_TYPE_EVO2_FRNN\n");
            break;
        case MY_TYPE_EVO3_FRNN:
            printf("MY_TYPE_EVO3_FRNN\n");
            break;
        case MY_TYPE_EVO4_FRNN:
            printf("MY_TYPE_EVO4_FRNN\n");
            break;
        case MY_TYPE_EVO5_FRNN:
            printf("MY_TYPE_EVO5_FRNN\n");
            break;
        case MY_TYPE_EVO_FRNN_PREDICT:
            printf("MY_TYPE_EVO_FRNN_PREDICT\n");
            break;
        case MY_TYPE_EVO_DFRNN_PREDICT:
            printf("MY_TYPE_EVO_DFRNN_PREDICT\n");
            break;
        case MY_TYPE_INTRUSION_DETECTION_CLASSIFY:
            printf("MY_TYPE_INTRUSION_DETECTION_CLASSIFY\n");
            break;
        case MY_TYPE_ACTIVITY_DETECTION_CLASSIFY:
            printf("MY_TYPE_ACTIVITY_DETECTION_CLASSIFY\n");
            break;
        case MY_TYPE_RecSys_SmartCity:
            printf("MY_TYPE_RecSys_SmartCity\n");
            break;
        case MY_TYPE_EVO_CNN:
            printf("MY_TYPE_EVO_CNN\n");
            break;
        case MY_TYPE_EVO_CFRNN:
            printf("MY_TYPE_EVO_CFRNN\n");
            break;
        case MY_TYPE_EVO_MOBILE_SINK:
            printf("MY_TYPE_EVO_MOBILE_SINK\n");
            break;
        //case MY_TYPE_RS_SC:
        //    printf("MY_TYPE_RS_SC\n");
        //    break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.optimizer_type:\t");
        switch(st_ctrl_p.optimizer_type) {
        case EC_DE_CUR_1:
            printf("EC_DE_CUR_1\n");
            break;
        case EC_DE_CUR_2:
            printf("EC_DE_CUR_2\n");
            break;
        case EC_DE_RAND_1:
            printf("EC_DE_RAND_1\n");
            break;
        case EC_DE_RAND_2:
            printf("EC_DE_RAND_2\n");
            break;
        case EC_DE_ARCHIVE:
            printf("EC_DE_ARCHIVE\n");
            break;
        case EC_DE_ARCHIVE_RAND:
            printf("EC_DE_RCHIVE_RAND\n");
            break;
        case EC_DE_2SELECTED:
            printf("EC_DE_2SELECTED\n");
            break;
        case EC_SBX_CUR:
            printf("EC_SBX_CUR\n");
            break;
        case EC_SBX_RAND:
            printf("EC_SBX_RAND\n");
            break;
        case SI_PSO:
            printf("SI_PSO\n");
            break;
        case SI_QPSO:
            printf("SI_QPSO\n");
            break;
        case EC_MIX_DE_R_SBX_R:
            printf("EC_MIX_DE_R_SBX_R\n");
            break;
        case EC_MIX_DE_C_SBX_R:
            printf("EC_MIX_DE_C_SBX_R\n");
            break;
        case EC_MIX_DE_C_SBX_C:
            printf("EC_MIX_DE_C_SBX_C\n");
            break;
        case EC_MIX_SBX_C_R:
            printf("EC_MIX_SBX_C_R\n");
            break;
        case EC_SI_MIX_DE_C_PSO:
            printf("EC_SI_MIX_DE_C_PSO\n");
            break;
        case EC_MIX_DE_C_R:
            printf("EC_MIX_DE_C_R\n");
            break;
        case EC_MIX_DE_C_1_2:
            printf("EC_MIX_DE_C_1_2\n");
            break;
        case EC_MIX_DE_R_1_2:
            printf("EC_MIX_DE_R_1_2\n");
            break;
        case OPTIMIZER_BLEND:
            printf("OPTIMIZER_BLEND\n");
            break;
        case OPTIMIZER_ENSEMBLE:
            printf("OPTIMIZER_ENSEMBLE\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.DE_F_type:\t");
        switch(st_ctrl_p.DE_F_type) {
        case DE_F_FIXED:
            printf("DE_F_FIXED\n");
            break;
        case DE_F_JADE:
            printf("DE_F_JADE\n");
            break;
        //case DE_F_JADE_UNLIMITED:
        //    printf("DE_F_JADE_UNLIMITED\n");
        //    break;
        case DE_F_SaNSDE:
            printf("DE_F_SaNSDE\n");
            break;
        case DE_F_SaNSDE_a:
            printf("DE_F_SaNSDE_a\n");
            break;
        case DE_F_NSDE:
            printf("DE_F_NSDE\n");
            break;
        case DE_F_SHADE:
            printf("DE_F_SHADE\n");
            break;
        case DE_F_jDE:
            printf("DE_F_jDE\n");
            break;
        case DE_F_DISC:
            printf("DE_F_DISC\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.DE_CR_type:\t");
        switch(st_ctrl_p.DE_CR_type) {
        case DE_CR_FIXED:
            printf("DE_CR_FIXED\n");
            break;
        case DE_CR_LINEAR:
            printf("DE_CR_LINEAR\n");
            break;
        case DE_CR_JADE:
            printf("DE_CR_JADE\n");
            break;
        case DE_CR_SaNSDE:
            printf("DE_CR_SaNSDE\n");
            break;
        case DE_CR_NSDE:
            printf("DE_CR_NSDE\n");
            break;
        case DE_CR_SHADE:
            printf("DE_CR_SHADE\n");
            break;
        case DE_CR_jDE:
            printf("DE_CR_jDE\n");
            break;
        case DE_CR_DISC:
            printf("DE_CR_DISC\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.PSO_para_type:\t");
        switch(st_ctrl_p.PSO_para_type) {
        case PSO_PARA_FIXED:
            printf("PSO_PARA_FIXED\n");
            break;
        case PSO_PARA_ADAP:
            printf("PSO_PARA_ADAP\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.updatePop_type:\t");
        switch(st_ctrl_p.updatePop_type) {
        case UPDATE_POP_MOEAD:
            printf("UPDATE_POP_MOEAD\n");
            break;
        case UPDATE_POP_1TO1:
            printf("UPDATE_POP_1TO1\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_clone_selection:\t");
        switch(st_ctrl_p.type_clone_selection) {
        case CLONE_SLCT_ND1:
            printf("CLONE_SLCT_ND1\n");
            break;
        case CLONE_SLCT_ND2:
            printf("CLONE_SLCT_ND2\n");
            break;
        case CLONE_SLCT_ND_TOUR:
            printf("CLONE_SLCT_ND_TOUR\n");
            break;
        case CLONE_SLCT_UTILITY_TOUR:
            printf("CLONE_SLCT_UTILITY_TOUR\n");
            break;
        case CLONE_SLCT_AGGFIT_G:
            printf("CLONE_SLCT_AGGFIT_G\n");
            break;
        case CLONE_SLCT_AGGFIT_L:
            printf("CLONE_SLCT_AGGFIT_L\n");
            break;
        case CLONE_SLCT_PREFER:
            printf("CLONE_SLCT_PREFER\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.tag_prefer_which_obj:\t");
        switch(st_ctrl_p.tag_prefer_which_obj) {
        case PREFER_FIRST_OBJ:
            printf("PREFER_FIRST_OBJ\n");
            break;
        case PREFER_SECOND_OBJ:
            printf("PREFER_SECOND_OBJ\n");
            break;
        case PREFER_THIRD_OBJ:
            printf("PREFER_THIRD_OBJ\n");
            break;
        case PREFER_NONE_OBJ:
            printf("PREFER_NONE_OBJ\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_clone_evo:\t");
        switch(st_ctrl_p.type_clone_evo) {
        case CLONE_EVO_LOCAL:
            printf("CLONE_EVO_LOCAL\n");
            break;
        case CLONE_EVO_GLOBAL:
            printf("CLONE_EVO_GLOBAL\n");
            break;
        case CLONE_EVO_NONE:
            printf("CLONE_EVO_NONE\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.multiPop_mode:\t");
        switch(st_ctrl_p.multiPop_mode) {
        case MP_0:
            printf("MP_0\n");
            break;
        case MP_I:
            printf("MP_I\n");
            break;
        case MP_II:
            printf("MP_II\n");
            break;
        case MP_III:
            printf("MP_III\n");
            break;
        case MP_ADAP:
            printf("MP_ADAP\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_join_xor:\t");
        switch(st_ctrl_p.type_join_xor) {
        case JOIN_XOR_UTILITY:
            printf("JOIN_XOR_UTILITY\n");
            break;
        case JOIN_XOR_AGGFIT:
            printf("JOIN_XOR_AGGFIT\n");
            break;
        case JOIN_XOR_RAND:
            printf("JOIN_XOR_RAND\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_var_encoding:\t");
        switch(st_ctrl_p.type_var_encoding) {
        case VAR_DOUBLE:
            printf("VAR_DOUBLE\n");
            break;
        case VAR_BINARY:
            printf("VAR_BINARY\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_xor_rem_vars:\t");
        switch(st_ctrl_p.type_xor_rem_vars) {
        case XOR_REMVARS_COPY:
            printf("XOR_REMVARS_COPY\n");
            break;
        case XOR_REMVARS_INHERIT:
            printf("XOR_REMVARS_INHERIT\n");
            break;
        case XOR_REMVARS_XOR_MIXED:
            printf("XOR_REMVARS_XOR_MIXED\n");
            break;
        case XOR_REMVARS_XOR_POP:
            printf("XOR_REMVARS_XOR_POP\n");
            break;
        case XOR_REMVARS_XOR_SAME_REGION:
            printf("XOR_REMVARS_XOR_SAME_REGION\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_grouping:\t");
        switch(st_ctrl_p.type_grouping) {
        case GROUPING_TYPE_CLASSIFY_NORMAL:
            printf("GROUPING_TYPE_CLASSIFY_NORMAL\n");
            break;
        case GROUPING_TYPE_CLASSIFY_RANDOM:
            printf("GROUPING_TYPE_CLASSIFY_RANDOM\n");
            break;
        case GROUPING_TYPE_CLASSIFY_ANALYS:
            printf("GROUPING_TYPE_CLASSIFY_ANALYS\n");
            break;
        case GROUPING_TYPE_SPECTRAL_CLUSTERING:
            printf("GROUPING_TYPE_SPECTRAL_CLUSTERING\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_feature_adjust:\t");
        switch(st_ctrl_p.type_feature_adjust) {
        case FEATURE_ADJUST_RAND:
            printf("FEATURE_ADJUST_RAND\n");
            break;
        case FEATURE_ADJUST_FILTER_MARKOV:
            printf("FEATURE_ADJUST_FILTER_MARKOV\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_xor_evo_mut:\t");
        switch(st_ctrl_p.type_xor_evo_mut) {
        case XOR_EVO_MUT_FIX:
            printf("XOR_EVO_MUT_FIX\n");
            break;
        case XOR_EVO_MUT_ADAP:
            printf("XOR_EVO_MUT_ADAP\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_xor_CNN:\t");
        switch(st_ctrl_p.type_xor_CNN) {
        case XOR_CNN_NORMAL:
            printf("XOR_CNN_NORMAL\n");
            break;
        case XOR_CNN_LeNet:
            printf("XOR_CNN_LeNet\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_del_var:\t");
        switch(st_ctrl_p.type_del_var) {
        case DEL_NORMAL:
            printf("DEL_NORMAL\n");
            break;
        case DEL_LeNet:
            printf("DEL_LeNet\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_dim_convert:\t");
        switch(st_ctrl_p.type_dim_convert) {
        case DIM_NORMAL:
            printf("DIM_NORMAL\n");
            break;
        case DIM_CONVERT_CNN:
            printf("DIM_CONVERT_CNN\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_limit_exceed_proc:\t");
        switch(st_ctrl_p.type_limit_exceed_proc) {
        case LIMIT_ADJUST:
            printf("LIMIT_ADJUST\n");
            break;
        case LIMIT_TRUNCATION:
            printf("LIMIT_TRUNCATION\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.indicator_tag:\t");
        switch(st_ctrl_p.indicator_tag) {
        case INDICATOR_IGD:
            printf("INDICATOR_IGD\n");
            break;
        case INDICATOR_HV:
            printf("INDICATOR_HV\n");
            break;
        case INDICATOR_IGD_HV:
            printf("INDICATOR_IGD_HV\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.CLONALG_tag:\t");
        switch(st_ctrl_p.CLONALG_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.MFI_update_tag:\t");
        switch(st_ctrl_p.MFI_update_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.ScalePara_tag:\t");
        switch(st_ctrl_p.ScalePara_tag) {
        case SCALE_NONE:
            printf("SCALE_NONE\n");
            break;
        case SCALE_QUANTUM:
            printf("SCALE_QUANTUM\n");
            break;
        case SCALE_LEVY:
            printf("SCALE_LEVY\n");
            break;
        case SCALE_CAUCHY:
            printf("SCALE_CAUCHY\n");
            break;
        case SCALE_GAUSS:
            printf("SCALE_GAUSS\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.type_select_obj_best:\t");
        switch(st_ctrl_p.type_select_obj_best) {
        case SELECT_OBJ_BEST_TOP_N:
            printf("SELECT_OBJ_BEST_TOP_N\n");
            break;
        case SELECT_OBJ_BEST_HISTORY:
            printf("SELECT_OBJ_BEST_HISTORY\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.Qubits_angle_opt_tag:\t");
        switch(st_ctrl_p.Qubits_angle_opt_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.Qubits_transform_tag:\t");
        switch(st_ctrl_p.Qubits_transform_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.commonality_xor_remvar_tag:\t");
        switch(st_ctrl_p.commonality_xor_remvar_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.optimize_binaryVar_as_doubleVar_tag:\t");
        switch(st_ctrl_p.opt_binVar_as_realVar_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- strct_ctrl_para.F_para_limit_tag:\t");
        switch(st_ctrl_p.F_para_limit_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- rand_type:\t");
        switch(rand_type) {
        case RAND_UNIF:
            printf("RAND_UNIF\n");
            break;
        case RAND_CHEBYSHEV:
            printf("RAND_CHEBYSHEV\n");
            break;
        case RAND_PIECEWISE_LINEAR:
            printf("RAND_PIECEWISE_LINEAR\n");
            break;
        case RAND_SINUS:
            printf("RAND_SINUS\n");
            break;
        case RAND_LOGISTIC:
            printf("RAND_LOGISTIC\n");
            break;
        case RAND_CIRCLE:
            printf("RAND_CIRCLE\n");
            break;
        case RAND_GAUSS:
            printf("RAND_GAUSS\n");
            break;
        case RAND_TENT:
            printf("RAND_TENT\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- st_ctrl_p.flag_save_trace_PS:\t");
        switch(st_ctrl_p.flag_save_trace_PS) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("A%02d", count);
        printf("-- st_ctrl_p.collect_pop_type:\t");
        switch(st_ctrl_p.collect_pop_type) {
        case COLLECT_WEIGHTED:
            printf("COLLECT_WEIGHTED\n");
            break;
        case COLLECT_NONDOMINATED:
            printf("COLLECT_NONDOMINATED\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        ////
        //count++;
        //printf("A%02d", count);
        //printf("-- type_sync:\t\t");
        //switch (type_sync) {
        //case SYNC_random:
        //    printf("SYNC_random\n");
        //    break;
        //case SYNC_vonNeumann:
        //    printf("SYNC_vonNeumann\n");
        //    break;
        //default:
        //    printf("Unknown\n");
        //    break;
        //}
        //
        printf("\n");
        //
        if(fail_tag) {
            printf("%s: Setting para_A failed, one or more parameters are %d `Unknown', exiting...\n", AT, fail_tag);
        }
    }

    return;
}

void show_para_B()
{
    if(0 == st_MPI_p.mpi_rank) {
        int fail_tag = 0;
        int count = 0;
        //
        count++;
        printf("B%02d", count);
        printf("-- strct_ctrl_para.mixed_var_types_tag:\t");
        switch(st_ctrl_p.mixed_var_types_tag) {
        case FLAG_ON:
            printf("FLAG_ON\n");
            break;
        case FLAG_OFF:
            printf("FLAG_OFF\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("B%02d", count);
        printf("-- strct_ctrl_para.type_pop_loop:\t");
        switch(st_ctrl_p.type_pop_loop) {
        case LOOP_NONE:
            printf("LOOP_NONE\n");
            break;
        case LOOP_POP:
            printf("LOOP_POP\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        count++;
        printf("B%02d", count);
        printf("-- strct_ctrl_para.type_grp_loop:\t");
        switch(st_ctrl_p.type_grp_loop) {
        case LOOP_NONE:
            printf("LOOP_NONE\n");
            break;
        case LOOP_GRP:
            printf("LOOP_GRP\n");
            break;
        default:
            printf("Unknown\n");
            fail_tag++;
            break;
        }
        //
        printf("\n");
        //
        if(fail_tag) {
            printf("%s: Setting para_B failed, one or more parameters are %d `Unknown', exiting...\n", AT, fail_tag);
        }
    }

    return;
}

void show_para_Prob()
{
    if(0 == st_MPI_p.mpi_rank) {
        if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
            int fail_tag = 0;
            int count = 0;
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_RAND_Classify_CNN_Indus:\t");
            switch(TAG_RAND_Classify_CNN_Indus) {
            case FLAG_ON_Classify_CNN_Indus:
                printf("FLAG_ON_Classify_CNN_Indus\n");
                break;
            case FLAG_OFF_Classify_CNN_Indus:
                printf("FLAG_OFF_Classify_CNN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus:\t");
            switch(TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus) {
            case FLAG_ON_Classify_CNN_Indus:
                printf("FLAG_ON_Classify_CNN_Indus\n");
                break;
            case FLAG_OFF_Classify_CNN_Indus:
                printf("FLAG_OFF_Classify_CNN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus:\t");
            switch(TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus) {
            case GENERALIZATION_NONE_Classify_CNN_Indus:
                printf("GENERALIZATION_NONE_Classify_CNN_Indus\n");
                break;
            case GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus:
                printf("GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus\n");
                break;
            case GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus:
                printf("GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus\n");
                break;
            case GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus:
                printf("GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus\n");
                break;
            case GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus:
                printf("GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            printf("MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN = %lf\n",
                   MAX_NOISE_LEVEL_MOP_CLASSIFY_CNN);
            //
            printf("\n");
            //
            if(fail_tag) {
                printf("%s: Setting para_Prob failed, one or more parameters are %d `Unknown', exiting...\n", AT, fail_tag);
            }
        }
        if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
            int fail_tag = 0;
            int count = 0;
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_RAND_Classify_NN_Indus:\t");
            switch(TAG_RAND_Classify_NN_Indus) {
            case FLAG_ON_Classify_NN_Indus:
                printf("FLAG_ON_Classify_NN_Indus\n");
                break;
            case FLAG_OFF_Classify_NN_Indus:
                printf("FLAG_OFF_Classify_NN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_NN_Indus:\t");
            switch(TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_NN_Indus) {
            case FLAG_ON_Classify_NN_Indus:
                printf("FLAG_ON_Classify_NN_Indus\n");
                break;
            case FLAG_OFF_Classify_NN_Indus:
                printf("FLAG_OFF_Classify_NN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus:\t");
            switch(TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus) {
            case GENERALIZATION_NONE_Classify_NN_Indus:
                printf("GENERALIZATION_NONE_Classify_NN_Indus\n");
                break;
            case GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_NN_Indus:
                printf("GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_NN_Indus\n");
                break;
            case GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus:
                printf("GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus\n");
                break;
            case GENERALIZATION_ONE_INDEPENDENDT_Classify_NN_Indus:
                printf("GENERALIZATION_ONE_INDEPENDENDT_Classify_NN_Indus\n");
                break;
            case GENERALIZATION_EACH_INDEPENDENDT_Classify_NN_Indus:
                printf("GENERALIZATION_EACH_INDEPENDENDT_Classify_NN_Indus\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            printf("MAX_NOISE_LEVEL_MOP_CLASSIFY_NN = %lf\n",
                   MAX_NOISE_LEVEL_MOP_CLASSIFY_NN);
            //
            printf("\n");
            //
            if(fail_tag) {
                printf("%s: Setting para_Prob failed, one or more parameters are %d `Unknown', exiting...\n", AT, fail_tag);
            }
        }
        if(st_ctrl_p.type_test == MY_TYPE_CFRNN_CLASSIFY) {
            int fail_tag = 0;
            int count = 0;
            //
            count++;
            printf("Prob%02d", count);
            printf("-- TAG_RAND_Classify_NN_Indus:\t");
            switch(CFRNN_MODEL_MOP_CLASSIFY_CFRNN_CUR) {
            case CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I:
                printf("CFRNN_MODEL_MOP_CLASSIFY_CFRNN_I\n");
                break;
            case CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II:
                printf("CFRNN_MODEL_MOP_CLASSIFY_CFRNN_II\n");
                break;
            case CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III:
                printf("CFRNN_MODEL_MOP_CLASSIFY_CFRNN_III\n");
                break;
            case CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV:
                printf("CFRNN_MODEL_MOP_CLASSIFY_CFRNN_IV\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            count++;
            printf("Prob%02d", count);
            printf("-- CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR:\t");
            switch(CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_CUR) {
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_ORIGIN\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_ALL_NORMED\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVERAG\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVERAG\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_AVG_NORM\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_MAP_ALL_AVG_NORM\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_IN_FEATURE_FIX_INPUTS\n");
                break;
            case CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE:
                printf("CFRNN_CONSEQUENCE_MOP_CLASSIFY_CFRNN_NONE\n");
                break;
            default:
                printf("Unknown\n");
                fail_tag++;
                break;
            }
            //
            printf("\n");
            //
            if(fail_tag) {
                printf("%s: Setting para_Prob failed, one or more parameters are %d `Unknown', exiting...\n", AT, fail_tag);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_PROBLEM_PARA);
            }
        }
    }
    //
    return;
}

void show_DeBug_info()
{
    //printf("nDim=%d\n", st_global_p.nDim);
    //printf("ratio_mut=%lf\n", st_optimizer_p.ratio_mut);
    //printf("type_xor_rem_vars=%d\n", st_ctrl_p.type_xor_rem_vars);
    //printf("type_test=%d\n", st_ctrl_p.type_test);
    //printf("commonality_xor_remvar_tag=%d\n", st_ctrl_p.commonality_xor_remvar_tag);
    //printf("type_feature_adjust=%d\n", st_ctrl_p.type_feature_adjust);
    //printf("type_mut_general=%d\n", st_ctrl_p.type_mut_general);
    //printf("type_del_var=%d\n", st_ctrl_p.type_del_var);
    if(st_MPI_p.mpi_rank == 0) {
        //printf("nPop=%d\n", st_global_p.nPop);
        //printf("nPop_mine=%d\n", st_global_p.nPop_mine);
        //printf("nDim=%d\n", st_global_p.nDim);
        //printf("type_xor_rem_vars=%d\n", st_ctrl_p.type_xor_rem_vars);
        ////int* parent_type = st_decomp_p.parent_type;
        ////int* tableNeighbor = st_decomp_p.tableNeighbor;
        ////printf("maxNneighb=%d\n", nPop); //
        //printf("niche=%d\n", st_decomp_p.niche);
        //printf("type_join_xor=%d\n", st_ctrl_p.type_join_xor);
        //printf("color_pop=%d\n", st_MPI_p.color_pop);
        ////double* cur_obj = st_pop_evo_cur.obj;
        ////double* cur_var = st_pop_evo_cur.var;
        //printf("commonality_xor_remvar_tag=%d\n", st_ctrl_p.commonality_xor_remvar_tag);
        ////int* table_mine_flag = st_grp_info_p.table_mine_flag;
        ////int* types_var_all = st_ctrl_p.types_var_all;
        ////double* rate_Commonality = st_optimizer_p.rate_Commonality;
        //printf("type_test=%d\n", st_ctrl_p.type_test);
        //printf("type_xor_CNN=%d\n", st_ctrl_p.type_xor_CNN);
        ////printf("nPop_cur=%d\n", nPop);
        ////printf("nPop_candid_all=%d\n", nPop);
        //printf("algo_mech_type=%d\n", st_ctrl_p.algo_mech_type);
        ////
        ////printf("rank%d-ADDR-%d\n", st_MPI_p.mpi_rank, st_optimizer_p.optimizer_types_all);
        //for(int iID = 0; iID < st_global_p.nPop; iID++)
        //    if(st_optimizer_p.optimizer_types_all[iID] != st_ctrl_p.optimizer_type)
        //        printf("rank%d-%d(%d)", st_MPI_p.mpi_rank, st_optimizer_p.optimizer_types_all[iID], iID);
    }
    //
    return;
}
