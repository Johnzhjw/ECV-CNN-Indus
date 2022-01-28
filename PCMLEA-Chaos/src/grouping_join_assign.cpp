# include "global.h"
# include <iostream>
# include <algorithm>

void localAssignGroup(int iPop, int iGrp)
{
    int iGrp_cur = iGrp;
    if(st_ctrl_p.optimization_tag == OPTIMIZE_DIVER_VARS) {
        iGrp_cur = 0;
    } else {
        if(st_ctrl_p.opt_diverVar_separately == FLAG_ON)
            iGrp_cur = iGrp + 1;
    }
    int tmp_ind = iPop * st_global_p.nDim + iGrp_cur;
    st_grp_info_p.table_mine_size = st_grp_info_p.Groups_sub_sizes[tmp_ind];

    for(int i = 0; i < st_global_p.nDim; i++) st_grp_info_p.table_mine_flag[i] = 0;
    for(int i = 0; i < st_global_p.nDim; i++) st_grp_info_p.group_mine_flag[i] = 0;
    st_grp_info_p.group_mine_flag[iGrp_cur] = 1;

    for(int i = 0; i < st_grp_info_p.table_mine_size; i++) {
        int tmp_id = iPop * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_ind] + i;
        st_grp_info_p.table_mine[i] = st_grp_info_p.Groups[tmp_id];
        st_grp_info_p.table_mine_flag[st_grp_info_p.table_mine[i]] = 1;
    }

    int count = 0;
    for(int i = 0; i < st_global_p.nDim; i++) {
        if(!st_grp_info_p.table_mine_flag[i]) {
            st_grp_info_p.table_remain[count++] = i;
        }
    }
    //
    return;
}

void grouping_variables()
{
    exec_DVA();
    // all
    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        int i, j;
        int* flag_single = (int*)calloc(st_global_p.nDim, sizeof(int));
        int  num_single = 0;
        int* flag_group_id = (int*)calloc(st_global_p.nDim, sizeof(int));
        for(i = 0; i < st_global_p.nDim; i++) {
            flag_single[i] = 0;
            flag_group_id[i] = -1;
        }

        for(i = 0; i < st_grp_info_p.Groups_raw_sizes[iObj]; i++) {
            int tmp_i = iObj * st_global_p.nDim + i;
            for(j = 0; j < st_grp_info_p.Groups_raw_sub_sizes[tmp_i]; j++) {
                int tmp_ind = iObj * st_global_p.nDim + st_grp_info_p.Groups_raw_sub_disps[tmp_i] + j;
                int tmp_var = st_grp_info_p.Groups_raw[tmp_ind];
                flag_group_id[tmp_var] = i;
                if(st_grp_info_p.Groups_raw_sub_sizes[tmp_i] == 1) {
                    flag_single[tmp_var] = 1;
                    num_single++;
                }
            }
        }

        int max_size = st_grp_info_p.maxGroupSize;
        int min_size = st_grp_info_p.minGroupSize;
        if(num_single) {
            int tmp_size = num_single;
            int tmp_num = 1;
            while((tmp_size / tmp_num) > min_size) tmp_num++;
            int quo = tmp_size / tmp_num;
            int rem = tmp_size % tmp_num;

            int cur_group_id = -1;
            for(i = 0; i < st_global_p.nDim; i++) {
                if(flag_single[i]) {
                    cur_group_id = flag_group_id[i];
                    break;
                }
            }
            for(int iNum = 0; iNum < tmp_num; iNum++) {
                int cur_size = quo;
                if(iNum < rem) cur_size++;
                int tmp_count = 0;
                for(i = 0; i < st_global_p.nDim; i++) {
                    if(tmp_count >= cur_size)
                        break;
                    if(flag_single[i]) {
                        flag_single[i] = 0;
                        flag_group_id[i] = cur_group_id;
                        tmp_count++;
                    }
                }
                for(i = 0; i < st_global_p.nDim; i++) {
                    if(flag_single[i]) {
                        cur_group_id = flag_group_id[i];
                        break;
                    }
                }
            }
        }

        int* tmp_group_sizes = (int*)calloc(st_global_p.nDim, sizeof(int));
        for(i = 0; i < st_global_p.nDim; i++) tmp_group_sizes[i] = 0;
        for(i = 0; i < st_global_p.nDim; i++) {
            if(flag_group_id[i] >= 0 /*&& flag_single[i] == 0*/) {
                tmp_group_sizes[flag_group_id[i]]++;
            }
        }

        bool changed = true;
        int min1, min2;
        int minI1, minI2;
        while(changed) {
            changed = false;

            min1 = min2 = st_global_p.nDim;
            minI1 = minI2 = -1;
            for(i = 0; i < st_global_p.nDim; i++) {
                if(tmp_group_sizes[i] > 0 &&
                   tmp_group_sizes[i] < min2) {
                    if(tmp_group_sizes[i] > 0 &&
                       tmp_group_sizes[i] < min1) {
                        min2 = min1;
                        minI2 = minI1;
                        min1 = tmp_group_sizes[i];
                        minI1 = i;
                    } else {
                        min2 = tmp_group_sizes[i];
                        minI2 = i;
                    }
                }
            }

            if(minI1 != -1 && minI2 != -1 &&
               min1 + min2 <= st_grp_info_p.minGroupSize) {
                changed = true;

                tmp_group_sizes[minI2] += tmp_group_sizes[minI1];
                tmp_group_sizes[minI1] = 0;

                for(j = 0; j < st_global_p.nDim; j++) {
                    if(flag_group_id[j] == minI1) {
                        flag_group_id[j] = minI2;
                    }
                }
            }
        }

        int* flag_split_num = (int*)calloc(st_global_p.nDim, sizeof(int));
        for(i = 0; i < st_global_p.nDim; i++) tmp_group_sizes[i] = 0;
        for(i = 0; i < st_global_p.nDim; i++) {
            flag_split_num[i] = 1;
            if(flag_group_id[i] >= 0 /*&& flag_single[i] == 0*/) {
                tmp_group_sizes[flag_group_id[i]]++;
            }
        }

        //int max_index;
        for(i = 0; i < st_global_p.nDim; i++) {
            if(tmp_group_sizes[i] > max_size) {
                int tmp_size = tmp_group_sizes[i];
                int tmp_num = 1;
                while((tmp_size / tmp_num) > min_size) tmp_num++;
                flag_split_num[i] = tmp_num;
            }
        }

        int count_var = 0;
        int count_grp = 0;

        //Diversity variables
        for(i = 0; i < st_grp_ana_p.numDiverIndexes; i++) {
            int tmp_idx = iObj * st_global_p.nDim + count_var;
            st_grp_info_p.Groups[tmp_idx] = st_grp_info_p.DiversityIndexs[i];
            count_var++;
        }
        if(st_grp_ana_p.numDiverIndexes) {
            int tmp_idx = iObj * st_global_p.nDim + count_grp;
            st_grp_info_p.Groups_sub_sizes[tmp_idx] = st_grp_ana_p.numDiverIndexes;
            if(count_grp == 0) {
                st_grp_info_p.Groups_sub_disps[tmp_idx] = 0;
            } else {
                st_grp_info_p.Groups_sub_disps[tmp_idx] =
                    st_grp_info_p.Groups_sub_sizes[tmp_idx - 1] +
                    st_grp_info_p.Groups_sub_disps[tmp_idx - 1];
            }
            count_grp++;
        }

        //Convergence variables
        for(i = 0; i < st_global_p.nDim; i++) {
            if(tmp_group_sizes[i] > 0) {
                int tmp_size = tmp_group_sizes[i];
                int tmp_num = flag_split_num[i];
                int quo = tmp_size / tmp_num;
                int rem = tmp_size % tmp_num;

                for(j = 0; j < st_global_p.nDim; j++) {
                    if(flag_group_id[j] == i) {
                        int tmp_idx = iObj * st_global_p.nDim + count_var;
                        st_grp_info_p.Groups[tmp_idx] = j;
                        count_var++;
                    }
                }

                for(int iNum = 0; iNum < tmp_num; iNum++) {
                    int tmp_idx = iObj * st_global_p.nDim + count_grp;
                    st_grp_info_p.Groups_sub_sizes[tmp_idx] = quo;
                    if(iNum < rem) st_grp_info_p.Groups_sub_sizes[tmp_idx]++;
                    if(count_grp == 0) {
                        st_grp_info_p.Groups_sub_disps[tmp_idx] = 0;
                    } else {
                        st_grp_info_p.Groups_sub_disps[tmp_idx] =
                            st_grp_info_p.Groups_sub_sizes[tmp_idx - 1] +
                            st_grp_info_p.Groups_sub_disps[tmp_idx - 1];
                    }
                    count_grp++;
                }
            }
        }
        st_grp_info_p.Groups_sizes[iObj] = count_grp;

        free(flag_single);
        free(flag_group_id);
        free(tmp_group_sizes);
        free(flag_split_num);
    }

    return;
}

void grouping_variables_unif(int numGrps, int flag_rand)
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = numGrps;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            int tmp_size = st_global_p.nDim;
            int tmp_num = tmp_nGroup;
            int quo = tmp_size / tmp_num;
            int rem = tmp_size % tmp_num;

            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[iObj * st_global_p.nDim + i] = i;
            }
            if(flag_rand)
                shuffle(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_global_p.nDim);

            st_grp_info_p.Groups_sizes[iObj] = tmp_num;

            for(int i = 0; i < tmp_num; i++) {
                int tmp_ind = iObj * st_global_p.nDim + i;
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = quo;
                if(i < rem) st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                if(i == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                //sort
                for(int a = st_grp_info_p.Groups_sub_sizes[tmp_ind] - 1; a > 0; a--) {
                    for(int b = 0; b < a; b++) {
                        int tmp_offset = iObj * st_global_p.nDim +
                                         st_grp_info_p.Groups_sub_disps[tmp_ind];
                        if(st_grp_info_p.Groups[tmp_offset + b] > st_grp_info_p.Groups[tmp_offset + b + 1]) {
                            int tmp = st_grp_info_p.Groups[tmp_offset + b];
                            st_grp_info_p.Groups[tmp_offset + b] = st_grp_info_p.Groups[tmp_offset + b + 1];
                            st_grp_info_p.Groups[tmp_offset + b + 1] = tmp;
                        }
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_WDCN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 4;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(i / N_RADIO_PER_COL < N_RADIO_PER_ROW / 2) {
                        if(i % N_RADIO_PER_COL < N_RADIO_PER_COL / 2) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        }
                    } else {
                        if(i % N_RADIO_PER_COL < N_RADIO_PER_COL / 2) {
                            if(iNum == 2) {
                                tmp_flag = 1;
                            }
                        } else {
                            if(iNum == 3) {
                                tmp_flag = 1;
                            }
                        }
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_ARRANGE2D()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 4;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(i / N_COL_ARRANGE2D < N_ROW_ARRANGE2D / 2) {
                        if(i % N_COL_ARRANGE2D < N_COL_ARRANGE2D / 2) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        }
                    } else {
                        if(i % N_COL_ARRANGE2D < N_COL_ARRANGE2D / 2) {
                            if(iNum == 2) {
                                tmp_flag = 1;
                            }
                        } else {
                            if(iNum == 3) {
                                tmp_flag = 1;
                            }
                        }
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    return;
}

void grouping_variables_HDSN_URBAN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(i < n_sensor_URBAN * UNIT_URBAN) {
                        if(i % 4 < 2) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        }
                    } else {
                        if(iNum == 0) {
                            tmp_flag = 1;
                        }
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_RS_SC()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 3;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = N_USR_RS_SC * N_IMP_RS_SC;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = N_LOC_RS_SC * N_IMP_RS_SC;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = N_ACT_RS_SC * N_IMP_RS_SC + N_FEA_RS_SC * N_IMP_RS_SC + 1;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_IWSN_S_1F()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_offset = iObj * st_global_p.nDim;
            int cur_count = 0;
            for(int i = 0; i < N_DIREC_S_1F; i++) {
                int cur_ind = cur_offset + cur_count;
                st_grp_info_p.Groups[cur_ind++] = i * D_DIREC_S_1F;
                st_grp_info_p.Groups[cur_ind] = i * D_DIREC_S_1F + 1;
                cur_count += 2;
            }
            for(int i = 0; i < N_RELAY_S_1F; i++) {
                int cur_ind = cur_offset + cur_count;
                st_grp_info_p.Groups[cur_ind++] = N_DIREC_S_1F * D_DIREC_S_1F + i * D_RELAY_S_1F;
                st_grp_info_p.Groups[cur_ind] = N_DIREC_S_1F * D_DIREC_S_1F + i * D_RELAY_S_1F + 1;
                cur_count += 2;
            }
            for(int i = 0; i < N_DIREC_S_1F; i++) {
                int cur_ind = cur_offset + cur_count;
                st_grp_info_p.Groups[cur_ind++] = i * D_DIREC_S_1F + 2;
                st_grp_info_p.Groups[cur_ind] = i * D_DIREC_S_1F + 3;
                cur_count += 2;
            }
            //for(int i = 0; i < strct_global_paras.nDim; i++) {
            //    int cur_ind = iObj * strct_global_paras.nDim + i;
            //    strct_grp_info_vals.Groups[cur_ind] = i;
            //}

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                if(iNum == 0)
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = (N_DIREC_S_1F + N_RELAY_S_1F) * 2;
                else
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = N_DIREC_S_1F * 2;
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_IWSN_S_1F_6()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 6;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_offset = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                int cur_ind = cur_offset + i;
                st_grp_info_p.Groups[cur_ind] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                if(iNum < tmp_nGroup - 1)
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = D_DIREC_S_1F * 6;
                else
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = N_RELAY_S_1F * 2;
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_IWSN_S_1F_with_nG(int numGROUP)
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = numGROUP;

    int quo_sn_in_1 = N_DIREC_S_1F / tmp_nGroup;
    int rem_sn_in_1 = N_DIREC_S_1F % tmp_nGroup;
    int quo_rn_in_1 = N_RELAY_S_1F / tmp_nGroup;
    int rem_rn_in_1 = N_RELAY_S_1F % tmp_nGroup;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_offset = iObj * st_global_p.nDim;

            int tmp_offset_sensor = 0;
            int tmp_offset_relay = N_DIREC_S_1F * D_DIREC_S_1F;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_size = 0;
                int tmp;
                tmp = quo_sn_in_1;
                if(iNum < rem_sn_in_1)
                    tmp++;
                for(int i = 0; i < tmp * D_DIREC_S_1F; i++) {
                    int cur_ind = cur_offset++;
                    st_grp_info_p.Groups[cur_ind] = tmp_offset_sensor++;
                    ++tmp_size;
                }
                tmp = quo_rn_in_1;
                if(iNum < rem_rn_in_1)
                    tmp++;
                for(int i = 0; i < tmp * D_RELAY_S_1F; i++) {
                    int cur_ind = cur_offset++;
                    st_grp_info_p.Groups[cur_ind] = tmp_offset_relay++;
                    ++tmp_size;
                }

                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                st_grp_info_p.Groups_sub_sizes[tmp_ind] = tmp_size;
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EdgeComputation()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 0;
    int* var_clusterIDs = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* classSizes = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* clusterIDs = (int*)calloc(st_global_p.nDim, sizeof(int));
    for(int i = 0; i < st_global_p.nDim; i++) {
        classSizes[i] = 0;
    }
    //////////////////////////////////////////////////////////////////////////
    char file[1024];
    FILE* ifs;
    int max_buf_size = 1000 * 20 + 1;
    char tmp_delim[] = " ,\t\r\n";
    char* buff = (char*)calloc(max_buf_size, sizeof(char));
    char* p;
    sprintf(file, "../Data_all/Data_EdgeComputation/%dall.csv", RSU_SET_CARDINALITY_CUR);
    ifs = fopen(file, "r");
    if(!ifs) {
        printf("%s(%d): Open file %s error, exiting...\n",
               __FILE__, __LINE__, file);
        exit(-111007);
    }
    if(!fgets(buff, max_buf_size, ifs)) {
        printf("%s(%d): Not enough data for file %s, exiting...\n",
               __FILE__, __LINE__, file);
        exit(-111006);
    }
    int tmp_ind = 0;
    while(fgets(buff, max_buf_size, ifs)) {
        int k = 0;
        for(p = strtok(buff, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
            int tmp_val;
            if(sscanf(p, "%d", &tmp_val) != 1) {
                printf("%s(%d): No more data file %s, exiting...\n",
                       __FILE__, __LINE__, file);
                exit(-111006);
            }
            if(k == 5) {
                var_clusterIDs[tmp_ind++] = tmp_val - 1;
                if(tmp_val > st_global_p.nDim || tmp_val < 1) {
                    printf("%s(%d): Invalid cluster ID for row %d (%d not in %d~%d) file %s, exiting...\n",
                           __FILE__, __LINE__, tmp_ind, tmp_val, 1, st_global_p.nDim, file);
                    exit(-1110055);
                }
                classSizes[tmp_val - 1]++;
                //printf("var_c_ID - %d ", tmp_val - 1);
            }
            k++;
        }
    }
    //printf("\n");
    if(tmp_ind != st_global_p.nDim) {
        printf("%s(%d): The number of IDs is not consistent withe the setting (%d != %d), exiting...\n",
               __FILE__, __LINE__, tmp_ind, st_global_p.nDim);
        exit(-111006);
    }
    fclose(ifs);
    free(buff);
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < st_global_p.nDim; i++) {
        if(classSizes[i]) {
            //printf("size - %d\n", classSizes[i]);
            clusterIDs[tmp_nGroup++] = i;
        }
    }
    for(int i = 0; i < tmp_nGroup; i++) {
        //printf("%s(%d): ID - %d Size - %d\n",
        //       __FILE__, __LINE__, clusterIDs[i], classSizes[clusterIDs[i]]);
    }
    //////////////////////////////////////////////////////////////////////////
    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_offset = iObj * st_global_p.nDim;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_size = 0;

                int curTagID = clusterIDs[iNum];
                for(int j = 0; j < st_global_p.nDim; j++) {
                    if(var_clusterIDs[j] == curTagID) {
                        int cur_ind = cur_offset++;
                        st_grp_info_p.Groups[cur_ind] = j;
                        tmp_size++;
                        //printf("%d ", j);
                    }
                }
                //printf("\n");

                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = tmp_size;
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    //
    free(var_clusterIDs);
    free(classSizes);
    free(clusterIDs);
    //
    return;
}

void grouping_variables_LeNet()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 1 + NUM_CHANNEL_C3_OUT / 2;

    if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE)
        tmp_nGroup = 3;
    else if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
            st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN)
        tmp_nGroup = 2;
    int tmp_offset = DIM_ALL_PARA_CNN;
    if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
       st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN)
        tmp_offset = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS * st_global_p.nDim_MAP +
                     NUM_PARA_C3_B + NUM_PARA_O5;
#if OPTIMIZE_STRUCTURE_CNN == 1
    int ind_var_struc_beg = tmp_offset;
    tmp_offset += DIM_ALL_STRU_CNN;
    int ind_var_struc_end = tmp_offset;
    int ind_grp_struc = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet ||
       st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
    int ind_var_pixel_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_pixel_end = tmp_offset;
    int ind_grp_pixel = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    if(0 == st_MPI_p.mpi_rank) {
        printf("NG = %d\n", tmp_nGroup);
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE) {
                    int iGroup;
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        iGroup = i / st_global_p.nPop;
                        if(iNum == iGroup) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                } else if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
                          st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
                    int thresh_ori4 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                                      st_global_p.nDim_MAP + NUM_PARA_C3_B;
                    int thresh_ori5 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                                      st_global_p.nDim_MAP + NUM_PARA_C3_B + NUM_PARA_O5;
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        int tmp_flag = 0;
                        if(i < thresh_ori4) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else if(i < thresh_ori5) {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        } else {
#if OPTIMIZE_STRUCTURE_CNN == 1
                            if(i >= ind_var_struc_beg && i < ind_var_struc_end) {
                                if(iNum == ind_grp_struc) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
                            if(i >= ind_var_pixel_beg && i < ind_var_pixel_end) {
                                if(iNum == ind_grp_pixel) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#endif
                        }
                        if(tmp_flag) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                } else {
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        int tmp_flag = 0;
                        if(i < NUM_PARA_C1) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else if(i < NUM_PARA_C1 + NUM_PARA_C3) {
                            int tmp_i = (i - NUM_PARA_C1) / NUM_PARA_C3_U / 2 + 1;
                            if(iNum == tmp_i) {
                                tmp_flag = 1;
                            }
                        } else if(i < DIM_ALL_PARA_CNN) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else {
#if OPTIMIZE_STRUCTURE_CNN == 1
                            if(i >= ind_var_struc_beg && i < ind_var_struc_end) {
                                if(iNum == ind_grp_struc) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
                            if(i >= ind_var_pixel_beg && i < ind_var_pixel_end) {
                                if(iNum == ind_grp_pixel) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#endif
                        }
                        if(tmp_flag) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_LeNet_less()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE)
        tmp_nGroup = 3;
    else if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
            st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN)
        tmp_nGroup = 2;
    int tmp_offset = DIM_ALL_PARA_CNN;
    if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
       st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN)
        tmp_offset = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS * st_global_p.nDim_MAP +
                     NUM_PARA_C3_B + NUM_PARA_O5;
#if OPTIMIZE_STRUCTURE_CNN == 1
    int ind_var_struc_beg = tmp_offset;
    tmp_offset += DIM_ALL_STRU_CNN;
    int ind_var_struc_end = tmp_offset;
    int ind_grp_struc = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet ||
       st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
    int ind_var_pixel_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_pixel_end = tmp_offset;
    int ind_grp_pixel = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_CNN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    if(0 == st_MPI_p.mpi_rank) {
        printf("NG = %d\n", tmp_nGroup);
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                if(st_ctrl_p.type_test == MY_TYPE_LeNet_ENSEMBLE) {
                    int iGroup;
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        iGroup = i / st_global_p.nPop;
                        if(iNum == iGroup) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                } else if((st_ctrl_p.type_test == MY_TYPE_LeNet || st_ctrl_p.type_test == MY_TYPE_LeNet_CLASSIFY_Indus) &&
                          st_ctrl_p.type_dim_convert == DIM_CONVERT_CNN) {
                    int thresh_ori4 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                                      st_global_p.nDim_MAP + NUM_PARA_C3_B;
                    int thresh_ori5 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                                      st_global_p.nDim_MAP + NUM_PARA_C3_B + NUM_PARA_O5;
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        int tmp_flag = 0;
                        if(i < thresh_ori4) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else if(i < thresh_ori5) {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        } else {
#if OPTIMIZE_STRUCTURE_CNN == 1
                            if(i >= ind_var_struc_beg && i < ind_var_struc_end) {
                                if(iNum == ind_grp_struc) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
                            if(i >= ind_var_pixel_beg && i < ind_var_pixel_end) {
                                if(iNum == ind_grp_pixel) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#endif
                        }
                        if(tmp_flag) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                } else {
                    for(int i = 0; i < st_global_p.nDim; i++) {
                        int tmp_flag = 0;
                        if(i < NUM_PARA_C1) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else if(i < NUM_PARA_C1 + NUM_PARA_C3) {
                            if(iNum == 1) {
                                tmp_flag = 1;
                            }
                        } else if(i < DIM_ALL_PARA_CNN) {
                            if(iNum == 0) {
                                tmp_flag = 1;
                            }
                        } else {
#if OPTIMIZE_STRUCTURE_CNN == 1
                            if(i >= ind_var_struc_beg && i < ind_var_struc_end) {
                                if(iNum == ind_grp_struc) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_CNN_Indus == FLAG_ON_Classify_CNN_Indus
                            if(i >= ind_var_pixel_beg && i < ind_var_pixel_end) {
                                if(iNum == ind_grp_pixel) {
                                    tmp_flag = 1;
                                }
                            }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_CNN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_CNN_Indus
                            if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                                if(iNum == ind_grp_generalization) {
                                    tmp_flag = 1;
                                }
                            }
#endif
                        }
                        if(tmp_flag) {
                            int cur_ind = iObj * st_global_p.nDim +
                                          st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                          st_grp_info_p.Groups_sub_sizes[tmp_ind];
                            st_grp_info_p.Groups[cur_ind] = i;
                            st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                        }
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("void grouping_variables_LeNet_less()\n");
    //}

    return;
}

void grouping_variables_NN_indus()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    int tmp_offset = DIM_ALL_PARA_NN;

#if OPTIMIZE_STRUCTURE_NN == 1
    int ind_var_struc_beg = tmp_offset;
    tmp_offset += DIM_ALL_STRU_NN;
    int ind_var_struc_end = tmp_offset;
    int ind_grp_struc = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        tmp_nGroup += 2;
    }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_NN_Indus == FLAG_ON_Classify_NN_Indus
    int ind_var_pixel_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_NN_Indus;
    int ind_var_pixel_end = tmp_offset;
    int ind_grp_pixel = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_NN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_NN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_NN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += 1;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = 0;// tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        //tmp_nGroup++;
    }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_NN_Indus
    int ind_var_generalization_beg = tmp_offset;
    tmp_offset += NUM_feature_Classify_NN_Indus;
    int ind_var_generalization_end = tmp_offset;
    int ind_grp_generalization = tmp_nGroup;
    if(st_ctrl_p.type_test == MY_TYPE_NN_CLASSIFY_Indus) {
        tmp_nGroup++;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    if(0 == st_MPI_p.mpi_rank) {
        printf("NG = %d\n", tmp_nGroup);
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                int num_para_O1 =
                    NN_Classify->O1->outputNum * NN_Classify->O1->inputNum + NN_Classify->O1->outputNum;
                int num_para_O2 =
                    NN_Classify->O2->outputNum * NN_Classify->O2->inputNum + NN_Classify->O2->outputNum;
                int num_para_O3 =
                    NN_Classify->O3->outputNum * NN_Classify->O3->inputNum + NN_Classify->O3->outputNum;
                int num_conn_O1 =
                    NN_Classify->O1->outputNum * NN_Classify->O1->inputNum;
                int num_conn_O2 =
                    NN_Classify->O2->outputNum * NN_Classify->O2->inputNum;
                int num_conn_O3 =
                    NN_Classify->O3->outputNum * NN_Classify->O3->inputNum;

                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(i < num_para_O1) {
                        if(iNum == 0) {
                            tmp_flag = 1;
                        }
                    } else if(i < num_para_O1 + num_para_O2) {
                        if(iNum == 1) {
                            tmp_flag = 1;
                        }
                    } else if(i < num_para_O1 + num_para_O2 + num_para_O3) {
                        if(iNum == 1) {
                            tmp_flag = 1;
                        }
                    } else {
#if OPTIMIZE_STRUCTURE_NN == 1
                        if(i < ind_var_struc_beg + num_conn_O1) {
                            if(iNum == ind_grp_struc + 0) {
                                tmp_flag = 1;
                            }
                        } else if(i < ind_var_struc_beg + num_conn_O1 + num_conn_O2) {
                            if(iNum == ind_grp_struc + 1) {
                                tmp_flag = 1;
                            }
                        } else if(i < ind_var_struc_beg + num_conn_O1 + num_conn_O2 + num_conn_O3) {
                            if(iNum == ind_grp_struc + 1) {
                                tmp_flag = 1;
                            }
                        }
#endif
#if TAG_OPTIMIZE_PIXEL_ARRANGEMENT_Classify_NN_Indus == FLAG_ON_Classify_NN_Indus
                        if(i >= ind_var_pixel_beg && i < ind_var_pixel_end) {
                            if(iNum == ind_grp_pixel) {
                                tmp_flag = 1;
                            }
                        }
#endif
#if TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_ONE_DEPENDENDT_ON_RANGE_Classify_NN_Indus
                        if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                            if(iNum == ind_grp_generalization) {
                                tmp_flag = 1;
                            }
                        }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_EACH_DEPENDENDT_ON_RANGE_Classify_NN_Indus
                        if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                            if(iNum == ind_grp_generalization) {
                                tmp_flag = 1;
                            }
                        }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_ONE_INDEPENDENDT_Classify_NN_Indus
                        if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                            if(iNum == ind_grp_generalization) {
                                tmp_flag = 1;
                            }
                        }
#elif TAG_OPTIMIZE_GENERALIZATION_Classify_NN_Indus == GENERALIZATION_EACH_INDEPENDENDT_Classify_NN_Indus
                        if(i >= ind_var_generalization_beg && i < ind_var_generalization_end) {
                            if(iNum == ind_grp_generalization) {
                                tmp_flag = 1;
                            }
                        }
#endif
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("void grouping_variables_LeNet_less()\n");
    //}

    return;
}

void grouping_variables_CFRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    int tmp_offset = NDIM_Classify_CFRNN;

    MPI_Barrier(MPI_COMM_WORLD);
    if(0 == st_MPI_p.mpi_rank) {
        printf("NG = %d\n", tmp_nGroup);
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;

                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(cfrnn_Classify->xType[i] == VAR_TYPE_CONTINUOUS) {
                        if(iNum == 0) {
                            tmp_flag = 1;
                        }
                    } else if(cfrnn_Classify->xType[i] == VAR_TYPE_DISCRETE ||
                              cfrnn_Classify->xType[i] == VAR_TYPE_BINARY) {
                        if(iNum == 1) {
                            tmp_flag = 1;
                        }
                    } else {
                        MPI_Barrier(MPI_COMM_WORLD);
                        if(0 == st_MPI_p.mpi_rank) {
                            printf("%s: For grouping, the number of variables is not right, exiting...\n",
                                   AT);
                        }
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("void grouping_variables_LeNet_less()\n");
    //}

    return;
}

void grouping_variables_EVO1_FRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_TYPE_EVO1_FRNN + DIM_CLUSTER_BIN_RULE_EVO1_FRNN +
                            DIM_FUZZY_BIN_ROUGH_EVO1_FRNN;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA_EVO1_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO1_FRNN +
                            NUM_CLASS_EVO1_FRNN;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EVO2_FRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_TYPE_EVO2_FRNN + DIM_CLUSTER_BIN_RULE_EVO2_FRNN +
                            DIM_FUZZY_BIN_ROUGH_EVO2_FRNN;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA_EVO2_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO2_FRNN +
                            DIM_CONSEQUENCE_EVO2_FRNN;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EVO3_FRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_TYPE_EVO3_FRNN + DIM_CLUSTER_BIN_RULE_EVO3_FRNN +
                            DIM_FUZZY_BIN_ROUGH_EVO3_FRNN;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA_EVO3_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO3_FRNN +
                            NUM_CLASS_EVO3_FRNN;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EVO4_FRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_TYPE_EVO4_FRNN + DIM_CLUSTER_BIN_RULE_EVO4_FRNN +
                            DIM_FUZZY_BIN_ROUGH_EVO4_FRNN;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA_EVO4_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO4_FRNN +
                            DIM_CONSEQUENCE_EVO4_FRNN;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EVO5_FRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_TYPE_EVO5_FRNN + DIM_CLUSTER_BIN_RULE_EVO5_FRNN +
                            DIM_FUZZY_BIN_ROUGH_EVO5_FRNN;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA_EVO5_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVO5_FRNN +
                            DIM_CONSEQUENCE_EVO5_FRNN + DIM_CONSE_WEIGHT_EVO5_FRNN;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_EVO_FRNN(int* xType)
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 2;

    MPI_Barrier(MPI_COMM_WORLD);
    if(0 == st_MPI_p.mpi_rank) {
        printf("NG = %d\n", tmp_nGroup);
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;
                //
                for(int i = 0; i < st_global_p.nDim; i++) {
                    int tmp_flag = 0;
                    if(xType[i] == VAR_TYPE_CONTINUOUS) {
                        if(iNum == 0) {
                            tmp_flag = 1;
                        }
                    } else if(xType[i] == VAR_TYPE_DISCRETE ||
                              xType[i] == VAR_TYPE_BINARY) {
                        if(iNum == 1) {
                            tmp_flag = 1;
                        }
                    } else {
                        MPI_Barrier(MPI_COMM_WORLD);
                        if(0 == st_MPI_p.mpi_rank) {
                            printf("%s: For grouping, the number of variables is not right, exiting...\n",
                                   AT);
                        }
                        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    }
                    if(tmp_flag) {
                        int cur_ind = iObj * st_global_p.nDim +
                                      st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                      st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        st_grp_info_p.Groups[cur_ind] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) {
    //    printf("void grouping_variables_LeNet_less()\n");
    //}

    return;
}

void grouping_variables_IntrusionDetection_Classify()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 3;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_id_c->M1->numParaLocal;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_id_c->F2->numParaLocal;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_id_c->R3->numParaLocal + frnn_id_c->O4->numParaLocal;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_ActivityDetection_Classify()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 3;

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_act_c->M1->numParaLocal;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_act_c->F2->numParaLocal;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] = frnn_act_c->R3->numParaLocal + frnn_act_c->OL->numParaLocal;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_evoCNN()
{
}

void grouping_variables_evoCFRNN()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
    int tmp_nGroup = 3;
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
    int tmp_nGroup = 4;
#else
    int tmp_nGroup = 5;
#endif

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            int cur_ind = iObj * st_global_p.nDim;
            for(int i = 0; i < st_global_p.nDim; i++) {
                st_grp_info_p.Groups[cur_ind++] = i;
            }

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

#if CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_0
                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->M1->numParaLocal;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->F2->numParaLocal;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->R3->numParaLocal +
                        cnn_evoCFRNN_c->OL->numParaLocal;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }
#elif CFRNN_STRUCTURE_TYPE_CUR == CFRNN_STRUCTURE_TYPE_1
                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->C1->numParaLocal +
                        cnn_evoCFRNN_c->P2->numParaLocal;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->C3->numParaLocal +
                        cnn_evoCFRNN_c->P4->numParaLocal;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->M5->numParaLocal +
                        cnn_evoCFRNN_c->F6->numParaLocal;
                    break;
                case 3:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->R7->numParaLocal +
                        cnn_evoCFRNN_c->OL->numParaLocal;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }
#else
                switch(iNum) {
                case 0:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->C1->numParaLocal +
                        cnn_evoCFRNN_c->P2->numParaLocal;
                    break;
                case 1:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->C3->numParaLocal +
                        cnn_evoCFRNN_c->P4->numParaLocal;
                    break;
                case 2:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->I5->numParaLocal;
                    break;
                case 3:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->M6->numParaLocal +
                        cnn_evoCFRNN_c->F7->numParaLocal;
                    break;
                case 4:
                    st_grp_info_p.Groups_sub_sizes[tmp_ind] =
                        cnn_evoCFRNN_c->R8->numParaLocal +
                        cnn_evoCFRNN_c->OL->numParaLocal;
                    break;
                default:
                    printf("%s: Error grouping, exiting...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_GROUPING);
                    break;
                }
#endif

                //if (iNum < NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
                //else if (iNum == NUM_CLASS_FRNN / 2) {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = DIM_CLUSTER_MEM_PARA;
                //}
                //else {
                //	strct_grp_info_vals.Groups_sub_sizes[tmp_ind] = 2 * MAX_NUM_FUZZY_RULE;
                //}
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    return;
}

void grouping_variables_classify_random()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    int tmp_nGroup = 5;
    int* v_nGroup = (int*)calloc(tmp_nGroup, sizeof(int));
    for(int i = 0; i < tmp_nGroup; i++) {
        v_nGroup[i] = st_global_p.nDim / tmp_nGroup;
        if(i < st_global_p.nDim % tmp_nGroup)
            v_nGroup[i]++;
    }
    int* DimTag = (int*)calloc(st_global_p.nDim, sizeof(int));
    for(int i = 0; i < st_global_p.nDim; i++) {
        DimTag[i] = i;
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            int tmp_size = st_global_p.nDim;
            int tmp_num = tmp_nGroup;
            int quo = tmp_size / tmp_num;
            int rem = tmp_size % tmp_num;

            st_grp_info_p.Groups_sizes[iObj] = tmp_num;

            for(int iNum = 0; iNum < tmp_num; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = quo;
                if(iNum < rem) st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }
            }

            int tmp_count = 0;
            int tmp_thresh = st_global_p.nDim / 5;

            int selectIND;
            int remNum = st_global_p.nDim;
            //int depth = 100;
            int tmp_count_per = 0;

            while(tmp_count < tmp_thresh) {
                for(int i = 0; i < tmp_nGroup; i++) {
                    int tmp_ind = iObj * st_global_p.nDim + i;
                    //tour selection based on ReliefF
                    if(tmp_count_per < st_grp_info_p.Groups_sub_sizes[tmp_ind]) {
                        selectIND = rnd(0, remNum - 1);
                        //if (GROUPING_TYPE_CLASSIFY == GROUPING_TYPE_CLASSIFY_ANALYS){
                        //	for (int n = 0; n < depth; n++) {
                        //		candidIND = rnd(0, remNum - 1);
                        //		if (filterWeights[DimTag[candidIND]][0] > filterWeights[DimTag[selectIND]][0]) {
                        //			selectIND = candidIND;
                        //		}
                        //	}
                        //}
                        int cur_ind = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_ind];
                        st_grp_info_p.Groups[cur_ind + tmp_count_per] = DimTag[selectIND];
                        DimTag[selectIND] = DimTag[--remNum];
                        tmp_count++;
                    }
                }
                tmp_count_per++;
            }

            //int considIND;
            while(tmp_count < st_global_p.nDim) {
                for(int i = 0; i < tmp_nGroup; i++) {
                    int tmp_ind = iObj * st_global_p.nDim + i;
                    //tour selection based on correlation
                    if(tmp_count_per < st_grp_info_p.Groups_sub_sizes[tmp_ind]) {
                        //considIND = rnd(0, tmp_count_per - 1);
                        selectIND = rnd(0, remNum - 1);
                        //if (GROUPING_TYPE_CLASSIFY == GROUPING_TYPE_CLASSIFY_ANALYS){
                        //	for (int n = 0; n < depth; n++) {
                        //		candidIND = rnd(0, remNum - 1);
                        //		if (featureCorrelation2(tableGroup_allObjectives[i][considIND], DimTag[candidIND]) < featureCorrelation2(tableGroup_allObjectives[i][considIND], DimTag[selectIND])) {
                        //			selectIND = candidIND;
                        //		}
                        //	}
                        //}
                        int cur_ind = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_ind];
                        st_grp_info_p.Groups[cur_ind + tmp_count_per] = DimTag[selectIND];
                        DimTag[selectIND] = DimTag[--remNum];
                        tmp_count++;
                    }
                }
                tmp_count_per++;
            }

            //sort
            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                for(int a = 0; a < st_grp_info_p.Groups_sub_sizes[tmp_ind]; a++) {
                    for(int b = a + 1; b < st_grp_info_p.Groups_sub_sizes[tmp_ind]; b++) {
                        int tmp_offset = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_ind];
                        if(st_grp_info_p.Groups[tmp_offset + a] > st_grp_info_p.Groups[tmp_offset + b]) {
                            int tmp = st_grp_info_p.Groups[tmp_offset + a];
                            st_grp_info_p.Groups[tmp_offset + a] = st_grp_info_p.Groups[tmp_offset + b];
                            st_grp_info_p.Groups[tmp_offset + b] = tmp;
                        }
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    free(v_nGroup);
    free(DimTag);

    return;
}

void grouping_variables_classify_cluster_kmeans()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    const int tmp_nGroup = 5;
    int* DimTag = (int*)calloc(st_global_p.nDim, sizeof(int));
    for(int i = 0; i < st_global_p.nDim; i++) {
        DimTag[i] = -1;
    }

    //k-means
    double** centroids_kmeans = (double**)calloc(tmp_nGroup, sizeof(double*));
    int nInstance = N_sample_optimize;
    for(int i = 0; i < tmp_nGroup; i++) {
        int tmpFlag = 1;
        int tmpIND[tmp_nGroup];
        do {
            tmpIND[i] = rnd(0, st_global_p.nDim - 1);

            tmpFlag = 1;
            for(int j = 0; j < i; j++) {
                if(tmpIND[i] == tmpIND[j]) {
                    tmpFlag = 0;
                }
            }
        } while(!tmpFlag);
        centroids_kmeans[i] = (double*)calloc(nInstance, sizeof(double));
        for(int j = 0; j < nInstance; j++) {
            centroids_kmeans[i][j] = optimizeData[j][tmpIND[i]];
        }
    }
    int kmeansFlag = 0;
    do {
        kmeansFlag = 0;

        double minD;
        int minI;
        double tmpD;

        for(int i = 0; i < st_global_p.nDim; i++) {
            minD = 0.0;
            minI = 0;
            for(int k = 0; k < nInstance; k++) {
                minD += (optimizeData[k][i] - centroids_kmeans[0][k]) * (optimizeData[k][i] -
                        centroids_kmeans[0][k]);
            }
            for(int j = 1; j < tmp_nGroup; j++) {
                tmpD = 0.0;
                for(int k = 0; k < nInstance; k++) {
                    tmpD += (optimizeData[k][i] - centroids_kmeans[j][k]) * (optimizeData[k][i] -
                            centroids_kmeans[j][k]);
                }
                if(tmpD < minD) {
                    minD = tmpD;
                    minI = j;
                }
            }
            if(DimTag[i] != minI) {
                kmeansFlag = 1;
                DimTag[i] = minI;
            }
        }

        //update centroids
        for(int i = 0; i < tmp_nGroup; i++) {
            double tmpSum;
            int tmpCount;
            for(int j = 0; j < nInstance; j++) {
                tmpSum = 0.0;
                tmpCount = 0;
                for(int n = 0; n < st_global_p.nDim; n++) {
                    if(DimTag[n] == i) {
                        tmpSum += optimizeData[j][n];
                        tmpCount++;
                    }
                }
                centroids_kmeans[i][j] = tmpSum / tmpCount;
            }
        }
    } while(kmeansFlag);

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                for(int i = 0; i < st_global_p.nDim; i++) {
                    if(DimTag[i] == iNum) {
                        int tmp_i = st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                    st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        int tmp_iD = iObj * st_global_p.nDim + tmp_i;
                        st_grp_info_p.Groups[tmp_iD] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    for(int i = 0; i < tmp_nGroup; i++) {
        free(centroids_kmeans[i]);
    }
    free(centroids_kmeans);
    free(DimTag);

    return;
}

#define MAX_BUF_SIZE 1000
void grouping_variables_classify_cluster_spectral()
{
    st_grp_ana_p.numDiverIndexes = 0;
    st_grp_ana_p.numConverIndexes = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
    }

    const int tmp_nGroup = 5;
    int* DimTag = (int*)calloc(st_global_p.nDim, sizeof(int));
    for(int i = 0; i < st_global_p.nDim; i++) {
        DimTag[i] = -1;
    }

    char filename[1024];
    sprintf(filename, "DATA_alg/clusters/%s_%d.clus", st_global_p.testInstance, st_ctrl_p.cur_run);

    FILE* fpt = fopen(filename, "r");
    int m = -1, n = -1;

    if(fpt) {
        char buf[MAX_BUF_SIZE];
        char* p;
        m = 0;
        n = 0;

        fgets(buf, MAX_BUF_SIZE, fpt);
        for(p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
            n++;
        }
        while(!feof(fpt)) {
            m++;
            fgets(buf, MAX_BUF_SIZE, fpt);
        }
        fclose(fpt);
    } else {
        printf("%s:Open file %s error, exiting...\n", AT, filename);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_READING);
    }

    if(n != 1) {
        if(st_MPI_p.mpi_rank == 0)
            printf("The number of columns is not 1, but %d.\n", n);
    }
    if(m != st_global_p.nDim) {
        if(st_MPI_p.mpi_rank == 0)
            printf("The number of features is wrong --- m = %d != ndim = %d.\n", m, st_global_p.nDim);
    }

    int  count[5] = { 0, 0, 0, 0, 0 };

    fpt = fopen(filename, "r");
    if(fpt) {
        for(int i = 0; i < m; i++) {
            fscanf(fpt, "%d", &DimTag[i]); //printf("%d ", tags[i]);
            count[DimTag[i]]++;
        }
        if(st_MPI_p.mpi_rank == 0)
            printf("%d-%s:\t%d %d %d %d %d", st_ctrl_p.cur_run, st_global_p.testInstance, count[0], count[1], count[2],
                   count[3], count[4]);
        fclose(fpt);
    } else {
        printf("%s:Open file %s error, exiting...\n", AT, filename);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_READING);
    }
    double minV = 1e308;
    double maxV = -1e308;
    for(int in = 0; in < 5; in++) {
        if(count[in] < minV) minV = count[in];
        if(count[in] > maxV) maxV = count[in];
    }
    if(st_MPI_p.mpi_rank == 0)
        printf(" - %lf\n", maxV / minV);

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0) {
            st_grp_info_p.Groups_sizes[iObj] = tmp_nGroup;

            for(int iNum = 0; iNum < tmp_nGroup; iNum++) {
                int tmp_ind = iObj * st_global_p.nDim + iNum;
                st_grp_info_p.Groups_sub_sizes[tmp_ind] = 0;
                if(iNum == 0) {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] = 0;
                } else {
                    st_grp_info_p.Groups_sub_disps[tmp_ind] =
                        st_grp_info_p.Groups_sub_sizes[tmp_ind - 1] +
                        st_grp_info_p.Groups_sub_disps[tmp_ind - 1];
                }

                for(int i = 0; i < st_global_p.nDim; i++) {
                    if(DimTag[i] == iNum) {
                        int tmp_i = st_grp_info_p.Groups_sub_disps[tmp_ind] +
                                    st_grp_info_p.Groups_sub_sizes[tmp_ind];
                        int tmp_iD = iObj * st_global_p.nDim + tmp_i;
                        st_grp_info_p.Groups[tmp_iD] = i;
                        st_grp_info_p.Groups_sub_sizes[tmp_ind]++;
                    }
                }
            }
        } else {
            memcpy(&st_grp_info_p.Groups[iObj * st_global_p.nDim], st_grp_info_p.Groups,
                   st_global_p.nDim * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_sizes,
                   tmp_nGroup * sizeof(int));
            memcpy(&st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim], st_grp_info_p.Groups_sub_disps,
                   tmp_nGroup * sizeof(int));
            st_grp_info_p.Groups_sizes[iObj] = st_grp_info_p.Groups_sizes[0];
        }
    }

    free(DimTag);

    return;
}