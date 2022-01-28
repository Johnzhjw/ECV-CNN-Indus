# include "global.h"
# include <math.h>

void copyToArchiveFromRepository(int iA, int iR)
{
    memcpy(&st_archive_p.var[iA * st_global_p.nDim], &st_repo_p.var[iR * st_global_p.nDim],
           st_global_p.nDim * sizeof(double));
    memcpy(&st_archive_p.obj[iA * st_global_p.nObj], &st_repo_p.obj[iR * st_global_p.nObj],
           st_global_p.nObj * sizeof(double));
    st_archive_p.dens[iA] = st_repo_p.dens[iR];
    st_archive_p.indx[iA] = iR;
    st_DE_p.F__archive[iA] = st_repo_p.F[iR];
    st_DE_p.CR_archive[iA] = st_repo_p.CR[iR];
    st_DE_p.CR_evo_arc[iA] = st_repo_p.CR_evo[iR];
    st_PSO_p.w__archive[iA] = st_repo_p.w[iR];
    st_PSO_p.c1_archive[iA] = st_repo_p.c1[iR];
    st_PSO_p.c2_archive[iA] = st_repo_p.c2[iR];
    st_repo_p.flag[iR] = iA;

    return;
}

void copyFromRepository(int iR, int iDest, double* dest, double* destFit, double* destDens,
                        double* dest_one_group)
{
    memcpy(&dest[iDest * st_global_p.nDim], &st_repo_p.var[iR * st_global_p.nDim],
           st_global_p.nDim * sizeof(double));
    memcpy(&destFit[iDest * st_global_p.nObj], &st_repo_p.obj[iR * st_global_p.nObj],
           st_global_p.nObj * sizeof(double));
    destDens[iDest] = st_repo_p.dens[iR];
    if(dest_one_group) {
        int i;
        if(st_MPI_p.color_obj) {   //objectives
            for(i = 0; i < st_grp_info_p.table_mine_size; i++) {
                dest_one_group[iDest * st_global_p.nDim + i] =
                    dest[iDest * st_global_p.nDim + st_grp_info_p.table_mine[i]];
            }
        } else { //archive
            for(i = 0; i < st_grp_info_p.table_mine_size; i++) {
                dest_one_group[iDest * st_global_p.nDim + i] =
                    dest[iDest * st_global_p.nDim + st_grp_info_p.table_mine[i]];
            }
        }
    }
    st_repo_p.flag[iR] = iDest;
}

void selectSamples(int pp, int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3, int* r4, int* r5)
{
    if(r1) {
        do {
            *r1 = rnd(0, pp - 1);
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }

    if(r2) {
        do {
            *r2 = rnd(0, pp - 1);
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }

    if(r3) {
        do {
            *r3 = rnd(0, pp - 1);
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r1) || (*r3 == *r2));
    }

    if(r4) {
        do {
            *r4 = rnd(0, pp - 1);
        } while((*r4 == candidate1) || (*r4 == candidate2) || (*r4 == candidate3) || (*r4 == *r1) || (*r4 == *r2) || (*r4 == *r3));
    }

    if(r5) {
        do {
            *r5 = rnd(0, pp - 1);
        } while((*r5 == candidate1) || (*r5 == candidate2) || (*r5 == candidate3) ||
                (*r5 == *r1) || (*r5 == *r2) || (*r5 == *r3) || (*r5 == *r4));
    }
}

void selectSamples_niche(int* niche_table, int niche_size,
                         int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3, int* r4, int* r5)
{
    if(r1) {
        do {
            *r1 = rnd(0, niche_size - 1);
            *r1 = niche_table[*r1];
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }

    if(r2) {
        do {
            *r2 = rnd(0, niche_size - 1);
            *r2 = niche_table[*r2];
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }

    if(r3) {
        do {
            *r3 = rnd(0, niche_size - 1);
            *r3 = niche_table[*r3];
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r1) || (*r3 == *r2));
    }

    if(r4) {
        do {
            *r4 = rnd(0, niche_size - 1);
            *r4 = niche_table[*r4];
        } while((*r4 == candidate1) || (*r4 == candidate2) || (*r4 == candidate3) || (*r4 == *r1) || (*r4 == *r2) || (*r4 == *r3));
    }

    if(r5) {
        do {
            *r5 = rnd(0, niche_size - 1);
            *r5 = niche_table[*r5];
        } while((*r5 == candidate1) || (*r5 == candidate2) || (*r5 == candidate3) ||
                (*r5 == *r1) || (*r5 == *r2) || (*r5 == *r3) || (*r5 == *r4));
    }
}

void selectSamples_RR(int pp, double* prob,
                      int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3)
{
    double sum = 0.0;
    for(int i = 0; i < pp; i++)
        sum += prob[i];
    if(r1) {
        do {
            double tmpD = rndreal(0.0, sum);
            int  tmpI = 0;
            while(tmpD > prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r1 = tmpI;//printf("1-%d ",*r1);
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }
    if(r2) {
        do {
            double tmpD = rndreal(0.0, sum);
            int  tmpI = 0;
            while(tmpD > prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r2 = tmpI;//printf("2-%d ",*r2);
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }
    if(r3) {
        do {
            double tmpD = rndreal(0.0, sum);
            int  tmpI = 0;
            while(tmpD > prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r3 = tmpI;//printf("3-%d ",*r3);
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r1) || (*r3 == *r2));
    }
    return;
}

void selectSamples_clone_RR(int pp, int* prob,
                            int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3)
{
    int sum = 0;
    for(int i = 0; i < pp; i++)
        sum += prob[i];
    if(r1) {
        do {
            int tmpD = rnd(0, sum - 1);
            int  tmpI = 0;
            while(tmpD >= prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r1 = tmpI;//printf("1-%d ",*r1);
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }
    if(r2) {
        do {
            int tmpD = rnd(0, sum - 1);
            int  tmpI = 0;
            while(tmpD >= prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r2 = tmpI;//printf("2-%d ",*r2);
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }
    if(r3) {
        do {
            int tmpD = rnd(0, sum - 1);
            int  tmpI = 0;
            while(tmpD >= prob[tmpI]) {
                tmpD -= prob[tmpI];
                tmpI++;
            }
            *r3 = tmpI;//printf("3-%d ",*r3);
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r1) || (*r3 == *r2));
    }
    return;
}

void tourSelectSamples_sub(int pp, int depth, int iObj, double* data,
                           int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3)
{
    if(r1) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(data[can * st_global_p.nObj + iObj] < data[cur * st_global_p.nObj + iObj]) {
                    cur = can;
                }
            }
            *r1 = cur;
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }
    if(r2) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(data[can * st_global_p.nObj + iObj] < data[cur * st_global_p.nObj + iObj]) {
                    cur = can;
                }
            }
            *r2 = cur;
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }
    if(r3) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(data[can * st_global_p.nObj + iObj] < data[cur * st_global_p.nObj + iObj]) {
                    cur = can;
                }
            }
            *r3 = cur;
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r2) || (*r3 == *r1));
    }
    return;
}

void tourSelectSamples_rank_dist(int pp, int depth, int* rank, double* dist,
                                 int candidate1, int candidate2, int candidate3, int* r1, int* r2, int* r3)
{
    if(r1) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(rank[can] < rank[cur]) {
                    cur = can;
                } else if(rank[can] == rank[cur] && dist[can] > dist[cur]) {
                    cur = can;
                }
            }
            *r1 = cur;
        } while((*r1 == candidate1) || (*r1 == candidate2) || (*r1 == candidate3));
    }
    if(r2) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(rank[can] < rank[cur]) {
                    cur = can;
                } else if(rank[can] == rank[cur] && dist[can] > dist[cur]) {
                    cur = can;
                }
            }
            *r2 = cur;
        } while((*r2 == candidate1) || (*r2 == candidate2) || (*r2 == candidate3) || (*r2 == *r1));
    }
    if(r3) {
        do {
            int cur = rnd(0, pp - 1);
            int can;
            int i;
            for(i = 0; i < depth; i++) {
                can = rnd(0, pp - 1);
                if(rank[can] < rank[cur]) {
                    cur = can;
                } else if(rank[can] == rank[cur] && dist[can] > dist[cur]) {
                    cur = can;
                }
            }
            *r3 = cur;
        } while((*r3 == candidate1) || (*r3 == candidate2) || (*r3 == candidate3) || (*r3 == *r2) || (*r3 == *r1));
    }
    return;
}

bool isDuplicate(double* s, int i, int j)
{
    int k;
    for(k = 0; k < st_global_p.nDim; k++) {
        if(s[i * st_global_p.nDim + k] != s[j * st_global_p.nDim + k])
            return false;
    }
    return true;
}

void get_nonDominateSize()
{
    st_global_p.nonDominateSize = 0;
    while(st_archive_p.rank[st_global_p.nonDominateSize] == 1
          && st_global_p.nonDominateSize < st_archive_p.cnArch)
        st_global_p.nonDominateSize++;
}

void transform_var_feature(double* individual, int* var_transformed, int numIndiv)
{
    for(int iIndiv = 0; iIndiv < numIndiv; iIndiv++) {
        //////////////////////////////////////////////////////////////
        int numFeature = 0;
        if(st_ctrl_p.type_var_encoding == VAR_DOUBLE) {
            int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));

            numFeature = (int)(individual[iIndiv * st_global_p.nDim + st_global_p.nDim - 1] * TH_N_FEATURE) + 1;
            if(numFeature > TH_N_FEATURE)
                numFeature = TH_N_FEATURE;

            double tmpMAX = -1.0;
            int    tmpIND = -1;
            for(int i = 0; i < numFeature; i++) {
                tmpMAX = -1.0;
                for(int j = 0; j < N_FEATURE; j++) {
                    if(tmpFlag[j] == 0 && individual[iIndiv * st_global_p.nDim + j] > tmpMAX) {
                        tmpMAX = individual[iIndiv * st_global_p.nDim + j];
                        tmpIND = j;
                    }
                }
                tmpFlag[tmpIND] = 1;
            }
            int count = 0;
            for(int i = 0; i < N_FEATURE; i++) if(tmpFlag[i]) var_transformed[iIndiv * TH_N_FEATURE + count++] = i;

            free(tmpFlag);
        } else {
            for(int i = 0; i < N_FEATURE; i++) {
                if((int)individual[iIndiv * st_global_p.nDim + i]) {
                    var_transformed[iIndiv * TH_N_FEATURE + numFeature++] = i;
                    if(numFeature > TH_N_FEATURE) printf("Feature number overflow...%d\n", numFeature);
                }
            }
        }

        for(int i = numFeature; i < TH_N_FEATURE; i++) {
            var_transformed[iIndiv * TH_N_FEATURE + i] = -1;
        }
    }

    return;
}

void my_error_fun(const char* location, const char* msg, int error_code)
{
    printf(" Error at %s: %s\n ", location, msg);
    MPI_Abort(MPI_COMM_WORLD, error_code);
}

void convertVar_CNN(double* indiv_ori, double* indiv_cvt)
{
    int k;
    for(k = 0; k < st_global_p.nDim; k++) {
        if(indiv_ori[k] < st_global_p.minLimit[k] || indiv_ori[k] > st_global_p.maxLimit[k]) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, k, indiv_ori[k], st_global_p.minLimit[k], st_global_p.maxLimit[k]);
            exit(MY_ERROR_CHECK_LIMIT_WRONG);
        }
    }

    int direct;
    double v_scale = MAX_WEIGHT_BIAS_CNN;
    double angle_scale;
    double angle_bias;
    double dist;
    int thresh_ori1 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP;
    int thresh_ori2 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B;
    int thresh_ori3 = NUM_PARA_C1_MAPS * st_global_p.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS *
                      st_global_p.nDim_MAP;
    //int thresh_ori4 = NUM_PARA_C1_MAPS*strct_global_paras.nDim_MAP + NUM_PARA_C1_B + NUM_PARA_C3_MAPS*strct_global_paras.nDim_MAP + NUM_PARA_C3_B;
    int thresh_cvt1 = NUM_PARA_C1_M;
    int thresh_cvt2 = NUM_PARA_C1;
    int thresh_cvt3 = NUM_PARA_C1 + NUM_PARA_C3_M;
    //int thresh_cvt4 = NUM_PARA_C1 + NUM_PARA_C3;
    int iMap;
    int i;
    int cur_i;
    int n;
    int map_i;
    int map_j;
    double center_i = (MAP_SIZE_C - 1.0) / 2;
    double center_j = (MAP_SIZE_C - 1.0) / 2;
    for(i = 0; i < st_global_p.nDim;) {
        if(i < thresh_ori1) {
            cur_i = i;
            iMap = cur_i / st_global_p.nDim_MAP;
            direct = (int)indiv_ori[i];
            angle_scale = indiv_ori[i + 1];
            angle_bias = indiv_ori[i + 2];
            for(n = 0; n < MAP_SIZE_C * MAP_SIZE_C; n++) {
                map_i = n / MAP_SIZE_C;
                map_j = n % MAP_SIZE_C;
                if(direct == DIRECT_CNN_MAP_OMNI)
                    dist = sqrt((map_i - center_i) * (map_i - center_i) + (map_j - center_j) * (map_j - center_j));
                else if(direct == DIRECT_CNN_MAP_HORIZONTAL)
                    dist = map_j - center_j;
                else if(direct == DIRECT_CNN_MAP_VERTICAL)
                    dist = map_i - center_i;
                else {
                    if(0 == st_MPI_p.mpi_rank)
                        printf("%s: Wrong map direction, %d - %lf, for ENUM_DIRECT_CNN_MAP, exiting...\n",
                               AT, direct, indiv_ori[i]);
                    exit(MY_ERROR_ENUM_DIRECT_CNN_MAP);
                }
                indiv_cvt[iMap * MAP_SIZE_C * MAP_SIZE_C + n] =
                    v_scale * sin(angle_scale * dist + angle_bias);
            }
            i += st_global_p.nDim_MAP;
        } else if(i < thresh_ori2) {
            cur_i = i - thresh_ori1;
            indiv_cvt[thresh_cvt1 + cur_i] = indiv_ori[i];
            i++;
        } else if(i < thresh_ori3) {
            cur_i = i - thresh_ori2;
            iMap = cur_i / st_global_p.nDim_MAP;
            direct = (int)indiv_ori[i];
            angle_scale = indiv_ori[i + 1];
            angle_bias = indiv_ori[i + 2];
            for(n = 0; n < MAP_SIZE_C * MAP_SIZE_C; n++) {
                map_i = n / MAP_SIZE_C;
                map_j = n % MAP_SIZE_C;
                if(direct == DIRECT_CNN_MAP_OMNI)
                    dist = sqrt((map_i - center_i) * (map_i - center_i) + (map_j - center_j) * (map_j - center_j));
                else if(direct == DIRECT_CNN_MAP_HORIZONTAL)
                    dist = map_j - center_j;
                else if(direct == DIRECT_CNN_MAP_VERTICAL)
                    dist = map_i - center_i;
                else {
                    if(0 == st_MPI_p.mpi_rank)
                        printf("%s: Wrong map direction, %d - %lf, for ENUM_DIRECT_CNN_MAP, exiting...\n",
                               AT, direct, indiv_ori[i]);
                    exit(MY_ERROR_ENUM_DIRECT_CNN_MAP);
                }
                indiv_cvt[thresh_cvt2 + iMap * MAP_SIZE_C * MAP_SIZE_C + n] =
                    v_scale * sin(angle_scale * dist + angle_bias);
            }
            i += st_global_p.nDim_MAP;
        } else {
            cur_i = i - thresh_ori3;
            indiv_cvt[thresh_cvt3 + cur_i] = indiv_ori[i];
            i++;
        }
    }
}