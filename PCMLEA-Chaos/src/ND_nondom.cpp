# include "global.h"
#include <math.h>

void refineRepository_deleteTheSame(double* obj_store, double* var_store, int& n_store, int thresh)
{
    int i, j, k;
    int count = n_store;
    for(i = st_repo_p.nRep - 2; i >= 0; i--) {
        if(count <= thresh) break;
        j = count - 1;
        while(j > i) {
            if(count <= thresh) break;
            if(isTheSame(&(obj_store[i * st_global_p.nObj]), &(obj_store[j * st_global_p.nObj]))) {
                for(k = j; k < count - 1; k++) {
                    memcpy(&var_store[k * st_global_p.nDim], &var_store[(k + 1) * st_global_p.nDim],
                           st_global_p.nDim * sizeof(double));
                    memcpy(&obj_store[k * st_global_p.nObj], &obj_store[(k + 1) * st_global_p.nObj],
                           st_global_p.nObj * sizeof(double));
                }
                count--;
            }
            j--;
        }
    }

    //printf("%d---%d\n", strct_repo_info.nRep, count);

    n_store = count;

    return;
}

void refineRepository_markTheSame(double* obj_store, int* tag_store, int n_store)
{
    int i, j;
    int count = n_store;
    //
    for(i = 0; i < n_store; i++) tag_store[i] = 0;
    for(i = 0; i < n_store; i++) {
        for(j = i - 1; j >= 0; j--) {
            if(isTheSame(&(obj_store[i * st_global_p.nObj]), &(obj_store[j * st_global_p.nObj]))) {
                count--;
                tag_store[i] = tag_store[j] + 1;
                break;
            }
        }
    }
    return;
}

void refineRepository_generateArchive_sub()
{
    if(!st_MPI_p.color_obj) {
        printf("%s: The population simultaneously optimizes all objectives executes the routine for one objective, exiting...\n",
               AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_POP_WRONG);
    }

    //refineRepository_deleteTheSame(strct_repo_info.obj, strct_repo_info.var, strct_repo_info.nRep, strct_archive_info.nArch_sub);
    refineRepository_markTheSame(st_repo_p.obj, st_repo_p.tag, st_repo_p.nRep);

    int i;
    int ind_prob = st_MPI_p.color_pop - 1;
    int* ind_sep = (int*)calloc(st_repo_p.nRep, sizeof(int));
    for(i = 0; i < st_repo_p.nRep; i++) ind_sep[i] = i;
    qSortFrontObj(ind_prob, ind_sep, 0, st_repo_p.nRep - 1);
    //strct_repo_info.tag[ind_sep[0]] = 0;
    int tmp_count = 0;
    for(i = 0; i < st_repo_p.nRep && tmp_count < st_archive_p.nArch; i++) {
        if(st_repo_p.tag[ind_sep[i]] == 0) {
            copyToArchiveFromRepository(tmp_count, ind_sep[i]);
            tmp_count++;
        }
    }
    shuffle(ind_sep, st_repo_p.nRep);
    for(i = 0; i < st_repo_p.nRep && tmp_count < st_archive_p.nArch; i++) {
        if(st_repo_p.tag[ind_sep[i]] > 0) {
            copyToArchiveFromRepository(tmp_count, ind_sep[i]);
            tmp_count++;
        }
    }
    st_archive_p.cnArch = st_archive_p.nArch_sub;

    free(ind_sep);

    //printf("strct_archive_info.cnArch sep = %d\n", strct_archive_info.cnArch);

    //
    //double tmp_min = INF_DOUBLE;
    //for(int i = 0; i < strct_repo_info.nRep; i++) {
    //    if(tmp_min > strct_repo_info.obj[i * strct_global_paras.nObj + ind_prob])
    //        tmp_min = strct_repo_info.obj[i * strct_global_paras.nObj + ind_prob];
    //}
    //if(tmp_min != strct_archive_info.obj_archive[ind_prob]) {
    //    printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
    //           AT, tmp_min, strct_archive_info.obj_archive[ind_prob], strct_MPI_info.color_population);
    //}
    //
    return;
}

void refineRepository_generateArchive()
{
    //refineRepository_deleteTheSame(strct_repo_info.obj, strct_repo_info.var, strct_repo_info.nRep, strct_archive_info.nArch);
    for(int i = 0; i < st_global_p.nInd_max_repo; i++) st_repo_p.flag[i] = -1;

    int result;
    int i, j;
    int end;
    int frontSize;
    int popSize;
    int rank = 1;
    list* pool;
    list* elite;
    list* temp1, * temp2;
    pool = createList(-1);
    elite = createList(-1);
    frontSize = 0;
    popSize = 0;
    //
    //double tmp_min_all[100];
    //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
    //    tmp_min_all[iObj] = INF_DOUBLE;
    //    for(int i = 0; i < strct_repo_info.nRep; i++) {
    //        if(tmp_min_all[iObj] > strct_repo_info.obj[i * strct_global_paras.nObj + iObj])
    //            tmp_min_all[iObj] = strct_repo_info.obj[i * strct_global_paras.nObj + iObj];
    //    }
    //}
    //

    temp1 = pool;
    for(i = 0; i < st_repo_p.nRep; i++) {
        insert(temp1, i);
        temp1 = temp1->child;
    }
    i = 0;
    do {
        temp1 = pool->child;
        if(temp1 == NULL) {
            break;
        }
        insert(elite, temp1->index);
        frontSize = 1;
        temp2 = elite->child;
        temp1 = deleteNode(temp1);
        temp1 = temp1->child;
        do {
            temp2 = elite->child;
            if(temp1 == NULL) {
                break;
            }
            do {
                end = 0;
                result = dominanceComparator(&(st_repo_p.obj[(temp1->index) * st_global_p.nObj]),
                                             &(st_repo_p.obj[(temp2->index) * st_global_p.nObj]));
                if(result == 1) {
                    insert(pool, temp2->index);
                    temp2 = deleteNode(temp2);
                    frontSize--;
                    temp2 = temp2->child;
                }
                if(result == 0) {
                    temp2 = temp2->child;
                }
                if(result == -1 || result == 2) {
                    end = 1;
                }
            } while((end != 1) && (temp2 != NULL));

            if(result == 0 || result == 1) {
                insert(elite, temp1->index);
                frontSize++;
                temp1 = deleteNode(temp1);
            }
            if(result == 2) {
                temp1 = deleteNode(temp1);//////////////////////////////////////////////////////////////////////////
            }
            temp1 = temp1->child;
        } while(temp1 != NULL);

        if(rank == 1) {
            if(frontSize <= st_archive_p.nArch) {
                st_global_p.PF_size = frontSize;
            } else {
                st_global_p.PF_size = st_archive_p.nArch;
            }
            //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
            //    double tmp_min2 = INF_DOUBLE;
            //    list* tempL = elite->child;
            //    for(int i = 0; i < frontSize; i++) {
            //        int ind = tempL->index;
            //        if(tmp_min2 > strct_repo_info.obj[ind * strct_global_paras.nObj + iObj])
            //            tmp_min2 = strct_repo_info.obj[ind * strct_global_paras.nObj + iObj];
            //        tempL = tempL->child;
            //    }
            //    if(tmp_min_all[iObj] != tmp_min2) {
            //        printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
            //               AT, tmp_min_all[iObj], tmp_min2, iObj + 1);
            //    }
            //}
        }
        temp2 = elite->child;
        j = i;
        if((popSize + frontSize) <= st_archive_p.nArch) {
            assignCrowdingDistanceList(temp2, frontSize);
            do {
                copyToArchiveFromRepository(i, temp2->index);
                st_archive_p.rank[i] = rank;
                popSize += 1;
                temp2 = temp2->child;
                i += 1;
            } while(temp2 != NULL);
            rank += 1;
        } else {
            fillCrowdingDistance(i, frontSize, elite);
            //K_Neighbor_Nearest_SDE(i, frontSize, elite);
            popSize = st_archive_p.nArch;
            for(j = i; j < popSize; j++) {
                st_archive_p.rank[j] = rank;
            }
        }
        temp2 = elite->child;
        do {
            temp2 = deleteNode(temp2);
            temp2 = temp2->child;
        } while(elite->child != NULL);
        //
        //if(rank <= 2) {
        //    for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
        //        double tmp_min2 = INF_DOUBLE;
        //        for(int i = 0; i < popSize; i++) {
        //            if(tmp_min2 > strct_archive_info.obj_archive[i * strct_global_paras.nObj + iObj])
        //                tmp_min2 = strct_archive_info.obj_archive[i * strct_global_paras.nObj + iObj];
        //        }
        //        if(tmp_min_all[iObj] != tmp_min2) {
        //            printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
        //                   AT, tmp_min_all[iObj], tmp_min2, iObj + 1);
        //        }
        //    }
        //}
    } while(popSize < st_archive_p.nArch);

    // 	if (strct_repo_info.nRep<strct_archive_info.nArch)
    // 	{
    // 		strct_archive_info.cnArch=strct_repo_info.nRep;
    // 	}
    // 	else
    // 	{
    // 		strct_archive_info.cnArch=strct_archive_info.nArch;
    // 	}

    st_archive_p.cnArch = popSize < st_archive_p.nArch ? popSize : st_archive_p.nArch;
    //printf("strct_archive_info.cnArch = %d\n", strct_archive_info.cnArch);
    while(st_archive_p.cnArch < st_archive_p.nArch && st_archive_p.cnArch < st_repo_p.nRep) {
        int* tmpIND = (int*)malloc(st_repo_p.nRep * sizeof(int));
        for(int i = 0; i < st_repo_p.nRep; i++) tmpIND[i] = i;
        shuffle(tmpIND, st_repo_p.nRep);
        int tmpCNT = 0;
        for(int i = 0; i < st_repo_p.nRep && tmpCNT < st_archive_p.nArch - st_archive_p.cnArch; i++) {
            int curI = tmpIND[i];
            if(st_repo_p.flag[curI] == -1) {
                int ind_dst = st_archive_p.cnArch + tmpCNT;
                copyToArchiveFromRepository(ind_dst, curI);
                tmpCNT++;
            }
        }
        free(tmpIND);
        st_archive_p.cnArch += tmpCNT;
    }

    deleteList(pool);
    deleteList(elite);

    //
    //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
    //    double tmp_min = INF_DOUBLE;
    //    for(int i = 0; i < strct_repo_info.nRep; i++) {
    //        if(tmp_min > strct_repo_info.obj[i * strct_global_paras.nObj + iObj])
    //            tmp_min = strct_repo_info.obj[i * strct_global_paras.nObj + iObj];
    //    }
    //    double tmp_min2 = INF_DOUBLE;
    //    for(int i = 0; i < strct_archive_info.cnArch; i++) {
    //        if(tmp_min2 > strct_archive_info.obj_archive[i * strct_global_paras.nObj + iObj])
    //            tmp_min2 = strct_archive_info.obj_archive[i * strct_global_paras.nObj + iObj];
    //    }
    //    if(tmp_min != tmp_min2) {
    //        printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
    //               AT, tmp_min, tmp_min2, iObj + 1);
    //    }
    //}
    //
    return;
}

void refineRepository_generateND(double* _dest, double* _destFit, double* _destDens, int* _destClass,
                                 double* _dest_one_group, int& _n_d, int _n)
{
    //refineRepository_deleteTheSame(strct_repo_info.obj, strct_repo_info.var, strct_repo_info.nRep, _n);
    for(int i = 0; i < st_global_p.nInd_max_repo; i++) st_repo_p.flag[i] = -1;

    int result;
    int i, j;
    int end;
    int frontSize;
    int popSize;
    int rank = 1;
    list* pool;
    list* elite;
    list* temp1, * temp2;
    pool = createList(-1);
    elite = createList(-1);
    frontSize = 0;
    popSize = 0;
    //
    //double tmp_min_all[100];
    //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
    //    tmp_min_all[iObj] = INF_DOUBLE;
    //    for(int i = 0; i < strct_repo_info.nRep; i++) {
    //        if(tmp_min_all[iObj] > strct_repo_info.obj[i * strct_global_paras.nObj + iObj])
    //            tmp_min_all[iObj] = strct_repo_info.obj[i * strct_global_paras.nObj + iObj];
    //    }
    //}
    //

    temp1 = pool;
    for(i = 0; i < st_repo_p.nRep; i++) {
        insert(temp1, i);
        temp1 = temp1->child;
    }
    i = 0;
    do {
        temp1 = pool->child;
        if(temp1 == NULL) {
            break;
        }
        insert(elite, temp1->index);
        frontSize = 1;
        temp2 = elite->child;
        temp1 = deleteNode(temp1);
        temp1 = temp1->child;
        do {
            temp2 = elite->child;
            if(temp1 == NULL) {
                break;
            }
            do {
                end = 0;
                result = dominanceComparator(&(st_repo_p.obj[(temp1->index) * st_global_p.nObj]),
                                             &(st_repo_p.obj[(temp2->index) * st_global_p.nObj]));
                if(result == 1) {
                    insert(pool, temp2->index);
                    temp2 = deleteNode(temp2);
                    frontSize--;
                    temp2 = temp2->child;
                }
                if(result == 0) {
                    temp2 = temp2->child;
                }
                if(result == -1 || result == 2) {
                    end = 1;
                }
            } while((end != 1) && (temp2 != NULL));

            if(result == 0 || result == 1) {
                insert(elite, temp1->index);
                frontSize++;
                temp1 = deleteNode(temp1);
            }
            if(result == 2) {
                temp1 = deleteNode(temp1);//////////////////////////////////////////////////////////////////////////
            }
            temp1 = temp1->child;
        } while(temp1 != NULL);

        if(rank == 1) {
            if(frontSize <= st_archive_p.nArch) {
                st_global_p.PF_size = frontSize;
            } else {
                st_global_p.PF_size = st_archive_p.nArch;
            }
            //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
            //    double tmp_min2 = INF_DOUBLE;
            //    list* tempL = elite->child;
            //    for(int i = 0; i < frontSize; i++) {
            //        int ind = tempL->index;
            //        if(tmp_min2 > strct_repo_info.obj[ind * strct_global_paras.nObj + iObj])
            //            tmp_min2 = strct_repo_info.obj[ind * strct_global_paras.nObj + iObj];
            //        tempL = tempL->child;
            //    }
            //    if(tmp_min_all[iObj] != tmp_min2) {
            //        printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
            //               AT, tmp_min_all[iObj], tmp_min2, iObj + 1);
            //    }
            //}
        }
        temp2 = elite->child;
        j = i;
        if((popSize + frontSize) <= _n) {
            assignCrowdingDistanceList(temp2, frontSize);
            do {
                copyFromRepository(temp2->index, i, _dest, _destFit, _destDens, _dest_one_group);
                _destClass[i] = rank;
                popSize += 1;
                temp2 = temp2->child;
                i += 1;
            } while(temp2 != NULL);
            //assignCrowdingDistanceIndexes(j,i-1);
            rank += 1;
        } else {
            fillCrowdingDistance(i, frontSize, elite, _dest, _destFit, _destDens, _dest_one_group, _n);
            //K_Neighbor_Nearest_SDE(i, frontSize, elite, _dest, _destFit, _destDens, _dest_one_group, _n);
            popSize = _n;
            for(j = i; j < popSize; j++) {
                _destClass[j] = rank;
            }
        }
        temp2 = elite->child;
        do {
            temp2 = deleteNode(temp2);
            temp2 = temp2->child;
        } while(elite->child != NULL);
        //
        //if(rank <= 2) {
        //    for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
        //        double tmp_min2 = INF_DOUBLE;
        //        for(int i = 0; i < popSize; i++) {
        //            if(tmp_min2 > _destFit[i * strct_global_paras.nObj + iObj])
        //                tmp_min2 = _destFit[i * strct_global_paras.nObj + iObj];
        //        }
        //        if(tmp_min_all[iObj] != tmp_min2) {
        //            printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
        //                   AT, tmp_min_all[iObj], tmp_min2, iObj + 1);
        //        }
        //    }
        //}
    } while(popSize < _n);

    _n_d = popSize < _n ? popSize : _n;
    while(_n_d < _n && _n_d < st_repo_p.nRep) {
        int* tmpIND = (int*)malloc(st_repo_p.nRep * sizeof(int));
        for(int i = 0; i < st_repo_p.nRep; i++) tmpIND[i] = i;
        shuffle(tmpIND, st_repo_p.nRep);
        int tmpCNT = 0;
        for(int i = 0; i < st_repo_p.nRep && tmpCNT < _n - _n_d; i++) {
            int curI = tmpIND[i];
            if(st_repo_p.flag[curI] == -1) {
                int ind_dst = _n_d + tmpCNT;
                copyFromRepository(curI, ind_dst, _dest, _destFit, _destDens, _dest_one_group);
                tmpCNT++;
            }
        }
        free(tmpIND);
        _n_d += tmpCNT;
    }

    deleteList(pool);
    deleteList(elite);

    //
    //for(int iObj = 0; iObj < strct_global_paras.nObj; iObj++) {
    //    double tmp_min = INF_DOUBLE;
    //    for(int i = 0; i < strct_repo_info.nRep; i++) {
    //        if(tmp_min > strct_repo_info.obj[i * strct_global_paras.nObj + iObj])
    //            tmp_min = strct_repo_info.obj[i * strct_global_paras.nObj + iObj];
    //    }
    //    double tmp_min2 = INF_DOUBLE;
    //    for(int i = 0; i < _n_d; i++) {
    //        if(tmp_min2 > _destFit[i * strct_global_paras.nObj + iObj])
    //            tmp_min2 = _destFit[i * strct_global_paras.nObj + iObj];
    //    }
    //    if(tmp_min != tmp_min2) {
    //        printf("%s: The minimum value is changed from %g to %g for obj %d, not reasonable.\n",
    //               AT, tmp_min, tmp_min2, iObj + 1);
    //    }
    //}
    //
    return;
}

void refineRepository_generateArchive_SDE()
{
    //refineRepository_deleteTheSame(strct_repo_info.obj, strct_repo_info.var, strct_repo_info.nRep, strct_archive_info.nArch);
    for(int i = 0; i < st_global_p.nInd_max_repo; i++) st_repo_p.flag[i] = -1;

    int result;
    int i, j;
    int end;
    int frontSize;
    int popSize;
    int rank = 1;
    list* pool;
    list* elite;
    list* temp1, * temp2;
    pool = createList(-1);
    elite = createList(-1);
    frontSize = 0;
    popSize = 0;

    temp1 = pool;
    for(i = 0; i < st_repo_p.nRep; i++) {
        insert(temp1, i);
        temp1 = temp1->child;
    }
    i = 0;
    do {
        temp1 = pool->child;
        if(temp1 == NULL) {
            break;
        }
        insert(elite, temp1->index);
        frontSize = 1;
        temp2 = elite->child;
        temp1 = deleteNode(temp1);
        temp1 = temp1->child;
        do {
            temp2 = elite->child;
            if(temp1 == NULL) {
                break;
            }
            do {
                end = 0;
                result = dominanceComparator(&(st_repo_p.obj[(temp1->index) * st_global_p.nObj]),
                                             &(st_repo_p.obj[(temp2->index) * st_global_p.nObj]));
                if(result == 1) {
                    insert(pool, temp2->index);
                    temp2 = deleteNode(temp2);
                    frontSize--;
                    temp2 = temp2->child;
                }
                if(result == 0) {
                    temp2 = temp2->child;
                }
                if(result == -1 || result == 2) {
                    end = 1;
                }
            } while((end != 1) && (temp2 != NULL));

            if(result == 0 || result == 1) {
                insert(elite, temp1->index);
                frontSize++;
                temp1 = deleteNode(temp1);
            }
            if(result == 2) {
                temp1 = deleteNode(temp1);//////////////////////////////////////////////////////////////////////////
            }
            temp1 = temp1->child;
        } while(temp1 != NULL);

        if(rank == 1) {
            if(frontSize <= st_archive_p.nArch) {
                st_global_p.PF_size = frontSize;
            } else {
                st_global_p.PF_size = st_archive_p.nArch;
            }
        }
        temp2 = elite->child;
        j = i;
        if((popSize + frontSize) <= st_archive_p.nArch) {
            do {
                copyToArchiveFromRepository(i, temp2->index);
                st_archive_p.rank[i] = rank;
                popSize += 1;
                temp2 = temp2->child;
                i += 1;
            } while(temp2 != NULL);
            // 			assignCrowdingDistanceIndexes(j,i-1);
            rank += 1;
        } else {
            K_Neighbor_Nearest_SDE(i, frontSize, elite);
            popSize = st_archive_p.nArch;
            for(j = i; j < popSize; j++) {
                st_archive_p.rank[j] = rank;
            }
        }
        temp2 = elite->child;
        do {
            temp2 = deleteNode(temp2);
            temp2 = temp2->child;
        } while(elite->child != NULL);
    } while(popSize < st_archive_p.nArch);

    // 	if (strct_repo_info.nRep<strct_archive_info.nArch)
    // 	{
    // 		strct_archive_info.cnArch=strct_repo_info.nRep;
    // 	}
    // 	else
    // 	{
    // 		strct_archive_info.cnArch=strct_archive_info.nArch;
    // 	}

    st_archive_p.cnArch = popSize < st_archive_p.nArch ? popSize : st_archive_p.nArch;
    while(st_archive_p.cnArch < st_archive_p.nArch && st_archive_p.cnArch < st_repo_p.nRep) {
        int* tmpIND = (int*)malloc(st_repo_p.nRep * sizeof(int));
        for(int i = 0; i < st_repo_p.nRep; i++) tmpIND[i] = i;
        shuffle(tmpIND, st_repo_p.nRep);
        int tmpCNT = 0;
        for(int i = 0; i < st_repo_p.nRep && tmpCNT < st_archive_p.nArch - st_archive_p.cnArch; i++) {
            int curI = tmpIND[i];
            if(st_repo_p.flag[curI] == -1) {
                int ind_dst = st_archive_p.cnArch + tmpCNT;
                copyToArchiveFromRepository(ind_dst, curI);
                tmpCNT++;
            }
        }
        free(tmpIND);
        st_archive_p.cnArch += tmpCNT;
    }

    deleteList(pool);
    deleteList(elite);
    return;
}

void K_Neighbor_Nearest_SDE(int count, int frontSize, list* elite)
{
    int non_size = frontSize;
    int* non_indexes;

    non_indexes = (int*)malloc(non_size * sizeof(int));
    list* tmp = elite->child;
    for(int i = 0; i < non_size; i++) {
        non_indexes[i] = tmp->index;
        tmp = tmp->child;
    }

    //²éÕÒ²¢¼ÇÂ¼Ã¿¸öÄ¿±êÉÏµÄ×î´óÖµºÍ×îÐ¡Öµ
    for(int i = 0; i < st_global_p.nObj; ++i) {
        st_decomp_p.fun_min[i] = INF_DOUBLE;
        st_decomp_p.fun_max[i] = -100;
    }
    for(int i = 0; i < non_size; ++i) {
        for(int j = 0; j < st_global_p.nObj; ++j) {
            if(st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j] < st_decomp_p.fun_min[j])
                st_decomp_p.fun_min[j] = st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j];
            if(st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j] > st_decomp_p.fun_max[j])
                st_decomp_p.fun_max[j] = st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j];
        }
    }

    //¼ÆËã¸÷¸ö·ÇÕ¼ÓÅ½â£¬¾­¹ýshift-basedµÄÅ·¼¸ÀïµÃ¾àÀë¾ØÕó
    double** dist;				//¼ÇÂ¼¾àÀë¾ØÕó
    int** distIndex;			//¼ÇÂ¼ÅÅÐòºóµÄ¾àÀë¾ØÕó¶ÔÓ¦µÄ·ÇÕ¼ÓÅ½âÏÂ±ê
    int* flag;					//±ê¼ÇÒÑ±»É¾³ýµÄ½âÎª0£¬Î´±»É¾³ýµÄ½âÎª1
    dist = (double**)malloc(non_size * sizeof(double*));
    distIndex = (int**)malloc(non_size * sizeof(int*));
    flag = (int*)malloc(non_size * sizeof(int));

    double* distMatrix = generateDistMatrix(st_repo_p.obj, non_size, non_indexes);

    for(int i = 0; i < non_size; ++i) {
        flag[i] = 1;
        dist[i] = (double*)malloc(non_size * sizeof(double));
        distIndex[i] = (int*)malloc(non_size * sizeof(int));
        for(int j = 0; j < non_size; ++j) {
            if(i == j)
                dist[i][j] = INF_DOUBLE;
            else
                dist[i][j] = distMatrix[i * non_size + j];
            distIndex[i][j] = j;
        }
    }

    //¶Ô¾àÀë¾ØÕó½øÐÐ´øÏÂ±êµÄÅÅÐò
    for(int i = 0; i < non_size; ++i)
        sort_dist_index(dist[i], distIndex[i], 0, non_size - 1);

    //µü´úÌÞ³ýnon_size-NA¸öÓµ¼··ÇÕ¼ÓÅ½â
    int cur_size = non_size;
    while(cur_size > st_archive_p.nArch - count) {
        //É¾³ýÁÚ½ü¾àÀë×îÐ¡µÄ´æµµ½â²Ù×÷£¬Í¨¹ý¶Ô¸Ã½â½øÐÐÒÑ´¦Àí±ê¼ÇÀ´ÊµÏÖ
        int crowd_index = -1;						//¼ÇÂ¼ÁÚ½ü¾àÀë×îÐ¡µÄ´æµµ½âÏÂ±ê
        for(int i = 0; i < non_size; ++i) {
            if(flag[i]) {
                crowd_index = i;
                break;
            }
        }
        if(crowd_index == -1) {
            printf("%s, crowd_index not changed, ERROR\n", AT);
        }
        for(int i = crowd_index + 1; i < non_size; ++i) {
            if(flag[i] && cmp_crowd(dist[i], dist[crowd_index]))
                crowd_index = i;
        }
        flag[crowd_index] = 0;

        //ÐÞ¸ÄÁÚ½ü¾àÀë¾ØÕó£¬ÕÒµ½É¾³ý½âÏÂ±ê£¬È»ºó½«ÁÚ½ü¾àÀëÒ»ÖÂÇ°ÒÆ
        for(int i = 0; i < non_size; ++i) {
            if(flag[i] == 0) continue;
            for(int j = 0; j < cur_size; ++j) {
                if(distIndex[i][j] == crowd_index) {
                    while(j < cur_size - 1) {
                        dist[i][j] = dist[i][j + 1];
                        distIndex[i][j] = distIndex[i][j + 1];
                        j++;
                    }
                    dist[i][j] = INF_DOUBLE;				//´ËÊ±j=cur_size-1
                    distIndex[i][j] = crowd_index;
                    break;
                }
            }
        }
        cur_size -= 1;
    }
    //µü´úÌÞ³ý½áÊø

    //±£´æ¾­¹ýÌÞ³ý²Ù×÷ºóµÄ´æµµ½âµ½AÖÐ
    int p_arch = count;
    for(int i = 0; i < non_size; ++i) {
        if(flag[i]) {
            copyToArchiveFromRepository(p_arch, non_indexes[i]);
            p_arch++;
        }
    }
    for(int i = 0; i < non_size; ++i) {
        free(dist[i]);
        free(distIndex[i]);
    }
    free(dist);
    free(distIndex);
    free(flag);
    free(non_indexes);
    free(distMatrix);
}

void K_Neighbor_Nearest_SDE(int count, int frontSize, list* elite, double* _dest, double* _destFit, double* _destDens,
                            double* _dest_one_group, int _n)
{
    int non_size = frontSize;
    int* non_indexes;

    non_indexes = (int*)malloc(non_size * sizeof(int));
    list* tmp = elite->child;
    for(int i = 0; i < non_size; i++) {
        non_indexes[i] = tmp->index;
        tmp = tmp->child;
    }

    //²éÕÒ²¢¼ÇÂ¼Ã¿¸öÄ¿±êÉÏµÄ×î´óÖµºÍ×îÐ¡Öµ
    for(int i = 0; i < st_global_p.nObj; ++i) {
        st_decomp_p.fun_min[i] = INF_DOUBLE;
        st_decomp_p.fun_max[i] = -100;
    }
    for(int i = 0; i < non_size; ++i) {
        for(int j = 0; j < st_global_p.nObj; ++j) {
            if(st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j] < st_decomp_p.fun_min[j])
                st_decomp_p.fun_min[j] = st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j];
            if(st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j] > st_decomp_p.fun_max[j])
                st_decomp_p.fun_max[j] = st_repo_p.obj[non_indexes[i] * st_global_p.nObj + j];
        }
    }

    //¼ÆËã¸÷¸ö·ÇÕ¼ÓÅ½â£¬¾­¹ýshift-basedµÄÅ·¼¸ÀïµÃ¾àÀë¾ØÕó
    double** dist;				//¼ÇÂ¼¾àÀë¾ØÕó
    int** distIndex;			//¼ÇÂ¼ÅÅÐòºóµÄ¾àÀë¾ØÕó¶ÔÓ¦µÄ·ÇÕ¼ÓÅ½âÏÂ±ê
    int* flag;					//±ê¼ÇÒÑ±»É¾³ýµÄ½âÎª0£¬Î´±»É¾³ýµÄ½âÎª1
    dist = (double**)malloc(non_size * sizeof(double*));
    distIndex = (int**)malloc(non_size * sizeof(int*));
    flag = (int*)malloc(non_size * sizeof(int));

    double* distMatrix = generateDistMatrix(st_repo_p.obj, non_size, non_indexes);

    for(int i = 0; i < non_size; ++i) {
        flag[i] = 1;
        dist[i] = (double*)malloc(non_size * sizeof(double));
        distIndex[i] = (int*)malloc(non_size * sizeof(int));
        for(int j = 0; j < non_size; ++j) {
            if(i == j)
                dist[i][j] = INF_DOUBLE;
            else
                dist[i][j] = distMatrix[i * non_size + j];
            distIndex[i][j] = j;
        }
    }

    //¶Ô¾àÀë¾ØÕó½øÐÐ´øÏÂ±êµÄÅÅÐò
    for(int i = 0; i < non_size; ++i)
        sort_dist_index(dist[i], distIndex[i], 0, non_size - 1);

    //µü´úÌÞ³ýnon_size-NA¸öÓµ¼··ÇÕ¼ÓÅ½â
    int cur_size = non_size;
    while(cur_size > st_archive_p.nArch - count) {
        //É¾³ýÁÚ½ü¾àÀë×îÐ¡µÄ´æµµ½â²Ù×÷£¬Í¨¹ý¶Ô¸Ã½â½øÐÐÒÑ´¦Àí±ê¼ÇÀ´ÊµÏÖ
        int crowd_index = -1;						//¼ÇÂ¼ÁÚ½ü¾àÀë×îÐ¡µÄ´æµµ½âÏÂ±ê
        for(int i = 0; i < non_size; ++i) {
            if(flag[i]) {
                crowd_index = i;
                break;
            }
        }
        if(crowd_index == -1) {
            printf("%s, crowd_index not changed, ERROR\n", AT);
        }
        for(int i = crowd_index + 1; i < non_size; ++i) {
            if(flag[i] && cmp_crowd(dist[i], dist[crowd_index]))
                crowd_index = i;
        }
        flag[crowd_index] = 0;

        //ÐÞ¸ÄÁÚ½ü¾àÀë¾ØÕó£¬ÕÒµ½É¾³ý½âÏÂ±ê£¬È»ºó½«ÁÚ½ü¾àÀëÒ»ÖÂÇ°ÒÆ
        for(int i = 0; i < non_size; ++i) {
            if(flag[i] == 0) continue;
            for(int j = 0; j < cur_size; ++j) {
                if(distIndex[i][j] == crowd_index) {
                    while(j < cur_size - 1) {
                        dist[i][j] = dist[i][j + 1];
                        distIndex[i][j] = distIndex[i][j + 1];
                        j++;
                    }
                    dist[i][j] = INF_DOUBLE;				//´ËÊ±j=cur_size-1
                    distIndex[i][j] = crowd_index;
                    break;
                }
            }
        }
        cur_size -= 1;
    }
    //µü´úÌÞ³ý½áÊø

    //±£´æ¾­¹ýÌÞ³ý²Ù×÷ºóµÄ´æµµ½âµ½AÖÐ
    int p_arch = count;
    for(int i = 0; i < non_size; ++i) {
        if(flag[i]) {
            copyToArchiveFromRepository(p_arch, non_indexes[i]);
            copyFromRepository(non_indexes[i], p_arch, _dest, _destFit, _destDens, _dest_one_group);
            p_arch++;
        }
    }
    for(int i = 0; i < non_size; ++i) {
        free(dist[i]);
        free(distIndex[i]);
    }
    free(dist);
    free(distIndex);
    free(flag);
    free(non_indexes);
    free(distMatrix);
}

double* generateDistMatrix(double* f, int non_size, int* non_indexes)
{
    double* distMatrix = (double*)malloc(non_size * non_size * sizeof(double));

    int i, j;
    for(i = 0; i < non_size; i++) {
        distMatrix[i * non_size + i] = 0.0;
        for(j = i + 1; j < non_size; j++) {
            distMatrix[i * non_size + j] =
                distMatrix[j * non_size + i] =
                    Euclid_Dist(&f[non_indexes[i] * st_global_p.nObj], &f[non_indexes[j] * st_global_p.nObj]);
        }
    }
    return distMatrix;
}

// ¼ÆËã¹éÒ»»¯µÄÅ·¼¸ÀïµÃ¾àÀë£¬vec1 µ½ÆäËüvec2µÄ¾àÀë
double Euclid_Dist(double* vec1, double* vec2)
{
    double sum = 0;
    for(int n = 0; n < st_global_p.nObj; n++) {
        if(vec1[n] < vec2[n])			//¾­¹ýSPEA2ÖÐµÄSDE²Ù×÷ºóµÄÅ·¼¸ÀïµÃ¾àÀë
            sum += ((vec1[n] - vec2[n]) * (vec1[n] - vec2[n])) / ((st_decomp_p.fun_max[n] - st_decomp_p.fun_min[n]) *
                    (st_decomp_p.fun_max[n] - st_decomp_p.fun_min[n]));
    }
    return sqrt(sum);
}

// record index orser after sorting
void sort_dist_index(double* a, int* b, int left, int right)
{
    int pivot;
    int i, j;
    double temp;
    double temp_a;
    int temp_b;
    if(left < right) {
        temp = a[right];
        i = left - 1;
        for(j = left; j < right; ++j) {
            if(a[j] <= temp) {
                i += 1;
                temp_a = a[i];
                a[i] = a[j];
                a[j] = temp_a;
                temp_b = b[i];
                b[i] = b[j];
                b[j] = temp_b;
            }
        }
        pivot = i + 1;
        temp_a = a[pivot];
        a[pivot] = a[right];
        a[right] = temp_a;
        temp_b = b[pivot];
        b[pivot] = b[right];
        b[right] = temp_b;
        sort_dist_index(a, b, left, pivot - 1);
        sort_dist_index(a, b, pivot + 1, right);
    }
}

//K½üÁÚ±È½Ïº¯Êý
bool cmp_crowd(double* a, double* b)
{
    for(int i = 0; i < 10; ++i)				//ÕâÀï×î¶à±È½ÏÇ°NA¸öÁÚ½ü¾àÀë
        if(a[i] != b[i])
            return a[i] < b[i];
    return  false;
}
