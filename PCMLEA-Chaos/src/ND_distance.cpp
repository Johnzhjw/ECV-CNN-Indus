# include "global.h"

void fillCrowdingDistance(int count, int frontSize, list* elite)
{
    int* distance = NULL;
    list* temp;
    int i, j;
    int missing = st_archive_p.nArch - count;
    while(missing < frontSize) {
        assignCrowdingDistanceList(elite->child, frontSize);
        distance = (int*)calloc(frontSize, sizeof(int));
        temp = elite->child;
        for(j = 0; j < frontSize; j++) {
            distance[j] = temp->index;
            temp = temp->child;
        }
        // 		quickSortDistance(distance, frontSize);
        double min_dist = st_repo_p.dens[distance[0]];
        int min_ind = distance[0];
        for(int k = 1; k < frontSize; k++) {
            if(st_repo_p.dens[distance[k]] < min_dist) {
                min_dist = st_repo_p.dens[distance[k]];
                min_ind = distance[k];
            }
        }
        // 		deleteInd(elite,distance[0]);
        deleteInd(elite, min_ind);
        frontSize--;
        free(distance);
    }
    temp = elite->child;
    for(i = count, j = frontSize - 1; i < st_archive_p.nArch; i++, j--) {
        copyToArchiveFromRepository(i, temp->index);
        temp = temp->child;
    }
    return;
}

void fillCrowdingDistance(int count, int frontSize, list* elite, double* _dest, double* _destFit, double* _destDens,
                          double* _dest_one_group, int _n)
{
    int* distance = NULL;
    list* temp;
    int i, j;
    int missing = _n - count;
    while(missing < frontSize) {
        assignCrowdingDistanceList(elite->child, frontSize);
        distance = (int*)calloc(frontSize, sizeof(int));
        temp = elite->child;
        for(j = 0; j < frontSize; j++) {
            distance[j] = temp->index;
            temp = temp->child;
        }
        //quickSortDistance(distance, frontSize);
        double min_dist = st_repo_p.dens[distance[0]];
        int min_ind = distance[0];
        for(int k = 1; k < frontSize; k++) {
            if(st_repo_p.dens[distance[k]] < min_dist) {
                min_dist = st_repo_p.dens[distance[k]];
                min_ind = distance[k];
            }
        }
        //deleteInd(elite,distance[0]);
        deleteInd(elite, min_ind);
        frontSize--;
        free(distance);
    }
    temp = elite->child;
    for(i = count, j = frontSize - 1; i < _n; i++, j--) {
        copyFromRepository(temp->index, i, _dest, _destFit, _destDens, _dest_one_group);
        temp = temp->child;
    }
    return;
}

void assignCrowdingDistanceList(list* lst, int frontSize)
{
    int** arrayFx;
    int* distance;
    int i, j;
    list* temp;
    temp = lst;
    if(frontSize == 1) {
        st_repo_p.dens[lst->index] = INF_DOUBLE;
        return;
    }
    if(frontSize == 2) {
        st_repo_p.dens[lst->index] = INF_DOUBLE;
        st_repo_p.dens[lst->child->index] = INF_DOUBLE;
        return;
    }
    arrayFx = (int**)calloc(st_global_p.nObj, sizeof(int*));
    distance = (int*)calloc(frontSize, sizeof(int));
    for(i = 0; i < st_global_p.nObj; i++) {
        arrayFx[i] = (int*)calloc(frontSize, sizeof(int));
    }
    for(j = 0; j < frontSize; j++) {
        distance[j] = temp->index;
        temp = temp->child;
    }
    assignCrowdingDistance(distance, arrayFx, frontSize);
    free(distance);
    for(i = 0; i < st_global_p.nObj; i++) {
        free(arrayFx[i]);
    }
    free(arrayFx);
    return;
}

void assignCrowdingDistanceIndexes(int c1, int c2)
{
    int** arrayFx;
    int* distance;
    int i, j;
    int frontSize;
    frontSize = c2 - c1 + 1;
    if(frontSize == 1) {
        st_repo_p.dens[c1] = INF_DOUBLE;
        return;
    }
    if(frontSize == 2) {
        st_repo_p.dens[c1] = INF_DOUBLE;
        st_repo_p.dens[c2] = INF_DOUBLE;
        return;
    }
    arrayFx = (int**)calloc(st_global_p.nObj, sizeof(int*));
    distance = (int*)calloc(frontSize, sizeof(int));
    for(i = 0; i < st_global_p.nObj; i++) {
        arrayFx[i] = (int*)calloc(frontSize, sizeof(int));
    }
    for(j = 0; j < frontSize; j++) {
        distance[j] = c1++;
    }
    assignCrowdingDistance(distance, arrayFx, frontSize);
    free(distance);
    for(i = 0; i < st_global_p.nObj; i++) {
        free(arrayFx[i]);
    }
    free(arrayFx);
    return;
}

void assignCrowdingDistance(int* distance, int** arrayFx, int frontSize)
{
    int i, j;
    for(i = 0; i < st_global_p.nObj; i++) {
        for(j = 0; j < frontSize; j++) {
            arrayFx[i][j] = distance[j];
        }
        quickSortFrontObj(i, arrayFx[i], frontSize);
    }
    for(j = 0; j < frontSize; j++) {
        st_repo_p.dens[distance[j]] = 0.0;
    }
    for(i = 0; i < st_global_p.nObj; i++) {
        st_repo_p.dens[arrayFx[i][0]] = INF_DOUBLE;
    }
    for(i = 0; i < st_global_p.nObj; i++) {
        for(j = 1; j < frontSize - 1; j++) {
            if(st_repo_p.dens[arrayFx[i][j]] != INF_DOUBLE) {
                if(st_repo_p.obj[(arrayFx[i][frontSize - 1]) * st_global_p.nObj + i] == st_repo_p.obj[(arrayFx[i][0]) *
                        st_global_p.nObj + i]) {
                    st_repo_p.dens[arrayFx[i][j]] += 0.0;
                } else {
                    st_repo_p.dens[arrayFx[i][j]] +=
                        (st_repo_p.obj[arrayFx[i][j + 1] * st_global_p.nObj + i] - st_repo_p.obj[arrayFx[i][j - 1] *
                                st_global_p.nObj + i]) /
                        (st_repo_p.obj[arrayFx[i][frontSize - 1] * st_global_p.nObj + i] - st_repo_p.obj[arrayFx[i][0] *
                                st_global_p.nObj + i]);
                }
            }
        }
    }

    for(j = 0; j < frontSize; j++) {
        //if(strct_repo_info.dens[distance[j]] > INF_DOUBLE) {
        //    printf("%s(%d): Too large value occurs - %g\n", strct_repo_info.dens[distance[j]]);
        //}
        if(st_repo_p.dens[distance[j]] != INF_DOUBLE) {
            st_repo_p.dens[distance[j]] = (st_repo_p.dens[distance[j]]) / st_global_p.nObj;
        }
    }
    return;
}
