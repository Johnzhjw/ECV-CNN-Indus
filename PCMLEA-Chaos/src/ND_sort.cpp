# include "global.h"

void quickSortDistance(int* distance, int frontSize)
{
    qSortDistance(distance, 0, frontSize - 1);
    return;
}

void qSortDistance(int* distance, int left, int right)
{
    int index;
    int temp;
    int i, j;
    double pivot;
    if(left < right) {
        index = rnd(left, right);
        temp = distance[right];
        distance[right] = distance[index];
        distance[index] = temp;
        pivot = st_repo_p.dens[distance[right]];
        i = left - 1;
        for(j = left; j < right; j++) {
            if(st_repo_p.dens[distance[j]] <= pivot) {
                i += 1;
                temp = distance[j];
                distance[j] = distance[i];
                distance[i] = temp;
            }
        }
        index = i + 1;
        temp = distance[index];
        distance[index] = distance[right];
        distance[right] = temp;
        qSortDistance(distance, left, index - 1);
        qSortDistance(distance, index + 1, right);
    }
    return;
}

void quickSortFrontObj(int objcount, int arrayFx[], int sizeArrayFx)
{
    qSortFrontObj(objcount, arrayFx, 0, sizeArrayFx - 1);
    return;
}

void qSortFrontObj(int objcount, int arrayFx[], int left, int right)
{
    int index;
    int temp;
    int i, j;
    double pivot;
    if(left < right) {
        index = rnd(left, right);
        temp = arrayFx[right];
        arrayFx[right] = arrayFx[index];
        arrayFx[index] = temp;
        pivot = st_repo_p.obj[(arrayFx[right]) * st_global_p.nObj + objcount];
        i = left - 1;
        for(j = left; j < right; j++) {
            if(st_repo_p.obj[(arrayFx[j]) * st_global_p.nObj + objcount] <= pivot) {
                i += 1;
                temp = arrayFx[j];
                arrayFx[j] = arrayFx[i];
                arrayFx[i] = temp;
            }
        }
        index = i + 1;
        temp = arrayFx[index];
        arrayFx[index] = arrayFx[right];
        arrayFx[right] = temp;
        qSortFrontObj(objcount, arrayFx, left, index - 1);
        qSortFrontObj(objcount, arrayFx, index + 1, right);
    }
    return;
}