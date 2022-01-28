#include "global.h"
#include <math.h>

void initializePopulation_grouping()
{
    int i, j;
    int realInd;
    //////////////////////////////////////////////////////////////////////////
    int nDiv = st_grp_ana_p.numDiverIndexes;
    if(nDiv == 1) {
        for(i = 0; i < st_global_p.nPop; i++) {
            realInd = st_grp_info_p.DiversityIndexs[0];
            st_grp_ana_p.var_current_grp[i * st_global_p.nDim + realInd] =
                st_global_p.minLimit[realInd] +
                (double)(i) / (double)(st_global_p.nPop - 1) *
                (st_global_p.maxLimit[realInd] - st_global_p.minLimit[realInd]);
        }
    } else if(nDiv >= 2 && nDiv <= 4) {
        char FileName[256];
        sprintf(FileName, "DATA_alg/SamplePoint/SamplePoint_Dim%dN%d.txt", nDiv, st_global_p.nPop);
        FILE* fpt;
        fpt = fopen(FileName, "r");
        if(fpt == NULL) {
            printf("%s:\nsample point file error...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_SAMPLE_POINT_READING);
        }
        int tmp;
        double elem;
        for(i = 0; i < st_global_p.nPop; i++) {
            for(j = 0; j < nDiv; j++) {
                realInd = st_grp_info_p.DiversityIndexs[j];
                tmp = fscanf(fpt, "%lf", &elem);
                if(tmp == EOF) {
                    printf("%s:\nsample points are not enough...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_SAMPLE_POINT_NOT_ENOUGH);
                }
                st_grp_ana_p.var_current_grp[i * st_global_p.nDim + realInd] =
                    st_global_p.minLimit[realInd] +
                    elem *
                    (st_global_p.maxLimit[realInd] - st_global_p.minLimit[realInd]);
            }
        }
        fclose(fpt);
    } else {
        for(i = 0; i < st_global_p.nPop; i++) {
            for(j = 0; j < nDiv; j++) {
                realInd = st_grp_info_p.DiversityIndexs[j];
                st_grp_ana_p.var_current_grp[i * st_global_p.nDim + realInd] = rndreal(st_global_p.minLimit[realInd],
                        st_global_p.maxLimit[realInd]);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    int nConv = st_grp_ana_p.numConverIndexes;
    for(i = 0; i < st_global_p.nPop; i++) {
        for(j = 0; j < nConv; j++) {
            realInd = st_grp_info_p.ConvergenceIndexs[j];
            st_grp_ana_p.var_current_grp[i * st_global_p.nDim + realInd] = rndreal(st_global_p.minLimit[realInd],
                    st_global_p.maxLimit[realInd]);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    int* recv_size_grp = (int*)calloc(st_MPI_p.mpi_size, sizeof(int));
    int* disp_size_grp = (int*)calloc(st_MPI_p.mpi_size, sizeof(int));
    int* each_size_grp = (int*)calloc(st_MPI_p.mpi_size, sizeof(int));
    int quo = st_global_p.nPop / st_MPI_p.mpi_size;
    int rem = st_global_p.nPop % st_MPI_p.mpi_size;
    for(i = 0; i < st_MPI_p.mpi_size; i++) {
        each_size_grp[i] = quo;
        if(i < rem) each_size_grp[i]++;
    }
    update_recv_disp(each_size_grp, st_global_p.nDim, st_MPI_p.mpi_size, recv_size_grp, disp_size_grp);
    MPI_Scatterv(st_grp_ana_p.var_current_grp, recv_size_grp, disp_size_grp, MPI_DOUBLE,
                 st_grp_ana_p.var_repository_grp, recv_size_grp[st_MPI_p.mpi_rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    for(i = 0; i < each_size_grp[st_MPI_p.mpi_rank]; i++) {
        EMO_evaluate_problems(st_global_p.testInstance, &st_grp_ana_p.var_repository_grp[i * st_global_p.nDim],
                              &st_grp_ana_p.obj_repository_grp[i * st_global_p.nObj], st_global_p.nDim, 1, st_global_p.nObj);
    }
    update_recv_disp(each_size_grp, st_global_p.nDim, st_MPI_p.mpi_size, recv_size_grp, disp_size_grp);
    MPI_Allgatherv(st_grp_ana_p.var_repository_grp, recv_size_grp[st_MPI_p.mpi_rank], MPI_DOUBLE,
                   st_grp_ana_p.var_current_grp, recv_size_grp, disp_size_grp, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    update_recv_disp(each_size_grp, st_global_p.nObj, st_MPI_p.mpi_size, recv_size_grp, disp_size_grp);
    MPI_Allgatherv(st_grp_ana_p.obj_repository_grp, recv_size_grp[st_MPI_p.mpi_rank], MPI_DOUBLE,
                   st_grp_ana_p.obj_current_grp, recv_size_grp, disp_size_grp, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    free(recv_size_grp);
    free(disp_size_grp);
    free(each_size_grp);
    //////////////////////////////////////////////////////////////////////////
    MPI_Bcast(st_grp_ana_p.var_current_grp, st_global_p.nPop * st_global_p.nDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(st_grp_ana_p.obj_current_grp, st_global_p.nPop * st_global_p.nObj, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    return;
}

int dominate_main(double* Data)
{
    int i, j, count;
    int* front = (int*)calloc(st_grp_ana_p.NumControlAnalysis * st_grp_ana_p.NumControlAnalysis, sizeof(int));
    int* size_front = (int*)calloc(st_grp_ana_p.NumControlAnalysis, sizeof(int));
    int* dominate = (int*)calloc(st_grp_ana_p.NumControlAnalysis * st_grp_ana_p.NumControlAnalysis, sizeof(int));
    int* size_dominate = (int*)calloc(st_grp_ana_p.NumControlAnalysis * st_grp_ana_p.NumControlAnalysis, sizeof(int));
    int* num_dominated = (int*)calloc(st_grp_ana_p.NumControlAnalysis, sizeof(int));

    int temp;
    for(i = 0; i < st_grp_ana_p.NumControlAnalysis; i++) {
        for(j = i + 1; j < st_grp_ana_p.NumControlAnalysis; j++) {
            temp = dominate_judge(&Data[i * st_global_p.nObj], &Data[j * st_global_p.nObj]);
            if(temp == -1) {   //i dominate j
                int tmp = i * st_grp_ana_p.NumControlAnalysis + size_dominate[i];
                dominate[tmp] = j;
                num_dominated[j]++;
                size_dominate[i]++;
            } else if(temp == 1) { //j dominate i
                int tmp = j * st_grp_ana_p.NumControlAnalysis + size_dominate[j];
                dominate[tmp] = i;
                num_dominated[i]++;
                size_dominate[j]++;
            }
        }
        if(num_dominated[i] == 0) {
            front[size_front[0]] = i;
            size_front[0]++;
        }
    }
    if(size_front[0] == st_grp_ana_p.NumControlAnalysis) {
        free(front);
        free(size_front);
        free(dominate);
        free(size_dominate);
        free(num_dominated);
        return -1;//0;//Data.size();//-1;
    } else if(size_front[0] > 1) {
        free(front);
        free(size_front);
        free(dominate);
        free(size_dominate);
        free(num_dominated);
        return 0;//(front[0].size()-Data.size());//0;
    } else {
        count = 1;
        while(count <= st_grp_ana_p.NumControlAnalysis && size_front[count - 1] > 0) {
            for(i = 0; i < size_front[count - 1]; i++) {
                for(j = 0; j < size_dominate[front[(count - 1) * st_grp_ana_p.NumControlAnalysis + i]]; j++) {
                    int temp_index = dominate[front[(count - 1) * st_grp_ana_p.NumControlAnalysis + i] * st_grp_ana_p.NumControlAnalysis
                                                           + j];
                    num_dominated[temp_index]--;
                    if(num_dominated[temp_index] == 0) {
                        int tmp = count * st_grp_ana_p.NumControlAnalysis + size_front[count];
                        front[tmp] = temp_index;
                        size_front[count]++;
                    }
                }
            }
            if(count < st_grp_ana_p.NumControlAnalysis && size_front[count] > 1) {
                free(front);
                free(size_front);
                free(dominate);
                free(size_dominate);
                free(num_dominated);
                return 0;//(front[count].size()-(count+1)*Data.size());//0;
            }
            count++;
        }
        free(front);
        free(size_front);
        free(dominate);
        free(size_dominate);
        free(num_dominated);
        return 1;//(-Data.size()*Data.size());//1;
    }
}

int dominate_judge(double* pf1, double* pf2)
{
    //return 1:  pf2 donimate pf1;
    //return -1: pf1 donimate pf2;
    //return 0:  pf1,pf2 is non donimate;
    double big = 0, small = 0;
    for(int i = 0; i < st_global_p.nObj; i++) {
        if(pf1[i] >= pf2[i])	big++;
        if(pf1[i] <= pf2[i]) small++;
    }
    if(small == st_global_p.nObj)	return -1;
    else if(big == st_global_p.nObj) return 1;
    else return 0;
}

bool IsDistanceVariable(int j)
{
    int i = 0;
    for(i = 0; i < st_grp_ana_p.numConverIndexes && st_grp_info_p.ConvergenceIndexs[i] != j; i++) {}
    if(i >= st_grp_ana_p.numConverIndexes) return false;
    else return true;
}

bool IsDiversityVariable(int j)
{
    int i = 0;
    for(i = 0; i < st_grp_ana_p.numDiverIndexes && st_grp_info_p.DiversityIndexs[i] != j; i++) {}
    if(i >= st_grp_ana_p.numDiverIndexes) return false;
    else return true;
}

bool UpdateSolution(double* Parent, double* Offspring)
{
    //int temp = donimate_judge(population[i].indiv.y_obj, child.y_obj);
    //if (temp == 1) population[i].indiv = child;
    //else if (temp == 0) cout<<"The optimized variables have position or mixed variable(s)"<<endl;
    //double ParentASFValue = 0.0, OffspringASFValue = 0.0;
    //for (int j=0; j<nobj; j++) ParentASFValue += population[i].indiv.y_obj[j];
    //for (int j=0; j<nobj; j++) OffspringASFValue += child.y_obj[j];
    //if (OffspringASFValue < ParentASFValue) population[i].indiv = child;
    if(!strcmp(st_grp_ana_p.strUpdateSolutionMode, "Dominate")) {
        int temp = dominate_judge(Parent, Offspring);
        if(temp == 1) return true;
        else return false;
    } else if(!strcmp(st_grp_ana_p.strUpdateSolutionMode, "ValueSum")) {
        double ParentASFValue = 0.0, OffspringASFValue = 0.0;
        for(int j = 0; j < st_global_p.nObj; j++) ParentASFValue += Parent[j];
        for(int j = 0; j < st_global_p.nObj; j++) OffspringASFValue += Offspring[j];
        if(OffspringASFValue < ParentASFValue) return true;
        else return false;
    } else
        return false;
}

double CalcDistance(double* Data)
{
    int** arrIndex = (int**)calloc(st_global_p.nObj, sizeof(int*));
    for(int i = 0; i < st_global_p.nObj; i++) {
        arrIndex[i] = (int*)calloc(st_grp_ana_p.NumControlAnalysis, sizeof(int));
    }
    double** tObj = (double**)calloc(st_global_p.nObj, sizeof(double*));
    for(int i = 0; i < st_global_p.nObj; i++) {
        tObj[i] = (double*)calloc(st_grp_ana_p.NumControlAnalysis, sizeof(double));
    }

    //int i, j;
    for(int i = 0; i < st_global_p.nObj; i++) {
        for(int j = 0; j < st_grp_ana_p.NumControlAnalysis; j++) {
            arrIndex[i][j] = j;
            tObj[i][j] = Data[j * st_grp_ana_p.NumControlAnalysis + i];
        }
        myQuickSort(tObj[i], arrIndex[i], 0, st_grp_ana_p.NumControlAnalysis - 1);
    }

    double result = 0.0;
    for(int i = 0; i < st_global_p.nObj; i++) {
        double maxV = tObj[i][arrIndex[i][st_grp_ana_p.NumControlAnalysis - 1]] - tObj[i][arrIndex[i][0]];
        if(maxV == 0.0)
            result += 0.0;
        else {
            for(int j = 1; j < st_grp_ana_p.NumControlAnalysis - 1; j++) {
                result += (tObj[i][arrIndex[i][j + 1]] - tObj[i][arrIndex[i][j - 1]]) / maxV;
            }
        }
    }

    for(int i = 0; i < st_global_p.nObj; i++) {
        free(arrIndex[i]);
    }
    free(arrIndex);
    for(int i = 0; i < st_global_p.nObj; i++) {
        free(tObj[i]);
    }
    free(tObj);

    return (result / st_global_p.nObj);
}

void myQuickSort(double Data[], int indexArr[], int left, int right)
{
    int index;
    int temp;
    int i, j;
    double pivot;
    if(left < right) {
        index = rnd(left, right);
        temp = indexArr[right];
        indexArr[right] = indexArr[index];
        indexArr[index] = temp;
        pivot = Data[(indexArr[right])];
        i = left - 1;
        for(j = left; j < right; j++) {
            if(Data[(indexArr[j])] <= pivot) {
                i += 1;
                temp = indexArr[j];
                indexArr[j] = indexArr[i];
                indexArr[i] = temp;
            }
        }
        index = i + 1;
        temp = indexArr[index];
        indexArr[index] = indexArr[right];
        indexArr[right] = temp;
        myQuickSort(Data, indexArr, left, index - 1);
        myQuickSort(Data, indexArr, index + 1, right);
    }
    return;
}