#include "global.h"
#include <math.h>

void ControlVarAnalysis()
{
    // Analyze the decision variable strct_grp_ana_vals.Control the convergence or spread aspects
    //distance parameter,  position parameter or mixed parameter.
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int* recv = (int*)malloc(size * sizeof(int));
    int* disp = (int*)malloc(size * sizeof(int));
    int quo, rem;
    quo = st_global_p.nDim / size;
    rem = st_global_p.nDim % size;
    int i, j;
    for(i = 0; i < size; i++) {
        recv[i] = quo;
        if(i < rem)recv[i]++;
    }
    disp[0] = 0;
    for(i = 1; i < size; i++) {
        disp[i] = disp[i - 1] + recv[i - 1];
    }
    double* randomIndiv = (double*)calloc(st_global_p.nDim, sizeof(double));
    double* randomFit = (double*)calloc(st_global_p.nObj, sizeof(double));
    double* S_store = (double*)calloc(st_grp_ana_p.NumControlAnalysis * st_global_p.nObj, sizeof(double));
    for(i = 0; i < st_global_p.nDim; i++) {
        st_grp_ana_p.Control_Mean[i] = 0.0;
        st_grp_ana_p.Control_Dist_Mean[i] = 0.0;
    }
    int task_start = disp[rank];
    int task_num = recv[rank];
    int task_end = task_start + task_num;
    int* tmp = (int*)malloc((task_num + 1) * sizeof(int));
    double* tmpD = (double*)malloc((task_num + 1) * sizeof(double));

    //	for(n=0;n<strct_grp_ana_vals.NumRepControlAnalysis;n++)
    {
        for(i = task_start; i < task_end; i++) {
            for(j = 0; j < st_global_p.nDim;
                j++) randomIndiv[j] = st_global_p.minLimit[j] + 0.5 * (st_global_p.maxLimit[j] -
                                          st_global_p.minLimit[j]);   //rndreal(strct_global_paras.minLimit[j],strct_global_paras.maxLimit[j]);
            for(j = 0; j < st_grp_ana_p.NumControlAnalysis; j++) {
                randomIndiv[i] = (j + 0.5) / st_grp_ana_p.NumControlAnalysis * (st_global_p.maxLimit[i] -
                                 st_global_p.minLimit[i]) + st_global_p.minLimit[i];
                EMO_evaluate_problems(st_global_p.testInstance, randomIndiv, randomFit, st_global_p.nDim, 1,
                                      st_global_p.nObj);
                memcpy(&S_store[j * st_global_p.nObj], randomFit, st_global_p.nObj * sizeof(double));
            } //end for j
            // 		strct_grp_ana_vals.Control[i] = dominate(S_store);
            tmp[i - task_start] = dominate_main(S_store);
            /*			if(tmp[i-task_start]==0)
            tmpD[i-task_start] = -CalcDistance(S_store);
            else
            tmpD[i-task_start] = 0.0;*/
        }//end for i
        // Allgather
        MPI_Allgatherv(tmp, task_num, MPI_INT,
                       st_grp_ana_p.Control, recv, disp, MPI_INT,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(tmpD, task_num, MPI_DOUBLE,
                       st_grp_ana_p.Control_Dist, recv, disp, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for(i = 0; i < st_global_p.nDim; i++) {
            st_grp_ana_p.Control_Mean[i] += (double)st_grp_ana_p.Control[i] /
                                            st_grp_ana_p.NumRepControlAnalysis;
            //			strct_grp_ana_vals.Control_Dist_Mean[i]+=(double)strct_grp_ana_vals.Control_Dist[i]/strct_grp_ana_vals.NumRepControlAnalysis;
        }
    }

    // statistics the variables which strct_grp_ana_vals.Control the diversity or/and convergence.
    st_grp_ana_p.numConverIndexes = 0;
    st_grp_ana_p.numDiverIndexes = 0;

    /*	int tSum=0;
    for(i=0;i<strct_global_paras.nDim;i++) if(strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1) tSum++;

    if(tSum<strct_grp_info_vals.limitDiverIndex)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    tSum=0;
    for(i=0;i<strct_global_paras.nDim;i++) if(strct_grp_ana_vals.Control[i]==0) tSum++;
    int tRem=strct_grp_info_vals.limitDiverIndex-tSum;

    int *tInd=(int*)calloc(strct_global_paras.nDim,sizeof(int));
    bool* tFlag=(bool*)calloc(strct_global_paras.nDim,sizeof(bool));

    if(tSum<strct_grp_info_vals.limitDiverIndex)
    {
    for(i=0;i<strct_global_paras.nDim;i++)
    {
    tInd[i]=i;
    if(strct_grp_ana_vals.Control[i]==0)
    tFlag[i]=true;
    else
    tFlag[i]=false;
    }
    shuffle(tInd,strct_global_paras.nDim);
    int tmp=0;

    for(i=0;i<strct_global_paras.nDim;i++)
    {
    if(strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1 && tmp<tRem)
    {
    tFlag[tInd[i]]=true;
    tmp++;
    }
    }
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tFlag[i])
    strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=i;
    else
    strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=i;
    }
    else
    {
    for(i=0;i<strct_global_paras.nDim;i++)
    {
    tInd[i]=i;
    tFlag[i]=false;
    }
    shuffle(tInd,strct_global_paras.nDim);
    int tmp=0;

    for(i=0;i<strct_global_paras.nDim;i++)
    {
    if(strct_grp_ana_vals.Control[tInd[i]]==0 && tmp<strct_grp_info_vals.limitDiverIndex)
    {
    tFlag[tInd[i]]=true;
    tmp++;
    }
    }
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tFlag[i])
    strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=i;
    else
    strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=i;
    }

    free(tInd);
    free(tFlag);
    }*/

    /*	if(tSum<=strct_grp_info_vals.limitDiverIndex)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control_Mean[i] <= 0 && strct_grp_ana_vals.Control_Mean[i]>-strct_grp_ana_vals.NumControlAnalysis*strct_grp_ana_vals.NumControlAnalysis) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    double tN=-strct_grp_ana_vals.NumControlAnalysis*strct_grp_ana_vals.NumControlAnalysis;
    while(tSum>strct_grp_info_vals.limitDiverIndex)
    {
    double tmp=999;
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tmp>strct_grp_ana_vals.Control_Mean[i]&&strct_grp_ana_vals.Control_Mean[i]>tN)
    tmp=strct_grp_ana_vals.Control_Mean[i];
    tN=tmp;
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]==tN)
    tSum--;
    }
    if(tN<=0)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control_Mean[i] <= 0 && strct_grp_ana_vals.Control_Mean[i] > tN) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    tSum=0;
    int index[strct_global_paras.nDim];
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]==0)
    {
    index[tSum++]=i;
    }

    myQuickSort(strct_grp_ana_vals.Control_Dist_Mean, index, 0, tSum-1);

    for(i=0;i<strct_grp_info_vals.limitDiverIndex;i++) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(index[i]);
    for(i=0;i<strct_global_paras.nDim;i++)
    if(!IsDiversityVariable(i)) strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);

    for(i=0;i<strct_grp_ana_vals.numDiverIndexes;i++)
    {
    for(j=i+1;j<strct_grp_ana_vals.numDiverIndexes;j++)
    {
    if(strct_grp_info_vals.DiversityIndexs[i]>strct_grp_info_vals.DiversityIndexs[j])
    {
    int temp=strct_grp_info_vals.DiversityIndexs[i];
    strct_grp_info_vals.DiversityIndexs[i]=strct_grp_info_vals.DiversityIndexs[j];
    strct_grp_info_vals.DiversityIndexs[j]=temp;
    }
    }
    }
    }
    }*/

    for(i = 0; i < st_global_p.nDim; i++) {
        if(st_grp_ana_p.Control[i] == 0
           || st_grp_ana_p.Control[i] == -1) st_grp_info_p.DiversityIndexs[st_grp_ana_p.numDiverIndexes++] = (i);
        else st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = (i);
    }

    if(st_grp_ana_p.numDiverIndexes == st_global_p.nDim) {
        st_grp_ana_p.numDiverIndexes = 0;
        st_grp_ana_p.numConverIndexes = 0;
        for(i = 0; i < st_global_p.nDim; i++) {
            if(st_grp_ana_p.Control[i] == -1) st_grp_info_p.DiversityIndexs[st_grp_ana_p.numDiverIndexes++] = (i);
            else st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = (i);
        }
    }

    free(recv);
    free(disp);
    free(tmp);
    free(tmpD);
    free(randomIndiv);
    free(randomFit);
    free(S_store);

    MPI_Bcast(&st_grp_ana_p.numConverIndexes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(st_grp_ana_p.numConverIndexes)
        MPI_Bcast(st_grp_info_p.ConvergenceIndexs, st_grp_ana_p.numConverIndexes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&st_grp_ana_p.numDiverIndexes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(st_grp_ana_p.numDiverIndexes)
        MPI_Bcast(st_grp_info_p.DiversityIndexs, st_grp_ana_p.numDiverIndexes, MPI_INT, 0, MPI_COMM_WORLD);

    if(st_grp_ana_p.numDiverIndexes != (st_global_p.nObj - 1)) {
        if(0 == st_MPI_p.mpi_rank) {
            printf("strct_grp_ana_vals.Control ANALYSIS MAYBE WRONG...RANK%d_%s_OBJ%d_RUN%d_NDIV%d\n",
                   rank,
                   st_global_p.testInstance,
                   st_global_p.nObj,
                   st_ctrl_p.cur_run,
                   st_grp_ana_p.numDiverIndexes);
        }
    }

    /*	if(strct_grp_ana_vals.numConverIndexes==0)
    {
    printf("NO convergence variables, exiting...\n");
    exit(-999);
    }*/
}

void ControlVarAnalysis_serial()
{
    // Analyze the decision variable strct_grp_ana_vals.Control the convergence or spread aspects
    //distance parameter,  position parameter or mixed parameter.
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int i, j;
    double* randomIndiv = (double*)calloc(st_global_p.nDim, sizeof(double));
    double* randomFit = (double*)calloc(st_global_p.nObj, sizeof(double));
    double* S_store = (double*)calloc(st_grp_ana_p.NumControlAnalysis * st_global_p.nObj, sizeof(double));
    for(i = 0; i < st_global_p.nDim; i++) {
        st_grp_ana_p.Control_Mean[i] = 0.0;
        st_grp_ana_p.Control_Dist_Mean[i] = 0.0;
    }

    //	for(n=0;n<strct_grp_ana_vals.NumRepControlAnalysis;n++)
    {
        for(i = 0; i < st_global_p.nDim; i++) {
            for(j = 0; j < st_global_p.nDim;
                j++) randomIndiv[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);
            for(j = 0; j < st_grp_ana_p.NumControlAnalysis; j++) {
                randomIndiv[i] = (j + 0.5) / st_grp_ana_p.NumControlAnalysis * (st_global_p.maxLimit[i] -
                                 st_global_p.minLimit[i]) + st_global_p.minLimit[i];
                EMO_evaluate_problems(st_global_p.testInstance, randomIndiv, randomFit, st_global_p.nDim, 1,
                                      st_global_p.nObj);
                memcpy(&S_store[j * st_global_p.nObj], randomFit, st_global_p.nObj * sizeof(double));
            } //end for j
            st_grp_ana_p.Control[i] = dominate_main(S_store);
            /*			if(strct_grp_ana_vals.Control[i]==0)
            strct_grp_ana_vals.Control_Dist[i] = -CalcDistance(S_store);
            else
            strct_grp_ana_vals.Control_Dist[i] = 0.0;*/
        }//end for i
        // Allgather

        for(i = 0; i < st_global_p.nDim; i++) {
            //			strct_grp_ana_vals.Control_Mean[i]+=(double)strct_grp_ana_vals.Control[i]/strct_grp_ana_vals.NumRepControlAnalysis;
            //			strct_grp_ana_vals.Control_Dist_Mean[i]+=(double)strct_grp_ana_vals.Control_Dist[i]/strct_grp_ana_vals.NumRepControlAnalysis;
        }
    }

    MPI_Bcast(st_grp_ana_p.Control, st_global_p.nDim, MPI_INT, 0, MPI_COMM_WORLD);
    //	MPI_Bcast(strct_grp_ana_vals.Control_Mean,strct_global_paras.nDim,MPI_DOUBLE,0,MPI_COMM_WORLD);
    //	MPI_Bcast(strct_grp_ana_vals.Control_Dist_Mean,strct_global_paras.nDim,MPI_DOUBLE,0,MPI_COMM_WORLD);

    // statistics the variables which strct_grp_ana_vals.Control the diversity or/and convergence.
    st_grp_ana_p.numConverIndexes = 0;
    st_grp_ana_p.numDiverIndexes = 0;

    /*	int tSum=0;
    for(i=0;i<strct_global_paras.nDim;i++) if(strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1) tSum++;

    if(tSum<strct_grp_info_vals.limitDiverIndex)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    tSum=0;
    for(i=0;i<strct_global_paras.nDim;i++) if(strct_grp_ana_vals.Control[i]==0) tSum++;
    int tRem=strct_grp_info_vals.limitDiverIndex-tSum;

    int *tInd=(int*)calloc(strct_global_paras.nDim,sizeof(int));
    bool* tFlag=(bool*)calloc(strct_global_paras.nDim,sizeof(bool));

    if(tSum<strct_grp_info_vals.limitDiverIndex)
    {
    for(i=0;i<strct_global_paras.nDim;i++)
    {
    tInd[i]=i;
    if(strct_grp_ana_vals.Control[i]==0)
    tFlag[i]=true;
    else
    tFlag[i]=false;
    }
    shuffle(tInd,strct_global_paras.nDim);
    int tmp=0;

    for(i=0;i<strct_global_paras.nDim;i++)
    {
    if(strct_grp_ana_vals.Control[i]==0 || strct_grp_ana_vals.Control[i]==-1 && tmp<tRem)
    {
    tFlag[tInd[i]]=true;
    tmp++;
    }
    }
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tFlag[i])
    strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=i;
    else
    strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=i;
    }
    else
    {
    for(i=0;i<strct_global_paras.nDim;i++)
    {
    tInd[i]=i;
    tFlag[i]=false;
    }
    shuffle(tInd,strct_global_paras.nDim);
    int tmp=0;

    for(i=0;i<strct_global_paras.nDim;i++)
    {
    if(strct_grp_ana_vals.Control[tInd[i]]==0 && tmp<strct_grp_info_vals.limitDiverIndex)
    {
    tFlag[tInd[i]]=true;
    tmp++;
    }
    }
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tFlag[i])
    strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=i;
    else
    strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=i;
    }

    free(tInd);
    free(tFlag);
    }*/

    /*	if(tSum<=strct_grp_info_vals.limitDiverIndex)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control_Mean[i] <= 0 && strct_grp_ana_vals.Control_Mean[i]>-strct_grp_ana_vals.NumControlAnalysis*strct_grp_ana_vals.NumControlAnalysis) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    double tN=-strct_grp_ana_vals.NumControlAnalysis*strct_grp_ana_vals.NumControlAnalysis;
    while(tSum>strct_grp_info_vals.limitDiverIndex)
    {
    double tmp=999;
    for(i=0;i<strct_global_paras.nDim;i++)
    if(tmp>strct_grp_ana_vals.Control_Mean[i]&&strct_grp_ana_vals.Control_Mean[i]>tN)
    tmp=strct_grp_ana_vals.Control_Mean[i];
    tN=tmp;
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]==tN)
    tSum--;
    }
    if(tN<=0)
    {
    for (i=0; i<strct_global_paras.nDim; i++)	{
    if (strct_grp_ana_vals.Control_Mean[i] <= 0 && strct_grp_ana_vals.Control_Mean[i] > tN) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(i);
    else strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);
    }
    }
    else
    {
    tSum=0;
    int index[strct_global_paras.nDim];
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]==0)
    {
    index[tSum++]=i;
    }

    myQuickSort(strct_grp_ana_vals.Control_Dist_Mean, index, 0, tSum-1);

    for(i=0;i<strct_grp_info_vals.limitDiverIndex;i++) strct_grp_info_vals.DiversityIndexs[strct_grp_ana_vals.numDiverIndexes++]=(index[i]);
    for(i=0;i<strct_global_paras.nDim;i++)
    if(!IsDiversityVariable(i)) strct_grp_info_vals.ConvergenceIndexs[strct_grp_ana_vals.numConverIndexes++]=(i);

    for(i=0;i<strct_grp_ana_vals.numDiverIndexes;i++)
    {
    for(j=i+1;j<strct_grp_ana_vals.numDiverIndexes;j++)
    {
    if(strct_grp_info_vals.DiversityIndexs[i]>strct_grp_info_vals.DiversityIndexs[j])
    {
    int temp=strct_grp_info_vals.DiversityIndexs[i];
    strct_grp_info_vals.DiversityIndexs[i]=strct_grp_info_vals.DiversityIndexs[j];
    strct_grp_info_vals.DiversityIndexs[j]=temp;
    }
    }
    }
    }
    }*/

    for(i = 0; i < st_global_p.nDim; i++) {
        if(st_grp_ana_p.Control[i] == 0
           || st_grp_ana_p.Control[i] == -1) st_grp_info_p.DiversityIndexs[st_grp_ana_p.numDiverIndexes++] = (i);
        else st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = (i);
    }

    free(randomIndiv);
    free(randomFit);

    MPI_Bcast(&st_grp_ana_p.numConverIndexes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(st_grp_ana_p.numConverIndexes)
        MPI_Bcast(st_grp_info_p.ConvergenceIndexs, st_grp_ana_p.numConverIndexes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&st_grp_ana_p.numDiverIndexes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(st_grp_ana_p.numDiverIndexes)
        MPI_Bcast(st_grp_info_p.DiversityIndexs, st_grp_ana_p.numDiverIndexes, MPI_INT, 0, MPI_COMM_WORLD);

    //if(st_grp_ana_p.numConverIndexes == 0) {
    //    printf("%s:NO convergence variables, exiting...\n", AT);
    //    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_CONVER_VAR);
    //}
}
