#include "global.h"
#include <math.h>

void InterdependenceAnalysis_master_slave()
{
    initializePopulation_grouping();
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) printf("INIT_DONE\n");

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int myPopNum;
    int* recv = (int*)malloc(size * sizeof(int));
    int* disp = (int*)malloc(size * sizeof(int));
    int quo, rem;
    int task_sum = st_global_p.nPop;
    quo = task_sum / size;
    rem = task_sum % size;
    int i, j, l, k;
    for(i = 0; i < size; i++) {
        recv[i] = quo;
        if(i < rem)recv[i]++;
    }
    disp[0] = 0;
    for(i = 1; i < size; i++) {
        disp[i] = disp[i - 1] + recv[i - 1];
    }
    myPopNum = recv[rank];

    if(rank == 0) {
        //		for(i=0;i<strct_global_paras.nPop;i++)
        //		{
        //			for(j=0;j<strct_global_paras.nDim;j++)
        //			{
        //				xCurrent[i*strct_global_paras.nDim+j]=strct_global_paras.minLimit[j]+
        //					rnd_uni(&rnd_uni_init)*(strct_global_paras.maxLimit[j]-strct_global_paras.minLimit[j]);
        //			}
        //			for(j=0;j<strct_grp_ana_vals.numDiverIndexes;j++)
        //			{
        //				xCurrent[i*strct_global_paras.nDim+strct_grp_info_vals.DiversityIndexs[j]]=diver_pos_store_all[i*strct_global_paras.nDim+j];
        //			}
        //			evaluate_problems(strct_global_paras.testInstance,&xCurrent[i*strct_global_paras.nDim],
        //				&fCurrent[i*strct_global_paras.nObj],strct_global_paras.nDim,1,strct_global_paras.nObj);
        //		}

        // 		for(i=0;i<strct_global_paras.nPop;i++)
        // 		{
        // 			printf("NO.%d\n",i+1);
        // 			for(j=0;j<strct_global_paras.nDim;j++)
        // 			{
        // 				printf("%lf\t",xCurrent[i*strct_global_paras.nDim+j]);
        // 			}
        // 			printf("\n");
        // 			for(j=0;j<strct_global_paras.nObj;j++)
        // 			{
        // 				printf("%lf\t",fCurrent[i*strct_global_paras.nObj+j]);
        // 			}
        // 			printf("\n");
        // 		}

        st_repo_p.nRep = 0;
        int tmp_dest, tmp_tag;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(i % size) {
                tmp_dest = i % size;
                tmp_tag = i / size;
                MPI_Send(&st_grp_ana_p.var_current_grp[i * st_global_p.nDim], st_global_p.nDim, MPI_DOUBLE, tmp_dest,
                         tmp_tag, MPI_COMM_WORLD);
                MPI_Send(&st_grp_ana_p.obj_current_grp[i * st_global_p.nObj], st_global_p.nObj, MPI_DOUBLE, tmp_dest,
                         tmp_tag, MPI_COMM_WORLD);
            } else {
                memcpy(&st_grp_ana_p.var_repository_grp[st_repo_p.nRep * st_global_p.nDim],
                       &st_grp_ana_p.var_current_grp[i * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
                memcpy(&st_grp_ana_p.obj_repository_grp[st_repo_p.nRep * st_global_p.nObj],
                       &st_grp_ana_p.obj_current_grp[i * st_global_p.nObj],
                       st_global_p.nObj * sizeof(double));
                st_repo_p.nRep++;
            }
        }
    } else {
        st_repo_p.nRep = 0;
        for(i = 0; i < myPopNum; i++) {
            MPI_Recv(&st_grp_ana_p.var_repository_grp[i * st_global_p.nDim], st_global_p.nDim, MPI_DOUBLE, 0, i,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&st_grp_ana_p.obj_repository_grp[i * st_global_p.nObj], st_global_p.nObj, MPI_DOUBLE, 0, i,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // 	MPI_Bcast(xCurrent,strct_global_paras.nPop*strct_global_paras.nDim,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // 	MPI_Bcast(fCurrent,strct_global_paras.nPop*strct_global_paras.nObj,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // 	for(l=0;l<size;l++)
    // 	{
    // 		MPI_Barrier(MPI_COMM_WORLD);
    // 		if(rank==l)
    // 		{
    // 			int tag=1;
    // 			for(i=0;i<myPopNum;i++)
    // 			{
    // 				for(j=0;j<strct_global_paras.nDim;j++)
    // 				{
    // 					if(repository[i*strct_global_paras.nDim+j]!=xCurrent[(i*size+rank)*strct_global_paras.nDim+j])
    // 						tag=0;
    // 				}
    // 				for(j=0;j<strct_global_paras.nObj;j++)
    // 				{
    // 					if(repositFit[i*strct_global_paras.nObj+j]!=fCurrent[(i*size+rank)*strct_global_paras.nObj+j])
    // 						tag=0;
    // 				}
    // 			}
    // 			if(tag==0)
    // 			{
    // 				printf("RANK:%d\n",rank);
    // 				printf("TAG:%d\n",tag);
    // 			}
    // 		}
    // 		MPI_Barrier(MPI_COMM_WORLD);
    // 	}

    // task allocation
    task_sum = st_global_p.nDim * (st_global_p.nDim - 1) / 2;
    int* task_detail = (int*)calloc(task_sum * st_grp_ana_p.NumDependentAnalysis1 * (3 + st_global_p.nObj),
                                    sizeof(int));
    int myTaskNum = 0;
    int* task_tmp = (int*)calloc(3 + st_global_p.nObj, sizeof(int));
    if(rank == 0) {
        int SeleIndivIndex, tmp_dest, tmp_tag;
        for(i = 0; i < st_global_p.nDim; i++) {
            for(j = i + 1; j < st_global_p.nDim; j++) {
                for(k = 0; k < st_grp_ana_p.NumDependentAnalysis1; k++) {
                    SeleIndivIndex = int(st_global_p.nPop * pointer_gen_rand());
                    tmp_dest = SeleIndivIndex % size;
                    tmp_tag = SeleIndivIndex / size;
                    task_tmp[0] = tmp_tag;
                    task_tmp[1] = i;
                    task_tmp[2] = j;
                    if(tmp_dest) {
                        MPI_Send(task_tmp, (3), MPI_INT, tmp_dest, tmp_dest, MPI_COMM_WORLD);
                    } else {
                        memcpy(&task_detail[myTaskNum * (3 + st_global_p.nObj)], task_tmp, (3) * sizeof(int));
                        myTaskNum++;
                    }
                }
            }
        }

        // signal to end
        task_tmp[0] = -1;
        for(i = 1; i < size; i++) {
            MPI_Send(task_tmp, (3), MPI_INT, i, i, MPI_COMM_WORLD);
        }
    } else {
        do {
            MPI_Recv(task_tmp, (3), MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(task_tmp[0] != -1) {
                memcpy(&task_detail[myTaskNum * (3 + st_global_p.nObj)], task_tmp, (3 + st_global_p.nObj) * sizeof(int));
                myTaskNum++;
            }
        } while(task_tmp[0] != -1);
    }

    //
    // 	int summmm=0;
    // 	MPI_Reduce(&myTaskNum,&summmm,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    // 	if(summmm!=task_sum*strct_grp_ana_vals.NumDependentAnalysis1&&rank==0)
    // 		printf("ERROR\n");

    //
    double* x_a1_b1, * x_a2_b1, * x_a1_b2, * x_a2_b2;
    double* f_a1_b1, * f_a2_b1, * f_a1_b2, * f_a2_b2;
    x_a1_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a1_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    f_a1_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a1_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    double* interdependence_detail = (double*)calloc(task_sum * st_grp_ana_p.NumDependentAnalysis1 * st_global_p.nObj,
                                     sizeof(double));
    double* tmp_weight_min = (double*)calloc(st_global_p.nObj, sizeof(double)),
            * tmp_weight_max = (double*)calloc(st_global_p.nObj, sizeof(double));
    for(i = 0; i < st_global_p.nObj; i++) {
        tmp_weight_min[i] = 1e130;
        tmp_weight_max[i] = -1e130;
    }
    //double tmp_strct_grp_ana_vals.weight_thresh = 1e-6;
    for(i = 0; i < myTaskNum; i++) {
        int iPop, ii, jj;
        iPop = task_detail[i * (3 + st_global_p.nObj) + 0];
        ii = task_detail[i * (3 + st_global_p.nObj) + 1];
        jj = task_detail[i * (3 + st_global_p.nObj) + 2];

        memcpy(x_a1_b1, &st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        memcpy(f_a1_b1, &st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj],
               st_global_p.nObj * sizeof(double));
        memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
        memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
        memcpy(x_a2_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
        double a2 = rndreal(st_global_p.minLimit[ii], st_global_p.maxLimit[ii]);
        while(fabs(a2 - x_a1_b1[ii]) < (st_global_p.maxLimit[ii] - st_global_p.minLimit[ii]) / 10)
            a2 = rndreal(st_global_p.minLimit[ii], st_global_p.maxLimit[ii]);
        double b2 = rndreal(st_global_p.minLimit[jj], st_global_p.maxLimit[jj]);
        while(fabs(b2 - x_a1_b1[jj]) < (st_global_p.maxLimit[jj] - st_global_p.minLimit[jj]) / 10)   // i or j
            b2 = rndreal(st_global_p.minLimit[jj], st_global_p.maxLimit[jj]);
        x_a2_b1[ii] = a2;
        EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
        x_a1_b2[jj] = b2;
        EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
        x_a2_b2[ii] = a2;
        x_a2_b2[jj] = b2;
        EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
        for(l = 0; l < st_global_p.nObj; l++) {
            double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
            double temp2 = (f_a1_b2[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a2_b1[l]);
            /*			interdependence_detail[i*strct_global_paras.nObj+l]=(fabs(temp1) + fabs(temp2))/2.0;
            if((fabs(temp1) + fabs(temp2))/2.0<strct_grp_ana_vals.weight_min[l])
            strct_grp_ana_vals.weight_min[l]=(fabs(temp1) + fabs(temp2))/2.0;
            if((fabs(temp1) + fabs(temp2))/2.0>strct_grp_ana_vals.weight_max[l])
            strct_grp_ana_vals.weight_max[l]=(fabs(temp1) + fabs(temp2))/2.0;*/
            if(fabs(temp1) > 1e-6 || fabs(temp2) > 1e-6) {
                task_detail[i * (3 + st_global_p.nObj) + 3 + l] = 1;
            }  // variable x_i has a depentant relationship with x_j.
            //			if ( fabs(x_a2_b1[l]-f_a1_b1[l]) > 1e-6) task_detail[i*(3+strct_global_paras.nObj)+3+l]=1;
        }
        if((IsDistanceVariable(jj) || st_grp_ana_p.numDiverIndexes > 4)
           && UpdateSolution(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a1_b2)) {
            st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim + jj] = b2;
            memcpy(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a1_b2,
                   st_global_p.nObj * sizeof(double));
        }
        if((IsDistanceVariable(ii) || st_grp_ana_p.numDiverIndexes > 4)
           && UpdateSolution(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a2_b1)) {
            st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim + ii] = a2;
            st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim + jj] = x_a1_b1[jj];
            memcpy(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a2_b1,
                   st_global_p.nObj * sizeof(double));
        }
        if(((IsDistanceVariable(ii) && IsDistanceVariable(jj)) || st_grp_ana_p.numDiverIndexes > 4)
           && UpdateSolution(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a2_b2)) {
            st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim + ii] = a2;
            st_grp_ana_p.var_repository_grp[iPop * st_global_p.nDim + jj] = b2;
            memcpy(&st_grp_ana_p.obj_repository_grp[iPop * st_global_p.nObj], f_a2_b2,
                   st_global_p.nObj * sizeof(double));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);//if(rank==0)printf("CALCULATION DONE\n");

    /*	for(l=0;l<strct_global_paras.nObj;l++)
    {
    double tmp=strct_grp_ana_vals.weight_min[l];
    MPI_Allreduce(&tmp,&strct_grp_ana_vals.weight_min[l],1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    tmp=strct_grp_ana_vals.weight_max[l];
    MPI_Allreduce(&tmp,&strct_grp_ana_vals.weight_max[l],1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    }

    for(i=0;i<myTaskNum;i++)
    {
    for(l=0;l<strct_global_paras.nObj;l++)
    {
    if(strct_grp_ana_vals.weight_max[l]-strct_grp_ana_vals.weight_min[l]>1e-6)
    interdependence_detail[i*strct_global_paras.nObj+l]=
    (interdependence_detail[i*strct_global_paras.nObj+l]-strct_grp_ana_vals.weight_min[l])/(strct_grp_ana_vals.weight_max[l]-strct_grp_ana_vals.weight_min[l]);
    if(interdependence_detail[i*strct_global_paras.nObj+l]<1e-6)
    task_detail[i*(3+strct_global_paras.nObj)+3+l]=0;
    else
    task_detail[i*(3+strct_global_paras.nObj)+3+l]=1;
    }
    }*/

    //
    // 	if(rank==0)
    // 	{
    // 		for(l=0;l<strct_global_paras.nObj;l++)
    // 		{
    // 			printf("OBJ:%d\n",l+1);
    // 			for(i=0;i<strct_global_paras.nDim;i++)
    // 			{
    // 				printf("DIM:%d\n",i+1);
    // 				for(j=0;j<strct_global_paras.nDim;j++)
    // 				{
    // 					printf("%d\t",strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+j]);
    // 				}
    // 				printf("\n");
    // 			}
    // 			printf("\n");
    // 		}
    // 	}

    //
    if(rank == 0) {
        for(i = 0; i < myTaskNum; i++) {
            int ii, jj;
            ii = task_detail[i * (3 + st_global_p.nObj) + 1];
            jj = task_detail[i * (3 + st_global_p.nObj) + 2];

            for(l = 0; l < st_global_p.nObj; l++) {
                if(task_detail[i * (3 + st_global_p.nObj) + 3 + l]) {
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + ii * st_global_p.nDim + jj] =
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + jj * st_global_p.nDim + ii] = 1;
                }
            }
        }
    }
    if(rank) {
        MPI_Send(&myTaskNum, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
        if(myTaskNum) {
            MPI_Send(task_detail, myTaskNum * (3 + st_global_p.nObj), MPI_INT, 0, rank, MPI_COMM_WORLD);
        }
        // 		printf("RANK:%d\tSENT\n",rank);
    } else {
        for(i = 1; i < size; i++) {
            MPI_Recv(&myTaskNum, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(myTaskNum) {
                MPI_Recv(task_detail, myTaskNum * (3 + st_global_p.nObj), MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for(k = 0; k < myTaskNum; k++) {
                    int ii, jj;
                    ii = task_detail[k * (3 + st_global_p.nObj) + 1];
                    jj = task_detail[k * (3 + st_global_p.nObj) + 2];

                    for(l = 0; l < st_global_p.nObj; l++) {
                        if(task_detail[k * (3 + st_global_p.nObj) + 3 + l]) {
                            st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + ii * st_global_p.nDim + jj] =
                                st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + jj * st_global_p.nDim + ii] = 1;
                        }
                    }
                }
            }
            // 			printf("RANK:%d\tRECEIVED\n",i);
        }
    }

    //
    MPI_Bcast(st_grp_ana_p.Dependent, st_global_p.nObj * st_global_p.nDim * st_global_p.nDim, MPI_INT, 0,
              MPI_COMM_WORLD);

    //
    // 	if(rank==0)
    // 	{
    // 		for(l=0;l<strct_global_paras.nObj;l++)
    // 		{
    // 			printf("OBJ:%d\n",l+1);
    // 			for(i=0;i<strct_global_paras.nDim;i++)
    // 			{
    // 				printf("DIM:%d\n",i+1);
    // 				for(j=0;j<strct_global_paras.nDim;j++)
    // 				{
    // 					printf("%d\t",strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+j]);
    // 				}
    // 				printf("\n");
    // 			}
    // 			printf("\n");
    // 		}
    // 	}

    if(rank) {
        for(i = 0; i < myPopNum; i++) {
            MPI_Send(&st_grp_ana_p.var_repository_grp[i * st_global_p.nDim], st_global_p.nDim, MPI_DOUBLE, 0, i,
                     MPI_COMM_WORLD);
            MPI_Send(&st_grp_ana_p.obj_repository_grp[i * st_global_p.nObj], st_global_p.nObj, MPI_DOUBLE, 0, i,
                     MPI_COMM_WORLD);
        }
    } else {
        int tmp_src, tmp_tag;
        st_repo_p.nRep = 0;
        for(i = 0; i < st_global_p.nPop; i++) {
            if(i % size) {
                tmp_src = i % size;
                tmp_tag = i / size;
                MPI_Recv(&st_grp_ana_p.var_current_grp[i * st_global_p.nDim], st_global_p.nDim, MPI_DOUBLE, tmp_src,
                         tmp_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&st_grp_ana_p.obj_current_grp[i * st_global_p.nObj], st_global_p.nObj, MPI_DOUBLE, tmp_src,
                         tmp_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                memcpy(&st_grp_ana_p.var_current_grp[i * st_global_p.nDim],
                       &st_grp_ana_p.var_repository_grp[st_repo_p.nRep * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
                memcpy(&st_grp_ana_p.obj_current_grp[i * st_global_p.nObj],
                       &st_grp_ana_p.obj_repository_grp[st_repo_p.nRep * st_global_p.nObj],
                       st_global_p.nObj * sizeof(double));
                st_repo_p.nRep++;
            }
        }
    }

    //
    MPI_Bcast(st_grp_ana_p.var_current_grp, st_global_p.nPop * st_global_p.nDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(st_grp_ana_p.obj_current_grp, st_global_p.nPop * st_global_p.nObj, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //
    // 	if(rank==0)
    // 	{
    // 		for(i=0;i<strct_global_paras.nPop;i++)
    // 		{
    // 			printf("NO.%d\n",i+1);
    // 			for(j=0;j<strct_global_paras.nDim;j++)
    // 			{
    // 				printf("%lf\t",xCurrent[i*strct_global_paras.nDim+j]);
    // 			}
    // 			printf("\n");
    // 			for(j=0;j<strct_global_paras.nObj;j++)
    // 			{
    // 				printf("%lf\t",fCurrent[i*strct_global_paras.nObj+j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 	}

    free(recv);
    free(disp);
    free(task_detail);
    free(task_tmp);
    free(x_a1_b1);
    free(x_a1_b2);
    free(x_a2_b1);
    free(x_a2_b2);
    free(f_a1_b1);
    free(f_a1_b2);
    free(f_a2_b1);
    free(f_a2_b2);
    free(interdependence_detail);
    free(tmp_weight_min);
    free(tmp_weight_max);

    return;
}

void InterdependenceAnalysis()
{
    initializePopulation_grouping();
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) printf("INIT_DONE\n");

    //Analyze the strct_grp_ana_vals.Dependent relationship among the decision variables.
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int* recv = (int*)malloc(size * sizeof(int));
    int* disp = (int*)malloc(size * sizeof(int));
    int quo, rem;
    int task_sum = st_global_p.nDim * (st_global_p.nDim - 1) / 2;
    quo = task_sum / size;
    rem = task_sum % size;
    int i, j, l, k;

    for(i = 0; i < size; i++) {
        recv[i] = quo;
        if(i < rem)recv[i]++;
    }
    disp[0] = 0;
    for(i = 1; i < size; i++) {
        disp[i] = disp[i - 1] + recv[i - 1];
    }

    double* x_a1_b1, * x_a2_b1, * x_a1_b2, * x_a2_b2;
    double* f_a1_b1, * f_a2_b1, * f_a1_b2, * f_a2_b2;
    x_a1_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a1_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    f_a1_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a1_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    int i_s, j_s;
    int num_mine = recv[rank];
    int num_start = disp[rank];
    int count = 0;
    for(i = 0; i < st_global_p.nDim; i++) {
        for(j = i + 1; j < st_global_p.nDim; j++) {
            if(count == num_start)break;
            count++;
        }
        if(count == num_start)break;
    }
    i_s = i;
    j_s = j;
    count = 0;
    int* tmp = (int*)calloc(st_global_p.nObj * (num_mine + 1), sizeof(int));
    i = i_s;
    {
        for(j = j_s; j < st_global_p.nDim; j++) {
            if(count >= num_mine)break;
            for(k = 0; k < st_grp_ana_p.NumDependentAnalysis1; k++) {
                //generate another three solution ind(...a2,...,b1,...), ind(...a1,...,b2,...), ind(...a2,...,b2,...).
                int SeleIndivIndex = int(st_global_p.nPop * pointer_gen_rand());// strct_global_paras.nPop?????????
                memcpy(x_a1_b1, &st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
                memcpy(f_a1_b1, &st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj],
                       st_global_p.nObj * sizeof(double));
                memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a2_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                double a2 = rndreal(st_global_p.minLimit[i], st_global_p.maxLimit[i]);
                while(fabs(a2 - x_a1_b1[i]) < (st_global_p.maxLimit[i] - st_global_p.minLimit[i]) / 10)
                    a2 = rndreal(st_global_p.minLimit[i], st_global_p.maxLimit[i]);
                double b2 = rndreal(st_global_p.minLimit[j], st_global_p.maxLimit[j]);
                while(fabs(b2 - x_a1_b1[j]) < (st_global_p.maxLimit[j] - st_global_p.minLimit[j]) / 10)   // i or j
                    b2 = rndreal(st_global_p.minLimit[j], st_global_p.maxLimit[j]);
                x_a2_b1[i] = a2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
                x_a1_b2[j] = b2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
                x_a2_b2[i] = a2;
                x_a2_b2[j] = b2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
                for(l = 0; l < st_global_p.nObj; l++) {
                    double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
                    //double temp2 = (f_a1_b2[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a2_b1[l]);
                    //					if ( temp1 < 0 || temp2 < 0 ) tmp[l*num_mine+count]=1; // variable x_i has a depentant relationship with x_j.
                    if(fabs(temp1) > 1e-6) tmp[l * num_mine + count] = 1;
                    else tmp[l * num_mine + count] = 0;
                    //					printf("<%s~%04d~%04d~%04d~%lf~%lf~~~%d> ",strct_global_paras.testInstance,l,i,j,temp1,temp2,tmp[l*num_mine+count]);
                }
                if((IsDistanceVariable(j) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a1_b2)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = b2;
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a1_b2,
                           st_global_p.nObj * sizeof(double));
                }
                if((IsDistanceVariable(i) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b1)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + i] = a2;
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = x_a1_b1[j];
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b1,
                           st_global_p.nObj * sizeof(double));
                }
                if(((IsDistanceVariable(i) && IsDistanceVariable(j)) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b2)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + i] = a2;
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = b2;
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b2,
                           st_global_p.nObj * sizeof(double));
                }
            }
            count++;
        }
    }
    for(i = i_s + 1; i < st_global_p.nDim; i++) {
        if(count >= num_mine)break;
        for(j = i + 1; j < st_global_p.nDim; j++) {
            if(count >= num_mine)break;
            for(k = 0; k < st_grp_ana_p.NumDependentAnalysis1; k++) {
                //generate another three solution ind(...a2,...,b1,...), ind(...a1,...,b2,...), ind(...a2,...,b2,...).
                int SeleIndivIndex = int(st_global_p.nPop * pointer_gen_rand());//strct_global_paras.nPop?????????
                memcpy(x_a1_b1, &st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim],
                       st_global_p.nDim * sizeof(double));
                memcpy(f_a1_b1, &st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj],
                       st_global_p.nObj * sizeof(double));
                memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a2_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                double a2 = rndreal(st_global_p.minLimit[i], st_global_p.maxLimit[i]);
                while(fabs(a2 - x_a1_b1[i]) < (st_global_p.maxLimit[i] - st_global_p.minLimit[i]) / 10)
                    a2 = rndreal(st_global_p.minLimit[i], st_global_p.maxLimit[i]);
                double b2 = rndreal(st_global_p.minLimit[j], st_global_p.maxLimit[j]);
                while(fabs(b2 - x_a1_b1[j]) < (st_global_p.maxLimit[j] - st_global_p.minLimit[j]) / 10)   // i or j
                    b2 = rndreal(st_global_p.minLimit[j], st_global_p.maxLimit[j]);
                x_a2_b1[i] = a2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
                x_a1_b2[j] = b2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
                x_a2_b2[i] = a2;
                x_a2_b2[j] = b2;
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
                for(l = 0; l < st_global_p.nObj; l++) {
                    double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
                    //double temp2 = (f_a1_b2[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a2_b1[l]);
                    //					if ( temp1 < 0 || temp2 < 0 ) tmp[l*num_mine+count]=1; // variable x_i has a depentant relationship with x_j.
                    if(fabs(temp1) > 1e-6) tmp[l * num_mine + count] = 1;
                    else tmp[l * num_mine + count] = 0;
                    //					printf("<%s~%04d~%04d~%04d~%lf~%lf~~~%d> ",strct_global_paras.testInstance,l,i,j,temp1,temp2,tmp[l*num_mine+count]);
                }
                if((IsDistanceVariable(j) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a1_b2)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = b2;
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a1_b2,
                           st_global_p.nObj * sizeof(double));
                }
                if((IsDistanceVariable(i) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b1)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + i] = a2;
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = x_a1_b1[j];
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b1,
                           st_global_p.nObj * sizeof(double));
                }
                if(((IsDistanceVariable(i) && IsDistanceVariable(j)) || st_grp_ana_p.numDiverIndexes > 4)
                   && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b2)) {
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + i] = a2;
                    st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + j] = b2;
                    memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_a2_b2,
                           st_global_p.nObj * sizeof(double));
                }
            }
            count++;
        }
    }
    int* tmp_all = (int*)calloc(st_global_p.nObj * task_sum, sizeof(int));
    //all gather
    for(l = 0; l < st_global_p.nObj; l++) {
        MPI_Allgatherv(&tmp[l * num_mine], num_mine, MPI_INT,
                       &tmp_all[l * task_sum], recv, disp, MPI_INT,
                       MPI_COMM_WORLD);
    }

    count = 0;
    for(i = 0; i < st_global_p.nDim; i++) {
        for(j = i + 1; j < st_global_p.nDim; j++) {
            for(l = 0; l < st_global_p.nObj; l++) {
                st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j] =
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim + i] =
                        tmp_all[l * task_sum + count];
            }
            count++;
        }
        for(l = 0; l < st_global_p.nObj; l++)
            st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + i] = 1;
    }

    free(recv);
    free(disp);
    free(tmp);
    free(tmp_all);
    free(x_a1_b1);
    free(x_a1_b2);
    free(x_a2_b1);
    free(x_a2_b2);
    free(f_a1_b1);
    free(f_a1_b2);
    free(f_a2_b1);
    free(f_a2_b2);
}

void InterdependenceAnalysis_gDG2_serial()
{
    //    printf("%lf\n",strct_grp_ana_vals.weight_thresh);

    //Analyze the strct_grp_ana_vals.Dependent relationship among the decision variables.
    int i, j, l, k;

    double* x_a1_b1, * x_a2_b1, * x_a1_b2, * x_a2_b2;
    double* f_a1_b1, * f_a2_b1, * f_a1_b2, * f_a2_b2;
    x_a1_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a1_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    f_a1_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a1_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));

    for(i = 0; i < st_global_p.nObj; i++) {
        st_grp_ana_p.weight_min[i] = 1e130;
        st_grp_ana_p.weight_max[i] = -1e130;
    }
    for(i = 0; i < st_global_p.nDim; i++) {
        if(IsDiversityVariable(i)) continue;   //the corresponding weights will be 0
        for(k = 0; k < st_global_p.nDim;
            k++) x_a1_b1[k] = st_global_p.minLimit[k] + (st_global_p.maxLimit[k] - st_global_p.minLimit[k]) /
                                  st_grp_ana_p.div_ratio;
        EMO_evaluate_problems(st_global_p.testInstance, x_a1_b1, f_a1_b1, st_global_p.nDim, 1, st_global_p.nObj);
        memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
        x_a2_b1[i] = st_global_p.maxLimit[i] - (st_global_p.maxLimit[i] - st_global_p.minLimit[i]) /
                     st_grp_ana_p.div_ratio;
        EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
        for(j = i + 1; j < st_global_p.nDim; j++) {
            if(IsDiversityVariable(j)) continue;   //the corresponding weights will be 0
            {
                //generate another three solution ind(...a2,...,b1,...), ind(...a1,...,b2,...), ind(...a2,...,b2,...).
                memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a2_b2, x_a2_b1, st_global_p.nDim * sizeof(double));
                x_a1_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);;
                EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
                x_a2_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);;
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
                for(l = 0; l < st_global_p.nObj; l++) {
                    double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
                    st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                          j] =
                                                            st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim +
                                                                      i] =
                                                                            fabs(temp1);//only for strct_grp_ana_vals.NumDependentAnalysis1=1
                    if(fabs(temp1) < st_grp_ana_p.weight_min[l])
                        st_grp_ana_p.weight_min[l] = fabs(temp1);
                    if(fabs(temp1) > st_grp_ana_p.weight_max[l])
                        st_grp_ana_p.weight_max[l] = fabs(temp1);
                }
            }
        }
    }

    for(l = 0; l < st_global_p.nObj; l++) {
        for(i = 0; i < st_global_p.nDim; i++) {
            for(j = i + 1; j < st_global_p.nDim; j++) {
                if(st_grp_ana_p.weight_max[l] - st_grp_ana_p.weight_min[l] > st_grp_ana_p.weight_thresh)
                    st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                          j] =
                                                            st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim +
                                                                      i] =
                                                                            (st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                                                    j] - st_grp_ana_p.weight_min[l]) /
                                                                            (st_grp_ana_p.weight_max[l] - st_grp_ana_p.weight_min[l]);
                if(st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim
                                                         + j] < st_grp_ana_p.weight_thresh)
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j] =
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim + i] = 0;
                else
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j] =
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim + i] = 1;
            }
            st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + i] = 1;
            st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                  i] = 1;
        }
    }

    /*	MPI_Bcast(strct_grp_ana_vals.Interdependence_Weight,strct_global_paras.nObj*strct_global_paras.nDim*strct_global_paras.nDim,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(strct_grp_ana_vals.Dependent,strct_global_paras.nObj*strct_global_paras.nDim*strct_global_paras.nDim,MPI_INT,0,MPI_COMM_WORLD);*/

    free(x_a1_b1);
    free(x_a1_b2);
    free(x_a2_b1);
    free(x_a2_b2);
    free(f_a1_b1);
    free(f_a1_b2);
    free(f_a2_b1);
    free(f_a2_b2);
}

void InterdependenceAnalysis_gDG2()
{
    //Analyze the strct_grp_ana_vals.Dependent relationship among the decision variables.
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int* recv = (int*)malloc(size * sizeof(int));
    int* disp = (int*)malloc(size * sizeof(int));
    int quo, rem;
    int task_sum = st_global_p.nDim * (st_global_p.nDim - 1) / 2;
    quo = task_sum / size;
    rem = task_sum % size;
    int i, j, l, k;

    for(i = 0; i < size; i++) {
        recv[i] = quo;
        if(i < rem)recv[i]++;
    }
    disp[0] = 0;
    for(i = 1; i < size; i++) {
        disp[i] = disp[i - 1] + recv[i - 1];
    }

    double* x_a1_b1, * x_a2_b1, * x_a1_b2, * x_a2_b2;
    double* f_a1_b1, * f_a2_b1, * f_a1_b2, * f_a2_b2;
    x_a1_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b1 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a1_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    x_a2_b2 = (double*)malloc(st_global_p.nDim * sizeof(double));
    f_a1_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b1 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a1_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    f_a2_b2 = (double*)malloc(st_global_p.nObj * sizeof(double));
    int i_s, j_s;
    int num_mine = recv[rank];
    int num_start = disp[rank];
    int count = 0;
    //for(i = 0; i < st_global_p.nDim; i++) {
    //    for(j = i + 1; j < st_global_p.nDim; j++) {
    //        if(count == num_start)break;
    //        count++;
    //    }
    //    if(count == num_start)break;
    //}
    for(i = 0; i < st_global_p.nDim; i++) {
        for(j = i + 1; j < st_global_p.nDim; j++) {
            count++;
            if(count - 1 == num_start) break;
        }
        if(count - 1 == num_start) break;
    }
    i_s = i;
    j_s = j;
    count = 0;
    double* interdependence_detail = (double*)calloc(st_global_p.nObj * (num_mine + 1), sizeof(double));
    for(i = 0; i < st_global_p.nObj; i++) {
        st_grp_ana_p.weight_min[i] = 1e130;
        st_grp_ana_p.weight_max[i] = -1e130;
    }
    i = i_s;
    {
        //    	if(!IsDiversityVariable(i))
        {
            for(k = 0; k < st_global_p.nDim;
                k++) x_a1_b1[k] = st_global_p.minLimit[k] + (st_global_p.maxLimit[k] - st_global_p.minLimit[k]) /
                                      st_grp_ana_p.div_ratio;
            EMO_evaluate_problems(st_global_p.testInstance, x_a1_b1, f_a1_b1, st_global_p.nDim, 1, st_global_p.nObj);
            memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
            x_a2_b1[i] = st_global_p.maxLimit[i] - (st_global_p.maxLimit[i] - st_global_p.minLimit[i]) /
                         st_grp_ana_p.div_ratio;
            EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
        }
        for(j = j_s; j < st_global_p.nDim; j++) {
            if(count >= num_mine)break;
            //            if(!IsDiversityVariable(i) && !IsDiversityVariable(j))
            {
                //generate another three solution ind(...a2,...,b1,...), ind(...a1,...,b2,...), ind(...a2,...,b2,...).
                memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a2_b2, x_a2_b1, st_global_p.nDim * sizeof(double));
                x_a1_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);
                EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
                x_a2_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
                for(l = 0; l < st_global_p.nObj; l++) {
                    double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
                    interdependence_detail[l * num_mine + count] = fabs(temp1); //only for strct_grp_ana_vals.NumDependentAnalysis1=1
                    if(fabs(temp1) < st_grp_ana_p.weight_min[l])
                        st_grp_ana_p.weight_min[l] = fabs(temp1);
                    if(fabs(temp1) > st_grp_ana_p.weight_max[l])
                        st_grp_ana_p.weight_max[l] = fabs(temp1);
                }
            }
            count++;
        }
    }
    for(i = i_s + 1; i < st_global_p.nDim; i++) {
        if(count >= num_mine)break;
        //        if(!IsDiversityVariable(i))
        {
            for(k = 0; k < st_global_p.nDim;
                k++) x_a1_b1[k] = st_global_p.minLimit[k] + (st_global_p.maxLimit[k] - st_global_p.minLimit[k]) /
                                      st_grp_ana_p.div_ratio;
            EMO_evaluate_problems(st_global_p.testInstance, x_a1_b1, f_a1_b1, st_global_p.nDim, 1, st_global_p.nObj);
            memcpy(x_a2_b1, x_a1_b1, st_global_p.nDim * sizeof(double));
            x_a2_b1[i] = st_global_p.maxLimit[i] - (st_global_p.maxLimit[i] - st_global_p.minLimit[i]) /
                         st_grp_ana_p.div_ratio;
            EMO_evaluate_problems(st_global_p.testInstance, x_a2_b1, f_a2_b1, st_global_p.nDim, 1, st_global_p.nObj);
        }
        for(j = i + 1; j < st_global_p.nDim; j++) {
            if(count >= num_mine)break;
            //            if(!IsDiversityVariable(i) && !IsDiversityVariable(j))
            {
                //generate another three solution ind(...a2,...,b1,...), ind(...a1,...,b2,...), ind(...a2,...,b2,...).
                memcpy(x_a1_b2, x_a1_b1, st_global_p.nDim * sizeof(double));
                memcpy(x_a2_b2, x_a2_b1, st_global_p.nDim * sizeof(double));
                x_a1_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);
                EMO_evaluate_problems(st_global_p.testInstance, x_a1_b2, f_a1_b2, st_global_p.nDim, 1, st_global_p.nObj);
                x_a2_b2[j] = 0.5 * (st_global_p.minLimit[j] + st_global_p.maxLimit[j]);
                EMO_evaluate_problems(st_global_p.testInstance, x_a2_b2, f_a2_b2, st_global_p.nDim, 1, st_global_p.nObj);
                for(l = 0; l < st_global_p.nObj; l++) {
                    double temp1 = (f_a2_b1[l] - f_a1_b1[l]) - (f_a2_b2[l] - f_a1_b2[l]);
                    interdependence_detail[l * num_mine + count] = fabs(temp1); //only for strct_grp_ana_vals.NumDependentAnalysis1=1
                    if(fabs(temp1) < st_grp_ana_p.weight_min[l])
                        st_grp_ana_p.weight_min[l] = fabs(temp1);
                    if(fabs(temp1) > st_grp_ana_p.weight_max[l])
                        st_grp_ana_p.weight_max[l] = fabs(temp1);
                }
            }
            count++;
        }
    }
    double* interdependence_all = (double*)calloc(st_global_p.nObj * task_sum, sizeof(double));
    //all gather
    for(l = 0; l < st_global_p.nObj; l++) {
        MPI_Allgatherv(&interdependence_detail[l * num_mine], num_mine, MPI_DOUBLE,
                       &interdependence_all[l * task_sum], recv, disp, MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    for(l = 0; l < st_global_p.nObj; l++) {
        double tmpw = st_grp_ana_p.weight_min[l];
        MPI_Allreduce(&tmpw, &st_grp_ana_p.weight_min[l], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        tmpw = st_grp_ana_p.weight_max[l];
        MPI_Allreduce(&tmpw, &st_grp_ana_p.weight_max[l], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    for(l = 0; l < st_global_p.nObj; l++) {
        for(i = 0; i < st_global_p.nDim; i++) {
            for(j = 0; j < st_global_p.nDim; j++) {
                st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                      j] = -1e10;
            }
            st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                  i] = 0.0;
        }
    }

    count = 0;
    double temp;
    /*    for(i=0;i<strct_global_paras.nDim;i++){
    for(j=i+1;j<strct_global_paras.nDim;j++){
    for(l=0;l<strct_global_paras.nObj;l++){
    if(strct_grp_ana_vals.weight_max[l]-strct_grp_ana_vals.weight_min[l]>strct_grp_ana_vals.weight_thresh)
    temp=
    (interdependence_all[l*task_sum+count]-strct_grp_ana_vals.weight_min[l])/(strct_grp_ana_vals.weight_max[l]-strct_grp_ana_vals.weight_min[l]);
    else
    temp=interdependence_all[l*task_sum+count];
    if(temp<strct_grp_ana_vals.weight_thresh)
    strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+j]=
    strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+j*strct_global_paras.nDim+i]=0;
    else
    strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+j]=
    strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+j*strct_global_paras.nDim+i]=1;
    strct_grp_ana_vals.Interdependence_Weight[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+j]=
    strct_grp_ana_vals.Interdependence_Weight[l*strct_global_paras.nDim*strct_global_paras.nDim+j*strct_global_paras.nDim+i]=
    interdependence_all[l*task_sum+count];
    }
    count++;
    }
    for(l=0;l<strct_global_paras.nObj;l++)
    {
    strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+i]=0;
    strct_grp_ana_vals.Interdependence_Weight[l*strct_global_paras.nDim*strct_global_paras.nDim+i*strct_global_paras.nDim+i]=0;
    }
    }*/
    for(i = 0; i < st_global_p.nDim; i++) {
        for(j = i + 1; j < st_global_p.nDim; j++) {
            for(l = 0; l < st_global_p.nObj; l++) {
                temp = interdependence_all[l * task_sum + count];
                if(temp > 0.0)
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j] =
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim + i] = 1;
                else
                    st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j] =
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim + i] = 0;
                st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                      j] =
                                                        st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + j * st_global_p.nDim +
                                                                  i] =
                                                                        interdependence_all[l * task_sum + count];
            }
            count++;
        }
        for(l = 0; l < st_global_p.nObj; l++) {
            st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + i] = 0;
            st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                                  i] = 0;
        }
    }

    free(recv);
    free(disp);
    free(x_a1_b1);
    free(x_a1_b2);
    free(x_a2_b1);
    free(x_a2_b2);
    free(f_a1_b1);
    free(f_a1_b2);
    free(f_a2_b1);
    free(f_a2_b2);
    free(interdependence_detail);
    free(interdependence_all);
}

void DependentVarAnalysis()
{
    // Analyze whether a decision variable is strct_grp_ana_vals.Effective for individual function.
    int i, j, l;
    for(l = 0; l < st_global_p.nObj; l++)
        for(i = 0; i < st_global_p.nDim; i++) {
            st_grp_ana_p.Effect[l * st_global_p.nDim + i] = 0;
        }
    double* x_RandIndiv = (double*)malloc(st_global_p.nDim * sizeof(double));
    double* f_RandIndiv = (double*)malloc(st_global_p.nObj * sizeof(double));
    for(i = 0; i < st_global_p.nDim; i++) {
        int SeleIndivIndex = int(st_global_p.nPop * pointer_gen_rand());
        memcpy(x_RandIndiv, &st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim],
               st_global_p.nDim * sizeof(double));
        //evaluate_problems(strct_global_paras.testInstance, x_RandIndiv, f_RandIndiv, strct_global_paras.nDim, 1, strct_global_paras.nObj);
        for(j = 0; j < st_grp_ana_p.NumDependentAnalysis; j++) {
            x_RandIndiv[i] = (j + pointer_gen_rand()) / st_grp_ana_p.NumDependentAnalysis;
            EMO_evaluate_problems(st_global_p.testInstance, x_RandIndiv, f_RandIndiv, st_global_p.nDim, 1,
                                  st_global_p.nObj);
            for(l = 0; l < st_global_p.nObj; l++) {
                if(fabs(st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj + l] - f_RandIndiv[l]) > 1e-20)
                    st_grp_ana_p.Effect[l *
                                          st_global_p.nDim + i] = 1;
            }
            if(IsDistanceVariable(i)
               && UpdateSolution(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_RandIndiv)) {
                st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim + i] = x_RandIndiv[i];
                memcpy(&st_grp_ana_p.obj_current_grp[SeleIndivIndex * st_global_p.nObj], f_RandIndiv,
                       st_global_p.nObj * sizeof(double));
                memcpy(&st_grp_ana_p.var_current_grp[SeleIndivIndex * st_global_p.nDim], x_RandIndiv,
                       st_global_p.nDim * sizeof(double));
            }
        }
    }
    free(x_RandIndiv);
    free(f_RandIndiv);
}