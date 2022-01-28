#include "output_csv.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void output_results_to_csv_files(int the_rank, int nExp, int iRun, int iTrace, int tag_prob,
                                 char* _algName, char* _probName, int numObj, int numVar, long curTime,
                                 double* IGDs_TRAIN, double* IGDs_TEST, double* IGDs_all,
                                 double* HVs_TRAIN, double* HVs_TEST, double* HVs_all,
                                 double TIMEs_all)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if(the_rank) {
        MPI_Send(IGDs_TRAIN, iTrace + 2, MPI_DOUBLE, 0, 10000 + the_rank, MPI_COMM_WORLD);
        if(tag_prob) {
            MPI_Send(IGDs_TEST, iTrace + 2, MPI_DOUBLE, 0, 20000 + the_rank, MPI_COMM_WORLD);
            MPI_Send(IGDs_all, iTrace + 2, MPI_DOUBLE, 0, 30000 + the_rank, MPI_COMM_WORLD);
        }
        MPI_Send(HVs_TRAIN, iTrace + 2, MPI_DOUBLE, 0, 1000 + the_rank, MPI_COMM_WORLD);
        if(tag_prob) {
            MPI_Send(HVs_TEST, iTrace + 2, MPI_DOUBLE, 0, 2000 + the_rank, MPI_COMM_WORLD);
            MPI_Send(HVs_all, iTrace + 2, MPI_DOUBLE, 0, 3000 + the_rank, MPI_COMM_WORLD);
        }
        MPI_Send(&TIMEs_all, 1, MPI_DOUBLE, 0, 4000 + the_rank, MPI_COMM_WORLD);
    } else {
        char tmpName[1024];
        FILE* tmpFile;
        double*** ALL_IGD_TRAIN;// [nExpNum][NRUN][NTRACE + 2];
        double*** ALL_IGD_TEST;// [nExpNum][NRUN][NTRACE + 2];
        double*** ALL_IGD;// [nExpNum][NRUN][NTRACE + 2];
        double*** ALL_HV_TRAIN;// [nExpNum][NRUN][NTRACE + 2];
        double*** ALL_HV_TEST;// [nExpNum][NRUN][NTRACE + 2];
        double*** ALL_HV;// [nExpNum][NRUN][NTRACE + 2];
        double**  ALL_TIME;// [nExpNum][NRUN];
        double*   tmp;// [NTRACE + 2];
        //
        ALL_IGD_TRAIN = (double***)malloc(nExp * sizeof(double**));
        ALL_IGD_TEST = (double***)malloc(nExp * sizeof(double**));
        ALL_IGD = (double***)malloc(nExp * sizeof(double**));
        ALL_HV_TRAIN = (double***)malloc(nExp * sizeof(double**));
        ALL_HV_TEST = (double***)malloc(nExp * sizeof(double**));
        ALL_HV = (double***)malloc(nExp * sizeof(double**));
        ALL_TIME = (double**)malloc(nExp * sizeof(double*));
        tmp = (double*)malloc((iTrace + 2) * sizeof(double));
        for(int i = 0; i < nExp; i++) {
            ALL_IGD_TRAIN[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_IGD_TEST[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_IGD[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_HV_TRAIN[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_HV_TEST[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_HV[i] = (double**)malloc(iRun * sizeof(double*));
            ALL_TIME[i] = (double*)malloc(iRun * sizeof(double));
            for(int j = 0; j < iRun; j++) {
                ALL_IGD_TRAIN[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
                ALL_IGD_TEST[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
                ALL_IGD[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
                ALL_HV_TRAIN[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
                ALL_HV_TEST[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
                ALL_HV[i][j] = (double*)malloc((iTrace + 2) * sizeof(double));
            }
        }
        //
        double** IGD_TRAIN_mean;// [nExpNum][NTRACE + 2];
        double** IGD_TRAIN_std;// [nExpNum][NTRACE + 2];
        double** IGD_TEST_mean;// [nExpNum][NTRACE + 2];
        double** IGD_TEST_std;// [nExpNum][NTRACE + 2];
        double** IGD_mean;// [nExpNum][NTRACE + 2];
        double** IGD_std;// [nExpNum][NTRACE + 2];
        double** HV_TRAIN_mean;// [nExpNum][NTRACE + 2];
        double** HV_TRAIN_std;// [nExpNum][NTRACE + 2];
        double** HV_TEST_mean;// [nExpNum][NTRACE + 2];
        double** HV_TEST_std;// [nExpNum][NTRACE + 2];
        double** HV_mean;// [nExpNum][NTRACE + 2];
        double** HV_std;// [nExpNum][NTRACE + 2];
        IGD_TRAIN_mean = (double**)malloc(nExp * sizeof(double*));
        IGD_TRAIN_std = (double**)malloc(nExp * sizeof(double*));
        IGD_TEST_mean = (double**)malloc(nExp * sizeof(double*));
        IGD_TEST_std = (double**)malloc(nExp * sizeof(double*));
        IGD_mean = (double**)malloc(nExp * sizeof(double*));
        IGD_std = (double**)malloc(nExp * sizeof(double*));
        HV_TRAIN_mean = (double**)malloc(nExp * sizeof(double*));
        HV_TRAIN_std = (double**)malloc(nExp * sizeof(double*));
        HV_TEST_mean = (double**)malloc(nExp * sizeof(double*));
        HV_TEST_std = (double**)malloc(nExp * sizeof(double*));
        HV_mean = (double**)malloc(nExp * sizeof(double*));
        HV_std = (double**)malloc(nExp * sizeof(double*));
        for(int i = 0; i < nExp; i++) {
            IGD_TRAIN_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            IGD_TRAIN_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            IGD_TEST_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            IGD_TEST_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            IGD_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            IGD_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_TRAIN_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_TRAIN_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_TEST_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_TEST_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_mean[i] = (double*)malloc((iTrace + 2) * sizeof(double));
            HV_std[i] = (double*)malloc((iTrace + 2) * sizeof(double));
        }
        //
        for(int i = 0; i < iTrace + 2; i++) {
            ALL_IGD_TRAIN[0][0][i] = IGDs_TRAIN[i];
            if(tag_prob) {
                ALL_IGD_TEST[0][0][i] = IGDs_TEST[i];
                ALL_IGD[0][0][i] = IGDs_all[i];
            }
            ALL_HV_TRAIN[0][0][i] = HVs_TRAIN[i];
            if(tag_prob) {
                ALL_HV_TEST[0][0][i] = HVs_TEST[i];
                ALL_HV[0][0][i] = HVs_all[i];
            }
        }
        ALL_TIME[0][0] = TIMEs_all;
        for(int lll = 0; lll < nExp; lll++) {
            for(int rrr = 0; rrr < iRun; rrr++) {
                if(!lll && !rrr) continue;
                int tmpRank = lll * iRun + rrr;
                //
                MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 10000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(&ALL_IGD_TRAIN[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                if(tag_prob) {
                    MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 20000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    memcpy(&ALL_IGD_TEST[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                    MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 30000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    memcpy(&ALL_IGD[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                }
                //
                MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 1000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(&ALL_HV_TRAIN[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                if(tag_prob) {
                    MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 2000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    memcpy(&ALL_HV_TEST[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                    MPI_Recv(tmp, iTrace + 2, MPI_DOUBLE, tmpRank, 3000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    memcpy(&ALL_HV[lll][rrr][0], tmp, (iTrace + 2) * sizeof(double));
                }
                //
                MPI_Recv(tmp, 1, MPI_DOUBLE, tmpRank, 4000 + tmpRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ALL_TIME[lll][rrr] = tmp[0];
            }
        }
        //////////////////////////////////////////////////////////////////////////
        //IGD_TRAIN
        sprintf(tmpName, "OUTPUT/IGD_TRAIN_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj, numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_IGD_TRAIN[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_IGD_TRAIN[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //IGD_TRAIN mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                IGD_TRAIN_mean[lll][i] = 0.0;
                IGD_TRAIN_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_TRAIN_mean[lll][j] += ALL_IGD_TRAIN[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_TRAIN_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_TRAIN_std[lll][j] += (ALL_IGD_TRAIN[lll][i - 1][j] - IGD_TRAIN_mean[lll][j]) *
                                             (ALL_IGD_TRAIN[lll][i - 1][j] - IGD_TRAIN_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_TRAIN_std[lll][j] = sqrt(IGD_TRAIN_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_IGD_TRAIN_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_TRAIN_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_IGD_TRAIN_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_TRAIN_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //IGD_TEST
        sprintf(tmpName, "OUTPUT/IGD_TEST_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj, numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_IGD_TEST[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_IGD_TEST[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //IGD_TEST mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                IGD_TEST_mean[lll][i] = 0.0;
                IGD_TEST_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_TEST_mean[lll][j] += ALL_IGD_TEST[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_TEST_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_TEST_std[lll][j] += (ALL_IGD_TEST[lll][i - 1][j] - IGD_TEST_mean[lll][j]) *
                                            (ALL_IGD_TEST[lll][i - 1][j] - IGD_TEST_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_TEST_std[lll][j] = sqrt(IGD_TEST_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_IGD_TEST_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_TEST_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_IGD_TEST_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_TEST_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //IGD
        sprintf(tmpName, "OUTPUT/IGD_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj, numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_IGD[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_IGD[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //IGD mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                IGD_mean[lll][i] = 0.0;
                IGD_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_mean[lll][j] += ALL_IGD[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    IGD_std[lll][j] += (ALL_IGD[lll][i - 1][j] - IGD_mean[lll][j]) *
                                       (ALL_IGD[lll][i - 1][j] - IGD_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                IGD_std[lll][j] = sqrt(IGD_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_IGD_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_IGD_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", IGD_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //HV_TRAIN
        sprintf(tmpName, "OUTPUT/HV_TRAIN_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj,
                numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_HV_TRAIN[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_HV_TRAIN[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //HV_TRAIN mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                HV_TRAIN_mean[lll][i] = 0.0;
                HV_TRAIN_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_TRAIN_mean[lll][j] += ALL_HV_TRAIN[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_TRAIN_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_TRAIN_std[lll][j] += (ALL_HV_TRAIN[lll][i - 1][j] - HV_TRAIN_mean[lll][j]) *
                                            (ALL_HV_TRAIN[lll][i - 1][j] - HV_TRAIN_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_TRAIN_std[lll][j] = sqrt(HV_TRAIN_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_HV_TRAIN_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_TRAIN_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_HV_TRAIN_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_TRAIN_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //HV_TEST
        sprintf(tmpName, "OUTPUT/HV_TEST_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj, numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_HV_TEST[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_HV_TEST[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //HV_TEST mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                HV_TEST_mean[lll][i] = 0.0;
                HV_TEST_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_TEST_mean[lll][j] += ALL_HV_TEST[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_TEST_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_TEST_std[lll][j] += (ALL_HV_TEST[lll][i - 1][j] - HV_TEST_mean[lll][j]) *
                                           (ALL_HV_TEST[lll][i - 1][j] - HV_TEST_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_TEST_std[lll][j] = sqrt(HV_TEST_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_HV_TEST_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_TEST_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_HV_TEST_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_TEST_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //HV
        sprintf(tmpName, "OUTPUT/HV_%s_%s_OBJ%d_VAR%d_%ld.csv", _algName, _probName, numObj, numVar, curTime);
        tmpFile = fopen(tmpName, "w");
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i <= iRun; i++) {
                if(i)
                    fprintf(tmpFile, "EXP%d_RUN%d", lll, i);
                for(int j = 0; j <= iTrace; j++) {
                    if(i == 0)
                        fprintf(tmpFile, ",TRACE%d", j);
                    else
                        fprintf(tmpFile, ",%.16e", ALL_HV[lll][i - 1][j]);
                }
                if(i == 0)
                    fprintf(tmpFile, ",FINAL\n");
                else
                    fprintf(tmpFile, ",%.16e\n", ALL_HV[lll][i - 1][iTrace + 1]);
            }
        }
        fclose(tmpFile);
        //HV mean & std
        for(int lll = 0; lll < nExp; lll++) {
            for(int i = 0; i < iTrace + 2; i++) {
                HV_mean[lll][i] = 0.0;
                HV_std[lll][i] = 0.0;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_mean[lll][j] += ALL_HV[lll][i - 1][j];
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_mean[lll][j] /= iRun;
            }
            for(int i = 1; i <= iRun; i++) {
                for(int j = 0; j <= iTrace + 1; j++) {
                    HV_std[lll][j] += (ALL_HV[lll][i - 1][j] - HV_mean[lll][j]) *
                                      (ALL_HV[lll][i - 1][j] - HV_mean[lll][j]);
                }
            }
            for(int j = 0; j <= iTrace + 1; j++) {
                HV_std[lll][j] = sqrt(HV_std[lll][j] / iRun);
            }
        }
        sprintf(tmpName, "OUTPUT/MEAN_HV_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_mean[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        sprintf(tmpName, "OUTPUT/STD_HV_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            for(int j = 0; j <= iTrace + 1; j++) {
                fprintf(tmpFile, ",%.16e", HV_std[lll][j]);
            }
            fprintf(tmpFile, "\n");
        }
        fclose(tmpFile);
        //////////////////////////////////////////////////////////////////////////
        //time
        double time_mean;
        sprintf(tmpName, "OUTPUT/TIME_%s_%ld.csv", _algName, curTime);
        tmpFile = fopen(tmpName, "a");
        for(int lll = 0; lll < nExp; lll++) {
            fprintf(tmpFile, "EXP%d_%s_OBJ%d_VAR%d", lll, _probName, numObj, numVar);
            time_mean = 0.0;
            for(int i = 1; i <= iRun; i++) {
                fprintf(tmpFile, ",%.16e", ALL_TIME[lll][i - 1]);
                time_mean += ALL_TIME[lll][i - 1];
            }
            time_mean /= iRun;
            fprintf(tmpFile, ",%.16e\n", time_mean);
        }
        fclose(tmpFile);
        //
        for(int i = 0; i < nExp; i++) {
            for(int j = 0; j < iRun; j++) {
                free(ALL_IGD_TRAIN[i][j]);
                free(ALL_IGD_TEST[i][j]);
                free(ALL_IGD[i][j]);
                free(ALL_HV_TRAIN[i][j]);
                free(ALL_HV_TEST[i][j]);
                free(ALL_HV[i][j]);
            }
            free(ALL_IGD_TRAIN[i]);
            free(ALL_IGD_TEST[i]);
            free(ALL_IGD[i]);
            free(ALL_HV_TRAIN[i]);
            free(ALL_HV_TEST[i]);
            free(ALL_HV[i]);
            free(ALL_TIME[i]);
        }
        free(ALL_IGD_TRAIN);
        free(ALL_IGD_TEST);
        free(ALL_IGD);
        free(ALL_HV_TRAIN);
        free(ALL_HV_TEST);
        free(ALL_HV);
        free(ALL_TIME);
        free(tmp);
        //
        for(int i = 0; i < nExp; i++) {
            free(IGD_TRAIN_mean[i]);
            free(IGD_TRAIN_std[i]);
            free(IGD_TEST_mean[i]);
            free(IGD_TEST_std[i]);
            free(IGD_mean[i]);
            free(IGD_std[i]);
            free(HV_TRAIN_mean[i]);
            free(HV_TRAIN_std[i]);
            free(HV_TEST_mean[i]);
            free(HV_TEST_std[i]);
            free(HV_mean[i]);
            free(HV_std[i]);
        }
        free(IGD_TRAIN_mean);
        free(IGD_TRAIN_std);
        free(IGD_TEST_mean);
        free(IGD_TEST_std);
        free(IGD_mean);
        free(IGD_std);
        free(HV_TRAIN_mean);
        free(HV_TRAIN_std);
        free(HV_TEST_mean);
        free(HV_TEST_std);
        free(HV_mean);
        free(HV_std);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
