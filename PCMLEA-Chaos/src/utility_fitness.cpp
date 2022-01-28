# include "global.h"
# include <math.h>

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// DECOMPOSITION
double fitnessFunction(double* solution, double* lamda)
{
    double fvalue = 0;

    double alpha = 2.0;

    // Tchebycheff approach
    if(!strcmp(st_global_p.strFunctionType, "_TCH1")) {
        double max_fun = -1.0e+30;
        for(int n = 0; n < st_global_p.nObj; n++) {
            double diff = fabs(solution[n] - st_decomp_p.idealpoint[n]);
            double feval;
            if(lamda[n] == 0)
                feval = 0.00001 * diff;
            else
                feval = diff * lamda[n];
            if(feval > max_fun) max_fun = feval;
        }
        fvalue = max_fun;
        return fvalue;
    }

    // normalized Tchebycheff approach
    if(!strcmp(st_global_p.strFunctionType, "_TCH2")) {
        double max_fun = -1.0e+30;
        for(int n = 0; n < st_global_p.nObj; n++) {
            double diff = (solution[n] - st_decomp_p.idealpoint[n]) / (st_decomp_p.nadirpoint[n] -
                          st_decomp_p.idealpoint[n] + 1e-6);
            double feval;
            if(lamda[n] == 0)
                feval = 0.0001 * diff;
            else
                feval = diff * lamda[n];
            if(feval > max_fun) max_fun = feval;
        }
        fvalue = max_fun;
        return fvalue;
    }

    if(!strcmp(st_global_p.strFunctionType, "_MTCH1")) {
        double max_fun = -1.0e+30;
        for(int n = 0; n < st_global_p.nObj; n++) {
            double diff = fabs(solution[n] - st_decomp_p.idealpoint[n]);
            double feval;
            if(lamda[n] == 0)
                feval = diff / 0.00001;
            else
                feval = diff / lamda[n];
            if(feval > max_fun) max_fun = feval;
        }
        fvalue = max_fun;
        return fvalue;
    }

    // normalized Tchebycheff approach
    if(!strcmp(st_global_p.strFunctionType, "_MTCH2")) {
        double max_fun = -1.0e+30;
        for(int n = 0; n < st_global_p.nObj; n++) {
            double diff = (solution[n] - st_decomp_p.idealpoint[n]) / (st_decomp_p.nadirpoint[n] -
                          st_decomp_p.idealpoint[n] + 1e-6);
            double feval;
            if(lamda[n] == 0)
                feval = diff / 0.0001;
            else
                feval = diff / lamda[n];
            if(feval > max_fun) max_fun = feval;
        }
        fvalue = max_fun;
        return fvalue;
    }

    //* Boundary intersection approach
    if(!strcmp(st_global_p.strFunctionType, "_PBI")) {
        // normalize the weight vector (line segment)
        double nd = norm(lamda, st_global_p.nObj);
        int n;
        for(int i = 0; i < st_global_p.nObj; i++)
            lamda[i] = lamda[i] / nd;

        double* realA = (double*)calloc(st_global_p.nObj, sizeof(double));
        double* realB = (double*)calloc(st_global_p.nObj, sizeof(double));

        // difference between current point and reference point
        for(n = 0; n < st_global_p.nObj; n++)
            realA[n] = (solution[n] - st_decomp_p.idealpoint[n]);

        // distance along the line segment
        double d1 = fabs(innerproduct(realA, lamda, st_global_p.nObj));

        // distance to the line segment
        for(n = 0; n < st_global_p.nObj; n++)
            realB[n] = (solution[n] - (st_decomp_p.idealpoint[n] + d1 * lamda[n]));
        double d2 = norm(realB, st_global_p.nObj);

        fvalue = d1 + 25 * d2;
        free(realA);
        free(realB);
        return fvalue;
    }

    //* Normalized Boundary intersection approach
    if(!strcmp(st_global_p.strFunctionType, "_MPBI")) {
        // normalize the weight vector (line segment)
        double nd = norm(lamda, st_global_p.nObj);
        int n;
        for(int i = 0; i < st_global_p.nObj; i++)
            lamda[i] = lamda[i] / nd;

        double* realA = (double*)calloc(st_global_p.nObj, sizeof(double));
        double* realB = (double*)calloc(st_global_p.nObj, sizeof(double));

        // difference between current point and reference point
        for(n = 0; n < st_global_p.nObj; n++)
            realA[n] = (solution[n] - st_decomp_p.idealpoint[n]) / (st_decomp_p.nadirpoint[n] -
                       st_decomp_p.idealpoint[n] + 1e-6);

        // distance along the line segment
        double d1 = fabs(innerproduct(realA, lamda, st_global_p.nObj));

        // distance to the line segment
        for(n = 0; n < st_global_p.nObj; n++)
            realB[n] = (realA[n] - (d1 * lamda[n]));
        double d2 = norm(realB, st_global_p.nObj);

        fvalue = d1 + st_global_p.nObj * d2 * pow((double)(st_global_p.iter - st_global_p.usedIter_init) /
                 (st_global_p.maxIter - st_global_p.usedIter_init + 1e-6),
                 alpha);
        free(realA);
        free(realB);
        return fvalue;
    }

    //* Angle-Penalized Distance (APD)
    if(!strcmp(st_global_p.strFunctionType, "_APD")) {
        double* realA = (double*)calloc(st_global_p.nObj, sizeof(double));
        int n;

        // difference between current point and reference point
        for(n = 0; n < st_global_p.nObj; n++)
            realA[n] = (solution[n] - st_decomp_p.idealpoint[n]);

        // distance along the line segment
        double d1 = fabs(innerproduct(realA, lamda, st_global_p.nObj));
        double quantity = norm(realA, st_global_p.nObj);
        double theta = acos(d1 / quantity);

        fvalue = (1 + st_global_p.nObj * pow((double)(st_global_p.iter - st_global_p.usedIter_init) /
                                             (st_global_p.maxIter - st_global_p.usedIter_init + 1e-6),
                                             alpha) * theta) * quantity;
        free(realA);
        return fvalue;
    }

    // Weight Sum approach
    if(!strcmp(st_global_p.strFunctionType, "_WS")) {
        for(int n = 0; n < st_global_p.nObj; n++) {
            fvalue += lamda[n] * solution[n];
        }
        return fvalue;
    }

    return fvalue;
}

double norm(double* vec, int len)
{
    double sum = 0;
    for(int n = 0; n < len; n++)
        sum += vec[n] * vec[n];
    return sqrt(sum);
}

double innerproduct(double* vec1, double* vec2, int len)
{
    int dim = len;
    double sum = 0;
    for(int n = 0; n < dim; n++)	sum = sum + vec1[n] * vec2[n];
    return sum;
}

void update_idealpoint(double* candidate)
{
    for(int n = 0; n < st_global_p.nObj; n++) {
        if(candidate[n] < st_decomp_p.idealpoint[n]) {
            st_decomp_p.idealpoint[n] = candidate[n];
        }
    }

    return;
}

void update_nadirpoint(double* solutionAddress, int pop_size, int nObj)
{
    int i, j;

    for(i = 0; i < pop_size; i++)
        for(j = 0; j < nObj; j++)
            if(st_decomp_p.nadirpoint[j] < solutionAddress[i * nObj + j])
                st_decomp_p.nadirpoint[j] = solutionAddress[i * nObj + j];

    return;
}

void load_samplePoints()
{
    int nPop = st_global_p.nPop;
    int nDim = st_global_p.nDim;
    int nObj = st_global_p.nObj;
    double* minLimit = st_global_p.minLimit;
    double* maxLimit = st_global_p.maxLimit;
    int* DiversityIndexs = st_grp_info_p.DiversityIndexs;
    double* diver_var_store_all = st_grp_info_p.diver_var_store_all;
    int type_test = st_ctrl_p.type_test;
    double* weights_all = st_decomp_p.weights_all;
    //
    int i, j;
    int nDiv = st_grp_ana_p.numDiverIndexes;
    int indReal;
    if(nDiv == 1) {
        for(i = 0; i < nPop; i++) {
            indReal = DiversityIndexs[0];
            diver_var_store_all[i * nDim + 0] =
                minLimit[indReal] + (double)(i) / (double)(nPop - 1) * (maxLimit[indReal] - minLimit[indReal]);
        }
    } else if(nDiv >= 2 && nDiv <= 4) {
        char FileName[256];
        sprintf(FileName, "DATA_alg/SamplePoint/SamplePoint_Dim%dN%d.txt", nDiv, nPop);
        FILE* fpt;
        fpt = fopen(FileName, "r");
        if(fpt == NULL) {
            printf("%s:\nsample point file error...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_SAMPLE_POINT_READING);
        }
        int tmp;
        double elem;
        for(i = 0; i < nPop; i++) {
            for(j = 0; j < nDiv; j++) {
                tmp = fscanf(fpt, "%lf", &elem);
                if(tmp == EOF) {
                    printf("%s:\nsample points are not enough...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_SAMPLE_POINT_NOT_ENOUGH);
                }
                indReal = DiversityIndexs[j];
                diver_var_store_all[i * nDim + j] =
                    minLimit[indReal] + elem * (maxLimit[indReal] - minLimit[indReal]);
            }
        }
        fclose(fpt);
    } else {
        for(i = 0; i < nPop; i++) {
            for(j = 0; j < nDiv; j++) {
                indReal = DiversityIndexs[j];
                diver_var_store_all[i * nDim + j] = rndreal(minLimit[indReal], maxLimit[indReal]);
            }
        }
    }

    if(type_test == MY_TYPE_LeNet_ENSEMBLE) {
        char FileName[256];
        sprintf(FileName, "DATA_alg/Weight/W%dD_%d.dat", nObj, nPop);
        FILE* fpt;
        fpt = fopen(FileName, "r");
        if(fpt == NULL) {
            printf("\n%s:Weight file error...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_READING);
        }
        int tmp;
        double elem;
        for(i = 0; i < nPop; i++) {
            for(j = 0; j < nObj; j++) {
                tmp = fscanf(fpt, "%lf", &elem);
                if(tmp == EOF) {
                    printf("\n%s:weights are not enough...\n", AT);
                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_NOT_ENOUGH);
                }
                weights_all[i * nObj + j] = elem;
            }
        }
        fclose(fpt);
    } else {
    }

    return;
}

void load_weights()
{
    int i, j;
    //load weights
    //if(!strct_MPI_info.color_population) {
    //    char FileName[256];
    //    sprintf(FileName, "Weight/W%dD_%d.dat", strct_global_paras.nObj, strct_global_paras.nPop);
    //    FILE* fpt;
    //    fpt = fopen(FileName, "r");
    //    if(fpt == NULL) {
    //        printf("\n%s:Weight file error...\n", AT);
    //        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_READING);
    //    }
    //    int tmp;
    //    double elem;
    //    for(i = 0; i < strct_global_paras.nPop; i++) {
    //        strct_decomp_paras.weight_prefer_tag[i] = 0;
    //        for(j = 0; j < strct_global_paras.nObj; j++) {
    //            tmp = fscanf(fpt, "%lf", &elem);
    //            if(tmp == EOF) {
    //                printf("\n%s:weights are not enough...\n", AT);
    //                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_NOT_ENOUGH);
    //            }
    //            strct_decomp_paras.weights_all[i * strct_global_paras.nObj + j] = elem;
    //            if(elem >= 1.0) strct_decomp_paras.weight_prefer_tag[i] = strct_decomp_paras.prefer_intensity;
    //        }
    //    }
    //    fclose(fpt);
    //} else {
    //    char FileName[256];
    //    sprintf(FileName, "Weight/W%dD_%d.dat", strct_global_paras.nObj - 1, strct_global_paras.nPop);
    //    FILE* fpt;
    //    fpt = fopen(FileName, "r");
    //    if(fpt == NULL) {
    //        printf("\n%s:Weight file error...\n", AT);
    //        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_READING);
    //    }
    //    int tmp;
    //    double elem;
    //    for(i = 0; i < strct_global_paras.nPop; i++) {
    //        strct_decomp_paras.weight_prefer_tag[i] = 0;
    //        for(j = 0; j < strct_global_paras.nObj; j++) {
    //            if(j == strct_MPI_info.color_population - 1) {
    //                elem = 0;
    //            } else {
    //                tmp = fscanf(fpt, "%lf", &elem);
    //                if(tmp == EOF) {
    //                    printf("\n%s:weights are not enough...\n", AT);
    //                    MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_NOT_ENOUGH);
    //                }
    //            }
    //            strct_decomp_paras.weights_all[i * strct_global_paras.nObj + j] = elem;
    //            if(elem >= 1.0) strct_decomp_paras.weight_prefer_tag[i] = strct_decomp_paras.prefer_intensity;
    //        }
    //    }
    //    fclose(fpt);
    //}
    //
    char FileName[256];
    sprintf(FileName, "DATA_alg/Weight/W%dD_%d.dat", st_global_p.nObj, st_global_p.nPop);
    FILE* fpt;
    fpt = fopen(FileName, "r");
    if(fpt == NULL) {
        printf("\n%s:Weight file error...\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_READING);
    }
    int tmp;
    double elem;
    for(i = 0; i < st_global_p.nPop; i++) {
        st_decomp_p.weight_prefer_tag[i] = 0;
        for(j = 0; j < st_global_p.nObj; j++) {
            tmp = fscanf(fpt, "%lf", &elem);
            if(tmp == EOF) {
                printf("\n%s:weights are not enough...\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_NOT_ENOUGH);
            }
            if(elem >= 1.0)
                st_decomp_p.weight_prefer_tag[i] = st_decomp_p.prefer_intensity;
            if(st_MPI_p.color_pop && j == st_MPI_p.color_pop - 1) {
                elem = 0;
            }
            st_decomp_p.weights_all[i * st_global_p.nObj + j] = elem;
        }
    }
    fclose(fpt);
    //
    if(!st_MPI_p.color_pop) {
        double* weight_std = (double*)calloc(st_global_p.nPop, sizeof(double));
        double  weight_std_min = INF_DOUBLE;
        for(i = 0; i < st_global_p.nPop; i++) {
            double w_mean = 0.0;
            for(j = 0; j < st_global_p.nObj; j++) {
                w_mean += st_decomp_p.weights_all[i * st_global_p.nObj + j];
            }
            w_mean /= st_global_p.nObj;
            for(j = 0; j < st_global_p.nObj; j++) {
                weight_std[i] += (st_decomp_p.weights_all[i * st_global_p.nObj + j] - w_mean) *
                                 (st_decomp_p.weights_all[i * st_global_p.nObj + j] - w_mean);
            }
            weight_std[i] /= st_global_p.nObj;
            weight_std[i] = sqrt(weight_std[i]);
            if(weight_std[i] < weight_std_min)
                weight_std_min = weight_std[i];
        }
        for(i = 0; i < st_global_p.nPop; i++) {
            if(weight_std[i] <= weight_std_min)
                st_decomp_p.weight_prefer_tag[i] = st_decomp_p.prefer_intensity;
        }
        free(weight_std);
    } else {
        sprintf(FileName, "DATA_alg/Weight/W%dD_%d.dat", st_global_p.nObj - 1, st_global_p.nPop);
        fpt = fopen(FileName, "r");
        if(fpt == NULL) {
            printf("\n%s:Weight file error...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_READING);
        }
        int i_obj = st_MPI_p.color_pop - 1;
        int tmp;
        double elem;
        for(i = 0; i < st_global_p.nPop; i++) {
            st_decomp_p.weight_prefer_tag[i] = 0;
            for(j = 0; j < st_global_p.nObj; j++) {
                if(j == i_obj) {
                    st_decomp_p.weights_all[i * st_global_p.nObj + j] = 0;
                } else {
                    tmp = fscanf(fpt, "%lf", &elem);
                    if(tmp == EOF) {
                        printf("\n%s:weights are not enough...\n", AT);
                        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_WEIGHT_NOT_ENOUGH);
                    }
                    if(elem >= 1.0)
                        st_decomp_p.weight_prefer_tag[i] = st_decomp_p.prefer_intensity;
                    st_decomp_p.weights_all[i * st_global_p.nObj + j] = elem;
                }
            }
        }
        fclose(fpt);
        for(int i = 0; i < st_global_p.nPop; i++) {
            //for(int j = 0; j < strct_global_paras.nObj; j++) {
            //    strct_decomp_paras.weights_all[i * strct_global_paras.nObj + j] = rndreal(0.01, 1.0);
            //}
            st_decomp_p.weights_all[i * st_global_p.nObj + st_MPI_p.color_pop - 1] = 0;
        }
    }
    //
    for(i = 0; i < st_global_p.nPop; i++) {
        for(j = 0; j < st_global_p.nObj; j++) {
            if(st_ctrl_p.tag_prefer_which_obj == j && st_decomp_p.weights_all[i * st_global_p.nObj + j] <= 0) {
                st_decomp_p.weight_prefer_tag[i] = st_decomp_p.prefer_intensity;
            }
        }
    }
    //
    if(!strcmp(st_global_p.strFunctionType, "_APD")) {
        for(i = 0; i < st_global_p.nPop; i++) {
            double nm = norm(&st_decomp_p.weights_all[i * st_global_p.nObj], st_global_p.nObj);
            for(j = 0; j < st_global_p.nObj; j++) {
                st_decomp_p.weights_all[i * st_global_p.nObj + j] /= nm;
                st_decomp_p.weights_unit[i * st_global_p.nObj + j] = st_decomp_p.weights_all[i * st_global_p.nObj +
                        j];
            }
        }
    }
    //
    return;
}

void normalize_weights()
{
    update_recv_disp(st_MPI_p.each_size_subPop, 1, st_MPI_p.mpi_size_subPop, st_MPI_p.recv_size_subPop,
                     st_MPI_p.disp_size_subPop);

    double* diff = (double*)calloc(st_global_p.nObj, sizeof(double));
    int i, j;
    // 	if(strct_MPI_info.mpi_rank==240)
    // 	{
    // 		for(i=0;i<strct_global_paras.nPop_mine;i++)
    // 		{
    // 			printf("ID: %d\n",i+1);
    // 			for(j=0;j<strct_global_paras.nObj;j++)
    // 			{
    // 				printf("%lf\t",weights[i*strct_global_paras.nObj+j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 		for(i=0;i<strct_global_paras.nPop;i++)
    // 		{
    // 			printf("ID: %d\n",i+1);
    // 			for(j=0;j<strct_global_paras.nObj;j++)
    // 			{
    // 				printf("%lf\t",strct_decomp_paras.weights_all[i*strct_global_paras.nObj+j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 	}
    for(i = 0; i < st_global_p.nObj; i++)
        diff[i] = st_decomp_p.nadirpoint[i] - st_decomp_p.idealpoint[i];
    for(i = 0; i < st_global_p.nPop; i++) {
        double sum = 0.0;
        for(j = 0; j < st_global_p.nObj; j++) {
            st_decomp_p.weights_all[i * st_global_p.nObj + j] =
                st_decomp_p.weights_unit[i * st_global_p.nObj + j] * diff[j];
            sum += st_decomp_p.weights_all[i * st_global_p.nObj + j] * st_decomp_p.weights_all[i *
                    st_global_p.nObj + j];
        }
        for(j = 0; j < st_global_p.nObj; j++)
            st_decomp_p.weights_all[i * st_global_p.nObj + j] /= sqrt(sum);
    }
    //for (i = 0; i < strct_global_paras.nPop_mine; i++) {
    //	for (j = 0; j < strct_global_paras.nObj; j++)
    //		weights[i * strct_global_paras.nObj + j] =
    //		strct_decomp_paras.weights_all[(strct_MPI_info.disp_size_species[strct_MPI_info.mpi_rank_species] + i) * strct_global_paras.nObj + j];
    //}
    // 	if(strct_MPI_info.mpi_rank==240)
    // 	{
    // 		for(i=0;i<strct_global_paras.nPop_mine;i++)
    // 		{
    // 			printf("ID: %d\n",i+1);
    // 			for(j=0;j<strct_global_paras.nObj;j++)
    // 			{
    // 				printf("%lf\t",weights[i*strct_global_paras.nObj+j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 		for(i=0;i<strct_global_paras.nPop;i++)
    // 		{
    // 			printf("ID: %d\n",i+1);
    // 			for(j=0;j<strct_global_paras.nObj;j++)
    // 			{
    // 				printf("%lf\t",strct_decomp_paras.weights_all[i*strct_global_paras.nObj+j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 	}
    free(diff);

    return;
}

void generate_new_weights(double* new_weights, int nWT, int nOBJ)
{
    int i, j;
    for(i = 0; i < nWT; i++) {
        double tmp_sum = 0.0;
        for(j = 0; j < nOBJ; j++) {
            new_weights[i * nOBJ + j] = pointer_gen_rand();
            tmp_sum += new_weights[i * nOBJ + j];
        }
        for(j = 0; j < nOBJ; j++) {
            new_weights[i * nOBJ + j] =
                new_weights[i * nOBJ + j] / tmp_sum;
        }
    }
    //
    return;
}

void calculate_utility()
{
    double f1, f2, delta;
    st_utility_p.utility_min = VAL_MAX;
    st_utility_p.utility_max = VAL_MIN;

    for(int i = 0; i < st_global_p.nPop; i++) {
        f1 = fitnessFunction(&st_pop_evo_cur.obj[i * st_global_p.nObj],
                             &st_decomp_p.weights_all[i * st_global_p.nObj]);
        f2 = fitnessFunction(&st_pop_evo_cur.obj_saved[i * st_global_p.nObj],
                             &st_decomp_p.weights_all[i * st_global_p.nObj]);
        delta = f2 - f1;
        if(delta > 0) st_utility_p.utility_mean += delta;
        if(delta > 0.001) st_utility_p.utility[i] = 1.0;
        else {
            st_utility_p.utility[i] = (0.95 + 0.05 * delta / 0.001) * st_utility_p.utility[i];
        }
        if(st_utility_p.utility_max < st_utility_p.utility[i])
            st_utility_p.utility_max = st_utility_p.utility[i];
        if(st_utility_p.utility_min > st_utility_p.utility[i])
            st_utility_p.utility_min = st_utility_p.utility[i];
    }

    {
        memcpy(st_pop_evo_cur.var_saved, st_pop_evo_cur.var, st_global_p.nPop * st_global_p.nDim * sizeof(double));
        memcpy(st_pop_evo_cur.obj_saved, st_pop_evo_cur.obj, st_global_p.nPop * st_global_p.nObj * sizeof(double));
    }

    /*strct_utility_info.utility_mid1=strct_utility_info.utility_min+(strct_utility_info.utility_max-strct_utility_info.utility_min)/3.0;
    strct_utility_info.utility_mid2=strct_utility_info.utility_min+(strct_utility_info.utility_max-strct_utility_info.utility_min)/3.0*2.0;

    double tmp[strct_global_paras.nPop];
    memcpy(tmp,strct_utility_info.utility,strct_global_paras.nPop*sizeof(double));
    int i,j;
    for(i=0;i<strct_global_paras.nPop/3;i++)
    {
    for(j=i+1;j<strct_global_paras.nPop;j++)
    {
    if(tmp[i]>tmp[j])
    {
    double ttt=tmp[i];
    tmp[i]=tmp[j];
    tmp[j]=ttt;
    }
    }
    }
    strct_utility_info.utility_mid1=tmp[i];
    for(i=strct_global_paras.nPop-1;i>strct_global_paras.nPop/3*2;i--)
    {
    for(j=i-1;j>=strct_global_paras.nPop/3;j--)
    {
    if(tmp[i]<tmp[j])
    {
    double ttt=tmp[i];
    tmp[i]=tmp[j];
    tmp[j]=ttt;
    }
    }
    }
    strct_utility_info.utility_mid2=tmp[i];*/

    return;
}

void total_utility(int nPop, double* weights)
{
    double* utility_mean = &st_utility_p.utility_mean;
    double* utility = st_utility_p.utility;
    //int nPop = st_global_p.nPop;//
    int nObj = st_global_p.nObj;
    int nDim = st_global_p.nDim;
    double* cur_obj = st_pop_evo_cur.obj;
    double* cur_var = st_pop_evo_cur.var;
    double* cur_obj_saved = st_pop_evo_cur.obj_saved;
    double* cur_var_saved = st_pop_evo_cur.var_saved;
    //double* weights_all = st_decomp_p.weights_all;
    //
    double f1, f2, delta;
    (*utility_mean) = 0.0;
    //
    for(int i = 0; i < nPop; i++) {
        f1 = fitnessFunction(&cur_obj[i * nObj], &weights[i * nObj]);
        f2 = fitnessFunction(&cur_obj_saved[i * nObj], &weights[i * nObj]);
        delta = f2 - f1;
        if(delta > 0)
            (*utility_mean) += delta;
        if(delta > 0.001)
            utility[i] = 1.0;
        else
            utility[i] = (0.95 + 0.05 * delta / 0.001) * utility[i];
        if(delta > 0)
            (*utility_mean) += delta;
    }
    (*utility_mean) /= nPop;
    //
    memcpy(cur_var_saved, cur_var, nPop * nDim * sizeof(double));
    memcpy(cur_obj_saved, cur_obj, nPop * nObj * sizeof(double));
    //
    return;
}

void one_utility(double objVal_before, double objVal_current, int iP)
{
    double delta = objVal_before - objVal_current;
    if(delta > 0.001) st_utility_p.utility_cur[iP] = 1.0;
    else {
        st_utility_p.utility_cur[iP] = 0.95 * (1.0 + delta / 0.001) * st_utility_p.utility_cur[iP];
    }
}
