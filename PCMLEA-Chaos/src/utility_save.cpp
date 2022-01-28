# include "global.h"
# include <math.h>

void save_double(FILE* fpt, double* pTarget, int num, int dim, int tag)
{
    if(tag)
        fprintf(fpt, "void save_double(FILE *fpt, double* pTarget, int num, int dim) with Addr %d\n", pTarget);
    int i, j;
    for(i = 0; i < num; i++) {
        for(j = 0; j < dim; j++) {
            fprintf(fpt, "%.16e\t", pTarget[i * dim + j]);
        }
        fprintf(fpt, "\n");
    }
    fprintf(fpt, "\n");
}

void save_double_as_int(FILE* fpt, double* pTarget, int num, int dim, int tag)
{
    if(tag)
        fprintf(fpt, "void save_double(FILE *fpt, double* pTarget, int num, int dim) with Addr %d\n", pTarget);
    int i, j;
    for(i = 0; i < num; i++) {
        for(j = 0; j < dim; j++) {
            fprintf(fpt, "%d\t", (int)(pTarget[i * dim + j]));
        }
        fprintf(fpt, "\n");
    }
    fprintf(fpt, "\n");
}

void save_int(FILE* fpt, int* pTarget, int num, int dim, int tag)
{
    if(tag)
        fprintf(fpt, "void save_int(FILE *fpt, int* pTarget, int num, int dim) with Addr %d\n", pTarget);
    int i, j;
    for(i = 0; i < num; i++) {
        for(j = 0; j < dim; j++) {
            fprintf(fpt, "%d\t", pTarget[i * dim + j]);
        }
        fprintf(fpt, "\n");
    }
    fprintf(fpt, "\n");
}

void output_group_info()
{
    int iObj;
    int i, j;
    int l;
    int size = 1;
    fprintf(st_global_p.debugFpt, "\nselect_count...%d...\n", get_select_count_uti_rand());
    fprintf(st_global_p.debugFpt, "\ngroup_info...\n");
    fprintf(st_global_p.debugFpt, "Diversity index:\n");
    for(i = 0; i < st_grp_ana_p.numDiverIndexes; i++) {
        fprintf(st_global_p.debugFpt, "%06d->(", st_grp_info_p.DiversityIndexs[i]);
        for(j = 0; j < size; j++)
            fprintf(st_global_p.debugFpt, "%lf ",
                    st_grp_ana_p.Control_Mean[st_grp_info_p.DiversityIndexs[i] * size + j]);
        fprintf(st_global_p.debugFpt, ")");
    }
    /*	fprintf(strct_global_paras.debugFpt,"\n");
    fprintf(strct_global_paras.debugFpt,"Diversity index of MOEA/DVA:\n");
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]>strct_grp_ana_vals.NumControlAnalysis*0.64)
    fprintf(strct_global_paras.debugFpt,"%06d->%lf ",i,strct_grp_ana_vals.Control_Mean[i]);*/
    /*	fprintf(strct_global_paras.debugFpt,"\n");
    fprintf(strct_global_paras.debugFpt,"\nControl_Mean_all...\n");
    for(i=0;i<strct_global_paras.nDim;i++)
    {
    fprintf(strct_global_paras.debugFpt,"%06d->(",i);
    for(j=0;j<size;j++)
    {
    fprintf(strct_global_paras.debugFpt, "%lf ",strct_grp_ana_vals.Control_Mean[i*size+j]);
    }
    fprintf(strct_global_paras.debugFpt,")");
    }*/
    fprintf(st_global_p.debugFpt, "\n");
    fprintf(st_global_p.debugFpt, "\nControl_all...\n");
    for(i = 0; i < st_global_p.nDim; i++) {
        fprintf(st_global_p.debugFpt, "%06d->(", i);
        for(j = 0; j < size; j++) {
            fprintf(st_global_p.debugFpt, "%d ", st_grp_ana_p.Control[i * size + j]);
        }
        fprintf(st_global_p.debugFpt, ")");
    }
    /*	fprintf(strct_global_paras.debugFpt,"\n");
    fprintf(strct_global_paras.debugFpt,"\nControlDist...\n");
    for(i=0;i<strct_global_paras.nDim;i++)
    fprintf(strct_global_paras.debugFpt,"%06d->%lf ",i,strct_grp_ana_vals.Control_Dist_Mean[i]);
    fprintf(strct_global_paras.debugFpt,"\n");
    fprintf(strct_global_paras.debugFpt,"Diver_Dist\n");
    int sum=0;
    for(i=0;i<strct_global_paras.nDim;i++)
    if(strct_grp_ana_vals.Control_Mean[i]==0)
    {
    fprintf(strct_global_paras.debugFpt,"%06d->%lf ",i,strct_grp_ana_vals.Control_Dist_Mean[i]);
    ++sum;
    }
    fprintf(strct_global_paras.debugFpt,"\nSUM=%d\n",sum);*/
    fprintf(st_global_p.debugFpt, "\nConvergence index:\n");
    for(i = 0; i < st_grp_ana_p.numConverIndexes; i++) {
        fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.ConvergenceIndexs[i]);
    }
    fprintf(st_global_p.debugFpt, "\n");
    fprintf(st_global_p.debugFpt, "\nstrct_grp_ana_vals.Dependent...\n");
    for(l = 0; l < st_global_p.nObj; l++) {
        fprintf(st_global_p.debugFpt, "\n\nOBJ: %d minWeight: %lf maxWeight: %lf\n", l + 1, st_grp_ana_p.weight_min[l],
                st_grp_ana_p.weight_max[l]);
        for(i = 0; i < st_global_p.nDim; i++) {
            fprintf(st_global_p.debugFpt, "\ni: %d\n", i + 1);
            for(j = 0; j < st_global_p.nDim; j++) {
                fprintf(st_global_p.debugFpt, "(%06d:%06d)->%d->%.16lf\t", i, j,
                        st_grp_ana_p.Dependent[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j],
                        st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim +
                                  j]);
            }
        }
        fprintf(st_global_p.debugFpt, "\n");
    }

    for(iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0)
            fprintf(st_global_p.debugFpt, "\ngroup_table_allObjectives...%d...\n", st_grp_info_p.Groups_sizes[iObj]);
        else
            fprintf(st_global_p.debugFpt, "\ngroup_table_separatedObjective...NO%d...%d...\n", iObj,
                    st_grp_info_p.Groups_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_sizes[iObj]; l++) {
            fprintf(st_global_p.debugFpt, "GROUP: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(i = 0; i < st_grp_info_p.Groups_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_i] + i;
                fprintf(st_global_p.debugFpt, "%06d\t", st_grp_info_p.Groups[tmp_j]);
            }
            fprintf(st_global_p.debugFpt, "\n");
        }
    }

    fprintf(st_global_p.debugFpt, "\nRAW INFORMATION\n");
    for(iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0)
            fprintf(st_global_p.debugFpt, "\ngroup_raw_table_allObjectives...%d...\n", st_grp_info_p.Groups_raw_sizes[iObj]);
        else
            fprintf(st_global_p.debugFpt, "\ngroup_raw_table_separatedObjective...NO%d...%d...\n", iObj,
                    st_grp_info_p.Groups_raw_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_raw_sizes[iObj]; l++) {
            fprintf(st_global_p.debugFpt, "GROUP_RAW: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(i = 0; i < st_grp_info_p.Groups_raw_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_raw_sub_disps[tmp_i] + i;
                fprintf(st_global_p.debugFpt, "%06d\t", st_grp_info_p.Groups_raw[tmp_j]);
            }
            fprintf(st_global_p.debugFpt, "\n");
        }
    }

    return;
}

void output_group_info_brief()
{
    int l;
    //int size = 1;
    fprintf(st_global_p.debugFpt, "\nselect_count...%d...\n", get_select_count_uti_rand());
    fprintf(st_global_p.debugFpt, "\ngroup_info...\n");
    fprintf(st_global_p.debugFpt, "\n");
    fprintf(st_global_p.debugFpt, "\nControl_all...\n");

    fprintf(st_global_p.debugFpt, "\nConvergence index:\n");
    for(int i = 0; i < st_grp_ana_p.numConverIndexes; i++) {
        fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.ConvergenceIndexs[i]);
    }
    fprintf(st_global_p.debugFpt, "\n");

    fprintf(st_global_p.debugFpt, "\nAbstract Info\n");
    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        fprintf(st_global_p.debugFpt, "iObj = %d --- size = %d\n", iObj, st_grp_info_p.Groups_sizes[iObj]);
        fprintf(st_global_p.debugFpt, "VARS:\n");
        for(int i = 0; i < st_global_p.nDim; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
        fprintf(st_global_p.debugFpt, "DISPS:\n");
        for(int i = 0; i < st_grp_info_p.Groups_sizes[iObj]; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_sub_disps[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
        fprintf(st_global_p.debugFpt, "SIZES:\n");
        for(int i = 0; i < st_grp_info_p.Groups_sizes[iObj]; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_sub_sizes[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0)
            fprintf(st_global_p.debugFpt, "\ngroup_table_allObjectives...%d...\n", st_grp_info_p.Groups_sizes[iObj]);
        else
            fprintf(st_global_p.debugFpt, "\ngroup_table_separatedObjective...NO%d...%d...\n", iObj,
                    st_grp_info_p.Groups_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_sizes[iObj]; l++) {
            fprintf(st_global_p.debugFpt, "GROUP: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(int i = 0; i < st_grp_info_p.Groups_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_sub_disps[tmp_i] + i;
                fprintf(st_global_p.debugFpt, "%06d\t", st_grp_info_p.Groups[tmp_j]);
            }
            fprintf(st_global_p.debugFpt, "\n");
        }
    }

    return;
}

void output_group_raw_info_brief()
{
    //int i, j;
    int l;
    //int size = 1;
    fprintf(st_global_p.debugFpt, "\nselect_count...%d...\n", get_select_count_uti_rand());
    fprintf(st_global_p.debugFpt, "\ngroup_info...\n");
    fprintf(st_global_p.debugFpt, "\n");
    fprintf(st_global_p.debugFpt, "\nControl_all...\n");

    fprintf(st_global_p.debugFpt, "\nConvergence index:\n");
    for(int i = 0; i < st_grp_ana_p.numConverIndexes; i++) {
        fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.ConvergenceIndexs[i]);
    }
    fprintf(st_global_p.debugFpt, "\n");

    fprintf(st_global_p.debugFpt, "\nRAW INFORMATION\n");
    fprintf(st_global_p.debugFpt, "\nAbstract Info\n");
    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        fprintf(st_global_p.debugFpt, "iObj = %d --- size = %d\n", iObj, st_grp_info_p.Groups_raw_sizes[iObj]);
        fprintf(st_global_p.debugFpt, "FLAGS: \n");
        for(int i = 0; i < st_global_p.nDim; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_raw_flags[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
        fprintf(st_global_p.debugFpt, "VARS: \n");
        for(int i = 0; i < st_global_p.nDim; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_raw[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
        fprintf(st_global_p.debugFpt, "DISPS: \n");
        for(int i = 0; i < st_grp_info_p.Groups_raw_sizes[iObj]; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_raw_sub_disps[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
        fprintf(st_global_p.debugFpt, "SIZES: \n");
        for(int i = 0; i < st_grp_info_p.Groups_raw_sizes[iObj]; i++) {
            fprintf(st_global_p.debugFpt, "%d\t", st_grp_info_p.Groups_raw_sub_sizes[iObj * st_global_p.nDim + i]);
        }
        fprintf(st_global_p.debugFpt, "\n");
    }

    for(int iObj = 0; iObj <= st_global_p.nObj; iObj++) {
        if(iObj == 0)
            fprintf(st_global_p.debugFpt, "\ngroup_raw_table_allObjectives...%d...\n", st_grp_info_p.Groups_raw_sizes[iObj]);
        else
            fprintf(st_global_p.debugFpt, "\ngroup_raw_table_separatedObjective...NO%d...%d...\n", iObj,
                    st_grp_info_p.Groups_raw_sizes[iObj]);

        for(l = 0; l < st_grp_info_p.Groups_raw_sizes[iObj]; l++) {
            fprintf(st_global_p.debugFpt, "GROUP_RAW: %d\n", l + 1);
            int tmp_i = iObj * st_global_p.nDim + l;
            for(int i = 0; i < st_grp_info_p.Groups_raw_sub_sizes[tmp_i]; i++) {
                int tmp_j = iObj * st_global_p.nDim + st_grp_info_p.Groups_raw_sub_disps[tmp_i] + i;
                fprintf(st_global_p.debugFpt, "%06d\t", st_grp_info_p.Groups_raw[tmp_j]);
            }
            fprintf(st_global_p.debugFpt, "\n");
        }
    }

    return;
}

void output_csv_matrix_recorded(const char* filename, double mat_data[][128], int nRun, int nTrace)
{
    FILE* theFile = fopen(filename, "w");
    for(int i = 0; i <= nRun; i++) {
        if(i)
            fprintf(theFile, "RUN%d", i);
        for(int j = 0; j <= nTrace; j++) {
            if(i == 0)
                fprintf(theFile, ",TRACE%d", j);
            else
                fprintf(theFile, ",%.16e", mat_data[i][j]);
        }
        if(i == 0)
            fprintf(theFile, ",FINAL\n");
        else
            fprintf(theFile, ",%.16e\n", mat_data[i][nTrace + 1]);
    }
    fclose(theFile);
}

void output_csv_mean_std(const char* fnm_mean, const char* fnm_std, double mat_data[][128],
                         int nRun, int nTrace, const char* prob, int nobj, int ndim)
{
    FILE* theFile = NULL;
    double data_mean[128];
    double data_std[128];
    for(int i = 0; i < 128; i++) {
        data_mean[i] = 0.0;
        data_std[i] = 0.0;
    }
    for(int i = 1; i <= nRun; i++) {
        for(int j = 0; j <= nTrace + 1; j++) {
            data_mean[j] += mat_data[i][j];
        }
    }
    for(int j = 0; j <= nTrace + 1; j++) {
        data_mean[j] /= nRun;
    }
    for(int i = 1; i <= nRun; i++) {
        for(int j = 0; j <= nTrace + 1; j++) {
            data_std[j] += (mat_data[i][j] - data_mean[j]) * (mat_data[i][j] - data_mean[j]);
        }
    }
    for(int j = 0; j <= nTrace + 1; j++) {
        data_std[j] = sqrt(data_std[j] / (double)nRun);
    }
    theFile = fopen(fnm_mean, "a");
    {
        fprintf(theFile, "%s_OBJ%d_VAR%d", prob, nobj, ndim);
        for(int j = 0; j <= nTrace; j++) {
            fprintf(theFile, ",%.16e", data_mean[j]);
        }
        fprintf(theFile, ",%.16e\n", data_mean[nTrace + 1]);
    }
    fclose(theFile);
    theFile = fopen(fnm_std, "a");
    {
        fprintf(theFile, "%s_OBJ%d_VAR%d", prob, nobj, ndim);
        for(int j = 0; j <= nTrace; j++) {
            fprintf(theFile, ",%.16e", data_std[j]);
        }
        fprintf(theFile, ",%.16e\n", data_std[nTrace + 1]);
    }
    fclose(theFile);
}

void output_csv_vec_int(const char* filename, int mat_data[],
                        int nRun, int nTrace, const char* prob, int nobj, int ndim)
{
    FILE* theFile = fopen(filename, "a");
    {
        fprintf(theFile, "%s_OBJ%d_VAR%d", prob, nobj, ndim);
        for(int i = 1; i <= nRun; i++) {
            fprintf(theFile, ",%d", mat_data[i]);
        }
        fprintf(theFile, "\n");
    }
    fclose(theFile);
}

void output_csv_vec_double_with_mean(const char* filename, double mat_data[],
                                     int nRun, int nTrace, const char* prob, int nobj, int ndim)
{
    double data_mean;
    FILE* theFile = fopen(filename, "a");
    {
        fprintf(theFile, "%s_OBJ%d_VAR%d", prob, nobj, ndim);
        data_mean = 0.0;
        for(int i = 1; i <= nRun; i++) {
            fprintf(theFile, ",%.16e", mat_data[i]);
            data_mean += mat_data[i];
        }
        data_mean /= nRun;
        fprintf(theFile, ",%.16e\n", data_mean);
    }
    fclose(theFile);
}
