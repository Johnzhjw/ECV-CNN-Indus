#include "global.h"
#include <math.h>

void exec_DVA()
{
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) printf("exec_DVA()\n");
    //
    strcpy(st_grp_ana_p.strUpdateSolutionMode, "ValueSum");
    // 	strcpy(strct_grp_ana_vals.strUpdateSolutionMode,"Dominate");
    //rand_type = 0;

    if(st_global_p.nObj == 2) {
        st_grp_ana_p.NumControlAnalysis = 5;
        st_grp_ana_p.weight_thresh = 1e-4;
    }
    if(st_global_p.nObj == 3) {
        st_grp_ana_p.NumControlAnalysis = 3;
        st_grp_ana_p.weight_thresh = 1e-6;
    }
    st_grp_ana_p.NumControlAnalysis = 20; //2;5;10;//same
    st_grp_ana_p.weight_thresh = 1e-6;

    ControlVarAnalysis();
    //	ControlVariableAnalysis_serial();
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) printf("ControlVarAnalysis_DONE\n");

    VarGrouping();
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(0 == strct_MPI_info.mpi_rank) printf("VarGrouping_DONE\n");

    // rand selection
    // 0 -> uniform
    // 1 -> chebyshevMap
    // 2 -> piecewise_linearMap
    // 3 -> sinusMap
    // 4 -> logisticMap
    // 5 -> circleMap
    // 6 -> gaussMap
    // 7 -> tentMap
    // rand_type = 0;
}

//Grouping decision variable based on linkage on all the objective functions
//to decompose distance variables
void VarGrouping()
{
    int l, i, j, k;

    //	InterdependenceAnalysis_master_slave();
    //	InterdependenceAnalysis();
    InterdependenceAnalysis_gDG2();
    // 	strct_grp_ana_vals.DependentVariableAnalysis();

    //	MPI_Barrier(MPI_COMM_WORLD);
    //	if(0==strct_MPI_info.mpi_rank) printf("gDG2_DONE\n");

    if(st_grp_ana_p.numDiverIndexes == st_global_p.nDim) {
        double maxDep = 0;
        double* DepCount = (double*)calloc(st_global_p.nDim, sizeof(double));

        for(l = 0; l < st_global_p.nObj; l++) {
            for(i = 0; i < st_global_p.nDim; i++) {
                for(j = 0; j < st_global_p.nDim; j++) {
                    DepCount[i] += st_grp_ana_p.Interdependence_Weight[l * st_global_p.nDim * st_global_p.nDim + i * st_global_p.nDim + j];
                }
            }
        }
        for(i = 0; i < st_global_p.nDim; i++)
            if(DepCount[i] > maxDep)
                maxDep = DepCount[i];

        st_grp_ana_p.numDiverIndexes = 0;
        st_grp_ana_p.numConverIndexes = 0;
        for(i = 0; i < st_global_p.nDim; i++) {
            if(DepCount[i] > 0.6 * maxDep)
                st_grp_info_p.DiversityIndexs[st_grp_ana_p.numDiverIndexes++] = i;
            else
                st_grp_info_p.ConvergenceIndexs[st_grp_ana_p.numConverIndexes++] = i;
        }

        free(DepCount);

        printf("AFTER...RANK%d_%s_OBJ%d_RUN%d_NDIV%d\n", st_MPI_p.mpi_rank, st_global_p.testInstance, st_global_p.nObj,
               st_ctrl_p.cur_run,
               st_grp_ana_p.numDiverIndexes);
    }

    bool myFlag;

    int* flag = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* checked = (int*)calloc(st_global_p.nDim, sizeof(int));
    int status;
    int loop;
    int pivot;

    // all objectives
    loop = 1;
    for(i = 0; i < st_global_p.nDim; i++) flag[i] = 0;
    for(i = 0; i < st_grp_ana_p.numDiverIndexes; i++) flag[st_grp_info_p.DiversityIndexs[i]] = -1;
    for(i = 0; i < st_global_p.nDim; i++) {
        if(flag[i] == 0) {
            status = 1;
            flag[i] = loop; //if (0 == strct_MPI_info.mpi_rank) printf("%d -- %d ", i, loop);
            pivot = i;
            for(j = 0; j < st_global_p.nDim; j++) checked[j] = 0;
            while(status) {
                checked[pivot] = 1;
                for(j = 0; j < st_global_p.nDim; j++) {
                    if(flag[j] < 0) continue;
                    myFlag = false;
                    for(k = 0; k < st_global_p.nObj; k++) {
                        if(st_grp_ana_p.Dependent[k * st_global_p.nDim * st_global_p.nDim + pivot * st_global_p.nDim + j] == 1) {
                            myFlag = true;
                            break;
                        }
                    }
                    if(flag[j] == 0 && myFlag) {
                        flag[j] = loop; //if (0 == strct_MPI_info.mpi_rank) printf("%d -- %d ", j, loop);
                    }
                }
                status = 0;
                for(j = 0; j < st_global_p.nDim; j++) {
                    if(flag[j] == loop && checked[j] == 0) {
                        pivot = j;
                        status = 1;
                        break;
                    }
                }
            }
            loop++;
        }
    }

    memcpy(&st_grp_info_p.Groups_raw_flags[0 * st_global_p.nDim], flag, st_global_p.nDim * sizeof(int));
    //if (0 == strct_MPI_info.mpi_rank) for (i = 0; i < strct_global_paras.nDim; i++) printf("%d ", strct_grp_info_vals.Groups_raw_flags[i]);
    st_grp_info_p.Groups_raw_sizes[0] = loop - 1;
    for(i = 1; i < loop; i++) {
        int tmp_count = 0;
        for(j = 0; j < st_global_p.nDim; j++) {
            if(flag[j] == i)
                tmp_count++;
        }
        int tmp_i = 0 * st_global_p.nDim + i - 1;
        st_grp_info_p.Groups_raw_sub_sizes[tmp_i] = tmp_count;
        if(i == 1) {
            st_grp_info_p.Groups_raw_sub_disps[tmp_i] = 0;
        } else {
            st_grp_info_p.Groups_raw_sub_disps[tmp_i] =
                st_grp_info_p.Groups_raw_sub_disps[tmp_i - 1] +
                st_grp_info_p.Groups_raw_sub_sizes[tmp_i - 1];
        }
        tmp_count = 0;
        for(j = 0; j < st_global_p.nDim; j++) {
            if(flag[j] == i) {
                int tmp_idx = tmp_count + st_grp_info_p.Groups_raw_sub_disps[tmp_i] + 0 * st_global_p.nDim;
                st_grp_info_p.Groups_raw[tmp_idx] = j;
                tmp_count++;
            }
        }
    }

    //separated objective
    for(int iObj = 0; iObj < st_global_p.nObj; iObj++) {
        loop = 1;
        for(i = 0; i < st_global_p.nDim; i++) flag[i] = 0;
        for(i = 0; i < st_grp_ana_p.numDiverIndexes; i++) flag[st_grp_info_p.DiversityIndexs[i]] = -1;
        for(i = 0; i < st_global_p.nDim; i++) {
            if(flag[i] == 0) {
                status = 1;
                flag[i] = loop;
                pivot = i;
                for(j = 0; j < st_global_p.nDim; j++) checked[j] = 0;
                while(status) {
                    checked[pivot] = 1;
                    for(j = 0; j < st_global_p.nDim; j++) {
                        if(flag[j] < 0) continue;
                        myFlag = false;
                        if(st_grp_ana_p.Dependent[iObj * st_global_p.nDim * st_global_p.nDim + pivot * st_global_p.nDim + j] == 1) {
                            myFlag = true;
                        }
                        if(flag[j] == 0 && myFlag) {
                            flag[j] = loop;
                        }
                    }
                    status = 0;
                    for(j = 0; j < st_global_p.nDim; j++) {
                        if(flag[j] == loop && checked[j] == 0) {
                            pivot = j;
                            status = 1;
                            break;
                        }
                    }
                }
                loop++;
            }
        }

        memcpy(&st_grp_info_p.Groups_raw_flags[(iObj + 1) * st_global_p.nDim], flag, st_global_p.nDim * sizeof(int));
        st_grp_info_p.Groups_raw_sizes[iObj + 1] = loop - 1;
        for(i = 1; i < loop; i++) {
            int tmp_count = 0;
            for(j = 0; j < st_global_p.nDim; j++) {
                if(flag[j] == i)
                    tmp_count++;
            }
            int tmp_i = (iObj + 1) * st_global_p.nDim + i - 1;
            st_grp_info_p.Groups_raw_sub_sizes[tmp_i] = tmp_count;
            if(i == 1) {
                st_grp_info_p.Groups_raw_sub_disps[tmp_i] = 0;
            } else {
                st_grp_info_p.Groups_raw_sub_disps[tmp_i] =
                    st_grp_info_p.Groups_raw_sub_disps[tmp_i - 1] +
                    st_grp_info_p.Groups_raw_sub_sizes[tmp_i - 1];
            }
            tmp_count = 0;
            for(j = 0; j < st_global_p.nDim; j++) {
                if(flag[j] == i) {
                    int tmp_idx = tmp_count +
                                  st_grp_info_p.Groups_raw_sub_disps[tmp_i] +
                                  (iObj + 1) * st_global_p.nDim;
                    st_grp_info_p.Groups_raw[tmp_idx] = j;
                    tmp_count++;
                }
            }
        }
    }
    /*	{
    		//all
    		{
    		vector<int> cluster;
    		vector<int> TmpConverIndex; for(int iCon=0;iCon<strct_grp_ana_vals.numConverIndexes;iCon++) TmpConverIndex.push_back(strct_grp_info_vals.ConvergenceIndexs[iCon]);	//MOEA/DVA just consider the linkages between distance variables to group variable.
    		while (TmpConverIndex.size() > 0) {
    		cluster.push_back(TmpConverIndex[0]);
    		TmpConverIndex.erase(TmpConverIndex.begin());
    		for (i=0; i<cluster.size(); i++) {
    		for (j=0; j<TmpConverIndex.size(); j++) { //just consider  the linkages between distance variables
    		for (k=0; k<strct_global_paras.nObj; k++) { //consider the linkage in all of objective functions
    		bool tTag=false;
    		if (strct_grp_ana_vals.Dependent[k*strct_global_paras.nDim*strct_global_paras.nDim+cluster[i]*strct_global_paras.nDim+TmpConverIndex[j]] == 1) {
    		cluster.push_back(TmpConverIndex[j]);
    		TmpConverIndex.erase(TmpConverIndex.begin() + j);
    		j--;
    		tTag=true;
    		}
    		if(tTag)
    		break;
    		}
    		}
    		}
    		strct_grp_info_vals.Groups.push_back(cluster);
    		cluster.clear();
    		}
    		vector<int>().swap(cluster);
    		vector<int>().swap(TmpConverIndex);

    		// obj
    		{
    		for(l=0;l<strct_global_paras.nObj;l++)
    		{
    		vector<int> cluster;
    		vector<vector<int>> grp_obj;
    		vector<int> TmpConverIndex; for(int iCon=0;iCon<strct_grp_ana_vals.numConverIndexes;iCon++) TmpConverIndex.push_back(strct_grp_info_vals.ConvergenceIndexs[iCon]);	//MOEA/DVA just consider the linkages between distance variables to group variable.
    		//					shuffle(&TmpConverIndex[0],TmpConverIndex.size());
    		//					MPI_Bcast(&TmpConverIndex[0],TmpConverIndex.size(),MPI_INT,0,MPI_COMM_WORLD);

    		while (TmpConverIndex.size() > 0) {
    		cluster.push_back(TmpConverIndex[0]);
    		TmpConverIndex.erase(TmpConverIndex.begin());
    		for (i=0; i<cluster.size(); i++) {
    		for (j=0; j<TmpConverIndex.size(); j++) { //just consider  the linkages between distance variables
    		bool tTag=false;
    		if (strct_grp_ana_vals.Dependent[l*strct_global_paras.nDim*strct_global_paras.nDim+cluster[i]*strct_global_paras.nDim+TmpConverIndex[j]] == 1) {
    		cluster.push_back(TmpConverIndex[j]);
    		TmpConverIndex.erase(TmpConverIndex.begin() + j);
    		j--;
    		}
    		}
    		}
    		grp_obj.push_back(cluster);
    		cluster.clear();
    		}
    		Groups_per_separatedObjective.push_back(grp_obj);

    		vector<int>().swap(cluster);
    		for(int m=0;m<grp_obj.size();m++)
    		vector<int>().swap(grp_obj[m]);
    		vector<vector<int>>().swap(grp_obj);
    		vector<int>().swap(TmpConverIndex);
    		}
    		}
    		}
    		}*/

    //	MPI_Barrier(MPI_COMM_WORLD);

    free(flag);
    free(checked);
}