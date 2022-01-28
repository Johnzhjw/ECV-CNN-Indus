#include "global.h"
#include <math.h>
#include <time.h>

int  seed = 36;		// seed for random number strct_global_paras.generation
int  seed_chaos = 36;
int  NP = 100;		// population size
int  N_arch = 100;	// archive size

int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.global_time = (long)time(NULL);
    //////////////////////////////////////////////////////////////////////////
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &st_MPI_p.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &st_MPI_p.mpi_size);
    MPI_Get_processor_name(st_MPI_p.my_name, &st_MPI_p.name_len);
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("MPI_Init(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    st_ctrl_p.algo_mech_type =
        DECOMPOSITION;
    NONDOMINANCE;
    LOCALIZATION;
    int fun_start_num;
    // the start number of test function, you can find the explanation in strct_global_paras.testInstance.txt
    int fun_end_num;
    // the end number of test function
    char tmpfilename[] = "./DATA_alg/para_file";
    get_alg_mech_func_id_to_test(tmpfilename, st_ctrl_p.algo_mech_type, fun_start_num, fun_end_num);
#ifdef DEBUG_TAG_TMP
    MPI_Barrier(MPI_COMM_WORLD);
    if(st_MPI_p.mpi_rank == 0) printf("get_alg_mech_func_id_to_test(); \n");
#endif
    //////////////////////////////////////////////////////////////////////////
    switch(st_ctrl_p.algo_mech_type) {
    case LOCALIZATION:
        sprintf(st_global_p.algorithmName, "DPCCMOEA");
        break;
    case DECOMPOSITION:
        sprintf(st_global_p.algorithmName, "DPCCMOLSEA");
        break;
    case NONDOMINANCE:
        sprintf(st_global_p.algorithmName, "DPCCMOLSIA");
        break;
    default:
        if(0 == st_MPI_p.mpi_rank)
            printf("%s: No such algorithm mechanism type\n", AT);
        MPI_Abort(MPI_COMM_WORLD, MY_ERROR_NO_SUCH_ALGO_MECH);
        break;
    }
    //////////////////////////////////////////////////////////////////////////
    int num_run = NRUN;        //the number of running test instrct_grp_ana_vals.Dependently
    int iRun;
    int seq;
    char prob[256];
    int ndim;
    int nobj;
    int para_1 = 0;
    int para_2 = 0;
    int para_3 = 0;
    char tmp_str[1024];
    FILE* readf = fopen("testInstance.txt", "r");
    //////////////////////////////////////////////////////////////////////////
    int iPro;
    for(iPro = 1; iPro < fun_start_num; iPro++) {
        if(!fgets(tmp_str, sizeof(tmp_str), readf)) {
            if(st_MPI_p.mpi_rank == 0)
                printf("%s: Reading file error - no more line, exiting...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_LINE);
        }
    }
    for(iPro = fun_start_num; iPro <= fun_end_num; iPro++) {
        if(fgets(tmp_str, sizeof(tmp_str), readf)) {
            int nb;
            nb = sscanf(tmp_str, "%d %s %d %d %d %d %d", &seq, prob, &ndim, &nobj, &para_1, &para_2, &para_3);
            if(nb < 4) {
                if(st_MPI_p.mpi_rank == 0)
                    printf("%s: Reading file error - no more para, exiting...\n", AT);
                MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_PARA);
            }
        } else {
            if(st_MPI_p.mpi_rank == 0)
                printf("%s: Reading file error - no more line, exiting...\n", AT);
            MPI_Abort(MPI_COMM_WORLD, MY_ERROR_FILE_LINE);
        }
        modify_num_run(prob, num_run);
#ifdef DEBUG_TAG_TMP
        MPI_Barrier(MPI_COMM_WORLD);
        if(st_MPI_p.mpi_rank == 0) printf("modify_num_run(); \n");
#endif
        st_ctrl_p.type_test = MY_TYPE_NORMAL;
        //////////////////////////////////////////////////////////////////////////
        seed = 36 + st_MPI_p.mpi_rank;
        //////////////////////////////////////////////////////////////////////////
        for(iRun = 1; iRun <= num_run; iRun++) {
            seed = (seed + 111) % 1235;
            seed_chaos = (seed_chaos + 19) % 1500;
            set_init_rand_para(seed, seed_chaos);
#ifdef DEBUG_TAG_TMP
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("set_init_rand_para(); \n");
#endif
            MPI_Barrier(MPI_COMM_WORLD);
            int maxIter;
            //////////////////////////////////////////////////////////////////////////
            EMO_initialization(prob, nobj, ndim, iRun - 1, num_run, st_MPI_p.mpi_rank, para_1, para_2, para_3);
#ifdef DEBUG_TAG_TMP
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("EMO_initialization(); \n");
#endif
            maxIter = (int)(ndim * 1e4);
            modify_hyper_paras(prob, nobj, ndim, NP, N_arch, maxIter, st_ctrl_p.type_test);
#ifdef DEBUG_TAG_TMP
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("modify_hyper_paras(); \n");
#endif
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0)
                printf("\n--   run %d   --\n\n\n-- PROBLEM %s\n--  variables: %d\n--  objectives: %d\n--  maxIter: %d\n--  MPI size: %d\n\n",
                       iRun, prob, ndim, nobj, maxIter, st_MPI_p.mpi_size);
            //////////////////////////////////////////////////////////////////////////
            set_para(NP, ndim, nobj, N_arch, maxIter, prob, iRun);
#ifdef DEBUG_TAG_TMP
            MPI_Barrier(MPI_COMM_WORLD);
            if(st_MPI_p.mpi_rank == 0) printf("set_para(); \n");
#endif
            {
#ifdef DEBUG_TAG_TMP
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("BEFORE - run_DPCC(); \n");
#endif
                run_DPCC();
#ifdef DEBUG_TAG_TMP
                MPI_Barrier(MPI_COMM_WORLD);
                if(st_MPI_p.mpi_rank == 0) printf("AFTER  - run_DPCC(); \n");
#endif
            }

            if(st_MPI_p.mpi_rank == 0) {
                double tmp_mean;
                if(st_ctrl_p.indicator_tag == INDICATOR_IGD || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    tmp_mean = 0.0;
                    for(int i = 1; i <= iRun; i++)
                        tmp_mean += st_indicator_p.mat_IGD_all[i][NTRACE + 1];
                    printf("\n%s-OBJ_%d-IGD: MEAN --- %lf\n", st_global_p.testInstance, st_global_p.nObj, tmp_mean / iRun);
                }
                if(st_ctrl_p.indicator_tag == INDICATOR_HV || st_ctrl_p.indicator_tag == INDICATOR_IGD_HV) {
                    tmp_mean = 0.0;
                    for(int i = 1; i <= iRun; i++)
                        tmp_mean += st_indicator_p.mat_HV_all[i][NTRACE + 1];
                    printf("%s-OBJ_%d-HV:  MEAN --- %lf\n\n", st_global_p.testInstance, st_global_p.nObj, tmp_mean / iRun);
                }
                printf("\nConsumed time - %lfs.\n\n", st_indicator_p.vec_TIME_all[iRun]);
            }
            //////////////////////////////////////////////////////////////////////////
            EMO_finalization(prob);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // OUTPUT csv - indicator values and time consumption
        if(0 == st_MPI_p.mpi_rank) {
            char theFileName[256];
            //////////////////////////////////////////////////////////////////////////
            //IGD
            sprintf(theFileName, "OUTPUT/IGD_%s_%s_OBJ%d_VAR%d_MPI%d_%ld.csv",
                    st_global_p.algorithmName, prob, nobj, ndim, st_MPI_p.mpi_size, st_ctrl_p.global_time);
            output_csv_matrix_recorded(theFileName, st_indicator_p.mat_IGD_all, num_run, NTRACE);
            //////////////////////////////////////////////////////////////////////////
            //HV_TRAIN
            sprintf(theFileName, "OUTPUT/HV_TRAIN_%s_%s_OBJ%d_VAR%d_MPI%d_%ld.csv", st_global_p.algorithmName, prob, nobj,
                    ndim, st_MPI_p.mpi_size, st_ctrl_p.global_time);
            output_csv_matrix_recorded(theFileName, st_indicator_p.mat_HV_all_TRAIN, num_run, NTRACE);
            //////////////////////////////////////////////////////////////////////////
            //HV_VALIDATION
            sprintf(theFileName, "OUTPUT/HV_VALIDATION_%s_%s_OBJ%d_VAR%d_MPI%d_%ld.csv", st_global_p.algorithmName, prob, nobj,
                    ndim, st_MPI_p.mpi_size, st_ctrl_p.global_time);
            output_csv_matrix_recorded(theFileName, st_indicator_p.mat_HV_all_VALIDATION, num_run, NTRACE);
            //////////////////////////////////////////////////////////////////////////
            //HV_TEST
            sprintf(theFileName, "OUTPUT/HV_TEST_%s_%s_OBJ%d_VAR%d_MPI%d_%ld.csv", st_global_p.algorithmName, prob, nobj,
                    ndim, st_MPI_p.mpi_size, st_ctrl_p.global_time);
            output_csv_matrix_recorded(theFileName, st_indicator_p.mat_HV_all_TEST, num_run, NTRACE);
            //////////////////////////////////////////////////////////////////////////
            //HV
            sprintf(theFileName, "OUTPUT/HV_%s_%s_OBJ%d_VAR%d_MPI%d_%ld.csv", st_global_p.algorithmName, prob, nobj, ndim,
                    st_MPI_p.mpi_size, st_ctrl_p.global_time);
            output_csv_matrix_recorded(theFileName, st_indicator_p.mat_HV_all, num_run, NTRACE);
            //////////////////////////////////////////////////////////////////////////
            //MEAN & STD --- IGD
            char theFileName2[256];
            sprintf(theFileName, "OUTPUT/MEAN_IGD_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            sprintf(theFileName2, "OUTPUT/STD_IGD_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_mean_std(theFileName, theFileName2, st_indicator_p.mat_IGD_all, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //MEAN & STD --- HV_TRAIN
            sprintf(theFileName, "OUTPUT/MEAN_HV_TRAIN_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            sprintf(theFileName2, "OUTPUT/STD_HV_TRAIN_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_mean_std(theFileName, theFileName2, st_indicator_p.mat_HV_all_TRAIN, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //MEAN & STD --- HV_VALIDATION
            sprintf(theFileName, "OUTPUT/MEAN_HV_VALIDATION_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            sprintf(theFileName2, "OUTPUT/STD_HV_VALIDATION_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_mean_std(theFileName, theFileName2, st_indicator_p.mat_HV_all_VALIDATION, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //MEAN & STD --- HV_TEST
            sprintf(theFileName, "OUTPUT/MEAN_HV_TEST_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            sprintf(theFileName2, "OUTPUT/STD_HV_TEST_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_mean_std(theFileName, theFileName2, st_indicator_p.mat_HV_all_TEST, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //MEAN & STD --- HV
            sprintf(theFileName, "OUTPUT/MEAN_HV_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            sprintf(theFileName2, "OUTPUT/STD_HV_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_mean_std(theFileName, theFileName2, st_indicator_p.mat_HV_all, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //vec_NTRACE_all_VALIDATION
            sprintf(theFileName, "OUTPUT/NTRACE_all_VALIDATION_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_vec_int(theFileName, st_indicator_p.vec_NTRACE_all_VALIDATION, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //time
            sprintf(theFileName, "OUTPUT/TIME_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_vec_double_with_mean(theFileName, st_indicator_p.vec_TIME_all, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //time grouping
            sprintf(theFileName, "OUTPUT/TIME_GROUPING_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_vec_double_with_mean(theFileName, st_indicator_p.vec_TIME_grouping, num_run, NTRACE, prob, nobj, ndim);
            //////////////////////////////////////////////////////////////////////////
            //time indicator && saving
            sprintf(theFileName, "OUTPUT/TIME_INDICATOR_SAVING_%s_MPI%d_%ld.csv", st_global_p.algorithmName, st_MPI_p.mpi_size,
                    st_ctrl_p.global_time);
            output_csv_vec_double_with_mean(theFileName, st_indicator_p.vec_TIME_indicator, num_run, NTRACE, prob, nobj, ndim);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    fclose(readf);
    //////////////////////////////////////////////////////////////////////////
    MPI_Finalize();
    return 0;
}
