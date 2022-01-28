#ifndef _OUTPUT_CSV_H__
#define _OUTPUT_CSV_H__

void output_results_to_csv_files(int the_rank, int nExp, int iRun, int iTrace, int tag_prob,
                                 char* _algName, char* _probName, int numObj, int numVar, long curTime,
                                 double* IGDs_TRAIN, double* IGDs_TEST, double* IGDs_all,
                                 double* HVs_TRAIN, double* HVs_TEST, double* HVs_all,
                                 double TIMEs_all);

#endif
