#ifndef _MOP_CLASSIFY_
#define _MOP_CLASSIFY_

#include <mpi.h>
/////////////////////////////////////////////////////////////////////////
//Classify
#define NOBJ_CLASSIFY 3
#define TH_N_FEATURE  50

//weight or not
//#define WEIGHT_ENCODING

extern int      N_FEATURE;
extern int      DIM_ClassifierFunc;
extern double** filterWeights;
extern double** optimizeData;
extern int      N_sample_optimize;
extern int* rank_INDX;

void   Initialize_ClassifierFunc(char prob[], int curN, int numN);
void   SetLimits_ClassifierFunc(double* minLimit, double* maxLimit, int nx);
void   Fitness_ClassifierFunc(double* individual, double* fitness, double* constrainV, int nx, int M);
void   Fitness_ClassifierFunc(int* individual, double* fitness);
void   Fitness_ClassifierFunc(double* individual, double* fitness,
                              MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);
int    CheckLimits_ClassifierFunc(double* x, int nx);
void   freeMemoryCLASS();
void   filter_ReliefF();
double featureCorrelation2(int iFeat, int jFeat);
void   testAccuracy(double* individual, double* fitness);
void   testAccuracy(int* individual, double* fitness);
void   testAccuracy(double* individual, double* fitness,
                    MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);
void   testAccuracy(int* individual, double* fitness,
                    MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);

#endif
