#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "MOP_Classify.h"

//#include <tchar.h>

//macro - opType
#define BOOTSTRAP     0 // .632 bootstrap
#define BOOTSTRAPplus 1 // .632+ bootstrap
#define MULTIFOLD     2 // ???????
#define RATIO         3 // ????????

#define MAX_BUF_SIZE  1024 //

//
int    N_CLASS;
int    N_MODEL;
int    N_FEATURE;
int    ind_class;
int    DIM_ClassifierFunc;
int    opNum;
int    opNumCur;
int    labelCLASS[256];//?????????
double ratioLabelCLASS[256];
int    arrSize[256];

//global variables in this cpp file
double** wholeData;
int** flag_wholeData;
int      N_row_whole_data;
int      N_col_whole_data;
int      N_sample_whole;
double** wholeDataMinMax;

double** optimizeData;
int** optimizeIdx;
int      N_sample_optimize;
int** optimizeDataDiscTag;

double** trainData;
int** trainIdx;
int      N_sample_train;

double** validationData;
int** validationIdx;
int      N_sample_validation;

double** testData;
int** testIdx;
int      N_sample_test;

double** filterWeights;
double** filterDataMinMax;
int* rank_INDX;

double** corrMatrix;
double** corrData;
int      N_sample_corr;
double* meanFeature;
int** selectionFlag;

int      innerTag = 0;
int      filterTag = 0;
int      corrTag = 0;

//stdout = freopen("out.txt", "w", stdout);

const char testInstNames[25][128] = {
    "ALLGSE412_poterapiji",
    "ALLGSE412_pred_poTh",
    "AMLGSE2191",
    "BC_CCGSE3726_frozen",
    "BCGSE349_350",
    "bladderGSE89",
    "braintumor",
    "CMLGSE2535",
    "DLBCL",
    "EWSGSE967",
    "EWSGSE967_3class",
    "gastricGSE2685",
    "gastricGSE2685_2razreda",
    "glioblastoma",
    "leukemia",
    "LL_GSE1577",
    "LL_GSE1577_2razreda",
    "lung",
    "lungGSE1987",
    "meduloblastomiGSE468",
    "MLL",
    "prostata",
    "prostateGSE2443",
    "SRBCT",
    "SECOM"
};

const int numROW[25] = {
    60,  110,   54,   52,   24,   40,
    40,   28,   77,   23,   23,   30,
    30,   50,   72,   29,   19,  203,
    34,   23,   72,  102,   20,   83,
    1567
};

const int numCOL[25] = {
    8281,   8281, 12626, 22284,
    12626,  5725,  7130, 12626,
    7071,   9946,  9946,  4523,
    4523,  12626,  5148, 15435,
    15435, 12601, 10542,  1466,
    12534, 12534, 12628,  2309,
    591
};

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define IM1_CLASS 2147483563
#define IM2_CLASS 2147483399
#define AM_CLASS (1.0/IM1_CLASS)
#define IMM1_CLASS (IM1_CLASS-1)
#define IA1_CLASS 40014
#define IA2_CLASS 40692
#define IQ1_CLASS 53668
#define IQ2_CLASS 52774
#define IR1_CLASS 12211
#define IR2_CLASS 3791
#define NTAB_CLASS 32
#define NDIV_CLASS (1+IMM1_CLASS/NTAB_CLASS)
#define EPS_CLASS 1.2e-7
#define RNMX_CLASS (1.0-EPS_CLASS)

//the random generator in [0,1)
double rnd_uni_CLASS(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_CLASS];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_CLASS + 7; j >= 0; j--) {
            k = (*idum) / IQ1_CLASS;
            *idum = IA1_CLASS * (*idum - k * IQ1_CLASS) - k * IR1_CLASS;
            if(*idum < 0) *idum += IM1_CLASS;
            if(j < NTAB_CLASS) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_CLASS;
    *idum = IA1_CLASS * (*idum - k * IQ1_CLASS) - k * IR1_CLASS;
    if(*idum < 0) *idum += IM1_CLASS;
    k = idum2 / IQ2_CLASS;
    idum2 = IA2_CLASS * (idum2 - k * IQ2_CLASS) - k * IR2_CLASS;
    if(idum2 < 0) idum2 += IM2_CLASS;
    j = iy / NDIV_CLASS;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_CLASS;   //printf("%lf\n", AM_CLASS*iy);
    if((temp = AM_CLASS * iy) > RNMX_CLASS) return RNMX_CLASS;
    else return temp;
}/*------End of rnd_uni_CLASS()--------------------------*/
int     seed_CLASS = 237;
long    rnd_uni_init_CLASS = -(long)seed_CLASS;

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
void   getSizeCLASS(char filename[], int& m, int& n);
double** allocDoubleCLASS(int m, int n);
int** allocIntCLASS(int m, int n);
void readDataCLASS(double**& MEM, int**& flag_MEM, char filename[], int& m, int& n);

void   splitData_outer(int opType, double ratio);
void   LOOCV_split(int* arrIDX, int nFeature, int opNum, int opNumCur);
void   classifierExecute(double* individual, int& nCorrectValidat, int& nNumValidat);
void classifierExecute2(int nFeature, int* cur_TP, int* cur_FP, int* cur_TN, int* cur_FN);
//void   Initialize_ClassifierFunc(char prob[], int curN, int numN); //////////////////////////////////////////////////////////////////
//void   SetLimits_ClassifierFunc(double* minLimit, double* maxLimit); //////////////////////////////////////////////////////////////////
//void   Fitness_ClassifierFunc(double* individual, double* fitness); //////////////////////////////////////////////////////////////////
//void   Fitness_ClassifierFunc(int* individual, double* fitness); //////////////////////////////////////////////////////////////////
//void   Fitness_ClassifierFunc(double* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);
//int    checkLimits_ClassifierFunc(double* x); //////////////////////////////////////////////////////////////////
void   freeMemoryCLASS();
void   filter_ReliefF();
void   featureCorrelation();
double featureCorrelation2(int iFeat, int jFeat);
void   testAccuracy(double* individual, double* fitness); //////////////////////////////////////////////////////////////////
void   testAccuracy(int* individual, double* fitness); //////////////////////////////////////////////////////////////////
void   testAccuracy(double* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);
void   testAccuracy(int* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species);
void   bootstrapInitialize(int**& index);
void   multifoldInitialize(int**& index);

//
void  discreteData();
double calcSymmetricalUncertainty(int indOne, int indTwo);

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void getSizeCLASS(char filename[], int& m, int& n)
{
    FILE* fpt = fopen(filename, "r");

    if(fpt) {
        char buf[MAX_BUF_SIZE];
        char* p;
        m = 0;
        n = 0;

        while(fgets(buf, MAX_BUF_SIZE, fpt)) {
            if(n == 0) {
                for(p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
                    n++;
                }
            } else {
                int tmp_n = 0;
                for(p = strtok(buf, " \t\r\n"); p; p = strtok(NULL, " \t\r\n")) {
                    tmp_n++;
                }
                if(n != tmp_n) {
                    printf("File %s error, column number not consistant, exiting...\n", filename);
                    exit(-1);
                }
            }
            m++;
        }
        // fgets(buf, MAX_BUF_SIZE, fpt);   //???????快???快?buf??
        // while (!feof(fpt)) {
        //     m++;//????fgets??
        //     fgets(buf, MAX_BUF_SIZE, fpt);   //???????快???快?buf??
        // }
        fclose(fpt);
    } else {
        printf("Open file %s error, exiting...\n", filename);
        exit(-1);
    }

    return;
}

double** allocDoubleCLASS(int m, int n)
{
    double** tmp;
    if((tmp = (double**)calloc(m, sizeof(double*))) == NULL) {
        printf("ERROR!! --> calloc: no memory for matrix\n");
        exit(1);
    }
    for(int i = 0; i < m; i++) {
        if((tmp[i] = (double*)calloc(n, sizeof(double))) == NULL) {
            printf("ERROR!! --> calloc: no memory for vector\n");
            exit(1);
        }
    }
    return tmp;
}

int** allocIntCLASS(int m, int n)
{
    int** tmp;
    if((tmp = (int**)calloc(m, sizeof(int*))) == NULL) {
        printf("ERROR!! --> calloc: no memory for matrix\n");
        exit(1);
    }
    for(int i = 0; i < m; i++) {
        if((tmp[i] = (int*)calloc(n, sizeof(int))) == NULL) {
            printf("ERROR!! --> calloc: no memory for vector\n");
            exit(1);
        }
    }
    return tmp;
}

void Initialize_ClassifierFunc(char prob[], int curN, int numN)
{
    seed_CLASS = 237;
    rnd_uni_init_CLASS = -(long)seed_CLASS;
    //for(int i = 0; i < curN; i++) {
    //    seed_CLASS = (seed_CLASS + 111) % 1235;
    //    rnd_uni_init_CLASS = -(long)seed_CLASS;
    //}

    opNumCur = curN;
    opNum = numN;

    char filename[1024];
    //strcpy(filename, prob);
    char tmpSTR0[256];
    char tmpSTR[256];
    sscanf(prob, "%[A-Za-z_]%[0-9]", tmpSTR0, tmpSTR);
    int tmpINT;
    sscanf(tmpSTR, "%d", &tmpINT);
    tmpINT--;
    sprintf(filename, "../Data_all/Data_FeatureSelection/%s", testInstNames[tmpINT]);

    ///////////////////////////////////////////////////////////////////////////////
    //READING DATA
    ///////////////////////////////////////////////////////////////////////////////
    N_row_whole_data = numROW[tmpINT];
    N_col_whole_data = numCOL[tmpINT];
    readDataCLASS(wholeData, flag_wholeData, filename, N_row_whole_data, N_col_whole_data);
    for(int i = 0; i < 25; i++) {
        if(!strcmp(testInstNames[i], testInstNames[tmpINT])) {
            if(N_row_whole_data != numROW[i] || N_col_whole_data != numCOL[i]) {
                printf("%s(%d): READING FILE ERROR, size (%d,%d) is wrong, should be (%d,%d), exiting...\n",
                       __FILE__, __LINE__, N_row_whole_data, N_col_whole_data, numROW[i], numCOL[i]);
                exit(90909);
            }
        }
    }
    N_sample_whole = N_row_whole_data;
    N_FEATURE = N_col_whole_data - 1;
    ind_class = N_FEATURE;
    //for (int i = 0; i < M_whole_data; i++) printf("%lf ", wholeData[i][N_FEATURE]);

    //??????????(-999999999.876)???t?????
    for(int i = 0; i < N_FEATURE; i++) {
        double sum = 0.0;
        int count = 0;
        for(int j = 0; j < N_sample_whole; j++) {
            if(flag_wholeData[j][i]) {
                sum += wholeData[j][i];
                count++;
            }
        }
        if(count < N_sample_whole && count) {
            sum /= count;
            for(int j = 0; j < N_sample_whole; j++) {
                if(!flag_wholeData[j][i]) {
                    wholeData[j][i] = sum;
                }
            }
        }
    }

    N_CLASS = 0;
    labelCLASS[N_CLASS++] = (int)wholeData[0][ind_class];
    for(int i = 1; i < N_sample_whole; i++) {
        int flag = 1;
        for(int j = 0; j < N_CLASS; j++) {
            if(labelCLASS[j] == wholeData[i][ind_class])
                flag = 0;
        }
        if(flag) {
            labelCLASS[N_CLASS++] = (int)wholeData[i][ind_class];
        }
        //printf("%lf ", wholeData[i, N_FEATURE]);
    }
    for(int i = 0; i < N_CLASS; i++) {
        arrSize[i] = 0;
    }
    for(int j = 0; j < N_row_whole_data; j++) {
        for(int i = 0; i < N_CLASS; i++)
            if((int)wholeData[j][ind_class] == labelCLASS[i])
                arrSize[i]++;
    }

    //for (int i = 0; i < N_CLASS; i++)
    //  cout << labelCLASS[i];
    //cout << endl;

    N_MODEL = 1;// only kNN

#ifdef WEIGHT_ENCODING
    DIM_ClassifierFunc = N_FEATURE + 1;
#else
    DIM_ClassifierFunc = N_FEATURE;
#endif

    ///////////////////////////////////////////////////////////////////////////////
    //DATA SAVING
    ///////////////////////////////////////////////////////////////////////////////
    //sprintf(filename, "FILES%s_ori", prob);
    //FILE* fpt1 = fopen(filename, "w");

    //for(int i = 0; i < N_row_whole_data; i++) {
    //    for(int j = 0; j < N_FEATURE; j++) {
    //        if(fabs(wholeData[i][j]) > 1) {
    //            int test = 0;
    //        }
    //        fprintf(fpt1, "%d:%e", j + 1, wholeData[i][j]);
    //        if(j == N_FEATURE - 1) {
    //            fprintf(fpt1, "\n");
    //        } else {
    //            fprintf(fpt1, " ");
    //        }
    //    }
    //}
    //fclose(fpt1);

    ///////////////////////////////////////////////////////////////////////////////
    //DATA NORMALIZATION
    ///////////////////////////////////////////////////////////////////////////////
    wholeDataMinMax = allocDoubleCLASS(N_FEATURE, 2);
    for(int i = 0; i < N_FEATURE; i++) {
        wholeDataMinMax[i][0] = 1e308;
        wholeDataMinMax[i][1] = -1e308;
    }
    double tmp;
    for(int i = 0; i < N_row_whole_data; i++) {
        for(int j = 0; j < N_FEATURE; j++) {
            tmp = wholeData[i][j];
            if(tmp < wholeDataMinMax[j][0]) {
                wholeDataMinMax[j][0] = tmp;
            }
            if(tmp > wholeDataMinMax[j][1]) {
                wholeDataMinMax[j][1] = tmp;
            }
        }
    }
    for(int i = 0; i < N_row_whole_data; i++) {
        for(int j = 0; j < N_FEATURE; j++) {
            if(wholeDataMinMax[j][1] == wholeDataMinMax[j][0]) {
                wholeData[i][j] = 0;
            } else {
                wholeData[i][j] = (wholeData[i][j] - wholeDataMinMax[j][0]) /
                                  (wholeDataMinMax[j][1] - wholeDataMinMax[j][0]);
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////
    //DATA SAVING
    ///////////////////////////////////////////////////////////////////////////////
    //sprintf(filename, "%s", prob);
    //FILE* fpt = fopen(filename, "w");

    //for(int i = 0; i < N_row_whole_data; i++) {
    //    for(int j = 0; j < N_FEATURE; j++) {
    //        fprintf(fpt, "%d:%e", j + 1, wholeData[i][j]);
    //        if(j == N_FEATURE - 1) {
    //            fprintf(fpt, "\n");
    //        } else {
    //            fprintf(fpt, " ");
    //        }
    //    }
    //}
    //fclose(fpt);

    //printf("88888-%lf\n", wholeData[7][0]);

    //// #define BOOTSTRAP     0 // .632 bootstrap
    //// #define BOOTSTRAPplus 1 // .632+ bootstrap
    //// #define MULTIFOLD     2 // ???????
    //// #define RATIO         3 // ????????
    if(tmpINT == 24)
        splitData_outer(MULTIFOLD, 0.7);
    else
        splitData_outer(BOOTSTRAP, 0.7);

    filter_ReliefF();
    discreteData();
    //featureCorrelation();

    return;
}

void SetLimits_ClassifierFunc(double* minLimit, double* maxLimit, int nx)
{
    for(int i = 0; i < DIM_ClassifierFunc; i++) {
        minLimit[i] = 0.0;
        maxLimit[i] = 2.0 - 1e-6;
    }

    return;
}

int CheckLimits_ClassifierFunc(double* x, int nx)
{
    for(int i = 0; i < DIM_ClassifierFunc; i++) {
        if(x[i] < 0.0 || x[i] > 2.0 - 1e-6) {
            return false;
        }
    }

    return true;
}

void readDataCLASS(double**& MEM, int**& flag_MEM, char filename[], int& m, int& n)
{
    //printf("%s\n", filename);
    //getSizeCLASS(filename, m, n);
    MEM = allocDoubleCLASS(m, n);
    flag_MEM = allocIntCLASS(m, n);
    FILE* fpt;
    fpt = fopen(filename, "r");
    if(!fpt) {
        printf("\n%s(%d): Load file failed", __FILE__, __LINE__);
        exit(-1);
    }
    int i, j;
    char tmp_str[1024];
    for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
            int tmp = fscanf(fpt, "%s", tmp_str); //printf("%lf ", MEM[i][j]);
            if(tmp == EOF) {
                printf("\n%s(%d):weights are not enough...\n", __FILE__, __LINE__);
                exit(9);
            }
            if(!strcmp(tmp_str, "NaN")) {
                flag_MEM[i][j] = 0;
                MEM[i][j] = 0.0;
            } else {
                flag_MEM[i][j] = 1;
                MEM[i][j] = atof(tmp_str);
                if(MEM[i][j] == -999999999.876) {
                    flag_MEM[i][j] = 0;
                    MEM[i][j] = 0.0;
                }
            }
        }
    }
    fclose(fpt);
    //for (i = 0; i < m; i++) {
    //  for (j = 0; j < n && j < 9; j++) {
    //      printf("%lf ", MEM[i][j]);
    //  }
    //  printf("\n");
    //}

    return;
}

//void saveMat(Mat inputMat, char* filename)
//{
//  FILE* fpt = fopen(filename, "w");
//  for (int i = 0; i < inputMat.rows; i++) {
//      for (int j = 0; j < inputMat.cols; j++) {
//          if (j < inputMat.cols - 1)
//              fprintf(fpt, "%f,", inputMat.at<float>(i, j));
//          else
//              fprintf(fpt, "%f\n", inputMat.at<float>(i, j));
//      }
//  }
//  fclose(fpt);
//}

void filter_ReliefF()
{
    if(filterTag) {
        for(int i = 0; i < N_FEATURE; i++) {
            free(filterDataMinMax[i]);
            free(filterWeights[i]);
        }
        free(filterDataMinMax);
        free(filterWeights);
        free(rank_INDX);

        filterTag = 0;
    }

    filterWeights = allocDoubleCLASS(N_FEATURE, 1);
    for(int i = 0; i < N_FEATURE; i++) {
        filterWeights[i][0] = 0.0;
    }
    //////////////////
    filterDataMinMax = allocDoubleCLASS(N_FEATURE, 2);
    for(int i = 0; i < N_FEATURE; i++) {
        filterDataMinMax[i][0] = 1e308;
        filterDataMinMax[i][1] = -1e308;
    }
    for(int i = 0; i < N_CLASS; i++) {
        ratioLabelCLASS[i] = 0.0;
    }
    for(int i = 0; i < N_sample_optimize; i++) {
        for(int j = 0; j < N_CLASS; j++) {
            if((int)optimizeData[i][N_FEATURE] == labelCLASS[j]) {
                ratioLabelCLASS[j] += 1.0;
            }
        }
        for(int j = 0; j < N_FEATURE; j++) {
            if(filterDataMinMax[j][0] > optimizeData[i][j])
                filterDataMinMax[j][0] = optimizeData[i][j];
            if(filterDataMinMax[j][1] < optimizeData[i][j])
                filterDataMinMax[j][1] = optimizeData[i][j];
        }
    }
    for(int i = 0; i < N_CLASS; i++) {
        ratioLabelCLASS[i] /= N_sample_optimize;
        if(ratioLabelCLASS[i] == 0) {
            printf("%s(%d): ZERO ERROR! - No samples optimized for class %d\n", __FILE__, __LINE__, i + 1);
            exit(-100);
        }
    }
    //////////////////
    const int K = (N_sample_optimize > 10) ? 10 : (N_sample_optimize);
    double** dists = allocDoubleCLASS(N_CLASS, K);
    int** indices = allocIntCLASS(N_CLASS, K);
    //////////////////
    for(int i = 0; i < N_sample_optimize; i++) {
        int idx = (int)(rnd_uni_CLASS(&rnd_uni_init_CLASS) * N_sample_optimize);
        int curClass = -1;
        for(int n = 0; n < N_CLASS; n++) {
            if((int)optimizeData[idx][N_FEATURE] == labelCLASS[n]) {
                curClass = n;
            }
        }
        if(curClass == -1) {
            printf("%s(%d): index 'tmpClass' is -1, ERROR, ... \n", __FILE__, __LINE__);
            exit(-111111);
        }
        for(int j = 0; j < N_CLASS; j++) {
            for(int k = 0; k < K; k++) {
                dists[j][k] = -1;
                indices[j][k] = -1;
            }
        }
        for(int j = 0; j < N_sample_optimize; j++) {
            double d = 0;
            for(int k = 0; k < N_FEATURE; k++) {
                d += (optimizeData[idx][k] - optimizeData[j][k]) * (optimizeData[idx][k] - optimizeData[j][k]);
            }
            int tmpClass = -1;
            for(int k = 0; k < N_CLASS; k++) {
                if((int)optimizeData[j][N_FEATURE] == labelCLASS[k]) {
                    tmpClass = k;
                }
            }
            if(tmpClass == -1) {
                printf("%s(%d): index 'tmpClass' is -1, ERROR, ... \n", __FILE__, __LINE__);
                exit(-111112);
            }
            int tmpIdx = -1;
            for(int k = 0; k < K; k++) {
                if(dists[tmpClass][k] < 0) {
                    tmpIdx = k;
                    break;
                }
                if(dists[tmpClass][k] > d) {
                    if(tmpIdx == -1) {
                        tmpIdx = k;
                    } else {
                        if(dists[tmpClass][k] > dists[tmpClass][tmpIdx]) {
                            tmpIdx = k;
                        }
                    }
                }
            }
            if(tmpIdx >= 0) {
                dists[tmpClass][tmpIdx] = d;
                indices[tmpClass][tmpIdx] = j;
            }
        }
        ////////////////////////////////
        for(int j = 0; j < N_FEATURE; j++) {
            // double sum1 = 0.0;
            // double sum2 = 0.0;
            double* allSUM = (double*)calloc(N_CLASS, sizeof(double));
            int* tmpCount = (int*)calloc(N_CLASS, sizeof(int));
            for(int m = 0; m < N_CLASS; m++) {
                allSUM[m] = 0.0;
                tmpCount[m] = 0;
            }
            for(int m = 0; m < N_CLASS; m++) {
                for(int k = 0; k < K; k++) {
                    if(indices[m][k] >= 0) {
                        double tmp = optimizeData[idx][j] - optimizeData[indices[m][k]][j];
                        if(filterDataMinMax[j][1] == filterDataMinMax[j][0]) {
                            tmp = 0.0;
                        } else {
                            tmp /= (filterDataMinMax[j][1] - filterDataMinMax[j][0]);
                        }
                        if(tmp < 0.0) tmp = -tmp;
                        if((int)optimizeData[indices[m][k]][N_FEATURE] == labelCLASS[curClass]) {
                            allSUM[curClass] += tmp;
                            tmpCount[curClass]++;
                        } else {
                            int tmpClass = -1;
                            for(int n = 0; n < N_CLASS; n++) {
                                if((int)optimizeData[indices[m][k]][N_FEATURE] == labelCLASS[n]) {
                                    tmpClass = n;
                                }
                            }
                            if(tmpClass == -1) {
                                printf("%s(%d): index 'tmpClass' is -1, ERROR, ... \n", __FILE__, __LINE__);
                                exit(-111113);
                            }
                            allSUM[tmpClass] += ratioLabelCLASS[tmpClass] / (1 - ratioLabelCLASS[curClass]) * tmp;
                            tmpCount[tmpClass]++;
                        }
                    } else {
                        //printf("%s(%d): index of one neighbor is -1, ERROR, ... \n", __FILE__, __LINE__);
                        //exit(-111114);
                    }
                }
            }
            //filterWeights[j][0] = 0.0;
            //filterWeights[j][0] = filterWeights[j][0] - sum1 / (N_sample_optimize*K) + sum2 / (N_sample_optimize*K);
            for(int m = 0; m < N_CLASS; m++) {
                if(m == curClass) {
                    filterWeights[j][0] -= allSUM[m] / (N_sample_optimize * tmpCount[m]);
                } else {
                    filterWeights[j][0] += allSUM[m] / (N_sample_optimize * tmpCount[m]);
                }
            }
            free(allSUM);
            free(tmpCount);
        }
    }

    rank_INDX = (int*)calloc(N_FEATURE, sizeof(int));
    for(int i = 0; i < N_FEATURE; i++) {
        rank_INDX[i] = i;
    }
    int tmp;
    for(int i = 0; i < N_FEATURE; i++) {
        for(int j = i + 1; j < N_FEATURE; j++) {
            if(filterWeights[rank_INDX[i]][0] < filterWeights[rank_INDX[j]][0]) {
                tmp = rank_INDX[i];
                rank_INDX[i] = rank_INDX[j];
                rank_INDX[j] = tmp;
            }
        }
        //printf("%.16e --- %6d\n", filterWeights[rank_INDX[i]][0], rank_INDX[i]);
    }

    for(int i = 0; i < N_CLASS; i++) {
        free(dists[i]);
        free(indices[i]);
    }
    free(dists);
    free(indices);
    //for (int i = 0; i < N_sample_whole; i++){
    //  free(selectionFlag[i]);
    //}
    //free(selectionFlag);

    filterTag = 1;

    return;
}

void discreteData()
{
    optimizeDataDiscTag = allocIntCLASS(N_sample_optimize, N_col_whole_data);

    return;
}

//
//  wholeData;
//  optimizeData;
//  testData;
// int opNum, int opNumCur --- ????????? ?? ???????????? ??????????
void splitData_outer(int opType, double ratio)
{
    //free
    //for (int i = 0; i < N_sample_optimize; i++){
    //  if (optimizeData[i]) free(optimizeData[i]);
    //  if (optimizeIdx[i]) free(optimizeIdx[i]);
    //}
    //if (optimizeData) free(optimizeData);
    //if (optimizeIdx) free(optimizeIdx);
    //for (int i = 0; i < N_sample_test; i++){
    //  if (testData[i]) free(testData[i]);
    //  if (testIdx[i]) free(testIdx[i]);
    //}
    //if (testData) free(testData);
    //if (testIdx) free(testIdx);

    //seed_CLASS = (seed_CLASS + 111) % 1235;
    //rnd_uni_init_CLASS = -(long)seed_CLASS;
    //rnd_uni_CLASS(&rnd_uni_init_CLASS)

    int count = 0;
    int count2 = 0;
    int testSize;
    int optimizeSize;
    switch(opType) {
    case BOOTSTRAP:
        //????????????????
        N_sample_optimize = N_row_whole_data;
        optimizeData = allocDoubleCLASS(N_sample_optimize, N_col_whole_data);
        optimizeIdx = allocIntCLASS(N_sample_optimize, 1);
        bootstrapInitialize(optimizeIdx);
        //??????????????
        selectionFlag = allocIntCLASS(N_sample_optimize, 1);
        for(int i = 0; i < N_sample_optimize; i++)
            selectionFlag[i][0] = 0;
        count = N_sample_optimize;
        //?戒?????????
        for(int i = 0; i < N_row_whole_data; i++) {
            memcpy(optimizeData[i], wholeData[optimizeIdx[i][0]], N_col_whole_data * sizeof(double));
            if(0 == selectionFlag[optimizeIdx[i][0]][0]) {
                count--;
                selectionFlag[optimizeIdx[i][0]][0] = 1;
            }
            //printf("%d-%d ", (int)wholeData[optimizeIdx[i][0]][DIM_ClassifierFunc], (int)optimizeData[i][DIM_ClassifierFunc]);
        }
        //汛?????????????????
        N_sample_test = count;
        if(N_sample_test == 0) {
            printf("%s(%d): the number of test instances is ZEROS for run - %d, exiting...\n", __FILE__, __LINE__, opNumCur);
            exit(-100);
        }
        testData = allocDoubleCLASS(N_sample_test, N_col_whole_data);
        testIdx = allocIntCLASS(N_sample_test, 1);
        count = 0;
        for(int i = 0; i < N_row_whole_data; i++) {
            if(0 == selectionFlag[i][0]) {
                memcpy(testData[count], wholeData[i], N_col_whole_data * sizeof(double));
                testIdx[count][0] = i;
                count++;
            }
        }
        for(int i = 0; i < N_sample_optimize; i++) free(selectionFlag[i]);
        free(selectionFlag);
        //for (int i = 0; i < class_mpi_size; i++){
        //  if (class_mpi_rank == i){
        //      printf("%d-TRAIN ", i);
        //      for (int j = 0; j < M_whole_data; j++){
        //          printf("%2d ", optimizeIdx[j][0]);
        //      }
        //      printf("\n");
        //      printf("%d-TESTT ", i);
        //      for (int j = 0; j < N_sample_test; j++){
        //          printf("%2d ", testIdx[j][0]);
        //      }
        //      printf("\n");
        //  }
        //  MPI_Barrier(MPI_COMM_WORLD);
        //}
        break;

    case BOOTSTRAPplus:
        printf("Unavailable currently, EXITING...\n");
        exit(-77);
        break;

    case MULTIFOLD:
        //?????opNum
        if(opNumCur < 0) {
            printf("opNumCur should not be negative, exiting!!!\n");
            exit(-111);
        }
        if(opNumCur >= opNum) {
            printf("opNumCur %d should be less than opNum %d\n", opNumCur, opNum);
            exit(-111);
        }
        //????????????????????????
        testSize = N_row_whole_data / opNum;
        if(opNumCur < N_row_whole_data % opNum) testSize++;
        N_sample_optimize = N_row_whole_data - testSize;
        optimizeData = allocDoubleCLASS(N_sample_optimize, N_col_whole_data);
        optimizeIdx = allocIntCLASS(N_sample_optimize, 1);
        N_sample_test = testSize;
        testData = allocDoubleCLASS(N_sample_test, N_col_whole_data);
        testIdx = allocIntCLASS(N_sample_test, 1);
        multifoldInitialize(optimizeIdx);
        //?????????????????
        selectionFlag = allocIntCLASS(N_row_whole_data, 1);
        //??????
        for(int i = 0; i < N_sample_optimize; i++) {
            memcpy(optimizeData[i], wholeData[optimizeIdx[i][0]], N_col_whole_data * sizeof(double));
            selectionFlag[optimizeIdx[i][0]][0] = 1;
            //printf("%d-%d ", (int)wholeData[optimizeIdx[i][0]][DIM_ClassifierFunc], (int)optimizeData[i][DIM_ClassifierFunc]);
        }
        count = 0;
        for(int i = 0; i < N_row_whole_data; i++) {
            if(0 == selectionFlag[i][0]) {
                memcpy(testData[count], wholeData[i], N_col_whole_data * sizeof(double));
                testIdx[count][0] = i;
                count++;
            }
        }
        //
        for(int i = 0; i < N_sample_optimize; i++) free(selectionFlag[i]);
        free(selectionFlag);
        break;

    case RATIO:
        // double ratio --- ?????????????????
        if(ratio <= 0.0 || ratio >= 1.0) {
            printf("ratio should be in (0.0,1.0), but not %lf\n", ratio);
            exit(-222);
        }
        //????????????????
        optimizeSize = 0;
        for(int i = 0; i < N_CLASS; i++) {
            arrSize[i] = 0;
            for(int j = 0; j < N_row_whole_data; j++) {
                if((int)wholeData[j][ind_class] == labelCLASS[i])
                    arrSize[i]++;
            }
            int tmp = (int)(arrSize[i] * ratio);
            if(tmp == 0) tmp = 1;
            optimizeSize += tmp;
        }
        //?????
        N_sample_optimize = optimizeSize;
        optimizeData = allocDoubleCLASS(N_sample_optimize, N_col_whole_data);
        optimizeIdx = allocIntCLASS(N_sample_optimize, 1);
        N_sample_test = N_row_whole_data - optimizeSize;
        testData = allocDoubleCLASS(N_sample_test, N_col_whole_data);
        testIdx = allocIntCLASS(N_sample_test, 1);
        //??????????????
        selectionFlag = allocIntCLASS(N_sample_whole, 1);
        for(int i = 0; i < N_sample_whole; i++)
            selectionFlag[i][0] = 0;
        //??????
        for(int p = 0; p < N_CLASS; p++) {
            count = 0;
            int tmp = (int)(arrSize[p] * ratio);
            if(tmp == 0) tmp = 1;
            while(count < tmp) {
                int idx = (int)(rnd_uni_CLASS(&rnd_uni_init_CLASS) * N_sample_whole);
                if((int)wholeData[idx][ind_class] == labelCLASS[p] && selectionFlag[idx][0] == 0) {
                    memcpy(optimizeData[count], wholeData[idx], N_col_whole_data * sizeof(double));
                    optimizeIdx[count][0] = idx;
                    count++;
                    selectionFlag[idx][0] = 1;
                }
            }
        }
        count = 0;
        for(int i = 0; i < N_row_whole_data; i++) {
            if(selectionFlag[i][0] == 0) {
                memcpy(testData[count], wholeData[i], N_col_whole_data * sizeof(double));
                testIdx[count][0] = i;
                count++;
            }
        }
        //for (int i = 0; i < N_sample_test; i++){
        //  for (int j = 0; j < N_whole_data; j++){
        //      printf("%lf ", testData[i][j]);
        //  }
        //  printf("\n");
        //}
        for(int i = 0; i < N_sample_whole; i++) free(selectionFlag[i]);
        free(selectionFlag);
        break;

    default:
        printf("%s(%d): INVALID OP TYPE, EXITING...\n", __FILE__, __LINE__);
        exit(-77);
        break;
    }

    return;
}

void LOOCV_split(int* arrIDX, int nFeature, int i_opNum, int i_opNumCur)
{
    if(innerTag) {
        for(int i = 0; i < N_sample_train; i++) {
            free(trainData[i]);
            free(trainIdx[i]);
        }
        free(trainData);
        free(trainIdx);
        for(int i = 0; i < N_sample_validation; i++) {
            free(validationData[i]);
            free(validationIdx[i]);
        }
        free(validationData);
        free(validationIdx);

        innerTag = 0;
    }

    int count = 0;
    int count2 = 0;
    int validationSize;
    int ccc1;
    int ccc2;

    //????????????????????????
    ccc1 = (int)(N_sample_optimize * (i_opNumCur + 1.0) / i_opNum);
    if(ccc1 > N_sample_optimize) ccc1 = N_sample_optimize;
    ccc2 = (int)(N_sample_optimize * (i_opNumCur + 0.0) / i_opNum);
    validationSize = ccc1 - ccc2;
    //?????????????????
    N_sample_train = N_sample_optimize - validationSize;
    trainData = allocDoubleCLASS(N_sample_train, nFeature + 1);
    trainIdx = allocIntCLASS(N_sample_train, 1);
    N_sample_validation = validationSize;
    validationData = allocDoubleCLASS(N_sample_validation, nFeature + 1);
    validationIdx = allocIntCLASS(N_sample_validation, 1);
    //??????
    count = 0;
    for(int i = 0; i < ccc2; i++) {
        for(int j = 0; j < nFeature; j++) {
            trainData[count][j] = optimizeData[i][arrIDX[j]];
        }
        trainData[count][nFeature] = optimizeData[i][ind_class];
        //printf("%d-%d ", (int)trainData[count][nFeature], (int)optimizeData[i][N_FEATURE]);
        //memcpy(trainData[count], optimizeData[i], nFeature*sizeof(double));
        trainIdx[count][0] = optimizeIdx[i][0];
        count++;
    }
    count2 = 0;
    for(int i = ccc2; i < ccc1; i++) {
        for(int j = 0; j < nFeature; j++) {
            validationData[count2][j] = optimizeData[i][arrIDX[j]];
        }
        validationData[count2][nFeature] = optimizeData[i][ind_class];
        //printf("%d-%d ", (int)validationData[count2][nFeature], (int)optimizeData[i][N_FEATURE]);
        //memcpy(validationData[count2], optimizeData[i], nFeature*sizeof(double));
        validationIdx[count2][0] = optimizeIdx[i][0];
        count2++;
    }
    for(int i = ccc1; i < N_sample_optimize; i++) {
        for(int j = 0; j < nFeature; j++) {
            trainData[count][j] = optimizeData[i][arrIDX[j]];
        }
        trainData[count][nFeature] = optimizeData[i][ind_class];
        //printf("%d-%d ", (int)trainData[count][nFeature], (int)optimizeData[i][N_FEATURE]);
        //memcpy(trainData[count], optimizeData[i], nFeature*sizeof(double));
        trainIdx[count][0] = optimizeIdx[i][0];
        count++;
    }

    //for (int i = 0; i < nFeature; i++){
    //  printf("%15d ", arrIDX[i]);
    //}
    //printf("\n");
    //for (int i = 0; i < N_sample_train; i++){
    //  for (int j = 0; j <= nFeature; j++)
    //      printf("%.8e ", trainData[i][j]);
    //  printf("\n");
    //}
    //for (int i = 0; i < N_sample_validation; i++){
    //  for (int j = 0; j <= nFeature; j++)
    //      printf("%.8e ", validationData[i][j]);
    //  printf("\n");
    //}

    innerTag = 1;

    return;
}

void classifierExecute(double* individual, int& nCorrectValidat, int& nNumValidat)
{
    nNumValidat = N_sample_validation;

    int predictedLabels;

    //kNN - mine
    {
        const int K = 1;
        double dists[K];
        int    labels[K];

        nCorrectValidat = 0;
        for(int i = 0; i < N_sample_validation; i++) {
            //cout << "SMP_IND: " << i << endl;
            for(int j = 0; j < K; j++) {
                dists[j] = -1;
                labels[j] = -1;
            }
            for(int j = 0; j < N_sample_train; j++) {
                double d = 0.0;
                for(int k = 0; k < N_FEATURE; k++) {
                    if((int)individual[k])
                        d += (trainData[j][k] - validationData[i][k]) * (trainData[j][k] - validationData[i][k]);
                }
                int index = -1;
                for(int k = 0; k < K; k++) {
                    if(dists[k] < 0) {  //uninitialized
                        index = k;
                        break;
                    }
                    if(dists[k] > d) {  //greater than d
                        if(index == -1) {
                            index = k;
                        } else {
                            if(dists[k] > dists[index]) {  //the greatest one greater than d
                                index = k;
                            }
                        }
                    }
                }
                if(index >= 0) {
                    dists[index] = d;
                    labels[index] = (int)trainData[index][ind_class];
                }
            }
            double minD = -1;
            int    minLabel = -1;
            for(int k = 0; k < K; k++) {
                if(minD < 0 || dists[k] < minD) {
                    minD = dists[k];
                    minLabel = labels[k];
                }
            }
            //cout << minD << endl;
            predictedLabels = minLabel;
            if(predictedLabels == validationData[i][ind_class])
                nCorrectValidat++;
        }
    }

    //free
    if(innerTag) {
        for(int i = 0; i < N_sample_train; i++) {
            free(trainData[i]);
            free(trainIdx[i]);
        }
        free(trainData);
        free(trainIdx);
        for(int i = 0; i < N_sample_validation; i++) {
            free(validationData[i]);
            free(validationIdx[i]);
        }
        free(validationData);
        free(validationIdx);

        innerTag = 0;
    }

    return;
}

void classifierExecute2(int nFeature, int* cur_TP, int* cur_FP, int* cur_TN, int* cur_FN)
{
    for(int i = 0; i < N_CLASS; i++) {
        cur_TP[i] = 0;
        cur_FP[i] = 0;
        cur_TN[i] = 0;
        cur_FN[i] = 0;
    }

    int cur_label;

    //kNN - mine
    {
        const int K = 1;
        double dists[K];
        int    labels[K];

        for(int i = 0; i < N_sample_validation; i++) {
            //cout << "SMP_IND: " << i << endl;
            for(int j = 0; j < K; j++) {
                dists[j] = -1;
                labels[j] = -1;
            }
            for(int j = 0; j < N_sample_train; j++) {
                double d = 0.0;
                for(int k = 0; k < nFeature; k++) {
                    d += (trainData[j][k] - validationData[i][k]) * (trainData[j][k] - validationData[i][k]);
                }
                int index = -1;
                for(int k = 0; k < K; k++) {
                    if(dists[k] < 0) {  //uninitialized
                        index = k;
                        break;
                    }
                    if(dists[k] > d) {  //greater than d
                        if(index == -1) {
                            index = k;
                        } else {
                            if(dists[k] > dists[index]) {  //seek the greatest one greater than d
                                index = k;
                            }
                        }
                    }
                }
                if(index >= 0) {
                    dists[index] = d;
                    labels[index] = (int)trainData[j][nFeature];
                    //printf("%d---%3d---%d\n", index, j, labels[index]);
                }
            }
            double minD = -1;
            int    minLabel = -1;
            for(int k = 0; k < K; k++) {
                if(minD < 0 || dists[k] < minD) {
                    minD = dists[k];
                    minLabel = labels[k];
                }
            }
            //cout << minD << endl;
            cur_label = minLabel;
            int true_label = (int)validationData[i][nFeature];
            for(int j = 0; j < N_CLASS; j++) {
                if(j == cur_label && j == true_label) cur_TP[j]++;
                if(j == cur_label && j != true_label) cur_FP[j]++;
                if(j != cur_label && j == true_label) cur_FN[j]++;
                if(j != cur_label && j != true_label) cur_TN[j]++;
            }
        }
    }

    //free
    if(innerTag) {
        for(int i = 0; i < N_sample_train; i++) {
            free(trainData[i]);
            free(trainIdx[i]);
        }
        free(trainData);
        free(trainIdx);
        for(int i = 0; i < N_sample_validation; i++) {
            free(validationData[i]);
            free(validationIdx[i]);
        }
        free(validationData);
        free(validationIdx);

        innerTag = 0;
    }

    return;
}

void Fitness_ClassifierFunc(double* individual, double* fitness, double* constrainV, int nx, int M)
{
    int numFeature = 0;
    int* arrIDX = (int*)calloc(N_FEATURE, sizeof(int));

#ifdef WEIGHT_ENCODING
    int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));

    numFeature = (int)(individual[N_FEATURE] * TH_N_FEATURE) + 1;
    if(numFeature > TH_N_FEATURE)
        numFeature = TH_N_FEATURE;

    double tmpMAX = -1.0;
    int    tmpIND = -1;
    for(int i = 0; i < numFeature; i++) {
        tmpMAX = -1.0;
        for(int j = 0; j < N_FEATURE; j++) {
            if(tmpFlag[j] == 0 && individual[j] > tmpMAX) {
                tmpMAX = individual[j];
                tmpIND = j;
            }
        }
        tmpFlag[tmpIND] = 1;
        arrIDX[i] = tmpIND;
    }

    free(tmpFlag);
#else
    for(int i = 0; i < N_FEATURE; i++) {
        if((int)individual[i])
            arrIDX[numFeature++] = i;
    }
#endif

    if(numFeature) {
        int nFold = N_sample_optimize;
        int *sum_TP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_TN = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FN = (int*)calloc(N_CLASS, sizeof(int));
        double f_prec = 0.0;
        for(int i = 0; i < nFold; i++) {
            int* cur_TP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_TN = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FN = (int*)calloc(N_CLASS, sizeof(int));
            // #define BOOTSTRAP     0 // .632 bootstrap
            // #define BOOTSTRAPplus 1 // .632+ bootstrap
            // #define MULTIFOLD     2 // ???????
            // #define RATIO         3 // ????????
            //splitData_inner(MULTIFOLD, nFold, i, 0.7);
            LOOCV_split(arrIDX, numFeature, nFold, i);
            classifierExecute2(numFeature, cur_TP, cur_FP, cur_TN, cur_FN);
            for(int j = 0; j < N_CLASS; j++) {
                sum_TP[j] += cur_TP[j];
                sum_FP[j] += cur_FP[j];
                sum_TN[j] += cur_TN[j];
                sum_FN[j] += cur_FN[j];
            }
            free(cur_TP);
            free(cur_FP);
            free(cur_TN);
            free(cur_FN);
        }
        int tmp1 = 0, tmp2 = 0;
        for(int j = 0; j < N_CLASS; j++) {
            tmp1 += sum_TP[j];
            tmp2 += sum_TP[j] + sum_FP[j];
        }
        f_prec = (double)tmp1 / (tmp1 + tmp2);
        free(sum_TP);
        free(sum_FP);
        free(sum_TN);
        free(sum_FN);
        fitness[0] = f_prec;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    } else {
        fitness[0] = 10.0;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    }
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    free(arrIDX);

    return;
}

void Fitness_ClassifierFunc(int* individual, double* fitness)
{
    int numFeature = 0;
    int arrIDX[TH_N_FEATURE];

    for(int i = 0; i < TH_N_FEATURE; i++) {
        if(individual[i] >= 0)
            arrIDX[numFeature++] = individual[i];
    }

    if(numFeature) {
        int nFold = N_sample_optimize;
        int *sum_TP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_TN = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FN = (int*)calloc(N_CLASS, sizeof(int));
        double f_prec = 0.0;
        for(int i = 0; i < nFold; i++) {
            int* cur_TP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_TN = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FN = (int*)calloc(N_CLASS, sizeof(int));
            // #define BOOTSTRAP     0 // .632 bootstrap
            // #define BOOTSTRAPplus 1 // .632+ bootstrap
            // #define MULTIFOLD     2 // ???????
            // #define RATIO         3 // ????????
            //splitData_inner(MULTIFOLD, nFold, i, 0.7);
            LOOCV_split(arrIDX, numFeature, nFold, i);
            classifierExecute2(numFeature, cur_TP, cur_FP, cur_TN, cur_FN);
            for(int j = 0; j < N_CLASS; j++) {
                sum_TP[j] += cur_TP[j];
                sum_FP[j] += cur_FP[j];
                sum_TN[j] += cur_TN[j];
                sum_FN[j] += cur_FN[j];
            }
            free(cur_TP);
            free(cur_FP);
            free(cur_TN);
            free(cur_FN);
        }
        int tmp1 = 0, tmp2 = 0;
        for(int j = 0; j < N_CLASS; j++) {
            tmp1 += sum_TP[j];
            tmp2 += sum_TP[j] + sum_FP[j];
        }
        f_prec = (double)tmp1 / (tmp1 + tmp2);
        free(sum_TP);
        free(sum_FP);
        free(sum_TN);
        free(sum_FN);
        fitness[0] = f_prec;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    } else {
        fitness[0] = 10.0;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    }
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    return;
}

void Fitness_ClassifierFunc(double* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species,
                            int mpi_size_species)
{
    int numFeature = 0;
    int* arrIDX = (int*)calloc(N_FEATURE, sizeof(int));

    if(mpi_rank_species == 0) {
#ifdef WEIGHT_ENCODING
        int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));

        numFeature = (int)(individual[N_FEATURE] * TH_N_FEATURE) + 1;
        if(numFeature > TH_N_FEATURE)
            numFeature = TH_N_FEATURE;

        double tmpMAX = -1.0;
        int    tmpIND = -1;
        for(int i = 0; i < numFeature; i++) {
            tmpMAX = -1.0;
            for(int j = 0; j < N_FEATURE; j++) {
                if(tmpFlag[j] == 0 && individual[j] > tmpMAX) {
                    tmpMAX = individual[j];
                    tmpIND = j;
                }
            }
            tmpFlag[tmpIND] = 1;
            arrIDX[i] = tmpIND;
        }

        free(tmpFlag);
#else
        for(int i = 0; i < N_FEATURE; i++) {
            if((int)individual[i])
                arrIDX[numFeature++] = i;
        }
#endif
    }

    MPI_Bcast(&numFeature, 1, MPI_INT, 0, comm_species);
    MPI_Bcast(arrIDX, numFeature, MPI_INT, 0, comm_species);

    if(numFeature) {
        int nFold = N_sample_optimize;
        int *sum_TP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FP = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_TN = (int*)calloc(N_CLASS, sizeof(int));
        int *sum_FN = (int*)calloc(N_CLASS, sizeof(int));
        double f_prec = 0.0;
        for(int i = mpi_rank_species; i < nFold; i += mpi_size_species) {
            int* cur_TP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FP = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_TN = (int*)calloc(N_CLASS, sizeof(int));
            int* cur_FN = (int*)calloc(N_CLASS, sizeof(int));
            // #define BOOTSTRAP     0 // .632 bootstrap
            // #define BOOTSTRAPplus 1 // .632+ bootstrap
            // #define MULTIFOLD     2 // ???????
            // #define RATIO         3 // ????????
            //splitData_inner(MULTIFOLD, nFold, i, 0.7);
            LOOCV_split(arrIDX, numFeature, nFold, i);
            classifierExecute2(numFeature, cur_TP, cur_FP, cur_TN, cur_FN);
            for(int j = 0; j < N_CLASS; j++) {
                sum_TP[j] += cur_TP[j];
                sum_FP[j] += cur_FP[j];
                sum_TN[j] += cur_TN[j];
                sum_FN[j] += cur_FN[j];
            }
            free(cur_TP);
            free(cur_FP);
            free(cur_TN);
            free(cur_FN);
        }
        int* all_TP = (int*)calloc(N_CLASS, sizeof(int));
        int* all_FP = (int*)calloc(N_CLASS, sizeof(int));
        int* all_TN = (int*)calloc(N_CLASS, sizeof(int));
        int* all_FN = (int*)calloc(N_CLASS, sizeof(int));

        MPI_Reduce(sum_TP, all_TP, N_CLASS, MPI_INT, MPI_SUM, 0, comm_species);
        MPI_Reduce(sum_FP, all_FP, N_CLASS, MPI_INT, MPI_SUM, 0, comm_species);
        MPI_Reduce(sum_TN, all_TN, N_CLASS, MPI_INT, MPI_SUM, 0, comm_species);
        MPI_Reduce(sum_FN, all_FN, N_CLASS, MPI_INT, MPI_SUM, 0, comm_species);

        int tmp1 = 0, tmp2 = 0;
        for(int j = 0; j < N_CLASS; j++) {
            tmp1 += all_TP[j];
            tmp2 += all_TP[j] + all_FP[j];
        }
        f_prec = (double)tmp1 / (tmp1 + tmp2);
        free(sum_TP);
        free(sum_FP);
        free(sum_TN);
        free(sum_FN);
        free(all_TP);
        free(all_FP);
        free(all_TN);
        free(all_FN);
        fitness[0] = f_prec;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    } else {
        fitness[0] = 10.0;
        fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    }
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    free(arrIDX);

    return;
}

void freeMemoryCLASS()
{
    if(filterTag) {
        for(int i = 0; i < N_FEATURE; i++) {
            //if (fabs(filterDataMinMax[i][0])>1e30 || fabs(filterDataMinMax[i][1]) > 1e30)
            //{
            //  int test = 1;
            //  for (int n = 0; n < N_sample_optimize; n++)
            //  {
            //      printf("%lf - %lf\n", optimizeData[n][i], wholeData[n][i]);
            //  }
            //}
            //printf("%lf - %lf\n", filterDataMinMax[i][0], filterDataMinMax[i][1]);
            free(filterDataMinMax[i]);
            free(filterWeights[i]);
        }
        free(filterDataMinMax);
        free(filterWeights);
        free(rank_INDX);

        filterTag = 0;
    }

    if(corrTag) {
        for(int i = 0; i < N_sample_corr; i++) {
            free(corrData[i]);
            free(corrMatrix[i]);
        }
        free(corrData);
        free(corrMatrix);
        free(meanFeature);

        corrTag = 0;
    }

    for(int i = 0; i < N_FEATURE; i++) {
        //if (fabs(wholeDataMinMax[i][0])>1e30 || fabs(wholeDataMinMax[i][1]) > 1e30)
        //{
        //  int test = 1;
        //  for (int n = 0; n < N_row_whole_data; n++)
        //  {
        //      printf("%lf\n", wholeData[n][i]);
        //  }
        //}
        free(wholeDataMinMax[i]);
    }
    free(wholeDataMinMax);

    for(int i = 0; i < N_row_whole_data; i++) {
        free(wholeData[i]);
        free(flag_wholeData[i]);
    }
    free(wholeData);
    free(flag_wholeData);

    for(int i = 0; i < N_sample_optimize; i++) {
        free(optimizeData[i]);
        free(optimizeIdx[i]);
    }
    free(optimizeData);
    free(optimizeIdx);
    for(int i = 0; i < N_sample_test; i++) {
        free(testData[i]);
        free(testIdx[i]);
    }
    free(testData);
    free(testIdx);

    if(innerTag) {
        for(int i = 0; i < N_sample_train; i++) {
            free(trainData[i]);
            free(trainIdx[i]);
        }
        free(trainData);
        free(trainIdx);
        for(int i = 0; i < N_sample_validation; i++) {
            free(validationData[i]);
            free(validationIdx[i]);
        }
        free(validationData);
        free(validationIdx);

        innerTag = 0;
    }

    return;
}

void featureCorrelation()
{
    if(corrTag) {
        for(int i = 0; i < N_sample_corr; i++) {
            free(corrData[i]);
            free(corrMatrix[i]);
        }
        free(corrData);
        free(corrMatrix);
        free(meanFeature);

        corrTag = 0;
    }

    //N_sample_corr = N_sample_whole - N_sample_test;
    //corrData = allocDoubleCLASS(N_sample_corr, N_whole_data);
    //corrMatrix = allocDoubleCLASS(N_FEATURE, N_FEATURE);

    //selectionFlag = allocIntCLASS(N_sample_whole, 1);
    //for (int i = 0; i < N_sample_whole; i++)
    //  selectionFlag[i][0] = 0;
    //for (int i = 0; i < N_sample_test; i++)
    //  selectionFlag[testIdx[i][0]][0] = 1;
    //int count = 0;
    //for (int i = 0; i < N_sample_whole; i++)
    //if (!selectionFlag[i][0])
    //  memcpy(corrData[count++], wholeData[i], N_whole_data*sizeof(double));
    N_sample_corr = N_sample_optimize;
    corrData = allocDoubleCLASS(N_sample_corr, N_col_whole_data);
    corrMatrix = allocDoubleCLASS(N_FEATURE, N_FEATURE);
    for(int i = 0; i < N_sample_optimize; i++) {
        memcpy(corrData[i], optimizeData[i], N_col_whole_data * sizeof(double));
    }
    meanFeature = (double*)calloc(N_FEATURE, sizeof(double));

    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    for(int i = 0; i < N_FEATURE; i++) {
        meanFeature[i] = 0.0;
    }
    for(int i = 0; i < N_sample_corr; i++) {
        for(int j = 0; j < N_FEATURE; j++) {
            meanFeature[j] += corrData[i][j];
        }
    }
    for(int i = 0; i < N_FEATURE; i++) {
        meanFeature[i] /= N_sample_corr;
    }

    double r, diff1, diff2;
    double sum, sx, sy;
    int count = 0;
    double minV = 1.0, maxV = 0.0, meanV = 0.0;
    for(int i = 0; i < N_FEATURE; i++) {
        for(int j = 0; j < i; j++) {
            sum = 0.0;
            sx = 0.0;
            sy = 0.0;
            for(int k = 0; k < N_sample_corr; k++) {
                diff1 = corrData[k][i] - meanFeature[i];
                diff2 = corrData[k][j] - meanFeature[j];
                sum += (diff1 * diff2);
                sx += (diff1 * diff1);
                sy += (diff2 * diff2);
            }
            if((sx * sy) > 0.0) {
                r = (sum / sqrt(sx * sy));
                if(r < 0.0) r = (-r);
                corrMatrix[i][j] = corrMatrix[j][i] = r; //printf("%lf\t", r);
            } else {
                corrMatrix[i][j] = corrMatrix[j][i] = 0.0;
            }
            if(corrMatrix[i][j] < minV && corrMatrix[i][j] > 0.0)
                minV = corrMatrix[i][j];
            if(corrMatrix[i][j] > maxV)
                maxV = corrMatrix[i][j];
            meanV += corrMatrix[i][j];
            count++;
        }
    }
    printf("minV = %.16e\nmaxV =%.16e\nmeanV = %.16e\n", minV, maxV, meanV / count);
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////

    //for (int i = 0; i < N_sample_whole; i++){
    //  free(selectionFlag[i]);
    //}
    //free(selectionFlag);

    corrTag = 1;

    return;
}

double featureCorrelation2(int iFeat, int jFeat)
{
    double r, diff1, diff2;
    double sum = 0.0, sx = 0.0, sy = 0.0;
    double mx = 0.0, my = 0.0;

    for(int i = 0; i < N_sample_optimize; i++) {
        mx += optimizeData[i][iFeat];
        my += optimizeData[i][jFeat];
    }
    mx /= N_sample_optimize;
    my /= N_sample_optimize;

    for(int k = 0; k < N_sample_optimize; k++) {
        diff1 = optimizeData[k][iFeat] - mx;
        diff2 = optimizeData[k][jFeat] - my;
        sum += (diff1 * diff2);
        sx += (diff1 * diff1);
        sy += (diff2 * diff2);
    }
    if((sx * sy) > 0.0) {
        r = (sum / sqrt(sx * sy));
        if(r < 0.0) r = (-r);
        return r; //printf("%lf\t", r);
    } else {
        return 0.0;
    }
}

void testAccuracy(double* individual, double* fitness)
{
    int numFeature = 0;

    int* arrIDX = (int*)calloc(N_FEATURE, sizeof(int));

#ifdef WEIGHT_ENCODING
    int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));

    numFeature = (int)(individual[N_FEATURE] * TH_N_FEATURE) + 1;
    if(numFeature > TH_N_FEATURE)
        numFeature = TH_N_FEATURE;

    double tmpMAX = -1.0;
    int    tmpIND = -1;
    for(int i = 0; i < numFeature; i++) {
        tmpMAX = -1.0;
        for(int j = 0; j < N_FEATURE; j++) {
            if(tmpFlag[j] == 0 && individual[j] > tmpMAX) {
                tmpMAX = individual[j];
                tmpIND = j;
            }
        }
        tmpFlag[tmpIND] = 1;
        arrIDX[i] = tmpIND;
    }
#else
    for(int i = 0; i < N_FEATURE; i++) {
        if((int)individual[i])
            arrIDX[numFeature++] = i;
    }
#endif

    //printf("--- ---%d\n", numFeature);

    int nTest = N_sample_test;
    int cur_TP = 0;
    int cur_FP = 0;
    int cur_TN = 0;
    int cur_FN = 0;
    double f_prec = 0.0;

    for(int i = 0; i < nTest; i++) {
        // #define BOOTSTRAP     0 // .632 bootstrap
        // #define BOOTSTRAPplus 1 // .632+ bootstrap
        // #define MULTIFOLD     2 // ???????
        // #define RATIO         3 // ????????
        {
            int predictedLabels;
            //kNN - mine
            {
                const int K = 1;
                double dists[K];
                int    labels[K];
                //cout << "SMP_IND: " << i << endl;
                for(int j = 0; j < K; j++) {
                    dists[j] = -1;
                    labels[j] = -1;
                }
                for(int j = 0; j < N_sample_optimize; j++) {
                    double d = 0.0;
                    for(int k = 0; k < N_FEATURE; k++) {
#ifdef WEIGHT_ENCODING
                        if(tmpFlag[k])
#else
                        if((int)individual[k])
#endif
                            d += (optimizeData[j][k] - testData[i][k]) * (optimizeData[j][k] - testData[i][k]);
                    }
                    int index = -1;
                    for(int k = 0; k < K; k++) {
                        if(dists[k] < 0) {  //uninitialized
                            index = k;
                            break;
                        }
                        if(dists[k] > d) {  //greater than d
                            if(index == -1) {
                                index = k;
                            } else {
                                if(dists[k] > dists[index]) {  //the greatest one greater than d
                                    index = k;
                                }
                            }
                        }
                    }
                    if(index >= 0) {
                        dists[index] = d;
                        labels[index] = (int)optimizeData[j][N_FEATURE];
                    }
                }
                double minD = -1;
                int    minLabel = -1;
                for(int k = 0; k < K; k++) {
                    if(minD < 0 || dists[k] < minD) {
                        minD = dists[k];
                        minLabel = labels[k];
                    }
                }
                //cout << minD << endl;
                predictedLabels = minLabel;
                if(predictedLabels == (int)testData[i][N_FEATURE]) {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_TP++;
                    } else {
                        cur_TN++;
                    }
                } else {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_FP++;
                    } else {
                        cur_FN++;
                    }
                }
            }
        }
    }

    f_prec = ((double)cur_FP / (cur_FP + cur_TN) + (double)cur_FN / (cur_FN + cur_TP)) / 2.0;
    fitness[0] = f_prec;
    fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    free(arrIDX);
#ifdef WEIGHT_ENCODING
    free(tmpFlag);
#endif

    return;
}

void testAccuracy(int* individual, double* fitness)
{
    int numFeature = 0;

    int arrIDX[TH_N_FEATURE];

    for(int i = 0; i < TH_N_FEATURE; i++) {
        if(individual[i] >= 0)
            arrIDX[numFeature++] = individual[i];
    }

    //printf("%d\n", numFeature);

    int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));
    for(int i = 0; i < numFeature; i++) {
        tmpFlag[arrIDX[i]] = 1;
    }

    int nTest = N_sample_test;
    int cur_TP = 0;
    int cur_FP = 0;
    int cur_TN = 0;
    int cur_FN = 0;
    double f_prec = 0.0;

    for(int i = 0; i < nTest; i++) {
        // #define BOOTSTRAP     0 // .632 bootstrap
        // #define BOOTSTRAPplus 1 // .632+ bootstrap
        // #define MULTIFOLD     2 // ???????
        // #define RATIO         3 // ????????
        {
            int predictedLabels;
            //kNN - mine
            {
                const int K = 1;
                double dists[K];
                int    labels[K];
                //cout << "SMP_IND: " << i << endl;
                for(int j = 0; j < K; j++) {
                    dists[j] = -1;
                    labels[j] = -1;
                }
                for(int j = 0; j < N_sample_optimize; j++) {
                    double d = 0.0;
                    for(int k = 0; k < N_FEATURE; k++) {
                        if(tmpFlag[k])
                            d += (optimizeData[j][k] - testData[i][k]) * (optimizeData[j][k] - testData[i][k]);
                    }
                    int index = -1;
                    for(int k = 0; k < K; k++) {
                        if(dists[k] < 0) {  //uninitialized
                            index = k;
                            break;
                        }
                        if(dists[k] > d) {  //greater than d
                            if(index == -1) {
                                index = k;
                            } else {
                                if(dists[k] > dists[index]) {  //the greatest one greater than d
                                    index = k;
                                }
                            }
                        }
                    }
                    if(index >= 0) {
                        dists[index] = d;
                        labels[index] = (int)optimizeData[j][N_FEATURE];
                    }
                }
                double minD = -1;
                int    minLabel = -1;
                for(int k = 0; k < K; k++) {
                    if(minD < 0 || dists[k] < minD) {
                        minD = dists[k];
                        minLabel = labels[k];
                    }
                }
                //cout << minD << endl;
                predictedLabels = minLabel;
                if(predictedLabels == (int)testData[i][N_FEATURE]) {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_TP++;
                    } else {
                        cur_TN++;
                    }
                } else {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_FP++;
                    } else {
                        cur_FN++;
                    }
                }
            }
        }
    }

    f_prec = ((double)cur_FP / (cur_FP + cur_TN) + (double)cur_FN / (cur_FN + cur_TP)) / 2.0;
    fitness[0] = f_prec;
    fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    free(tmpFlag);

    return;
}

void testAccuracy(double* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species)
{
    int numFeature = 0;
    int* arrIDX = (int*)calloc(N_FEATURE, sizeof(int));

    if(mpi_rank_species == 0) {
#ifdef WEIGHT_ENCODING
        int* tmpFlag = (int*)calloc(N_FEATURE, sizeof(int));

        numFeature = (int)(individual[N_FEATURE] * TH_N_FEATURE) + 1;
        if(numFeature > TH_N_FEATURE)
            numFeature = TH_N_FEATURE;

        double tmpMAX = -1.0;
        int    tmpIND = -1;
        for(int i = 0; i < numFeature; i++) {
            tmpMAX = -1.0;
            for(int j = 0; j < N_FEATURE; j++) {
                if(tmpFlag[j] == 0 && individual[j] > tmpMAX) {
                    tmpMAX = individual[j];
                    tmpIND = j;
                }
            }
            tmpFlag[tmpIND] = 1;
            arrIDX[i] = tmpIND;
        }

        free(tmpFlag);
#else
        for(int i = 0; i < N_FEATURE; i++) {
            if((int)individual[i])
                arrIDX[numFeature++] = i;
        }
#endif
    }

    MPI_Bcast(&numFeature, 1, MPI_INT, 0, comm_species);
    MPI_Bcast(arrIDX, numFeature, MPI_INT, 0, comm_species);

    int nTest = N_sample_test;
    int cur_TP = 0;
    int cur_FP = 0;
    int cur_TN = 0;
    int cur_FN = 0;
    double f_prec = 0.0;
    for(int i = mpi_rank_species; i < nTest; i += mpi_size_species) {
        // #define BOOTSTRAP     0 // .632 bootstrap
        // #define BOOTSTRAPplus 1 // .632+ bootstrap
        // #define MULTIFOLD     2 // ???????
        // #define RATIO         3 // ????????
        {
            int predictedLabels;
            //kNN - mine
            {
                const int K = 1;
                double dists[K];
                int    labels[K];
                //cout << "SMP_IND: " << i << endl;
                for(int j = 0; j < K; j++) {
                    dists[j] = -1;
                    labels[j] = -1;
                }
                for(int j = 0; j < N_sample_optimize; j++) {
                    double d = 0.0;
                    for(int k = 0; k < numFeature; k++) {
                        d += (optimizeData[j][arrIDX[k]] - testData[i][arrIDX[k]]) * (optimizeData[j][arrIDX[k]] - testData[i][arrIDX[k]]);
                    }
                    int index = -1;
                    for(int k = 0; k < K; k++) {
                        if(dists[k] < 0) {  //uninitialized
                            index = k;
                            break;
                        }
                        if(dists[k] > d) {  //greater than d
                            if(index == -1) {
                                index = k;
                            } else {
                                if(dists[k] > dists[index]) {  //the greatest one greater than d
                                    index = k;
                                }
                            }
                        }
                    }
                    if(index >= 0) {
                        dists[index] = d;
                        labels[index] = (int)optimizeData[j][N_FEATURE];
                    }
                }
                double minD = -1;
                int    minLabel = -1;
                for(int k = 0; k < K; k++) {
                    if(minD < 0 || dists[k] < minD) {
                        minD = dists[k];
                        minLabel = labels[k];
                    }
                }
                //cout << minD << endl;
                predictedLabels = minLabel;
                if(predictedLabels == (int)testData[i][N_FEATURE]) {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_TP++;
                    } else {
                        cur_TN++;
                    }
                } else {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_FP++;
                    } else {
                        cur_FN++;
                    }
                }
            }
        }
    }

    int sum_TP = 0;
    int sum_FP = 0;
    int sum_TN = 0;
    int sum_FN = 0;

    MPI_Reduce(&cur_TP, &sum_TP, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_FP, &sum_FP, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_TN, &sum_TN, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_FN, &sum_FN, 1, MPI_INT, MPI_SUM, 0, comm_species);

    f_prec = ((double)sum_FP / (sum_FP + sum_TN) + (double)sum_FN / (sum_FN + sum_TP)) / 2.0;
    fitness[0] = f_prec;
    fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    free(arrIDX);

    return;
}

void testAccuracy(int* individual, double* fitness, MPI_Comm comm_species, int mpi_rank_species, int mpi_size_species)
{
    int numFeature = 0;
    int arrIDX[TH_N_FEATURE];

    if(mpi_rank_species == 0) {
        for(int i = 0; i < TH_N_FEATURE; i++) {
            if(individual[i] >= 0)
                arrIDX[numFeature++] = individual[i];
        }

        //printf("---%d\n", numFeature);
    }

    MPI_Bcast(&numFeature, 1, MPI_INT, 0, comm_species);
    MPI_Bcast(arrIDX, numFeature, MPI_INT, 0, comm_species);

    int nTest = N_sample_test;
    int cur_TP = 0;
    int cur_FP = 0;
    int cur_TN = 0;
    int cur_FN = 0;
    double f_prec = 0.0;
    for(int i = mpi_rank_species; i < nTest; i += mpi_size_species) {
        // #define BOOTSTRAP     0 // .632 bootstrap
        // #define BOOTSTRAPplus 1 // .632+ bootstrap
        // #define MULTIFOLD     2 // ???????
        // #define RATIO         3 // ????????
        {
            int predictedLabels;
            //kNN - mine
            {
                const int K = 1;
                double dists[K];
                int    labels[K];
                //cout << "SMP_IND: " << i << endl;
                for(int j = 0; j < K; j++) {
                    dists[j] = -1;
                    labels[j] = -1;
                }
                for(int j = 0; j < N_sample_optimize; j++) {
                    double d = 0.0;
                    for(int k = 0; k < numFeature; k++) {
                        d += (optimizeData[j][arrIDX[k]] - testData[i][arrIDX[k]]) * (optimizeData[j][arrIDX[k]] - testData[i][arrIDX[k]]);
                    }
                    int index = -1;
                    for(int k = 0; k < K; k++) {
                        if(dists[k] < 0) {  //uninitialized
                            index = k;
                            break;
                        }
                        if(dists[k] > d) {  //greater than d
                            if(index == -1) {
                                index = k;
                            } else {
                                if(dists[k] > dists[index]) {  //the greatest one greater than d
                                    index = k;
                                }
                            }
                        }
                    }
                    if(index >= 0) {
                        dists[index] = d;
                        labels[index] = (int)optimizeData[j][N_FEATURE];
                    }
                }
                double minD = -1;
                int    minLabel = -1;
                for(int k = 0; k < K; k++) {
                    if(minD < 0 || dists[k] < minD) {
                        minD = dists[k];
                        minLabel = labels[k];
                    }
                }
                //cout << minD << endl;
                predictedLabels = minLabel;
                if(predictedLabels == (int)testData[i][N_FEATURE]) {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_TP++;
                    } else {
                        cur_TN++;
                    }
                } else {
                    if(predictedLabels == labelCLASS[0]) {
                        cur_FP++;
                    } else {
                        cur_FN++;
                    }
                }
            }
        }
    }

    int sum_TP = 0;
    int sum_FP = 0;
    int sum_TN = 0;
    int sum_FN = 0;

    MPI_Reduce(&cur_TP, &sum_TP, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_FP, &sum_FP, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_TN, &sum_TN, 1, MPI_INT, MPI_SUM, 0, comm_species);
    MPI_Reduce(&cur_FN, &sum_FN, 1, MPI_INT, MPI_SUM, 0, comm_species);

    f_prec = ((double)sum_FP / (sum_FP + sum_TN) + (double)sum_FN / (sum_FN + sum_TP)) / 2.0;
    fitness[0] = f_prec;
    fitness[1] = (double)numFeature / TH_N_FEATURE;// N_FEATURE;
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);
    int count = 0;
    double corrV = 0.0;
    // double corrV2 = 0.0;
    for(int i = 0; i < numFeature; i++) {
        for(int j = 0; j < i; j++) {
            corrV += featureCorrelation2(arrIDX[i], arrIDX[j]);
            //corrV += corrMatrix[arrIDX[i]][arrIDX[j]];
            count++;
        }
        if(numFeature == 1) {
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrIDX[i] == j) continue;
                corrV += featureCorrelation2(arrIDX[i], j);
                //corrV += corrMatrix[arrIDX[i]][j];
                count++;
            }
        }
    }
    if(count) {
        fitness[2] = corrV / count;
        //fitness[3] = corrV2 / count;
    } else {
        fitness[2] = 10.0;
        //fitness[3] = 10.0;
    }
    //printf("%lf\n", (double)sumCorrect / sumSample);
    //printf("%d\n", numFeature);

    return;
}

void bootstrapInitialize(int**& index)
{
    ////////////////////////////////////////////
    ////////////////////////////////////////////
    ////////////////////////////////////////////
    int* selectIndicator = (int*)calloc(N_row_whole_data * opNum, sizeof(int));

    int count = 0;
    for(int i = 0; i < N_CLASS; i++) {
        for(int j = 0; j < N_row_whole_data; j++) {
            if((int)wholeData[j][ind_class] == labelCLASS[i]) {
                for(int k = count; k < count + opNum; k++) {
                    selectIndicator[k] = -j;
                }
                count += opNum;
            }
        }
    }

    int* tmpIDX = (int*)calloc(N_row_whole_data, sizeof(int));

    for(int n = 0; n < opNum; n++) {
        int ind_begin = 0;
        int ind_end = 0;

        for(int i = 0; i < N_CLASS; i++) {
            ind_begin = ind_end;
            ind_end = ind_begin + arrSize[i] * opNum;

            int j = ind_begin / opNum;
            while(j < ind_end / opNum) {
                int tmp = (int)(rnd_uni_CLASS(&rnd_uni_init_CLASS) * arrSize[i] * opNum);
                if(selectIndicator[tmp] <= 0) {
                    tmpIDX[j] = -selectIndicator[tmp];
                    selectIndicator[tmp] = 1 - selectIndicator[tmp];
                    j++;
                }
            }
        }

        if(n == opNumCur) {
            for(int i = 0; i < N_row_whole_data; i++) {
                index[i][0] = tmpIDX[i];
            }
            //break;
        }

        //for (int j = 0; j < M_whole_data; j++){
        //  printf("%2d ", tmpIDX[j]);
        //}
        //printf("\n");
    }

    free(selectIndicator);
    free(tmpIDX);

    return;
}

void multifoldInitialize(int**& index)
{
    ////////////////////////////////////////////
    ////////////////////////////////////////////
    ////////////////////////////////////////////
    int* stratufiedIndx = (int*)calloc(N_row_whole_data, sizeof(int));
    int* testIndxTag = (int*)calloc(N_row_whole_data, sizeof(int));

    int count = 0;
    for(int i = 0; i < N_CLASS; i++) {
        for(int j = 0; j < N_row_whole_data; j++) {
            if((int)wholeData[j][ind_class] == labelCLASS[i]) {
                stratufiedIndx[count++] = j;
            }
        }
    }

    int n = opNumCur;
    while(n < N_row_whole_data) {
        testIndxTag[stratufiedIndx[n]] = 1;
        //printf("%d ", stratufiedIndx[n]);
        n += opNum;
    }
    //printf("\n");
    //for(n = 0; n < N_row_whole_data; n++) {
    //    if(testIndxTag[n])
    //        printf("%d ", n);
    //}
    //printf("\n");
    //printf("\n");

    count = 0;
    for(n = 0; n < N_row_whole_data; n++) {
        if(!testIndxTag[n])
            index[count++][0] = n;
    }

    //for(n = 0; n < count; n++) {
    //    printf("%d ", index[n][0]);
    //}
    //printf("\n");

    for(n = count - 1; n >= 0; n--) {
        int tmpInd = (int)(rnd_uni_CLASS(&rnd_uni_init_CLASS) * (n + 1));
        int tmp_val = index[n][0];
        index[n][0] = index[tmpInd][0];
        index[tmpInd][0] = tmp_val;
    }

    free(stratufiedIndx);
    free(testIndxTag);

    return;
}

//int main()
//{
//  //"ALLGSE412_poterapiji"
//  //"ALLGSE412_pred_poTh"
//  //"AMLGSE2191"
//  //"BC_CCGSE3726_frozen"
//  //"BCGSE349_350"
//  //"bladderGSE89"
//  //"braintumor"
//  //"CMLGSE2535"
//  //"DLBCL"
//  //"EWSGSE967"
//  //"EWSGSE967_3class"
//  //"gastricGSE2685"
//  //"gastricGSE2685_2razreda"
//  //"glioblastoma"
//  //"leukemia"
//  //"LL_GSE1577"
//  //"LL_GSE1577_2razreda"
//  //"lung"
//  //"lungGSE1987"
//  //"meduloblastomiGSE468"
//  //"MLL"
//  //"prostata"
//  //"prostateGSE2443"
//  //"SRBCT"
//  //"ALLGSE412_poterapiji_TREE"
//  //"ALLGSE412_pred_poTh_TREE"
//  //"AMLGSE2191_TREE"
//  //"BC_CCGSE3726_frozen_TREE"
//  //"BCGSE349_350_TREE"
//  //"bladderGSE89_TREE"
//  //"braintumor_TREE"
//  //"CMLGSE2535_TREE"
//  //"DLBCL_TREE"
//  //"EWSGSE967_TREE"
//  //"EWSGSE967_3class_TREE"
//  //"gastricGSE2685_TREE"
//  //"gastricGSE2685_2razreda_TREE"
//  //"glioblastoma_TREE"
//  //"leukemia_TREE"
//  //"LL_GSE1577_TREE"
//  //"LL_GSE1577_2razreda_TREE"
//  //"lung_TREE"
//  //"lungGSE1987_TREE"
//  //"meduloblastomiGSE468_TREE"
//  //"MLL_TREE"
//  //"prostata_TREE"
//  //"prostateGSE2443_TREE"
//  //"SRBCT_TREE"
//
//  Initialize_ClassifierFunc("LL_GSE1577_2razreda", 0);
//
//  double* LowLimit = (double*)calloc(DIM_ClassifierFunc, sizeof(double));
//  double* UppLimit = (double*)calloc(DIM_ClassifierFunc, sizeof(double));
//
//  SetLimits_ClassifierFunc(LowLimit, UppLimit);
//
//  double* tmpX = (double*)calloc(DIM_ClassifierFunc, sizeof(double));
//  double* tmpY = (double*)calloc(DIM_ClassifierFunc, sizeof(double));
//  int flag = 0;
//
//  for (int i = 0; i < DIM_ClassifierFunc; i++){
//      tmpX[i] = LowLimit[i] + rnd_uni_CLASS(&rnd_uni_init_CLASS)*(UppLimit[i] - LowLimit[i]);
//  }
//  {
//      int count = 0;
//      int *arrIDX = (int*)calloc(DIM_ClassifierFunc, sizeof(int));
//
//      for (int i = 0; i < DIM_ClassifierFunc; i++){
//          if (tmpX[i]>0.5){
//              arrIDX[count++] = i;
//          }
//      }
//
//      int thresh = TH_N_FEATURE;
//
//      if (count > thresh){
//          for (int i = count; i > thresh; i--){
//              int tmp = rnd_uni_CLASS(&rnd_uni_init_CLASS) * i;
//              int tmp2 = arrIDX[tmp];
//              tmpX[tmp2] = 1.0 - tmpX[tmp2];
//              arrIDX[tmp] = arrIDX[i - 1];
//              arrIDX[i - 1] = tmp2;
//          }
//      }
//
//      if (count == 0){
//          count = rnd_uni_CLASS(&rnd_uni_init_CLASS) * thresh + 1;
//          int i = 0;
//          while (i < count){
//              int tmp = rnd_uni_CLASS(&rnd_uni_init_CLASS) * DIM_ClassifierFunc;
//              int flag = 1;
//              for (int j = 0; j < i; j++){
//                  if (tmp == arrIDX[j])
//                      flag = 0;
//              }
//              if (flag){
//                  arrIDX[i++] = tmp;
//              }
//          }
//          for (i = 0; i < count; i++){
//              tmpX[arrIDX[i]] = 1.0 - tmpX[arrIDX[i]];
//          }
//      }
//
//      free(arrIDX);
//  }
//
//  double t = (double)clock();
//  Fitness_ClassifierFunc(tmpX, tmpY);
//  t = ((double)clock() - t) / CLOCKS_PER_SEC;
//  printf("CvKNearest --- %lf - %lf - %lf - %lf\n --- Times passed in seconds: %lf\n", tmpY[0], tmpY[1], tmpY[2], tmpY[3], t);
//
//  free(LowLimit);
//  free(UppLimit);
//  free(tmpX);
//  free(tmpY);
//
//  return (0);
//}