/*
* =============================================================
* IGD.cpp
*
* Matlab-C source codes
*
* IGD performance metric for CEC 2009 MOO Competition
*
* Usage: igd = IGD(PF*, PF, C) or igd = IGD(PF*, PF)
*
* Calculate the distance from the ideal Pareto Front (PF*) to an
* obtained nondominated front (PF), C is the constraints
*
* PF*, PF, C MUST be columnwise,
*
* Please refer to the report for more information.
* =============================================================
*/

#include "indicator_igd.h"

// #define BUFSIZ 512

static void trimLine(char line[])
{
    int i = 0;

    while(line[i] != '\0') {
        if(line[i] == '\r' || line[i] == '\n') {
            line[i] = '\0';
            break;
        }
        i++;
    }
}

FRONT* readFile(char filename[])
{
    FILE* fp;
    char line[BUFSIZ];
    int point = 0, objective = 0;
    FRONT* __front;

    __front = NULL;

    fp = fopen(filename, "r");
    if(fp == NULL) {
        fprintf(stderr, "File %s could not be opened\n", filename);
        exit(1);
    }

    __front = (FRONT*)calloc(1, sizeof(FRONT));
    __front->nPoints = 0;
    __front->points = NULL;

    while(fgets(line, sizeof line, fp) != NULL) {
        trimLine(line);

        FRONT* f = __front;
        point = f->nPoints;
        f->nPoints++;
        f->points = (POINT*)realloc(f->points, sizeof(POINT) * f->nPoints);
        f->n = 0;
        f->points[point].objectives = NULL;
        char* tok = strtok(line, " \t\n");
        do {
            POINT* p = &f->points[point];
            objective = f->n;
            f->n++;
            p->objectives = (OBJECTIVE*)realloc(p->objectives, sizeof(OBJECTIVE) * f->n);
            p->objectives[objective] = atof(tok);
        } while((tok = strtok(NULL, " \t\n")) != NULL);
    }

    // for (int i = 0; i < fc->nFronts; i++) fc->fronts[i].n = fc->fronts[i].points[0].nObjectives;
    fclose(fp);
    /* printf("Read %d fronts\n", fc->nFronts);
       printContents(fc); */
    return __front;
}

void printContents(FRONT* f)
{
    for(int j = 0; j < f->nPoints; j++) {
        printf("\t");
        for(int k = 0; k < f->n; k++) {
            printf("%f ", f->points[j].objectives[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void freeFRONT(FRONT* f)
{
    for(int j = 0; j < f->nPoints; j++) {
        free(f->points[j].objectives);
    }
    free(f);
    return;
}

double IGD(double* S, double* Q, int _nS, int _nQ, int _nobj)
{
    int i, j, k;
    double d, min, dis;
    int nS = _nS;
    int nQ = _nQ;
    int nobj = _nobj;

    /* Step 1: remove the infeasible points, i.e, constraint<-1.0E-6 */
    /* omitted, only for constrained*/

    /* Step 2: calculate the IGD value for feasible points */
    dis = 0.0;
    for(i = 0; i < nS; i++) {
        min = 1.0E200;
        for(j = 0; j < nQ; j++) {
            d = 0.0;
            for(k = 0; k < nobj; k++)
                d += (S[i * nobj + k] - Q[j * nobj + k]) * (S[i * nobj + k] - Q[j * nobj + k]);
            if(d < min) min = d;
        }
        dis += sqrt(min);
    }
    return dis / (double)(nS);
}

double generateIGD(char* _testName, char* _algoName, double* _estedPF, int _nEstedPF, int _nObj, int _nDim, int* iKey,
                   int iRun)
{
    if(!strcmp(_testName, "HDSN") ||
       !strcmp(_testName, "HDSN_URBAN") ||
       !strcmp(_testName, "IWSN") ||
       !strcmp(_testName, "IWSN_S_1F") ||
       !strcmp(_testName, "RS") ||
       !strcmp(_testName, "WDCN") ||
       !strcmp(_testName, "ARRANGE2D") ||
       !strcmp(_testName, "Classify_CNN_Indus") ||
       !strcmp(_testName, "Classify_NN_Indus") ||
       !strcmp(_testName, "Classify_CFRNN_Indus") ||
       !strcmp(_testName, "Classify_CFRNN_MNIST") ||
       !strcmp(_testName, "Classify_CFRNN_FashionMNIST") ||
       !strcmp(_testName, "EdgeComputation") ||
       !strncmp(_testName, "FeatureSelection", 16) ||
       !strncmp(_testName, "FINANCE", 7) ||
       !strncmp(_testName, "FRNN", 4) ||
       !strncmp(_testName, "EVO1_FRNN", 9) ||
       !strncmp(_testName, "EVO2_FRNN", 9) ||
       !strncmp(_testName, "EVO3_FRNN", 9) ||
       !strncmp(_testName, "EVO4_FRNN", 9) ||
       !strncmp(_testName, "EVO5_FRNN", 9) ||
       !strncmp(_testName, "evoFRNN_Predict_", 16) ||
       !strncmp(_testName, "evoGFRNN_Predict_", 17) ||
       !strncmp(_testName, "evoDFRNN_Predict_", 17) ||
       !strncmp(_testName, "evoFGRNN_Predict_", 17) ||
       !strncmp(_testName, "evoBFRNN_Predict_", 17) ||
       !strcmp(_testName, "IntrusionDetection_FRNN_Classify") ||
       !strcmp(_testName, "ActivityDetection_FRNN_Classify") ||
       !strcmp(_testName, "evoCFRNN_Classify") ||
       strstr(_testName, "evoMobileSink")) {
        return -10.0;
    }

    int nobj = _nObj;
    //int ndim = _nDim;
    int n_iPF;

    char fileName_iPF[256];
    if(_testName[0] == 'U' && _testName[1] == 'F')
        sprintf(fileName_iPF, "../Data_all/pareto_fronts/%s.pf", _testName);
    else
        sprintf(fileName_iPF, "../Data_all/pareto_fronts/%s.%dD.pf", _testName, nobj);

    FRONT* f_iPF = readFile(fileName_iPF); //printContents(f_iPF);

    if(f_iPF->n != nobj) {
        printf("ideal PF and PF do not have the same number of objectives\n");
        exit(2);
    }
    n_iPF = f_iPF->nPoints;

    double* iPF = (double*)malloc(n_iPF * nobj * sizeof(double));
    for(int i = 0; i < n_iPF; i++)
        for(int j = 0; j < nobj; j++) {
            iPF[i * nobj + j] =
                f_iPF->points[i].objectives[j];
        }

    double igd = IGD(iPF, _estedPF, n_iPF, _nEstedPF, nobj);

    /*	char fileName[256];
    	if(iKey)
    	sprintf(fileName,"IGD/trace/IGD_%s_%s_OBJ%d_VAR%d_RUN%d",_algoName,_testName,nobj,ndim,iRun);
    	else
    	sprintf(fileName,"IGD/IGD_%s_%s_OBJ%d_VAR%d_RUN%d",_algoName,_testName,nobj,ndim,iRun);
    	FILE* fpt=fopen(fileName,"a");
    	if(iKey)
    	fprintf(fpt,"%d\t",*iKey);
    	fprintf(fpt,"%1.16lf\n",igd);
    	fclose(fpt);*/

    //if (iKey)
    //    printf("IGD %s %s OBJ%d VAR%d RUN%d KEY%d: %lf\n", _algoName, _testName, nobj, ndim, iRun, *iKey, igd);
    //else
    //    printf("IGD %s %s OBJ%d VAR%d RUN%d FINAL: %lf\n", _algoName, _testName, nobj, ndim, iRun, igd);

    free(iPF);
    freeFRONT(f_iPF);

    return igd;
}