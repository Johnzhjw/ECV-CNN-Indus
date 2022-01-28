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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// #define BUFSIZ 512

typedef double OBJECTIVE;

typedef struct {
	OBJECTIVE* objectives;
	// 	struct avl_node_t * tnode;
} POINT;

typedef struct {
	int nPoints;
	int n;
	POINT* points;
} FRONT;

FRONT* readFile(char filename[]);

void printContents(FRONT* f);

void freeFRONT(FRONT* f);

double IGD(double* S, double* Q, int _nS, int _nQ, int _nobj);

double generateIGD(char* _testName, char* _algoName, double* _estedPF, int _nEstedPF, int _nObj, int _nDim, int* iKey, int iRun);
