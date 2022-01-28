#ifndef _EMO_TEST_SUITE_
#define _EMO_TEST_SUITE_

//////////////////////////////////////////////////////////////////////////
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include "MOP_FRNN_MODEL.h"
//benchmark problems
#include "MOP_UF_CF.h"
#include "MOP_DTLZ.h"
#include "MOP_WFG.h"
#include "MOP_LSMOP.h"
#include "MOP_SEC2018.h"
//Real-world problems
#include "MOP_Classify.h"
#include "MOP_Classify_TREE.h"
#include "MOP_HDSN.h"
#include "MOP_HDSN_URBAN.h"
#include "MOP_IWSN.h"
#include "MOP_RS.h"
#include "MOP_WDCN.h"
#include "MOP_LeNet.h"
#include "MOP_IWSN_1F.h" //Not used
#include "MOP_IWSN_Security_1F.h"
#include "MOP_RecSys_SmartCity.h"
#include "MOP_EVO1_FRNN.h"
#include "MOP_EVO2_FRNN.h"
#include "MOP_EVO3_FRNN.h"
#include "MOP_EVO4_FRNN.h"
#include "MOP_EVO5_FRNN.h"
#include "MOP_ARRANGE2D.h"
#include "MOP_Classify_CNN.h"
#include "MOP_Classify_NN.h"
#include "MOP_IntrusionDetection.h"
#include "MOP_ActivityDetection.h"
#include "MOP_evoCNN.h"
#include "MOP_evoCFRNN.h"
#include "MOP_Classify_CFRNN.h"
#include "MOP_EdgeComputation.h"
#include "MOP_Predict_FRNN.h"
#include "MOP_Predict_DFRNN.h"
#include "MOP_Mobile_Sink.h"

//////////////////////////////////////////////////////////////////////////

extern int LSMOP_D;
extern int EMO_test_suite_nvar, EMO_test_suite_nobj;                    //  the number of variables and objectives
extern int EMO_test_suite_position_parameters;
extern char EMO_test_suite_testInstName[1024];

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void EMO_setLimits(char* pro, double* minLimit, double* maxLimit, int dim);
void EMO_adjust_constraint_penalty(double cur_iter, double max_iter);
void EMO_evaluate_problems(char* pro, double* xreal, double* obj, int dim, int nx, int nobj);
void EMO_initialization(char* pro, int& nobj, int& ndim, int curN, int numN, int my_rank, int para_1, int para_2, int para_3);
void EMO_finalization(char* pro);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
#endif	//	_EMO_TEST_SUITE_
