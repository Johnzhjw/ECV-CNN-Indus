#include "MOP_FRNN_MODEL.h"

//#define MY_DEBUG_TAG_OUTPUT
//#define MY_DEBUG_TAG
//#define MY_DEBUG_TAG2
//#define MY_DEBUG_TAG3

static long debug_count_FRNN = 0;

typedef struct TreeNode_FRNN_MODEL {
    int  level;
    int  my_operator;
    int  flag_operator;
    int  my_terminal;
    int  flag_terminal;
    MY_FLT_TYPE my_val;
    int  num_children;
    struct TreeNode_FRNN_MODEL* parent, * children;
    TreeNode_FRNN_MODEL() : level(-1), my_operator(-1), flag_operator(0), my_terminal(-1), flag_terminal(0), my_val(0),
        num_children(0), parent(NULL), children(NULL)
    {
        //no other stuff
    }
} TreeNode_FRNN_MODEL, * Tree_FRNN_MODEL;

//static float KM_IT2Reduce(OutReduceLayer* oLayer, int i, float** degreesInput);
static MY_FLT_TYPE EIASC_IT2Reduce(OutReduceLayer* oLayer, int i, MY_FLT_TYPE** degreesInput);

void frnnsetup(FRNN* frnn, int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship, int* flagAdapMemship,
               int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
               int typeFuzzySet, int typeRules, int typeInRuleCorNum, int typeTypeReducer, int numFuzzyRules, int numRoughSets,
               int consequenceNodeStatus, int centroid_num_tag, int numInputConsequenceNode,
               MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq,
               int flagConnectStatus, int flagConnectWeight)
{
    frnn->typeFuzzySet = typeFuzzySet;
    frnn->typeRules = typeRules;
    frnn->typeInRuleCorNum = typeInRuleCorNum;
    frnn->typeTypeReducer = typeTypeReducer;
    frnn->consequenceNodeStatus = consequenceNodeStatus;
    frnn->flagConnectStatus = flagConnectStatus;
    frnn->flagConnectWeight = flagConnectWeight;

    frnn->layerNum = 4;
    frnn->numOutput = numOutput;

    int tmp_typeCoding = PARA_CODING_CANDECOMP_PARAFAC;

    frnn->M1 = setupMemberLayer(numInput, inputMin, inputMax, numMemship, flagAdapMemship, typeFuzzySet,
                                tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->F2 = setupFuzzyLayer(numInput, frnn->M1->numMembershipFun, numFuzzyRules, typeFuzzySet, typeRules, typeInRuleCorNum,
                               FLAG_STATUS_OFF,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->R3 = setupRoughLayer(frnn->F2->numRules, numRoughSets, typeFuzzySet,
                               1,
                               tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);
    frnn->O4 = setupOutReduceLayer(frnn->R3->numRoughSets, numOutput, outputMin, outputMax,
                                   typeFuzzySet, typeTypeReducer,
                                   consequenceNodeStatus, centroid_num_tag, numInputConsequenceNode, inputMin_cnsq, inputMax_cnsq,
                                   flagConnectStatus, flagConnectWeight, tmp_typeCoding, MAX_NUM_LOW_RANK_CFRNN_MODEL, 6, 1);

    frnn->e = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));

    frnn->N_sum = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));
    frnn->N_wrong = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));
    frnn->e_sum = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));

    frnn->N_TP = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));
    frnn->N_TN = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));
    frnn->N_FP = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));
    frnn->N_FN = (MY_FLT_TYPE*)malloc(numOutput * sizeof(MY_FLT_TYPE));

    if(typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
        frnn->dataflowMax = frnn->M1->numInput * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->O4->numOutput;
        frnn->connectionMax = frnn->M1->numInput * frnn->F2->numRules +
                              frnn->F2->numRules * frnn->R3->numRoughSets +
                              frnn->R3->numRoughSets * frnn->O4->numOutput;
    } else {
        frnn->dataflowMax = frnn->M1->outputSize * frnn->F2->numRules * frnn->R3->numRoughSets * frnn->O4->numOutput;
        frnn->connectionMax = frnn->M1->outputSize * frnn->F2->numRules +
                              frnn->F2->numRules * frnn->R3->numRoughSets +
                              frnn->R3->numRoughSets * frnn->O4->numOutput;
    }

    return;
}

void frnnfree(FRNN* frnn)
{
    freeMemberLayer(frnn->M1);
    freeFuzzyLayer(frnn->F2);
    freeRoughLayer(frnn->R3);
    freeOutReduceLayer(frnn->O4);

    free(frnn->e);

    free(frnn->N_sum);
    free(frnn->N_wrong);
    free(frnn->e_sum);

    free(frnn->N_TP);
    free(frnn->N_TN);
    free(frnn->N_FP);
    free(frnn->N_FN);

    free(frnn);

    return;
}

void frnninit(FRNN* frnn, double* x, int mode)
{
    int count = 0;
    switch(mode) {
    case INIT_MODE_FRNN:
    case INIT_BP_MODE_FRNN:
    case ASSIGN_MODE_FRNN:
    case OUTPUT_ALL_MODE_FRNN:
    case OUTPUT_DISCRETE_MODE_FRNN:
    case OUTPUT_CONTINUOUS_MODE_FRNN:
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    assignMemberLayer(frnn->M1, &x[count], mode);
    count += frnn->M1->numParaLocal;
    memcpy(frnn->F2->numMembershipFunCur, frnn->M1->numMembershipFunCur, frnn->M1->numInput * sizeof(int));
    assignFuzzyLayer(frnn->F2, &x[count], mode);
    count += frnn->F2->numParaLocal;
    assignRoughLayer(frnn->R3, &x[count], mode);
    count += frnn->R3->numParaLocal;
    assignOutReduceLayer(frnn->O4, &x[count], mode);

    return;
}

void ff_frnn(FRNN* frnn, MY_FLT_TYPE* valIn, MY_FLT_TYPE* valOut, MY_FLT_TYPE** inputConsequenceNode)
{
    MY_FLT_TYPE* dataflowStatus = (MY_FLT_TYPE*)malloc(frnn->M1->numInput * sizeof(MY_FLT_TYPE));
    for(int i = 0; i < frnn->M1->numInput; i++) dataflowStatus[i] = 1;
    ff_memberLayer(frnn->M1, valIn, dataflowStatus);
    free(dataflowStatus);
    ff_fuzzyLayer(frnn->F2, frnn->M1->degreeMembership, frnn->M1->dataflowStatus);
    ff_roughLayer(frnn->R3, frnn->F2->degreeRules, frnn->F2->dataflowStatus);
    if(frnn->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < frnn->O4->numOutput; i++)
            memcpy(frnn->O4->inputConsequenceNode[i], inputConsequenceNode[i], frnn->O4->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
    }
    ff_outReduceLayer(frnn->O4, frnn->R3->degreeRough, frnn->R3->dataflowStatus);
    memcpy(valOut, frnn->O4->valOutputFinal, frnn->O4->numOutput * sizeof(MY_FLT_TYPE));

    return;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
codingCP* setupCodingCP(int flag_adap_rank, int num_dim_CP, int* size_dim_CP, int numLowRankMax,
                        MY_FLT_TYPE para_min, MY_FLT_TYPE para_max)
{
    switch(flag_adap_rank) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_adap_rank %d, exit...\n",
               __FILE__, __LINE__, flag_adap_rank);
        exit(-1);
        break;
    }
    if(num_dim_CP < 2) {
        printf("%s(%d): Invalid value of num_dim_CP %d, exit...\n",
               __FILE__, __LINE__, num_dim_CP);
        exit(-1);
    }
    for(int i = 0; i < num_dim_CP; i++) {
        if(size_dim_CP[i] <= 0) {
            printf("%s(%d): Invalid value of size_dim_CP[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, size_dim_CP[i]);
            exit(-1);
        }
    }
    if(numLowRankMax <= 0 ||
       (numLowRankMax <= 1 && flag_adap_rank == FLAG_STATUS_ON)) {
        printf("%s(%d): Invalid value of numLowRankMax %d, exit...\n",
               __FILE__, __LINE__, numLowRankMax);
        exit(-1);
    }
    if(CHECK_INVALID(para_min) || CHECK_INVALID(para_max)) {
        printf("%s(%d): Invalid values of para_min %lf or para_max %lf, exit...\n",
               __FILE__, __LINE__, para_min, para_max);
        exit(-1);
    }
    if(para_max <= para_min) {
        printf("%s(%d): para_min %lf should be less than para_max %lf, exit...\n",
               __FILE__, __LINE__, para_min, para_max);
        exit(-1);
    }

    codingCP* cdCP = (codingCP*)malloc(1 * sizeof(codingCP));

    cdCP->flag_adap_rank = flag_adap_rank;

    cdCP->num_dim_CP = num_dim_CP;
    cdCP->size_dim_CP = (int*)malloc(cdCP->num_dim_CP * sizeof(int));
    memcpy(cdCP->size_dim_CP, size_dim_CP, cdCP->num_dim_CP * sizeof(int));

    cdCP->numLowRankMax_CP = numLowRankMax;

    cdCP->var_CP_4_w = (MY_FLT_TYPE***)malloc(cdCP->num_dim_CP * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < cdCP->num_dim_CP; i++) {
        cdCP->var_CP_4_w[i] = (MY_FLT_TYPE**)malloc(cdCP->size_dim_CP[i] * sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < cdCP->size_dim_CP[i]; j++) {
            cdCP->var_CP_4_w[i][j] = (MY_FLT_TYPE*)malloc(cdCP->numLowRankMax_CP * sizeof(MY_FLT_TYPE));
        }
    }

    cdCP->para_max = para_max;
    cdCP->para_min = para_min;
    cdCP->para_max_abs = cdCP->para_max;
    if(cdCP->para_max_abs < -cdCP->para_min)
        cdCP->para_max_abs = -cdCP->para_min;
    if(para_max <= 0) {
        printf("%s(%d): The value of ``para_max'', which is %lf now, should be greater than 0, exiting...\n",
               __FILE__, __LINE__, para_max);
    }

    //
    cdCP->numParaLocal = 0;
    cdCP->numParaLocal_disc = 0;
    for(int i = 0; i < cdCP->num_dim_CP; i++)
        cdCP->numParaLocal += cdCP->size_dim_CP[i] * cdCP->numLowRankMax_CP;
    if(cdCP->flag_adap_rank) {
        cdCP->numParaLocal++;
        cdCP->numParaLocal_disc++;
    }

    //
    cdCP->xMin = (MY_FLT_TYPE*)malloc((cdCP->numParaLocal + 1) * sizeof(MY_FLT_TYPE));
    cdCP->xMax = (MY_FLT_TYPE*)malloc((cdCP->numParaLocal + 1) * sizeof(MY_FLT_TYPE));
    cdCP->xType = (int*)malloc((cdCP->numParaLocal + 1) * sizeof(int));

    int count = 0;
    for(int i = 0; i < cdCP->numParaLocal; i++) {
        cdCP->xMin[count] = para_min;
        cdCP->xMax[count] = para_max;
        cdCP->xType[count] = VAR_TYPE_CONTINUOUS;
        count++;
    }
    if(cdCP->flag_adap_rank) {
        cdCP->xMin[count - 1] = 1;
        cdCP->xMax[count - 1] = cdCP->numLowRankMax_CP + 1 - 1e-6;
        cdCP->xType[count - 1] = VAR_TYPE_DISCRETE;
    }

    return cdCP;
}

void assignCodingCP(codingCP* cdCP, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    for(int i = 0; i < cdCP->num_dim_CP; i++) {
        for(int j = 0; j < cdCP->size_dim_CP[i]; j++) {
            for(int k = 0; k < cdCP->numLowRankMax_CP; k++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (cdCP->xMax[count] - cdCP->xMin[count]) + cdCP->xMin[count]);
                    cdCP->var_CP_4_w[i][j][k] = randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    cdCP->var_CP_4_w[i][j][k] = (MY_FLT_TYPE)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = cdCP->var_CP_4_w[i][j][k];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    x[count] = cdCP->var_CP_4_w[i][j][k];
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    break;
                default:
                    printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        }
    }
    if(cdCP->flag_adap_rank) {
        switch(mode) {
        case INIT_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (cdCP->xMax[count] - cdCP->xMin[count]) + cdCP->xMin[count]);
            cdCP->numLowRankCur_CP = (int)randnum;
            break;
        case ASSIGN_MODE_FRNN:
            cdCP->numLowRankCur_CP = (int)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = cdCP->numLowRankCur_CP;
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            x[count] = cdCP->numLowRankCur_CP;
            break;
        default:
            printf("%s(%d): mode error for assignCodingCP - %d, exiting...\n",
                   __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    }
    //
    MY_FLT_TYPE tmp_w = 0;
    for(int m = 0; m < cdCP->numLowRankCur_CP; m++) {
        MY_FLT_TYPE tmp_mul = 1;
        for(int n = 0; n < cdCP->num_dim_CP; n++) {
            tmp_mul *= cdCP->para_max;
        }
        tmp_w += tmp_mul;
    }
    cdCP->max_val = tmp_w;
    if(cdCP->para_min >= 0) {
        tmp_w = 0;
        for(int m = 0; m < cdCP->numLowRankCur_CP; m++) {
            MY_FLT_TYPE tmp_mul = 1;
            for(int n = 0; n < cdCP->num_dim_CP; n++) {
                tmp_mul *= cdCP->para_min;
            }
            tmp_w += tmp_mul;
        }
        cdCP->min_val = tmp_w;
    } else {
        if(fabs(cdCP->para_min) < fabs(cdCP->para_max)) {
            cdCP->min_val = cdCP->max_val * cdCP->para_min / cdCP->para_max;
        } else {
            for(int m = 0; m < cdCP->numLowRankCur_CP; m++) {
                MY_FLT_TYPE tmp_mul = 1;
                for(int n = 0; n < cdCP->num_dim_CP; n++) {
                    tmp_mul *= cdCP->para_max_abs;
                }
                tmp_w += tmp_mul;
            }
            cdCP->min_val = -tmp_w;
            if(cdCP->numLowRankCur_CP % 2 == 0) {
                cdCP->min_val = cdCP->min_val * cdCP->para_max / cdCP->para_max_abs;
            }
        }
    }

    return;
}

void print_para_codingCP(codingCP* cdCP)
{
    printf("codingCP:\n");
    printf("\n");
    for(int i = 0; i < cdCP->num_dim_CP; i++) {
        for(int j = 0; j < cdCP->size_dim_CP[i]; j++) {
            printf("D(%d~%d)", i + 1, j + 1);
            for(int k = 0; k < cdCP->numLowRankMax_CP; k++) {
                printf(",%e", cdCP->var_CP_4_w[i][j][k]);
            }
            printf("\n");
        }
    }
    printf("Cur rank num: %d\n", cdCP->numLowRankCur_CP);
    printf("Cur min max: %e ~ %e\n", cdCP->min_val, cdCP->max_val);
    printf("print para for codingCP done\n========================================\n");
    printf("\n");
}

void resetCodingCP(codingCP* cdCP)
{
    MY_FLT_TYPE val_tar = -100;


    return;
}

void freeCodingCP(codingCP* cdCP)
{
    free(cdCP->xMin);
    free(cdCP->xMax);
    free(cdCP->xType);

    for(int i = 0; i < cdCP->num_dim_CP; i++) {
        for(int j = 0; j < cdCP->size_dim_CP[i]; j++) {
            free(cdCP->var_CP_4_w[i][j]);
        }
        free(cdCP->var_CP_4_w[i]);
    }
    free(cdCP->var_CP_4_w);
    free(cdCP->size_dim_CP);

    free(cdCP);

    return;
}

codingGEP* setupCodingGEP(int GEP_num_input, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int dim_input, MY_FLT_TYPE op_ratio,
                          int flag_logic,
                          int GEP_head_length, int flag_GEP_weight, MY_FLT_TYPE para_min, MY_FLT_TYPE para_max)
{
    if(GEP_num_input <= 0) {
        printf("%s(%d): Invalid value of GEP_num_input %d, exit...\n",
               __FILE__, __LINE__, GEP_num_input);
        exit(-1);
    }
    if(inputMin && inputMax) {
        for(int i = 0; i < GEP_num_input; i++) {
            if(inputMax[i] <= inputMin[i]) {
                printf("%s(%d): Invalid value of inputMax[%d] and inputMin[%d] ~ %lf <= %lf, exit...\n",
                       __FILE__, __LINE__, i, i, inputMax[i], inputMin[i]);
                exit(-1);
            }
        }
    } else if((inputMin == NULL && inputMax) ||
              (inputMin && inputMax == NULL)) {
        printf("%s(%d): Inproper inputs of inputMax and inputMin ~ only one pointer is valid, exit...\n",
               __FILE__, __LINE__);
        exit(-1);
    }
    if(dim_input != 1 && dim_input != 2) {
        printf("%s(%d): The dimension of each input is wrong ~ %d, exit...\n",
               __FILE__, __LINE__, dim_input);
        exit(-1);
    }
    if(op_ratio <= 0 || op_ratio > 1) {
        printf("%s(%d): Invalid value of op_ratio ~ %lf, exit...\n",
               __FILE__, __LINE__, op_ratio);
        exit(-1);
    }
    switch(flag_logic) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_logic %d, exit...\n",
               __FILE__, __LINE__, flag_logic);
        exit(-1);
        break;
    }
    if(GEP_head_length <= 0) {
        printf("%s(%d): Invalid value of GEP_head_length %d, exit...\n",
               __FILE__, __LINE__, GEP_head_length);
        exit(-1);
    }
    switch(flag_GEP_weight) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_GEP_weight %d, exit...\n",
               __FILE__, __LINE__, flag_GEP_weight);
        exit(-1);
        break;
    }
    if(para_min >= para_max) {
        printf("%s(%d): Inproper values of paramin and paramax ~ %lf >= %lf, exit...\n",
               __FILE__, __LINE__, para_min, para_max);
        exit(-1);
    }

    codingGEP* cdGEP = (codingGEP*)malloc(1 * sizeof(codingGEP));

    cdGEP->GEP_num_input = GEP_num_input;
    cdGEP->inputMin = (MY_FLT_TYPE*)malloc(cdGEP->GEP_num_input * sizeof(MY_FLT_TYPE));
    cdGEP->inputMax = (MY_FLT_TYPE*)malloc(cdGEP->GEP_num_input * sizeof(MY_FLT_TYPE));
    if(inputMin)
        memcpy(cdGEP->inputMin, inputMin, cdGEP->GEP_num_input * sizeof(MY_FLT_TYPE));
    else
        for(int i = 0; i < cdGEP->GEP_num_input; i++) cdGEP->inputMin[i] = 0;
    if(inputMax)
        memcpy(cdGEP->inputMax, inputMax, cdGEP->GEP_num_input * sizeof(MY_FLT_TYPE));
    else
        for(int i = 0; i < cdGEP->GEP_num_input; i++) cdGEP->inputMax[i] = 1;
    cdGEP->dim_input = dim_input;

    cdGEP->op_ratio = op_ratio;

    cdGEP->flag_logic = flag_logic;

    cdGEP->GEP_head_length = GEP_head_length < GEP_HEAD_LENGTH_MAX_FRNN_MODEL ? GEP_head_length : GEP_HEAD_LENGTH_MAX_FRNN_MODEL;
    cdGEP->GEP_max_aug_num = 2;
    cdGEP->GEP_tail_length = (cdGEP->GEP_head_length * (cdGEP->GEP_max_aug_num - 1) + 1);
    cdGEP->GEP_weight_num = (2 * cdGEP->GEP_head_length);
    cdGEP->flag_GEP_weight = flag_GEP_weight;
    cdGEP->numPara_coding_GEP = cdGEP->GEP_head_length + cdGEP->GEP_tail_length;
    if(cdGEP->flag_GEP_weight == FLAG_STATUS_ON)
        cdGEP->numPara_coding_GEP += cdGEP->GEP_weight_num;

    cdGEP->para_coding_GEP = (MY_FLT_TYPE*)malloc(cdGEP->numPara_coding_GEP * sizeof(MY_FLT_TYPE));

    //
    cdGEP->numParaLocal = cdGEP->numPara_coding_GEP;
    cdGEP->numParaLocal_disc = cdGEP->GEP_head_length + cdGEP->GEP_tail_length;

    //
    cdGEP->xMin = (MY_FLT_TYPE*)malloc((cdGEP->numParaLocal + 1) * sizeof(MY_FLT_TYPE));
    cdGEP->xMax = (MY_FLT_TYPE*)malloc((cdGEP->numParaLocal + 1) * sizeof(MY_FLT_TYPE));
    cdGEP->xType = (int*)malloc((cdGEP->numParaLocal + 1) * sizeof(int));

    int count = 0;

    for(int k = 0; k < cdGEP->numPara_coding_GEP; k++) {
        if(k < cdGEP->GEP_head_length) {
            cdGEP->xMin[count] = 0;
            cdGEP->xMax[count] = 1;
            cdGEP->xType[count] = VAR_TYPE_CONTINUOUS; // TODO -var type changed !!!!!!
            count++;
        } else if(k < cdGEP->GEP_head_length + cdGEP->GEP_tail_length) {
            cdGEP->xMin[count] = 0;
            cdGEP->xMax[count] = cdGEP->GEP_num_input - 1e-6;
            cdGEP->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        } else {
            if(cdGEP->flag_GEP_weight == FLAG_STATUS_ON) {
                if(k < cdGEP->GEP_head_length + cdGEP->GEP_tail_length + cdGEP->GEP_weight_num) {
                    cdGEP->xMin[count] = para_min;
                    cdGEP->xMax[count] = para_max;
                    cdGEP->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                } else {
                    printf("%s(%d): parameter wrong, please check...\n", __FILE__, __LINE__);
                    exit(-1);
                }
            } else {
                printf("%s(%d): parameter wrong, please check...\n", __FILE__, __LINE__);
                exit(-1);
            }
        }
    }

    return cdGEP;
}

void assignCodingGEP(codingGEP* cdGEP, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    for(int k = 0; k < cdGEP->numPara_coding_GEP; k++) {
        switch(mode) {
        case INIT_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (cdGEP->xMax[count] - cdGEP->xMin[count]) + cdGEP->xMin[count]);
            cdGEP->para_coding_GEP[k] = randnum;
            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
            break;
        case ASSIGN_MODE_FRNN:
            cdGEP->para_coding_GEP[k] = (MY_FLT_TYPE)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = cdGEP->para_coding_GEP[k];
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            if(k < cdGEP->GEP_head_length && k >= cdGEP->GEP_head_length + cdGEP->GEP_tail_length)
                x[count] = cdGEP->para_coding_GEP[k];
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            if(k < cdGEP->GEP_head_length + cdGEP->GEP_tail_length && k >= cdGEP->GEP_head_length)
                x[count] = cdGEP->para_coding_GEP[k];
            break;
        default:
            printf("%s(%d): mode error for assignCodingGEP - %d, exiting...\n",
                   __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    }
    if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
        for(int k = 0; k < MAX_NUM_PARA_CODING_GEP_FRNN_MODEL; k++) {
            cdGEP->check_level[k] = 0;
            cdGEP->check_parent_ind[k] = -1;
            cdGEP->check_op[k] = -1;
            cdGEP->check_vInd[k] = -1;
            cdGEP->check_children_num[k] = 0;
            for(int d = 0; d < 2; d++)
                cdGEP->check_valR[k][d] = 0;
        }
        memcpy(cdGEP->check_para, cdGEP->para_coding_GEP, cdGEP->numPara_coding_GEP * sizeof(MY_FLT_TYPE));
        cdGEP->check_head = 0;
        cdGEP->check_tail = 1;
        int cur_level = 0;
        while(cdGEP->check_head < cdGEP->check_tail && cdGEP->check_head < cdGEP->GEP_head_length) {
            if(cdGEP->flag_logic == FLAG_STATUS_OFF) {
                int tmp_int = (int)(GEP_OP_F_NUM * cdGEP->check_para[cdGEP->check_head] / cdGEP->op_ratio);
                if(tmp_int >= 0 && tmp_int < GEP_OP_F_NUM) {
                    switch(tmp_int) {
                    case GEP_OP_F_ADD:
                    case GEP_OP_F_SUBTRACT:
                    case GEP_OP_F_MULTIPLY:
                    case GEP_OP_F_DIVIDE:
                    case GEP_OP_F_MAX:
                    case GEP_OP_F_MIN:
                    case GEP_OP_F_MEAN:
                        cdGEP->check_children_num[cdGEP->check_head] = 2;
                        break;
                    case GEP_OP_F_SIN:
                    case GEP_OP_F_COS:
                    case GEP_OP_F_EXP:
                    case GEP_OP_F_SQUARE:
                    case GEP_OP_F_SQUARE_ROOT:
                    case GEP_OP_F_LOG:
                        cdGEP->check_children_num[cdGEP->check_head] = 1;
                        break;
                    default:
                        printf("%s(%d): Unknown GEP_COMPN_FUNC_TYPE - %d ~ %d, exiting...\n",
                               __FILE__, __LINE__, tmp_int, cdGEP->check_head);
                        exit(-147801);
                        break;
                    }
                    cdGEP->check_op[cdGEP->check_head] = tmp_int;
                } else {
                    if(tmp_int < 0) {
                        printf("%s(%d): para error - %d - %d, exiting...\n",
                               __FILE__, __LINE__, cdGEP->check_head, tmp_int);
                        exit(-18745);
                    }
                    cdGEP->check_children_num[cdGEP->check_head] = 0;
                    cdGEP->check_op[cdGEP->check_head] = -1;
                }
            } else {
                int tmp_int = (int)(GEP_R_F_NUM * cdGEP->check_para[cdGEP->check_head] / cdGEP->op_ratio);
                if(tmp_int < GEP_R_F_NUM) {
                    switch(tmp_int) {
                    case GEP_R_F_AND:
                    case GEP_R_F_OR:
                        cdGEP->check_children_num[cdGEP->check_head] = 2;
                        break;
                    case GEP_R_F_NOT:
                    case GEP_R_F_SQUARE:
                    case GEP_R_F_SQUARE_ROOT:
                        cdGEP->check_children_num[cdGEP->check_head] = 1;
                        break;
                    default:
                        printf("%s(%d): Unknown GEP_RULE_FUNC_TYPE - %d ~ %d, exiting...\n",
                               __FILE__, __LINE__, tmp_int, cdGEP->check_head);
                        exit(-1);
                        break;
                    }
                    cdGEP->check_op[cdGEP->check_head] = tmp_int;
                } else {
                    if(tmp_int < 0) {
                        printf("%s(%d): para error - %d - %d, exiting...\n",
                               __FILE__, __LINE__, cdGEP->check_head, tmp_int);
                        exit(-18745);
                    }
                    cdGEP->check_children_num[cdGEP->check_head] = 0;
                    cdGEP->check_op[cdGEP->check_head] = -1;
                }
            }
            for(int k = 0; k < cdGEP->check_children_num[cdGEP->check_head]; k++) {
                cdGEP->check_parent_ind[cdGEP->check_tail] = cdGEP->check_head;
                cdGEP->check_level[cdGEP->check_tail] = cdGEP->check_level[cdGEP->check_head] + 1;
                cdGEP->check_tail++;
            }
            cdGEP->check_head++;
        }
        if(cdGEP->flag_GEP_weight == FLAG_STATUS_ON) {
            memcpy(cdGEP->check_weights,
                   &cdGEP->check_para[cdGEP->GEP_head_length + cdGEP->GEP_tail_length],
                   cdGEP->GEP_weight_num * sizeof(MY_FLT_TYPE));
        } else {
            for(int k = 0; k < cdGEP->GEP_weight_num; k++) {
                cdGEP->check_weights[k] = 1;
            }
        }
        getConnectGEP(cdGEP);
    }

    return;
}

void print_para_codingGEP(codingGEP* cdGEP)
{
    printf("codingGEP:\n");
    printf("\n");
    printf("GEP input num: %d\n", cdGEP->GEP_num_input);
    printf("GEP dim num: %d\n", cdGEP->dim_input);
    printf("GEP op_ratio: %e\n", cdGEP->op_ratio);
    printf("GEP head_length: %d\n", cdGEP->GEP_head_length);
    printf("GEP max_aug_num: %d\n", cdGEP->GEP_max_aug_num);
    printf("GEP tail_length: %d\n", cdGEP->GEP_tail_length);
    printf("GEP weight_num: %d\n", cdGEP->GEP_weight_num);
    printf("GEP flag_weight: %d\n", cdGEP->flag_GEP_weight);
    printf("GEP flag_weight: %d\n", cdGEP->flag_GEP_weight);
    printf("GEP para num: %d\n", cdGEP->numPara_coding_GEP);
    printf("GEP check_head: %d\n", cdGEP->check_head);
    printf("GEP check_tail: %d\n", cdGEP->check_tail);
    printf("GEP check para:\n");
    printf("GEP check level:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%d\t", cdGEP->check_level[i]);
    printf("\nGEP check parent ind:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%d\t", cdGEP->check_parent_ind[i]);
    printf("\nGEP check children num:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%d\t", cdGEP->check_children_num[i]);
    printf("\nGEP check vInd:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%d\t", cdGEP->check_vInd[i]);
    printf("\nGEP check op:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%d\t", cdGEP->check_op[i]);
    printf("\nGEP check valR:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) {
        printf("(%e", cdGEP->check_valR[i][0]);
        for(int j = 1; j < cdGEP->dim_input; j++) printf(" %e", cdGEP->check_valR[i][j]);
        printf(")\t");
    }
    printf("\nGEP check para:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%e\t", cdGEP->check_para[i]);
    printf("\nGEP check weights:\n");
    for(int i = 0; i < cdGEP->check_tail; i++) printf("%e\t", cdGEP->check_weights[i]);
    printf("\nprint para for codingGEP done\n========================================\n");
    printf("\n");
}

void freeCodingGEP(codingGEP* cdGEP)
{
    free(cdGEP->inputMin);
    free(cdGEP->inputMax);

    free(cdGEP->xMin);
    free(cdGEP->xMax);
    free(cdGEP->xType);

    free(cdGEP->para_coding_GEP);

    free(cdGEP);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
ConvolutionLayer* setupConvLayer(int inputHeightMax, int inputWidthMax, int channelsIn, int channelsOut, int channelsInMax,
                                 int typeKernelCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight,
                                 int flag_adapKernelSize, int default_kernelHeight, int default_kernelWidth,
                                 int kernelHeightMin, int kernelHeightMax, int kernelWidthMin, int kernelWidthMax,
                                 int flag_kernelFlagAdap, int default_kernelFlag,
                                 int flag_actFuncTypeAdap, int default_actFuncType,
                                 int flag_paddingTypeAdap, int default_paddingType)
{
    if(inputHeightMax <= 0) {
        printf("%s(%d): Invalid value of inputHeightMax %d, exit...\n",
               __FILE__, __LINE__, inputHeightMax);
        exit(-1);
    }
    if(inputWidthMax <= 0) {
        printf("%s(%d): Invalid value of inputWidthMax %d, exit...\n",
               __FILE__, __LINE__, inputWidthMax);
        exit(-1);
    }
    if(channelsIn <= 0) {
        printf("%s(%d): Invalid value of channelsIn %d, exit...\n",
               __FILE__, __LINE__, channelsIn);
        exit(-1);
    }
    if(channelsOut <= 0) {
        printf("%s(%d): Invalid value of channelsOut %d, exit...\n",
               __FILE__, __LINE__, channelsOut);
        exit(-1);
    }
    if(channelsInMax <= 0) {
        printf("%s(%d): Invalid value of channelsInMax %d, exit...\n",
               __FILE__, __LINE__, channelsInMax);
        exit(-1);
    }
    switch(typeKernelCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeKernelCoding %d, exit...\n",
               __FILE__, __LINE__, typeKernelCoding);
        exit(-1);
        break;
    }
    switch(flag_adapKernelSize) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_adapKernelSize %d, exit...\n",
               __FILE__, __LINE__, flag_adapKernelSize);
        exit(-1);
        break;
    }
    if(default_kernelHeight <= 0) {
        printf("%s(%d): Invalid value of default_kernelHeight %d, exit...\n",
               __FILE__, __LINE__, default_kernelHeight);
        exit(-1);
    }
    if(default_kernelWidth <= 0) {
        printf("%s(%d): Invalid value of default_kernelWidth %d, exit...\n",
               __FILE__, __LINE__, default_kernelWidth);
        exit(-1);
    }
    if(kernelHeightMin <= 0) {
        printf("%s(%d): Invalid value of kernelHeightMin %d, exit...\n",
               __FILE__, __LINE__, kernelHeightMin);
        exit(-1);
    }
    if(kernelHeightMax <= 0) {
        printf("%s(%d): Invalid value of kernelHeightMax %d, exit...\n",
               __FILE__, __LINE__, kernelHeightMax);
        exit(-1);
    }
    if(kernelHeightMax <= kernelHeightMin) {
        printf("%s(%d): kernelHeightMin %d should be less than kernelHeightMax %d, exit...\n",
               __FILE__, __LINE__, kernelHeightMin, kernelHeightMax);
        exit(-1);
    }
    if(kernelWidthMin <= 0) {
        printf("%s(%d): Invalid value of kernelWidthMin %d, exit...\n",
               __FILE__, __LINE__, kernelWidthMin);
        exit(-1);
    }
    if(kernelWidthMax <= 0) {
        printf("%s(%d): Invalid value of kernelWidthMax %d, exit...\n",
               __FILE__, __LINE__, kernelWidthMax);
        exit(-1);
    }
    if(kernelWidthMax <= kernelWidthMin) {
        printf("%s(%d): kernelWidthMin %d should be less than kernelWidthMax %d, exit...\n",
               __FILE__, __LINE__, kernelWidthMin, kernelWidthMax);
        exit(-1);
    }
    switch(flag_kernelFlagAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_kernelFlagAdap %d, exit...\n",
               __FILE__, __LINE__, flag_kernelFlagAdap);
        exit(-1);
        break;
    }
    switch(default_kernelFlag) {
    case KERNEL_FLAG_SKIP:
    case KERNEL_FLAG_OPERATE:
    case KERNEL_FLAG_COPY:
        break;
    default:
        printf("%s(%d): Unknown default_kernelFlag %d, exit...\n",
               __FILE__, __LINE__, default_kernelFlag);
        exit(-1);
        break;
    }
    switch(flag_actFuncTypeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_actFuncTypeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_actFuncTypeAdap);
        exit(-1);
        break;
    }
    switch(default_actFuncType) {
    case ACT_FUNC_RELU:
    case ACT_FUNC_LEAKYRELU:
    case ACT_FUNC_SIGMA:
    case ACT_FUNC_TANH:
    case ACT_FUNC_ELU:
        break;
    default:
        printf("%s(%d): Unknown default_actFuncType %d, exit...\n",
               __FILE__, __LINE__, default_actFuncType);
        exit(-1);
        break;
    }
    switch(flag_paddingTypeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_paddingTypeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_paddingTypeAdap);
        exit(-1);
        break;
    }
    switch(default_paddingType) {
    case PADDING_SAME:
    case PADDING_VALID:
        break;
    default:
        printf("%s(%d): Unknown default_paddingType %d, exit...\n",
               __FILE__, __LINE__, default_paddingType);
        exit(-1);
        break;
    }

    ConvolutionLayer* cLayer = (ConvolutionLayer*)malloc(1 * sizeof(ConvolutionLayer));

    cLayer->inputHeightMax = inputHeightMax;
    cLayer->inputWidthMax = inputWidthMax;

    cLayer->channelsIn = channelsIn;
    cLayer->channelsOut = channelsOut;
#if LAYER_SKIP_TAG_CFRNN_MODEL == 1
    cLayer->channelsInMax = channelsInMax;
    cLayer->channelsOutMax = cLayer->channelsInMax;
    if(cLayer->channelsOutMax < cLayer->channelsOut) cLayer->channelsOutMax = cLayer->channelsOut;
#else
    cLayer->channelsInMax = channelsIn;
    cLayer->channelsOutMax = channelsOut;
#endif

    cLayer->typeKernelCoding = typeKernelCoding;
    switch(cLayer->typeKernelCoding) {
    case PARA_CODING_DIRECT:
        cLayer->flag_kernelCoding_CP = 0;
        cLayer->flag_kernelCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        cLayer->flag_kernelCoding_CP = 1;
        cLayer->flag_kernelCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        cLayer->flag_kernelCoding_CP = 0;
        cLayer->flag_kernelCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeKernelCoding - %d, exiting...\n",
               __FILE__, __LINE__, cLayer->typeKernelCoding);
        exit(-1000);
        break;
    }

    cLayer->flag_adapKernelSize = flag_adapKernelSize;
    cLayer->kernelHeightDefault = default_kernelHeight;
    cLayer->kernelWidthDefault = default_kernelWidth;

    cLayer->kernelHeightMin = kernelHeightMin;
    cLayer->kernelHeightMax = kernelHeightMax;
    cLayer->kernelWidthMin = kernelWidthMin;
    cLayer->kernelWidthMax = kernelWidthMax;

    if(cLayer->flag_adapKernelSize == 0) {
        cLayer->kernelHeightMin = cLayer->kernelHeightDefault;
        cLayer->kernelHeightMax = cLayer->kernelHeightDefault;
        cLayer->kernelWidthMin = cLayer->kernelWidthDefault;
        cLayer->kernelWidthMax = cLayer->kernelWidthDefault;
    }

    cLayer->flag_kernelFlagAdap = flag_kernelFlagAdap;
    cLayer->kernelFlagDefault = default_kernelFlag;

    cLayer->flag_actFuncTypeAdap = flag_actFuncTypeAdap;
    cLayer->kernelTypeDefault = default_actFuncType;

    cLayer->inputHeight = (int*)malloc(cLayer->channelsInMax * sizeof(int));
    cLayer->inputWidth = (int*)malloc(cLayer->channelsInMax * sizeof(int));

    cLayer->kernelHeight = (int**)malloc(cLayer->channelsOut * sizeof(int*));
    cLayer->kernelWidth = (int**)malloc(cLayer->channelsOut * sizeof(int*));
    cLayer->kernelFlag = (int**)malloc(cLayer->channelsOut * sizeof(int*));
    cLayer->kernelFlagCountAll = (int**)malloc(cLayer->channelsOut * sizeof(int*));
    cLayer->kernelType = (int*)malloc(cLayer->channelsOut * sizeof(int));
    for(int i = 0; i < cLayer->channelsOut; i++) {
        cLayer->kernelHeight[i] = (int*)malloc(cLayer->channelsInMax * sizeof(int));
        cLayer->kernelWidth[i] = (int*)malloc(cLayer->channelsInMax * sizeof(int));
        cLayer->kernelFlag[i] = (int*)malloc(cLayer->channelsInMax * sizeof(int));
        cLayer->kernelFlagCountAll[i] = (int*)malloc(cLayer->channelsInMax * sizeof(int));
    }

    cLayer->kernelData = (MY_FLT_TYPE****)malloc(cLayer->channelsOut * sizeof(MY_FLT_TYPE***));
    cLayer->kernelDelta = (MY_FLT_TYPE****)malloc(cLayer->channelsOut * sizeof(MY_FLT_TYPE***));
    for(int i = 0; i < cLayer->channelsOut; i++) {
        cLayer->kernelData[i] = (MY_FLT_TYPE***)malloc(cLayer->channelsInMax * sizeof(MY_FLT_TYPE**));
        cLayer->kernelDelta[i] = (MY_FLT_TYPE***)malloc(cLayer->channelsInMax * sizeof(MY_FLT_TYPE**));
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            cLayer->kernelData[i][j] = (MY_FLT_TYPE**)malloc(cLayer->kernelHeightMax * sizeof(MY_FLT_TYPE*));
            cLayer->kernelDelta[i][j] = (MY_FLT_TYPE**)malloc(cLayer->kernelHeightMax * sizeof(MY_FLT_TYPE*));
            for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                cLayer->kernelData[i][j][h] = (MY_FLT_TYPE*)malloc(cLayer->kernelWidthMax * sizeof(MY_FLT_TYPE));
                cLayer->kernelDelta[i][j][h] = (MY_FLT_TYPE*)malloc(cLayer->kernelWidthMax * sizeof(MY_FLT_TYPE));
            }
        }
    }

    cLayer->flag_paddingTypeAdap = flag_paddingTypeAdap;
    cLayer->paddingTypeDefault = default_paddingType;
    cLayer->paddingType = (int**)malloc(cLayer->channelsOut * sizeof(int*));
    for(int i = 0; i < cLayer->channelsOut; i++) {
        cLayer->paddingType[i] = (int*)malloc(cLayer->channelsInMax * sizeof(int));
    }

    cLayer->biasData = (MY_FLT_TYPE*)malloc(cLayer->channelsOut * sizeof(MY_FLT_TYPE));
    cLayer->biasDelta = (MY_FLT_TYPE*)malloc(cLayer->channelsOut * sizeof(MY_FLT_TYPE));

    cLayer->featureMapHeight = (int*)malloc(cLayer->channelsOutMax * sizeof(int));
    cLayer->featureMapWidth = (int*)malloc(cLayer->channelsOutMax * sizeof(int));
    if(cLayer->flag_paddingTypeAdap == 0 && cLayer->paddingTypeDefault == PADDING_VALID) {
        cLayer->featureMapHeightMax = inputHeightMax - 2 * (int)(cLayer->kernelHeightMin / 2);
        cLayer->featureMapWidthMax = inputWidthMax - 2 * (int)(cLayer->kernelWidthMin / 2);
    } else {
        cLayer->featureMapHeightMax = inputHeightMax;
        cLayer->featureMapWidthMax = inputWidthMax;
    }
    cLayer->featureMapData = (MY_FLT_TYPE***)malloc(cLayer->channelsOutMax * sizeof(MY_FLT_TYPE**));
    cLayer->featureMapDelta = (MY_FLT_TYPE***)malloc(cLayer->channelsOutMax * sizeof(MY_FLT_TYPE**));
    cLayer->featureMapDerivative = (MY_FLT_TYPE***)malloc(cLayer->channelsOutMax * sizeof(MY_FLT_TYPE**));
    cLayer->featureMapTag = (int***)malloc(cLayer->channelsOutMax * sizeof(int**));
    for(int i = 0; i < cLayer->channelsOutMax; i++) {
        cLayer->featureMapData[i] = (MY_FLT_TYPE**)malloc(cLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        cLayer->featureMapDelta[i] = (MY_FLT_TYPE**)malloc(cLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        cLayer->featureMapDerivative[i] = (MY_FLT_TYPE**)malloc(cLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        cLayer->featureMapTag[i] = (int**)malloc(cLayer->featureMapHeightMax * sizeof(int*));
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            cLayer->featureMapData[i][h] = (MY_FLT_TYPE*)malloc(cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            cLayer->featureMapDelta[i][h] = (MY_FLT_TYPE*)malloc(cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            cLayer->featureMapDerivative[i][h] = (MY_FLT_TYPE*)malloc(cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            cLayer->featureMapTag[i][h] = (int*)malloc(cLayer->featureMapWidthMax * sizeof(int));
        }
    }

    cLayer->dataflowStatus = (MY_FLT_TYPE***)malloc(cLayer->channelsOutMax * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < cLayer->channelsOutMax; i++) {
        cLayer->dataflowStatus[i] = (MY_FLT_TYPE**)malloc(cLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            cLayer->dataflowStatus[i][h] = (MY_FLT_TYPE*)malloc(cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    //
    int count = 0;
    int count_disc = 0;
    if(cLayer->flag_adapKernelSize) {
        count += cLayer->channelsOut * cLayer->channelsInMax;
        count_disc += cLayer->channelsOut * cLayer->channelsInMax;
        count += cLayer->channelsOut * cLayer->channelsInMax;
        count_disc += cLayer->channelsOut * cLayer->channelsInMax;
    }
    if(cLayer->flag_kernelFlagAdap) {
        count += cLayer->channelsOut * cLayer->channelsInMax;
        count_disc += cLayer->channelsOut * cLayer->channelsInMax;
    }
    if(cLayer->flag_actFuncTypeAdap) {
        count += cLayer->channelsOut;
        count_disc += cLayer->channelsOut;
    }
    if(cLayer->flag_kernelCoding_CP) {
        int num_dim_CP = 4;
        int size_dim_CP[4];
        size_dim_CP[0] = cLayer->channelsOut;
        size_dim_CP[1] = cLayer->channelsInMax;
        size_dim_CP[2] = cLayer->kernelHeightMax;
        size_dim_CP[3] = cLayer->kernelWidthMax;
        cLayer->cdCP = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                     PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += cLayer->cdCP->numParaLocal;
        count_disc += cLayer->cdCP->numParaLocal_disc;
    } else if(cLayer->flag_kernelCoding_GEP) {
        int GEP_num_input = 4;
        MY_FLT_TYPE inputMin[4] = { 0, 0, 0, 0 };
        MY_FLT_TYPE inputMax[4];
        inputMax[0] = cLayer->channelsOut;
        inputMax[1] = cLayer->channelsInMax;
        inputMax[2] = cLayer->kernelHeightMax;
        inputMax[3] = cLayer->kernelWidthMax;
        cLayer->cdGEP = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                       GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += cLayer->cdGEP->numParaLocal;
        count_disc += cLayer->cdGEP->numParaLocal_disc;
    } else {
        count += cLayer->channelsOut * cLayer->channelsInMax * cLayer->kernelHeightMax * cLayer->kernelWidthMax;
    }
    count += cLayer->channelsOut;
    if(cLayer->flag_paddingTypeAdap) {
        count += cLayer->channelsOut;
        count_disc += cLayer->channelsOut;
    }
    cLayer->numParaLocal = count;
    cLayer->numParaLocal_disc = count_disc;

    //
    cLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    cLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    cLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(cLayer->flag_adapKernelSize) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->xMin[count] = cLayer->kernelHeightMin / 2;
                cLayer->xMax[count] = cLayer->kernelHeightMax / 2 + 1 - 1e-6;
                cLayer->xType[count] = VAR_TYPE_DISCRETE;
                count++;
                cLayer->xMin[count] = cLayer->kernelWidthMin / 2;
                cLayer->xMax[count] = cLayer->kernelWidthMax / 2 + 1 - 1e-6;
                cLayer->xType[count] = VAR_TYPE_DISCRETE;
                count++;
            }
        }
    }
    if(cLayer->flag_kernelFlagAdap) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->xMin[count] = KERNEL_FLAG_OPERATE;
                cLayer->xMax[count] = KERNEL_FLAG_COPY + 1 - 1e-6;
                cLayer->xType[count] = VAR_TYPE_DISCRETE;
                count++;
            }
        }
    }
    if(cLayer->flag_actFuncTypeAdap) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            cLayer->xMin[count] = 0;
            cLayer->xMax[count] = NUM_ACT_FUNC_TYPE - 1e-6;
            cLayer->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        }
    }
    if(cLayer->flag_kernelCoding_CP) {
        memcpy(&cLayer->xMin[count], cLayer->cdCP->xMin, cLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&cLayer->xMax[count], cLayer->cdCP->xMax, cLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&cLayer->xType[count], cLayer->cdCP->xType, cLayer->cdCP->numParaLocal * sizeof(int));
        count += cLayer->cdCP->numParaLocal;
    } else if(cLayer->flag_kernelCoding_GEP) {
        memcpy(&cLayer->xMin[count], cLayer->cdGEP->xMin, cLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&cLayer->xMax[count], cLayer->cdGEP->xMax, cLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&cLayer->xType[count], cLayer->cdGEP->xType, cLayer->cdGEP->numParaLocal * sizeof(int));
        count += cLayer->cdGEP->numParaLocal;
    } else {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                for(int k = 0; k < cLayer->kernelHeightMax * cLayer->kernelWidthMax; k++) {
                    cLayer->xMin[count] = PARA_MIN_KERNEL_DATA_CFRNN_MODEL;
                    cLayer->xMax[count] = PARA_MAX_KERNEL_DATA_CFRNN_MODEL;
                    cLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                }
            }
        }
    }
    for(int i = 0; i < cLayer->channelsOut; i++) {
        cLayer->xMin[count] = PARA_MIN_KERNEL_DATA_CFRNN_MODEL *
                              cLayer->channelsInMax * cLayer->kernelHeightMax * cLayer->kernelWidthMax;
        cLayer->xMax[count] = PARA_MAX_KERNEL_DATA_CFRNN_MODEL *
                              cLayer->channelsInMax * cLayer->kernelHeightMax * cLayer->kernelWidthMax;
        cLayer->xType[count] = VAR_TYPE_CONTINUOUS;
        count++;
    }
    if(cLayer->flag_paddingTypeAdap) {
        cLayer->xMin[count] = PADDING_SAME;
        cLayer->xMax[count] = NUM_PADDING_TYPE - 1e-6;
        cLayer->xType[count] = VAR_TYPE_DISCRETE;
        count++;
    }

    return cLayer;
}

void assignConvLayer(ConvolutionLayer* cLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    MY_FLT_TYPE tmp_para_cnt = 0;
    if(cLayer->flag_adapKernelSize) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                    cLayer->kernelHeight[i][j] = 2 * (int)randnum + 1;
                    count++;
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                    cLayer->kernelWidth[i][j] = 2 * (int)randnum + 1;
                    count++;
                    //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                    break;
                case ASSIGN_MODE_FRNN:
                    cLayer->kernelHeight[i][j] = 2 * (int)x[count++] + 1;
                    cLayer->kernelWidth[i][j] = 2 * (int)x[count++] + 1;
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count++] = cLayer->kernelHeight[i][j] / 2;
                    x[count++] = cLayer->kernelWidth[i][j] / 2;
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    count++;
                    count++;
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count++] = cLayer->kernelHeight[i][j] / 2;
                    x[count++] = cLayer->kernelWidth[i][j] / 2;
                    break;
                default:
                    printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                tmp_para_cnt += cLayer->kernelHeight[i][j] * cLayer->kernelWidth[i][j];
            }
        }
    } else {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->kernelHeight[i][j] = cLayer->kernelHeightDefault;
                cLayer->kernelWidth[i][j] = cLayer->kernelWidthDefault;
            }
        }
        tmp_para_cnt = cLayer->kernelHeightDefault * cLayer->kernelWidthDefault * cLayer->channelsOut * cLayer->channelsInMax;
    }

    if(cLayer->flag_kernelFlagAdap) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                    cLayer->kernelFlag[i][j] = (int)randnum;
                    //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                    break;
                case ASSIGN_MODE_FRNN:
                    cLayer->kernelFlag[i][j] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = cLayer->kernelFlag[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = cLayer->kernelFlag[i][j];
                    break;
                default:
                    printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
                if(cLayer->kernelFlag[i][j] != KERNEL_FLAG_OPERATE)
                    tmp_para_cnt -= cLayer->kernelHeight[i][j] * cLayer->kernelWidth[i][j];
            }
        }
    } else {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->kernelFlag[i][j] = cLayer->kernelFlagDefault;
                if(cLayer->kernelFlag[i][j] != KERNEL_FLAG_OPERATE)
                    tmp_para_cnt -= cLayer->kernelHeight[i][j] * cLayer->kernelWidth[i][j];
            }
        }
    }
    if(cLayer->flag_actFuncTypeAdap) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            switch(mode) {
            case INIT_MODE_FRNN:
            case INIT_BP_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                        (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                cLayer->kernelType[i] = (int)randnum;
                //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                break;
            case ASSIGN_MODE_FRNN:
                cLayer->kernelType[i] = (int)x[count];
                break;
            case OUTPUT_ALL_MODE_FRNN:
                x[count] = cLayer->kernelType[i];
                break;
            case OUTPUT_CONTINUOUS_MODE_FRNN:
                break;
            case OUTPUT_DISCRETE_MODE_FRNN:
                x[count] = cLayer->kernelType[i];
                break;
            default:
                printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                exit(-1000);
                break;
            }
            count++;
        }
    } else {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            cLayer->kernelType[i] = cLayer->kernelTypeDefault;
        }
    }

    if(cLayer->flag_kernelCoding_CP) {
        assignCodingCP(cLayer->cdCP, &x[count], mode);
        count += cLayer->cdCP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < cLayer->channelsOut; i++) {
                for(int j = 0; j < cLayer->channelsInMax; j++) {
                    for(int k = 0; k < cLayer->kernelHeightMax; k++) {
                        for(int l = 0; l < cLayer->kernelWidthMax; l++) {
                            int tmp_in[4] = { i, j, k, l };
                            MY_FLT_TYPE tmp_val = 0.0;
                            decodingCP(cLayer->cdCP, tmp_in, &tmp_val);
                            cLayer->kernelData[i][j][k][l] = tmp_val;
                        }
                    }
                }
            }
        }
    } else if(cLayer->flag_kernelCoding_GEP) {
        assignCodingGEP(cLayer->cdGEP, &x[count], mode);
        count += cLayer->cdGEP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < cLayer->channelsOut; i++) {
                for(int j = 0; j < cLayer->channelsInMax; j++) {
                    for(int k = 0; k < cLayer->kernelHeightMax; k++) {
                        for(int l = 0; l < cLayer->kernelWidthMax; l++) {
                            MY_FLT_TYPE tmp_O = i + 1;// (float)((i + 1.0) / (cLayer->channelsOut));
                            MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (cLayer->channelsInMax));
                            MY_FLT_TYPE tmp_X = k + 1;// (float)((k + 1.0) / (cLayer->kernelHeightMax));
                            MY_FLT_TYPE tmp_Y = l + 1;// (float)((l + 1.0) / (cLayer->kernelWidthMax));
                            MY_FLT_TYPE tmp_in[4] = { tmp_O, tmp_I, tmp_X, tmp_Y };
                            MY_FLT_TYPE tmp_val;
                            decodingGEP(cLayer->cdGEP, tmp_in, &tmp_val);
                            cLayer->kernelData[i][j][k][l] = tmp_val;
                        }
                    }
                }
            }
        }
    } else {
        tmp_para_cnt /= cLayer->channelsInMax * cLayer->channelsOut;
        tmp_para_cnt *= (cLayer->channelsInMax + cLayer->channelsOut);
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                    for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                        for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                    (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                            cLayer->kernelData[i][j][h][w] = randnum;
                            count++;
                        }
                    }
                    //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                    break;
                case INIT_BP_MODE_FRNN:
                    for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                        for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) - 0.5) * 2;
                            cLayer->kernelData[i][j][h][w] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(tmp_para_cnt));
                            count++;
                        }
                    }
                    //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                    break;
                case ASSIGN_MODE_FRNN:
                    for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                        for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                            cLayer->kernelData[i][j][h][w] = (MY_FLT_TYPE)x[count];
                            count++;
                        }
                    }
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                        for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                            x[count] = cLayer->kernelData[i][j][h][w];
                            count++;
                        }
                    }
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                        for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                            x[count] = cLayer->kernelData[i][j][h][w];
                            count++;
                        }
                    }
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    count += cLayer->kernelHeightMax * cLayer->kernelWidthMax;
                    break;
                default:
                    printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
            }
        }
    }

    for(int i = 0; i < cLayer->channelsOut; i++) {
        switch(mode) {
        case INIT_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
            cLayer->biasData[i] = randnum;
            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
            break;
        case INIT_BP_MODE_FRNN:
            cLayer->biasData[i] = 0;
            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
            break;
        case ASSIGN_MODE_FRNN:
            cLayer->biasData[i] = (MY_FLT_TYPE)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = cLayer->biasData[i];
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            x[count] = cLayer->biasData[i];
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            break;
        default:
            printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    }

    if(cLayer->flag_paddingTypeAdap) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            int tmp_paddingType = cLayer->paddingType[i][0];
            switch(mode) {
            case INIT_MODE_FRNN:
            case INIT_BP_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                        (cLayer->xMax[count] - cLayer->xMin[count]) + cLayer->xMin[count]);
                tmp_paddingType = (int)randnum;
                //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                break;
            case ASSIGN_MODE_FRNN:
                tmp_paddingType = (int)x[count];
                break;
            case OUTPUT_ALL_MODE_FRNN:
                x[count] = cLayer->paddingType[i][0];
                break;
            case OUTPUT_CONTINUOUS_MODE_FRNN:
                break;
            case OUTPUT_DISCRETE_MODE_FRNN:
                x[count] = cLayer->paddingType[i][0];
                break;
            default:
                printf("%s(%d): mode error for assignConvLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                exit(-1000);
                break;
            }
            count++;
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->paddingType[i][j] = tmp_paddingType;
            }
        }
    } else {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                cLayer->paddingType[i][j] = cLayer->paddingTypeDefault;
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Convolutional layer:\n");
    fprintf(debug_fpt, "\n");
    for(int i = 0; i < cLayer->channelsOut; i++) {
        fprintf(debug_fpt, "\nkernel,%d,type,%d\n", i + 1, cLayer->kernelType[i]);
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            fprintf(debug_fpt, "kernel-id,%d,%d\n", i + 1, j + 1);
            fprintf(debug_fpt, "kernel-size,%d,%d\n", cLayer->kernelHeight[i][j], cLayer->kernelWidth[i][j]);
            fprintf(debug_fpt, "kernel-flag,%d\n", cLayer->kernelFlag[i][j]);
            fprintf(debug_fpt, "kernel-data\n");
            for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                    fprintf(debug_fpt, ",%f", cLayer->kernelData[i][j][h][w]);
                }
                fprintf(debug_fpt, "\n");
            }
            fprintf(debug_fpt, "kernel-paddingType,%d\n", cLayer->paddingType[i][j]);
        }
        fprintf(debug_fpt, "biasData,%f\n", cLayer->biasData[i]);
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_convLayer(ConvolutionLayer* cLayer)
{
    if(cLayer->flag_kernelCoding_CP) {
        print_para_codingCP(cLayer->cdCP);
    } else if(cLayer->flag_kernelCoding_GEP) {
        print_para_codingGEP(cLayer->cdGEP);
    }

    printf("Convolutional layer:\n");
    printf("\n");
    for(int i = 0; i < cLayer->channelsOut; i++) {
        printf("\nkernel,%d,type,%d\n", i + 1, cLayer->kernelType[i]);
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            printf("kernel-id,%d,%d\n", i + 1, j + 1);
            printf("kernel-size,%d,%d\n", cLayer->kernelHeight[i][j], cLayer->kernelWidth[i][j]);
            printf("kernel-flag,%d\n", cLayer->kernelFlag[i][j]);
            printf("kernel-data\n");
            for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                    printf(",%f", cLayer->kernelData[i][j][h][w]);
                }
                printf("\n");
            }
            printf("kernel-paddingType,%d\n", cLayer->paddingType[i][j]);
        }
        printf("biasData,%f\n", cLayer->biasData[i]);
    }
    printf("print para for Convolutional layer done\n========================================\n");
    printf("\n");
}

void resetConvLayer(ConvolutionLayer* cLayer)
{
    return;
}

void freeConvLayer(ConvolutionLayer* cLayer)
{
    if(cLayer->flag_kernelCoding_CP) {
        freeCodingCP(cLayer->cdCP);
    } else if(cLayer->flag_kernelCoding_GEP) {
        freeCodingGEP(cLayer->cdGEP);
    }

    free(cLayer->inputHeight);
    free(cLayer->inputWidth);

    for(int i = 0; i < cLayer->channelsOut; i++) {
        free(cLayer->kernelHeight[i]);
        free(cLayer->kernelWidth[i]);
        free(cLayer->kernelFlag[i]);
        free(cLayer->kernelFlagCountAll[i]);
    }
    free(cLayer->kernelHeight);
    free(cLayer->kernelWidth);
    free(cLayer->kernelFlag);
    free(cLayer->kernelFlagCountAll);
    free(cLayer->kernelType);

    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                free(cLayer->kernelData[i][j][h]);
                free(cLayer->kernelDelta[i][j][h]);
            }
            free(cLayer->kernelData[i][j]);
            free(cLayer->kernelDelta[i][j]);
        }
        free(cLayer->kernelData[i]);
        free(cLayer->kernelDelta[i]);
    }
    free(cLayer->kernelData);
    free(cLayer->kernelDelta);

    for(int i = 0; i < cLayer->channelsOut; i++) {
        free(cLayer->paddingType[i]);
    }
    free(cLayer->paddingType);

    free(cLayer->biasData);
    free(cLayer->biasDelta);

    free(cLayer->featureMapHeight);
    free(cLayer->featureMapWidth);

    for(int i = 0; i < cLayer->channelsOutMax; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            free(cLayer->featureMapData[i][h]);
            free(cLayer->featureMapDelta[i][h]);
            free(cLayer->featureMapDerivative[i][h]);
            free(cLayer->featureMapTag[i][h]);
        }
        free(cLayer->featureMapData[i]);
        free(cLayer->featureMapDelta[i]);
        free(cLayer->featureMapDerivative[i]);
        free(cLayer->featureMapTag[i]);
    }
    free(cLayer->featureMapData);
    free(cLayer->featureMapDelta);
    free(cLayer->featureMapDerivative);
    free(cLayer->featureMapTag);

    for(int i = 0; i < cLayer->channelsOutMax; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            free(cLayer->dataflowStatus[i][h]);
        }
        free(cLayer->dataflowStatus[i]);
    }
    free(cLayer->dataflowStatus);

    free(cLayer->xMin);
    free(cLayer->xMax);
    free(cLayer->xType);

    free(cLayer);
}

PoolLayer* setupPoolLayer(int inputHeightMax, int inputWidthMax, int channelsInOut, int channelsInOutMax,
                          int flag_poolSizeAdap, int default_poolHeight, int default_poolWidth,
                          int poolHeightMin, int poolHeightMax, int poolWidthMin, int poolWidthMax,
                          int flag_poolTypeAdap, int default_poolType)
{
    if(inputHeightMax <= 0) {
        printf("%s(%d): Invalid value of inputHeightMax %d, exit...\n",
               __FILE__, __LINE__, inputHeightMax);
        exit(-1);
    }
    if(inputWidthMax <= 0) {
        printf("%s(%d): Invalid value of inputWidthMax %d, exit...\n",
               __FILE__, __LINE__, inputWidthMax);
        exit(-1);
    }
    if(channelsInOut <= 0) {
        printf("%s(%d): Invalid value of channelsInOut %d, exit...\n",
               __FILE__, __LINE__, channelsInOut);
        exit(-1);
    }
    if(channelsInOutMax <= 0) {
        printf("%s(%d): Invalid value of channelsInOutMax %d, exit...\n",
               __FILE__, __LINE__, channelsInOutMax);
        exit(-1);
    }
    switch(flag_poolSizeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_poolSizeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_poolSizeAdap);
        exit(-1);
        break;
    }
    if(default_poolHeight <= 0) {
        printf("%s(%d): Invalid value of default_poolHeight %d, exit...\n",
               __FILE__, __LINE__, default_poolHeight);
        exit(-1);
    }
    if(default_poolWidth <= 0) {
        printf("%s(%d): Invalid value of default_poolWidth %d, exit...\n",
               __FILE__, __LINE__, default_poolWidth);
        exit(-1);
    }
    if(poolHeightMin <= 0) {
        printf("%s(%d): Invalid value of poolHeightMin %d, exit...\n",
               __FILE__, __LINE__, poolHeightMin);
        exit(-1);
    }
    if(poolHeightMax <= 0) {
        printf("%s(%d): Invalid value of poolHeightMax %d, exit...\n",
               __FILE__, __LINE__, poolHeightMax);
        exit(-1);
    }
    if(poolHeightMax <= poolHeightMin) {
        printf("%s(%d): poolHeightMin %d should be less than poolHeightMax %d, exit...\n",
               __FILE__, __LINE__, poolHeightMin, poolHeightMax);
        exit(-1);
    }
    if(poolWidthMin <= 0) {
        printf("%s(%d): Invalid value of poolWidthMin %d, exit...\n",
               __FILE__, __LINE__, poolWidthMin);
        exit(-1);
    }
    if(poolWidthMax <= 0) {
        printf("%s(%d): Invalid value of poolWidthMax %d, exit...\n",
               __FILE__, __LINE__, poolWidthMax);
        exit(-1);
    }
    if(poolWidthMax <= poolWidthMin) {
        printf("%s(%d): poolWidthMin %d should be less than poolWidthMax %d, exit...\n",
               __FILE__, __LINE__, poolWidthMin, poolWidthMax);
        exit(-1);
    }
    switch(flag_poolTypeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_poolTypeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_poolTypeAdap);
        exit(-1);
        break;
    }
    switch(default_poolType) {
    case POOL_AVE:
    case POOL_MAX:
    case POOL_MIN:
        break;
    default:
        printf("%s(%d): Unknown default_poolType %d, exit...\n",
               __FILE__, __LINE__, default_poolType);
        exit(-1);
        break;
    }

    PoolLayer* pLayer = (PoolLayer*)malloc(1 * sizeof(PoolLayer));

    pLayer->inputHeightMax = inputHeightMax;
    pLayer->inputWidthMax = inputWidthMax;

    pLayer->channelsInOut = channelsInOut;
#if LAYER_SKIP_TAG_CFRNN_MODEL == 1
    pLayer->channelsInOutMax = channelsInOutMax;
    if(pLayer->channelsInOutMax < pLayer->channelsInOut) pLayer->channelsInOutMax = pLayer->channelsInOut;
#else
    pLayer->channelsInOutMax = channelsInOut;
#endif

    pLayer->flag_poolSizeAdap = flag_poolSizeAdap;
    pLayer->poolHeightDefault = default_poolHeight;
    pLayer->poolWidthDefault = default_poolWidth;

    pLayer->poolHeightMin = poolHeightMin;
    pLayer->poolHeightMax = poolHeightMax;
    pLayer->poolWidthMin = poolWidthMin;
    pLayer->poolWidthMax = poolWidthMax;

    if(pLayer->flag_poolSizeAdap == 0) {
        pLayer->poolHeightMin = pLayer->poolHeightDefault;
        pLayer->poolHeightMax = pLayer->poolHeightDefault;
        pLayer->poolWidthMin = pLayer->poolWidthDefault;
        pLayer->poolWidthMax = pLayer->poolWidthDefault;
    }

    pLayer->inputHeight = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));
    pLayer->inputWidth = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));

    pLayer->poolHeight = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));
    pLayer->poolWidth = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));
    pLayer->poolFlag = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));

    pLayer->flag_poolTypeAdap = flag_poolTypeAdap;
    pLayer->poolTypeDefault = default_poolType;
    pLayer->poolType = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));

    pLayer->featureMapHeight = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));
    pLayer->featureMapWidth = (int*)malloc(pLayer->channelsInOutMax * sizeof(int));
    pLayer->featureMapHeightMax = (inputHeightMax + pLayer->poolHeightMin - 1) / pLayer->poolHeightMin;
    pLayer->featureMapWidthMax = (inputWidthMax + pLayer->poolWidthMin - 1) / pLayer->poolWidthMin;
    pLayer->featureMapData = (MY_FLT_TYPE***)malloc(pLayer->channelsInOutMax * sizeof(MY_FLT_TYPE**));
    pLayer->featureMapDelta = (MY_FLT_TYPE***)malloc(pLayer->channelsInOutMax * sizeof(MY_FLT_TYPE**));
    pLayer->featureMapDerivative = (MY_FLT_TYPE***)malloc(pLayer->channelsInOutMax * sizeof(MY_FLT_TYPE**));
    pLayer->featureMapPos = (int***)malloc(pLayer->channelsInOutMax * sizeof(int**));
    pLayer->featureMapTag = (int***)malloc(pLayer->channelsInOutMax * sizeof(int**));
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        pLayer->featureMapData[i] = (MY_FLT_TYPE**)malloc(pLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        pLayer->featureMapDelta[i] = (MY_FLT_TYPE**)malloc(pLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        pLayer->featureMapDerivative[i] = (MY_FLT_TYPE**)malloc(pLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        pLayer->featureMapPos[i] = (int**)malloc(pLayer->featureMapHeightMax * sizeof(int*));
        pLayer->featureMapTag[i] = (int**)malloc(pLayer->featureMapHeightMax * sizeof(int*));
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            pLayer->featureMapData[i][h] = (MY_FLT_TYPE*)malloc(pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            pLayer->featureMapDelta[i][h] = (MY_FLT_TYPE*)malloc(pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            pLayer->featureMapDerivative[i][h] = (MY_FLT_TYPE*)malloc(pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            pLayer->featureMapPos[i][h] = (int*)malloc(pLayer->featureMapWidthMax * sizeof(int));
            pLayer->featureMapTag[i][h] = (int*)malloc(pLayer->featureMapWidthMax * sizeof(int));
        }
    }

    pLayer->dataflowStatus = (MY_FLT_TYPE***)malloc(pLayer->channelsInOutMax * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        pLayer->dataflowStatus[i] = (MY_FLT_TYPE**)malloc(pLayer->featureMapHeightMax * sizeof(MY_FLT_TYPE*));
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            pLayer->dataflowStatus[i][h] = (MY_FLT_TYPE*)malloc(pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    //
    int count = 0;
    int count_disc = 0;
    //count += pLayer->channelsInOut;
    if(pLayer->flag_poolSizeAdap) {
        count += 2;
        count_disc += 2;
    }
    if(pLayer->flag_poolTypeAdap) {
        count += pLayer->channelsInOutMax;
        count_disc += pLayer->channelsInOutMax;
    }
    pLayer->numParaLocal = count;
    pLayer->numParaLocal_disc = count_disc;

    //
    pLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    pLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    pLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    //for (int i = 0; i < pLayer->channelsInOut; i++) {
    //    pLayer->xMin[count] = 0;
    //    pLayer->xMax[count] = 2 - 1e-6;
    //    count++;
    //}
    if(pLayer->flag_poolSizeAdap) {
        // height
        pLayer->xMin[count] = pLayer->poolHeightMin;
        pLayer->xMax[count] = pLayer->poolHeightMax + 1 - 1e-6;
        pLayer->xType[count] = VAR_TYPE_DISCRETE;
        count++;
        pLayer->xMin[count] = pLayer->poolWidthMin;
        pLayer->xMax[count] = pLayer->poolWidthMax + 1 - 1e-6;
        pLayer->xType[count] = VAR_TYPE_DISCRETE;
        count++;
    }
    if(pLayer->flag_poolTypeAdap) {
        for(int i = 0; i < pLayer->channelsInOutMax; i++) {
            pLayer->xMin[count] = 0;
            pLayer->xMax[count] = NUM_POOL_TYPE - 1e-6;
            pLayer->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        }
    }

    return pLayer;
}

void assignPoolLayer(PoolLayer* pLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    if(pLayer->flag_poolSizeAdap) {
        switch(mode) {
        case INIT_MODE_FRNN:
        case INIT_BP_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (pLayer->xMax[count] - pLayer->xMin[count]) + pLayer->xMin[count]);
            pLayer->poolHeightAll = (int)randnum;
            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
            break;
        case ASSIGN_MODE_FRNN:
            pLayer->poolHeightAll = (int)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = pLayer->poolHeightAll;
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            x[count] = pLayer->poolHeightAll;
            break;
        default:
            printf("%s(%d): mode error for assignPoolLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
        switch(mode) {
        case INIT_MODE_FRNN:
        case INIT_BP_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (pLayer->xMax[count] - pLayer->xMin[count]) + pLayer->xMin[count]);
            pLayer->poolWidthAll = (int)randnum;
            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
            break;
        case ASSIGN_MODE_FRNN:
            pLayer->poolWidthAll = (int)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = pLayer->poolWidthAll;
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            x[count] = pLayer->poolWidthAll;
            break;
        default:
            printf("%s(%d): mode error for assignPoolLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    } else {
        pLayer->poolHeightAll = pLayer->poolHeightDefault;
        pLayer->poolWidthAll = pLayer->poolWidthDefault;
    }
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        pLayer->poolHeight[i] = pLayer->poolHeightAll;
        pLayer->poolWidth[i] = pLayer->poolWidthAll;
    }
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        pLayer->poolFlag[i] = 1;
    }
    //// uniform or separately !!!!!!!!!!!!!!!!!!!!!!!
    //for (int i = 0; i < pLayer->channelsInOut; i++) {
    //    switch (mode) {
    //    case INIT_MODE_FRNN:
    //        randnum = (float)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
    //            (pLayer->xMax[count] - pLayer->xMin[count]) + pLayer->xMin[count]);
    //        pLayer->kernelFlag[i] = (int)randnum;
    //        //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
    //        break;
    //    case ASSIGN_MODE_FRNN:
    //        pLayer->kernelFlag[i] = (int)x[count];
    //        break;
    //    case OUTPUT_MODE_FRNN:
    //        x[count] = pLayer->kernelFlag[i];
    //        break;
    //    default:
    //        printf("%s(%d): mode error for assignPoolLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
    //        exit(-1000);
    //        break;
    //    }
    //    count++;
    //}
    if(pLayer->flag_poolTypeAdap) {
        for(int i = 0; i < pLayer->channelsInOutMax; i++) {
            switch(mode) {
            case INIT_MODE_FRNN:
            case INIT_BP_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                        (pLayer->xMax[count] - pLayer->xMin[count]) + pLayer->xMin[count]);
                pLayer->poolType[i] = (int)randnum;
                //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                break;
            case ASSIGN_MODE_FRNN:
                pLayer->poolType[i] = (int)x[count];
                break;
            case OUTPUT_ALL_MODE_FRNN:
                x[count] = pLayer->poolType[i];
                break;
            case OUTPUT_CONTINUOUS_MODE_FRNN:
                break;
            case OUTPUT_DISCRETE_MODE_FRNN:
                x[count] = pLayer->poolType[i];
                break;
            default:
                printf("%s(%d): mode error for assignPoolLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                exit(-1000);
                break;
            }
            count++;
        }
    } else {
        for(int i = 0; i < pLayer->channelsInOutMax; i++) {
            pLayer->poolType[i] = pLayer->poolTypeDefault;
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Pooling layer:\n");
    fprintf(debug_fpt, "\n");
    fprintf(debug_fpt, "id,H,W,Flag,Type\n");
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        fprintf(debug_fpt, "%d,", i + 1);
        fprintf(debug_fpt, "%d,%d,%d,%d\n",
                pLayer->poolHeight[i], pLayer->poolWidth[i], pLayer->poolFlag[i], pLayer->poolType[i]);
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_poolLayer(PoolLayer* pLayer)
{
    printf("Pooling layer:\n");
    printf("\n");
    printf("id,H,W,Flag,Type\n");
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        printf("%d,", i + 1);
        printf("%d,%d,%d,%d\n",
               pLayer->poolHeight[i], pLayer->poolWidth[i], pLayer->poolFlag[i], pLayer->poolType[i]);
    }
    printf("print para for Pooling layer done\n========================================\n");
    printf("\n");
}

void freePoolLayer(PoolLayer* pLayer)
{
    free(pLayer->inputHeight);
    free(pLayer->inputWidth);

    free(pLayer->poolHeight);
    free(pLayer->poolWidth);
    free(pLayer->poolFlag);
    free(pLayer->poolType);

    free(pLayer->featureMapHeight);
    free(pLayer->featureMapWidth);

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            free(pLayer->featureMapData[i][h]);
            free(pLayer->featureMapDelta[i][h]);
            free(pLayer->featureMapDerivative[i][h]);
            free(pLayer->featureMapPos[i][h]);
            free(pLayer->featureMapTag[i][h]);
        }
        free(pLayer->featureMapData[i]);
        free(pLayer->featureMapDelta[i]);
        free(pLayer->featureMapDerivative[i]);
        free(pLayer->featureMapPos[i]);
        free(pLayer->featureMapTag[i]);
    }
    free(pLayer->featureMapData);
    free(pLayer->featureMapDelta);
    free(pLayer->featureMapDerivative);
    free(pLayer->featureMapPos);
    free(pLayer->featureMapTag);

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            free(pLayer->dataflowStatus[i][h]);
        }
        free(pLayer->dataflowStatus[i]);
    }
    free(pLayer->dataflowStatus);

    free(pLayer->xMin);
    free(pLayer->xMax);
    free(pLayer->xType);

    free(pLayer);
}

InterCPCLayer* setupInterCFCLayer(int preFeatureMapChannels, int preFeatureMapHeightMax, int preFeatureMapWidthMax,
                                  int numOutput,
                                  int flagActFunc, int flag_actFuncTypeAdap, int default_actFuncType,
                                  int flag_connectAdap, int typeConnectParaCoding,
                                  int layerNum, int* numNodesAll, int numLowRankMax,
                                  int GEP_head_length, int flag_GEP_weight,
                                  int flag_wt_positive, int flag_normalize_outData)
{
    if(preFeatureMapChannels <= 0) {
        printf("%s(%d): Invalid value of preFeatureMapChannels %d, exit...\n",
               __FILE__, __LINE__, preFeatureMapChannels);
        exit(-1);
    }
    if(preFeatureMapHeightMax <= 0) {
        printf("%s(%d): Invalid value of preFeatureMapHeightMax %d, exit...\n",
               __FILE__, __LINE__, preFeatureMapHeightMax);
        exit(-1);
    }
    if(preFeatureMapWidthMax <= 0) {
        printf("%s(%d): Invalid value of preFeatureMapWidthMax %d, exit...\n",
               __FILE__, __LINE__, preFeatureMapWidthMax);
        exit(-1);
    }
    if(numOutput <= 0) {
        printf("%s(%d): Invalid value of numOutput %d, exit...\n",
               __FILE__, __LINE__, numOutput);
        exit(-1);
    }
    switch(flagActFunc) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flagActFunc %d, exit...\n",
               __FILE__, __LINE__, flagActFunc);
        exit(-1);
        break;
    }
    switch(flag_actFuncTypeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_actFuncTypeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_actFuncTypeAdap);
        exit(-1);
        break;
    }
    switch(default_actFuncType) {
    case ACT_FUNC_RELU:
    case ACT_FUNC_LEAKYRELU:
    case ACT_FUNC_SIGMA:
    case ACT_FUNC_TANH:
    case ACT_FUNC_ELU:
        break;
    default:
        printf("%s(%d): Unknown default_actFuncType %d, exit...\n",
               __FILE__, __LINE__, default_actFuncType);
        exit(-1);
        break;
    }
    switch(flag_connectAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_connectAdap %d, exit...\n",
               __FILE__, __LINE__, flag_connectAdap);
        exit(-1);
        break;
    }
    switch(typeConnectParaCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
    case PARA_CODING_NN:
        break;
    default:
        printf("%s(%d): Unknown typeConnectParaCoding %d, exit...\n",
               __FILE__, __LINE__, typeConnectParaCoding);
        exit(-1);
        break;
    }
    if(typeConnectParaCoding == PARA_CODING_NN) {
        if(layerNum <= 1) {
            printf("%s(%d): Invalid value of layerNum %d, exit...\n",
                   __FILE__, __LINE__, layerNum);
            exit(-1);
        }
        if((flag_connectAdap == FLAG_STATUS_ON && numNodesAll[layerNum - 1] != 2) ||
           (flag_connectAdap == FLAG_STATUS_OFF && numNodesAll[layerNum - 1] != 1)) {
            printf("%s(%d): Invalid value of numNodesAll[layerNum - 1] %d, exit...\n",
                   __FILE__, __LINE__, numNodesAll[layerNum - 1]);
            exit(-1);
        }
    }
    switch(flag_wt_positive) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_wt_positive %d, exit...\n",
               __FILE__, __LINE__, flag_wt_positive);
        exit(-1);
        break;
    }
    switch(flag_normalize_outData) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_normalize_outData %d, exit...\n",
               __FILE__, __LINE__, flag_normalize_outData);
        exit(-1);
        break;
    }

    InterCPCLayer* icfcLayer = (InterCPCLayer*)malloc(1 * sizeof(InterCPCLayer));

    icfcLayer->numOutput = numOutput;

    icfcLayer->preFeatureMapChannels = preFeatureMapChannels;
    icfcLayer->preFeatureMapHeightMax = preFeatureMapHeightMax;
    icfcLayer->preFeatureMapWidthMax = preFeatureMapWidthMax;

    icfcLayer->preInputHeight = (int*)malloc(icfcLayer->preFeatureMapChannels * sizeof(int));
    icfcLayer->preInputWidth = (int*)malloc(icfcLayer->preFeatureMapChannels * sizeof(int));

    icfcLayer->flagActFunc = flagActFunc;
    icfcLayer->flag_actFuncTypeAdap = flag_actFuncTypeAdap;
    icfcLayer->actFuncTypeDefault = default_actFuncType;
    if(icfcLayer->flagActFunc == FLAG_STATUS_ON)
        icfcLayer->actFuncType = (int*)malloc(icfcLayer->numOutput * sizeof(int));

    icfcLayer->flag_connectAdap = flag_connectAdap;

    icfcLayer->typeConnectParaCoding = typeConnectParaCoding;
    switch(icfcLayer->typeConnectParaCoding) {
    case PARA_CODING_DIRECT:
        icfcLayer->flag_connectCoding_CP = 0;
        icfcLayer->flag_connectCoding_GEP = 0;
        icfcLayer->flag_NN4Para = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        icfcLayer->flag_connectCoding_CP = 1;
        icfcLayer->flag_connectCoding_GEP = 0;
        icfcLayer->flag_NN4Para = 0;
        break;
    case PARA_CODING_GEP:
        icfcLayer->flag_connectCoding_CP = 0;
        icfcLayer->flag_connectCoding_GEP = 1;
        icfcLayer->flag_NN4Para = 0;
        break;
    case PARA_CODING_NN:
        icfcLayer->flag_connectCoding_CP = 0;
        icfcLayer->flag_connectCoding_GEP = 0;
        icfcLayer->flag_NN4Para = 1;
        break;
    default:
        printf("%s(%d): Unknown typeConnectParaCoding - %d, exiting...\n",
               __FILE__, __LINE__, icfcLayer->typeConnectParaCoding);
        exit(-1000);
        break;
    }

    icfcLayer->flag_wt_positive = flag_wt_positive;
    icfcLayer->flag_normalize_outData = flag_normalize_outData;

    icfcLayer->connectStatusAll = (int****)malloc(icfcLayer->numOutput * sizeof(int***));
    icfcLayer->connectWeightAll = (MY_FLT_TYPE****)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE***));
    icfcLayer->connectWtDeltaAll = (MY_FLT_TYPE****)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE***));
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        icfcLayer->connectStatusAll[i] = (int***)malloc(icfcLayer->preFeatureMapChannels * sizeof(int**));
        icfcLayer->connectWeightAll[i] = (MY_FLT_TYPE***)malloc(icfcLayer->preFeatureMapChannels * sizeof(MY_FLT_TYPE**));
        icfcLayer->connectWtDeltaAll[i] = (MY_FLT_TYPE***)malloc(icfcLayer->preFeatureMapChannels * sizeof(MY_FLT_TYPE**));
        for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
            icfcLayer->connectStatusAll[i][j] = (int**)malloc(icfcLayer->preFeatureMapHeightMax * sizeof(int*));
            icfcLayer->connectWeightAll[i][j] = (MY_FLT_TYPE**)malloc(icfcLayer->preFeatureMapHeightMax * sizeof(MY_FLT_TYPE*));
            icfcLayer->connectWtDeltaAll[i][j] = (MY_FLT_TYPE**)malloc(icfcLayer->preFeatureMapHeightMax * sizeof(MY_FLT_TYPE*));
            for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                icfcLayer->connectStatusAll[i][j][k] = (int*)malloc(icfcLayer->preFeatureMapWidthMax * sizeof(int));
                icfcLayer->connectWeightAll[i][j][k] = (MY_FLT_TYPE*)malloc(icfcLayer->preFeatureMapWidthMax * sizeof(MY_FLT_TYPE));
                icfcLayer->connectWtDeltaAll[i][j][k] = (MY_FLT_TYPE*)malloc(icfcLayer->preFeatureMapWidthMax * sizeof(MY_FLT_TYPE));
            }
        }
    }
    icfcLayer->connectCountAll = (int*)malloc(icfcLayer->numOutput * sizeof(int));

    icfcLayer->biasData = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    icfcLayer->biasDelta = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));

    icfcLayer->outputData = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    icfcLayer->outputDelta = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    icfcLayer->outputDerivative = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));

    icfcLayer->dataflowStatus = (MY_FLT_TYPE*)malloc(icfcLayer->numOutput * sizeof(MY_FLT_TYPE));

    //
    int count = 0;
    int count_disc = 0;
    if(icfcLayer->flagActFunc == FLAG_STATUS_ON &&
       icfcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
        count += icfcLayer->numOutput;
        count_disc += icfcLayer->numOutput;
    }
    if(icfcLayer->flag_NN4Para) {
        icfcLayer->NN4Para = (MLP_mine*)malloc(1 * sizeof(MLP_mine));
        icfcLayer->NN4Para->layerNum = layerNum;
        icfcLayer->NN4Para->numNodesAll = (int*)malloc(icfcLayer->NN4Para->layerNum * sizeof(int));
        memcpy(icfcLayer->NN4Para->numNodesAll, numNodesAll, icfcLayer->NN4Para->layerNum * sizeof(int));
        icfcLayer->NN4Para->numOutput = icfcLayer->NN4Para->numNodesAll[icfcLayer->NN4Para->layerNum - 1];
        icfcLayer->NN4Para->LayersPnt = (FCLayer**)malloc((icfcLayer->NN4Para->layerNum - 1) * sizeof(FCLayer*));
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            int numInputMax;
            if(i == 0) {
                numInputMax = icfcLayer->NN4Para->numNodesAll[0];
            } else {
                numInputMax = icfcLayer->NN4Para->LayersPnt[i - 1]->numOutputMax;
            }
            int tmp_flagActFunc = 1;
            int tmp_flag_actFuncTypeAdap = 0;
            int tmp_default_actFuncType = ACT_FUNC_LEAKYRELU;
            int tmp_flag_connectAdap = 0;
            //if(i < icfcLayer->NN4Para->layerNum - 2) flagActFunc = 1;
            icfcLayer->NN4Para->LayersPnt[i] = setupFCLayer(icfcLayer->NN4Para->numNodesAll[i], numInputMax,
                                               icfcLayer->NN4Para->numNodesAll[i + 1],
                                               tmp_flagActFunc, tmp_flag_actFuncTypeAdap, tmp_default_actFuncType,
                                               tmp_flag_connectAdap);
        }
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            count += icfcLayer->NN4Para->LayersPnt[i]->numParaLocal;
            count_disc += icfcLayer->NN4Para->LayersPnt[i]->numParaLocal_disc;
        }
    } else if(icfcLayer->flag_connectCoding_CP) {
        int num_dim_CP = 4;
        int size_dim_CP[4];
        size_dim_CP[0] = icfcLayer->numOutput;
        size_dim_CP[1] = icfcLayer->preFeatureMapChannels;
        size_dim_CP[2] = icfcLayer->preFeatureMapHeightMax;
        size_dim_CP[3] = icfcLayer->preFeatureMapWidthMax;
        icfcLayer->cdCP_w = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                          PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += icfcLayer->cdCP_w->numParaLocal;
        count_disc += icfcLayer->cdCP_w->numParaLocal_disc;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            icfcLayer->cdCP_c = setupCodingCP(0, num_dim_CP, size_dim_CP, 1,
                                              PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
            count += icfcLayer->cdCP_c->numParaLocal;
            count_disc += icfcLayer->cdCP_c->numParaLocal_disc;
        }
    } else if(icfcLayer->flag_connectCoding_GEP) {
        int GEP_num_input = 4;
        MY_FLT_TYPE inputMin[4] = { 0, 0, 0, 0 };
        MY_FLT_TYPE inputMax[4];
        inputMax[0] = icfcLayer->numOutput;
        inputMax[1] = icfcLayer->preFeatureMapChannels;
        inputMax[2] = icfcLayer->preFeatureMapHeightMax;
        inputMax[3] = icfcLayer->preFeatureMapWidthMax;
        icfcLayer->cdGEP_w = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                            GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += icfcLayer->cdGEP_w->numParaLocal;
        count_disc += icfcLayer->cdGEP_w->numParaLocal_disc;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            icfcLayer->cdGEP_c = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                                GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
            count += icfcLayer->cdGEP_c->numParaLocal;
            count_disc += icfcLayer->cdGEP_c->numParaLocal_disc;
        }
    } else {
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            count += icfcLayer->numOutput * icfcLayer->preFeatureMapChannels * icfcLayer->preFeatureMapHeightMax *
                     icfcLayer->preFeatureMapWidthMax;
            count_disc += icfcLayer->numOutput * icfcLayer->preFeatureMapChannels * icfcLayer->preFeatureMapHeightMax *
                          icfcLayer->preFeatureMapWidthMax;
        }
        count += icfcLayer->numOutput * icfcLayer->preFeatureMapChannels * icfcLayer->preFeatureMapHeightMax *
                 icfcLayer->preFeatureMapWidthMax;
    }
    count += icfcLayer->numOutput;
    icfcLayer->numParaLocal = count;
    icfcLayer->numParaLocal_disc = count_disc;

    //
    icfcLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    icfcLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    icfcLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(icfcLayer->flagActFunc == FLAG_STATUS_ON &&
       icfcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < icfcLayer->numOutput; i++) {
            icfcLayer->xMin[count] = 0;
            icfcLayer->xMax[count] = NUM_ACT_FUNC_TYPE - 1e-6;
            icfcLayer->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        }
    }
    if(icfcLayer->flag_NN4Para) {
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            memcpy(&icfcLayer->xMin[count], icfcLayer->NN4Para->LayersPnt[i]->xMin,
                   icfcLayer->NN4Para->LayersPnt[i]->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xMax[count], icfcLayer->NN4Para->LayersPnt[i]->xMax,
                   icfcLayer->NN4Para->LayersPnt[i]->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xType[count], icfcLayer->NN4Para->LayersPnt[i]->xType,
                   icfcLayer->NN4Para->LayersPnt[i]->numParaLocal * sizeof(int));
            count += icfcLayer->NN4Para->LayersPnt[i]->numParaLocal;
        }
    } else if(icfcLayer->flag_connectCoding_CP) {
        memcpy(&icfcLayer->xMin[count], icfcLayer->cdCP_w->xMin, icfcLayer->cdCP_w->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&icfcLayer->xMax[count], icfcLayer->cdCP_w->xMax, icfcLayer->cdCP_w->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&icfcLayer->xType[count], icfcLayer->cdCP_w->xType, icfcLayer->cdCP_w->numParaLocal * sizeof(int));
        count += icfcLayer->cdCP_w->numParaLocal;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            memcpy(&icfcLayer->xMin[count], icfcLayer->cdCP_c->xMin, icfcLayer->cdCP_c->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xMax[count], icfcLayer->cdCP_c->xMax, icfcLayer->cdCP_c->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xType[count], icfcLayer->cdCP_c->xType, icfcLayer->cdCP_c->numParaLocal * sizeof(int));
            count += icfcLayer->cdCP_c->numParaLocal;
        }
    } else if(icfcLayer->flag_connectCoding_GEP) {
        memcpy(&icfcLayer->xMin[count], icfcLayer->cdGEP_w->xMin, icfcLayer->cdGEP_w->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&icfcLayer->xMax[count], icfcLayer->cdGEP_w->xMax, icfcLayer->cdGEP_w->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&icfcLayer->xType[count], icfcLayer->cdGEP_w->xType, icfcLayer->cdGEP_w->numParaLocal * sizeof(int));
        count += icfcLayer->cdGEP_w->numParaLocal;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            memcpy(&icfcLayer->xMin[count], icfcLayer->cdGEP_c->xMin, icfcLayer->cdGEP_c->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xMax[count], icfcLayer->cdGEP_c->xMax, icfcLayer->cdGEP_c->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&icfcLayer->xType[count], icfcLayer->cdGEP_c->xType, icfcLayer->cdGEP_c->numParaLocal * sizeof(int));
            count += icfcLayer->cdGEP_c->numParaLocal;
        }
    } else {
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            icfcLayer->xMin[count] = 0;
                            icfcLayer->xMax[count] = 2 - 1e-6;
                            icfcLayer->xType[count] = VAR_TYPE_BINARY;
                            count++;
                        }
                    }
                }
            }
        }
    }
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
            for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                    if(icfcLayer->flag_wt_positive)
                        icfcLayer->xMin[count] = 0;
                    else
                        icfcLayer->xMin[count] = PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL;
                    icfcLayer->xMax[count] = PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL;
                    icfcLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                }
            }
        }
    }
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        if(icfcLayer->flag_wt_positive)
            icfcLayer->xMin[count] = 0;
        else
            icfcLayer->xMin[count] = PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL *
                                     icfcLayer->preFeatureMapChannels *
                                     icfcLayer->preFeatureMapHeightMax *
                                     icfcLayer->preFeatureMapWidthMax;
        icfcLayer->xMax[count] = PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL *
                                 icfcLayer->preFeatureMapChannels *
                                 icfcLayer->preFeatureMapHeightMax *
                                 icfcLayer->preFeatureMapWidthMax;
        icfcLayer->xType[count] = VAR_TYPE_CONTINUOUS;
        count++;
    }

    return icfcLayer;
}

void assignInterCFCLayer(InterCPCLayer* icfcLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;

    if(icfcLayer->flagActFunc == FLAG_STATUS_ON) {
        if(icfcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (icfcLayer->xMax[count] - icfcLayer->xMin[count]) + icfcLayer->xMin[count]);
                    icfcLayer->actFuncType[i] = (int)randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    icfcLayer->actFuncType[i] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = icfcLayer->actFuncType[i];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = icfcLayer->actFuncType[i];
                    break;
                default:
                    printf("%s(%d): mode error for assignInterCFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        } else {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                icfcLayer->actFuncType[i] = icfcLayer->actFuncTypeDefault;
            }
        }
    }
    if(icfcLayer->flag_NN4Para) {
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            assignFCLayer(icfcLayer->NN4Para->LayersPnt[i], &x[count], mode);
            count += icfcLayer->NN4Para->LayersPnt[i]->numParaLocal;
        }
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            MY_FLT_TYPE valIn[4];
                            valIn[0] = (MY_FLT_TYPE)(i + 1.0);// / icfcLayer->numOutput);
                            valIn[1] = (MY_FLT_TYPE)(j + 1.0);// / icfcLayer->preFeatureMapChannels);
                            valIn[2] = (MY_FLT_TYPE)(k + 1.0);// / icfcLayer->preFeatureMapHeightMax);
                            valIn[3] = (MY_FLT_TYPE)(l + 1.0);// / icfcLayer->preFeatureMapWidthMax);
                            int tagIn[4] = { 1, 1, 1, 1 };
                            MY_FLT_TYPE dataflow[4] = { 0.25, 0.25, 0.25, 0.25 };
                            ff_fcLayer(icfcLayer->NN4Para->LayersPnt[0], valIn, tagIn, 4, dataflow);
                            for(int n = 1; n < icfcLayer->NN4Para->layerNum - 1; n++) {
                                ff_fcLayer(icfcLayer->NN4Para->LayersPnt[n],
                                           icfcLayer->NN4Para->LayersPnt[n - 1]->outputData,
                                           icfcLayer->NN4Para->LayersPnt[n - 1]->connectCountAll,
                                           icfcLayer->NN4Para->LayersPnt[n - 1]->numOutputMax,
                                           icfcLayer->NN4Para->LayersPnt[n - 1]->dataflowStatus);
                            }
                            MY_FLT_TYPE* tmpOut = icfcLayer->NN4Para->LayersPnt[icfcLayer->NN4Para->layerNum - 2]->outputData;
                            if(icfcLayer->flag_wt_positive && tmpOut[0] < 0) tmpOut[0] = -tmpOut[0];
                            icfcLayer->connectWeightAll[i][j][k][l] = tmpOut[0];
                            if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
                                if(tmpOut[1] > 0) icfcLayer->connectStatusAll[i][j][k][l] = 1;
                                else              icfcLayer->connectStatusAll[i][j][k][l] = 0;
                            } else {
                                icfcLayer->connectStatusAll[i][j][k][l] = 1;
                            }
                        }
                    }
                }
            }
        }
    } else if(icfcLayer->flag_connectCoding_CP) {
        assignCodingCP(icfcLayer->cdCP_w, &x[count], mode);
        count += icfcLayer->cdCP_w->numParaLocal;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            assignCodingCP(icfcLayer->cdCP_c, &x[count], mode);
            count += icfcLayer->cdCP_c->numParaLocal;
        }
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            int tmp_in[4] = { i, j, k, l };
                            MY_FLT_TYPE tmp_val = 0.0;
                            decodingCP(icfcLayer->cdCP_w, tmp_in, &tmp_val);
                            if(icfcLayer->flag_wt_positive && tmp_val < 0) tmp_val = -tmp_val;
                            icfcLayer->connectWeightAll[i][j][k][l] = tmp_val;
                            if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
                                decodingCP(icfcLayer->cdCP_c, tmp_in, &tmp_val);
                                if(tmp_val > 0)
                                    icfcLayer->connectStatusAll[i][j][k][l] = 1;
                                else
                                    icfcLayer->connectStatusAll[i][j][k][l] = 0;
                            } else {
                                icfcLayer->connectStatusAll[i][j][k][l] = 1;
                            }
                        }
                    }
                }
            }
        }
    } else if(icfcLayer->flag_connectCoding_GEP) {
        assignCodingGEP(icfcLayer->cdGEP_w, &x[count], mode);
        count += icfcLayer->cdGEP_w->numParaLocal;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            assignCodingGEP(icfcLayer->cdGEP_c, &x[count], mode);
            count += icfcLayer->cdGEP_c->numParaLocal;
        }
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            MY_FLT_TYPE tmp_in[4];
                            tmp_in[0] = i + 1;
                            tmp_in[1] = j + 1;
                            tmp_in[2] = k + 1;
                            tmp_in[3] = l + 1;
                            MY_FLT_TYPE tmp_val = 0.0;
                            decodingGEP(icfcLayer->cdGEP_w, tmp_in, &tmp_val);
                            if(icfcLayer->flag_wt_positive && tmp_val < 0) tmp_val = -tmp_val;
                            icfcLayer->connectWeightAll[i][j][k][l] = tmp_val;
                            if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
                                decodingGEP(icfcLayer->cdGEP_c, tmp_in, &tmp_val);
                                if(tmp_val > 0)
                                    icfcLayer->connectStatusAll[i][j][k][l] = 1;
                                else
                                    icfcLayer->connectStatusAll[i][j][k][l] = 0;
                            } else {
                                icfcLayer->connectStatusAll[i][j][k][l] = 1;
                            }
                        }
                    }
                }
            }
        }
    } else {
        MY_FLT_TYPE tmp_connect_cnt = 0;
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            switch(mode) {
                            case INIT_MODE_FRNN:
                            case INIT_BP_MODE_FRNN:
                                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                        (icfcLayer->xMax[count] - icfcLayer->xMin[count]) + icfcLayer->xMin[count]);
                                icfcLayer->connectStatusAll[i][j][k][l] = (int)randnum;
                                break;
                            case ASSIGN_MODE_FRNN:
                                icfcLayer->connectStatusAll[i][j][k][l] = (int)x[count];
                                break;
                            case OUTPUT_ALL_MODE_FRNN:
                                x[count] = icfcLayer->connectStatusAll[i][j][k][l];
                                break;
                            case OUTPUT_CONTINUOUS_MODE_FRNN:
                                break;
                            case OUTPUT_DISCRETE_MODE_FRNN:
                                x[count] = icfcLayer->connectStatusAll[i][j][k][l];
                                break;
                            default:
                                printf("%s(%d): mode error for assignInterCFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                                exit(-1000);
                                break;
                            }
                            count++;
                            if(icfcLayer->connectStatusAll[i][j][k][l])
                                tmp_connect_cnt++;
                        }
                    }
                }
            }
        } else {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                    for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                            icfcLayer->connectStatusAll[i][j][k][l] = 1;
                        }
                    }
                }
            }
        }
        tmp_connect_cnt /= icfcLayer->numOutput * icfcLayer->preFeatureMapChannels *
                           icfcLayer->preFeatureMapHeightMax * icfcLayer->preFeatureMapWidthMax;
        tmp_connect_cnt *= icfcLayer->numOutput + (icfcLayer->preFeatureMapChannels *
                           icfcLayer->preFeatureMapHeightMax * icfcLayer->preFeatureMapWidthMax);
        for(int i = 0; i < icfcLayer->numOutput; i++) {
            for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
                for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                    for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                        switch(mode) {
                        case INIT_MODE_FRNN:
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                    (icfcLayer->xMax[count] - icfcLayer->xMin[count]) + icfcLayer->xMin[count]);
                            icfcLayer->connectWeightAll[i][j][k][l] = randnum;
                            break;
                        case INIT_BP_MODE_FRNN:
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) - 0.5) * 2;
                            icfcLayer->connectWeightAll[i][j][k][l] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(tmp_connect_cnt));
                            break;
                        case ASSIGN_MODE_FRNN:
                            icfcLayer->connectWeightAll[i][j][k][l] = (MY_FLT_TYPE)x[count];
                            break;
                        case OUTPUT_ALL_MODE_FRNN:
                            x[count] = icfcLayer->connectWeightAll[i][j][k][l];
                            break;
                        case OUTPUT_CONTINUOUS_MODE_FRNN:
                            x[count] = icfcLayer->connectWeightAll[i][j][k][l];
                            break;
                        case OUTPUT_DISCRETE_MODE_FRNN:
                            break;
                        default:
                            printf("%s(%d): mode error for assignInterCFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                            exit(-1000);
                            break;
                        }
                        count++;
                    }
                }
            }
        }
    }
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        switch(mode) {
        case INIT_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (icfcLayer->xMax[count] - icfcLayer->xMin[count]) + icfcLayer->xMin[count]);
            icfcLayer->biasData[i] = randnum;
            break;
        case INIT_BP_MODE_FRNN:
            icfcLayer->biasData[i] = 0;
            break;
        case ASSIGN_MODE_FRNN:
            icfcLayer->biasData[i] = (MY_FLT_TYPE)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = icfcLayer->biasData[i];
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            x[count] = icfcLayer->biasData[i];
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            break;
        default:
            printf("%s(%d): mode error for assignInterCFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "InterCFC layer:\n");
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
            fprintf(debug_fpt, "id,(%d - %d)\n", i + 1, j + 1);
            for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                    fprintf(debug_fpt, ",%d,%f",
                            icfcLayer->connectStatusAll[i][j][k][l],
                            icfcLayer->connectWeightAll[i][j][k][l]);
                }
                fprintf(debug_fpt, "\n");
            }
            fprintf(debug_fpt, "\n");
        }
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_icfcLayer(InterCPCLayer* icfcLayer)
{
    if(icfcLayer->flag_NN4Para) {
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            print_para_fcLayer(icfcLayer->NN4Para->LayersPnt[i]);
        }
    } else if(icfcLayer->flag_connectCoding_CP) {
        print_para_codingCP(icfcLayer->cdCP_w);
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            print_para_codingCP(icfcLayer->cdCP_c);
        }
    } else if(icfcLayer->flag_connectCoding_GEP) {
        print_para_codingGEP(icfcLayer->cdGEP_w);
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            print_para_codingGEP(icfcLayer->cdGEP_c);
        }
    }
    printf("InterCFC layer:\n");
    for(int i = 0; i < icfcLayer->numOutput; i++) {
        for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
            printf("id,(%d - %d)\n", i + 1, j + 1);
            for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                for(int l = 0; l < icfcLayer->preFeatureMapWidthMax; l++) {
                    printf(",%d,%f",
                           icfcLayer->connectStatusAll[i][j][k][l],
                           icfcLayer->connectWeightAll[i][j][k][l]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    printf("print para for InterCFC layer done\n========================================\n");
    printf("\n");
}

void freeInterCFCLayer(InterCPCLayer* icfcLayer)
{
    free(icfcLayer->preInputHeight);
    free(icfcLayer->preInputWidth);

    if(icfcLayer->flagActFunc == FLAG_STATUS_ON)
        free(icfcLayer->actFuncType);

    if(icfcLayer->flag_NN4Para) {
        for(int i = 0; i < icfcLayer->NN4Para->layerNum - 1; i++) {
            freeFCLayer(icfcLayer->NN4Para->LayersPnt[i]);
        }
        free(icfcLayer->NN4Para->numNodesAll);
        free(icfcLayer->NN4Para->LayersPnt);
        free(icfcLayer->NN4Para);
    } else if(icfcLayer->flag_connectCoding_CP) {
        freeCodingCP(icfcLayer->cdCP_w);
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            freeCodingCP(icfcLayer->cdCP_c);
        }
    } else if(icfcLayer->flag_connectCoding_GEP) {
        freeCodingGEP(icfcLayer->cdGEP_w);
        if(icfcLayer->flag_connectAdap == FLAG_STATUS_ON) {
            freeCodingGEP(icfcLayer->cdGEP_c);
        }
    }

    for(int i = 0; i < icfcLayer->numOutput; i++) {
        for(int j = 0; j < icfcLayer->preFeatureMapChannels; j++) {
            for(int k = 0; k < icfcLayer->preFeatureMapHeightMax; k++) {
                free(icfcLayer->connectStatusAll[i][j][k]);
                free(icfcLayer->connectWeightAll[i][j][k]);
                free(icfcLayer->connectWtDeltaAll[i][j][k]);
            }
            free(icfcLayer->connectStatusAll[i][j]);
            free(icfcLayer->connectWeightAll[i][j]);
            free(icfcLayer->connectWtDeltaAll[i][j]);
        }
        free(icfcLayer->connectStatusAll[i]);
        free(icfcLayer->connectWeightAll[i]);
        free(icfcLayer->connectWtDeltaAll[i]);
    }
    free(icfcLayer->connectStatusAll);
    free(icfcLayer->connectWeightAll);
    free(icfcLayer->connectWtDeltaAll);
    free(icfcLayer->connectCountAll);

    free(icfcLayer->biasData);
    free(icfcLayer->biasDelta);

    free(icfcLayer->outputData);
    free(icfcLayer->outputDelta);
    free(icfcLayer->outputDerivative);

    free(icfcLayer->dataflowStatus);

    free(icfcLayer->xMin);
    free(icfcLayer->xMax);
    free(icfcLayer->xType);

    free(icfcLayer);
}

FCLayer* setupFCLayer(int numInput, int numInputMax, int numOutput,
                      int flagActFunc, int flag_actFuncTypeAdap, int default_actFuncType, int flag_connectAdap)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    if(numInputMax <= 0) {
        printf("%s(%d): Invalid value of numInputMax %d, exit...\n",
               __FILE__, __LINE__, numInputMax);
        exit(-1);
    }
    if(numOutput <= 0) {
        printf("%s(%d): Invalid value of numOutput %d, exit...\n",
               __FILE__, __LINE__, numOutput);
        exit(-1);
    }
    switch(flagActFunc) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flagActFunc %d, exit...\n",
               __FILE__, __LINE__, flagActFunc);
        exit(-1);
        break;
    }
    switch(flag_actFuncTypeAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_actFuncTypeAdap %d, exit...\n",
               __FILE__, __LINE__, flag_actFuncTypeAdap);
        exit(-1);
        break;
    }
    switch(default_actFuncType) {
    case ACT_FUNC_RELU:
    case ACT_FUNC_LEAKYRELU:
    case ACT_FUNC_SIGMA:
    case ACT_FUNC_TANH:
    case ACT_FUNC_ELU:
        break;
    default:
        printf("%s(%d): Unknown default_actFuncType %d, exit...\n",
               __FILE__, __LINE__, default_actFuncType);
        exit(-1);
        break;
    }
    switch(flag_connectAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_connectAdap %d, exit...\n",
               __FILE__, __LINE__, flag_connectAdap);
        exit(-1);
        break;
    }

    FCLayer* fcLayer = (FCLayer*)malloc(1 * sizeof(FCLayer));

    fcLayer->numInput = numInput;
    fcLayer->numOutput = numOutput;
#if LAYER_SKIP_TAG_CFRNN_MODEL == 1
    fcLayer->numInputMax = numInputMax;
    fcLayer->numOutputMax = fcLayer->numInputMax;
    if(fcLayer->numOutputMax < fcLayer->numOutput) fcLayer->numOutputMax = fcLayer->numOutput;
#else
    fcLayer->numInputMax = numInput;
    fcLayer->numOutputMax = numOutput;
#endif

    fcLayer->flagActFunc = flagActFunc;
    if(fcLayer->flagActFunc)
        fcLayer->actFuncType = (int*)malloc(fcLayer->numOutput * sizeof(int));
    fcLayer->flag_actFuncTypeAdap = flag_actFuncTypeAdap;
    fcLayer->actFuncTypeDefault = default_actFuncType;

    fcLayer->flag_connectAdap = flag_connectAdap;

    fcLayer->connectStatus = (int**)malloc(fcLayer->numOutput * sizeof(int*));
    fcLayer->connectWeight = (MY_FLT_TYPE**)malloc(fcLayer->numOutput * sizeof(MY_FLT_TYPE*));
    fcLayer->connectWtDelta = (MY_FLT_TYPE**)malloc(fcLayer->numOutput * sizeof(MY_FLT_TYPE*));
    fcLayer->connectCountAll = (int*)malloc(fcLayer->numOutputMax * sizeof(int));
    for(int i = 0; i < fcLayer->numOutput; i++) {
        fcLayer->connectStatus[i] = (int*)malloc(fcLayer->numInputMax * sizeof(int));
        fcLayer->connectWeight[i] = (MY_FLT_TYPE*)malloc(fcLayer->numInputMax * sizeof(MY_FLT_TYPE));
        fcLayer->connectWtDelta[i] = (MY_FLT_TYPE*)malloc(fcLayer->numInputMax * sizeof(MY_FLT_TYPE));
    }

    fcLayer->biasData = (MY_FLT_TYPE*)malloc(fcLayer->numOutput * sizeof(MY_FLT_TYPE));
    fcLayer->biasDelta = (MY_FLT_TYPE*)malloc(fcLayer->numOutput * sizeof(MY_FLT_TYPE));

    fcLayer->outputData = (MY_FLT_TYPE*)malloc(fcLayer->numOutputMax * sizeof(MY_FLT_TYPE));
    fcLayer->outputDelta = (MY_FLT_TYPE*)malloc(fcLayer->numOutputMax * sizeof(MY_FLT_TYPE));
    fcLayer->outputDerivative = (MY_FLT_TYPE*)malloc(fcLayer->numOutputMax * sizeof(MY_FLT_TYPE));

    fcLayer->dataflowStatus = (MY_FLT_TYPE*)malloc(fcLayer->numOutputMax * sizeof(MY_FLT_TYPE));

    //
    int count = 0;
    int count_disc = 0;
    if(fcLayer->flagActFunc == FLAG_STATUS_ON &&
       fcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
        count += fcLayer->numOutput;
        count_disc += fcLayer->numOutput;
    }
    if(fcLayer->flag_connectAdap == FLAG_STATUS_ON) {
        count += fcLayer->numOutput * fcLayer->numInputMax;
        count_disc += fcLayer->numOutput * fcLayer->numInputMax;
    }
    count += fcLayer->numOutput * fcLayer->numInputMax;
    count += fcLayer->numOutput;
    fcLayer->numParaLocal = count;
    fcLayer->numParaLocal_disc = count_disc;

    //
    fcLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    fcLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    fcLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(fcLayer->flagActFunc == FLAG_STATUS_ON &&
       fcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            fcLayer->xMin[count] = 0;
            fcLayer->xMax[count] = NUM_ACT_FUNC_TYPE - 1e-6;
            fcLayer->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        }
    }
    if(fcLayer->flag_connectAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            for(int j = 0; j < fcLayer->numInputMax; j++) {
                fcLayer->xMin[count] = 0;
                fcLayer->xMax[count] = 2 - 1e-6;
                fcLayer->xType[count] = VAR_TYPE_BINARY;
                count++;
            }
        }
    }
    for(int i = 0; i < fcLayer->numOutput; i++) {
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            fcLayer->xMin[count] = PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL;
            fcLayer->xMax[count] = PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL;
            fcLayer->xType[count] = VAR_TYPE_CONTINUOUS;
            count++;
        }
    }
    for(int i = 0; i < fcLayer->numOutput; i++) {
        fcLayer->xMin[count] = PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL * fcLayer->numInputMax;
        fcLayer->xMax[count] = PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL * fcLayer->numInputMax;
        fcLayer->xType[count] = VAR_TYPE_CONTINUOUS;
        count++;
    }

    return fcLayer;
}

void assignFCLayer(FCLayer* fcLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;

    if(fcLayer->flagActFunc == FLAG_STATUS_ON) {
        if(fcLayer->flag_actFuncTypeAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < fcLayer->numOutput; i++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (fcLayer->xMax[count] - fcLayer->xMin[count]) + fcLayer->xMin[count]);
                    fcLayer->actFuncType[i] = (int)randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    fcLayer->actFuncType[i] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = fcLayer->actFuncType[i];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = fcLayer->actFuncType[i];
                    break;
                default:
                    printf("%s(%d): mode error for assignInterCFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        } else {
            for(int i = 0; i < fcLayer->numOutput; i++) {
                fcLayer->actFuncType[i] = fcLayer->actFuncTypeDefault;
            }
        }
    }

    MY_FLT_TYPE tmp_connect_cnt = 0;
    if(fcLayer->flag_connectAdap == FLAG_STATUS_ON) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            for(int j = 0; j < fcLayer->numInputMax; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (fcLayer->xMax[count] - fcLayer->xMin[count]) + fcLayer->xMin[count]);
                    fcLayer->connectStatus[i][j] = (int)randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    fcLayer->connectStatus[i][j] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = fcLayer->connectStatus[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = fcLayer->connectStatus[i][j];
                    break;
                default:
                    printf("%s(%d): mode error for assignFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
                if(fcLayer->connectStatus[i][j])
                    tmp_connect_cnt++;
            }
        }
    } else {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            for(int j = 0; j < fcLayer->numInputMax; j++) {
                fcLayer->connectStatus[i][j] = 1;
            }
        }
        tmp_connect_cnt = fcLayer->numOutput * fcLayer->numInputMax;
    }

    tmp_connect_cnt = tmp_connect_cnt / (fcLayer->numOutput * fcLayer->numInputMax);
    tmp_connect_cnt *= fcLayer->numOutput + fcLayer->numInputMax;
    for(int i = 0; i < fcLayer->numOutput; i++) {
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            switch(mode) {
            case INIT_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                        (fcLayer->xMax[count] - fcLayer->xMin[count]) + fcLayer->xMin[count]);
                fcLayer->connectWeight[i][j] = randnum;
                break;
            case INIT_BP_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) - 0.5) * 2;
                fcLayer->connectWeight[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(tmp_connect_cnt));
                break;
            case ASSIGN_MODE_FRNN:
                fcLayer->connectWeight[i][j] = (MY_FLT_TYPE)x[count];
                break;
            case OUTPUT_ALL_MODE_FRNN:
                x[count] = fcLayer->connectWeight[i][j];
                break;
            case OUTPUT_CONTINUOUS_MODE_FRNN:
                x[count] = fcLayer->connectWeight[i][j];
                break;
            case OUTPUT_DISCRETE_MODE_FRNN:
                break;
            default:
                printf("%s(%d): mode error for assignFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                exit(-1000);
                break;
            }
            count++;
        }
    }

    for(int i = 0; i < fcLayer->numOutput; i++) {
        switch(mode) {
        case INIT_MODE_FRNN:
            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                    (fcLayer->xMax[count] - fcLayer->xMin[count]) + fcLayer->xMin[count]);
            fcLayer->biasData[i] = randnum;
            break;
        case INIT_BP_MODE_FRNN:
            fcLayer->biasData[i] = 0;
            break;
        case ASSIGN_MODE_FRNN:
            fcLayer->biasData[i] = (MY_FLT_TYPE)x[count];
            break;
        case OUTPUT_ALL_MODE_FRNN:
            x[count] = fcLayer->biasData[i];
            break;
        case OUTPUT_CONTINUOUS_MODE_FRNN:
            x[count] = fcLayer->biasData[i];
            break;
        case OUTPUT_DISCRETE_MODE_FRNN:
            break;
        default:
            printf("%s(%d): mode error for assignFCLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
            exit(-1000);
            break;
        }
        count++;
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "FC layer:\n");
    for(int i = 0; i < fcLayer->numOutput; i++) {
        fprintf(debug_fpt, "%d", i + 1);
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            fprintf(debug_fpt, ",%d,%f", fcLayer->connectStatus[i][j], fcLayer->connectWeight[i][j]);
        }
        fprintf(debug_fpt, "\n");
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_fcLayer(FCLayer* fcLayer)
{
    printf("FC layer:\n");
    for(int i = 0; i < fcLayer->numOutput; i++) {
        printf("%d", i + 1);
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            printf(",%d,%f", fcLayer->connectStatus[i][j], fcLayer->connectWeight[i][j]);
        }
        printf("\n");
    }
    printf("print para for FC done\n========================================\n");
    printf("\n");
}

void freeFCLayer(FCLayer* fcLayer)
{
    if(fcLayer->flagActFunc == FLAG_STATUS_ON)
        free(fcLayer->actFuncType);

    for(int i = 0; i < fcLayer->numOutput; i++) {
        free(fcLayer->connectStatus[i]);
        free(fcLayer->connectWeight[i]);
        free(fcLayer->connectWtDelta[i]);
    }
    free(fcLayer->connectStatus);
    free(fcLayer->connectWeight);
    free(fcLayer->connectWtDelta);
    free(fcLayer->connectCountAll);

    free(fcLayer->biasData);
    free(fcLayer->biasDelta);

    free(fcLayer->outputData);
    free(fcLayer->outputDelta);
    free(fcLayer->outputDerivative);

    free(fcLayer->dataflowStatus);

    free(fcLayer->xMin);
    free(fcLayer->xMax);
    free(fcLayer->xType);

    free(fcLayer);

    return;
}

MemberLayer* setupMemberLayer(int numInput, MY_FLT_TYPE* inputMin, MY_FLT_TYPE* inputMax, int* numMemship,
                              int* flag_adapMemship,
                              int typeFuzzySet,
                              int typeMFCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    for(int i = 0; i < numInput; i++) {
        if(inputMax[i] <= inputMin[i]) {
            printf("%s(%d): Invalid value of inputMax[%d] and inputMin[%d] ~ %lf <= %lf, exit...\n",
                   __FILE__, __LINE__, i, i, inputMax[i], inputMin[i]);
            exit(-1);
        }
    }
    for(int i = 0; i < numInput; i++) {
        if(numMemship[i] <= 0) {
            printf("%s(%d): Invalid value of numMemship[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, numMemship[i]);
            exit(-1);
        }
    }
    for(int i = 0; i < numInput; i++) {
        switch(flag_adapMemship[i]) {
        case FLAG_STATUS_OFF:
        case FLAG_STATUS_ON:
            break;
        default:
            printf("%s(%d): Unknown flag_adapMemship[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, flag_adapMemship[i]);
            exit(-1);
            break;
        }
    }
    switch(typeFuzzySet) {
    case FUZZY_SET_I:
    case FUZZY_INTERVAL_TYPE_II:
        break;
    default:
        printf("%s(%d): Unknown typeFuzzySet %d, exit...\n",
               __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }
    switch(typeMFCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeMFCoding - %d, exiting...\n",
               __FILE__, __LINE__, typeMFCoding);
        exit(-1000);
        break;
    }

    MemberLayer* mLayer = (MemberLayer*)malloc(1 * sizeof(MemberLayer));

    mLayer->typeMFCoding = typeMFCoding;
    switch(mLayer->typeMFCoding) {
    case PARA_CODING_DIRECT:
        mLayer->flag_MFCoding_CP = 0;
        mLayer->flag_MFCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        mLayer->flag_MFCoding_CP = 1;
        mLayer->flag_MFCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        mLayer->flag_MFCoding_CP = 0;
        mLayer->flag_MFCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeMFCoding - %d, exiting...\n",
               __FILE__, __LINE__, mLayer->typeMFCoding);
        exit(-1000);
        break;
    }

    switch(typeFuzzySet) {
    case FUZZY_SET_I:
        mLayer->dim_degree = 1;
        mLayer->numParaMembershipFun = NUM_PARA_MEM_FUNC_I_FRNN_MODEL;
        break;
    case FUZZY_INTERVAL_TYPE_II:
        mLayer->dim_degree = 2;
        mLayer->numParaMembershipFun = NUM_PARA_MEM_FUNC_II_FRNN_MODEL;
        break;
    default:
        printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    mLayer->typeFuzzySet = typeFuzzySet;

    mLayer->numInput = numInput;
    mLayer->valInput = (MY_FLT_TYPE*)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE));
    mLayer->inputMin = (MY_FLT_TYPE*)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE));
    mLayer->inputMax = (MY_FLT_TYPE*)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE));
    memcpy(mLayer->inputMin, inputMin, mLayer->numInput * sizeof(MY_FLT_TYPE));
    memcpy(mLayer->inputMax, inputMax, mLayer->numInput * sizeof(MY_FLT_TYPE));

    mLayer->numMembershipFunCur = (int*)malloc(mLayer->numInput * sizeof(int));
    mLayer->numMembershipFun = (int*)malloc(mLayer->numInput * sizeof(int));
    memcpy(mLayer->numMembershipFun, numMemship, mLayer->numInput * sizeof(int));
    mLayer->flag_adapMembershipFun = (int*)malloc(mLayer->numInput * sizeof(int));
    memcpy(mLayer->flag_adapMembershipFun, flag_adapMemship, mLayer->numInput * sizeof(int));
    mLayer->outputSize = 0;
    for(int i = 0; i < mLayer->numInput; i++) mLayer->outputSize += mLayer->numMembershipFun[i];
    mLayer->typeMembershipFun = (int**)malloc(mLayer->numInput * sizeof(int*));
    for(int i = 0; i < mLayer->numInput; i++) {
        mLayer->typeMembershipFun[i] = (int*)malloc(mLayer->numMembershipFun[i] * sizeof(int));
    }

    mLayer->paraMembershipFun = (MY_FLT_TYPE***)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE**));
    mLayer->degreeMembership = (MY_FLT_TYPE***)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < mLayer->numInput; i++) {
        mLayer->paraMembershipFun[i] = (MY_FLT_TYPE**)malloc(mLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE*));
        mLayer->degreeMembership[i] = (MY_FLT_TYPE**)malloc(mLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            mLayer->paraMembershipFun[i][j] = (MY_FLT_TYPE*)malloc(MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL * sizeof(MY_FLT_TYPE));
            mLayer->degreeMembership[i][j] = (MY_FLT_TYPE*)malloc(mLayer->dim_degree * sizeof(MY_FLT_TYPE));
        }
    }

    mLayer->dataflowStatus = (MY_FLT_TYPE**)malloc(mLayer->numInput * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < mLayer->numInput; i++) {
        mLayer->dataflowStatus[i] = (MY_FLT_TYPE*)malloc(mLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE));
    }

    //
    int count = 0;
    int count_disc = 0;
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) continue;
        count += mLayer->numMembershipFun[i];
        count_disc += mLayer->numMembershipFun[i];
    }
    int max_num_MF = 0;
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) continue;
        if(max_num_MF < mLayer->numMembershipFun[i])
            max_num_MF = mLayer->numMembershipFun[i];
    }
    if(mLayer->flag_MFCoding_CP) {
        const int num_dim_CP = 3;
        int size_dim_CP[num_dim_CP];
        size_dim_CP[0] = mLayer->numInput;
        size_dim_CP[1] = max_num_MF;
        size_dim_CP[2] = mLayer->numParaMembershipFun;
        mLayer->cdCP = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                     PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += mLayer->cdCP->numParaLocal;
        count_disc += mLayer->cdCP->numParaLocal_disc;
    } else if(mLayer->flag_MFCoding_GEP) {
        int GEP_num_input = 3;
        MY_FLT_TYPE inputMin[3] = { 0, 0, 0 };
        MY_FLT_TYPE inputMax[3];
        inputMax[0] = mLayer->numInput;
        inputMax[1] = max_num_MF;
        inputMax[2] = mLayer->numParaMembershipFun;
        mLayer->cdGEP = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                       GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += mLayer->cdGEP->numParaLocal;
        count_disc += mLayer->cdGEP->numParaLocal_disc;
    } else {
        for(int i = 0; i < mLayer->numInput; i++) {
            if(mLayer->flag_adapMembershipFun[i] == 0) continue;
            count += mLayer->numMembershipFun[i] * mLayer->numParaMembershipFun;
        }
    }
    mLayer->numParaLocal = count;
    mLayer->numParaLocal_disc = count_disc;

    //
    mLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    mLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    mLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) continue;
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            mLayer->xMin[count] = 0;
            mLayer->xMax[count] = NUM_MEM_FUNC - 1e-6;
            mLayer->xType[count] = VAR_TYPE_DISCRETE;
            count++;
        }
    }
    if(mLayer->flag_MFCoding_CP) {
        memcpy(&mLayer->xMin[count], mLayer->cdCP->xMin, mLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&mLayer->xMax[count], mLayer->cdCP->xMax, mLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&mLayer->xType[count], mLayer->cdCP->xType, mLayer->cdCP->numParaLocal * sizeof(int));
        count += mLayer->cdCP->numParaLocal;
    } else if(mLayer->flag_MFCoding_GEP) {
        memcpy(&mLayer->xMin[count], mLayer->cdGEP->xMin, mLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&mLayer->xMax[count], mLayer->cdGEP->xMax, mLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&mLayer->xType[count], mLayer->cdGEP->xType, mLayer->cdGEP->numParaLocal * sizeof(int));
        count += mLayer->cdGEP->numParaLocal;
    } else {
        MY_FLT_TYPE cur_min, cur_max;
        MY_FLT_TYPE tmp_min[MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL], tmp_max[MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL];
        cur_min = 0; // mLayer->inputMin[i];
        cur_max = 1; // mLayer->inputMax[i];
        tmp_min[0] = 1e-2;
        tmp_min[1] = cur_min;
        tmp_min[2] = 1e-2;
        tmp_min[3] = cur_min;
        tmp_min[4] = -PARA_MAX_MEM_SIGMOID_FRNN_MODEL;
        tmp_min[5] = cur_min;
        tmp_min[6] = 1.0 + 1e-5;
        tmp_min[7] = 0.5 + 1e-5;
        tmp_max[0] = cur_max - cur_min; // delta
        tmp_max[1] = cur_max;
        tmp_max[2] = cur_max - cur_min; // delta
        tmp_max[3] = cur_max;
        tmp_max[4] = PARA_MAX_MEM_SIGMOID_FRNN_MODEL;
        tmp_max[5] = cur_max;
        tmp_max[6] = 2.0 - 1e-5;
        tmp_max[7] = 1.0 - 1e-5;
        for(int i = 0; i < mLayer->numInput; i++) {
            if(mLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                    mLayer->xMin[count] = tmp_min[k];
                    mLayer->xMax[count] = tmp_max[k];
                    mLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                }
            }
        }
    }

    return mLayer;
}

void assignMemberLayer(MemberLayer* mLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) continue;
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            switch(mode) {
            case INIT_MODE_FRNN:
            case INIT_BP_MODE_FRNN:
                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                        (mLayer->xMax[count] - mLayer->xMin[count]) + mLayer->xMin[count]);
                mLayer->typeMembershipFun[i][j] = (int)randnum;
                //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                break;
            case ASSIGN_MODE_FRNN:
                mLayer->typeMembershipFun[i][j] = (int)x[count];
                break;
            case OUTPUT_ALL_MODE_FRNN:
                x[count] = mLayer->typeMembershipFun[i][j];
                break;
            case OUTPUT_CONTINUOUS_MODE_FRNN:
                break;
            case OUTPUT_DISCRETE_MODE_FRNN:
                x[count] = mLayer->typeMembershipFun[i][j];
                break;
            default:
                printf("%s(%d): mode error for assignMemberLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                exit(-1000);
                break;
            }
            count++;
        }
    }
    if(mLayer->flag_MFCoding_CP) {
        assignCodingCP(mLayer->cdCP, &x[count], mode);
        count += mLayer->cdCP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            MY_FLT_TYPE cur_min, cur_max;
            MY_FLT_TYPE tmp_min[MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL], tmp_max[MAX_NUM_PARA_MEM_FUNC_FRNN_MODEL];
            cur_min = 0; // mLayer->inputMin[i];
            cur_max = 1; // mLayer->inputMax[i];
            tmp_min[0] = 1e-2;
            tmp_min[1] = cur_min;
            tmp_min[2] = 1e-2;
            tmp_min[3] = cur_min;
            tmp_min[4] = -PARA_MAX_MEM_SIGMOID_FRNN_MODEL;
            tmp_min[5] = cur_min;
            tmp_min[6] = 1.0 + 1e-5;
            tmp_min[7] = 0.5 + 1e-5;
            tmp_max[0] = cur_max - cur_min; // delta
            tmp_max[1] = cur_max;
            tmp_max[2] = cur_max - cur_min; // delta
            tmp_max[3] = cur_max;
            tmp_max[4] = PARA_MAX_MEM_SIGMOID_FRNN_MODEL;
            tmp_max[5] = cur_max;
            tmp_max[6] = 2.0 - 1e-5;
            tmp_max[7] = 1.0 - 1e-5;
            for(int i = 0; i < mLayer->numInput; i++) {
                if(mLayer->flag_adapMembershipFun[i] == 0) continue;
                for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
                    for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                        int tmp_in[4] = { i, j, k };
                        MY_FLT_TYPE tmp_val = 0.0;
                        decodingCP(mLayer->cdCP, tmp_in, &tmp_val);
                        tmp_val = (tmp_val - mLayer->cdCP->min_val) / (mLayer->cdCP->max_val - mLayer->cdCP->min_val);
                        tmp_val = tmp_val * (tmp_max[k] - tmp_min[k]) + tmp_min[k];
                        mLayer->paraMembershipFun[i][j][k] = tmp_val;
                    }
                }
            }
        }
    } else if(mLayer->flag_MFCoding_GEP) {
        assignCodingGEP(mLayer->cdGEP, &x[count], mode);
        count += mLayer->cdGEP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < mLayer->numInput; i++) {
                if(mLayer->flag_adapMembershipFun[i] == 0) continue;
                for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
                    for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                        MY_FLT_TYPE tmp_R = i + 1;// (float)((i + 1.0) / (fLayer->numRules));
                        MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (fLayer->numInput));
                        MY_FLT_TYPE tmp_M = k + 1;// (float)((k + 1.0) / (fLayer->numMembershipFun[j]));
                        MY_FLT_TYPE tmp_in[3] = { tmp_R, tmp_I, tmp_M };
                        MY_FLT_TYPE tmp_val = 0.0;
                        decodingGEP(mLayer->cdGEP, tmp_in, &tmp_val);
                        mLayer->paraMembershipFun[i][j][k] = tmp_val;
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < mLayer->numInput; i++) {
            if(mLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                    switch(mode) {
                    case INIT_MODE_FRNN:
                    case INIT_BP_MODE_FRNN:
                        randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                (mLayer->xMax[count] - mLayer->xMin[count]) + mLayer->xMin[count]);
                        mLayer->paraMembershipFun[i][j][k] = randnum;
                        //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                        break;
                    case ASSIGN_MODE_FRNN:
                        mLayer->paraMembershipFun[i][j][k] = (MY_FLT_TYPE)x[count];
                        break;
                    case OUTPUT_ALL_MODE_FRNN:
                        x[count] = mLayer->paraMembershipFun[i][j][k];
                        break;
                    case OUTPUT_CONTINUOUS_MODE_FRNN:
                        x[count] = mLayer->paraMembershipFun[i][j][k];
                        break;
                    case OUTPUT_DISCRETE_MODE_FRNN:
                        break;
                    default:
                        printf("%s(%d): mode error for assignMemberLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                        exit(-1000);
                        break;
                    }
                    count++;
                }
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Membership layer:\n");
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) {
            continue;
        }
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            fprintf(debug_fpt, "%d", i + 1);
            fprintf(debug_fpt, ",%d,%d", j + 1, mLayer->typeMembershipFun[i][j]);
            for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                fprintf(debug_fpt, ",%f", mLayer->paraMembershipFun[i][j][k]);
            }
            fprintf(debug_fpt, "\n");
        }
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_memberLayer(MemberLayer* mLayer)
{
    if(mLayer->flag_MFCoding_CP) {
        print_para_codingCP(mLayer->cdCP);
    } else if(mLayer->flag_MFCoding_GEP) {
        print_para_codingGEP(mLayer->cdGEP);
    }

    printf("Membership layer:\n");
    for(int i = 0; i < mLayer->numInput; i++) {
        if(mLayer->flag_adapMembershipFun[i] == 0) {
            continue;
        }
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            printf("%d", i + 1);
            printf(",%d,%d", j + 1, mLayer->typeMembershipFun[i][j]);
            for(int k = 0; k < mLayer->numParaMembershipFun; k++) {
                printf(",%f", mLayer->paraMembershipFun[i][j][k]);
            }
            printf("\n");
        }
    }
    printf("print para for Membership layer done\n========================================\n");
    printf("\n");
}

void freeMemberLayer(MemberLayer* mLayer)
{
    free(mLayer->xMin);
    free(mLayer->xMax);
    free(mLayer->xType);

    free(mLayer->valInput);
    free(mLayer->inputMin);
    free(mLayer->inputMax);
    free(mLayer->flag_adapMembershipFun);

    if(mLayer->flag_MFCoding_CP) {
        freeCodingCP(mLayer->cdCP);
    } else if(mLayer->flag_MFCoding_GEP) {
        freeCodingGEP(mLayer->cdGEP);
    }

    for(int i = 0; i < mLayer->numInput; i++) {
        free(mLayer->typeMembershipFun[i]);
    }
    free(mLayer->typeMembershipFun);

    for(int i = 0; i < mLayer->numInput; i++) {
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            free(mLayer->paraMembershipFun[i][j]);
            free(mLayer->degreeMembership[i][j]);
        }
        free(mLayer->paraMembershipFun[i]);
        free(mLayer->degreeMembership[i]);
    }
    free(mLayer->paraMembershipFun);
    free(mLayer->degreeMembership);

    for(int i = 0; i < mLayer->numInput; i++) {
        free(mLayer->dataflowStatus[i]);
    }
    free(mLayer->dataflowStatus);

    free(mLayer->numMembershipFunCur);
    free(mLayer->numMembershipFun);

    free(mLayer);

    return;
}

Member2DLayer* setupMember2DLayer(int numInput,
                                  int typeMFFeatMapCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight,
                                  int preFeatureMapHeightMax, int preFeatureMapWidthMax,
                                  int flag_typeMembershipFunAdap, int default_typeMembershipFun,
                                  int* numMemship, int* flag_adapMemship, int typeFuzzySet)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    switch(typeMFFeatMapCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeMFFeatMapCoding - %d, exiting...\n",
               __FILE__, __LINE__, typeMFFeatMapCoding);
        exit(-1000);
        break;
    }
    if(preFeatureMapHeightMax) {
        printf("%s(%d): Invalid value of preFeatureMapHeightMax %d, exit...\n",
               __FILE__, __LINE__, preFeatureMapHeightMax);
        exit(-1);
    }
    if(preFeatureMapWidthMax) {
        printf("%s(%d): Invalid value of preFeatureMapWidthMax %d, exit...\n",
               __FILE__, __LINE__, preFeatureMapWidthMax);
        exit(-1);
    }
    switch(flag_typeMembershipFunAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flag_typeMembershipFun %d, exit...\n",
               __FILE__, __LINE__, flag_typeMembershipFunAdap);
        exit(-1);
        break;
    }
    switch(default_typeMembershipFun) {
    case MAT_SIMILARITY_T_COS:
    //case MAT_SIMILARITY_T_ACOS:
    case MAT_SIMILARITY_T_Norm2:
        break;
    default:
        printf("%s(%d): Unknown default_typeMembershipFun %d, exit...\n",
               __FILE__, __LINE__, default_typeMembershipFun);
        exit(-1);
        break;
    }
    for(int i = 0; i < numInput; i++) {
        if(numMemship[i] <= 0) {
            printf("%s(%d): Invalid value of numMemship[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, numMemship[i]);
            exit(-1);
        }
    }
    for(int i = 0; i < numInput; i++) {
        switch(flag_adapMemship[i]) {
        case FLAG_STATUS_OFF:
        case FLAG_STATUS_ON:
            break;
        default:
            printf("%s(%d): Unknown flag_adapMemship[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, flag_adapMemship[i]);
            exit(-1);
            break;
        }
    }
    switch(typeFuzzySet) {
    case FUZZY_SET_I:
    case FUZZY_INTERVAL_TYPE_II:
        break;
    default:
        printf("%s(%d): Unknown typeFuzzySet %d, exit...\n",
               __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    Member2DLayer* m2DLayer = (Member2DLayer*)malloc(1 * sizeof(Member2DLayer));

    m2DLayer->typeMFFeatMapCoding = typeMFFeatMapCoding;
    switch(m2DLayer->typeMFFeatMapCoding) {
    case PARA_CODING_DIRECT:
        m2DLayer->flag_MFFeatMapCoding_CP = 0;
        m2DLayer->flag_MFFeatMapCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        m2DLayer->flag_MFFeatMapCoding_CP = 1;
        m2DLayer->flag_MFFeatMapCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        m2DLayer->flag_MFFeatMapCoding_CP = 0;
        m2DLayer->flag_MFFeatMapCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeMFFeatMapCoding - %d, exiting...\n",
               __FILE__, __LINE__, m2DLayer->typeMFFeatMapCoding);
        exit(-1000);
        break;
    }

    switch(typeFuzzySet) {
    case FUZZY_SET_I:
        m2DLayer->dim_degree = 1;
        break;
    case FUZZY_INTERVAL_TYPE_II:
        m2DLayer->dim_degree = 2;
        //m2DLayer->numPara_MFFeatMapCoding_GEP += 2;
        break;
    default:
        printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    m2DLayer->typeFuzzySet = typeFuzzySet;

    m2DLayer->numInput = numInput;
    m2DLayer->preFeatureMapHeightMax = preFeatureMapHeightMax;
    m2DLayer->preFeatureMapWidthMax = preFeatureMapWidthMax;

    m2DLayer->flag_typeMembershipFunAdap = flag_typeMembershipFunAdap;
    m2DLayer->typeMembershipFunDefault = default_typeMembershipFun;

    m2DLayer->numMembershipFunCur = (int*)malloc(m2DLayer->numInput * sizeof(int));
    m2DLayer->numMembershipFun = (int*)malloc(m2DLayer->numInput * sizeof(int));
    memcpy(m2DLayer->numMembershipFun, numMemship, m2DLayer->numInput * sizeof(int));
    m2DLayer->flag_adapMembershipFun = (int*)malloc(m2DLayer->numInput * sizeof(int));
    memcpy(m2DLayer->flag_adapMembershipFun, flag_adapMemship, m2DLayer->numInput * sizeof(int));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        if(m2DLayer->flag_adapMembershipFun[i] == 0) {
            printf("%s(%d): For 2-D MFs, the input should be continuous, exiting...\n",
                   __FILE__, __LINE__);
            exit(-1);
        }
    }
    m2DLayer->outputSize = 0;
    for(int i = 0; i < m2DLayer->numInput; i++) m2DLayer->outputSize += m2DLayer->numMembershipFun[i];

    m2DLayer->typeMembershipFun = (int**)malloc(m2DLayer->numInput * sizeof(int*));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        m2DLayer->typeMembershipFun[i] = (int*)malloc(m2DLayer->numMembershipFun[i] * sizeof(int));
    }

    m2DLayer->degreeMembership = (MY_FLT_TYPE***)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        m2DLayer->degreeMembership[i] = (MY_FLT_TYPE**)malloc(m2DLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            m2DLayer->degreeMembership[i][j] = (MY_FLT_TYPE*)malloc(m2DLayer->dim_degree * sizeof(MY_FLT_TYPE));
        }
    }

    m2DLayer->mat_MFFeatMap = (MY_FLT_TYPE****)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE***));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        m2DLayer->mat_MFFeatMap[i] = (MY_FLT_TYPE***)malloc(m2DLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE**));
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            m2DLayer->mat_MFFeatMap[i][j] = (MY_FLT_TYPE**)malloc(m2DLayer->preFeatureMapHeightMax * sizeof(MY_FLT_TYPE*));
            for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                m2DLayer->mat_MFFeatMap[i][j][k] = (MY_FLT_TYPE*)malloc(m2DLayer->preFeatureMapWidthMax * sizeof(MY_FLT_TYPE));
            }
        }
    }

    m2DLayer->norm_mat_MFFeatMap = (MY_FLT_TYPE**)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        m2DLayer->norm_mat_MFFeatMap[i] = (MY_FLT_TYPE*)malloc(m2DLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE));
    }

    m2DLayer->mean_featureMapDataIn = (MY_FLT_TYPE*)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE));

    if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        m2DLayer->para_MF_II_ratios = (MY_FLT_TYPE***)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE**));
        for(int i = 0; i < m2DLayer->numInput; i++) {
            m2DLayer->para_MF_II_ratios[i] = (MY_FLT_TYPE**)malloc(m2DLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE*));
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                m2DLayer->para_MF_II_ratios[i][j] = (MY_FLT_TYPE*)malloc(2 * sizeof(MY_FLT_TYPE));
            }
        }
    }

    m2DLayer->dataflowStatus = (MY_FLT_TYPE**)malloc(m2DLayer->numInput * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < m2DLayer->numInput; i++) {
        m2DLayer->dataflowStatus[i] = (MY_FLT_TYPE*)malloc(m2DLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE));
    }

    //
    int count = 0;
    int count_disc = 0;
    if(m2DLayer->flag_typeMembershipFunAdap) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            count += m2DLayer->numMembershipFun[i];
            count_disc += m2DLayer->numMembershipFun[i];
        }
    }
    int max_num_MF = 0;
    for(int i = 0; i < m2DLayer->numInput; i++) {
        if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
        if(max_num_MF < m2DLayer->numMembershipFun[i])
            max_num_MF = m2DLayer->numMembershipFun[i];
    }
    if(m2DLayer->flag_MFFeatMapCoding_CP) {
        int num_dim_CP = 4;
        int size_dim_CP[4];
        size_dim_CP[0] = m2DLayer->numInput;
        size_dim_CP[1] = max_num_MF;
        size_dim_CP[2] = m2DLayer->preFeatureMapHeightMax;
        size_dim_CP[3] = m2DLayer->preFeatureMapWidthMax;
        m2DLayer->cdCP = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                       PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += m2DLayer->cdCP->numParaLocal;
        count_disc += m2DLayer->cdCP->numParaLocal_disc;
    } else if(m2DLayer->flag_MFFeatMapCoding_GEP) {
        int GEP_num_input = 4;
        MY_FLT_TYPE inputMin[4] = { 0, 0, 0, 0 };
        MY_FLT_TYPE inputMax[4];
        inputMax[0] = m2DLayer->numInput;
        inputMax[1] = max_num_MF;
        inputMax[2] = m2DLayer->preFeatureMapHeightMax;
        inputMax[3] = m2DLayer->preFeatureMapWidthMax;
        m2DLayer->cdGEP = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                         GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += m2DLayer->cdGEP->numParaLocal;
        count_disc += m2DLayer->cdGEP->numParaLocal_disc;
    } else {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            count += m2DLayer->numMembershipFun[i] * m2DLayer->preFeatureMapHeightMax * m2DLayer->preFeatureMapWidthMax;
        }
    }
    if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            count += m2DLayer->numMembershipFun[i] * 2;
        }
    }
    m2DLayer->numParaLocal = count;
    m2DLayer->numParaLocal_disc = count_disc;

    //
    m2DLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    m2DLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    m2DLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(m2DLayer->flag_typeMembershipFunAdap) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                m2DLayer->xMin[count] = 0;
                m2DLayer->xMax[count] = MAT_SIMILARITY_T_NUM - 1e-6;
                m2DLayer->xType[count] = VAR_TYPE_DISCRETE;
                count++;
            }
        }
    }
    if(m2DLayer->flag_MFFeatMapCoding_CP) {
        memcpy(&m2DLayer->xMin[count], m2DLayer->cdCP->xMin, m2DLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&m2DLayer->xMax[count], m2DLayer->cdCP->xMax, m2DLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&m2DLayer->xType[count], m2DLayer->cdCP->xType, m2DLayer->cdCP->numParaLocal * sizeof(int));
        count += m2DLayer->cdCP->numParaLocal;
    } else if(m2DLayer->flag_MFFeatMapCoding_GEP) {
        memcpy(&m2DLayer->xMin[count], m2DLayer->cdGEP->xMin, m2DLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&m2DLayer->xMax[count], m2DLayer->cdGEP->xMax, m2DLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&m2DLayer->xType[count], m2DLayer->cdGEP->xType, m2DLayer->cdGEP->numParaLocal * sizeof(int));
        count += m2DLayer->cdGEP->numParaLocal;
    } else {
        // CHECK ~ value ranges good or bad
        MY_FLT_TYPE cur_min = -PARA_MAX_W_MF_2D_FM_FRNN_MODEL * 100;
        MY_FLT_TYPE cur_max = PARA_MAX_W_MF_2D_FM_FRNN_MODEL * 100;
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                    for(int l = 0; l < m2DLayer->preFeatureMapWidthMax; l++) {
                        m2DLayer->xMin[count] = cur_min;
                        m2DLayer->xMax[count] = cur_max;
                        m2DLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                        count++;
                    }
                }
            }
        }
    }
    if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        MY_FLT_TYPE tmp_min[2], tmp_max[2];
        tmp_min[0] = 1.0 + 1e-6;
        tmp_min[1] = 0.5 + 1e-6;
        tmp_max[0] = 2.0 - 1e-6;
        tmp_max[1] = 1.0 - 1e-6;
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < 2; k++) {
                    m2DLayer->xMin[count] = tmp_min[k];
                    m2DLayer->xMax[count] = tmp_max[k];
                    m2DLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                }
            }
        }
    }

    return m2DLayer;
}

void assignMember2DLayer(Member2DLayer* m2DLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    if(m2DLayer->flag_typeMembershipFunAdap) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (m2DLayer->xMax[count] - m2DLayer->xMin[count]) + m2DLayer->xMin[count]);
                    m2DLayer->typeMembershipFun[i][j] = (int)randnum;
                    //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                    break;
                case ASSIGN_MODE_FRNN:
                    m2DLayer->typeMembershipFun[i][j] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = m2DLayer->typeMembershipFun[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = m2DLayer->typeMembershipFun[i][j];
                    break;
                default:
                    printf("%s(%d): mode error for assignMember2DLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        }
    } else {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                m2DLayer->typeMembershipFun[i][j] = m2DLayer->typeMembershipFunDefault;
            }
        }
    }
    //
    if(m2DLayer->flag_MFFeatMapCoding_CP) {
        assignCodingCP(m2DLayer->cdCP, &x[count], mode);
        count += m2DLayer->cdCP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < m2DLayer->numInput; i++) {
                if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
                for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                    for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < m2DLayer->preFeatureMapWidthMax; l++) {
                            int tmp_in[4] = { i, j, k, l };
                            MY_FLT_TYPE tmp_val = 0.0;
                            decodingCP(m2DLayer->cdCP, tmp_in, &tmp_val);
                            m2DLayer->mat_MFFeatMap[i][j][k][l] = tmp_val;
                        }
                    }
                }
            }
        }
    } else if(m2DLayer->flag_MFFeatMapCoding_GEP) {
        assignCodingGEP(m2DLayer->cdGEP, &x[count], mode);
        count += m2DLayer->cdGEP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < m2DLayer->numInput; i++) {
                if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
                for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                    for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                        for(int l = 0; l < m2DLayer->preFeatureMapWidthMax; l++) {
                            MY_FLT_TYPE tmp_I = i + 1;// (float)((i + 1.0) / (m2DLayer->numInput));
                            MY_FLT_TYPE tmp_O = j + 1;// (float)((j + 1.0) / (m2DLayer->numMembershipFun[i]));
                            MY_FLT_TYPE tmp_X = k + 1;// (float)((k + 1.0) / (m2DLayer->preFeatureMapHeightMax));
                            MY_FLT_TYPE tmp_Y = l + 1;// (float)((l + 1.0) / (m2DLayer->preFeatureMapWidthMax));
                            MY_FLT_TYPE tmp_in[4] = { tmp_I, tmp_O, tmp_X, tmp_Y };
                            MY_FLT_TYPE tmp_val = 0;
                            decodingGEP(m2DLayer->cdGEP, tmp_in, &tmp_val);
                            m2DLayer->mat_MFFeatMap[i][j][k][l] = tmp_val;
                        }
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                    for(int l = 0; l < m2DLayer->preFeatureMapWidthMax; l++) {
                        switch(mode) {
                        case INIT_MODE_FRNN:
                        case INIT_BP_MODE_FRNN:
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                    (m2DLayer->xMax[count] - m2DLayer->xMin[count]) + m2DLayer->xMin[count]);
                            m2DLayer->mat_MFFeatMap[i][j][k][l] = randnum;
                            //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                            break;
                        case ASSIGN_MODE_FRNN:
                            m2DLayer->mat_MFFeatMap[i][j][k][l] = (MY_FLT_TYPE)x[count];
                            break;
                        case OUTPUT_ALL_MODE_FRNN:
                            x[count] = m2DLayer->mat_MFFeatMap[i][j][k][l];
                            break;
                        case OUTPUT_CONTINUOUS_MODE_FRNN:
                            x[count] = m2DLayer->mat_MFFeatMap[i][j][k][l];
                            break;
                        case OUTPUT_DISCRETE_MODE_FRNN:
                            break;
                        default:
                            printf("%s(%d): mode error for assignMember2DLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                            exit(-1000);
                            break;
                        }
                        count++;
                    }
                }
            }
        }
    }
    //
    //for (int i = 0; i < m2DLayer->numInput; i++) {
    //	if (m2DLayer->flag_adapMembershipFun[i] == 0) {
//           memset(m2DLayer->norm_matMembershipFeatureMap[i], 0, m2DLayer->numMembershipFun[i] * sizeof(float));
//           continue;
//       }
    //	for (int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
    //		float tmp_sum = 0;
    //		for (int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
    //			for (int l = 0; l < m2DLayer->preFeatureMapWidthMax; l++) {
//                   tmp_sum +=
//                       m2DLayer->matMembershipFeatureMap[i][j][k][l] *
//                       m2DLayer->matMembershipFeatureMap[i][j][k][l];
    //			}
    //		}
//           m2DLayer->norm_matMembershipFeatureMap[i][j] = sqrt(tmp_sum);
    //	}
    //}
    //
    if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            if(m2DLayer->flag_adapMembershipFun[i] == 0) continue;
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                for(int k = 0; k < 2; k++) {
                    switch(mode) {
                    case INIT_MODE_FRNN:
                    case INIT_BP_MODE_FRNN:
                        randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                (m2DLayer->xMax[count] - m2DLayer->xMin[count]) + m2DLayer->xMin[count]);
                        m2DLayer->para_MF_II_ratios[i][j][k] = randnum;
                        //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                        break;
                    case ASSIGN_MODE_FRNN:
                        m2DLayer->para_MF_II_ratios[i][j][k] = (MY_FLT_TYPE)x[count];
                        break;
                    case OUTPUT_ALL_MODE_FRNN:
                        x[count] = m2DLayer->para_MF_II_ratios[i][j][k];
                        break;
                    case OUTPUT_CONTINUOUS_MODE_FRNN:
                        x[count] = m2DLayer->para_MF_II_ratios[i][j][k];
                        break;
                    case OUTPUT_DISCRETE_MODE_FRNN:
                        break;
                    default:
                        printf("%s(%d): mode error for assignMember2DLayer - %d, exiting...\n",
                               __FILE__, __LINE__, mode);
                        exit(-1000);
                        break;
                    }
                    count++;
                }
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Membership 2D layer:\n");
    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            fprintf(debug_fpt, "In, %d,", i + 1);
            fprintf(debug_fpt, "MB,%d\n", j + 1);
            for(int h = 0; h < m2DLayer->preFeatureMapHeightMax; h++) {
                for(int w = 0; w < m2DLayer->preFeatureMapWidthMax; w++) {
                    fprintf(debug_fpt, "%e,", m2DLayer->mat_MFFeatMap[i][j][h][w]);
                }
                fprintf(debug_fpt, "\n");
            }
        }
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_member2DLayer(Member2DLayer* m2DLayer)
{
    if(m2DLayer->flag_MFFeatMapCoding_CP) {
        print_para_codingCP(m2DLayer->cdCP);
    } else if(m2DLayer->flag_MFFeatMapCoding_GEP) {
        print_para_codingGEP(m2DLayer->cdGEP);
    }

    printf("Membership 2D layer:\n");
    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            printf("In, %d,", i + 1);
            printf("MB,%d\n", j + 1);
            for(int h = 0; h < m2DLayer->preFeatureMapHeightMax; h++) {
                for(int w = 0; w < m2DLayer->preFeatureMapWidthMax; w++) {
                    printf("%e,", m2DLayer->mat_MFFeatMap[i][j][h][w]);
                }
                printf("\n");
            }
        }
    }
    printf("print para for Membership 2D layer done\n========================================\n");
    printf("\n");
}

void freeMember2DLayer(Member2DLayer* m2DLayer)
{
    free(m2DLayer->xMin);
    free(m2DLayer->xMax);
    free(m2DLayer->xType);

    free(m2DLayer->flag_adapMembershipFun);

    for(int i = 0; i < m2DLayer->numInput; i++) {
        free(m2DLayer->typeMembershipFun[i]);
    }
    free(m2DLayer->typeMembershipFun);

    if(m2DLayer->flag_MFFeatMapCoding_CP) {
        freeCodingCP(m2DLayer->cdCP);
    } else if(m2DLayer->flag_MFFeatMapCoding_GEP) {
        freeCodingGEP(m2DLayer->cdGEP);
    }

    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            free(m2DLayer->degreeMembership[i][j]);
        }
        free(m2DLayer->degreeMembership[i]);
    }
    free(m2DLayer->degreeMembership);

    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            for(int k = 0; k < m2DLayer->preFeatureMapHeightMax; k++) {
                free(m2DLayer->mat_MFFeatMap[i][j][k]);
            }
            free(m2DLayer->mat_MFFeatMap[i][j]);
        }
        free(m2DLayer->mat_MFFeatMap[i]);
    }
    free(m2DLayer->mat_MFFeatMap);

    for(int i = 0; i < m2DLayer->numInput; i++) {
        free(m2DLayer->norm_mat_MFFeatMap[i]);
    }
    free(m2DLayer->norm_mat_MFFeatMap);

    free(m2DLayer->mean_featureMapDataIn);

    if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        for(int i = 0; i < m2DLayer->numInput; i++) {
            for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
                free(m2DLayer->para_MF_II_ratios[i][j]);
            }
            free(m2DLayer->para_MF_II_ratios[i]);
        }
        free(m2DLayer->para_MF_II_ratios);
    }

    for(int i = 0; i < m2DLayer->numInput; i++) {
        free(m2DLayer->dataflowStatus[i]);
    }
    free(m2DLayer->dataflowStatus);

    free(m2DLayer->numMembershipFunCur);
    free(m2DLayer->numMembershipFun);

    free(m2DLayer);

    return;
}

FuzzyLayer* setupFuzzyLayer(int numInput, int* numMemship, int numRules, int typeFuzzySet, int typeRules, int typeInRuleCorNum,
                            int tag_GEP_rule, int typeConnectCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    for(int i = 0; i < numInput; i++) {
        if(numMemship[i] <= 0) {
            printf("%s(%d): Invalid value of numMemship[%d] %d, exit...\n",
                   __FILE__, __LINE__, i, numMemship[i]);
            exit(-1);
        }
    }
    if(numRules <= 0) {
        printf("%s(%d): Invalid value of numRules %d, exit...\n",
               __FILE__, __LINE__, numRules);
        exit(-1);
    }
    switch(typeFuzzySet) {
    case FUZZY_SET_I:
    case FUZZY_INTERVAL_TYPE_II:
        break;
    default:
        printf("%s(%d): Unknown typeFuzzySet %d, exit...\n",
               __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }
    switch(typeRules) {
    case PRODUCT_INFERENCE_ENGINE:
    case MINIMUM_INFERENCE_ENGINE:
        break;
    default:
        printf("%s(%d): Unknown typeRules %d, exit...\n",
               __FILE__, __LINE__, typeRules);
        exit(-1);
        break;
    }
    switch(typeInRuleCorNum) {
    case ONE_EACH_IN_TO_ONE_RULE:
    case MUL_EACH_IN_TO_ONE_RULE:
        break;
    default:
        printf("%s(%d): Unknown typeInRuleCorNum %d, exit...\n",
               __FILE__, __LINE__, typeInRuleCorNum);
        exit(-1);
        break;
    }
    switch(tag_GEP_rule) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown tag_GEP_rule %d, exit...\n",
               __FILE__, __LINE__, tag_GEP_rule);
        exit(-1);
        break;
    }
    switch(typeConnectCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, typeConnectCoding);
        exit(-1000);
        break;
    }

    FuzzyLayer* fLayer = (FuzzyLayer*)malloc(1 * sizeof(FuzzyLayer));

    fLayer->typeConnectCoding = typeConnectCoding;
    switch(fLayer->typeConnectCoding) {
    case PARA_CODING_DIRECT:
        fLayer->flag_connectCoding_CP = 0;
        fLayer->flag_connectCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        fLayer->flag_connectCoding_CP = 1;
        fLayer->flag_connectCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        fLayer->flag_connectCoding_CP = 0;
        fLayer->flag_connectCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, fLayer->typeConnectCoding);
        exit(-1000);
        break;
    }

    switch(typeFuzzySet) {
    case FUZZY_SET_I:
        fLayer->dim_degree = 1;
        break;
    case FUZZY_INTERVAL_TYPE_II:
        fLayer->dim_degree = 2;
        break;
    default:
        printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    fLayer->typeFuzzySet = typeFuzzySet;
    fLayer->typeRules = typeRules;
    fLayer->typeInRuleCorNum = typeInRuleCorNum;
    fLayer->tag_GEP_rule = tag_GEP_rule;
    fLayer->numInput = numInput;
    fLayer->numMembershipFunCur = (int*)malloc(fLayer->numInput * sizeof(int));
    fLayer->numMembershipFun = (int*)malloc(fLayer->numInput * sizeof(int));
    memcpy(fLayer->numMembershipFun, numMemship, fLayer->numInput * sizeof(int));
    fLayer->numMembershipAll = 0;
    for(int i = 0; i < fLayer->numInput; i++)
        fLayer->numMembershipAll += fLayer->numMembershipFun[i];
    fLayer->numRules = numRules;

    if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)
        fLayer->ruleGEP_numInput = fLayer->numInput;
    else
        fLayer->ruleGEP_numInput = fLayer->numMembershipAll;

    fLayer->connectStatusAll = (int***)malloc(fLayer->numRules * sizeof(int**));
    for(int i = 0; i < fLayer->numRules; i++) {
        fLayer->connectStatusAll[i] = (int**)malloc(fLayer->numInput * sizeof(int*));
        for(int j = 0; j < fLayer->numInput; j++) {
            fLayer->connectStatusAll[i][j] = (int*)malloc(fLayer->numMembershipFun[j] * sizeof(int));
        }
    }
    fLayer->dataflowStatus = (MY_FLT_TYPE*)malloc(fLayer->numRules * sizeof(MY_FLT_TYPE));

    fLayer->degreeMembs = (MY_FLT_TYPE*)malloc(fLayer->numMembershipAll * fLayer->dim_degree * sizeof(MY_FLT_TYPE));
    fLayer->degreeRules = (MY_FLT_TYPE**)malloc(fLayer->numRules * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < fLayer->numRules; i++) {
        fLayer->degreeRules[i] = (MY_FLT_TYPE*)malloc(fLayer->dim_degree * sizeof(MY_FLT_TYPE));
    }

    //
    int count = 0;
    int count_disc = 0;
    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF ||
       (fLayer->tag_GEP_rule == FLAG_STATUS_ON && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)) {
        int max_num_MF = 0;
        for(int i = 0; i < fLayer->numInput; i++) {
            if(max_num_MF < fLayer->numMembershipFun[i])
                max_num_MF = fLayer->numMembershipFun[i];
        }
        if(fLayer->flag_connectCoding_CP) {
            const int num_dim_CP = 3;
            int size_dim_CP[num_dim_CP];
            size_dim_CP[0] = fLayer->numRules;
            size_dim_CP[1] = fLayer->numInput;
            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)
                size_dim_CP[2] = max_num_MF + 1;
            else
                size_dim_CP[2] = max_num_MF;
            fLayer->cdCP = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                         PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
            count += fLayer->cdCP->numParaLocal;
            count_disc += fLayer->cdCP->numParaLocal_disc;
        } else if(fLayer->flag_connectCoding_GEP) {
            int GEP_num_input = 3;
            MY_FLT_TYPE inputMin[3] = { 0, 0, 0 };
            MY_FLT_TYPE inputMax[3];
            inputMax[0] = fLayer->numRules;
            inputMax[1] = fLayer->numInput;
            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)
                inputMax[2] = max_num_MF + 1;
            else
                inputMax[2] = max_num_MF;
            fLayer->cdGEP = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                           GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
            count += fLayer->cdGEP->numParaLocal;
            count_disc += fLayer->cdGEP->numParaLocal_disc;
        } else {
            for(int j = 0; j < fLayer->numInput; j++) {
                if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                    count += fLayer->numRules;
                    count_disc += fLayer->numRules;
                } else if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                    count += fLayer->numMembershipFun[j] * fLayer->numRules;
                    count_disc += fLayer->numMembershipFun[j] * fLayer->numRules;
                }
            }
        }
    }
    if(fLayer->tag_GEP_rule == FLAG_STATUS_ON) {
        fLayer->ruleGEP = (codingGEP**)malloc(fLayer->numRules * sizeof(codingGEP*));
        for(int i = 0; i < fLayer->numRules; i++) {
            fLayer->ruleGEP[i] = setupCodingGEP(fLayer->ruleGEP_numInput, NULL, NULL, fLayer->dim_degree, 0.5, FLAG_STATUS_ON,
                                                GEP_head_length,
                                                flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
            count += fLayer->ruleGEP[i]->numParaLocal;
            count_disc += fLayer->ruleGEP[i]->numParaLocal_disc;
        }
    }
    fLayer->numParaLocal = count;
    fLayer->numParaLocal_disc = count_disc;

    //
    fLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    fLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    fLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF ||
       (fLayer->tag_GEP_rule == FLAG_STATUS_ON && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)) {
        if(fLayer->flag_connectCoding_CP) {
            memcpy(&fLayer->xMin[count], fLayer->cdCP->xMin, fLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xMax[count], fLayer->cdCP->xMax, fLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xType[count], fLayer->cdCP->xType, fLayer->cdCP->numParaLocal * sizeof(int));
            count += fLayer->cdCP->numParaLocal;
        } else if(fLayer->flag_connectCoding_GEP) {
            memcpy(&fLayer->xMin[count], fLayer->cdGEP->xMin, fLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xMax[count], fLayer->cdGEP->xMax, fLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xType[count], fLayer->cdGEP->xType, fLayer->cdGEP->numParaLocal * sizeof(int));
            count += fLayer->cdGEP->numParaLocal;
        } else {
            for(int i = 0; i < fLayer->numRules; i++) {
                for(int j = 0; j < fLayer->numInput; j++) {
                    if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                        fLayer->xMin[count] = 0;
                        if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF)
                            fLayer->xMax[count] = 2 * fLayer->numMembershipFun[j] - 1e-6;
                        else
                            fLayer->xMax[count] = fLayer->numMembershipFun[j] - 1e-6;
                        fLayer->xType[count] = VAR_TYPE_DISCRETE;
                        count++;
                    } else if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                        for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                            fLayer->xMin[count] = 0;
                            fLayer->xMax[count] = 2 - 1e-6;
                            fLayer->xType[count] = VAR_TYPE_BINARY;
                            count++;
                        }
                    }
                }
            }
        }
    }
    if(fLayer->tag_GEP_rule == FLAG_STATUS_ON) {
        for(int i = 0; i < fLayer->numRules; i++) {
            memcpy(&fLayer->xMin[count], fLayer->ruleGEP[i]->xMin, fLayer->ruleGEP[i]->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xMax[count], fLayer->ruleGEP[i]->xMax, fLayer->ruleGEP[i]->numParaLocal * sizeof(MY_FLT_TYPE));
            memcpy(&fLayer->xType[count], fLayer->ruleGEP[i]->xType, fLayer->ruleGEP[i]->numParaLocal * sizeof(int));
            count += fLayer->ruleGEP[i]->numParaLocal;
        }
    }

    return fLayer;
}

void assignFuzzyLayer(FuzzyLayer* fLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;

    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF ||
       (fLayer->tag_GEP_rule == FLAG_STATUS_ON && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)) {
        if(fLayer->flag_connectCoding_CP) {
            assignCodingCP(fLayer->cdCP, &x[count], mode);
            count += fLayer->cdCP->numParaLocal;
            if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
                for(int i = 0; i < fLayer->numRules; i++) {
                    for(int j = 0; j < fLayer->numInput; j++) {
                        if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                            memset(fLayer->connectStatusAll[i][j], 0, fLayer->numMembershipFun[j] * sizeof(int));
                            MY_FLT_TYPE tmp_max = -1e30;
                            int tmp_ind = -1;
                            int tmp_n = 2 * fLayer->numMembershipFun[j];
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_ON)
                                tmp_n = fLayer->numMembershipFun[j];
                            for(int k = 0; k < tmp_n; k++) {
                                int tmp_in[3] = { i, j, k };
                                MY_FLT_TYPE tmp_val = 0.0;
                                decodingCP(fLayer->cdCP, tmp_in, &tmp_val);
                                if(tmp_max < tmp_val) {
                                    tmp_max = tmp_val;
                                    tmp_ind = k;
                                }
                            }
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && tmp_ind >= fLayer->numMembershipFun[j])
                                fLayer->connectStatusAll[i][j][tmp_ind - fLayer->numMembershipFun[j]] = 1;
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_ON && tmp_ind >= 0)
                                fLayer->connectStatusAll[i][j][tmp_ind] = 1;
                        } else if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                                int tmp_in[4] = { i, j, k };
                                MY_FLT_TYPE tmp_val = 0.0;
                                decodingCP(fLayer->cdCP, tmp_in, &tmp_val);
                                if(tmp_val > 0)
                                    fLayer->connectStatusAll[i][j][k] = 1;
                                else
                                    fLayer->connectStatusAll[i][j][k] = 0;
                            }
                        }
                    }
                }
            }
        } else if(fLayer->flag_connectCoding_GEP) {
            assignCodingGEP(fLayer->cdGEP, &x[count], mode);
            count += fLayer->cdGEP->numParaLocal;
            if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
                for(int i = 0; i < fLayer->numRules; i++) {
                    for(int j = 0; j < fLayer->numInput; j++) {
                        if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                            memset(fLayer->connectStatusAll[i][j], 0, fLayer->numMembershipFun[j] * sizeof(int));
                            MY_FLT_TYPE tmp_max = -1e30;
                            int tmp_ind = -1;
                            int tmp_n = 2 * fLayer->numMembershipFun[j];
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_ON)
                                tmp_n = fLayer->numMembershipFun[j];
                            for(int k = 0; k < tmp_n; k++) {
                                MY_FLT_TYPE tmp_R = i + 1;// (float)((i + 1.0) / (fLayer->numRules));
                                MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (fLayer->numInput));
                                MY_FLT_TYPE tmp_M = k + 1;// (float)((k + 1.0) / (fLayer->numMembershipFun[j]));
                                MY_FLT_TYPE tmp_in[3] = { tmp_R, tmp_I, tmp_M };
                                MY_FLT_TYPE tmp_val = 0.0;
                                decodingGEP(fLayer->cdGEP, tmp_in, &tmp_val);
                                if(tmp_max < tmp_val) {
                                    tmp_max = tmp_val;
                                    tmp_ind = k;
                                }
                            }
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && tmp_ind >= fLayer->numMembershipFun[j])
                                fLayer->connectStatusAll[i][j][tmp_ind - fLayer->numMembershipFun[j]] = 1;
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_ON && tmp_ind >= 0)
                                fLayer->connectStatusAll[i][j][tmp_ind] = 1;
                        } else if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                                MY_FLT_TYPE tmp_R = i + 1;// (float)((i + 1.0) / (fLayer->numRules));
                                MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (fLayer->numInput));
                                MY_FLT_TYPE tmp_M = k + 1;// (float)((k + 1.0) / (fLayer->numMembershipFun[j]));
                                MY_FLT_TYPE tmp_in[3] = { tmp_R, tmp_I, tmp_M };
                                MY_FLT_TYPE tmp_val = 0.0;
                                decodingGEP(fLayer->cdGEP, tmp_in, &tmp_val);
                                if(tmp_val > 0)
                                    fLayer->connectStatusAll[i][j][k] = 1;
                                else
                                    fLayer->connectStatusAll[i][j][k] = 0;
                            }
                        }
                    }
                }
            }
        } else {
            for(int i = 0; i < fLayer->numRules; i++) {
                for(int j = 0; j < fLayer->numInput; j++) {
                    if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
                        switch(mode) {
                        case INIT_MODE_FRNN:
                        case INIT_BP_MODE_FRNN:
                            memset(fLayer->connectStatusAll[i][j], 0, fLayer->numMembershipFun[j] * sizeof(int));
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                    (fLayer->xMax[count] - fLayer->xMin[count]) + fLayer->xMin[count]);
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && (int)randnum >= fLayer->numMembershipFun[j])
                                fLayer->connectStatusAll[i][j][(int)randnum - fLayer->numMembershipFun[j]] = 1;
                            else if(fLayer->tag_GEP_rule == FLAG_STATUS_ON)
                                fLayer->connectStatusAll[i][j][(int)randnum] = 1;
                            break;
                        case ASSIGN_MODE_FRNN:
                            memset(fLayer->connectStatusAll[i][j], 0, fLayer->numMembershipFun[j] * sizeof(int));
                            if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF && (int)x[count] >= fLayer->numMembershipFun[j])
                                fLayer->connectStatusAll[i][j][(int)x[count] - fLayer->numMembershipFun[j]] = 1;
                            else if(fLayer->tag_GEP_rule == FLAG_STATUS_ON)
                                fLayer->connectStatusAll[i][j][(int)x[count]] = 1;
                            //for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                            //    printf("%d ", fLayer->connectStatusAll[i][j][k]);
                            //}
                            //printf("\n");
                            break;
                        case OUTPUT_ALL_MODE_FRNN:
                            x[count] = 0;
                            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                                if(fLayer->connectStatusAll[i][j][k]) {
                                    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF)
                                        x[count] = k + fLayer->numMembershipFun[j];
                                    else
                                        x[count] = k;
                                    break;
                                }
                            }
                            break;
                        case OUTPUT_CONTINUOUS_MODE_FRNN:
                            break;
                        case OUTPUT_DISCRETE_MODE_FRNN:
                            x[count] = 0;
                            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                                if(fLayer->connectStatusAll[i][j][k]) {
                                    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF)
                                        x[count] = k + fLayer->numMembershipFun[j];
                                    else
                                        x[count] = k;
                                    break;
                                }
                            }
                            break;
                        default:
                            printf("%s(%d): mode error for assignFuzzyRuleLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                            exit(-1000);
                            break;
                        }
                        count++;
                    } else if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                        for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                            switch(mode) {
                            case INIT_MODE_FRNN:
                            case INIT_BP_MODE_FRNN:
                                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                        (fLayer->xMax[count] - fLayer->xMin[count]) + fLayer->xMin[count]);
                                fLayer->connectStatusAll[i][j][k] = (int)randnum;
                                break;
                            case ASSIGN_MODE_FRNN:
                                fLayer->connectStatusAll[i][j][k] = (int)x[count];
                                break;
                            case OUTPUT_ALL_MODE_FRNN:
                                x[count] = fLayer->connectStatusAll[i][j][k];
                                break;
                            case OUTPUT_CONTINUOUS_MODE_FRNN:
                                break;
                            case OUTPUT_DISCRETE_MODE_FRNN:
                                x[count] = fLayer->connectStatusAll[i][j][k];
                                break;
                            default:
                                printf("%s(%d): mode error for assignFuzzyRuleLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                                exit(-1000);
                                break;
                            }
                            count++;
                        }
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < fLayer->numRules; i++) {
            for(int j = 0; j < fLayer->numInput; j++) {
                memset(fLayer->connectStatusAll[i][j], 0, fLayer->numMembershipFun[j] * sizeof(int));
            }
        }
    }
    if(fLayer->tag_GEP_rule == FLAG_STATUS_ON) {
        for(int i = 0; i < fLayer->numRules; i++) {
            assignCodingGEP(fLayer->ruleGEP[i], &x[count], mode);
            count += fLayer->ruleGEP[i]->numParaLocal;
            if(fLayer->typeInRuleCorNum == MUL_EACH_IN_TO_ONE_RULE) {
                if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
                    //getConnectGEP(fLayer->ruleGEP[i]);
                    for(int n = 0; n < fLayer->ruleGEP[i]->check_tail; n++) {
                        int tmp_ind = fLayer->ruleGEP[i]->check_vInd[n];
                        if(tmp_ind >= 0 && tmp_ind < fLayer->numMembershipAll) {
                            int tmp_j = 0;
                            int tmp_k = 0;
                            int tmp_count = 0;
                            for(tmp_j = 0; tmp_j < fLayer->numInput; tmp_j++) {
                                for(tmp_k = 0; tmp_k < fLayer->numMembershipFun[tmp_j]; tmp_k++) {
                                    tmp_count++;
                                    if(tmp_count - 1 == tmp_ind) break;
                                }
                                if(tmp_count - 1 == tmp_ind) break;
                            }
                            fLayer->connectStatusAll[i][tmp_j][tmp_k]++;
                        }
                    }
                }
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Fuzzy Rules:\n");
    for(int i = 0; i < fLayer->numRules; i++) {
        fprintf(debug_fpt, "%d", i + 1);
        for(int j = 0; j < fLayer->numInput; j++) {
            int flag = 0;
            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                if(fLayer->connectStatusAll[i][j][k]) {
                    fprintf(debug_fpt, ",%d", k + 1);
                    flag = 1;
                }
            }
            if(!flag)
                fprintf(debug_fpt, ",0");
        }
        fprintf(debug_fpt, "\n");
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_fuzzyLayer(FuzzyLayer* fLayer)
{
    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF ||
       (fLayer->tag_GEP_rule == FLAG_STATUS_ON && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)) {
        if(fLayer->flag_connectCoding_CP) {
            print_para_codingCP(fLayer->cdCP);
        } else if(fLayer->flag_connectCoding_GEP) {
            print_para_codingGEP(fLayer->cdGEP);
        }
    }

    if(fLayer->tag_GEP_rule == FLAG_STATUS_ON) {
        for(int i = 0; i < fLayer->numRules; i++) {
            print_para_codingGEP(fLayer->ruleGEP[i]);
        }
    }

    printf("Fuzzy Rules:\n");
    for(int i = 0; i < fLayer->numRules; i++) {
        printf("%d", i + 1);
        for(int j = 0; j < fLayer->numInput; j++) {
            int flag = 0;
            for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                if(fLayer->connectStatusAll[i][j][k]) {
                    printf(",%d", k + 1);
                    flag = 1;
                }
            }
            if(!flag)
                printf(",0");
        }
        printf("\n");
    }
    printf("print para for Fuzzy Rules done\n========================================\n");
    printf("\n");
}

void freeFuzzyLayer(FuzzyLayer* fLayer)
{
    free(fLayer->xMin);
    free(fLayer->xMax);
    free(fLayer->xType);

    free(fLayer->numMembershipFunCur);
    free(fLayer->numMembershipFun);

    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF ||
       (fLayer->tag_GEP_rule == FLAG_STATUS_ON && fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE)) {
        if(fLayer->flag_connectCoding_CP) {
            freeCodingCP(fLayer->cdCP);
        } else if(fLayer->flag_connectCoding_GEP) {
            freeCodingGEP(fLayer->cdGEP);
        }
    }

    if(fLayer->tag_GEP_rule == FLAG_STATUS_ON) {
        for(int i = 0; i < fLayer->numRules; i++) {
            freeCodingGEP(fLayer->ruleGEP[i]);
        }
        free(fLayer->ruleGEP);
    }

    for(int i = 0; i < fLayer->numRules; i++) {
        for(int j = 0; j < fLayer->numInput; j++) {
            free(fLayer->connectStatusAll[i][j]);
        }
        free(fLayer->connectStatusAll[i]);
        free(fLayer->degreeRules[i]);
    }
    free(fLayer->connectStatusAll);
    free(fLayer->degreeRules);
    free(fLayer->dataflowStatus);

    free(fLayer->degreeMembs);

    free(fLayer);

    return;
}

RoughLayer* setupRoughLayer(int numInput, int numRoughSets, int typeFuzzySet,
                            int flagConnectStatusAdap,
                            int typeConnectCoding,
                            int numLowRankMax, int GEP_head_length, int flag_GEP_weight)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    if(numRoughSets <= 0) {
        printf("%s(%d): Invalid value of numRoughSets %d, exit...\n",
               __FILE__, __LINE__, numRoughSets);
        exit(-1);
    }
    switch(typeFuzzySet) {
    case FUZZY_SET_I:
    case FUZZY_INTERVAL_TYPE_II:
        break;
    default:
        printf("%s(%d): Unknown typeFuzzySet %d, exit...\n",
               __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }
    switch(flagConnectStatusAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flagConnectStatusAdap %d, exit...\n",
               __FILE__, __LINE__, flagConnectStatusAdap);
        exit(-1);
        break;
    }
    switch(typeConnectCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, typeConnectCoding);
        exit(-1000);
        break;
    }

    RoughLayer* rLayer = (RoughLayer*)malloc(1 * sizeof(RoughLayer));

    rLayer->typeConnectCoding = typeConnectCoding;
    switch(rLayer->typeConnectCoding) {
    case PARA_CODING_DIRECT:
        rLayer->flag_connectCoding_CP = 0;
        rLayer->flag_connectCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        rLayer->flag_connectCoding_CP = 1;
        rLayer->flag_connectCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        rLayer->flag_connectCoding_CP = 0;
        rLayer->flag_connectCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, rLayer->typeConnectCoding);
        exit(-1000);
        break;
    }

    switch(typeFuzzySet) {
    case FUZZY_SET_I:
        rLayer->dim_degree = 1;
        break;
    case FUZZY_INTERVAL_TYPE_II:
        rLayer->dim_degree = 2;
        break;
    default:
        printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    rLayer->typeFuzzySet = typeFuzzySet;

    rLayer->flagConnectStatusAdap = flagConnectStatusAdap;

    rLayer->numInput = numInput;
    rLayer->numRoughSets = numRoughSets;

    rLayer->connectStatus = (int**)malloc(rLayer->numRoughSets * sizeof(int*));
    rLayer->connectWeight = (MY_FLT_TYPE**)malloc(rLayer->numRoughSets * sizeof(MY_FLT_TYPE*));
    rLayer->dataflowStatus = (MY_FLT_TYPE*)malloc(rLayer->numRoughSets * sizeof(MY_FLT_TYPE));
    rLayer->degreeRough = (MY_FLT_TYPE**)malloc(rLayer->numRoughSets * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        rLayer->connectStatus[i] = (int*)malloc(rLayer->numInput * sizeof(int));
        rLayer->connectWeight[i] = (MY_FLT_TYPE*)malloc(rLayer->numInput * sizeof(MY_FLT_TYPE));
        rLayer->degreeRough[i] = (MY_FLT_TYPE*)malloc(rLayer->dim_degree * sizeof(MY_FLT_TYPE));
    }

    //
    int count = 0;
    int count_disc = 0;
    if(rLayer->flag_connectCoding_CP) {
        const int num_dim_CP = 2;
        int size_dim_CP[num_dim_CP];
        size_dim_CP[0] = rLayer->numRoughSets;
        size_dim_CP[1] = rLayer->numInput;
        rLayer->cdCP = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                     PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += rLayer->cdCP->numParaLocal;
        count_disc += rLayer->cdCP->numParaLocal_disc;
    } else if(rLayer->flag_connectCoding_GEP) {
        int GEP_num_input = 2;
        MY_FLT_TYPE inputMin[2] = { 0, 0};
        MY_FLT_TYPE inputMax[2];
        inputMax[0] = rLayer->numRoughSets;
        inputMax[1] = rLayer->numInput;
        rLayer->cdGEP = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                       GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += rLayer->cdGEP->numParaLocal;
        count_disc += rLayer->cdGEP->numParaLocal_disc;
    } else {
        if(rLayer->flagConnectStatusAdap == FLAG_STATUS_ON) {
            count += rLayer->numRoughSets * rLayer->numInput;
            count_disc += rLayer->numRoughSets * rLayer->numInput;
        }
        count += rLayer->numRoughSets * rLayer->numInput;
    }
    rLayer->numParaLocal = count;
    rLayer->numParaLocal_disc = count_disc;

    //
    rLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    rLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    rLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(rLayer->flag_connectCoding_CP) {
        memcpy(&rLayer->xMin[count], rLayer->cdCP->xMin, rLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&rLayer->xMax[count], rLayer->cdCP->xMax, rLayer->cdCP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&rLayer->xType[count], rLayer->cdCP->xType, rLayer->cdCP->numParaLocal * sizeof(int));
        count += rLayer->cdCP->numParaLocal;
    } else if(rLayer->flag_connectCoding_GEP) {
        memcpy(&rLayer->xMin[count], rLayer->cdGEP->xMin, rLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&rLayer->xMax[count], rLayer->cdGEP->xMax, rLayer->cdGEP->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&rLayer->xType[count], rLayer->cdGEP->xType, rLayer->cdGEP->numParaLocal * sizeof(int));
        count += rLayer->cdGEP->numParaLocal;
    } else {
        if(rLayer->flagConnectStatusAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < rLayer->numRoughSets; i++) {
                for(int j = 0; j < rLayer->numInput; j++) {
                    rLayer->xMin[count] = 0;
                    rLayer->xMax[count] = 2 - 1e-6;
                    rLayer->xType[count] = VAR_TYPE_BINARY;
                    count++;
                }
            }
        }
        for(int i = 0; i < rLayer->numRoughSets; i++) {
            for(int j = 0; j < rLayer->numInput; j++) {
                rLayer->xMin[count] = 0;
                rLayer->xMax[count] = 1;
                rLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                count++;
            }
        }
    }

    return rLayer;
}

void assignRoughLayer(RoughLayer* rLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    if(rLayer->flag_connectCoding_CP) {
        assignCodingCP(rLayer->cdCP, &x[count], mode);
        count += rLayer->cdCP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < rLayer->numRoughSets; i++) {
                for(int j = 0; j < rLayer->numInput; j++) {
                    int tmp_in[2] = { i, j };
                    MY_FLT_TYPE tmp_val = 0.0;
                    decodingCP(rLayer->cdCP, tmp_in, &tmp_val);
                    rLayer->connectWeight[i][j] = fabs(tmp_val);
                    if(rLayer->flagConnectStatusAdap == FLAG_STATUS_ON)
                        rLayer->connectStatus[i][j] = tmp_val > 0 ? 1 : 0;
                    else
                        rLayer->connectStatus[i][j] = 1;
                }
            }
        }
    } else if(rLayer->flag_connectCoding_GEP) {
        assignCodingGEP(rLayer->cdGEP, &x[count], mode);
        count += rLayer->cdGEP->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < rLayer->numRoughSets; i++) {
                for(int j = 0; j < rLayer->numInput; j++) {
                    MY_FLT_TYPE tmp_R = i + 1;// (float)((i + 1.0) / (rLayer->numRoughSets));
                    MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (rLayer->numInput));
                    MY_FLT_TYPE tmp_in[2] = { tmp_R, tmp_I };
                    MY_FLT_TYPE tmp_val = 0.0;
                    decodingGEP(rLayer->cdGEP, tmp_in, &tmp_val);
                    rLayer->connectWeight[i][j] = fabs(tmp_val);
                    if(rLayer->flagConnectStatusAdap == FLAG_STATUS_ON)
                        rLayer->connectStatus[i][j] = tmp_val > 0 ? 1 : 0;
                    else
                        rLayer->connectStatus[i][j] = 1;
                }
            }
        }
    } else {
        if(rLayer->flagConnectStatusAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < rLayer->numRoughSets; i++) {
                for(int j = 0; j < rLayer->numInput; j++) {
                    switch(mode) {
                    case INIT_MODE_FRNN:
                    case INIT_BP_MODE_FRNN:
                        randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                (rLayer->xMax[count] - rLayer->xMin[count]) + rLayer->xMin[count]);
                        rLayer->connectStatus[i][j] = (int)randnum;
                        break;
                    case ASSIGN_MODE_FRNN:
                        rLayer->connectStatus[i][j] = (int)x[count];
                        break;
                    case OUTPUT_ALL_MODE_FRNN:
                        x[count] = rLayer->connectStatus[i][j];
                        break;
                    case OUTPUT_CONTINUOUS_MODE_FRNN:
                        break;
                    case OUTPUT_DISCRETE_MODE_FRNN:
                        x[count] = rLayer->connectStatus[i][j];
                        break;
                    default:
                        printf("%s(%d): mode error for assignRoughLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                        exit(-1000);
                        break;
                    }
                    count++;
                }
            }
        } else {
            for(int i = 0; i < rLayer->numRoughSets; i++) {
                for(int j = 0; j < rLayer->numInput; j++) {
                    rLayer->connectStatus[i][j] = 1;
                }
            }
        }
        for(int i = 0; i < rLayer->numRoughSets; i++) {
            for(int j = 0; j < rLayer->numInput; j++) {
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (rLayer->xMax[count] - rLayer->xMin[count]) + rLayer->xMin[count]);
                    rLayer->connectWeight[i][j] = randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    rLayer->connectWeight[i][j] = (MY_FLT_TYPE)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = rLayer->connectWeight[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    x[count] = rLayer->connectWeight[i][j];
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    break;
                default:
                    printf("%s(%d): mode error for assignRoughLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Rough layer:\n");
    fprintf(debug_fpt, "\n");
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        fprintf(debug_fpt, "%d", i + 1);
        for(int j = 0; j < rLayer->numInput; j++) {
            fprintf(debug_fpt, ",%d,%f", rLayer->connectStatus[i][j], rLayer->connectWeight[i][j]);
        }
        fprintf(debug_fpt, "\n");
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
#endif
    return;
}

void print_para_roughLayer(RoughLayer* rLayer)
{
    if(rLayer->flag_connectCoding_CP) {
        print_para_codingCP(rLayer->cdCP);
    } else if(rLayer->flag_connectCoding_GEP) {
        print_para_codingGEP(rLayer->cdGEP);
    }

    printf("Rough layer:\n");
    printf("\n");
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        printf("%d", i + 1);
        for(int j = 0; j < rLayer->numInput; j++) {
            printf(",%d,%f", rLayer->connectStatus[i][j], rLayer->connectWeight[i][j]);
        }
        printf("\n");
    }
    printf("print para for Rough layer done\n========================================\n");
    printf("\n");
}

void freeRoughLayer(RoughLayer* rLayer)
{
    free(rLayer->xMin);
    free(rLayer->xMax);
    free(rLayer->xType);

    if(rLayer->flag_connectCoding_CP) {
        freeCodingCP(rLayer->cdCP);
    } else if(rLayer->flag_connectCoding_GEP) {
        freeCodingGEP(rLayer->cdGEP);
    }

    for(int i = 0; i < rLayer->numRoughSets; i++) {
        free(rLayer->connectStatus[i]);
        free(rLayer->connectWeight[i]);
        free(rLayer->degreeRough[i]);
    }
    free(rLayer->connectStatus);
    free(rLayer->connectWeight);
    free(rLayer->dataflowStatus);
    free(rLayer->degreeRough);

    free(rLayer);

    return;
}

OutReduceLayer* setupOutReduceLayer(int numInput, int numOutput, MY_FLT_TYPE* outputMin, MY_FLT_TYPE* outputMax,
                                    int typeFuzzySet, int typeTypeReducer,
                                    int consequenceNodeStatus, int centroid_num_tag,
                                    int numInputConsequenceNode, MY_FLT_TYPE* inputMin_cnsq, MY_FLT_TYPE* inputMax_cnsq,
                                    int flagConnectStatusAdap, int flagConnectWeightAdap,
                                    int typeConnectCoding, int numLowRankMax, int GEP_head_length, int flag_GEP_weight)
{
    if(numInput <= 0) {
        printf("%s(%d): Invalid value of numInput %d, exit...\n",
               __FILE__, __LINE__, numInput);
        exit(-1);
    }
    if(numOutput <= 0) {
        printf("%s(%d): Invalid value of numOutput %d, exit...\n",
               __FILE__, __LINE__, numOutput);
        exit(-1);
    }
    for(int i = 0; i < numOutput; i++) {
        if(outputMax[i] <= outputMin[i]) {
            printf("%s(%d): Invalid value of outputMax[%d] and outputMin[%d] ~ %lf <= %lf, exit...\n",
                   __FILE__, __LINE__, i, i, outputMax[i], outputMin[i]);
            exit(-1);
        }
    }
    switch(typeFuzzySet) {
    case FUZZY_SET_I:
    case FUZZY_INTERVAL_TYPE_II:
        break;
    default:
        printf("%s(%d): Unknown typeFuzzySet %d, exit...\n",
               __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }
    switch(typeTypeReducer) {
    case NIE_TAN_TYPE_REDUCER:
    case CENTER_OF_SETS_TYPE_REDUCER:
        break;
    default:
        printf("%s(%d): Unknown typeTypeReducer %d, exit...\n",
               __FILE__, __LINE__, typeTypeReducer);
        exit(-1);
        break;
    }
    switch(consequenceNodeStatus) {
    case NO_CONSEQUENCE_CENTROID:
    case FIXED_CONSEQUENCE_CENTROID:
    case ADAPTIVE_CONSEQUENCE_CENTROID:
        break;
    default:
        printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n",
               __FILE__, __LINE__, consequenceNodeStatus);
        exit(-1);
        break;
    }
    switch(centroid_num_tag) {
    case CENTROID_ALL_ONESET:
    case CENTROID_ONESET_EACH:
        break;
    default:
        printf("%s(%d): Unknown centroid_num_tag %d, exit...\n",
               __FILE__, __LINE__, centroid_num_tag);
        exit(-1);
        break;
    }
    if(consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID &&
       numInputConsequenceNode <= 0) {
        printf("%s(%d): Invalid value of numInputConsequenceNode %d, exit...\n",
               __FILE__, __LINE__, numInputConsequenceNode);
        exit(-1);
    }
    if(consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID &&
       (inputMin_cnsq == NULL || inputMax_cnsq == NULL)) {
        printf("%s(%d): Invalid value of inputMin_cnsq %d or inputMax_cnsq %d, exit...\n",
               __FILE__, __LINE__, inputMin_cnsq, inputMax_cnsq);
        exit(-1);
    }
    if(consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < numInputConsequenceNode; i++) {
            if(inputMax_cnsq[i] <= inputMin_cnsq[i]) {
                printf("%s(%d): Invalid value of inputMax_cnsq[%d] and inputMin_cnsq[%d] ~ %lf <= %lf, exit...\n",
                       __FILE__, __LINE__, i, i, inputMax_cnsq[i], inputMin_cnsq[i]);
                exit(-1);
            }
        }
    }
    switch(flagConnectStatusAdap) {
    case FLAG_STATUS_OFF:
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flagConnectStatusAdap %d, exit...\n",
               __FILE__, __LINE__, flagConnectStatusAdap);
        exit(-1);
        break;
    }
    switch(flagConnectWeightAdap) {
    case FLAG_STATUS_OFF:
        if(numOutput > 1 &&
           (consequenceNodeStatus == NO_CONSEQUENCE_CENTROID ||
            centroid_num_tag == CENTROID_ALL_ONESET)) {
            printf("%s(%d): Parameter setting error, if there are no consequence centroids, weights should be adaptive || multiple output with only one set of centroids, exit...\n",
                   __FILE__, __LINE__);
            exit(-1);
        }
        break;
    case FLAG_STATUS_ON:
        break;
    default:
        printf("%s(%d): Unknown flagConnectWeightAdap %d, exit...\n",
               __FILE__, __LINE__, flagConnectWeightAdap);
        exit(-1);
        break;
    }
    switch(typeConnectCoding) {
    case PARA_CODING_DIRECT:
    case PARA_CODING_CANDECOMP_PARAFAC:
    case PARA_CODING_GEP:
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, typeConnectCoding);
        exit(-1000);
        break;
    }

    OutReduceLayer* oLayer = (OutReduceLayer*)malloc(1 * sizeof(OutReduceLayer));

    oLayer->typeConnectCoding = typeConnectCoding;
    switch(oLayer->typeConnectCoding) {
    case PARA_CODING_DIRECT:
        oLayer->flag_connectCoding_CP = 0;
        oLayer->flag_connectCoding_GEP = 0;
        oLayer->flag_consqCoding_CP = 0;
        oLayer->flag_consqCoding_GEP = 0;
        break;
    case PARA_CODING_CANDECOMP_PARAFAC:
        oLayer->flag_connectCoding_CP = 1;
        oLayer->flag_connectCoding_GEP = 0;
        oLayer->flag_consqCoding_CP = 1;
        oLayer->flag_consqCoding_GEP = 0;
        break;
    case PARA_CODING_GEP:
        oLayer->flag_connectCoding_CP = 0;
        oLayer->flag_connectCoding_GEP = 1;
        oLayer->flag_consqCoding_CP = 0;
        oLayer->flag_consqCoding_GEP = 1;
        break;
    default:
        printf("%s(%d): Unknown typeConnectCoding - %d, exiting...\n",
               __FILE__, __LINE__, oLayer->typeConnectCoding);
        exit(-1000);
        break;
    }
    oLayer->flagConnectStatusAdap = flagConnectStatusAdap;
    oLayer->flagConnectWeightAdap = flagConnectWeightAdap;

    if(oLayer->flagConnectStatusAdap == FLAG_STATUS_OFF &&
       oLayer->flagConnectWeightAdap == FLAG_STATUS_OFF) {
        oLayer->flag_connectCoding_CP = 0;
        oLayer->flag_connectCoding_GEP = 0;
    }

    switch(typeFuzzySet) {
    case FUZZY_SET_I:
        oLayer->dim_degree = 1;
        break;
    case FUZZY_INTERVAL_TYPE_II:
        oLayer->dim_degree = 2;
        break;
    default:
        printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, typeFuzzySet);
        exit(-1);
        break;
    }

    oLayer->typeFuzzySet = typeFuzzySet;
    oLayer->typeTypeReducer = typeTypeReducer;

    oLayer->numInput = numInput;
    oLayer->numOutput = numOutput;
    if(numLowRankMax > oLayer->numOutput ||
       numLowRankMax > oLayer->numInput) {
        oLayer->flag_connectCoding_CP = 0;
        //oLayer->flag_connectCoding_GEP = 0;
    }

    oLayer->outputMin = (MY_FLT_TYPE*)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE));
    oLayer->outputMax = (MY_FLT_TYPE*)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE));
    memcpy(oLayer->outputMin, outputMin, oLayer->numOutput * sizeof(MY_FLT_TYPE));
    memcpy(oLayer->outputMax, outputMax, oLayer->numOutput * sizeof(MY_FLT_TYPE));

    oLayer->connectStatus = (int**)malloc(oLayer->numOutput * sizeof(int*));
    oLayer->connectWeight = (MY_FLT_TYPE**)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < oLayer->numOutput; i++) {
        oLayer->connectStatus[i] = (int*)malloc(oLayer->numInput * sizeof(int));
        oLayer->connectWeight[i] = (MY_FLT_TYPE*)malloc(oLayer->numInput * sizeof(MY_FLT_TYPE));
    }
    oLayer->dataflowStatus = (MY_FLT_TYPE*)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE));

    oLayer->consequenceNodeStatus = consequenceNodeStatus;
    switch(oLayer->consequenceNodeStatus) {
    case NO_CONSEQUENCE_CENTROID:
        break;
    case FIXED_CONSEQUENCE_CENTROID:
        break;
    case ADAPTIVE_CONSEQUENCE_CENTROID:
        oLayer->numInputConsequenceNode = numInputConsequenceNode;
        if(oLayer->numInputConsequenceNode <= 0) {
            printf("%s(%d): For ADAPTIVE_ROUGH_CENTROID, the input number should not be zero, exiting...\n",
                   __FILE__, __LINE__);
            exit(-1);
        }
        oLayer->inputMin_cnsq = (MY_FLT_TYPE*)malloc(oLayer->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        oLayer->inputMax_cnsq = (MY_FLT_TYPE*)malloc(oLayer->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        memcpy(oLayer->inputMin_cnsq, inputMin_cnsq, oLayer->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        memcpy(oLayer->inputMax_cnsq, inputMax_cnsq, oLayer->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
        oLayer->inputConsequenceNode = (MY_FLT_TYPE**)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE*));
        oLayer->paraConsequenceNode = (MY_FLT_TYPE****)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE***));
        for(int i = 0; i < oLayer->numOutput; i++) {
            oLayer->inputConsequenceNode[i] = (MY_FLT_TYPE*)malloc(oLayer->numInputConsequenceNode * sizeof(MY_FLT_TYPE));
            oLayer->paraConsequenceNode[i] = (MY_FLT_TYPE***)malloc(oLayer->numInput * sizeof(MY_FLT_TYPE**));
            for(int j = 0; j < oLayer->numInput; j++) {
                oLayer->paraConsequenceNode[i][j] = (MY_FLT_TYPE**)malloc(oLayer->dim_degree * sizeof(MY_FLT_TYPE*));
                for(int k = 0; k < oLayer->dim_degree; k++) {
                    oLayer->paraConsequenceNode[i][j][k] =
                        (MY_FLT_TYPE*)malloc((oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                }
            }
        }
        break;
    default:
        printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
        exit(-1);
        break;
    }
    oLayer->centroid_num_tag = centroid_num_tag;
    oLayer->centroidsRough = (MY_FLT_TYPE***)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE**));
    for(int i = 0; i < oLayer->numOutput; i++) {
        oLayer->centroidsRough[i] = (MY_FLT_TYPE**)malloc(oLayer->numInput * sizeof(MY_FLT_TYPE*));
        for(int j = 0; j < oLayer->numInput; j++) {
            oLayer->centroidsRough[i][j] = (MY_FLT_TYPE*)malloc(oLayer->dim_degree * sizeof(MY_FLT_TYPE));
        }
    }

    oLayer->valInputFinal = (MY_FLT_TYPE**)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < oLayer->numOutput; i++) {
        oLayer->valInputFinal[i] = (MY_FLT_TYPE*)malloc(oLayer->numInput * sizeof(MY_FLT_TYPE));
    }
    oLayer->valOutputFinal = (MY_FLT_TYPE*)malloc(oLayer->numOutput * sizeof(MY_FLT_TYPE));

    //
    int count = 0;
    int count_disc = 0;
    if(oLayer->flag_connectCoding_CP) {
        int num_dim_CP = 2;
        int size_dim_CP[2];
        size_dim_CP[0] = oLayer->numOutput;
        size_dim_CP[1] = oLayer->numInput;
        oLayer->cdCP_cw = setupCodingCP(1, num_dim_CP, size_dim_CP, numLowRankMax,
                                        PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += oLayer->cdCP_cw->numParaLocal;
        count_disc += oLayer->cdCP_cw->numParaLocal_disc;
    } else if(oLayer->flag_connectCoding_GEP) {
        int GEP_num_input = 2;
        MY_FLT_TYPE inputMin[2] = { 0, 0 };
        MY_FLT_TYPE inputMax[2];
        inputMax[0] = oLayer->numOutput;
        inputMax[1] = oLayer->numInput;
        oLayer->cdGEP_cw = setupCodingGEP(GEP_num_input, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                          GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += oLayer->cdGEP_cw->numParaLocal;
        count_disc += oLayer->cdGEP_cw->numParaLocal_disc;
    } else {
        if(oLayer->flagConnectStatusAdap == FLAG_STATUS_ON) {
            count += oLayer->numOutput * oLayer->numInput;
            count_disc += oLayer->numOutput * oLayer->numInput;
        }
        if(oLayer->flagConnectWeightAdap == FLAG_STATUS_ON)
            count += oLayer->numOutput * oLayer->numInput;
    }
    int num_dim_CP_GEP;
    int size_dim_CP_GEP[4];
    switch(oLayer->consequenceNodeStatus) {
    case NO_CONSEQUENCE_CENTROID:
        oLayer->flag_consqCoding_CP = 0;
        oLayer->flag_consqCoding_GEP = 0;
        break;
    case FIXED_CONSEQUENCE_CENTROID:
        if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER) {
                num_dim_CP_GEP = 1;
                size_dim_CP_GEP[0] = oLayer->numInput;
                oLayer->flag_consqCoding_CP = 0;
                //oLayer->flag_consqCoding_GEP = 0;
            } else {
                num_dim_CP_GEP = 1;
                size_dim_CP_GEP[0] = oLayer->numInput;
                oLayer->flag_consqCoding_CP = 0;
                //oLayer->flag_consqCoding_GEP = 0;
                if(oLayer->dim_degree > 1) {
                    num_dim_CP_GEP++;
                    size_dim_CP_GEP[1] = oLayer->dim_degree;
                }
            }
        } else {
            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER) {
                num_dim_CP_GEP = 2;
                size_dim_CP_GEP[0] = oLayer->numOutput;
                size_dim_CP_GEP[1] = oLayer->numInput;
                if(numLowRankMax > size_dim_CP_GEP[0] ||
                   numLowRankMax > size_dim_CP_GEP[1]) {
                    oLayer->flag_consqCoding_CP = 0;
                    //oLayer->flag_consqCoding_GEP = 0;
                }
            } else {
                num_dim_CP_GEP = 2;
                size_dim_CP_GEP[0] = oLayer->numOutput;
                size_dim_CP_GEP[1] = oLayer->numInput;
                if(numLowRankMax > size_dim_CP_GEP[0] ||
                   numLowRankMax > size_dim_CP_GEP[1]) {
                    oLayer->flag_consqCoding_CP = 0;
                    //oLayer->flag_consqCoding_GEP = 0;
                }
                if(oLayer->dim_degree > 1) {
                    num_dim_CP_GEP++;
                    size_dim_CP_GEP[2] = oLayer->dim_degree;
                }
            }
        }
        break;
    case ADAPTIVE_CONSEQUENCE_CENTROID:
        if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER) {
                num_dim_CP_GEP = 2;
                size_dim_CP_GEP[0] = oLayer->numInput;
                size_dim_CP_GEP[1] = oLayer->numInputConsequenceNode + 1;
                if(numLowRankMax > size_dim_CP_GEP[0] ||
                   numLowRankMax > size_dim_CP_GEP[1]) {
                    oLayer->flag_consqCoding_CP = 0;
                    //oLayer->flag_consqCoding_GEP = 0;
                }
            } else {
                num_dim_CP_GEP = 2;
                size_dim_CP_GEP[0] = oLayer->numInput;
                size_dim_CP_GEP[1] = oLayer->numInputConsequenceNode + 1;
                if(numLowRankMax > size_dim_CP_GEP[0] ||
                   numLowRankMax > size_dim_CP_GEP[1]) {
                    if(oLayer->dim_degree == 1) {
                        oLayer->flag_consqCoding_CP = 0;
                        //oLayer->flag_consqCoding_GEP = 0;
                    }
                }
                if(oLayer->dim_degree > 1) {
                    num_dim_CP_GEP++;
                    size_dim_CP_GEP[2] = size_dim_CP_GEP[1];
                    size_dim_CP_GEP[1] = oLayer->dim_degree;
                }
            }
        } else {
            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER) {
                num_dim_CP_GEP = 3;
                size_dim_CP_GEP[0] = oLayer->numOutput;
                size_dim_CP_GEP[1] = oLayer->numInput;
                size_dim_CP_GEP[2] = oLayer->numInputConsequenceNode + 1;
            } else {
                num_dim_CP_GEP = 3;
                size_dim_CP_GEP[0] = oLayer->numOutput;
                size_dim_CP_GEP[1] = oLayer->numInput;
                size_dim_CP_GEP[2] = oLayer->numInputConsequenceNode + 1;
                if(oLayer->dim_degree > 1) {
                    num_dim_CP_GEP++;
                    size_dim_CP_GEP[3] = size_dim_CP_GEP[2];
                    size_dim_CP_GEP[2] = oLayer->dim_degree;
                }
            }
        }
        break;
    default:
        printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n",
               __FILE__, __LINE__, oLayer->consequenceNodeStatus);
        exit(-1);
        break;
    }
    if(oLayer->flag_consqCoding_CP) {
        oLayer->cdCP_cq = setupCodingCP(1, num_dim_CP_GEP, size_dim_CP_GEP, numLowRankMax,
                                        PARA_MIN_VAL_CP_CFRNN_MODEL, PARA_MAX_VAL_CP_CFRNN_MODEL);
        count += oLayer->cdCP_cq->numParaLocal;
        count_disc += oLayer->cdCP_cq->numParaLocal_disc;
    } else if(oLayer->flag_consqCoding_GEP) {
        MY_FLT_TYPE inputMin[4] = { 0, 0, 0, 0 };
        MY_FLT_TYPE inputMax[4];
        for(int i = 0; i < num_dim_CP_GEP; i++) inputMax[i] = size_dim_CP_GEP[i];
        oLayer->cdGEP_cq = setupCodingGEP(num_dim_CP_GEP, inputMin, inputMax, 1, 0.5, FLAG_STATUS_OFF,
                                          GEP_head_length, flag_GEP_weight, PARA_MIN_VAL_GEP_CFRNN_MODEL, PARA_MAX_VAL_GEP_CFRNN_MODEL);
        count += oLayer->cdGEP_cq->numParaLocal;
        count_disc += oLayer->cdGEP_cq->numParaLocal_disc;
    } else {
        switch(oLayer->consequenceNodeStatus) {
        case NO_CONSEQUENCE_CENTROID:
            break;
        case FIXED_CONSEQUENCE_CENTROID:
            if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
                if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                    count += oLayer->numInput;
                else
                    count += oLayer->numInput * oLayer->dim_degree;
            } else {
                if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                    count += oLayer->numOutput * oLayer->numInput;
                else
                    count += oLayer->numOutput * oLayer->numInput * oLayer->dim_degree;
            }
            break;
        case ADAPTIVE_CONSEQUENCE_CENTROID:
            if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
                if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                    count += oLayer->numInput * (oLayer->numInputConsequenceNode + 1);
                else
                    count += oLayer->numInput * oLayer->dim_degree * (oLayer->numInputConsequenceNode + 1);
            } else {
                if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                    count += oLayer->numOutput * oLayer->numInput * (oLayer->numInputConsequenceNode + 1);
                else
                    count += oLayer->numOutput * oLayer->numInput * oLayer->dim_degree * (oLayer->numInputConsequenceNode + 1);
            }
            break;
        default:
            printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n",
                   __FILE__, __LINE__, oLayer->consequenceNodeStatus);
            exit(-1);
            break;
        }
    }
    oLayer->numParaLocal = count;
    oLayer->numParaLocal_disc = count_disc;

    //
    oLayer->xMin = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    oLayer->xMax = (MY_FLT_TYPE*)malloc((count + 1) * sizeof(MY_FLT_TYPE));
    oLayer->xType = (int*)malloc((count + 1) * sizeof(int));

    count = 0;
    if(oLayer->flag_connectCoding_CP) {
        memcpy(&oLayer->xMin[count], oLayer->cdCP_cw->xMin, oLayer->cdCP_cw->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xMax[count], oLayer->cdCP_cw->xMax, oLayer->cdCP_cw->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xType[count], oLayer->cdCP_cw->xType, oLayer->cdCP_cw->numParaLocal * sizeof(int));
        count += oLayer->cdCP_cw->numParaLocal;
    } else if(oLayer->flag_connectCoding_GEP) {
        memcpy(&oLayer->xMin[count], oLayer->cdGEP_cw->xMin, oLayer->cdGEP_cw->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xMax[count], oLayer->cdGEP_cw->xMax, oLayer->cdGEP_cw->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xType[count], oLayer->cdGEP_cw->xType, oLayer->cdGEP_cw->numParaLocal * sizeof(int));
        count += oLayer->cdGEP_cw->numParaLocal;
    } else {
        if(oLayer->flagConnectStatusAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < oLayer->numOutput; i++) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    oLayer->xMin[count] = 0;
                    oLayer->xMax[count] = 2 - 1e-6;
                    oLayer->xType[count] = VAR_TYPE_BINARY;
                    count++;
                }
            }
        }
        if(oLayer->flagConnectWeightAdap == FLAG_STATUS_ON) {
            for(int i = 0; i < oLayer->numOutput; i++) {
                for(int j = 0; j < oLayer->numInput; j++) {
#ifdef UTILIZE_MKL_LAPACKE_IN_MOPS_LINUX_ONLY
#if MY_FLT_TYPE == double
                    oLayer->xMin[count] = -1e306;
                    oLayer->xMax[count] = 1e306;
#else
                    oLayer->xMin[count] = -1e36;
                    oLayer->xMax[count] = 1e36;
#endif
#else
                    oLayer->xMin[count] = -10;
                    oLayer->xMax[count] = 10;
#endif
                    oLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                    count++;
                }
            }
        }
    }
    if(oLayer->flag_consqCoding_CP) {
        memcpy(&oLayer->xMin[count], oLayer->cdCP_cq->xMin, oLayer->cdCP_cq->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xMax[count], oLayer->cdCP_cq->xMax, oLayer->cdCP_cq->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xType[count], oLayer->cdCP_cq->xType, oLayer->cdCP_cq->numParaLocal * sizeof(int));
        count += oLayer->cdCP_cq->numParaLocal;
    } else if(oLayer->flag_consqCoding_GEP) {
        memcpy(&oLayer->xMin[count], oLayer->cdGEP_cq->xMin, oLayer->cdGEP_cq->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xMax[count], oLayer->cdGEP_cq->xMax, oLayer->cdGEP_cq->numParaLocal * sizeof(MY_FLT_TYPE));
        memcpy(&oLayer->xType[count], oLayer->cdGEP_cq->xType, oLayer->cdGEP_cq->numParaLocal * sizeof(int));
        count += oLayer->cdGEP_cq->numParaLocal;
    } else {
        MY_FLT_TYPE tmp_max_in_sum = 0;
        switch(oLayer->consequenceNodeStatus) {
        case NO_CONSEQUENCE_CENTROID:
            break;
        case FIXED_CONSEQUENCE_CENTROID:
            for(int i = 0; i < oLayer->numOutput; i++) {
                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) continue;
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) continue;
                        oLayer->xMin[count] = oLayer->outputMin[i];
                        oLayer->xMax[count] = oLayer->outputMax[i];
                        oLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                        count++;
                    }
                }
            }
            break;
        case ADAPTIVE_CONSEQUENCE_CENTROID:
            for(int i = 0; i < oLayer->numOutput; i++) {
                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) continue;
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) continue;
                        tmp_max_in_sum = 0;
                        for(int m = 0; m < oLayer->numInputConsequenceNode; m++) {
                            tmp_max_in_sum += inputMax_cnsq[m] * PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL;
                            oLayer->xMin[count] = PARA_MIN_CONNECT_WEIGHT_CFRNN_MODEL;
                            oLayer->xMax[count] = PARA_MAX_CONNECT_WEIGHT_CFRNN_MODEL;
                            oLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                            count++;
                        }
                        oLayer->xMin[count] = -tmp_max_in_sum - outputMax[i];
                        oLayer->xMax[count] = tmp_max_in_sum + outputMax[i];
                        oLayer->xType[count] = VAR_TYPE_CONTINUOUS;
                        count++;
                    }
                }
            }
            break;
        default:
            printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
            exit(-1);
            break;
        }
    }

    return oLayer;
}

void assignOutReduceLayer(OutReduceLayer* oLayer, double* x, int mode)
{
    MY_FLT_TYPE randnum;
    int count = 0;
    if(oLayer->flag_connectCoding_CP) {
        assignCodingCP(oLayer->cdCP_cw, &x[count], mode);
        count += oLayer->cdCP_cw->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < oLayer->numOutput; i++) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    int tmp_in[2] = { i, j };
                    MY_FLT_TYPE tmp_val = 0.0;
                    decodingCP(oLayer->cdCP_cw, tmp_in, &tmp_val);
                    oLayer->connectWeight[i][j] = oLayer->flagConnectWeightAdap == FLAG_STATUS_OFF ? 1 : tmp_val;
                    oLayer->connectStatus[i][j] = tmp_val > 0 ? 1 : 0;
                    if(oLayer->flagConnectStatusAdap == FLAG_STATUS_OFF)
                        oLayer->connectStatus[i][j] = 1;
                }
            }
        }
    } else if(oLayer->flag_connectCoding_GEP) {
        assignCodingGEP(oLayer->cdGEP_cw, &x[count], mode);
        count += oLayer->cdGEP_cw->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            for(int i = 0; i < oLayer->numOutput; i++) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    MY_FLT_TYPE tmp_R = i + 1;// (float)((i + 1.0) / (oLayer->numOutput));
                    MY_FLT_TYPE tmp_I = j + 1;// (float)((j + 1.0) / (oLayer->numInput));
                    MY_FLT_TYPE tmp_in[2] = { tmp_R, tmp_I };
                    MY_FLT_TYPE tmp_val = 0.0;
                    decodingGEP(oLayer->cdGEP_cw, tmp_in, &tmp_val);
                    oLayer->connectWeight[i][j] = oLayer->flagConnectWeightAdap == FLAG_STATUS_OFF ? 1 : tmp_val;
                    oLayer->connectStatus[i][j] = tmp_val > 0 ? 1 : 0;
                    if(oLayer->flagConnectStatusAdap == FLAG_STATUS_OFF)
                        oLayer->connectStatus[i][j] = 1;
                }
            }
        }
    } else {
        for(int i = 0; i < oLayer->numOutput; i++) {
            for(int j = 0; j < oLayer->numInput; j++) {
                if(oLayer->flagConnectStatusAdap == FLAG_STATUS_OFF) {
                    oLayer->connectStatus[i][j] = 1;
                    continue;
                }
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (oLayer->xMax[count] - oLayer->xMin[count]) + oLayer->xMin[count]);
                    oLayer->connectStatus[i][j] = (int)(randnum);
                    break;
                case ASSIGN_MODE_FRNN:
                    oLayer->connectStatus[i][j] = (int)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = oLayer->connectStatus[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    x[count] = oLayer->connectStatus[i][j];
                    break;
                default:
                    printf("%s(%d): mode error for assignOutputLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        }
        for(int i = 0; i < oLayer->numOutput; i++) {
            for(int j = 0; j < oLayer->numInput; j++) {
                if(oLayer->flagConnectWeightAdap == FLAG_STATUS_OFF) {
                    oLayer->connectWeight[i][j] = 1;
                    continue;
                }
                switch(mode) {
                case INIT_MODE_FRNN:
                case INIT_BP_MODE_FRNN:
                    randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                            (oLayer->xMax[count] - oLayer->xMin[count]) + oLayer->xMin[count]);
                    oLayer->connectWeight[i][j] = randnum;
                    break;
                case ASSIGN_MODE_FRNN:
                    oLayer->connectWeight[i][j] = (MY_FLT_TYPE)x[count];
                    break;
                case OUTPUT_ALL_MODE_FRNN:
                    x[count] = oLayer->connectWeight[i][j];
                    break;
                case OUTPUT_CONTINUOUS_MODE_FRNN:
                    x[count] = oLayer->connectWeight[i][j];
                    break;
                case OUTPUT_DISCRETE_MODE_FRNN:
                    break;
                default:
                    printf("%s(%d): mode error for assignOutputLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                    exit(-1000);
                    break;
                }
                count++;
            }
        }
    }
    //
    if(oLayer->flag_consqCoding_CP) {
        assignCodingCP(oLayer->cdCP_cq, &x[count], mode);
        count += oLayer->cdCP_cq->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            switch(oLayer->consequenceNodeStatus) {
            case NO_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            oLayer->centroidsRough[i][j][k] = 1.0;
                        }
                    }
                }
                break;
            case FIXED_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            for(int k = 0; k < oLayer->dim_degree; k++) {
                                oLayer->centroidsRough[i][j][k] =
                                    oLayer->centroidsRough[0][j][k];
                            }
                        }
                        continue;
                    }
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                                oLayer->centroidsRough[i][j][k] = oLayer->centroidsRough[i][j][0];
                                continue;
                            }
                            int tmp_in[3] = { i, j, k };
                            MY_FLT_TYPE tmp_val = 0;
                            decodingCP(oLayer->cdCP_cq, tmp_in, &tmp_val);
                            oLayer->centroidsRough[i][j][k] = tmp_val;
                        }
                    }
                }
                break;
            case ADAPTIVE_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            for(int k = 0; k < oLayer->dim_degree; k++) {
                                memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[0][j][k],
                                       (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                            }
                        }
                        continue;
                    }
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                                memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[i][j][0],
                                       (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                                continue;
                            }
                            for(int m = 0; m < oLayer->numInputConsequenceNode + 1; m++) {
                                int tmp_in[4];
                                MY_FLT_TYPE tmp_val = 0;
                                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
                                    if(oLayer->dim_degree == 1) {
                                        tmp_in[0] = j;
                                        tmp_in[1] = m;
                                    } else {
                                        tmp_in[0] = j;
                                        tmp_in[1] = k;
                                        tmp_in[2] = m;
                                    }
                                    decodingCP(oLayer->cdCP_cq, tmp_in, &tmp_val);
                                    oLayer->paraConsequenceNode[i][j][k][m] = tmp_val;
                                } else {
                                    if(oLayer->dim_degree == 1) {
                                        tmp_in[0] = i;
                                        tmp_in[1] = j;
                                        tmp_in[2] = m;
                                    } else {
                                        tmp_in[0] = i;
                                        tmp_in[1] = j;
                                        tmp_in[2] = k;
                                        tmp_in[2] = m;
                                    }
                                    decodingCP(oLayer->cdCP_cq, tmp_in, &tmp_val);
                                    oLayer->paraConsequenceNode[i][j][k][m] = tmp_val;
                                }
                            }
                        }
                    }
                }
                break;
            default:
                printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
                exit(-1);
                break;
            }
        }
    } else if(oLayer->flag_consqCoding_GEP) {
        assignCodingGEP(oLayer->cdGEP_cq, &x[count], mode);
        count += oLayer->cdGEP_cq->numParaLocal;
        if(mode == INIT_MODE_FRNN || mode == ASSIGN_MODE_FRNN) {
            switch(oLayer->consequenceNodeStatus) {
            case NO_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            oLayer->centroidsRough[i][j][k] = 1.0;
                        }
                    }
                }
                break;
            case FIXED_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            for(int k = 0; k < oLayer->dim_degree; k++) {
                                oLayer->centroidsRough[i][j][k] =
                                    oLayer->centroidsRough[0][j][k];
                            }
                        }
                        continue;
                    }
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                                oLayer->centroidsRough[i][j][k] = oLayer->centroidsRough[i][j][0];
                                continue;
                            }
                            MY_FLT_TYPE tmp_in[3];
                            tmp_in[0] = i + 1;
                            tmp_in[1] = j + 1;
                            tmp_in[2] = k + 1;
                            MY_FLT_TYPE tmp_val = 0;
                            decodingGEP(oLayer->cdGEP_cq, tmp_in, &tmp_val);
                            oLayer->centroidsRough[i][j][k] = tmp_val;
                        }
                    }
                }
                break;
            case ADAPTIVE_CONSEQUENCE_CENTROID:
                for(int i = 0; i < oLayer->numOutput; i++) {
                    if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            for(int k = 0; k < oLayer->dim_degree; k++) {
                                memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[0][j][k],
                                       (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                            }
                        }
                        continue;
                    }
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                                memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[i][j][0],
                                       (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                                continue;
                            }
                            for(int m = 0; m < oLayer->numInputConsequenceNode + 1; m++) {
                                MY_FLT_TYPE tmp_in[4];
                                MY_FLT_TYPE tmp_val = 0;
                                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET) {
                                    if(oLayer->dim_degree == 1) {
                                        tmp_in[0] = j + 1;
                                        tmp_in[1] = m + 1;
                                    } else {
                                        tmp_in[0] = j + 1;
                                        tmp_in[1] = k + 1;
                                        tmp_in[2] = m + 1;
                                    }
                                    decodingGEP(oLayer->cdGEP_cq, tmp_in, &tmp_val);
                                    oLayer->paraConsequenceNode[i][j][k][m] = tmp_val;
                                } else {
                                    if(oLayer->dim_degree == 1) {
                                        tmp_in[0] = i + 1;
                                        tmp_in[1] = j + 1;
                                        tmp_in[2] = m + 1;
                                    } else {
                                        tmp_in[0] = i + 1;
                                        tmp_in[1] = j + 1;
                                        tmp_in[2] = k + 1;
                                        tmp_in[2] = m + 1;
                                    }
                                    decodingGEP(oLayer->cdGEP_cq, tmp_in, &tmp_val);
                                    oLayer->paraConsequenceNode[i][j][k][m] = tmp_val;
                                }
                            }
                        }
                    }
                }
                break;
            default:
                printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
                exit(-1);
                break;
            }
        }
    } else {
        switch(oLayer->consequenceNodeStatus) {
        case NO_CONSEQUENCE_CENTROID:
            for(int i = 0; i < oLayer->numOutput; i++) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        oLayer->centroidsRough[i][j][k] = 1.0;
                    }
                }
            }
            break;
        case FIXED_CONSEQUENCE_CENTROID:
            for(int i = 0; i < oLayer->numOutput; i++) {
                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            oLayer->centroidsRough[i][j][k] =
                                oLayer->centroidsRough[0][j][k];
                        }
                    }
                    continue;
                }
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                            oLayer->centroidsRough[i][j][k] = oLayer->centroidsRough[i][j][0];
                            continue;
                        }
                        switch(mode) {
                        case INIT_MODE_FRNN:
                        case INIT_BP_MODE_FRNN:
                            randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                    (oLayer->xMax[count] - oLayer->xMin[count]) + oLayer->xMin[count]);
                            oLayer->centroidsRough[i][j][k] = randnum;
                            break;
                        case ASSIGN_MODE_FRNN:
                            oLayer->centroidsRough[i][j][k] = (MY_FLT_TYPE)x[count];
                            break;
                        case OUTPUT_ALL_MODE_FRNN:
                            x[count] = oLayer->centroidsRough[i][j][k];
                            break;
                        case OUTPUT_CONTINUOUS_MODE_FRNN:
                            x[count] = oLayer->centroidsRough[i][j][k];
                            break;
                        case OUTPUT_DISCRETE_MODE_FRNN:
                            break;
                        default:
                            printf("%s(%d): mode error for assignOutputLayer - %d, exiting...\n", __FILE__, __LINE__, mode);
                            exit(-1000);
                            break;
                        }
                        count++;
                    }
                }
            }
            break;
        case ADAPTIVE_CONSEQUENCE_CENTROID:
            for(int i = 0; i < oLayer->numOutput; i++) {
                if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                    for(int j = 0; j < oLayer->numInput; j++) {
                        for(int k = 0; k < oLayer->dim_degree; k++) {
                            memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[0][j][k],
                                   (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                        }
                    }
                    continue;
                }
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) {
                            memcpy(oLayer->paraConsequenceNode[i][j][k], oLayer->paraConsequenceNode[i][j][0],
                                   (oLayer->numInputConsequenceNode + 1) * sizeof(MY_FLT_TYPE));
                            continue;
                        }
                        for(int m = 0; m <= oLayer->numInputConsequenceNode; m++) {
                            switch(mode) {
                            case INIT_MODE_FRNN:
                            case INIT_BP_MODE_FRNN:
                                randnum = (MY_FLT_TYPE)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) *
                                                        (oLayer->xMax[count] - oLayer->xMin[count]) + oLayer->xMin[count]);
                                oLayer->paraConsequenceNode[i][j][k][m] = randnum;
                                break;
                            case ASSIGN_MODE_FRNN:
                                oLayer->paraConsequenceNode[i][j][k][m] = (MY_FLT_TYPE)x[count];
                                break;
                            case OUTPUT_ALL_MODE_FRNN:
                                x[count] = oLayer->paraConsequenceNode[i][j][k][m];
                                break;
                            case OUTPUT_CONTINUOUS_MODE_FRNN:
                                x[count] = oLayer->paraConsequenceNode[i][j][k][m];
                                break;
                            case OUTPUT_DISCRETE_MODE_FRNN:
                                break;
                            default:
                                printf("%s(%d): mode error for assignOutputLayer - %d, exiting...\n",
                                       __FILE__, __LINE__, mode);
                                exit(-1000);
                                break;
                            }
                            count++;
                        }
                    }
                }
            }
            break;
        default:
            printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
            exit(-1);
            break;
        }
    }
    //
#ifdef MY_DEBUG_TAG3
    char debug_fn[1024];
    sprintf(debug_fn, "FRNN_info_%d.csv", debug_count_FRNN);
    FILE* debug_fpt = fopen(debug_fn, "a");
    fprintf(debug_fpt, "Output layer:\n");
    fprintf(debug_fpt, "\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        fprintf(debug_fpt, "%d", i + 1);
        for(int j = 0; j < oLayer->numInput; j++) {
            fprintf(debug_fpt, ",%d", oLayer->connectStatus[i][j]);
        }
        fprintf(debug_fpt, "\n");
    }
    fprintf(debug_fpt, "\n");
    fclose(debug_fpt);
    debug_count_FRNN++;
#endif
    return;
}

void print_para_outReduceLayer(OutReduceLayer* oLayer)
{
    if(oLayer->flag_connectCoding_CP) {
        print_para_codingCP(oLayer->cdCP_cw);
    } else if(oLayer->flag_connectCoding_GEP) {
        print_para_codingGEP(oLayer->cdGEP_cw);
    }

    if(oLayer->flag_consqCoding_CP) {
        print_para_codingCP(oLayer->cdCP_cq);
    } else if(oLayer->flag_consqCoding_GEP) {
        print_para_codingGEP(oLayer->cdGEP_cq);
    }

    printf("Output layer:\n");
    printf("\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        printf("%d", i + 1);
        for(int j = 0; j < oLayer->numInput; j++) {
            printf(",%d", oLayer->connectStatus[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        printf("%d", i + 1);
        for(int j = 0; j < oLayer->numInput; j++) {
            printf(",%lf", oLayer->connectWeight[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    switch(oLayer->consequenceNodeStatus) {
    case NO_CONSEQUENCE_CENTROID:
        break;
    case FIXED_CONSEQUENCE_CENTROID:
        break;
    case ADAPTIVE_CONSEQUENCE_CENTROID:
        for(int i = 0; i < oLayer->numOutput; i++) {
            if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) continue;
            for(int j = 0; j < oLayer->numInput; j++) {
                for(int k = 0; k < oLayer->dim_degree; k++) {
                    if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER && k) continue;
                    for(int m = 0; m < oLayer->numInputConsequenceNode; m++) {
                        printf("%e,", oLayer->paraConsequenceNode[i][j][k][m]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
        break;
    default:
        printf("%s(%d): Unknown ROUGH_CENTROID_TYPE %d, exit...\n", __FILE__, __LINE__, oLayer->consequenceNodeStatus);
        exit(-1);
        break;
    }
    printf("print para for Output layer done\n========================================\n");
    printf("\n");
}

void freeOutReduceLayer(OutReduceLayer* oLayer)
{
    if(oLayer->flag_connectCoding_CP) {
        freeCodingCP(oLayer->cdCP_cw);
    } else if(oLayer->flag_connectCoding_GEP) {
        freeCodingGEP(oLayer->cdGEP_cw);
    }

    if(oLayer->flag_consqCoding_CP) {
        freeCodingCP(oLayer->cdCP_cq);
    } else if(oLayer->flag_consqCoding_GEP) {
        freeCodingGEP(oLayer->cdGEP_cq);
    }

    free(oLayer->xMin);
    free(oLayer->xMax);
    free(oLayer->xType);

    free(oLayer->outputMin);
    free(oLayer->outputMax);

    for(int i = 0; i < oLayer->numOutput; i++) {
        free(oLayer->connectStatus[i]);
        free(oLayer->connectWeight[i]);
    }
    free(oLayer->connectStatus);
    free(oLayer->connectWeight);
    free(oLayer->dataflowStatus);

    if(oLayer->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        free(oLayer->inputMin_cnsq);
        free(oLayer->inputMax_cnsq);
        for(int i = 0; i < oLayer->numOutput; i++) {
            free(oLayer->inputConsequenceNode[i]);
            for(int j = 0; j < oLayer->numInput; j++) {
                for(int k = 0; k < oLayer->dim_degree; k++) {
                    free(oLayer->paraConsequenceNode[i][j][k]);
                }
                free(oLayer->paraConsequenceNode[i][j]);
            }
            free(oLayer->paraConsequenceNode[i]);
        }
        free(oLayer->inputConsequenceNode);
        free(oLayer->paraConsequenceNode);
    }

    for(int i = 0; i < oLayer->numOutput; i++) {
        for(int j = 0; j < oLayer->numInput; j++) {
            free(oLayer->centroidsRough[i][j]);
        }
        free(oLayer->centroidsRough[i]);
    }
    free(oLayer->centroidsRough);

    for(int i = 0; i < oLayer->numOutput; i++) {
        free(oLayer->valInputFinal[i]);
    }
    free(oLayer->valInputFinal);
    free(oLayer->valOutputFinal);

    free(oLayer);

    return;
}

//////////////////////////////////////////////////////////////////////////
void ff_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus)
{
    memcpy(cLayer->inputHeight, inputHeight, cLayer->channelsInMax * sizeof(int));
    memcpy(cLayer->inputWidth, inputWidth, cLayer->channelsInMax * sizeof(int));

    for(int i = 0; i < cLayer->channelsOutMax; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            memset(cLayer->featureMapData[i][h], 0, cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            memset(cLayer->featureMapTag[i][h], 0, cLayer->featureMapWidthMax * sizeof(int));
            memset(cLayer->dataflowStatus[i][h], 0, cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }
    memset(cLayer->featureMapHeight, 0, cLayer->channelsOutMax * sizeof(int));
    memset(cLayer->featureMapWidth, 0, cLayer->channelsOutMax * sizeof(int));

    for(int i = 0; i < cLayer->channelsOut; i++) {
        memset(cLayer->kernelFlagCountAll[i], 0, cLayer->channelsInMax * sizeof(int));
    }

    cLayer->kernelFlagCount = 0;
    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_SKIP) continue;
            int heightInCur = cLayer->inputHeight[j];
            int widthInCur = cLayer->inputWidth[j];
            int kernelHeightCur = cLayer->kernelHeight[i][j];
            int kernelWidthCur = cLayer->kernelWidth[i][j];
            int heightInOffset = (cLayer->inputHeightMax - heightInCur) / 2;
            int widthInOffset = (cLayer->inputWidthMax - widthInCur) / 2;
            int kernelHeightOffset = (cLayer->kernelHeightMax - kernelHeightCur) / 2;
            int kernelWidthOffset = (cLayer->kernelWidthMax - kernelWidthCur) / 2;
            int heightOutCur = heightInCur;
            int widthOutCur = widthInCur;
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_COPY) {
                if(heightOutCur <= 0 || widthOutCur <= 0) continue;
                if(heightOutCur > cLayer->featureMapHeight[i]) cLayer->featureMapHeight[i] = heightOutCur;
                if(widthOutCur > cLayer->featureMapWidth[i]) cLayer->featureMapWidth[i] = widthOutCur;
                int heightOutOffset = (cLayer->featureMapHeightMax - heightOutCur) / 2;
                int widthOutOffset = (cLayer->featureMapWidthMax - widthOutCur) / 2;
                int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
                int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
                int tmp_count_pixel = 0;
                for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
                    for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                        if(h < heightOutOffset ||
                           w < widthOutOffset ||
                           h >= heightOutOffset + heightOutCur ||
                           w >= widthOutOffset + widthOutCur ||
                           featureMapTagIn[j][heightOutOffsetIn + h][widthOutOffsetIn + w] == 0)
                            continue;
                        MY_FLT_TYPE tmp_conv = 0;
                        int tmp_count = 0;
                        MY_FLT_TYPE tmp_count_data = 0;
                        int heightPadding = kernelHeightCur / 2;
                        int widthPadding = kernelWidthCur / 2;
                        int inHCur = heightOutOffsetIn + h;
                        int inWCur = widthOutOffsetIn + w;
                        if(inHCur < heightInOffset ||
                           inWCur < widthInOffset ||
                           inHCur >= heightInOffset + heightInCur ||
                           inWCur >= widthInOffset + widthInCur ||
                           featureMapTagIn[j][inHCur][inWCur] == 0)
                            continue;
                        tmp_count++;
                        tmp_count_data += dataflowStatus[j][inHCur][inWCur];
                        tmp_conv += featureMapDataIn[j][inHCur][inWCur];
                        cLayer->featureMapData[i][h][w] += tmp_conv;
                        cLayer->featureMapTag[i][h][w]++;
                        tmp_count_pixel++;
                        cLayer->dataflowStatus[i][h][w] += tmp_count_data;
                    }
                }
                cLayer->kernelFlagCountAll[i][j] = tmp_count_pixel;
                if(tmp_count_pixel) cLayer->kernelFlagCount++;
                continue;
            }
            int kernelHeightHalf = kernelHeightCur / 2;
            int kernelWidthHalf = kernelWidthCur / 2;
            if(cLayer->paddingType[i][j] == PADDING_VALID) {
                heightOutCur -= 2 * kernelHeightHalf;
                widthOutCur -= 2 * kernelWidthHalf;
            }
            if(heightOutCur <= 0 || widthOutCur <= 0) continue;
            if(heightOutCur > cLayer->featureMapHeight[i]) cLayer->featureMapHeight[i] = heightOutCur;
            if(widthOutCur > cLayer->featureMapWidth[i]) cLayer->featureMapWidth[i] = widthOutCur;
            int heightOutOffset = (cLayer->featureMapHeightMax - heightOutCur) / 2;
            int widthOutOffset = (cLayer->featureMapWidthMax - widthOutCur) / 2;
            int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
            int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
            int tmp_count_pixel = 0;
            for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
                for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                    if(h < heightOutOffset ||
                       w < widthOutOffset ||
                       h >= heightOutOffset + heightOutCur ||
                       w >= widthOutOffset + widthOutCur ||
                       featureMapTagIn[j][heightOutOffsetIn + h][widthOutOffsetIn + w] == 0)
                        continue;
                    MY_FLT_TYPE tmp_conv = 0;
                    int tmp_count = 0;
                    MY_FLT_TYPE tmp_count_data = 0;
                    int heightPadding = kernelHeightCur / 2;
                    int widthPadding = kernelWidthCur / 2;
                    for(int a = 0; a < kernelHeightCur; a++) {
                        for(int b = 0; b < kernelWidthCur; b++) {
                            int inHCur = heightOutOffsetIn + h + a - heightPadding;
                            int inWCur = widthOutOffsetIn + w + b - widthPadding;
                            if(inHCur < heightInOffset ||
                               inWCur < widthInOffset ||
                               inHCur >= heightInOffset + heightInCur ||
                               inWCur >= widthInOffset + widthInCur ||
                               featureMapTagIn[j][inHCur][inWCur] == 0)
                                continue;
                            tmp_count++;
                            tmp_count_data += dataflowStatus[j][inHCur][inWCur];
                            tmp_conv += featureMapDataIn[j][inHCur][inWCur] *
                                        cLayer->kernelData[i][j][a + kernelHeightOffset][b + kernelWidthOffset];
                        }
                    }
                    if(tmp_count == kernelHeightCur * kernelWidthCur ||
                       (tmp_count > 0 && cLayer->paddingType[i][j] == PADDING_SAME)) {
                        cLayer->featureMapData[i][h][w] += tmp_conv;
                        cLayer->featureMapTag[i][h][w]++;
                        tmp_count_pixel++;
                        cLayer->dataflowStatus[i][h][w] += tmp_count_data;
                    }
                }
            }
            cLayer->kernelFlagCountAll[i][j] = tmp_count_pixel;
            if(tmp_count_pixel) cLayer->kernelFlagCount++;
        }
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                if(cLayer->featureMapTag[i][h][w])
                    cLayer->featureMapData[i][h][w] += cLayer->biasData[i];
            }
        }
    }

    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                if(!cLayer->featureMapTag[i][h][w]) continue;
                MY_FLT_TYPE temp_in = cLayer->featureMapData[i][h][w];
                MY_FLT_TYPE tmp1;
                MY_FLT_TYPE tmp2;
                MY_FLT_TYPE temp_out;
                switch(cLayer->kernelType[i]) {
                case ACT_FUNC_SIGMA:
                    temp_out = (MY_FLT_TYPE)1.0 / ((MY_FLT_TYPE)(1.0 + exp(-temp_in)));
                    break;
                case ACT_FUNC_RELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.0;
                    break;
                case ACT_FUNC_TANH:
                    tmp1 = exp(temp_in);
                    tmp2 = exp(-temp_in);
                    temp_out = (tmp1 - tmp2) / (tmp1 + tmp2);
                    break;
                case ACT_FUNC_LEAKYRELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * temp_in;
                    break;
                case ACT_FUNC_ELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * (exp(temp_in) - 1);
                    break;
                default:
                    printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                           __FILE__, __LINE__, cLayer->kernelType[i]);
                    exit(-1);
                    break;
                }
                cLayer->featureMapData[i][h][w] = temp_out;
            }
        }
    }

    if(cLayer->kernelFlagCount == 0) {
#if LAYER_SKIP_TAG_CFRNN_MODEL == 1
        for(int i = 0; i < cLayer->channelsInMax; i++) {
            int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
            int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
            for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
                for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                    if(heightOutOffsetIn + h < 0 ||
                       heightOutOffsetIn + h >= cLayer->inputHeightMax ||
                       widthOutOffsetIn + w < 0 ||
                       widthOutOffsetIn + w >= cLayer->inputWidthMax)
                        continue;
                    cLayer->featureMapData[i][h][w] = featureMapDataIn[i][heightOutOffsetIn + h][widthOutOffsetIn + w];
                    cLayer->featureMapTag[i][h][w] = featureMapTagIn[i][heightOutOffsetIn + h][widthOutOffsetIn + w];
                    cLayer->dataflowStatus[i][h][w] = dataflowStatus[i][heightOutOffsetIn + h][widthOutOffsetIn + w];
                }
            }
            cLayer->featureMapHeight[i] = cLayer->featureMapHeightMax < inputHeight[i] ?
                                          cLayer->featureMapHeightMax : inputHeight[i];
            cLayer->featureMapWidth[i] = cLayer->featureMapWidthMax < inputWidth[i] ?
                                         cLayer->featureMapWidthMax : inputWidth[i];
        }
#else

#endif
    }

#ifdef MY_DEBUG_TAG
    printf("\ncLayer->featureMapData[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%f ", cLayer->featureMapData[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\ncLayer->featureMapTag[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%d ", cLayer->featureMapTag[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\ncLayer->dataflowStatus[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%f ", cLayer->dataflowStatus[iOut][h][w]);
            }
            printf("\n");
        }
    }
#endif

    return;
}

void print_data_convLayer(ConvolutionLayer* cLayer)
{
    printf("\nConvolutional layer:\n");
    printf("\ncLayer->featureMapData[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%f ", cLayer->featureMapData[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\ncLayer->featureMapTag[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%d ", cLayer->featureMapTag[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\ncLayer->dataflowStatus[][][]\n");
    for(int iOut = 0; iOut < cLayer->channelsOutMax; iOut++) {
        printf("H: %d, W: %d\n", cLayer->featureMapHeight[iOut], cLayer->featureMapWidth[iOut]);
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                printf("%f ", cLayer->dataflowStatus[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("print data for Convolutional layer done\n----------------------------------------\n\n");
    //
    return;
}

void ff_poolLayer(PoolLayer* pLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus)
{
    memcpy(pLayer->inputHeight, inputHeight, pLayer->channelsInOutMax * sizeof(int));
    memcpy(pLayer->inputWidth, inputWidth, pLayer->channelsInOutMax * sizeof(int));

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            memset(pLayer->featureMapData[i][h], 0, pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
            memset(pLayer->featureMapPos[i][h], 0, pLayer->featureMapWidthMax * sizeof(int));
            memset(pLayer->featureMapTag[i][h], 0, pLayer->featureMapWidthMax * sizeof(int));
            memset(pLayer->dataflowStatus[i][h], 0, pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }
    memset(pLayer->featureMapHeight, 0, pLayer->channelsInOutMax * sizeof(int));
    memset(pLayer->featureMapWidth, 0, pLayer->channelsInOutMax * sizeof(int));

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        if(pLayer->poolFlag[i] == 0) continue;
        int heightInCur = pLayer->inputHeight[i];
        int widthInCur = pLayer->inputWidth[i];
        int kernelHeightCur = pLayer->poolHeight[i];
        int kernelWidthCur = pLayer->poolWidth[i];
        int heightInOffset = (pLayer->inputHeightMax - heightInCur) / 2;
        int widthInOffset = (pLayer->inputWidthMax - widthInCur) / 2;
        int kernelHeightOffset = (pLayer->poolHeightMax - kernelHeightCur) / 2;
        int kernelWidthOffset = (pLayer->poolWidthMax - kernelWidthCur) / 2;
        int heightOutCur = (heightInCur + kernelHeightCur - 1) / kernelHeightCur;
        int widthOutCur = (widthInCur + kernelWidthCur - 1) / kernelWidthCur;
        if(heightOutCur > pLayer->featureMapHeight[i]) pLayer->featureMapHeight[i] = heightOutCur;
        if(widthOutCur > pLayer->featureMapWidth[i]) pLayer->featureMapWidth[i] = widthOutCur;
        int heightOutOffset = (pLayer->featureMapHeightMax - heightOutCur) / 2;
        int widthOutOffset = (pLayer->featureMapWidthMax - widthOutCur) / 2;
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                if(h < heightOutOffset ||
                   w < widthOutOffset ||
                   h >= heightOutOffset + heightOutCur ||
                   w >= widthOutOffset + widthOutCur)
                    continue;
                MY_FLT_TYPE tmp_pool = 0;
                int tmp_count = 0;
                MY_FLT_TYPE tmp_count_data = 0;
                int tmp_pos = 0;
                //int tmp_inHcur;
                //int tmp_inWcur;
                for(int a = 0; a < kernelHeightCur; a++) {
                    for(int b = 0; b < kernelWidthCur; b++) {
                        int inHCur = heightInOffset + (h - heightOutOffset) * kernelHeightCur + a;
                        int inWCur = widthInOffset + (w - widthOutOffset) * kernelWidthCur + b;
                        if(inHCur < heightInOffset ||
                           inWCur < widthInOffset ||
                           inHCur >= heightInOffset + heightInCur ||
                           inWCur >= widthInOffset + widthInCur ||
                           featureMapTagIn[i][inHCur][inWCur] == 0)
                            continue;
                        if(tmp_count == 0) {
                            tmp_pool = featureMapDataIn[i][inHCur][inWCur];
                            tmp_count_data = dataflowStatus[i][inHCur][inWCur];
                            tmp_pos = a * kernelWidthCur + b;
                        } else {
                            switch(pLayer->poolType[i]) {
                            case POOL_AVE:
                                tmp_pool += featureMapDataIn[i][inHCur][inWCur];
                                tmp_count_data += dataflowStatus[i][inHCur][inWCur];
                                break;
                            case POOL_MIN:
                                if(tmp_pool > featureMapDataIn[i][inHCur][inWCur]) {
                                    tmp_pool = featureMapDataIn[i][inHCur][inWCur];
                                    tmp_count_data = dataflowStatus[i][inHCur][inWCur];
                                    tmp_pos = a * kernelWidthCur + b;
                                }
                                break;
                            case POOL_MAX:
                                if(tmp_pool < featureMapDataIn[i][inHCur][inWCur]) {
                                    tmp_pool = featureMapDataIn[i][inHCur][inWCur];
                                    tmp_count_data = dataflowStatus[i][inHCur][inWCur];
                                    tmp_pos = a * kernelWidthCur + b;
                                }
                                break;
                            default:
                                printf("%s(%d): Unknown POOL_TYPE %d, exit...\n",
                                       __FILE__, __LINE__, pLayer->poolType[i]);
                                exit(-1);
                                break;
                            }
                        }
                        tmp_count++;
                    }
                }
                if(tmp_count > 0) {
                    switch(pLayer->poolType[i]) {
                    case POOL_AVE:
                        pLayer->featureMapData[i][h][w] = tmp_pool / tmp_count;
                        pLayer->featureMapTag[i][h][w] = tmp_count;
                        break;
                    case POOL_MIN:
                    case POOL_MAX:
                        pLayer->featureMapData[i][h][w] = tmp_pool;
                        pLayer->featureMapTag[i][h][w] = tmp_count;
                        pLayer->featureMapPos[i][h][w] = tmp_pos;
                        break;
                    default:
                        printf("%s(%d): Unknown POOL_TYPE %d, exit...\n",
                               __FILE__, __LINE__, pLayer->poolType[i]);
                        exit(-1);
                        break;
                    }
                    pLayer->dataflowStatus[i][h][w] += tmp_count_data;
                }
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("\npLayer->featureMapData[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%f ", pLayer->featureMapData[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\npLayer->featureMapTag[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%d ", pLayer->featureMapTag[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\npLayer->dataflowStatus[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%f ", pLayer->dataflowStatus[iOut][h][w]);
            }
            printf("\n");
        }
    }
#endif

    return;
}

void print_data_poolLayer(PoolLayer* pLayer)
{
    printf("\nPool layer:\n");
    printf("\npLayer->featureMapData[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%f ", pLayer->featureMapData[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\npLayer->featureMapTag[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%d ", pLayer->featureMapTag[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("\npLayer->dataflowStatus[][][]\n");
    for(int iOut = 0; iOut < pLayer->channelsInOutMax; iOut++) {
        printf("H: %d, W: %d\n", pLayer->featureMapHeight[iOut], pLayer->featureMapWidth[iOut]);
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                printf("%f ", pLayer->dataflowStatus[iOut][h][w]);
            }
            printf("\n");
        }
    }
    printf("print data for Pool layer done\n----------------------------------------\n\n");
}

void ff_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                  int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus)
{
    memcpy(icfcLayer->preInputHeight, inputHeight, icfcLayer->preFeatureMapChannels * sizeof(int));
    memcpy(icfcLayer->preInputWidth, inputWidth, icfcLayer->preFeatureMapChannels * sizeof(int));

    memset(icfcLayer->outputData, 0, icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    memset(icfcLayer->dataflowStatus, 0, icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    memset(icfcLayer->connectCountAll, 0, icfcLayer->numOutput * sizeof(int));
    icfcLayer->connectCountSum = 0;

    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
            int heightInCur = icfcLayer->preInputHeight[iCh];
            int widthInCur = icfcLayer->preInputWidth[iCh];
            if(heightInCur == 0 || widthInCur == 0) continue;
            int quoHeight = icfcLayer->preFeatureMapHeightMax / heightInCur;
            int remHeight = icfcLayer->preFeatureMapHeightMax % heightInCur;
            int quoWidth = icfcLayer->preFeatureMapWidthMax / widthInCur;
            int remWidth = icfcLayer->preFeatureMapWidthMax % widthInCur;
            int heightInOffset = (icfcLayer->preFeatureMapHeightMax - heightInCur) / 2;
            int widthInOffset = (icfcLayer->preFeatureMapWidthMax - widthInCur) / 2;
            int offsetHeight = 0;
            int offsetWidth = 0;
            for(int iH = 0; iH < icfcLayer->preInputHeight[iCh]; iH++) {
                int kernelHeightCur = quoHeight;
                if(iH < remHeight) kernelHeightCur++;
                offsetWidth = 0;
                for(int iW = 0; iW < icfcLayer->preInputWidth[iCh]; iW++) {
                    int kernelWidthCur = quoWidth;
                    if(iW < remWidth) kernelWidthCur++;
                    int inHcur = heightInOffset + iH;
                    int inWcur = widthInOffset + iW;
                    if(inHcur < heightInOffset ||
                       inWcur < widthInOffset ||
                       inHcur >= heightInOffset + heightInCur ||
                       inWcur >= widthInOffset + widthInCur ||
                       featureMapTagIn[iCh][inHcur][inWcur] == 0)
                        continue;
                    MY_FLT_TYPE tmp_weight = 0;
                    int tmp_count = 0;
                    for(int h = 0; h < kernelHeightCur; h++) {
                        for(int w = 0; w < kernelWidthCur; w++) {
                            int wtHcur = offsetHeight + h;
                            int wtWcur = offsetWidth + w;
                            if(wtHcur < 0 ||
                               wtWcur < 0 ||
                               wtHcur >= icfcLayer->preFeatureMapHeightMax ||
                               wtWcur >= icfcLayer->preFeatureMapWidthMax) {
                                printf("%s(%d): Error out of range, exiting...\n",
                                       __FILE__, __LINE__);
                                exit(-1);
                            }
                            if(icfcLayer->connectStatusAll[iOut][iCh][wtHcur][wtWcur]) {
                                tmp_weight += icfcLayer->connectWeightAll[iOut][iCh][wtHcur][wtWcur];
                                tmp_count++;
                            }
                        }
                    }
                    if(tmp_count) {
                        tmp_weight /= tmp_count;
                        icfcLayer->outputData[iOut] += featureMapDataIn[iCh][inHcur][inWcur] * tmp_weight;
                        icfcLayer->dataflowStatus[iOut] += dataflowStatus[iCh][inHcur][inWcur];
                        icfcLayer->connectCountAll[iOut]++;
                        icfcLayer->connectCountSum++;
                    }
                    offsetWidth += kernelWidthCur;
                }
                offsetHeight += kernelHeightCur;
            }
        }
        if(icfcLayer->connectCountAll[iOut]) {
            icfcLayer->outputData[iOut] += icfcLayer->biasData[iOut];
        }
    }

    if(icfcLayer->flagActFunc == FLAG_STATUS_ON) {
        for(int i = 0; i < icfcLayer->numOutput; i++) {
            if(!icfcLayer->connectCountAll[i]) continue;
            MY_FLT_TYPE temp_in = icfcLayer->outputData[i];
            MY_FLT_TYPE tmp1;
            MY_FLT_TYPE tmp2;
            MY_FLT_TYPE temp_out;
            switch(icfcLayer->actFuncType[i]) {
            case ACT_FUNC_SIGMA:
                temp_out = (MY_FLT_TYPE)1.0 / ((MY_FLT_TYPE)(1.0 + exp(-temp_in)));
                break;
            case ACT_FUNC_RELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.0;
                break;
            case ACT_FUNC_TANH:
                tmp1 = exp(temp_in);
                tmp2 = exp(-temp_in);
                temp_out = (tmp1 - tmp2) / (tmp1 + tmp2);
                break;
            case ACT_FUNC_LEAKYRELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * temp_in;
                break;
            case ACT_FUNC_ELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * (exp(temp_in) - 1);
                break;
            default:
                printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                       __FILE__, __LINE__, icfcLayer->actFuncType[i]);
                exit(-1);
                break;
            }
            icfcLayer->outputData[i] = temp_out;
        }
    }

    if(icfcLayer->flag_normalize_outData) {
        MY_FLT_TYPE tmp_min = icfcLayer->outputData[0];
        MY_FLT_TYPE tmp_max = icfcLayer->outputData[0];
        for(int i = 1; i < icfcLayer->numOutput; i++) {
            if(tmp_min > icfcLayer->outputData[i])
                tmp_min = icfcLayer->outputData[i];
            if(tmp_max < icfcLayer->outputData[i])
                tmp_max = icfcLayer->outputData[i];
        }
        if(tmp_max > tmp_min) {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                icfcLayer->outputData[i] = (icfcLayer->outputData[i] - tmp_min) / (tmp_max - tmp_min);
            }
        } else {
            for(int i = 0; i < icfcLayer->numOutput; i++) {
                if(tmp_min >= 0 && tmp_min <= 1)
                    icfcLayer->outputData[i] = tmp_min;
                else
                    icfcLayer->outputData[i] = 0;
            }
        }
    }

#ifdef MY_DEBUG_TAG_OUTPUT
    printf("\nicfcLayer->outputData[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%f ", icfcLayer->outputData[iOut]);
    }
    printf("\nicfcLayer->dataflowStatus[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%f ", icfcLayer->dataflowStatus[iOut]);
    }
    printf("\nicfcLayer->connectCountAll[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%d ", icfcLayer->connectCountAll[iOut]);
    }
    printf("\nicfcLayer->connectCountSum\n");
    printf("%d ", icfcLayer->connectCountSum);
#endif

    return;
}

void print_data_icfcLayer(InterCPCLayer* icfcLayer)
{
    printf("\nInterCFC layer:\n");
    printf("\nicfcLayer->outputData[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%f ", icfcLayer->outputData[iOut]);
    }
    printf("\nicfcLayer->dataflowStatus[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%f ", icfcLayer->dataflowStatus[iOut]);
    }
    printf("\nicfcLayer->connectCountAll[iOut]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%d ", icfcLayer->connectCountAll[iOut]);
    }
    printf("\nicfcLayer->connectCountSum\n");
    printf("%d ", icfcLayer->connectCountSum);
    printf("print data for InterCFC layer done\n----------------------------------------\n\n");
}

void ff_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* theDataIn, int* theTagIn, int numDataIn, MY_FLT_TYPE* dataflowStatus)
{
    memset(fcLayer->outputData, 0, fcLayer->numOutputMax * sizeof(MY_FLT_TYPE));
    memset(fcLayer->connectCountAll, 0, fcLayer->numOutputMax * sizeof(int));
    fcLayer->connectCount = 0;

    for(int i = 0; i < fcLayer->numOutput; i++) {
        MY_FLT_TYPE temp = 0;
        fcLayer->dataflowStatus[i] = 0;
        for(int j = 0; j < numDataIn; j++) {
            if(theTagIn[j] && fcLayer->connectStatus[i][j]) {
                temp += theDataIn[j] * fcLayer->connectWeight[i][j];
                fcLayer->connectCountAll[i]++;
                fcLayer->connectCount++;
                fcLayer->dataflowStatus[i] += dataflowStatus[j];
            }
        }
        fcLayer->outputData[i] = temp + fcLayer->biasData[i];
    }
    fcLayer->numOutputCur = fcLayer->numOutput;

    if(fcLayer->flagActFunc == FLAG_STATUS_ON) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            MY_FLT_TYPE temp_in = fcLayer->outputData[i];
            MY_FLT_TYPE tmp1;
            MY_FLT_TYPE tmp2;
            MY_FLT_TYPE temp_out;
            switch(fcLayer->actFuncType[i]) {
            case ACT_FUNC_SIGMA:
                temp_out = (MY_FLT_TYPE)1.0 / ((MY_FLT_TYPE)(1.0 + exp(-temp_in)));
                break;
            case ACT_FUNC_RELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.0;
                break;
            case ACT_FUNC_TANH:
                tmp1 = exp(temp_in);
                tmp2 = exp(-temp_in);
                temp_out = (tmp1 - tmp2) / (tmp1 + tmp2);
                break;
            case ACT_FUNC_LEAKYRELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * temp_in;
                break;
            case ACT_FUNC_ELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)temp_in : (MY_FLT_TYPE)0.01 * (exp(temp_in) - 1);
                break;
            default:
                printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                       __FILE__, __LINE__, fcLayer->actFuncType[i]);
                exit(-1);
                break;
            }
            fcLayer->outputData[i] = temp_out;
        }
    }

    if(fcLayer->connectCount == 0) {
#if LAYER_SKIP_TAG_CFRNN_MODEL == 1
        memcpy(fcLayer->outputData, theDataIn, numDataIn * sizeof(MY_FLT_TYPE));
        memcpy(fcLayer->connectCountAll, theTagIn, numDataIn * sizeof(int));
        memcpy(fcLayer->dataflowStatus, dataflowStatus, numDataIn * sizeof(MY_FLT_TYPE));
        fcLayer->numOutputCur = numDataIn;
#else

#endif
    }

#ifdef MY_DEBUG_TAG
    printf("\nfcLayer->outputData[iOut]\n");
    for(int iOut = 0; iOut < fcLayer->numOutputMax; iOut++) {
        printf("%f ", fcLayer->outputData[iOut]);
    }
    printf("\nfcLayer->dataflowStatus[iOut]\n");
    for(int iOut = 0; iOut < fcLayer->numOutputMax; iOut++) {
        printf("%f ", fcLayer->dataflowStatus[iOut]);
    }
    printf("\nfcLayer->connectCount\n");
    printf("%d ", fcLayer->connectCount);
#endif

    return;
}

void print_data_fcLayer(FCLayer* fcLayer)
{
    printf("print data for FC layer:\n");
    printf("\nfcLayer->outputData[iOut]\n");
    for(int iOut = 0; iOut < fcLayer->numOutputMax; iOut++) {
        printf("%f ", fcLayer->outputData[iOut]);
    }
    printf("\nfcLayer->dataflowStatus[iOut]\n");
    for(int iOut = 0; iOut < fcLayer->numOutputMax; iOut++) {
        printf("%f ", fcLayer->dataflowStatus[iOut]);
    }
    printf("\nfcLayer->connectCount\n");
    printf("%d ", fcLayer->connectCount);
    printf("print data for FC layer done\n----------------------------------------\n\n");
}

void ff_memberLayer(MemberLayer* mLayer, MY_FLT_TYPE* valInput, MY_FLT_TYPE* dataflowStatus)
{
    memcpy(mLayer->valInput, valInput, mLayer->numInput * sizeof(MY_FLT_TYPE));

    for(int i = 0; i < mLayer->numInput; i++) {
        memset(mLayer->dataflowStatus[i], 0, mLayer->numMembershipFun[i] * sizeof(MY_FLT_TYPE));
        if(dataflowStatus[i] == 0) continue;
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            MY_FLT_TYPE cur_in = mLayer->valInput[i];
            MY_FLT_TYPE cur_out;
            if(mLayer->flag_adapMembershipFun[i]) {
                cur_in = (cur_in - mLayer->inputMin[i]) / (mLayer->inputMax[i] - mLayer->inputMin[i]);
                MY_FLT_TYPE sigma1 = mLayer->paraMembershipFun[i][j][0];
                MY_FLT_TYPE c1 = mLayer->paraMembershipFun[i][j][1];
                MY_FLT_TYPE sigma2 = mLayer->paraMembershipFun[i][j][2];
                MY_FLT_TYPE c2 = mLayer->paraMembershipFun[i][j][3];
                MY_FLT_TYPE gamma = mLayer->paraMembershipFun[i][j][4];
                MY_FLT_TYPE c3 = mLayer->paraMembershipFun[i][j][5];
                MY_FLT_TYPE a1 = fabs(mLayer->paraMembershipFun[i][j][6]) + 1e-6;
                MY_FLT_TYPE a2 = fabs(mLayer->paraMembershipFun[i][j][7]) + 1e-6;
                int t11 = 0, t12 = 0, t21 = 0, t22 = 0;
                switch(mLayer->typeMembershipFun[i][j]) {
                case GAUSSIAN_MEM_FUNC:
                    cur_out = (MY_FLT_TYPE)(exp(-(cur_in - c1) * (cur_in - c1) / 2 / (sigma1 * sigma1 + 1e-6)));
                    break;
                case GAUSSIAN_COMB_MEM_FUNC:
                    if(cur_in <= c1) {
                        t11 = 1;
                        t12 = 0;
                    } else {
                        t11 = 0;
                        t12 = 1;
                    }
                    if(cur_in >= c2) {
                        t21 = 1;
                        t22 = 0;
                    } else {
                        t21 = 0;
                        t22 = 1;
                    }
                    cur_out = (t11 * exp(-(cur_in - c1) * (cur_in - c1) / 2 / (sigma1 * sigma1 + 1e-6)) + t12) *
                              (t21 * exp(-(cur_in - c2) * (cur_in - c2) / 2 / (sigma2 * sigma2 + 1e-6)) + t22);
                    break;
                case SIGMOID_MEM_FUNC:
                    cur_out = 1 / (1 + exp(-gamma * (cur_in - c3)));
                    break;
                default:
                    printf("%s(%d): Unknown MF, exiting...\n", __FILE__, __LINE__);
                    //printf("%d\n", mLayer->typeMembershipFun[i][j]);
                    exit(-1234567);
                    break;
                }
                switch(mLayer->typeFuzzySet) {
                case FUZZY_SET_I:
                    mLayer->degreeMembership[i][j][0] = cur_out;
                    break;
                case FUZZY_INTERVAL_TYPE_II:
                    mLayer->degreeMembership[i][j][1] = pow((1 - pow((1 - cur_out), a1)), (1 / a1));
                    mLayer->degreeMembership[i][j][0] = pow((1 - pow((1 - cur_out), a2)), (1 / a2));
                    break;
                default:
                    printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, mLayer->typeFuzzySet);
                    exit(-1);
                    break;
                }
            } else {
                switch(mLayer->typeFuzzySet) {
                case FUZZY_SET_I:
                    if((int)cur_in == j)
                        mLayer->degreeMembership[i][j][0] = 1;
                    else
                        mLayer->degreeMembership[i][j][0] = 0;
                    break;
                case FUZZY_INTERVAL_TYPE_II:
                    if((int)cur_in == j)
                        mLayer->degreeMembership[i][j][0] = mLayer->degreeMembership[i][j][1] = 1;
                    else
                        mLayer->degreeMembership[i][j][0] = mLayer->degreeMembership[i][j][1] = 0;
                    break;
                default:
                    printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, mLayer->typeFuzzySet);
                    exit(-1);
                    break;
                }
            }
            mLayer->dataflowStatus[i][j] = dataflowStatus[i];
        }
    }

#ifdef MY_DEBUG_TAG
    printf("\nmLayer->degreeMembership[i][j][0]\n");
    for(int i = 0; i < mLayer->numInput; i++) {
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            printf("%e ", mLayer->degreeMembership[i][j][0]);
        }
        printf("\n");
    }
#endif
    //
    return;
}

void print_data_memberLayer(MemberLayer* mLayer)
{
    printf("\nMembership layer:\n");
    printf("\nmLayer->degreeMembership[i][j][]\n");
    for(int i = 0; i < mLayer->numInput; i++) {
        for(int j = 0; j < mLayer->numMembershipFun[i]; j++) {
            printf("(%e", mLayer->degreeMembership[i][j][0]);
            for(int k = 1; k < mLayer->dim_degree; k++)
                printf(" %e", mLayer->degreeMembership[i][j][0]);
            printf("),");

        }
        printf("\n");
    }
    printf("print data for Membership layer done\n----------------------------------------\n\n");
}

void ff_member2DLayer(Member2DLayer* m2DLayer, MY_FLT_TYPE*** featureMapDataIn, int*** featureMapTagIn,
                      int* inputHeight, int* inputWidth, MY_FLT_TYPE*** dataflowStatus)
{
    for(int i = 0; i < m2DLayer->numInput; i++) {
        int heightInCur = inputHeight[i];
        int widthInCur = inputWidth[i];
        int heightInOffset = (m2DLayer->preFeatureMapHeightMax - heightInCur) / 2;
        int widthInOffset = (m2DLayer->preFeatureMapWidthMax - widthInCur) / 2;
        MY_FLT_TYPE tmp_in_norm = 0;
        MY_FLT_TYPE dataflowCount = 0;
        MY_FLT_TYPE tmp_feat_sum = 0;
        int tmp_feat_cnt = 0;
        for(int j = 0; j < inputHeight[i]; j++) {
            for(int k = 0; k < inputWidth[i]; k++) {
                if(featureMapTagIn[i][heightInOffset + j][widthInOffset + k]) {
                    tmp_in_norm +=
                        featureMapDataIn[i][heightInOffset + j][widthInOffset + k] *
                        featureMapDataIn[i][heightInOffset + j][widthInOffset + k];
                    dataflowCount += dataflowStatus[i][heightInOffset + j][widthInOffset + k];
                    tmp_feat_sum += featureMapDataIn[i][heightInOffset + j][widthInOffset + k];
                    tmp_feat_cnt++;
                }
            }
        }
        if(tmp_feat_cnt > 0)
            m2DLayer->mean_featureMapDataIn[i] = tmp_feat_sum / tmp_feat_cnt;
        else
            m2DLayer->mean_featureMapDataIn[i] = 0;
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            if(m2DLayer->flag_adapMembershipFun[i]) {
                MY_FLT_TYPE cur_out = 0;
                MY_FLT_TYPE tmp_mf_norm = 0;
                for(int k = 0; k < inputHeight[i]; k++) {
                    for(int l = 0; l < inputWidth[i]; l++) {
                        if(featureMapTagIn[i][heightInOffset + k][widthInOffset + l]) {
                            tmp_mf_norm +=
                                m2DLayer->mat_MFFeatMap[i][j][heightInOffset + k][widthInOffset + l] *
                                m2DLayer->mat_MFFeatMap[i][j][heightInOffset + k][widthInOffset + l];
                        }
                    }
                }
                MY_FLT_TYPE tmp_diff = 0;
                MY_FLT_TYPE tmp_denom = 0;
                switch(m2DLayer->typeMembershipFun[i][j]) {
                case MAT_SIMILARITY_T_COS:
                    for(int k = 0; k < inputHeight[i]; k++) {
                        for(int l = 0; l < inputWidth[i]; l++) {
                            if(featureMapTagIn[i][heightInOffset + k][widthInOffset + l]) {
                                tmp_diff +=
                                    featureMapDataIn[i][heightInOffset + k][widthInOffset + l] *
                                    m2DLayer->mat_MFFeatMap[i][j][heightInOffset + k][widthInOffset + l];
                            }
                        }
                    }
                    tmp_denom = sqrt(tmp_in_norm) * sqrt(tmp_mf_norm);
                    tmp_denom = tmp_denom > FLT_EPSILON ? tmp_denom : FLT_EPSILON;
                    cur_out = tmp_diff / tmp_denom;
                    cur_out = (1 + cur_out) / 2;
                    break;
                case MAT_SIMILARITY_T_Norm2:
                    for(int k = 0; k < inputHeight[i]; k++) {
                        for(int l = 0; l < inputWidth[i]; l++) {
                            if(featureMapTagIn[i][heightInOffset + k][widthInOffset + l]) {
                                tmp_diff +=
                                    (featureMapDataIn[i][heightInOffset + k][widthInOffset + l] -
                                     m2DLayer->mat_MFFeatMap[i][j][heightInOffset + k][widthInOffset + l]) *
                                    (featureMapDataIn[i][heightInOffset + k][widthInOffset + l] -
                                     m2DLayer->mat_MFFeatMap[i][j][heightInOffset + k][widthInOffset + l]);
                            }
                        }
                    }
                    tmp_denom = (0.5 * (sqrt(tmp_in_norm) + sqrt(tmp_mf_norm)));
                    tmp_denom = tmp_denom > FLT_EPSILON ? tmp_denom : FLT_EPSILON;
                    cur_out = sqrt(tmp_diff) / tmp_denom;
                    cur_out = 1 - cur_out;
                    break;
                default:
                    printf("%s(%d): Unknown MEMBERSHIP_FUNC_2D_MAT_SIMILARITY_TYPE %d, exit...\n",
                           __FILE__, __LINE__, m2DLayer->typeMembershipFun[i][j]);
                    exit(-1);
                    break;
                }
                MY_FLT_TYPE a1 = 0, a2 = 0;
                switch(m2DLayer->typeFuzzySet) {
                case FUZZY_SET_I:
                    m2DLayer->degreeMembership[i][j][0] = cur_out;
                    break;
                case FUZZY_INTERVAL_TYPE_II:
                    a1 = m2DLayer->para_MF_II_ratios[i][j][0];
                    a2 = m2DLayer->para_MF_II_ratios[i][j][1];
                    m2DLayer->degreeMembership[i][j][1] = pow((1 - pow((1 - cur_out), a1)), (1 / a1));
                    m2DLayer->degreeMembership[i][j][0] = pow((1 - pow((1 - cur_out), a2)), (1 / a2));
                    break;
                default:
                    printf("%s(%d): Unknown FUZZY_SET_TYPE %d, exit...\n", __FILE__, __LINE__, m2DLayer->typeFuzzySet);
                    exit(-1);
                    break;
                }
            } else {
                printf("%s(%d): For 2-D MFs, the input should be continuous, exiting...\n",
                       __FILE__, __LINE__);
                exit(-1);
            }
            m2DLayer->dataflowStatus[i][j] = dataflowCount;
            if(dataflowCount == 0) {
                m2DLayer->degreeMembership[i][j][0] = 0;
                if(m2DLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                    m2DLayer->degreeMembership[i][j][1] = 0;
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("\nmLayer->degreeMembership[i][j]\n");
    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            printf("(%e  ", m2DLayer->degreeMembership[i][j][0]);
            if(m2DLayer->typeFuzzySet == FUZZY_SET_I)
                printf(") ");
            else
                printf("%e) ", m2DLayer->degreeMembership[i][j][1]);
        }
        printf("\n");
    }
#endif

    return;
}

void print_data_member2DLayer(Member2DLayer* m2DLayer)
{
    printf("\nMembership 2D layer:\n");
    printf("\nmLayer->degreeMembership[i][j]\n");
    for(int i = 0; i < m2DLayer->numInput; i++) {
        for(int j = 0; j < m2DLayer->numMembershipFun[i]; j++) {
            printf("(%e  ", m2DLayer->degreeMembership[i][j][0]);
            if(m2DLayer->typeFuzzySet == FUZZY_SET_I)
                printf(") ");
            else
                printf("%e) ", m2DLayer->degreeMembership[i][j][1]);
        }
        printf("\n");
    }
    printf("print data for Membership 2D layer done\n----------------------------------------\n\n");
}

void ff_fuzzyLayer(FuzzyLayer* fLayer, MY_FLT_TYPE*** degreesMemb, MY_FLT_TYPE** dataflowStatus)
{
    if(fLayer->tag_GEP_rule == FLAG_STATUS_OFF) {
        for(int i = 0; i < fLayer->numRules; i++) {
            MY_FLT_TYPE temp1 = 1;
            MY_FLT_TYPE temp2 = 1;
            int count = 0;
            MY_FLT_TYPE dataflowCount = 0;
            for(int j = 0; j < fLayer->numInput; j++) {
                for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                    if(fLayer->connectStatusAll[i][j][k]) {
                        switch(fLayer->typeRules) {
                        case PRODUCT_INFERENCE_ENGINE:
                            temp1 *= degreesMemb[j][k][0];
                            if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                                temp2 *= degreesMemb[j][k][1];
                            break;
                        case MINIMUM_INFERENCE_ENGINE:
                            if(temp1 > degreesMemb[j][k][0]) temp1 = degreesMemb[j][k][0];
                            if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                                if(temp2 > degreesMemb[j][k][1]) temp2 = degreesMemb[j][k][1];
                            break;
                        default:
                            printf("%s(%d): Unknown FUZZY_RULE_TYPE %d, exit...\n", __FILE__, __LINE__, fLayer->typeRules);
                            exit(-1);
                            break;
                        }
                        count++;
                        dataflowCount += dataflowStatus[j][k];
                    }
                }
            }
            if(count) {
                fLayer->degreeRules[i][0] = temp1;
                if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                    fLayer->degreeRules[i][1] = temp2;
            } else {
                fLayer->degreeRules[i][0] = 0;
                if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                    fLayer->degreeRules[i][1] = 0;
            }
            fLayer->dataflowStatus[i] = dataflowCount;
        }
    } else {
        if(fLayer->typeInRuleCorNum == ONE_EACH_IN_TO_ONE_RULE) {
            for(int i = 0; i < fLayer->numRules; i++) {
                int count = 0;
                MY_FLT_TYPE dataflowCount = 0;
                for(int j = 0; j < fLayer->numInput; j++) {
                    for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                        if(fLayer->connectStatusAll[i][j][k]) {
                            for(int d = 0; d < fLayer->dim_degree; d++)
                                fLayer->degreeMembs[count * fLayer->dim_degree + d] = degreesMemb[j][k][d];
                            count++;
                            dataflowCount += dataflowStatus[j][k];
                        }
                    }
                }
                if(count != fLayer->numInput) {
                    printf("%s(%d): The number of selected mem degrees (%d) should be (%d), exiting...\n",
                           __FILE__, __LINE__, count, fLayer->numInput);
                    exit(-1000);
                }
                MY_FLT_TYPE tempV[2];
                decodingGEP(fLayer->ruleGEP[i], fLayer->degreeMembs, tempV);
                if(tempV[0] > tempV[1]) {
                    MY_FLT_TYPE temp = tempV[0];
                    tempV[0] = tempV[1];
                    tempV[1] = temp;
                }
                if(count) {
                    fLayer->degreeRules[i][0] = tempV[0];
                    if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                        fLayer->degreeRules[i][1] = tempV[1];
                } else {
                    fLayer->degreeRules[i][0] = 0;
                    if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                        fLayer->degreeRules[i][1] = 0;
                }
                fLayer->dataflowStatus[i] = dataflowCount;
            }
        } else {
            int count = 0;
            for(int j = 0; j < fLayer->numInput; j++) {
                for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                    for(int d = 0; d < fLayer->dim_degree; d++)
                        fLayer->degreeMembs[count * fLayer->dim_degree + d] = degreesMemb[j][k][d];
                    count++;
                }
            }
            for(int i = 0; i < fLayer->numRules; i++) {
                count = 0;
                MY_FLT_TYPE dataflowCount = 0;
                for(int j = 0; j < fLayer->numInput; j++) {
                    for(int k = 0; k < fLayer->numMembershipFun[j]; k++) {
                        if(fLayer->connectStatusAll[i][j][k]) {
                            count++;
                            dataflowCount += dataflowStatus[j][k];
                        }
                    }
                }
                MY_FLT_TYPE tempV[2];
                decodingGEP(fLayer->ruleGEP[i], fLayer->degreeMembs, tempV);
                if(tempV[0] > tempV[1]) {
                    MY_FLT_TYPE temp = tempV[0];
                    tempV[0] = tempV[1];
                    tempV[1] = temp;
                }
                if(count) {
                    fLayer->degreeRules[i][0] = tempV[0];
                    if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                        fLayer->degreeRules[i][1] = tempV[1];
                } else {
                    fLayer->degreeRules[i][0] = 0;
                    if(fLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                        fLayer->degreeRules[i][1] = 0;
                }
                fLayer->dataflowStatus[i] = dataflowCount;
            }
        }
    }
    //
#ifdef MY_DEBUG_TAG
    printf("degreeRules[i][0]\n");
    for(int i = 0; i < fLayer->numRules; i++) {
        printf("(%e  ", fLayer->degreeRules[i][0]);
        if(fLayer->typeFuzzySet == FUZZY_SET_I)
            printf(") ");
        else
            printf("%e) ", fLayer->degreeRules[i][1]);
    }
    printf("\n");
#endif

    return;
}

void print_data_fuzzyLayer(FuzzyLayer* fLayer)
{
    printf("\nFuzzy rule layer:\n");
    printf("degreeRules[i][0]\n");
    for(int i = 0; i < fLayer->numRules; i++) {
        printf("(%e  ", fLayer->degreeRules[i][0]);
        if(fLayer->typeFuzzySet == FUZZY_SET_I)
            printf(") ");
        else
            printf("%e) ", fLayer->degreeRules[i][1]);
    }
    printf("print data for Fuzzy rule layer done\n----------------------------------------\n\n");
}

void ff_roughLayer(RoughLayer* rLayer, MY_FLT_TYPE** degreesInput, MY_FLT_TYPE* dataflowStatus)
{
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        MY_FLT_TYPE temp1 = 0;
        MY_FLT_TYPE temp2 = 0;
        rLayer->dataflowStatus[i] = 0;
        for(int j = 0; j < rLayer->numInput; j++) {
            if(rLayer->connectStatus[i][j]) {
                temp1 += degreesInput[j][0] * rLayer->connectWeight[i][j];
                if(rLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                    temp2 += degreesInput[j][1] * rLayer->connectWeight[i][j];
                rLayer->dataflowStatus[i] += dataflowStatus[j];
            }
        }
        rLayer->degreeRough[i][0] = temp1;
        if(rLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
            rLayer->degreeRough[i][1] = temp2;
    }

#ifdef MY_DEBUG_TAG
    printf("rLayer->degreeRough[i][0]\n");
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        printf("(%e ", rLayer->degreeRough[i][0]);
        if(rLayer->typeFuzzySet == FUZZY_SET_I)
            printf(") ");
        else
            printf("%e) ", rLayer->degreeRough[i][1]);
    }
    printf("\n");
#endif

    return;
}

void print_data_roughLayer(RoughLayer* rLayer)
{
    printf("\nRough layer:\n");
    printf("rLayer->degreeRough[i][0]\n");
    for(int i = 0; i < rLayer->numRoughSets; i++) {
        printf("(%e ", rLayer->degreeRough[i][0]);
        if(rLayer->typeFuzzySet == FUZZY_SET_I)
            printf(") ");
        else
            printf("%e) ", rLayer->degreeRough[i][1]);
    }
    printf("\n");
    printf("print data for Rough layer done\n----------------------------------------\n\n");
}

void ff_outReduceLayer(OutReduceLayer* oLayer, MY_FLT_TYPE** degreesInput, MY_FLT_TYPE* dataflowStatus)
{
    if(oLayer->consequenceNodeStatus == ADAPTIVE_CONSEQUENCE_CENTROID) {
        for(int i = 0; i < oLayer->numOutput; i++) {
            if(oLayer->centroid_num_tag == CENTROID_ALL_ONESET && i) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    for(int k = 0; k < oLayer->dim_degree; k++) {
                        oLayer->centroidsRough[i][j][k] = oLayer->centroidsRough[0][j][k];
                    }
                }
                continue;
            }
            for(int j = 0; j < oLayer->numInput; j++) {
                MY_FLT_TYPE temp1 = 0;
                MY_FLT_TYPE temp2 = 0;
                for(int k = 0; k < oLayer->numInputConsequenceNode; k++) {
                    MY_FLT_TYPE cur_val = (oLayer->inputConsequenceNode[i][k] - oLayer->inputMin_cnsq[k]) /
                                          (oLayer->inputMax_cnsq[k] - oLayer->inputMin_cnsq[k]);
                    temp1 += oLayer->paraConsequenceNode[i][j][0][k] * cur_val;
                    if(oLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
                        if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                            temp2 = temp1;
                        else if(oLayer->typeTypeReducer == CENTER_OF_SETS_TYPE_REDUCER)
                            temp2 += oLayer->paraConsequenceNode[i][j][1][k] * cur_val;
                    }
                }
                int k = oLayer->numInputConsequenceNode;
                temp1 += oLayer->paraConsequenceNode[i][j][0][k];
                if(oLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
                    if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER)
                        temp2 = temp1;
                    else if(oLayer->typeTypeReducer == CENTER_OF_SETS_TYPE_REDUCER)
                        temp2 += oLayer->paraConsequenceNode[i][j][1][k];
                }
                oLayer->centroidsRough[i][j][0] = temp1;
                if(oLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II)
                    oLayer->centroidsRough[i][j][1] = temp2;
                if(oLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II &&
                   oLayer->typeTypeReducer == CENTER_OF_SETS_TYPE_REDUCER) {
                    if(temp1 > temp2) {
                        oLayer->centroidsRough[i][j][0] = temp2;
                        oLayer->centroidsRough[i][j][1] = temp1;
                    }
                }
            }
        }
    }
    if(oLayer->typeFuzzySet == FUZZY_SET_I) {
        for(int i = 0; i < oLayer->numOutput; i++) {
            oLayer->dataflowStatus[i] = 0;
            MY_FLT_TYPE tempSum = 0;
            for(int j = 0; j < oLayer->numInput; j++) {
                oLayer->valInputFinal[i][j] = 0;
                if(oLayer->connectStatus[i][j]) {
                    tempSum += degreesInput[j][0];
                }
            }
            if(tempSum) {
                for(int j = 0; j < oLayer->numInput; j++) {
                    if(oLayer->connectStatus[i][j]) {
                        oLayer->valInputFinal[i][j] = degreesInput[j][0] / tempSum *
                                                      oLayer->centroidsRough[i][j][0];
                    }
                }
            }
            MY_FLT_TYPE tempVal = 0;
            for(int j = 0; j < oLayer->numInput; j++) {
                if(oLayer->connectStatus[i][j]) {
                    tempVal += oLayer->valInputFinal[i][j] * oLayer->connectWeight[i][j];
                    oLayer->dataflowStatus[i] += dataflowStatus[j];
                }
            }
            oLayer->valOutputFinal[i] = tempVal;
        }
    } else if(oLayer->typeFuzzySet == FUZZY_INTERVAL_TYPE_II) {
        if(oLayer->consequenceNodeStatus == NO_CONSEQUENCE_CENTROID) {
            for(int i = 0; i < oLayer->numOutput; i++) {
                oLayer->dataflowStatus[i] = 0;
                MY_FLT_TYPE tempSum = 0;
                for(int j = 0; j < oLayer->numInput; j++) {
                    oLayer->valInputFinal[i][j] = 0;
                    tempSum += (degreesInput[j][0] + degreesInput[j][1]);
                }
                if(tempSum > 0) {
                    for(int j = 0; j < oLayer->numInput; j++) {
                        oLayer->valInputFinal[i][j] =
                            (degreesInput[j][0] + degreesInput[j][1]) / tempSum *
                            oLayer->centroidsRough[i][j][0];
                    }
                }
                MY_FLT_TYPE tempVal = 0;
                for(int j = 0; j < oLayer->numInput; j++) {
                    if(oLayer->connectStatus[i][j]) {
                        tempVal += oLayer->valInputFinal[i][j] * oLayer->connectWeight[i][j];
                        oLayer->dataflowStatus[i] += dataflowStatus[j];
                    }
                }
                oLayer->valOutputFinal[i] = tempVal;
            }
        } else {
            for(int i = 0; i < oLayer->numOutput; i++) {
                oLayer->dataflowStatus[i] = 0;
                if(oLayer->typeTypeReducer == NIE_TAN_TYPE_REDUCER) {
                    MY_FLT_TYPE tempSum = 0;
                    for(int j = 0; j < oLayer->numInput; j++) {
                        oLayer->valInputFinal[i][j] = 0;
                        tempSum += (degreesInput[j][0] + degreesInput[j][1]);
                    }
                    if(tempSum > 0) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            oLayer->valInputFinal[i][j] =
                                (degreesInput[j][0] + degreesInput[j][1]) / tempSum *
                                oLayer->centroidsRough[i][j][0];
                        }
                    }
                    MY_FLT_TYPE tempVal = 0;
                    for(int j = 0; j < oLayer->numInput; j++) {
                        if(oLayer->connectStatus[i][j]) {
                            tempVal += oLayer->valInputFinal[i][j] * oLayer->connectWeight[i][j];
                            oLayer->dataflowStatus[i] += dataflowStatus[j];
                        }
                    }
                    oLayer->valOutputFinal[i] = tempVal;
                } else if(oLayer->typeTypeReducer == CENTER_OF_SETS_TYPE_REDUCER) {
                    MY_FLT_TYPE tempSum = 0;
                    for(int j = 0; j < oLayer->numInput; j++) {
                        oLayer->valInputFinal[i][j] = 0;
                        tempSum += (degreesInput[j][0] + degreesInput[j][1]);
                    }
                    if(tempSum > 0) {
                        for(int j = 0; j < oLayer->numInput; j++) {
                            oLayer->valInputFinal[i][j] =
                                (degreesInput[j][0] + degreesInput[j][1]) / tempSum *
                                oLayer->centroidsRough[i][j][0];
                        }
                    }
                    MY_FLT_TYPE tempVal = 0;
                    int count = 0;
                    for(int j = 0; j < oLayer->numInput; j++)
                        if(oLayer->connectStatus[i][j]) {
                            count++;
                        }
                    if(count) {
                        tempVal = EIASC_IT2Reduce(oLayer, i, degreesInput);
                    }
                    oLayer->valOutputFinal[i] = tempVal;
                }
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("oLayer->centroidsRough[][][]\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        for(int j = 0; j < oLayer->numInput; j++) {
            printf("(%e ", oLayer->centroidsRough[i][j][0]);
            if(oLayer->typeFuzzySet == FUZZY_SET_I)
                printf(") ");
            else
                printf("%e) ", oLayer->centroidsRough[i][j][1]);
        }
        printf("\n");
    }
    printf("\n");
    printf("oLayer->valOutputFinal[i]\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        printf("%e ", oLayer->valOutputFinal[i]);
    }
    printf("\n");
    debug_count_FRNN++;
#endif

    return;
}

void print_data_outReduceLayer(OutReduceLayer* oLayer)
{
    printf("\nOutput layer:\n");
    printf("oLayer->centroidsRough[][][]\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        for(int j = 0; j < oLayer->numInput; j++) {
            printf("(%e ", oLayer->centroidsRough[i][j][0]);
            if(oLayer->typeFuzzySet == FUZZY_SET_I)
                printf(") ");
            else
                printf("%e) ", oLayer->centroidsRough[i][j][1]);
        }
        printf("\n");
    }
    printf("\n");
    printf("oLayer->valOutputFinal[i]\n");
    for(int i = 0; i < oLayer->numOutput; i++) {
        printf("%e ", oLayer->valOutputFinal[i]);
    }
    printf("print data for Output layer done\n----------------------------------------\n\n");
}

static MY_FLT_TYPE KM_IT2Reduce(OutReduceLayer* oLayer, int i, MY_FLT_TYPE** degreesInput)
{
    MY_FLT_TYPE tempVal = 0;
    int count = 0;
    int sum_l_zero_count = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            count++;
            if(degreesInput[j][0] < FLT_MIN) sum_l_zero_count++;
        }
    }
    if(sum_l_zero_count == count) {
        MY_FLT_TYPE tempNum = 0;
        MY_FLT_TYPE tempDen = 0;
        for(int j = 0; j < oLayer->numInput; j++) {
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
                tempNum += oLayer->centroidsRough[i][j][1] * degreesInput[j][1];
                tempDen += degreesInput[j][1];
            }
        }
        if(count && tempDen > FLT_MIN)
            tempVal = tempNum / tempDen;
        else
            tempVal = 0;
        return tempVal;
    }
    MY_FLT_TYPE tempL, tempR;
    MY_FLT_TYPE tempCur1, tempCur2;
    MY_FLT_TYPE tempNum = 0;
    MY_FLT_TYPE tempDen = 0;
    int* sortedInd = (int*)malloc(oLayer->numInput * sizeof(int));
    // y_l
    tempNum = 0;
    tempDen = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            tempNum += oLayer->centroidsRough[i][j][0] *
                       (degreesInput[j][0] + degreesInput[j][1]) * oLayer->connectWeight[i][j] / 2;
            tempDen += (degreesInput[j][0] + degreesInput[j][1]) * oLayer->connectWeight[i][j] / 2;
#ifdef MY_DEBUG_TAG2
            printf("%d[%e,%e](%e,%e) \n",
                   j, oLayer->centroidsRough[i][j][0], oLayer->centroidsRough[i][j][1], degreesInput[j][0], degreesInput[j][1]);
#endif
        }
    }
    if(tempDen > FLT_MIN)
        tempCur1 = tempCur2 = tempNum / tempDen;
    else
        tempCur1 = tempCur2 = 0;
    if(count > 1) {
        count = 0;
        for(int j = 0; j < oLayer->numInput; j++) {
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
                sortedInd[count++] = j;
            }
        }
        for(int j = 0; j < count; j++) {
            for(int k = 1; k < count - j; k++) {
                if(oLayer->centroidsRough[i][sortedInd[k - 1]][0] >
                   oLayer->centroidsRough[i][sortedInd[k]][0]) {
                    int tempInt = sortedInd[k - 1];
                    sortedInd[k - 1] = sortedInd[k];
                    sortedInd[k] = tempInt;
                }
            }
        }
#ifdef MY_DEBUG_TAG2
        for(int j = 0; j < count; j++) printf("%d-%e ", sortedInd[j], oLayer->centroidsRough[i][sortedInd[j]][0]);
        printf("\n");
#endif
        do {
            tempCur1 = tempCur2;
            int tempFlag = -1;
            for(int j = 0; j < count - 1; j++) {
                int k = j + 1;
                if(oLayer->centroidsRough[i][sortedInd[j]][0] <= tempCur1 &&
                   oLayer->centroidsRough[i][sortedInd[k]][0] > tempCur1) {
                    tempFlag = j;
#ifdef MY_DEBUG_TAG2
                    printf("[%d %e,%e,%e]\n", tempFlag,
                           oLayer->centroidsRough[i][sortedInd[j]][0], tempCur1,
                           oLayer->centroidsRough[i][sortedInd[k]][0]);
#endif
                }
                if(tempFlag >= 0) break;
            }
            if(tempFlag == -1)
                break;
            tempNum = 0;
            tempDen = 0;
            for(int j = 0; j < count; j++) {
                if(j <= tempFlag) {
                    tempNum += oLayer->centroidsRough[i][sortedInd[j]][0] *
                               degreesInput[sortedInd[j]][1] * oLayer->connectWeight[i][sortedInd[j]];
                    tempDen += degreesInput[sortedInd[j]][1] * oLayer->connectWeight[i][sortedInd[j]];
                } else {
                    tempNum += oLayer->centroidsRough[i][sortedInd[j]][0] *
                               degreesInput[sortedInd[j]][0] * oLayer->connectWeight[i][sortedInd[j]];
                    tempDen += degreesInput[sortedInd[j]][0] * oLayer->connectWeight[i][sortedInd[j]];
                }
            }
            if(tempDen > FLT_MIN)
                tempCur2 = tempNum / tempDen;
            else
                tempCur2 = 0;
#ifdef MY_DEBUG_TAG2
            printf("%.16e\n", tempCur1 - tempCur2);
#endif
        } while(fabs(tempCur1 - tempCur2) >= FLT_EPSILON);
    }
    tempL = tempCur1;
    // y_r
    tempNum = 0;
    tempDen = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            tempNum += oLayer->centroidsRough[i][j][1] *
                       (degreesInput[j][0] + degreesInput[j][1]) * oLayer->connectWeight[i][j] / 2;
            tempDen += (degreesInput[j][0] + degreesInput[j][1]) * oLayer->connectWeight[i][j] / 2;
#ifdef MY_DEBUG_TAG2
            printf("%d[%e,%e](%e,%e) ",
                   j, oLayer->centroidsRough[i][j][0], oLayer->centroidsRough[i][j][1], degreesInput[j][0], degreesInput[j][1]);
#endif
        }
    }
#ifdef MY_DEBUG_TAG2
    printf("\n");
#endif
    if(tempDen > FLT_MIN)
        tempCur1 = tempCur2 = tempNum / tempDen;
    else
        tempCur1 = tempCur2 = 0;
    if(count > 1) {
        count = 0;
        for(int j = 0; j < oLayer->numInput; j++)
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN)
                sortedInd[count++] = j;
        for(int j = 0; j < count; j++) {
            for(int k = 1; k < count - j; k++) {
                if(oLayer->centroidsRough[i][sortedInd[k - 1]][1] >
                   oLayer->centroidsRough[i][sortedInd[k]][1]) {
                    int tempInt = sortedInd[k - 1];
                    sortedInd[k - 1] = sortedInd[k];
                    sortedInd[k] = tempInt;
                }
            }
        }
#ifdef MY_DEBUG_TAG2
        for(int j = 0; j < count; j++) printf("%e ", oLayer->centroidsRough[i][sortedInd[j]][1]);
        printf("\n");
#endif
        do {
            tempCur1 = tempCur2;
            int tempFlag = -1;
            for(int j = 0; j < count - 1; j++) {
                int k = j + 1;
                if(oLayer->centroidsRough[i][sortedInd[j]][1] < tempCur1 &&
                   oLayer->centroidsRough[i][sortedInd[k]][1] >= tempCur1) {
                    tempFlag = j;
#ifdef MY_DEBUG_TAG2
                    printf("[%d %e,%e,%e]\n", tempFlag,
                           oLayer->centroidsRough[i][sortedInd[j]][1], tempCur1,
                           oLayer->centroidsRough[i][sortedInd[k]][1]);
#endif
                }
                if(tempFlag >= 0) break;
            }
            if(tempFlag == -1)
                break;
            tempNum = 0;
            tempDen = 0;
            for(int j = 0; j < count; j++) {
                if(j <= tempFlag) {
                    tempNum += oLayer->centroidsRough[i][sortedInd[j]][1] *
                               degreesInput[sortedInd[j]][0] * oLayer->connectWeight[i][sortedInd[j]];
                    tempDen += degreesInput[sortedInd[j]][0] * oLayer->connectWeight[i][sortedInd[j]];
                } else {
                    tempNum += oLayer->centroidsRough[i][sortedInd[j]][1] *
                               degreesInput[sortedInd[j]][1] * oLayer->connectWeight[i][sortedInd[j]];
                    tempDen += degreesInput[sortedInd[j]][1] * oLayer->connectWeight[i][sortedInd[j]];
                }
            }
            if(tempDen > FLT_MIN)
                tempCur2 = tempNum / tempDen;
            else
                tempCur2 = 0;
        } while(fabs(tempCur1 - tempCur2) >= FLT_EPSILON);
    }
    tempR = tempCur1;
    //
    tempVal = (tempL + tempR) / 2;
    free(sortedInd);
    //
    return tempVal;
}

static MY_FLT_TYPE EIASC_IT2Reduce(OutReduceLayer* oLayer, int i, MY_FLT_TYPE** degreesInput)
{
    MY_FLT_TYPE tempVal = 0;
    int count = 0;
    int sum_l_zero_count = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            count++;
            if(degreesInput[j][0] < FLT_MIN) sum_l_zero_count++;
        }
    }
    if(sum_l_zero_count == count) {
        MY_FLT_TYPE tempNum = 0;
        MY_FLT_TYPE tempDen = 0;
        for(int j = 0; j < oLayer->numInput; j++) {
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
                tempNum += oLayer->centroidsRough[i][j][1] * degreesInput[j][1];
                tempDen += degreesInput[j][1];
            }
        }
        if(count && tempDen > FLT_MIN)
            tempVal = tempNum / tempDen;
        else
            tempVal = 0;
        return tempVal;
    }
    MY_FLT_TYPE tempL, tempR;
    MY_FLT_TYPE tempCur1;
    MY_FLT_TYPE tempNum = 0;
    MY_FLT_TYPE tempDen = 0;
    int* sortedInd = (int*)malloc(oLayer->numInput * sizeof(int));
    // y_l
    tempNum = 0;
    tempDen = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            tempNum += oLayer->centroidsRough[i][j][0] * degreesInput[j][0];
            tempDen += degreesInput[j][0];
#ifdef MY_DEBUG_TAG2
            printf("%d[%e,%e](%e,%e) \n",
                   j, oLayer->centroidsRough[i][j][0], oLayer->centroidsRough[i][j][1], degreesInput[j][0], degreesInput[j][1]);
#endif
        }
    }
    if(tempDen > FLT_MIN)
        tempCur1 = tempNum / tempDen;
    else
        tempCur1 = 0;
    if(count > 1) {
        count = 0;
        for(int j = 0; j < oLayer->numInput; j++) {
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
                sortedInd[count++] = j;
            }
        }
        for(int j = 0; j < count; j++) {
            for(int k = 1; k < count - j; k++) {
                if(oLayer->centroidsRough[i][sortedInd[k - 1]][0] >
                   oLayer->centroidsRough[i][sortedInd[k]][0]) {
                    int tempInt = sortedInd[k - 1];
                    sortedInd[k - 1] = sortedInd[k];
                    sortedInd[k] = tempInt;
                }
            }
        }
#ifdef MY_DEBUG_TAG2
        for(int j = 0; j < count; j++) printf("%d-%e ", sortedInd[j], oLayer->centroidsRough[i][sortedInd[j]][0]);
        printf("\n");
#endif
        int tempIND = -1;
        while(1) {
            tempIND++;
            int realIND = sortedInd[tempIND];
            tempNum += oLayer->centroidsRough[i][realIND][0] * (degreesInput[realIND][1] - degreesInput[realIND][0]);
            tempDen += degreesInput[realIND][1] - degreesInput[realIND][0];
            if(tempDen > FLT_MIN)
                tempCur1 = tempNum / tempDen;
            else
                tempCur1 = 0;
            if(tempIND >= count - 1) {
                printf("%s(%d): EIASC_IT2Reduce ERROR\n", __FILE__, __LINE__);
                break;
            }
            if(tempCur1 <= oLayer->centroidsRough[i][sortedInd[tempIND + 1]][0]) break;
        }
    }
    tempL = tempCur1;
    // y_r
    tempNum = 0;
    tempDen = 0;
    for(int j = 0; j < oLayer->numInput; j++) {
        if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
            tempNum += oLayer->centroidsRough[i][j][1] * degreesInput[j][0];
            tempDen += degreesInput[j][0];
#ifdef MY_DEBUG_TAG2
            printf("%d[%e,%e](%e,%e) \n",
                   j, oLayer->centroidsRough[i][j][0], oLayer->centroidsRough[i][j][1], degreesInput[j][0], degreesInput[j][1]);
#endif
        }
    }
    if(tempDen > FLT_MIN)
        tempCur1 = tempNum / tempDen;
    else
        tempCur1 = 0;
    if(count > 1) {
        count = 0;
        for(int j = 0; j < oLayer->numInput; j++) {
            if(oLayer->connectStatus[i][j] && degreesInput[j][1] > FLT_MIN) {
                sortedInd[count++] = j;
            }
        }
        for(int j = 0; j < count; j++) {
            for(int k = 1; k < count - j; k++) {
                if(oLayer->centroidsRough[i][sortedInd[k - 1]][1] >
                   oLayer->centroidsRough[i][sortedInd[k]][1]) {
                    int tempInt = sortedInd[k - 1];
                    sortedInd[k - 1] = sortedInd[k];
                    sortedInd[k] = tempInt;
                }
            }
        }
#ifdef MY_DEBUG_TAG2
        for(int j = 0; j < count; j++) printf("%d-%e ", sortedInd[j], oLayer->centroidsRough[i][sortedInd[j]][1]);
        printf("\n");
#endif
        int tempIND = count - 1;
        while(1) {
            int realIND = sortedInd[tempIND];
            tempNum += oLayer->centroidsRough[i][realIND][1] * (degreesInput[realIND][1] - degreesInput[realIND][0]);
            tempDen += degreesInput[realIND][1] - degreesInput[realIND][0];
            if(tempDen > FLT_MIN)
                tempCur1 = tempNum / tempDen;
            else
                tempCur1 = 0;
            tempIND--;
            if(tempIND < 0) {
                printf("%s(%d): EIASC_IT2Reduce ERROR\n", __FILE__, __LINE__);
                break;
            }
            if(tempCur1 >= oLayer->centroidsRough[i][sortedInd[tempIND]][1]) break;
        }
    }
    tempR = tempCur1;
    //
    tempVal = (tempL + tempR) / 2;
    free(sortedInd);
    //
    return tempVal;
}

void bp_derivative_convLayer(ConvolutionLayer* cLayer)
{
    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            memset(cLayer->featureMapDerivative[i][h], 0, cLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                if(!cLayer->featureMapTag[i][h][w]) continue;
                MY_FLT_TYPE temp_in = cLayer->featureMapData[i][h][w];
                MY_FLT_TYPE temp_out;
                switch(cLayer->kernelType[i]) {
                case ACT_FUNC_SIGMA:
                    temp_out = temp_in * (1 - temp_in);
                    break;
                case ACT_FUNC_RELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.0;
                    break;
                case ACT_FUNC_TANH:
                    temp_out = 1 - temp_in * temp_in;
                    break;
                case ACT_FUNC_LEAKYRELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.01;
                    break;
                case ACT_FUNC_ELU:
                    temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)(0.01 + temp_in);
                    break;
                default:
                    printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                           __FILE__, __LINE__, cLayer->kernelType[i]);
                    exit(-1);
                    break;
                }
                cLayer->featureMapDerivative[i][h][w] = temp_out;
                if(CHECK_INVALID(temp_out) || fabs(temp_out) > 1e10) {
                    int test = 0;
                }
            }
        }
    }

    return;
}

void bp_derivative_poolLayer(PoolLayer* pLayer)
{
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            memset(pLayer->featureMapDerivative[i][h], 0, pLayer->featureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        if(pLayer->poolFlag[i] == 0) continue;
        for(int h = 0; h < pLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < pLayer->featureMapWidthMax; w++) {
                if(pLayer->featureMapTag[i][h][w] == 0) continue;
                pLayer->featureMapDerivative[i][h][w] = 1;
            }
        }
    }

    return;
}

void bp_derivative_icfcLayer(InterCPCLayer* icfcLayer)
{
    memset(icfcLayer->outputDerivative, 0, icfcLayer->numOutput * sizeof(MY_FLT_TYPE));

    if(icfcLayer->flagActFunc == FLAG_STATUS_ON) {
        for(int i = 0; i < icfcLayer->numOutput; i++) {
            if(!icfcLayer->connectCountAll[i]) continue;
            MY_FLT_TYPE temp_in = icfcLayer->outputData[i];
            MY_FLT_TYPE temp_out;
            switch(icfcLayer->actFuncType[i]) {
            case ACT_FUNC_SIGMA:
                temp_out = temp_in * (1 - temp_in);
                break;
            case ACT_FUNC_RELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.0;
                break;
            case ACT_FUNC_TANH:
                temp_out = 1 - temp_in * temp_in;
                break;
            case ACT_FUNC_LEAKYRELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.01;
                break;
            case ACT_FUNC_ELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)(0.01 + temp_in);
                break;
            default:
                printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                       __FILE__, __LINE__, icfcLayer->actFuncType[i]);
                exit(-1);
                break;
            }
            icfcLayer->outputDerivative[i] = temp_out;
            if(CHECK_INVALID(temp_out) || fabs(temp_out) > 1e10) {
                int test = 0;
            }
        }
    } else {
        for(int i = 0; i < icfcLayer->numOutput; i++) {
            if(!icfcLayer->connectCountAll[i]) continue;
            icfcLayer->outputDerivative[i] = 1;
        }
    }

    return;
}

void bp_derivative_fcLayer(FCLayer* fcLayer)
{
    memset(fcLayer->outputDerivative, 0, fcLayer->numOutput * sizeof(MY_FLT_TYPE));

    if(fcLayer->flagActFunc) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            MY_FLT_TYPE temp_in = fcLayer->outputData[i];
            MY_FLT_TYPE temp_out;
            switch(fcLayer->actFuncType[i]) {
            case ACT_FUNC_SIGMA:
                temp_out = temp_in * (1 - temp_in);
                break;
            case ACT_FUNC_RELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.0;
                break;
            case ACT_FUNC_TANH:
                temp_out = 1 - temp_in * temp_in;
                break;
            case ACT_FUNC_LEAKYRELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.01;
                break;
            case ACT_FUNC_ELU:
                temp_out = temp_in > 0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)(0.01 + temp_in);
                break;
            default:
                printf("%s(%d): Unknown CONV_ACT_FUNC_TYPE %d, exit...\n",
                       __FILE__, __LINE__, fcLayer->actFuncType[i]);
                exit(-1);
                break;
            }
            fcLayer->outputDerivative[i] = temp_out;
            if(CHECK_INVALID(temp_out) || fabs(temp_out) > 1e10) {
                int test = 0;
            }
        }
    } else {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            fcLayer->outputDerivative[i] = 1;
        }
    }

    return;
}

void bp_delta_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer)
{
    for(int i = 0; i < cLayer->channelsInMax; i++) {
        for(int h = 0; h < cLayer->inputHeightMax; h++) {
            memset(deltaPriorLayer[i][h], 0, cLayer->inputWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_SKIP) continue;
            int heightInCur = cLayer->inputHeight[j];
            int widthInCur = cLayer->inputWidth[j];
            int kernelHeightCur = cLayer->kernelHeight[i][j];
            int kernelWidthCur = cLayer->kernelWidth[i][j];
            int heightInOffset = (cLayer->inputHeightMax - heightInCur) / 2;
            int widthInOffset = (cLayer->inputWidthMax - widthInCur) / 2;
            int kernelHeightOffset = (cLayer->kernelHeightMax - kernelHeightCur) / 2;
            int kernelWidthOffset = (cLayer->kernelWidthMax - kernelWidthCur) / 2;
            int heightOutCur = heightInCur;
            int widthOutCur = widthInCur;
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_COPY) {
                if(heightOutCur <= 0 || widthOutCur <= 0) continue;
                //if(heightOutCur > cLayer->featureMapHeight[i]) cLayer->featureMapHeight[i] = heightOutCur;
                //if(widthOutCur > cLayer->featureMapWidth[i]) cLayer->featureMapWidth[i] = widthOutCur;
                int heightOutOffset = (cLayer->featureMapHeightMax - heightOutCur) / 2;
                int widthOutOffset = (cLayer->featureMapWidthMax - widthOutCur) / 2;
                int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
                int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
                for(int h = 0; h < cLayer->inputHeightMax; h++) {
                    for(int w = 0; w < cLayer->inputWidthMax; w++) {
                        if(h < heightInOffset ||
                           h >= heightInOffset + heightInCur ||
                           w < widthInOffset ||
                           w >= widthInOffset + widthInCur ||
                           tagPriorLayer[j][h][w] == 0)
                            continue;
                        MY_FLT_TYPE tmp_delta = 0;
                        int outHCur = h - heightOutOffsetIn;
                        int outWCur = w - widthOutOffsetIn;
                        if(outHCur < heightOutOffset ||
                           outWCur < widthOutOffset ||
                           outHCur >= heightOutOffset + heightOutCur ||
                           outWCur >= widthOutOffset + widthOutCur ||
                           cLayer->featureMapTag[i][outHCur][outWCur] == 0)
                            continue;
                        tmp_delta += cLayer->featureMapDelta[i][outHCur][outWCur];
                        if(CHECK_INVALID(tmp_delta) || fabs(tmp_delta) > 1e10) {
                            int test = 0;
                        }
                        deltaPriorLayer[j][h][w] += tmp_delta;
                    }
                }
                continue;
            }
            int kernelHeightHalf = kernelHeightCur / 2;
            int kernelWidthHalf = kernelWidthCur / 2;
            if(cLayer->paddingType[i][j] == PADDING_VALID) {
                heightOutCur -= 2 * kernelHeightHalf;
                widthOutCur -= 2 * kernelWidthHalf;
            }
            if(heightOutCur <= 0 || widthOutCur <= 0) continue;
            //if(heightOutCur > cLayer->featureMapHeight[i]) cLayer->featureMapHeight[i] = heightOutCur;
            //if(widthOutCur > cLayer->featureMapWidth[i]) cLayer->featureMapWidth[i] = widthOutCur;
            int heightOutOffset = (cLayer->featureMapHeightMax - heightOutCur) / 2;
            int widthOutOffset = (cLayer->featureMapWidthMax - widthOutCur) / 2;
            int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
            int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
            for(int h = 0; h < cLayer->inputHeightMax; h++) {
                for(int w = 0; w < cLayer->inputWidthMax; w++) {
                    if(h < heightInOffset ||
                       h >= heightInOffset + heightInCur ||
                       w < widthInOffset ||
                       w >= widthInOffset + widthInCur ||
                       tagPriorLayer[j][h][w] == 0)
                        continue;
                    MY_FLT_TYPE tmp_conv = 0;
                    int heightPadding = kernelHeightCur / 2;
                    int widthPadding = kernelWidthCur / 2;
                    for(int a = 0; a < kernelHeightCur; a++) {
                        for(int b = 0; b < kernelWidthCur; b++) {
                            int outHCur = h - heightOutOffsetIn + a - heightPadding;
                            int outWCur = w - widthOutOffsetIn + b - widthPadding;
                            if(outHCur < heightOutOffset ||
                               outWCur < widthOutOffset ||
                               outHCur >= heightOutOffset + heightOutCur ||
                               outWCur >= widthOutOffset + widthOutCur ||
                               cLayer->featureMapTag[i][outHCur][outWCur] == 0)
                                continue;
                            int kHCur = kernelHeightCur - 1 - a + kernelHeightOffset;
                            int kWCur = kernelWidthCur - 1 - b + kernelWidthOffset;
                            tmp_conv += cLayer->featureMapDelta[i][outHCur][outWCur] *
                                        cLayer->kernelData[i][j][kHCur][kWCur];
                        }
                    }
                    if(CHECK_INVALID(tmp_conv) || fabs(tmp_conv) > 1e10) {
                        int test = 0;
                    }
                    deltaPriorLayer[j][h][w] += tmp_conv;
                }
            }
        }
    }

    for(int i = 0; i < cLayer->channelsInMax; i++) {
        for(int h = 0; h < cLayer->inputHeightMax; h++) {
            for(int w = 0; w < cLayer->inputWidthMax; w++) {
                deltaPriorLayer[i][h][w] *= derivativePriorLayer[i][h][w];
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_delta_convLayer - deltaPriorLayer[][][]\n");
    for(int i = 0; i < cLayer->channelsInMax; i++) {
        for(int h = 0; h < cLayer->inputHeightMax; h++) {
            printf("[%d][%d][] ", i, h);
            for(int w = 0; w < cLayer->inputWidthMax; w++) {
                printf("%e\t", deltaPriorLayer[i][h][w]);
            }
            printf("\n");
        }
    }
#endif

    return;
}

void bp_delta_poolLayer(PoolLayer* pLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer)
{
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->inputHeightMax; h++) {
            memset(deltaPriorLayer[i][h], 0, pLayer->inputWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        if(pLayer->poolFlag[i] == 0) continue;
        int heightInCur = pLayer->inputHeight[i];
        int widthInCur = pLayer->inputWidth[i];
        int kernelHeightCur = pLayer->poolHeight[i];
        int kernelWidthCur = pLayer->poolWidth[i];
        int heightInOffset = (pLayer->inputHeightMax - heightInCur) / 2;
        int widthInOffset = (pLayer->inputWidthMax - widthInCur) / 2;
        int kernelHeightOffset = (pLayer->poolHeightMax - kernelHeightCur) / 2;
        int kernelWidthOffset = (pLayer->poolWidthMax - kernelWidthCur) / 2;
        int heightOutCur = (heightInCur + kernelHeightCur - 1) / kernelHeightCur;
        int widthOutCur = (widthInCur + kernelWidthCur - 1) / kernelWidthCur;
        if(heightOutCur > pLayer->featureMapHeight[i]) pLayer->featureMapHeight[i] = heightOutCur;
        if(widthOutCur > pLayer->featureMapWidth[i]) pLayer->featureMapWidth[i] = widthOutCur;
        int heightOutOffset = (pLayer->featureMapHeightMax - heightOutCur) / 2;
        int widthOutOffset = (pLayer->featureMapWidthMax - widthOutCur) / 2;
        for(int h = 0; h < pLayer->inputHeightMax; h++) {
            for(int w = 0; w < pLayer->inputWidthMax; w++) {
                if(h < heightInOffset ||
                   h >= heightInOffset + heightInCur ||
                   w < widthInOffset ||
                   w >= widthInOffset + widthInCur ||
                   tagPriorLayer[i][h][w] == 0)
                    continue;
                int outHCur = heightOutOffset + (h - heightInOffset) / kernelHeightCur;
                int outWCur = widthOutOffset + (w - widthInOffset) / kernelWidthCur;
                if(outHCur < heightOutOffset ||
                   outWCur < widthOutOffset ||
                   outHCur >= heightOutOffset + heightOutCur ||
                   outWCur >= widthOutOffset + widthOutCur ||
                   pLayer->featureMapTag[i][outHCur][outWCur] == 0) {
                    printf("%s(%d): Error - Valid output (%d,%d,%d) corresponds to invalid input (%d,%d,%d), exiting ...\n",
                           __FILE__, __LINE__, i, h, w, i, outHCur, outWCur);
                    exit(-1);
                }
                if(pLayer->poolType[i] == POOL_AVE) {
                    deltaPriorLayer[i][h][w] = pLayer->featureMapDelta[i][outHCur][outWCur] /
                                               pLayer->featureMapTag[i][outHCur][outWCur];
                } else {
                    int curPos = ((h - heightInOffset) % kernelHeightCur) * kernelWidthCur +
                                 (w - widthInOffset) % kernelWidthCur;
                    if(curPos == pLayer->featureMapPos[i][outHCur][outWCur]) {
                        deltaPriorLayer[i][h][w] = pLayer->featureMapDelta[i][outHCur][outWCur];
                    }
                }
                if(CHECK_INVALID(deltaPriorLayer[i][h][w]) || fabs(deltaPriorLayer[i][h][w]) > 1e10) {
                    int test = 0;
                }
            }
        }
    }

    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->inputHeightMax; h++) {
            for(int w = 0; w < pLayer->inputWidthMax; w++) {
                deltaPriorLayer[i][h][w] *= derivativePriorLayer[i][h][w];
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_delta_poolLayer - deltaPriorLayer[][][]\n");
    for(int i = 0; i < pLayer->channelsInOutMax; i++) {
        for(int h = 0; h < pLayer->inputHeightMax; h++) {
            printf("[%d][%d][] ", i, h);
            for(int w = 0; w < pLayer->inputWidthMax; w++) {
                printf("%e\t", deltaPriorLayer[i][h][w]);
            }
            printf("\n");
        }
    }
#endif

    return;
}

void bp_delta_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** deltaPriorLayer, MY_FLT_TYPE*** derivativePriorLayer,
                        int*** tagPriorLayer)
{
    for(int i = 0; i < icfcLayer->preFeatureMapChannels; i++) {
        for(int h = 0; h < icfcLayer->preFeatureMapHeightMax; h++) {
            memset(deltaPriorLayer[i][h], 0, icfcLayer->preFeatureMapWidthMax * sizeof(MY_FLT_TYPE));
        }
    }

    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
            int heightInCur = icfcLayer->preInputHeight[iCh];
            int widthInCur = icfcLayer->preInputWidth[iCh];
            if(heightInCur == 0 || widthInCur == 0) continue;
            int quoHeight = icfcLayer->preFeatureMapHeightMax / heightInCur;
            int remHeight = icfcLayer->preFeatureMapHeightMax % heightInCur;
            int quoWidth = icfcLayer->preFeatureMapWidthMax / widthInCur;
            int remWidth = icfcLayer->preFeatureMapWidthMax % widthInCur;
            int heightInOffset = (icfcLayer->preFeatureMapHeightMax - heightInCur) / 2;
            int widthInOffset = (icfcLayer->preFeatureMapWidthMax - widthInCur) / 2;
            int offsetHeight = 0;
            int offsetWidth = 0;
            for(int iH = 0; iH < icfcLayer->preInputHeight[iCh]; iH++) {
                int kernelHeightCur = quoHeight;
                if(iH < remHeight) kernelHeightCur++;
                offsetWidth = 0;
                for(int iW = 0; iW < icfcLayer->preInputWidth[iCh]; iW++) {
                    int kernelWidthCur = quoWidth;
                    if(iW < remWidth) kernelWidthCur++;
                    int inHcur = heightInOffset + iH;
                    int inWcur = widthInOffset + iW;
                    if(inHcur < heightInOffset ||
                       inWcur < widthInOffset ||
                       inHcur >= heightInOffset + heightInCur ||
                       inWcur >= widthInOffset + widthInCur ||
                       tagPriorLayer[iCh][inHcur][inWcur] == 0)
                        continue;
                    MY_FLT_TYPE tmp_weight = 0;
                    int tmp_count = 0;
                    for(int h = 0; h < kernelHeightCur; h++) {
                        for(int w = 0; w < kernelWidthCur; w++) {
                            int wtHcur = offsetHeight + h;
                            int wtWcur = offsetWidth + w;
                            if(wtHcur < 0 ||
                               wtWcur < 0 ||
                               wtHcur >= icfcLayer->preFeatureMapHeightMax ||
                               wtWcur >= icfcLayer->preFeatureMapWidthMax) {
                                printf("%s(%d): Error out of range, exiting...\n",
                                       __FILE__, __LINE__);
                                exit(-1);
                            }
                            if(icfcLayer->connectStatusAll[iOut][iCh][wtHcur][wtWcur]) {
                                tmp_weight += icfcLayer->connectWeightAll[iOut][iCh][wtHcur][wtWcur];
                                tmp_count++;
                            }
                        }
                    }
                    if(tmp_count) {
                        tmp_weight /= tmp_count;
                        deltaPriorLayer[iCh][inHcur][inWcur] = icfcLayer->outputData[iOut] * tmp_weight;
                        if(CHECK_INVALID(deltaPriorLayer[iCh][inHcur][inWcur]) || fabs(deltaPriorLayer[iCh][inHcur][inWcur]) > 1e10) {
                            int test = 0;
                        }
                    }
                    offsetWidth += kernelWidthCur;
                }
                offsetHeight += kernelHeightCur;
            }
        }
    }

    for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
        for(int iH = 0; iH < icfcLayer->preFeatureMapHeightMax; iH++) {
            for(int iW = 0; iW < icfcLayer->preFeatureMapWidthMax; iW++) {
                deltaPriorLayer[iCh][iH][iW] *= derivativePriorLayer[iCh][iH][iW];
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_delta_icfcLayer - deltaPriorLayer[][][]\n");
    for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
        for(int iH = 0; iH < icfcLayer->preFeatureMapHeightMax; iH++) {
            printf("[%d][%d][] ", iCh, iH);
            for(int iW = 0; iW < icfcLayer->preFeatureMapWidthMax; iW++) {
                printf("%e\t", deltaPriorLayer[iCh][iH][iW]);
            }
            printf("\n");
        }
    }
#endif

    return;
}

void bp_delta_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* deltaPriorLayer, MY_FLT_TYPE* derivativePriorLayer, int* tagPriorLayer)
{
    memset(deltaPriorLayer, 0, fcLayer->numInputMax * sizeof(MY_FLT_TYPE));

    for(int i = 0; i < fcLayer->numInputMax; i++) {
        MY_FLT_TYPE temp = 0;
        for(int j = 0; j < fcLayer->numOutput; j++) {
            if(tagPriorLayer[i] && fcLayer->connectStatus[j][i]) {
                temp += fcLayer->outputDelta[j] * fcLayer->connectWeight[j][i];
            }
        }
        deltaPriorLayer[i] = temp;
        if(CHECK_INVALID(temp) || fabs(temp) > 1e10) {
            int test = 0;
        }
    }

    for(int i = 0; i < fcLayer->numInputMax; i++) {
        deltaPriorLayer[i] *= derivativePriorLayer[i];
    }

#ifdef MY_DEBUG_TAG
    printf("bp_delta_fcLayer - deltaPriorLayer[]\n");
    for(int i = 0; i < fcLayer->numInputMax; i++) {
        printf("%e\t", deltaPriorLayer[i]);
    }
    printf("\n");
#endif

    return;
}

void bp_update_convLayer(ConvolutionLayer* cLayer, MY_FLT_TYPE*** dataInput, int*** tagPriorLayer, FRNNOpts opts)
{
    if(opts.tag_init) {
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                    memset(cLayer->kernelDelta[i][j][h], 0, cLayer->kernelWidthMax * sizeof(MY_FLT_TYPE));
                }
            }
        }
        memset(cLayer->biasDelta, 0, cLayer->channelsOut * sizeof(MY_FLT_TYPE));
    }
    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_SKIP) continue;
            int heightInCur = cLayer->inputHeight[j];
            int widthInCur = cLayer->inputWidth[j];
            int kernelHeightCur = cLayer->kernelHeight[i][j];
            int kernelWidthCur = cLayer->kernelWidth[i][j];
            int heightInOffset = (cLayer->inputHeightMax - heightInCur) / 2;
            int widthInOffset = (cLayer->inputWidthMax - widthInCur) / 2;
            int kernelHeightOffset = (cLayer->kernelHeightMax - kernelHeightCur) / 2;
            int kernelWidthOffset = (cLayer->kernelWidthMax - kernelWidthCur) / 2;
            int heightOutCur = heightInCur;
            int widthOutCur = widthInCur;
            if(cLayer->kernelFlag[i][j] == KERNEL_FLAG_COPY)
                continue;
            int kernelHeightHalf = kernelHeightCur / 2;
            int kernelWidthHalf = kernelWidthCur / 2;
            if(cLayer->paddingType[i][j] == PADDING_VALID) {
                heightOutCur -= 2 * kernelHeightHalf;
                widthOutCur -= 2 * kernelWidthHalf;
            }
            if(heightOutCur <= 0 || widthOutCur <= 0) continue;
            //if(heightOutCur > cLayer->featureMapHeight[i]) cLayer->featureMapHeight[i] = heightOutCur;
            //if(widthOutCur > cLayer->featureMapWidth[i]) cLayer->featureMapWidth[i] = widthOutCur;
            int heightOutOffset = (cLayer->featureMapHeightMax - heightOutCur) / 2;
            int widthOutOffset = (cLayer->featureMapWidthMax - widthOutCur) / 2;
            int heightOutOffsetIn = (cLayer->inputHeightMax - cLayer->featureMapHeightMax) / 2;
            int widthOutOffsetIn = (cLayer->inputWidthMax - cLayer->featureMapWidthMax) / 2;
            int heightKnOffsetIn = (cLayer->inputHeightMax - kernelHeightCur) / 2;
            int widthKnOffsetIn = (cLayer->inputWidthMax - kernelWidthCur) / 2;
            for(int h = 0; h < kernelHeightCur; h++) {
                for(int w = 0; w < kernelWidthCur; w++) {
                    MY_FLT_TYPE tmp_conv = 0;
                    int heightPadding = kernelHeightCur / 2;
                    int widthPadding = kernelWidthCur / 2;
                    if(cLayer->paddingType[i][j] == PADDING_SAME) {
                        heightPadding = kernelHeightCur / 2;
                        widthPadding = kernelWidthCur / 2;
                    } else if(cLayer->paddingType[i][j] == PADDING_VALID) {
                        heightPadding = 0;
                        widthPadding = 0;
                    }
                    for(int a = 0; a < heightOutCur; a++) {
                        for(int b = 0; b < widthOutCur; b++) {
                            int inHCur = heightInOffset + a - heightPadding + h;
                            int inWCur = widthInOffset + b - widthPadding + w;
                            int outHCur = heightOutOffset + a;
                            int outWCur = widthOutOffset + b;
                            if(inHCur < heightInOffset ||
                               inWCur < widthInOffset ||
                               inHCur >= heightInOffset + heightInCur ||
                               inWCur >= widthInOffset + widthInCur ||
                               tagPriorLayer[j][inHCur][inWCur] == 0 ||
                               cLayer->featureMapTag[i][outHCur][outWCur] == 0)
                                continue;
                            tmp_conv += dataInput[j][inHCur][inWCur] *
                                        cLayer->featureMapDelta[i][outHCur][outWCur];
                        }
                    }
                    cLayer->kernelDelta[i][j][kernelHeightOffset + h][kernelWidthOffset + w] += tmp_conv;
                    if(CHECK_INVALID(cLayer->kernelDelta[i][j][kernelHeightOffset + h][kernelWidthOffset + w]) ||
                       fabs(cLayer->kernelDelta[i][j][kernelHeightOffset + h][kernelWidthOffset + w]) > 1e10) {
                        int test = 0;
                    }
                }
            }
        }
        MY_FLT_TYPE tmp_sum = 0;
        for(int h = 0; h < cLayer->featureMapHeightMax; h++) {
            for(int w = 0; w < cLayer->featureMapWidthMax; w++) {
                tmp_sum += cLayer->featureMapDelta[i][h][w];
            }
        }
        cLayer->biasDelta[i] += tmp_sum;
        if(CHECK_INVALID(cLayer->biasDelta[i]) || fabs(cLayer->biasDelta[i]) > 1e10) {
            int test = 0;
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_update_convLayer - cLayer->kernelDelta[][][][]\n");
    for(int i = 0; i < cLayer->channelsOut; i++) {
        for(int j = 0; j < cLayer->channelsInMax; j++) {
            for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                printf("[%d][%d][%d][] ", i, j, h);
                for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                    printf("%e\t", cLayer->kernelDelta[i][j][h][w]);
                }
                printf("\n");
            }
        }
    }
    printf("bp_update_convLayer - cLayer->biasDelta[]\n");
    for(int i = 0; i < cLayer->channelsOut; i++) {
        printf("%e\t", cLayer->biasDelta[i]);
    }
    printf("\n");
#endif

    //opts.cur_sample_num++;
    if(opts.tag_update) {
        MY_FLT_TYPE tmp_alpha = opts.alpha;
        for(int i = 0; i < cLayer->channelsOut; i++) {
            for(int j = 0; j < cLayer->channelsInMax; j++) {
                for(int h = 0; h < cLayer->kernelHeightMax; h++) {
                    for(int w = 0; w < cLayer->kernelWidthMax; w++) {
                        cLayer->kernelData[i][j][h][w] -= tmp_alpha * cLayer->kernelDelta[i][j][h][w];
                    }
                }
            }
            cLayer->biasData[i] -= tmp_alpha * cLayer->biasDelta[i];
        }
    }

    return;
}

void bp_update_icfcLayer(InterCPCLayer* icfcLayer, MY_FLT_TYPE*** dataInput, int*** tagPriorLayer, FRNNOpts opts)
{
    if(opts.tag_init) {
        for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
            for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
                for(int h = 0; h < icfcLayer->preFeatureMapHeightMax; h++) {
                    memset(icfcLayer->connectWtDeltaAll[iOut][iCh][h], 0, icfcLayer->preFeatureMapWidthMax * sizeof(MY_FLT_TYPE));
                }
            }
        }
        memset(icfcLayer->biasDelta, 0, icfcLayer->numOutput * sizeof(MY_FLT_TYPE));
    }
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
            int heightInCur = icfcLayer->preInputHeight[iCh];
            int widthInCur = icfcLayer->preInputWidth[iCh];
            if(heightInCur == 0 || widthInCur == 0) continue;
            int quoHeight = icfcLayer->preFeatureMapHeightMax / heightInCur;
            int remHeight = icfcLayer->preFeatureMapHeightMax % heightInCur;
            int quoWidth = icfcLayer->preFeatureMapWidthMax / widthInCur;
            int remWidth = icfcLayer->preFeatureMapWidthMax % widthInCur;
            int heightInOffset = (icfcLayer->preFeatureMapHeightMax - heightInCur) / 2;
            int widthInOffset = (icfcLayer->preFeatureMapWidthMax - widthInCur) / 2;
            int offsetHeight = 0;
            int offsetWidth = 0;
            for(int iH = 0; iH < icfcLayer->preInputHeight[iCh]; iH++) {
                int kernelHeightCur = quoHeight;
                if(iH < remHeight) kernelHeightCur++;
                offsetWidth = 0;
                for(int iW = 0; iW < icfcLayer->preInputWidth[iCh]; iW++) {
                    int kernelWidthCur = quoWidth;
                    if(iW < remWidth) kernelWidthCur++;
                    int inHcur = heightInOffset + iH;
                    int inWcur = widthInOffset + iW;
                    if(inHcur < heightInOffset ||
                       inWcur < widthInOffset ||
                       inHcur >= heightInOffset + heightInCur ||
                       inWcur >= widthInOffset + widthInCur ||
                       tagPriorLayer[iCh][inHcur][inWcur] == 0)
                        continue;
                    int tmp_count = 0;
                    for(int h = 0; h < kernelHeightCur; h++) {
                        for(int w = 0; w < kernelWidthCur; w++) {
                            int wtHcur = offsetHeight + h;
                            int wtWcur = offsetWidth + w;
                            if(wtHcur < 0 ||
                               wtWcur < 0 ||
                               wtHcur >= icfcLayer->preFeatureMapHeightMax ||
                               wtWcur >= icfcLayer->preFeatureMapWidthMax) {
                                printf("%s(%d): Error out of range, exiting...\n",
                                       __FILE__, __LINE__);
                                exit(-1);
                            }
                            if(icfcLayer->connectStatusAll[iOut][iCh][wtHcur][wtWcur]) {
                                tmp_count++;
                            }
                        }
                    }
                    if(tmp_count) {
                        for(int h = 0; h < kernelHeightCur; h++) {
                            for(int w = 0; w < kernelWidthCur; w++) {
                                int wtHcur = offsetHeight + h;
                                int wtWcur = offsetWidth + w;
                                if(wtHcur < 0 ||
                                   wtWcur < 0 ||
                                   wtHcur >= icfcLayer->preFeatureMapHeightMax ||
                                   wtWcur >= icfcLayer->preFeatureMapWidthMax) {
                                    printf("%s(%d): Error out of range, exiting...\n",
                                           __FILE__, __LINE__);
                                    exit(-1);
                                }
                                if(icfcLayer->connectStatusAll[iOut][iCh][wtHcur][wtWcur]) {
                                    icfcLayer->connectWtDeltaAll[iOut][iCh][wtHcur][wtWcur] +=
                                        icfcLayer->outputDelta[iOut] * dataInput[iCh][inHcur][inWcur] / tmp_count;
                                    if(CHECK_INVALID(icfcLayer->connectWtDeltaAll[iOut][iCh][wtHcur][wtWcur]) ||
                                       fabs(icfcLayer->connectWtDeltaAll[iOut][iCh][wtHcur][wtWcur]) > 1e10) {
                                        int test = 0;
                                    }
                                }
                            }
                        }
                    }
                    offsetWidth += kernelWidthCur;
                }
                offsetHeight += kernelHeightCur;
            }
        }
        if(icfcLayer->connectCountAll[iOut]) {
            icfcLayer->biasDelta[iOut] += icfcLayer->outputDelta[iOut];
            if(CHECK_INVALID(icfcLayer->biasDelta[iOut]) ||
               fabs(icfcLayer->biasDelta[iOut]) > 1e10) {
                int test = 0;
            }
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_update_icfcLayer - icfcLayer->connectWtDeltaAll[][][][]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
            for(int h = 0; h < icfcLayer->preFeatureMapHeightMax; h++) {
                printf("[%d][%d][%d][] ", iOut, iCh, h);
                for(int w = 0; w < icfcLayer->preFeatureMapWidthMax; w++) {
                    printf("%e\t", icfcLayer->connectWtDeltaAll[iOut][iCh][h][w]);
                }
                printf("\n");
            }
        }
    }
    printf("bp_update_convLayer - icfcLayer->biasDelta[]\n");
    for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
        printf("%e\t", icfcLayer->biasDelta[iOut]);
    }
    printf("\n");
#endif

    //opts.cur_sample_num++;
    if(opts.tag_update) {
        MY_FLT_TYPE tmp_alpha = opts.alpha;
        for(int iOut = 0; iOut < icfcLayer->numOutput; iOut++) {
            for(int iCh = 0; iCh < icfcLayer->preFeatureMapChannels; iCh++) {
                for(int h = 0; h < icfcLayer->preFeatureMapHeightMax; h++) {
                    for(int w = 0; w < icfcLayer->preFeatureMapWidthMax; w++) {
                        icfcLayer->connectWeightAll[iOut][iCh][h][w] -= tmp_alpha * icfcLayer->connectWtDeltaAll[iOut][iCh][h][w];
                    }
                }
            }
            icfcLayer->biasData[iOut] -= tmp_alpha * icfcLayer->biasDelta[iOut];
        }
    }

    return;
}

void bp_update_fcLayer(FCLayer* fcLayer, MY_FLT_TYPE* dataInput, int* tagPriorLayer, FRNNOpts opts)
{
    if(opts.tag_init) {
        for(int i = 0; i < fcLayer->numOutput; i++) {
            memset(fcLayer->connectWtDelta[i], 0, fcLayer->numInputMax * sizeof(MY_FLT_TYPE));
        }
        memset(fcLayer->biasDelta, 0, fcLayer->numOutput * sizeof(MY_FLT_TYPE));
    }
    for(int i = 0; i < fcLayer->numOutput; i++) {
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            if(tagPriorLayer[j]) {
                fcLayer->connectWtDelta[i][j] += fcLayer->outputDelta[i] * dataInput[j] * fcLayer->connectStatus[i][j];
                if(CHECK_INVALID(fcLayer->connectWtDelta[i][j]) ||
                   fabs(fcLayer->connectWtDelta[i][j]) > 1e10) {
                    int test = 0;
                }
            }
        }
        fcLayer->biasDelta[i] += fcLayer->outputDelta[i];
        if(CHECK_INVALID(fcLayer->biasDelta[i]) ||
           fabs(fcLayer->biasDelta[i]) > 1e10) {
            int test = 0;
        }
    }

#ifdef MY_DEBUG_TAG
    printf("bp_update_fcLayer - fcLayer->connectWtDelta[][]\n");
    for(int i = 0; i < fcLayer->numOutput; i++) {
        printf("[%d][] ", i);
        for(int j = 0; j < fcLayer->numInputMax; j++) {
            printf("%e\t", fcLayer->connectWtDelta[i][j]);
        }
        printf("\n");
    }
    printf("bp_update_convLayer - fcLayer->biasDelta[]\n");
    for(int i = 0; i < fcLayer->numOutput; i++) {
        printf("%e\t", fcLayer->biasDelta[i]);
    }
    printf("\n");
#endif

    //opts.cur_sample_num++;
    if(opts.tag_update) {
        MY_FLT_TYPE tmp_alpha = opts.alpha;
        for(int i = 0; i < fcLayer->numOutput; i++) {
            for(int j = 0; j < fcLayer->numInputMax; j++) {
                fcLayer->connectWeight[i][j] -= tmp_alpha * fcLayer->connectWtDelta[i][j];
            }
            fcLayer->biasData[i] -= tmp_alpha * fcLayer->biasDelta[i];
        }
    }

    return;
}

//////////////////////////////////////////////////////////////////
#define IM1_FRNN_MODEL 2147483563
#define IM2_FRNN_MODEL 2147483399
#define AM_FRNN_MODEL (1.0/IM1_FRNN_MODEL)
#define IMM1_FRNN_MODEL (IM1_FRNN_MODEL-1)
#define IA1_FRNN_MODEL 40014
#define IA2_FRNN_MODEL 40692
#define IQ1_FRNN_MODEL 53668
#define IQ2_FRNN_MODEL 52774
#define IR1_FRNN_MODEL 12211
#define IR2_FRNN_MODEL 3791
#define NTAB_FRNN_MODEL 32
#define NDIV_FRNN_MODEL (1+IMM1_FRNN_MODEL/NTAB_FRNN_MODEL)
#define EPS_FRNN_MODEL 1.2e-7
#define RNMX_FRNN_MODEL (1.0-EPS_FRNN_MODEL)

//the random generator in [0,1)
double rnd_uni_FRNN_MODEL(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_FRNN_MODEL];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_FRNN_MODEL + 7; j >= 0; j--) {
            k = (*idum) / IQ1_FRNN_MODEL;
            *idum = IA1_FRNN_MODEL * (*idum - k * IQ1_FRNN_MODEL) - k * IR1_FRNN_MODEL;
            if(*idum < 0) *idum += IM1_FRNN_MODEL;
            if(j < NTAB_FRNN_MODEL) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_FRNN_MODEL;
    *idum = IA1_FRNN_MODEL * (*idum - k * IQ1_FRNN_MODEL) - k * IR1_FRNN_MODEL;
    if(*idum < 0) *idum += IM1_FRNN_MODEL;
    k = idum2 / IQ2_FRNN_MODEL;
    idum2 = IA2_FRNN_MODEL * (idum2 - k * IQ2_FRNN_MODEL) - k * IR2_FRNN_MODEL;
    if(idum2 < 0) idum2 += IM2_FRNN_MODEL;
    j = iy / NDIV_FRNN_MODEL;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_FRNN_MODEL;    //printf("%lf\n", AM_CLASS*iy);
    if((temp = AM_FRNN_MODEL * iy) > RNMX_FRNN_MODEL) return RNMX_FRNN_MODEL;
    else return temp;
}/*------End of rnd_uni_CLASS()--------------------------*/
int     seed_FRNN_MODEL = 237;
long    rnd_uni_init_FRNN_MODEL = -(long)seed_FRNN_MODEL;

MY_FLT_TYPE rndreal_FRNN_MODEL(MY_FLT_TYPE low, MY_FLT_TYPE high)
{
    return (low + (high - low) * rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL));
}

MY_FLT_TYPE gaussrand_FRNN_MODEL(MY_FLT_TYPE a, MY_FLT_TYPE b)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL);
            double U2 = rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL);

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    //return X;
    return (X * b + a);
}

int rnd_FRNN_MODEL(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_FRNN_MODEL(&rnd_uni_init_FRNN_MODEL) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}

void shuffle_FRNN_MODEL(int* x, int size)
{
    int i, aux, k = 0;
    for(i = size - 1; i > 0; i--) {
        /* get a value between cero and i  */
        k = rnd_FRNN_MODEL(0, i);
        /* exchange of values */
        aux = x[i];
        x[i] = x[k];
        x[k] = aux;
    }
    //
    return;
}
