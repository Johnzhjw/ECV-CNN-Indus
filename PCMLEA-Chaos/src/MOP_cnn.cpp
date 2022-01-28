#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_cnn.h"
#include "MOP_LeNet.h"
#include "MOP_Classify_NN.h"

void cnnsetup(CNN* cnn, nSize inputSize, int outputSize)
{
    cnn->layerNum = 5;

    //cnn->core_mode = CORE_SIGMA;
    //cnn->core_mode = CORE_RELU;
    cnn->core_mode = CORE_LEAKYRELU;

    nSize inSize;
    int mapSize = MAP_SIZE_C;
    inSize.c = inputSize.c;
    inSize.r = inputSize.r;
    cnn->C1 = initCovLayer(inSize.c, inSize.r, MAP_SIZE_C, NUM_CHANNEL_C1_IN, NUM_CHANNEL_C1_OUT);
    inSize.c = inSize.c - mapSize + 1;
    inSize.r = inSize.r - mapSize + 1;
    cnn->S2 = initPoolLayer(inSize.c, inSize.r, MAP_SIZE_S, NUM_CHANNEL_C1_OUT, NUM_CHANNEL_C1_OUT, MaxPool);
    inSize.c = (inSize.c + 1) / 2;
    inSize.r = (inSize.r + 1) / 2;
    cnn->C3 = initCovLayer(inSize.c, inSize.r, MAP_SIZE_C, NUM_CHANNEL_C3_IN, NUM_CHANNEL_C3_OUT);
    inSize.c = inSize.c - mapSize + 1;
    inSize.r = inSize.r - mapSize + 1;
    cnn->S4 = initPoolLayer(inSize.c, inSize.r, MAP_SIZE_S, NUM_CHANNEL_C3_OUT, NUM_CHANNEL_C3_OUT, MaxPool);
    inSize.c = (inSize.c + 1) / 2;
    inSize.r = (inSize.r + 1) / 2;
    cnn->O5 = initOutLayer(inSize.c * inSize.r * NUM_CHANNEL_C3_OUT, outputSize);

    cnn->e = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));

    cnn->N_sum = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));
    cnn->N_wrong = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));
    cnn->e_sum = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));

    cnn->N_TP = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));
    cnn->N_TN = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));
    cnn->N_FP = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));
    cnn->N_FN = (MY_FLT_TYPE*)calloc(cnn->O5->outputNum, sizeof(MY_FLT_TYPE));

    return;
}

void cnnfree(CNN* cnn)
{
    freeCovLayer(cnn->C1);
    freePoolLayer(cnn->S2);
    freeCovLayer(cnn->C3);
    freePoolLayer(cnn->S4);
    freeOutLayer(cnn->O5);

    free(cnn->e);

    free(cnn->N_sum);
    free(cnn->N_wrong);
    free(cnn->e_sum);

    free(cnn->N_TP);
    free(cnn->N_TN);
    free(cnn->N_FP);
    free(cnn->N_FN);

    free(cnn);
}

void cnninit(CNN* cnn, double* x, int mode)
{
    MY_FLT_TYPE randnum;

    int count = 0;
    switch(mode) {
    case INIT_MODE:
        //rnd_uni_init_LeNet = -(long)seed_LeNet;
        break;
    case ASSIGN_MODE:
    case OUTPUT_MODE:
        count = 0;
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    int i, j, c, r;
    int mapSize;
    int inChannels;
    int outChannels;
    int inputNum;
    int outputNum;
    // C1的数据
    mapSize = cnn->C1->mapSize;
    inChannels = cnn->C1->inChannels;
    outChannels = cnn->C1->outChannels;
    for(j = 0; j < outChannels; j++) {
        for(i = 0; i < inChannels; i++) {
            for(r = 0; r < mapSize; r++) {
                for(c = 0; c < mapSize; c++) {
                    switch(mode) {
                    case INIT_MODE:
                        randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                        cnn->C1->mapData[i][j][r][c] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(mapSize * mapSize * (inChannels + outChannels)));
                        //printf("%lf ", cnn->C1->mapData[i][j][r][c]);
                        break;
                    case ASSIGN_MODE:
                        cnn->C1->mapData[i][j][r][c] = (MY_FLT_TYPE)x[count++];
                        break;
                    case OUTPUT_MODE:
                        x[count++] = cnn->C1->mapData[i][j][r][c];
                        break;
                    default:
                        printf("%s(%d): mode error for cnninit, exiting...\n",
                               __FILE__, __LINE__);
                        exit(1000);
                        break;
                    }
                }
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            cnn->C1->biasData[j] = 0;
            break;
        case ASSIGN_MODE:
            cnn->C1->biasData[j] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = cnn->C1->biasData[j];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    // C3网络
    mapSize = cnn->C3->mapSize;
    inChannels = cnn->C3->inChannels;
    outChannels = cnn->C3->outChannels;
    for(j = 0; j < outChannels; j++) {
        for(i = 0; i < inChannels; i++) {
            for(r = 0; r < mapSize; r++) {
                for(c = 0; c < mapSize; c++) {
                    switch(mode) {
                    case INIT_MODE:
                        randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                        cnn->C3->mapData[i][j][r][c] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(mapSize * mapSize * (inChannels + outChannels)));
                        break;
                    case ASSIGN_MODE:
                        cnn->C3->mapData[i][j][r][c] = (MY_FLT_TYPE)x[count++];
                        break;
                    case OUTPUT_MODE:
                        x[count++] = cnn->C3->mapData[i][j][r][c];
                        break;
                    default:
                        printf("%s(%d): mode error for cnninit, exiting...\n",
                               __FILE__, __LINE__);
                        exit(1000);
                        break;
                    }
                }
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            cnn->C3->biasData[j] = 0;
            break;
        case ASSIGN_MODE:
            cnn->C3->biasData[j] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = cnn->C3->biasData[j];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    // O5输出层
    inputNum = cnn->O5->inputNum;
    outputNum = cnn->O5->outputNum;
    for(i = 0; i < cnn->O5->outputNum; i++) {
        for(j = 0; j < cnn->O5->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                cnn->O5->wData[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(inputNum + outputNum));
                break;
            case ASSIGN_MODE:
                cnn->O5->wData[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = cnn->O5->wData[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            cnn->O5->biasData[i] = 0;
            break;
        case ASSIGN_MODE:
            cnn->O5->biasData[i] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = cnn->O5->biasData[i];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    //
#if OPTIMIZE_STRUCTURE_CNN == 1
    // C1的数据
    mapSize = cnn->C1->mapSize;
    inChannels = cnn->C1->inChannels;
    outChannels = cnn->C1->outChannels;
    for(j = 0; j < outChannels; j++) {
        for(i = 0; i < inChannels; i++) {
            switch(mode) {
            case INIT_MODE:
                cnn->C1->mapFlag[i][j] = 1;
                break;
            case ASSIGN_MODE:
                cnn->C1->mapFlag[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = cnn->C1->mapFlag[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }

    // C3网络
    mapSize = cnn->C3->mapSize;
    inChannels = cnn->C3->inChannels;
    outChannels = cnn->C3->outChannels;
    for(j = 0; j < outChannels; j++) {
        for(i = 0; i < inChannels; i++) {
            switch(mode) {
            case INIT_MODE:
                cnn->C3->mapFlag[i][j] = 1;
                break;
            case ASSIGN_MODE:
                cnn->C3->mapFlag[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = cnn->C3->mapFlag[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }

    // O5输出层
    inputNum = cnn->O5->inputNum;
    outputNum = cnn->O5->outputNum;
    for(i = 0; i < cnn->O5->outputNum; i++) {
        for(j = 0; j < cnn->O5->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                cnn->O5->connFalg[i][j] = 1;
                break;
            case ASSIGN_MODE:
                cnn->O5->connFalg[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = cnn->O5->connFalg[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }
#endif
}

void nnsetup(NN* nn, nSize inputSize, int outputSize)
{
    nn->layerNum = 5;

    //cnn->core_mode = CORE_SIGMA;
    //cnn->core_mode = CORE_RELU;
    nn->core_mode = CORE_LEAKYRELU;

    int inSize1 = inputSize.c * inputSize.r;
    int outSize1 = 600;
    nn->O1 = initOutLayer(inSize1, outSize1);
    int inSize2 = outSize1;
    int outSize2 = 192;
    nn->O2 = initOutLayer(inSize2, outSize2);
    int inSize3 = outSize2;
    int outSize3 = outputSize;
    nn->O3 = initOutLayer(inSize3, outSize3);

    nn->e = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));

    nn->N_sum = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));
    nn->N_wrong = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));
    nn->e_sum = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));

    nn->N_TP = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));
    nn->N_TN = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));
    nn->N_FP = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));
    nn->N_FN = (MY_FLT_TYPE*)calloc(nn->O3->outputNum, sizeof(MY_FLT_TYPE));

    return;
}

void nnfree(NN* nn)
{
    freeOutLayer(nn->O1);
    freeOutLayer(nn->O2);
    freeOutLayer(nn->O3);

    free(nn->e);

    free(nn->N_sum);
    free(nn->N_wrong);
    free(nn->e_sum);

    free(nn->N_TP);
    free(nn->N_TN);
    free(nn->N_FP);
    free(nn->N_FN);

    free(nn);
}

void nninit(NN* nn, double* x, int mode)
{
    MY_FLT_TYPE randnum;

    int count = 0;
    switch(mode) {
    case INIT_MODE:
        //rnd_uni_init_LeNet = -(long)seed_LeNet;
        break;
    case ASSIGN_MODE:
    case OUTPUT_MODE:
        count = 0;
        break;
    default:
        printf("%s(%d): mode error for cnninit, exiting...\n",
               __FILE__, __LINE__);
        exit(1000);
        break;
    }

    int i, j;
    int inputNum;
    int outputNum;
    // O1隐藏层
    inputNum = nn->O1->inputNum;
    outputNum = nn->O1->outputNum;
    for(i = 0; i < nn->O1->outputNum; i++) {
        for(j = 0; j < nn->O1->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                nn->O1->wData[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(inputNum + outputNum));
                break;
            case ASSIGN_MODE:
                nn->O1->wData[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O1->wData[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            nn->O1->biasData[i] = 0;
            break;
        case ASSIGN_MODE:
            nn->O1->biasData[i] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = nn->O1->biasData[i];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    // O2隐藏层
    inputNum = nn->O2->inputNum;
    outputNum = nn->O2->outputNum;
    for(i = 0; i < nn->O2->outputNum; i++) {
        for(j = 0; j < nn->O2->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                nn->O2->wData[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(inputNum + outputNum));
                break;
            case ASSIGN_MODE:
                nn->O2->wData[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O2->wData[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            nn->O2->biasData[i] = 0;
            break;
        case ASSIGN_MODE:
            nn->O2->biasData[i] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = nn->O2->biasData[i];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    // O3隐藏层
    inputNum = nn->O3->inputNum;
    outputNum = nn->O3->outputNum;
    for(i = 0; i < nn->O3->outputNum; i++) {
        for(j = 0; j < nn->O3->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                nn->O3->wData[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(inputNum + outputNum));
                break;
            case ASSIGN_MODE:
                nn->O3->wData[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O3->wData[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
        //
        switch(mode) {
        case INIT_MODE:
            nn->O3->biasData[i] = 0;
            break;
        case ASSIGN_MODE:
            nn->O3->biasData[i] = (MY_FLT_TYPE)x[count++];
            break;
        case OUTPUT_MODE:
            x[count++] = nn->O3->biasData[i];
            break;
        default:
            printf("%s(%d): mode error for cnninit, exiting...\n",
                   __FILE__, __LINE__);
            exit(1000);
            break;
        }
    }

    //
#if OPTIMIZE_STRUCTURE_NN == 1
    // O1隐藏层
    inputNum = nn->O1->inputNum;
    outputNum = nn->O1->outputNum;
    for(i = 0; i < nn->O1->outputNum; i++) {
        for(j = 0; j < nn->O1->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                nn->O1->connFalg[i][j] = 1;
                break;
            case ASSIGN_MODE:
                nn->O1->connFalg[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O1->connFalg[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }

    // O2隐藏层
    inputNum = nn->O2->inputNum;
    outputNum = nn->O2->outputNum;
    for(i = 0; i < nn->O2->outputNum; i++) {
        for(j = 0; j < nn->O2->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                nn->O2->connFalg[i][j] = 1;
                break;
            case ASSIGN_MODE:
                nn->O2->connFalg[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O2->connFalg[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }

    // O3隐藏层
    inputNum = nn->O3->inputNum;
    outputNum = nn->O3->outputNum;
    for(i = 0; i < nn->O3->outputNum; i++) {
        for(j = 0; j < nn->O3->inputNum; j++) {
            switch(mode) {
            case INIT_MODE:
                nn->O3->connFalg[i][j] = 1;
                break;
            case ASSIGN_MODE:
                nn->O3->connFalg[i][j] = (MY_FLT_TYPE)x[count++];
                break;
            case OUTPUT_MODE:
                x[count++] = nn->O3->connFalg[i][j];
                break;
            default:
                printf("%s(%d): mode error for cnninit, exiting...\n",
                       __FILE__, __LINE__);
                exit(1000);
                break;
            }
        }
    }
#endif
}

CovLayer_CNN* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
    CovLayer_CNN* covL = (CovLayer_CNN*)malloc(sizeof(CovLayer_CNN));

    covL->inputHeight = inputHeight;
    covL->inputWidth = inputWidth;
    covL->mapSize = mapSize;

    covL->inChannels = inChannels;
    covL->outChannels = outChannels;

    covL->isFullConnect = true; // 默认为全连接

    // 权重空间的初始化，先行再列调用，[r][c]
    int i, j, c, r;
    //srand((unsigned)time(NULL));
    covL->mapData = (MY_FLT_TYPE****)malloc(inChannels * sizeof(MY_FLT_TYPE***));
    for(i = 0; i < inChannels; i++) {
        covL->mapData[i] = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
        for(j = 0; j < outChannels; j++) {
            covL->mapData[i][j] = (MY_FLT_TYPE**)malloc(mapSize * sizeof(MY_FLT_TYPE*));
            for(r = 0; r < mapSize; r++) {
                covL->mapData[i][j][r] = (MY_FLT_TYPE*)malloc(mapSize * sizeof(MY_FLT_TYPE));
                for(c = 0; c < mapSize; c++) {
                    MY_FLT_TYPE randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
                    covL->mapData[i][j][r][c] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(mapSize * mapSize * (inChannels + outChannels)));
                }
            }
        }
    }
    // 权重梯度变化
    covL->dmapData = (MY_FLT_TYPE****)malloc(inChannels * sizeof(MY_FLT_TYPE***));
    for(i = 0; i < inChannels; i++) {
        covL->dmapData[i] = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
        for(j = 0; j < outChannels; j++) {
            covL->dmapData[i][j] = (MY_FLT_TYPE**)malloc(mapSize * sizeof(MY_FLT_TYPE*));
            for(r = 0; r < mapSize; r++) {
                covL->dmapData[i][j][r] = (MY_FLT_TYPE*)calloc(mapSize, sizeof(MY_FLT_TYPE));
            }
        }
    }
    //
    covL->mapFlag = (MY_FLT_TYPE**)malloc(inChannels * sizeof(MY_FLT_TYPE*));
    for(i = 0; i < inChannels; i++) {
        covL->mapFlag[i] = (MY_FLT_TYPE*)malloc(outChannels * sizeof(MY_FLT_TYPE));
        for(j = 0; j < outChannels; j++) {
            covL->mapFlag[i][j] = 1;
        }
    }

    covL->biasData = (MY_FLT_TYPE*)calloc(outChannels, sizeof(MY_FLT_TYPE));

    int outW = inputWidth - mapSize + 1;
    int outH = inputHeight - mapSize + 1;

    covL->d = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
    covL->v = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
    covL->y = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
    for(j = 0; j < outChannels; j++) {
        covL->d[j] = (MY_FLT_TYPE**)malloc(outH * sizeof(MY_FLT_TYPE*));
        covL->v[j] = (MY_FLT_TYPE**)malloc(outH * sizeof(MY_FLT_TYPE*));
        covL->y[j] = (MY_FLT_TYPE**)malloc(outH * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < outH; r++) {
            covL->d[j][r] = (MY_FLT_TYPE*)calloc(outW, sizeof(MY_FLT_TYPE));
            covL->v[j][r] = (MY_FLT_TYPE*)calloc(outW, sizeof(MY_FLT_TYPE));
            covL->y[j][r] = (MY_FLT_TYPE*)calloc(outW, sizeof(MY_FLT_TYPE));
        }
    }

    return covL;
}

void freeCovLayer(CovLayer_CNN* layer_C)
{
    int i, j, r;
    for(i = 0; i < layer_C->inChannels; i++) {
        for(j = 0; j < layer_C->outChannels; j++) {
            for(r = 0; r < layer_C->mapSize; r++) {
                free(layer_C->mapData[i][j][r]);
                free(layer_C->dmapData[i][j][r]);
            }
            free(layer_C->mapData[i][j]);
            free(layer_C->dmapData[i][j]);
        }
        free(layer_C->mapData[i]);
        free(layer_C->dmapData[i]);
    }
    free(layer_C->mapData);
    free(layer_C->dmapData);

    for(i = 0; i < layer_C->inChannels; i++) {
        free(layer_C->mapFlag[i]);
    }
    free(layer_C->mapFlag);

    free(layer_C->biasData);

    // int outW = layer_C->inputWidth - layer_C->mapSize + 1;
    int outH = layer_C->inputHeight - layer_C->mapSize + 1;

    for(j = 0; j < layer_C->outChannels; j++) {
        for(r = 0; r < outH; r++) {
            free(layer_C->d[j][r]);
            free(layer_C->v[j][r]);
            free(layer_C->y[j][r]);
        }
        free(layer_C->d[j]);
        free(layer_C->v[j]);
        free(layer_C->y[j]);
    }
    free(layer_C->d);
    free(layer_C->v);
    free(layer_C->y);

    free(layer_C);
}

PoolLayer_CNN* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
    PoolLayer_CNN* poolL = (PoolLayer_CNN*)malloc(sizeof(PoolLayer_CNN));

    poolL->inputHeight = inputHeight;
    poolL->inputWidth = inputWidth;
    poolL->mapSize = mapSize;
    poolL->inChannels = inChannels;
    poolL->outChannels = outChannels;
    poolL->poolType = poolType;

    poolL->biasData = (MY_FLT_TYPE*)calloc(outChannels, sizeof(MY_FLT_TYPE));

    int outW = (inputWidth + mapSize - 1) / mapSize;
    int outH = (inputHeight + mapSize - 1) / mapSize;

    int j, r;
    poolL->d = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
    poolL->y = (MY_FLT_TYPE***)malloc(outChannels * sizeof(MY_FLT_TYPE**));
    for(j = 0; j < outChannels; j++) {
        poolL->d[j] = (MY_FLT_TYPE**)malloc(outH * sizeof(MY_FLT_TYPE*));
        poolL->y[j] = (MY_FLT_TYPE**)malloc(outH * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < outH; r++) {
            poolL->d[j][r] = (MY_FLT_TYPE*)calloc(outW, sizeof(MY_FLT_TYPE));
            poolL->y[j][r] = (MY_FLT_TYPE*)calloc(outW, sizeof(MY_FLT_TYPE));
        }
    }

    return poolL;
}

void freePoolLayer(PoolLayer_CNN* layer_S)
{
    free(layer_S->biasData);

    // int outW = layer_S->inputWidth / layer_S->mapSize;
    int outH = layer_S->inputHeight / layer_S->mapSize;

    int j, r;
    for(j = 0; j < layer_S->outChannels; j++) {
        for(r = 0; r < outH; r++) {
            free(layer_S->d[j][r]);
            free(layer_S->y[j][r]);
        }
        free(layer_S->d[j]);
        free(layer_S->y[j]);
    }
    free(layer_S->d);
    free(layer_S->y);

    free(layer_S);
}

OutLayer_CNN* initOutLayer(int inputNum, int outputNum)
{
    OutLayer_CNN* outL = (OutLayer_CNN*)malloc(sizeof(OutLayer_CNN));

    outL->inputNum = inputNum;
    outL->outputNum = outputNum;

    outL->biasData = (MY_FLT_TYPE*)calloc(outputNum, sizeof(MY_FLT_TYPE));

    outL->d = (MY_FLT_TYPE*)calloc(outputNum, sizeof(MY_FLT_TYPE));
    outL->v = (MY_FLT_TYPE*)calloc(outputNum, sizeof(MY_FLT_TYPE));
    outL->y = (MY_FLT_TYPE*)calloc(outputNum, sizeof(MY_FLT_TYPE));

    // 权重的初始化
    outL->wData = (MY_FLT_TYPE**)malloc(outputNum * sizeof(MY_FLT_TYPE*)); // 输入行，输出列
    outL->connFalg = (MY_FLT_TYPE**)malloc(outputNum * sizeof(MY_FLT_TYPE*));
    int i, j;
    //srand((unsigned)time(NULL));
    for(i = 0; i < outputNum; i++) {
        outL->wData[i] = (MY_FLT_TYPE*)malloc(inputNum * sizeof(MY_FLT_TYPE));
        outL->connFalg[i] = (MY_FLT_TYPE*)malloc(inputNum * sizeof(MY_FLT_TYPE));
        for(j = 0; j < inputNum; j++) {
            MY_FLT_TYPE randnum = (MY_FLT_TYPE)((rnd_uni_CNN(&rnd_uni_init_CNN) - 0.5) * 2);
            outL->wData[i][j] = randnum * sqrt((MY_FLT_TYPE)6.0 / (MY_FLT_TYPE)(inputNum + outputNum));
            outL->connFalg[i][j] = 1;
        }
    }

    outL->isFullConnect = true;

    return outL;
}

void freeOutLayer(OutLayer_CNN* layer_O)
{
    free(layer_O->biasData);

    free(layer_O->d);
    free(layer_O->v);
    free(layer_O->y);

    int i;
    for(i = 0; i < layer_O->outputNum; i++) {
        free(layer_O->wData[i]);
        free(layer_O->connFalg[i]);
    }
    free(layer_O->wData);
    free(layer_O->connFalg);

    free(layer_O);
}

int vecmaxIndex(MY_FLT_TYPE* vec, int veclength)// 返回向量最大数的序号
{
    int i;
    MY_FLT_TYPE maxnum = vec[0];// -1.0;
    int maxIndex = 0;
    for(i = 1; i < veclength; i++) {
        if(maxnum < vec[i]) {
            maxnum = vec[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

// 测试cnn函数
MY_FLT_TYPE cnntest(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum)
{
    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    for(n = 0; n < testNum; n++) {
        cnnff(cnn, inputData->ImgPtr[n].ImgData, cnn->core_mode);

        int i;
        for(i = 0; i < cnn->O5->outputNum; i++)
            cnn->e[i] = cnn->O5->y[i] - outputData->LabelPtr[n].LabelData[i];
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);
        int a;
        for(a = 0; a < cnn->O5->outputNum; a++) {
            cnn->e_sum[a] += fabs(cnn->e[a]);
        }
        cnn->N_sum[tmp2]++;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2]++;
            incorrectnum++;
        }

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i]++;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i]++;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i]++;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i]++;
        }
        //
        cnnclear(cnn);
    }
    if(testNum > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)testNum;
    else
        return 1;
}
MY_FLT_TYPE cnntest_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum, int* vec_index)
{
    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    for(n = 0; n < testNum; n++) {
        int cur_index = vec_index[n];
        cnnff(cnn, inputData->ImgPtr[cur_index].ImgData, cnn->core_mode);

        int i;
        for(i = 0; i < cnn->O5->outputNum; i++)
            cnn->e[i] = cnn->O5->y[i] - outputData->LabelPtr[cur_index].LabelData[i];
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[cur_index].LabelData, cnn->O5->outputNum);
        int a;
        for(a = 0; a < cnn->O5->outputNum; a++) {
            cnn->e_sum[a] += fabs(cnn->e[a]);
        }
        cnn->N_sum[tmp2]++;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2]++;
            incorrectnum++;
        }

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i]++;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i]++;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i]++;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i]++;
        }

        cnnclear(cnn);
    }
    if(testNum > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)testNum;
    else
        return 1;
}

void cnn_train_bp(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int* indx_samples, int num_samples)
{
    int len_samples = num_samples;// inputData->ImgNum;
    // 学习训练误差曲线
    cnn->L = (MY_FLT_TYPE*)calloc(len_samples, sizeof(MY_FLT_TYPE));
    int e;
    for(e = 0; e < opts.numepochs; e++) {
        if(e % 10 == 0) {
            printf("%d/%d\n", e, opts.numepochs);
        }
        opts.alpha *= (MY_FLT_TYPE)0.95;
        for(int n = 0; n < len_samples; n++) {
            int cur_ind = indx_samples[n];
            if(cur_ind < 0) continue;
            //printf("%d-%d\n", e, n);
            cnnff(cnn, inputData->ImgPtr[cur_ind].ImgData, cnn->core_mode);  // 前向传播，这里主要计算各
            cnnbp(cnn, outputData->LabelPtr[cur_ind].LabelData,
                  cnn->core_mode); // 后向传播，这里主要计算各神经元的误差梯度
            //
            int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
            int tmp2 = vecmaxIndex(outputData->LabelPtr[cur_ind].LabelData, cnn->O5->outputNum);
            int a;
            for(a = 0; a < cnn->O5->outputNum; a++) {
                cnn->e_sum[a] += fabs(cnn->e[a]);
            }
            cnn->N_sum[tmp2]++;
            if(tmp1 != tmp2) {
                cnn->N_wrong[tmp2]++;
            }
            int i;
            for(i = 0; i < cnn->O5->outputNum; i++) {
                if(i == tmp1 && i == tmp2) cnn->N_TP[i]++;
                if(i == tmp1 && i != tmp2) cnn->N_FP[i]++;
                if(i != tmp1 && i == tmp2) cnn->N_FN[i]++;
                if(i != tmp1 && i != tmp2) cnn->N_TN[i]++;
            }

            //char* filedir = "PicTrans/CNNData/";
            //const char* filename = combine_strings(filedir, combine_strings(intTochar(n), ".cnn"));
            //savecnndata(cnn, filename, inputData->ImgPtr[n].ImgData);
            cnnapplygrads(cnn, opts, inputData->ImgPtr[cur_ind].ImgData); // 更新权重

            cnnclear(cnn);
            // 计算并保存误差能量
            MY_FLT_TYPE l = 0.0;
            //int i;
            for(i = 0; i < cnn->O5->outputNum; i++)
                l = l + cnn->e[i] * cnn->e[i];
            if(n == 0)
                cnn->L[n] = l / (MY_FLT_TYPE)2.0;
            else
                cnn->L[n] = (MY_FLT_TYPE)(cnn->L[n - 1] * 0.99 + 0.01 * l / 2.0);
        }
    }
}
void nn_train_bp(NN* nn, ImgArr inputData, LabelArr outputData, NNOpts opts, int* indx_samples, int num_samples)
{
    int len_samples = num_samples;// inputData->ImgNum;
    // 学习训练误差曲线
    nn->L = (MY_FLT_TYPE*)calloc(len_samples, sizeof(MY_FLT_TYPE));
    int e;
    for(e = 0; e < opts.numepochs; e++) {
        if(e % 10 == 0) {
            printf("%d/%d\n", e, opts.numepochs);
        }
        opts.alpha *= (MY_FLT_TYPE)0.95;
        for(int n = 0; n < len_samples; n++) {
            int cur_ind = indx_samples[n];
            if(cur_ind < 0) continue;
            //printf("%d-%d\n", e, n);
            int n_r = inputData->ImgPtr[cur_ind].r;
            int n_c = inputData->ImgPtr[cur_ind].c;
            MY_FLT_TYPE* tmp_ImgData = (MY_FLT_TYPE*)malloc(n_r * n_c * sizeof(MY_FLT_TYPE));
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[cur_ind].ImgData[i][j];
                }
            }
            nnff(nn, tmp_ImgData, nn->core_mode);
            nnbp(nn, outputData->LabelPtr[cur_ind].LabelData,
                 nn->core_mode); // 后向传播，这里主要计算各神经元的误差梯度
            //
            int tmp1 = vecmaxIndex(nn->O3->y, nn->O3->outputNum);
            int tmp2 = vecmaxIndex(outputData->LabelPtr[cur_ind].LabelData, nn->O3->outputNum);
            int a;
            for(a = 0; a < nn->O3->outputNum; a++) {
                nn->e_sum[a] += fabs(nn->e[a]);
            }
            nn->N_sum[tmp2]++;
            if(tmp1 != tmp2) {
                nn->N_wrong[tmp2]++;
            }
            int i;
            for(i = 0; i < nn->O3->outputNum; i++) {
                if(i == tmp1 && i == tmp2) nn->N_TP[i]++;
                if(i == tmp1 && i != tmp2) nn->N_FP[i]++;
                if(i != tmp1 && i == tmp2) nn->N_FN[i]++;
                if(i != tmp1 && i != tmp2) nn->N_TN[i]++;
            }

            //char* filedir = "PicTrans/CNNData/";
            //const char* filename = combine_strings(filedir, combine_strings(intTochar(n), ".cnn"));
            //savecnndata(cnn, filename, inputData->ImgPtr[n].ImgData);
            nnapplygrads(nn, opts, tmp_ImgData); // 更新权重
            free(tmp_ImgData);

            nnclear(nn);
            // 计算并保存误差能量
            MY_FLT_TYPE l = 0.0;
            //int i;
            for(i = 0; i < nn->O3->outputNum; i++)
                l = l + nn->e[i] * nn->e[i];
            if(n == 0)
                nn->L[n] = l / (MY_FLT_TYPE)2.0;
            else
                nn->L[n] = (MY_FLT_TYPE)(nn->L[n - 1] * 0.99 + 0.01 * l / 2.0);
        }
    }
}
MY_FLT_TYPE cnn_err_train(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index)
{
    cnnclear(cnn);

    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = outputData->LabelPtr[0].l;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag <= 0) continue;
        if(pixel_index) {
            int n_r = inputData->ImgPtr[n].r;
            int n_c = inputData->ImgPtr[n].c;
            MY_FLT_TYPE** tmp_ImgData = (MY_FLT_TYPE**)malloc(n_r * sizeof(MY_FLT_TYPE*));
            for(int i = 0; i < n_r; i++) {
                tmp_ImgData[i] = (MY_FLT_TYPE*)malloc(n_c * sizeof(MY_FLT_TYPE));
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i][j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
            cnnff(cnn, tmp_ImgData, cnn->core_mode);
            for(int i = 0; i < n_r; i++) {
                free(tmp_ImgData[i]);
            }
            free(tmp_ImgData);
        } else {
            cnnff(cnn, inputData->ImgPtr[n].ImgData, cnn->core_mode);
        }
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < cnn->O5->y[i])
                tmp_max = cnn->O5->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(cnn->O5->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < cnn->O5->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= cnn->O5->outputNum;
        cnn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        cnn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i] += cur_flag;
        }

        cnnclear(cnn);
    }
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}
MY_FLT_TYPE nn_err_train(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index)
{
    nnclear(nn);

    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = outputData->LabelPtr[0].l;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag <= 0) continue;
        int n_r = inputData->ImgPtr[n].r;
        int n_c = inputData->ImgPtr[n].c;
        MY_FLT_TYPE* tmp_ImgData = (MY_FLT_TYPE*)malloc(n_r * n_c * sizeof(MY_FLT_TYPE));
        if(pixel_index) {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        } else {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = i;
                    int cur_c = j;
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        }
        nnff(nn, tmp_ImgData, nn->core_mode);
        free(tmp_ImgData);
        //
        int tmp1 = vecmaxIndex(nn->O3->y, nn->O3->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, nn->O3->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < nn->O3->y[i])
                tmp_max = nn->O3->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(nn->O3->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < nn->O3->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= nn->O3->outputNum;
        nn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        nn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            nn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < nn->O3->outputNum; i++) {
            if(i == tmp1 && i == tmp2) nn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) nn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) nn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) nn->N_TN[i] += cur_flag;
        }

        nnclear(nn);
    }
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}

MY_FLT_TYPE cnn_err_train_with_noise(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
                                     MY_FLT_TYPE*** noise_level)
{
    cnnclear(cnn);

    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = outputData->LabelPtr[0].l;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag <= 0) continue;
        int n_r = inputData->ImgPtr[n].r;
        int n_c = inputData->ImgPtr[n].c;
        MY_FLT_TYPE** tmp_ImgData = (MY_FLT_TYPE**)malloc(n_r * sizeof(MY_FLT_TYPE*));
        for(int i = 0; i < n_r; i++) {
            tmp_ImgData[i] = (MY_FLT_TYPE*)malloc(n_c * sizeof(MY_FLT_TYPE));
        }
        if(pixel_index) {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i][j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        } else {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    tmp_ImgData[i][j] = inputData->ImgPtr[n].ImgData[i][j];
                }
            }
        }
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = outputData->LabelPtr[n].LabelData[0];
        for(int lab = 1; lab < len_labels; lab++) {
            if(tmp_fl_i < outputData->LabelPtr[n].LabelData[lab]) {
                tmp_fl_i = outputData->LabelPtr[n].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        int iClass = cur_lab_i;
        for(int i = 0; i < n_r; i++) {
            for(int j = 0; j < n_c; j++) {
                int cur_r;
                int cur_c;
                if(pixel_index) {
                    cur_r = pixel_index[i][j] / n_c;
                    cur_c = pixel_index[i][j] % n_c;
                } else {
                    cur_r = i;
                    cur_c = j;
                }
                tmp_ImgData[i][j] +=
                    (MY_FLT_TYPE)(gaussrand_CNN(0, noise_level[iClass][cur_r][cur_c]));
            }
        }
        cnnff(cnn, tmp_ImgData, cnn->core_mode);
        for(int i = 0; i < n_r; i++) {
            free(tmp_ImgData[i]);
        }
        free(tmp_ImgData);
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < cnn->O5->y[i])
                tmp_max = cnn->O5->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(cnn->O5->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < cnn->O5->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= cnn->O5->outputNum;
        cnn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        cnn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i] += cur_flag;
        }

        cnnclear(cnn);
    }
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}

MY_FLT_TYPE nn_err_train_with_noise(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
                                    MY_FLT_TYPE*** noise_level)
{
    nnclear(nn);

    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = outputData->LabelPtr[0].l;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag <= 0) continue;
        int n_r = inputData->ImgPtr[n].r;
        int n_c = inputData->ImgPtr[n].c;
        MY_FLT_TYPE* tmp_ImgData = (MY_FLT_TYPE*)malloc(n_r * n_c * sizeof(MY_FLT_TYPE));
        if(pixel_index) {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        } else {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[i][j];
                }
            }
        }
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = outputData->LabelPtr[n].LabelData[0];
        for(int lab = 1; lab < len_labels; lab++) {
            if(tmp_fl_i < outputData->LabelPtr[n].LabelData[lab]) {
                tmp_fl_i = outputData->LabelPtr[n].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        int iClass = cur_lab_i;
        for(int i = 0; i < n_r; i++) {
            for(int j = 0; j < n_c; j++) {
                int cur_r;
                int cur_c;
                if(pixel_index) {
                    cur_r = pixel_index[i][j] / n_c;
                    cur_c = pixel_index[i][j] % n_c;
                } else {
                    cur_r = i;
                    cur_c = j;
                }
                tmp_ImgData[i * n_c + j] +=
                    (MY_FLT_TYPE)(gaussrand_CNN(0, noise_level[iClass][cur_r][cur_c]));
            }
        }
        nnff(nn, tmp_ImgData, nn->core_mode);
        free(tmp_ImgData);
        //
        int tmp1 = vecmaxIndex(nn->O3->y, nn->O3->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, nn->O3->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < nn->O3->y[i])
                tmp_max = nn->O3->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(nn->O3->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < nn->O3->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= nn->O3->outputNum;
        nn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        nn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            nn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < nn->O3->outputNum; i++) {
            if(i == tmp1 && i == tmp2) nn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) nn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) nn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) nn->N_TN[i] += cur_flag;
        }

        nnclear(nn);
    }
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}

MY_FLT_TYPE cnn_err_train_less(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int num_selected)
{
    cnnclear(cnn);

    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = outputData->LabelPtr[0].l;
    int* vec_index = (int*)malloc(len_samples * sizeof(int));
    for(int i = 0; i < len_samples; i++) vec_index[i] = i;
    shuffle_CNN(vec_index, len_samples);
    int* vec_count = (int*)calloc(len_labels, sizeof(int));
    for(n = 0; n < len_samples; n++) {
        int cur_n = vec_index[n];
        int cur_flag = flag_samples[cur_n];
        if(cur_flag <= 0) continue;
        //
        int cur_label = vecmaxIndex(outputData->LabelPtr[cur_n].LabelData, len_labels);
        if(vec_count[cur_label] >= num_selected) {
            continue;
        } else {
            vec_count[cur_label]++;
        }
        //
        cnnff(cnn, inputData->ImgPtr[cur_n].ImgData, cnn->core_mode);
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[cur_n].LabelData, cnn->O5->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < cnn->O5->y[i])
                tmp_max = cnn->O5->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(cnn->O5->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < cnn->O5->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= cnn->O5->outputNum;
        cnn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        cnn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i] += cur_flag;
        }

        cnnclear(cnn);
    }
    //
    free(vec_index);
    free(vec_count);
    //
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}

MY_FLT_TYPE cnn_err_test(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index)
{
    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = cnn->O5->outputNum;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag != target_flag) continue;
        cur_flag = 1;
        if(pixel_index) {
            int n_r = inputData->ImgPtr[n].r;
            int n_c = inputData->ImgPtr[n].c;
            MY_FLT_TYPE** tmp_ImgData = (MY_FLT_TYPE**)malloc(n_r * sizeof(MY_FLT_TYPE*));
            for(int i = 0; i < n_r; i++) {
                tmp_ImgData[i] = (MY_FLT_TYPE*)malloc(n_c * sizeof(MY_FLT_TYPE));
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i][j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
            cnnff(cnn, tmp_ImgData, cnn->core_mode);
            for(int i = 0; i < n_r; i++) {
                free(tmp_ImgData[i]);
            }
            free(tmp_ImgData);
        } else {
            cnnff(cnn, inputData->ImgPtr[n].ImgData, cnn->core_mode);
        }
        //
        int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < cnn->O5->y[i])
                tmp_max = cnn->O5->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(cnn->O5->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < cnn->O5->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= cnn->O5->outputNum;
        cnn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        cnn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            cnn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < cnn->O5->outputNum; i++) {
            if(i == tmp1 && i == tmp2) cnn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) cnn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) cnn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) cnn->N_TN[i] += cur_flag;
        }

        cnnclear(cnn);
    }
    //
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}
MY_FLT_TYPE nn_err_test(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index)
{
    int n = 0;
    int incorrectnum = 0;  //错误预测的数目
    int len_samples = inputData->ImgNum;
    int num_samples = 0;
    int len_labels = nn->O3->outputNum;
    for(n = 0; n < len_samples; n++) {
        int cur_flag = flag_samples[n];
        if(cur_flag != target_flag) continue;
        cur_flag = 1;
        int n_r = inputData->ImgPtr[n].r;
        int n_c = inputData->ImgPtr[n].c;
        MY_FLT_TYPE* tmp_ImgData = (MY_FLT_TYPE*)malloc(n_r * n_c * sizeof(MY_FLT_TYPE));
        if(pixel_index) {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = pixel_index[i][j] / n_c;
                    int cur_c = pixel_index[i][j] % n_c;
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        } else {
            for(int i = 0; i < n_r; i++) {
                for(int j = 0; j < n_c; j++) {
                    int cur_r = i;
                    int cur_c = j;
                    tmp_ImgData[i * n_c + j] = inputData->ImgPtr[n].ImgData[cur_r][cur_c];
                }
            }
        }
        nnff(nn, tmp_ImgData, nn->core_mode);
        free(tmp_ImgData);
        //
        int tmp1 = vecmaxIndex(nn->O3->y, nn->O3->outputNum);
        int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, nn->O3->outputNum);

        MY_FLT_TYPE* tmp_prob = (MY_FLT_TYPE*)malloc(len_labels * sizeof(MY_FLT_TYPE));
        MY_FLT_TYPE sum_prob = 0;
        MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
        for(int i = 0; i < len_labels; i++) {
            if(tmp_max < nn->O3->y[i])
                tmp_max = nn->O3->y[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] = exp(nn->O3->y[i] - tmp_max);
            sum_prob += tmp_prob[i];
        }
        for(int i = 0; i < len_labels; i++) {
            tmp_prob[i] /= sum_prob;
        }
        MY_FLT_TYPE tmp_e = 0.0;
        int i;
        for(i = 0; i < nn->O3->outputNum; i++) {
            tmp_e += (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]) *
                     (tmp_prob[i] - outputData->LabelPtr[n].LabelData[i]);
        }
        tmp_e /= nn->O3->outputNum;
        nn->e_sum[tmp2] += tmp_e * cur_flag;
        free(tmp_prob);

        nn->N_sum[tmp2] += cur_flag;
        if(tmp1 != tmp2) {
            nn->N_wrong[tmp2] += cur_flag;
            incorrectnum += cur_flag;
        }
        num_samples += cur_flag;

        for(i = 0; i < nn->O3->outputNum; i++) {
            if(i == tmp1 && i == tmp2) nn->N_TP[i] += cur_flag;
            if(i == tmp1 && i != tmp2) nn->N_FP[i] += cur_flag;
            if(i != tmp1 && i == tmp2) nn->N_FN[i] += cur_flag;
            if(i != tmp1 && i != tmp2) nn->N_TN[i] += cur_flag;
        }

        nnclear(nn);
    }
    //
    if(num_samples > 0)
        return (MY_FLT_TYPE)incorrectnum / (MY_FLT_TYPE)num_samples;
    else
        return 1;
}

// 保存cnn
void savecnn(CNN* cnn, const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "wb");
    if(fp == NULL)
        printf("write file failed\n");

    int i, j, r;
    // C1的数据
    for(i = 0; i < cnn->C1->inChannels; i++)
        for(j = 0; j < cnn->C1->outChannels; j++)
            for(r = 0; r < cnn->C1->mapSize; r++)
                fwrite(cnn->C1->mapData[i][j][r], sizeof(MY_FLT_TYPE), cnn->C1->mapSize, fp);

    fwrite(cnn->C1->biasData, sizeof(MY_FLT_TYPE), cnn->C1->outChannels, fp);

    // C3网络
    for(i = 0; i < cnn->C3->inChannels; i++)
        for(j = 0; j < cnn->C3->outChannels; j++)
            for(r = 0; r < cnn->C3->mapSize; r++)
                fwrite(cnn->C3->mapData[i][j][r], sizeof(MY_FLT_TYPE), cnn->C3->mapSize, fp);

    fwrite(cnn->C3->biasData, sizeof(MY_FLT_TYPE), cnn->C3->outChannels, fp);

    // O5输出层
    for(i = 0; i < cnn->O5->outputNum; i++)
        fwrite(cnn->O5->wData[i], sizeof(MY_FLT_TYPE), cnn->O5->inputNum, fp);
    fwrite(cnn->O5->biasData, sizeof(MY_FLT_TYPE), cnn->O5->outputNum, fp);

    fclose(fp);
}
void savenn(NN* nn, const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "wb");
    if(fp == NULL)
        printf("write file failed\n");

    int i;
    //
    for(i = 0; i < nn->O1->outputNum; i++)
        fwrite(nn->O1->wData[i], sizeof(MY_FLT_TYPE), nn->O1->inputNum, fp);
    fwrite(nn->O1->biasData, sizeof(MY_FLT_TYPE), nn->O1->outputNum, fp);

    //
    for(i = 0; i < nn->O2->outputNum; i++)
        fwrite(nn->O2->wData[i], sizeof(MY_FLT_TYPE), nn->O2->inputNum, fp);
    fwrite(nn->O2->biasData, sizeof(MY_FLT_TYPE), nn->O2->outputNum, fp);

    //
    for(i = 0; i < nn->O3->outputNum; i++)
        fwrite(nn->O3->wData[i], sizeof(MY_FLT_TYPE), nn->O3->inputNum, fp);
    fwrite(nn->O3->biasData, sizeof(MY_FLT_TYPE), nn->O3->outputNum, fp);

    fclose(fp);
}
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("write file failed\n");

    int i, j, c, r;
    // C1的数据
    for(i = 0; i < cnn->C1->inChannels; i++)
        for(j = 0; j < cnn->C1->outChannels; j++)
            for(r = 0; r < cnn->C1->mapSize; r++)
                for(c = 0; c < cnn->C1->mapSize; c++) {
                    MY_FLT_TYPE* in = (MY_FLT_TYPE*)malloc(sizeof(MY_FLT_TYPE));
                    if(fread(in, sizeof(MY_FLT_TYPE), 1, fp) != 1) {
                        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                        exit(-1);
                    }
                    cnn->C1->mapData[i][j][r][c] = *in;
                }

    for(i = 0; i < cnn->C1->outChannels; i++)
        if(fread(&cnn->C1->biasData[i], sizeof(MY_FLT_TYPE), 1, fp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }

    // C3网络
    for(i = 0; i < cnn->C3->inChannels; i++)
        for(j = 0; j < cnn->C3->outChannels; j++)
            for(r = 0; r < cnn->C3->mapSize; r++)
                for(c = 0; c < cnn->C3->mapSize; c++)
                    if(fread(&cnn->C3->mapData[i][j][r][c], sizeof(MY_FLT_TYPE), 1, fp) != 1) {
                        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                        exit(-1);
                    }

    for(i = 0; i < cnn->C3->outChannels; i++)
        if(fread(&cnn->C3->biasData[i], sizeof(MY_FLT_TYPE), 1, fp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }

    // O5输出层
    for(i = 0; i < cnn->O5->outputNum; i++)
        for(j = 0; j < cnn->O5->inputNum; j++)
            if(fread(&cnn->O5->wData[i][j], sizeof(MY_FLT_TYPE), 1, fp) != 1) {
                printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                exit(-1);
            }

    for(i = 0; i < cnn->O5->outputNum; i++)
        if(fread(&cnn->O5->biasData[i], sizeof(MY_FLT_TYPE), 1, fp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }

    fclose(fp);
}

void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum)
{
    // 学习训练误差曲线
    cnn->L = (MY_FLT_TYPE*)calloc(trainNum, sizeof(MY_FLT_TYPE));
    int e;
    for(e = 0; e < opts.numepochs; e++) {
        int n = 0;
        for(n = 0; n < trainNum; n++) {
            //printf("%d\n", n);
            cnnff(cnn, inputData->ImgPtr[n].ImgData, cnn->core_mode);  // 前向传播，这里主要计算各
            cnnbp(cnn, outputData->LabelPtr[n].LabelData, cnn->core_mode); // 后向传播，这里主要计算各神经元的误差梯度

            //
            int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
            int tmp2 = vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);
            int a;
            for(a = 0; a < cnn->O5->outputNum; a++) {
                cnn->e_sum[a] += fabs(cnn->e[a]);
            }
            cnn->N_sum[tmp2]++;
            if(tmp1 != tmp2) {
                cnn->N_wrong[tmp2]++;
            }
            int i;
            for(i = 0; i < cnn->O5->outputNum; i++) {
                if(i == tmp1 && i == tmp2) cnn->N_TP[i]++;
                if(i == tmp1 && i != tmp2) cnn->N_FP[i]++;
                if(i != tmp1 && i == tmp2) cnn->N_FN[i]++;
                if(i != tmp1 && i != tmp2) cnn->N_TN[i]++;
            }

            //char* filedir = "PicTrans/CNNData/";
            //const char* filename = combine_strings(filedir, combine_strings(intTochar(n), ".cnn"));
            //savecnndata(cnn, filename, inputData->ImgPtr[n].ImgData);
            cnnapplygrads(cnn, opts, inputData->ImgPtr[n].ImgData); // 更新权重

            cnnclear(cnn);
            // 计算并保存误差能量
            MY_FLT_TYPE l = 0.0;
            //int i;
            for(i = 0; i < cnn->O5->outputNum; i++)
                l = l + cnn->e[i] * cnn->e[i];
            if(n == 0)
                cnn->L[n] = l / (MY_FLT_TYPE)2.0;
            else
                cnn->L[n] = (MY_FLT_TYPE)(cnn->L[n - 1] * 0.99 + 0.01 * l / 2.0);
        }
    }
}
void cnntrain_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum,
                       ArrLabelIndex arr_labelIndex)
{
    // 学习训练误差曲线
    cnn->L = (MY_FLT_TYPE*)calloc(trainNum, sizeof(MY_FLT_TYPE));
    int e;
    for(e = 0; e < opts.numepochs; e++) {
        int n = 0;
        for(n = 0; n < trainNum; n++) {
            int tmp_label = n % arr_labelIndex.LabelNum;
            int tmp_len = arr_labelIndex.LabelIndexPtr[tmp_label].len;
            int tmp_ind = (int)(rnd_uni_CNN(&rnd_uni_init_CNN) * tmp_len) % tmp_len;
            int cur_ind = arr_labelIndex.LabelIndexPtr[tmp_label].IndexData[tmp_ind];
            //printf("%d\n", n);
            cnnff(cnn, inputData->ImgPtr[cur_ind].ImgData, cnn->core_mode);  // 前向传播，这里主要计算各
            cnnbp(cnn, outputData->LabelPtr[cur_ind].LabelData,
                  cnn->core_mode); // 后向传播，这里主要计算各神经元的误差梯度

            //
            int tmp1 = vecmaxIndex(cnn->O5->y, cnn->O5->outputNum);
            int tmp2 = vecmaxIndex(outputData->LabelPtr[cur_ind].LabelData, cnn->O5->outputNum);
            if(tmp_label != tmp2) {
                printf("label index wrong...\n");
            }
            int a;
            for(a = 0; a < cnn->O5->outputNum; a++) {
                cnn->e_sum[a] += fabs(cnn->e[a]);
            }
            cnn->N_sum[tmp2]++;
            if(tmp1 != tmp2) {
                cnn->N_wrong[tmp2]++;
            }
            int i;
            for(i = 0; i < cnn->O5->outputNum; i++) {
                if(i == tmp1 && i == tmp2) cnn->N_TP[i]++;
                if(i == tmp1 && i != tmp2) cnn->N_FP[i]++;
                if(i != tmp1 && i == tmp2) cnn->N_FN[i]++;
                if(i != tmp1 && i != tmp2) cnn->N_TN[i]++;
            }

            //char* filedir = "PicTrans/CNNData/";
            //const char* filename = combine_strings(filedir, combine_strings(intTochar(n), ".cnn"));
            //savecnndata(cnn, filename, inputData->ImgPtr[n].ImgData);
            cnnapplygrads(cnn, opts, inputData->ImgPtr[cur_ind].ImgData); // 更新权重

            cnnclear(cnn);
            // 计算并保存误差能量
            MY_FLT_TYPE l = 0.0;
            //int i;
            for(i = 0; i < cnn->O5->outputNum; i++)
                l = l + cnn->e[i] * cnn->e[i];
            if(n == 0)
                cnn->L[n] = l / (MY_FLT_TYPE)2.0;
            else
                cnn->L[n] = (MY_FLT_TYPE)(cnn->L[n - 1] * 0.99 + 0.01 * l / 2.0);
        }
    }
}

// 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的
void cnnff(CNN* cnn, MY_FLT_TYPE** inputData, int core_mode)
{
    // int outSizeW = cnn->S2->inputWidth;
    // int outSizeH = cnn->S2->inputHeight;
    // 第一层的传播
    int i, j, r, c;
    // 第一层输出数据
    nSize mapSize = { cnn->C1->mapSize, cnn->C1->mapSize };
    nSize inSize = { cnn->C1->inputWidth, cnn->C1->inputHeight };
    nSize outSize = { cnn->S2->inputWidth, cnn->S2->inputHeight };
    for(i = 0; i < (cnn->C1->outChannels); i++) {
        for(j = 0; j < (cnn->C1->inChannels); j++) {
            if(!(int)cnn->C1->mapFlag[j][i]) continue;
            MY_FLT_TYPE** mapout = cov(cnn->C1->mapData[j][i], mapSize, inputData, inSize, valid);
            addmat(cnn->C1->v[i], cnn->C1->v[i], outSize, mapout, outSize);
            for(r = 0; r < outSize.r; r++)
                free(mapout[r]);
            free(mapout);
        }
        for(r = 0; r < outSize.r; r++)
            for(c = 0; c < outSize.c; c++) {
                switch(core_mode) {
                case CORE_SIGMA:
                    cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
                    break;
                case CORE_RELU:
                    cnn->C1->y[i][r][c] = activation_ReLu(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
                    break;
                case CORE_TANH:
                    cnn->C1->y[i][r][c] = activation_tanh(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
                    break;
                case CORE_LEAKYRELU:
                    cnn->C1->y[i][r][c] = activation_LeakyReLu(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
                    break;
                case CORE_ELU:
                    cnn->C1->y[i][r][c] = activation_ELU(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
                    break;
                default:
                    break;
                }
                //printf("%lf ", cnn->C1->y[i][r][c]);
            }
    }
    //printf("\n");

    // 第二层的输出传播S2，采样层
    outSize.c = cnn->C3->inputWidth;
    outSize.r = cnn->C3->inputHeight;
    inSize.c = cnn->S2->inputWidth;
    inSize.r = cnn->S2->inputHeight;
    for(i = 0; i < (cnn->S2->outChannels); i++) {
        if(cnn->S2->poolType == AvePool)
            avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
        else if(cnn->S2->poolType == MaxPool)
            maxPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
    }

    // 第三层输出传播,这里是全连接
    outSize.c = cnn->S4->inputWidth;
    outSize.r = cnn->S4->inputHeight;
    inSize.c = cnn->C3->inputWidth;
    inSize.r = cnn->C3->inputHeight;
    mapSize.c = cnn->C3->mapSize;
    mapSize.r = cnn->C3->mapSize;
    for(i = 0; i < (cnn->C3->outChannels); i++) {
        for(j = 0; j < (cnn->C3->inChannels); j++) {
            if(!(int)cnn->C3->mapFlag[j][i]) continue;
            MY_FLT_TYPE** mapout = cov(cnn->C3->mapData[j][i], mapSize, cnn->S2->y[j], inSize, valid);
            addmat(cnn->C3->v[i], cnn->C3->v[i], outSize, mapout, outSize);
            for(r = 0; r < outSize.r; r++)
                free(mapout[r]);
            free(mapout);
        }
        for(r = 0; r < outSize.r; r++)
            for(c = 0; c < outSize.c; c++) {
                switch(core_mode) {
                case CORE_SIGMA:
                    cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
                    break;
                case CORE_RELU:
                    cnn->C3->y[i][r][c] = activation_ReLu(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
                    break;
                case CORE_TANH:
                    cnn->C3->y[i][r][c] = activation_tanh(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
                    break;
                case CORE_LEAKYRELU:
                    cnn->C3->y[i][r][c] = activation_LeakyReLu(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
                    break;
                case CORE_ELU:
                    cnn->C3->y[i][r][c] = activation_ELU(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
                    break;
                default:
                    break;
                }
                //printf("%lf ", cnn->C3->y[i][r][c]);
            }
    }
    //printf("\n");

    // 第四层的输出传播
    inSize.c = cnn->S4->inputWidth;
    inSize.r = cnn->S4->inputHeight;
    outSize.c = (inSize.c + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    outSize.r = (inSize.r + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    for(i = 0; i < (cnn->S4->outChannels); i++) {
        if(cnn->S4->poolType == AvePool)
            avgPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
        else if(cnn->S4->poolType == MaxPool)
            maxPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
    }

    // 输出层O5的处理
    // 首先需要将前面的多维输出展开成一维向量
    MY_FLT_TYPE* O5inData = (MY_FLT_TYPE*)malloc((cnn->O5->inputNum) * sizeof(MY_FLT_TYPE));
    for(i = 0; i < (cnn->S4->outChannels); i++)
        for(r = 0; r < outSize.r; r++)
            for(c = 0; c < outSize.c; c++) {
                O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = cnn->S4->y[i][r][c];
                //printf("%lf ", cnn->S4->y[i][r][c]);
            }
    //printf("\n");

    nSize nnSize = { cnn->O5->inputNum, cnn->O5->outputNum };
    nnff_oneLayer(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->connFalg, cnn->O5->biasData, nnSize);
    for(i = 0; i < cnn->O5->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            cnn->O5->y[i] = cnn->O5->v[i] + cnn->O5->biasData[i];
            break;
        case CORE_RELU:
            cnn->O5->y[i] = cnn->O5->v[i] + cnn->O5->biasData[i];
            break;
        case CORE_TANH:
            cnn->O5->y[i] = cnn->O5->v[i] + cnn->O5->biasData[i];
            break;
        case CORE_LEAKYRELU:
            cnn->O5->y[i] = cnn->O5->v[i] + cnn->O5->biasData[i];
            break;
        case CORE_ELU:
            cnn->O5->y[i] = cnn->O5->v[i] + cnn->O5->biasData[i];
            break;
        default:
            break;
        }
        //printf("%lf ", cnn->O5->v[i]);
    }
    //printf("\n");
    free(O5inData);
}
void nnff(NN* nn, MY_FLT_TYPE* inputData, int core_mode)
{
    // int outSizeW = cnn->S2->inputWidth;
    // int outSizeH = cnn->S2->inputHeight;
    // 第一层的传播
    int i;
    // 第一层输出数据
    nSize nnSize1 = { nn->O1->inputNum, nn->O1->outputNum };
    nnff_oneLayer(nn->O1->v, inputData, nn->O1->wData, nn->O1->connFalg, nn->O1->biasData, nnSize1);
    for(i = 0; i < nn->O1->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O1->y[i] = activation_Sigma(nn->O1->v[i], nn->O1->biasData[i]);
            break;
        case CORE_RELU:
            nn->O1->y[i] = activation_ReLu(nn->O1->v[i], nn->O1->biasData[i]);
            break;
        case CORE_TANH:
            nn->O1->y[i] = activation_tanh(nn->O1->v[i], nn->O1->biasData[i]);
            break;
        case CORE_LEAKYRELU:
            nn->O1->y[i] = activation_LeakyReLu(nn->O1->v[i], nn->O1->biasData[i]);
            break;
        case CORE_ELU:
            nn->O1->y[i] = activation_ELU(nn->O1->v[i], nn->O1->biasData[i]);
            break;
        default:
            break;
        }
        //printf("%lf ", cnn->O5->v[i]);
    }

    // 第二层的输出传播S2，采样层
    nSize nnSize2 = { nn->O2->inputNum, nn->O2->outputNum };
    nnff_oneLayer(nn->O2->v, nn->O1->y, nn->O2->wData, nn->O2->connFalg, nn->O2->biasData, nnSize2);
    for(i = 0; i < nn->O2->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O2->y[i] = activation_Sigma(nn->O2->v[i], nn->O2->biasData[i]);
            break;
        case CORE_RELU:
            nn->O2->y[i] = activation_ReLu(nn->O2->v[i], nn->O2->biasData[i]);
            break;
        case CORE_TANH:
            nn->O2->y[i] = activation_tanh(nn->O2->v[i], nn->O2->biasData[i]);
            break;
        case CORE_LEAKYRELU:
            nn->O2->y[i] = activation_LeakyReLu(nn->O2->v[i], nn->O2->biasData[i]);
            break;
        case CORE_ELU:
            nn->O2->y[i] = activation_ELU(nn->O2->v[i], nn->O2->biasData[i]);
            break;
        default:
            break;
        }
        //printf("%lf ", cnn->O5->v[i]);
    }

    // 第三层输出传播,这里是全连接
    nSize nnSize3 = { nn->O3->inputNum, nn->O3->outputNum };
    nnff_oneLayer(nn->O3->v, nn->O2->y, nn->O3->wData, nn->O3->connFalg, nn->O3->biasData, nnSize3);
    for(i = 0; i < nn->O3->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O3->y[i] = nn->O3->v[i] + nn->O3->biasData[i];
            break;
        case CORE_RELU:
            nn->O3->y[i] = nn->O3->v[i] + nn->O3->biasData[i];
            break;
        case CORE_TANH:
            nn->O3->y[i] = nn->O3->v[i] + nn->O3->biasData[i];
            break;
        case CORE_LEAKYRELU:
            nn->O3->y[i] = nn->O3->v[i] + nn->O3->biasData[i];
            break;
        case CORE_ELU:
            nn->O3->y[i] = nn->O3->v[i] + nn->O3->biasData[i];
            break;
        default:
            break;
        }
        //printf("%lf ", cnn->O5->v[i]);
    }
    //printf("\n");
}

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
MY_FLT_TYPE activation_Sigma(MY_FLT_TYPE input, MY_FLT_TYPE bas) // sigma激活函数
{
    MY_FLT_TYPE temp = input + bas;
    return (MY_FLT_TYPE)1.0 / ((MY_FLT_TYPE)(1.0 + exp(-temp)));
}
MY_FLT_TYPE activation_ReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas) // sigma激活函数
{
    MY_FLT_TYPE temp = input + bas;
    return temp > 0 ? (MY_FLT_TYPE)temp : (MY_FLT_TYPE)0.0;
}
MY_FLT_TYPE activation_tanh(MY_FLT_TYPE input, MY_FLT_TYPE bas) // tanh激活函数
{
    MY_FLT_TYPE temp = input + bas;
    MY_FLT_TYPE tmp1 = exp(temp);
    MY_FLT_TYPE tmp2 = exp(-temp);
    return (tmp1 - tmp2) / (tmp1 + tmp2);
}
MY_FLT_TYPE activation_LeakyReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas) // leaky relu激活函数
{
    MY_FLT_TYPE temp = input + bas;
    return temp > 0 ? (MY_FLT_TYPE)temp : (MY_FLT_TYPE)0.01 * temp;
}
MY_FLT_TYPE activation_ELU(MY_FLT_TYPE input, MY_FLT_TYPE bas) // ELU激活函数
{
    MY_FLT_TYPE temp = input + bas;
    return temp > 0 ? (MY_FLT_TYPE)temp : (MY_FLT_TYPE)0.01 * (exp(temp) - 1);
}

void avgPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize) // 求平均值
{
    int outputW = (inputSize.c + mapSize - 1) / mapSize;
    int outputH = (inputSize.r + mapSize - 1) / mapSize;
    if(outputSize.c != outputW || outputSize.r != outputH)
        printf("ERROR: output size is wrong!!");

    int i, j, m, n;
    for(i = 0; i < outputH; i++)
        for(j = 0; j < outputW; j++) {
            MY_FLT_TYPE sum = 0.0;
            int cnt = 0;
            for(m = i * mapSize; m < i * mapSize + mapSize && m < inputSize.r; m++)
                for(n = j * mapSize; n < j * mapSize + mapSize && n < inputSize.c; n++) {
                    sum = sum + input[m][n];
                    cnt++;
                }
            output[i][j] = sum / (MY_FLT_TYPE)cnt;
        }
}

void maxPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize) // 求平均值
{
    int outputW = (inputSize.c + mapSize - 1) / mapSize;
    int outputH = (inputSize.r + mapSize - 1) / mapSize;
    if(outputSize.c != outputW || outputSize.r != outputH)
        printf("ERROR: output size is wrong!!");

    int i, j, m, n;
    for(i = 0; i < outputH; i++)
        for(j = 0; j < outputW; j++) {
            MY_FLT_TYPE tmp_max = (MY_FLT_TYPE)(-1e30);
            int cnt = 0;
            for(m = i * mapSize; m < i * mapSize + mapSize && m < inputSize.r; m++)
                for(n = j * mapSize; n < j * mapSize + mapSize && n < inputSize.c; n++) {
                    if(tmp_max < input[m][n])
                        tmp_max = input[m][n];
                }
            output[i][j] = tmp_max;
        }
}

// 单层全连接神经网络的前向传播
MY_FLT_TYPE vecMulti(MY_FLT_TYPE* vec1, MY_FLT_TYPE* vec2, MY_FLT_TYPE* vec3, int vecL)// 两向量相乘
{
    int i;
    MY_FLT_TYPE m = 0;
    for(i = 0; i < vecL; i++)
        m = m + vec1[i] * vec2[i] * (int)vec3[i];
    return m;
}

void nnff_oneLayer(MY_FLT_TYPE* output, MY_FLT_TYPE* input, MY_FLT_TYPE** wdata, MY_FLT_TYPE** connFlag, MY_FLT_TYPE* bas,
                   nSize nnSize)
{
    int w = nnSize.c;
    int h = nnSize.r;

    int i;
    for(i = 0; i < h; i++)
        output[i] = vecMulti(input, wdata[i], connFlag[i], w);// +bas[i];
}

MY_FLT_TYPE Sigma_derivation(MY_FLT_TYPE y)  // Logic激活函数的自变量微分
{
    return y * (1 - y); // 这里y是指经过激活函数的输出值，而不是自变量
}
MY_FLT_TYPE ReLu_derivation(MY_FLT_TYPE y)  // Logic激活函数的自变量微分
{
    return y > 0.0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.0; // 这里y是指经过激活函数的输出值，而不是自变量
}
MY_FLT_TYPE tanh_derivation(MY_FLT_TYPE y)  // Logic激活函数的自变量微分
{
    return (1 - y * y); // 这里y是指经过激活函数的输出值，而不是自变量
}
MY_FLT_TYPE LeakyReLu_derivation(MY_FLT_TYPE y)  // Logic激活函数的自变量微分
{
    return y > 0.0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)0.01; // 这里y是指经过激活函数的输出值，而不是自变量
}
MY_FLT_TYPE ELU_derivation(MY_FLT_TYPE y)  // Logic激活函数的自变量微分
{
    return y > 0.0 ? (MY_FLT_TYPE)1.0 : (MY_FLT_TYPE)(0.01 + y); // 这里y是指经过激活函数的输出值，而不是自变量
}

void cnnbp(CNN* cnn, MY_FLT_TYPE* outputData, int core_mode)
{
    int i, j, c, r; // 将误差保存到网络中
    for(i = 0; i < cnn->O5->outputNum; i++) {
        cnn->e[i] = cnn->O5->y[i] - outputData[i];
        //printf("%lf = %lf - %lf ", cnn->e[i], cnn->O5->y[i], outputData[i]);
    }
    //printf("\n");

    /*从后向前反向计算*/
    // 输出层O5
    for(i = 0; i < cnn->O5->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            cnn->O5->d[i] = cnn->e[i] * cnn->O5->y[i];
            break;
        case CORE_RELU:
            cnn->O5->d[i] = cnn->e[i] * cnn->O5->y[i];
            break;
        case CORE_TANH:
            cnn->O5->d[i] = cnn->e[i] * cnn->O5->y[i];
            break;
        case CORE_LEAKYRELU:
            cnn->O5->d[i] = cnn->e[i] * cnn->O5->y[i];
            break;
        case CORE_ELU:
            cnn->O5->d[i] = cnn->e[i] * cnn->O5->y[i];
            break;
        default:
            break;
        }
    }

    // S4层，传递到S4层的误差
    // 这里没有激活函数
    nSize outSize;
    outSize.c = (cnn->S4->inputWidth + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    outSize.r = (cnn->S4->inputHeight + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    for(i = 0; i < cnn->S4->outChannels; i++)
        for(r = 0; r < outSize.r; r++)
            for(c = 0; c < outSize.c; c++)
                for(j = 0; j < cnn->O5->outputNum; j++) {
                    int wInt = i * outSize.c * outSize.r + r * outSize.c + c;
                    cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
                }

    // C3层
    // 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍
    // int mapdata = cnn->S4->mapSize;
    nSize S4dSize;
    S4dSize.c = (cnn->S4->inputWidth + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    S4dSize.r = (cnn->S4->inputHeight + cnn->S4->mapSize - 1) / cnn->S4->mapSize;
    // 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
    for(i = 0; i < cnn->C3->outChannels; i++) {
        MY_FLT_TYPE** C3e = UpSample(cnn->S4->d[i], S4dSize, cnn->S4->mapSize, cnn->S4->mapSize);
        for(r = 0; r < cnn->S4->inputHeight; r++)
            for(c = 0; c < cnn->S4->inputWidth; c++) {
                switch(core_mode) {
                case CORE_SIGMA:
                    cnn->C3->d[i][r][c] = C3e[r][c] * Sigma_derivation(cnn->C3->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S4->mapSize * cnn->S4->mapSize);
                    break;
                case CORE_RELU:
                    cnn->C3->d[i][r][c] = C3e[r][c] * ReLu_derivation(cnn->C3->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S4->mapSize * cnn->S4->mapSize);
                    break;
                case CORE_TANH:
                    cnn->C3->d[i][r][c] = C3e[r][c] * tanh_derivation(cnn->C3->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S4->mapSize * cnn->S4->mapSize);
                    break;
                case CORE_LEAKYRELU:
                    cnn->C3->d[i][r][c] = C3e[r][c] * LeakyReLu_derivation(cnn->C3->y[i][r][c]) / (MY_FLT_TYPE)(
                                              cnn->S4->mapSize * cnn->S4->mapSize);
                    break;
                case CORE_ELU:
                    cnn->C3->d[i][r][c] = C3e[r][c] * ELU_derivation(cnn->C3->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S4->mapSize * cnn->S4->mapSize);
                    break;
                default:
                    break;
                }
            }
        for(r = 0; r < cnn->S4->inputHeight; r++)
            free(C3e[r]);
        free(C3e);
    }

    // S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
    // 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
    outSize.c = cnn->C3->inputWidth;
    outSize.r = cnn->C3->inputHeight;
    nSize inSize = { cnn->S4->inputWidth, cnn->S4->inputHeight };
    nSize mapSize = { cnn->C3->mapSize, cnn->C3->mapSize };
    for(i = 0; i < cnn->S2->outChannels; i++) {
        for(j = 0; j < cnn->C3->outChannels; j++) {
            MY_FLT_TYPE** corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);
            addmat(cnn->S2->d[i], cnn->S2->d[i], outSize, corr, outSize);
            for(r = 0; r < outSize.r; r++)
                free(corr[r]);
            free(corr);
        }
        /*
        for(r=0;r<cnn->C3->inputHeight;r++)
        for(c=0;c<cnn->C3->inputWidth;c++)
        // 这里本来用于采样的激活
        */
    }

    // C1层，卷积层
    // mapdata = cnn->S2->mapSize;
    nSize S2dSize;
    S2dSize.c = (cnn->S2->inputWidth + cnn->S2->mapSize - 1) / cnn->S2->mapSize;
    S2dSize.r = (cnn->S2->inputHeight + cnn->S2->mapSize - 1) / cnn->S2->mapSize;
    // 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
    for(i = 0; i < cnn->C1->outChannels; i++) {
        MY_FLT_TYPE** C1e = UpSample(cnn->S2->d[i], S2dSize, cnn->S2->mapSize, cnn->S2->mapSize);
        for(r = 0; r < cnn->S2->inputHeight; r++)
            for(c = 0; c < cnn->S2->inputWidth; c++) {
                switch(core_mode) {
                case CORE_SIGMA:
                    cnn->C1->d[i][r][c] = C1e[r][c] * Sigma_derivation(cnn->C1->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S2->mapSize * cnn->S2->mapSize);
                    break;
                case CORE_RELU:
                    cnn->C1->d[i][r][c] = C1e[r][c] * ReLu_derivation(cnn->C1->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S2->mapSize * cnn->S2->mapSize);
                    break;
                case CORE_TANH:
                    cnn->C1->d[i][r][c] = C1e[r][c] * tanh_derivation(cnn->C1->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S2->mapSize * cnn->S2->mapSize);
                    break;
                case CORE_LEAKYRELU:
                    cnn->C1->d[i][r][c] = C1e[r][c] * LeakyReLu_derivation(cnn->C1->y[i][r][c]) / (MY_FLT_TYPE)(
                                              cnn->S2->mapSize * cnn->S2->mapSize);
                    break;
                case CORE_ELU:
                    cnn->C1->d[i][r][c] = C1e[r][c] * ELU_derivation(cnn->C1->y[i][r][c]) / (MY_FLT_TYPE)(cnn->S2->mapSize * cnn->S2->mapSize);
                    break;
                default:
                    break;
                }
            }
        for(r = 0; r < cnn->S2->inputHeight; r++)
            free(C1e[r]);
        free(C1e);
    }
}
void nnbp(NN* nn, MY_FLT_TYPE* outputData, int core_mode)
{
    int i, j; // 将误差保存到网络中
    for(i = 0; i < nn->O3->outputNum; i++) {
        nn->e[i] = nn->O3->y[i] - outputData[i];
        //printf("%lf = %lf - %lf ", cnn->e[i], cnn->O5->y[i], outputData[i]);
    }
    //printf("\n");

    /*从后向前反向计算*/
    // 输出层O5
    for(i = 0; i < nn->O3->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O3->d[i] = nn->e[i] * nn->O3->y[i];
            break;
        case CORE_RELU:
            nn->O3->d[i] = nn->e[i] * nn->O3->y[i];
            break;
        case CORE_TANH:
            nn->O3->d[i] = nn->e[i] * nn->O3->y[i];
            break;
        case CORE_LEAKYRELU:
            nn->O3->d[i] = nn->e[i] * nn->O3->y[i];
            break;
        case CORE_ELU:
            nn->O3->d[i] = nn->e[i] * nn->O3->y[i];
            break;
        default:
            break;
        }
    }

    //
    for(i = 0; i < nn->O2->outputNum; i++)
        for(j = 0; j < nn->O3->outputNum; j++) {
            nn->O2->d[i] = nn->O2->d[i] + nn->O3->d[j] * nn->O3->wData[j][i];
        }

    for(i = 0; i < nn->O2->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O2->d[i] = nn->O2->d[i] * Sigma_derivation(nn->O2->y[i]);
            break;
        case CORE_RELU:
            nn->O2->d[i] = nn->O2->d[i] * ReLu_derivation(nn->O2->y[i]);
            break;
        case CORE_TANH:
            nn->O2->d[i] = nn->O2->d[i] * tanh_derivation(nn->O2->y[i]);
            break;
        case CORE_LEAKYRELU:
            nn->O2->d[i] = nn->O2->d[i] * LeakyReLu_derivation(nn->O2->y[i]);
            break;
        case CORE_ELU:
            nn->O2->d[i] = nn->O2->d[i] * ELU_derivation(nn->O2->y[i]);
            break;
        default:
            break;
        }
    }

    //
    for(i = 0; i < nn->O1->outputNum; i++)
        for(j = 0; j < nn->O2->outputNum; j++) {
            nn->O1->d[i] = nn->O1->d[i] + nn->O2->d[j] * nn->O2->wData[j][i];
        }

    for(i = 0; i < nn->O1->outputNum; i++) {
        switch(core_mode) {
        case CORE_SIGMA:
            nn->O1->d[i] = nn->O1->d[i] * Sigma_derivation(nn->O1->y[i]);
            break;
        case CORE_RELU:
            nn->O1->d[i] = nn->O1->d[i] * ReLu_derivation(nn->O1->y[i]);
            break;
        case CORE_TANH:
            nn->O1->d[i] = nn->O1->d[i] * tanh_derivation(nn->O1->y[i]);
            break;
        case CORE_LEAKYRELU:
            nn->O1->d[i] = nn->O1->d[i] * LeakyReLu_derivation(nn->O1->y[i]);
            break;
        case CORE_ELU:
            nn->O1->d[i] = nn->O1->d[i] * ELU_derivation(nn->O1->y[i]);
            break;
        default:
            break;
        }
    }

    return;
}

void cnnapplygrads(CNN* cnn, CNNOpts opts, MY_FLT_TYPE** inputData) // 更新权重
{
    // 这里存在权重的主要是卷积层和输出层
    // 更新这两个地方的权重就可以了
    int i, j, r, c;

    // C1层的权重更新
    nSize dSize = { cnn->S2->inputWidth, cnn->S2->inputHeight };
    nSize ySize = { cnn->C1->inputWidth, cnn->C1->inputHeight };
    nSize mapSize = { cnn->C1->mapSize, cnn->C1->mapSize };

    for(i = 0; i < cnn->C1->outChannels; i++) {
        for(j = 0; j < cnn->C1->inChannels; j++) {
            MY_FLT_TYPE** flipinputData = rotate180(inputData, ySize);
            MY_FLT_TYPE** C1dk = cov(cnn->C1->d[i], dSize, flipinputData, ySize, valid);
            multifactor(C1dk, C1dk, mapSize, -1 * opts.alpha);
            addmat(cnn->C1->mapData[j][i], cnn->C1->mapData[j][i], mapSize, C1dk, mapSize);
            for(r = 0; r < (dSize.r - (ySize.r - 1)); r++)
                free(C1dk[r]);
            free(C1dk);
            for(r = 0; r < ySize.r; r++)
                free(flipinputData[r]);
            free(flipinputData);
        }
        cnn->C1->biasData[i] = cnn->C1->biasData[i] - opts.alpha * summat(cnn->C1->d[i], dSize);
        //printf("%lf ", cnn->C1->d[i][0][0]);
    }
    //printf("\n");

    // C3层的权重更新
    dSize.c = cnn->S4->inputWidth;
    dSize.r = cnn->S4->inputHeight;
    ySize.c = cnn->C3->inputWidth;
    ySize.r = cnn->C3->inputHeight;
    mapSize.c = cnn->C3->mapSize;
    mapSize.r = cnn->C3->mapSize;
    for(i = 0; i < cnn->C3->outChannels; i++) {
        for(j = 0; j < cnn->C3->inChannels; j++) {
            MY_FLT_TYPE** flipinputData = rotate180(cnn->S2->y[j], ySize);
            MY_FLT_TYPE** C3dk = cov(cnn->C3->d[i], dSize, flipinputData, ySize, valid);
            multifactor(C3dk, C3dk, mapSize, (MY_FLT_TYPE)(-1.0 * opts.alpha));
            addmat(cnn->C3->mapData[j][i], cnn->C3->mapData[j][i], mapSize, C3dk, mapSize);
            for(r = 0; r < (dSize.r - (ySize.r - 1)); r++)
                free(C3dk[r]);
            free(C3dk);
            for(r = 0; r < ySize.r; r++)
                free(flipinputData[r]);
            free(flipinputData);
        }
        cnn->C3->biasData[i] = cnn->C3->biasData[i] - opts.alpha * summat(cnn->C3->d[i], dSize);
        //printf("%lf ", cnn->C3->d[i][0][0]);
    }
    //printf("\n");

    // 输出层
    // 首先需要将前面的多维输出展开成一维向量
    MY_FLT_TYPE* O5inData = (MY_FLT_TYPE*)malloc((cnn->O5->inputNum) * sizeof(MY_FLT_TYPE));
    nSize outSize = { cnn->S4->inputWidth / cnn->S4->mapSize, cnn->S4->inputHeight / cnn->S4->mapSize };
    for(i = 0; i < (cnn->S4->outChannels); i++)
        for(r = 0; r < outSize.r; r++)
            for(c = 0; c < outSize.c; c++)
                O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = cnn->S4->y[i][r][c];

    for(j = 0; j < cnn->O5->outputNum; j++) {
        for(i = 0; i < cnn->O5->inputNum; i++)
            cnn->O5->wData[j][i] = cnn->O5->wData[j][i] - opts.alpha * cnn->O5->d[j] * O5inData[i];
        cnn->O5->biasData[j] = cnn->O5->biasData[j] - opts.alpha * cnn->O5->d[j];
        //printf("%lf ", cnn->O5->d[j]);
    }
    //printf("\n");
    free(O5inData);
}
void nnapplygrads(NN* nn, NNOpts opts, MY_FLT_TYPE* inputData) // 更新权重
{
    // 这里存在权重的主要是卷积层和输出层
    // 更新这两个地方的权重就可以了
    int i, j;

    //
    for(j = 0; j < nn->O1->outputNum; j++) {
        for(i = 0; i < nn->O1->inputNum; i++)
            nn->O1->wData[j][i] = nn->O1->wData[j][i] - opts.alpha * nn->O1->d[j] * inputData[i];
        nn->O1->biasData[j] = nn->O1->biasData[j] - opts.alpha * nn->O1->d[j];
        //printf("%lf ", cnn->O5->d[j]);
    }
    //printf("\n");

    //
    for(j = 0; j < nn->O2->outputNum; j++) {
        for(i = 0; i < nn->O2->inputNum; i++)
            nn->O2->wData[j][i] = nn->O2->wData[j][i] - opts.alpha * nn->O2->d[j] * nn->O1->y[i];
        nn->O2->biasData[j] = nn->O2->biasData[j] - opts.alpha * nn->O2->d[j];
        //printf("%lf ", cnn->O5->d[j]);
    }
    //printf("\n");

    //
    for(j = 0; j < nn->O3->outputNum; j++) {
        for(i = 0; i < nn->O3->inputNum; i++)
            nn->O3->wData[j][i] = nn->O3->wData[j][i] - opts.alpha * nn->O3->d[j] * nn->O2->y[i];
        nn->O3->biasData[j] = nn->O3->biasData[j] - opts.alpha * nn->O3->d[j];
        //printf("%lf ", cnn->O5->d[j]);
    }
    //printf("\n");
}

void cnnclear(CNN* cnn)
{
    // 将神经元的部分数据清除
    int j, c, r;
    // C1网络
    for(j = 0; j < cnn->C1->outChannels; j++) {
        for(r = 0; r < cnn->S2->inputHeight; r++) {
            for(c = 0; c < cnn->S2->inputWidth; c++) {
                cnn->C1->d[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->C1->v[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->C1->y[j][r][c] = (MY_FLT_TYPE)0.0;
            }
        }
    }
    // S2网络
    for(j = 0; j < cnn->S2->outChannels; j++) {
        for(r = 0; r < cnn->C3->inputHeight; r++) {
            for(c = 0; c < cnn->C3->inputWidth; c++) {
                cnn->S2->d[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->S2->y[j][r][c] = (MY_FLT_TYPE)0.0;
            }
        }
    }
    // C3网络
    for(j = 0; j < cnn->C3->outChannels; j++) {
        for(r = 0; r < cnn->S4->inputHeight; r++) {
            for(c = 0; c < cnn->S4->inputWidth; c++) {
                cnn->C3->d[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->C3->v[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->C3->y[j][r][c] = (MY_FLT_TYPE)0.0;
            }
        }
    }
    // S4网络
    for(j = 0; j < cnn->S4->outChannels; j++) {
        for(r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++) {
            for(c = 0; c < cnn->S4->inputWidth / cnn->S4->mapSize; c++) {
                cnn->S4->d[j][r][c] = (MY_FLT_TYPE)0.0;
                cnn->S4->y[j][r][c] = (MY_FLT_TYPE)0.0;
            }
        }
    }
    // O5输出
    for(j = 0; j < cnn->O5->outputNum; j++) {
        cnn->O5->d[j] = (MY_FLT_TYPE)0.0;
        cnn->O5->v[j] = (MY_FLT_TYPE)0.0;
        cnn->O5->y[j] = (MY_FLT_TYPE)0.0;
    }
}
void nnclear(NN* nn)
{
    // 将神经元的部分数据清除
    int j;
    // O1输出
    for(j = 0; j < nn->O1->outputNum; j++) {
        nn->O1->d[j] = (MY_FLT_TYPE)0.0;
        nn->O1->v[j] = (MY_FLT_TYPE)0.0;
        nn->O1->y[j] = (MY_FLT_TYPE)0.0;
    }
    // O2输出
    for(j = 0; j < nn->O2->outputNum; j++) {
        nn->O2->d[j] = (MY_FLT_TYPE)0.0;
        nn->O2->v[j] = (MY_FLT_TYPE)0.0;
        nn->O2->y[j] = (MY_FLT_TYPE)0.0;
    }
    // O3输出
    for(j = 0; j < nn->O3->outputNum; j++) {
        nn->O3->d[j] = (MY_FLT_TYPE)0.0;
        nn->O3->v[j] = (MY_FLT_TYPE)0.0;
        nn->O3->y[j] = (MY_FLT_TYPE)0.0;
    }
}

// 这是用于测试的函数
void savecnndata(CNN* cnn, const char* filename, MY_FLT_TYPE** inputdata) // 保存CNN网络中的相关数据
{
    FILE* fp = NULL;
    fp = fopen(filename, "wb");
    if(fp == NULL)
        printf("write file failed\n");

    // C1的数据
    int i, j, r;
    // C1网络
    for(i = 0; i < cnn->C1->inputHeight; i++)
        fwrite(inputdata[i], sizeof(MY_FLT_TYPE), cnn->C1->inputWidth, fp);
    for(i = 0; i < cnn->C1->inChannels; i++)
        for(j = 0; j < cnn->C1->outChannels; j++)
            for(r = 0; r < cnn->C1->mapSize; r++)
                fwrite(cnn->C1->mapData[i][j][r], sizeof(MY_FLT_TYPE), cnn->C1->mapSize, fp);

    fwrite(cnn->C1->biasData, sizeof(MY_FLT_TYPE), cnn->C1->outChannels, fp);

    for(j = 0; j < cnn->C1->outChannels; j++) {
        for(r = 0; r < cnn->S2->inputHeight; r++) {
            fwrite(cnn->C1->v[j][r], sizeof(MY_FLT_TYPE), cnn->S2->inputWidth, fp);
        }
        for(r = 0; r < cnn->S2->inputHeight; r++) {
            fwrite(cnn->C1->d[j][r], sizeof(MY_FLT_TYPE), cnn->S2->inputWidth, fp);
        }
        for(r = 0; r < cnn->S2->inputHeight; r++) {
            fwrite(cnn->C1->y[j][r], sizeof(MY_FLT_TYPE), cnn->S2->inputWidth, fp);
        }
    }

    // S2网络
    for(j = 0; j < cnn->S2->outChannels; j++) {
        for(r = 0; r < cnn->C3->inputHeight; r++) {
            fwrite(cnn->S2->d[j][r], sizeof(MY_FLT_TYPE), cnn->C3->inputWidth, fp);
        }
        for(r = 0; r < cnn->C3->inputHeight; r++) {
            fwrite(cnn->S2->y[j][r], sizeof(MY_FLT_TYPE), cnn->C3->inputWidth, fp);
        }
    }
    // C3网络
    for(i = 0; i < cnn->C3->inChannels; i++)
        for(j = 0; j < cnn->C3->outChannels; j++)
            for(r = 0; r < cnn->C3->mapSize; r++)
                fwrite(cnn->C3->mapData[i][j][r], sizeof(MY_FLT_TYPE), cnn->C3->mapSize, fp);

    fwrite(cnn->C3->biasData, sizeof(MY_FLT_TYPE), cnn->C3->outChannels, fp);

    for(j = 0; j < cnn->C3->outChannels; j++) {
        for(r = 0; r < cnn->S4->inputHeight; r++) {
            fwrite(cnn->C3->v[j][r], sizeof(MY_FLT_TYPE), cnn->S4->inputWidth, fp);
        }
        for(r = 0; r < cnn->S4->inputHeight; r++) {
            fwrite(cnn->C3->d[j][r], sizeof(MY_FLT_TYPE), cnn->S4->inputWidth, fp);
        }
        for(r = 0; r < cnn->S4->inputHeight; r++) {
            fwrite(cnn->C3->y[j][r], sizeof(MY_FLT_TYPE), cnn->S4->inputWidth, fp);
        }
    }

    // S4网络
    for(j = 0; j < cnn->S4->outChannels; j++) {
        for(r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++) {
            fwrite(cnn->S4->d[j][r], sizeof(MY_FLT_TYPE), cnn->S4->inputWidth / cnn->S4->mapSize, fp);
        }
        for(r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++) {
            fwrite(cnn->S4->y[j][r], sizeof(MY_FLT_TYPE), cnn->S4->inputWidth / cnn->S4->mapSize, fp);
        }
    }

    // O5输出层
    for(i = 0; i < cnn->O5->outputNum; i++)
        fwrite(cnn->O5->wData[i], sizeof(MY_FLT_TYPE), cnn->O5->inputNum, fp);
    fwrite(cnn->O5->biasData, sizeof(MY_FLT_TYPE), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->v, sizeof(MY_FLT_TYPE), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->d, sizeof(MY_FLT_TYPE), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->y, sizeof(MY_FLT_TYPE), cnn->O5->outputNum, fp);

    fclose(fp);
}