#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_cnn_mat.h"
#include "MOP_cnn_data.h"
#include "MOP_NN_FLT_TYPE.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

// 卷积层
typedef struct convolutional_layer_CNN {
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	// 关于特征模板的权重分布，这里是一个四维数组
	// 其大小为inChannels*outChannels*mapSize*mapSize大小
	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	// 这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
	MY_FLT_TYPE**** mapData;     //存放特征模块的数据
	MY_FLT_TYPE**** dmapData;    //存放特征模块的数据的局部梯度
	MY_FLT_TYPE** mapFlag;

	MY_FLT_TYPE* biasData;   //偏置，偏置的大小，为outChannels
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	MY_FLT_TYPE*** v; // 进入激活函数的输入值
	MY_FLT_TYPE*** y; // 激活函数后神经元的输出

	// 输出像素的局部梯度
	MY_FLT_TYPE*** d; // 网络的局部梯度,δ值
} CovLayer_CNN;

// 采样层 pooling
typedef struct pooling_layer_CNN {
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	MY_FLT_TYPE* biasData;   //偏置

	MY_FLT_TYPE*** y; // 采样函数后神经元的输出,无激活函数
	MY_FLT_TYPE*** d; // 网络的局部梯度,δ值
} PoolLayer_CNN;

// 输出层 全连接的神经网络
typedef struct nn_layer_CNN {
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	MY_FLT_TYPE** wData; // 权重数据，为一个inputNum*outputNum大小
	MY_FLT_TYPE* biasData;   //偏置，大小为outputNum大小
	MY_FLT_TYPE** connFalg;

	// 下面三者的大小同输出的维度相同
	MY_FLT_TYPE* v; // 进入激活函数的输入值
	MY_FLT_TYPE* y; // 激活函数后神经元的输出
	MY_FLT_TYPE* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
} OutLayer_CNN;

typedef struct cnn_network {
	int layerNum;
	CovLayer_CNN* C1;
	PoolLayer_CNN* S2;
	CovLayer_CNN* C3;
	PoolLayer_CNN* S4;
	OutLayer_CNN* O5;

	MY_FLT_TYPE* e; // 训练误差
	MY_FLT_TYPE* L; // 瞬时误差能量

	MY_FLT_TYPE* N_sum;
	MY_FLT_TYPE* N_wrong;
	MY_FLT_TYPE* e_sum;

	MY_FLT_TYPE* N_TP;
	MY_FLT_TYPE* N_TN;
	MY_FLT_TYPE* N_FP;
	MY_FLT_TYPE* N_FN;

	int core_mode;
} CNN;

typedef struct nn_network {
	int layerNum;
	OutLayer_CNN* O1;
	OutLayer_CNN* O2;
	OutLayer_CNN* O3;

	MY_FLT_TYPE* e; // 训练误差
	MY_FLT_TYPE* L; // 瞬时误差能量

	MY_FLT_TYPE* N_sum;
	MY_FLT_TYPE* N_wrong;
	MY_FLT_TYPE* e_sum;

	MY_FLT_TYPE* N_TP;
	MY_FLT_TYPE* N_TN;
	MY_FLT_TYPE* N_FP;
	MY_FLT_TYPE* N_FN;

	int core_mode;
} NN;

typedef struct train_opts {
	int numepochs; // 训练的迭代次数
	int batchSize;
	MY_FLT_TYPE alpha; // 学习速率
} CNNOpts;

typedef struct nn_train_opts {
	int numepochs; // 训练的迭代次数
	int batchSize;
	MY_FLT_TYPE alpha; // 学习速率
} NNOpts;

enum CNN_INIT_MODE {
	INIT_MODE,
	ASSIGN_MODE,
	OUTPUT_MODE
};

void cnnsetup(CNN* cnn, nSize inputSize, int outputSize);
void cnnfree(CNN* cnn);
void cnninit(CNN* cnn, double* x, int mode);
void nnsetup(NN* nn, nSize inputSize, int outputSize);
void nnfree(NN* nn);
void nninit(NN* nn, double* x, int mode);
/*
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
	*/
void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum);
void cnntrain_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum,
	ArrLabelIndex arr_labelIndex);
// 测试cnn函数
MY_FLT_TYPE cnntest(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum);
MY_FLT_TYPE cnntest_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum, int* vec_index);
//
void cnn_train_bp(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int* indx_samples, int num_samples);
void nn_train_bp(NN* nn, ImgArr inputData, LabelArr outputData, NNOpts opts, int* indx_samples, int num_samples);
MY_FLT_TYPE cnn_err_train(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index);
MY_FLT_TYPE nn_err_train(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index);
MY_FLT_TYPE cnn_err_train_with_noise(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
	MY_FLT_TYPE*** noise_level);
MY_FLT_TYPE nn_err_train_with_noise(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
	MY_FLT_TYPE*** noise_level);
MY_FLT_TYPE cnn_err_train_less(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int num_selected);
MY_FLT_TYPE cnn_err_test(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index);
MY_FLT_TYPE nn_err_test(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index);
// 保存cnn
void savecnn(CNN* cnn, const char* filename);
void savenn(NN* nn, const char* filename);
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename);

// 初始化卷积层
CovLayer_CNN* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void CovLayerConnect(CovLayer_CNN* covL, bool* connectModel);
void freeCovLayer(CovLayer_CNN* layer_C);
// 初始化采样层
PoolLayer_CNN* initPoolLayer(int inputWidth, int inputHeigh, int mapSize, int inChannels, int outChannels, int poolType);
void PoolLayerConnect(PoolLayer_CNN* poolL, bool* connectModel);
void freePoolLayer(PoolLayer_CNN* layer_S);
// 初始化输出层
OutLayer_CNN* initOutLayer(int inputNum, int outputNum);
void freeOutLayer(OutLayer_CNN* layer_O);

enum CORE_MODE {
	CORE_SIGMA,
	CORE_RELU,
	CORE_TANH,
	CORE_LEAKYRELU,
	CORE_ELU
};

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
MY_FLT_TYPE activation_Sigma(MY_FLT_TYPE input, MY_FLT_TYPE bas); // sigma激活函数
MY_FLT_TYPE activation_ReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas);
MY_FLT_TYPE activation_tanh(MY_FLT_TYPE input, MY_FLT_TYPE bas);
MY_FLT_TYPE activation_LeakyReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas); //
MY_FLT_TYPE activation_ELU(MY_FLT_TYPE input, MY_FLT_TYPE bas); //

void cnnff(CNN* cnn, MY_FLT_TYPE** inputData, int core_mode); // 网络的前向传播
void nnff(NN* nn, MY_FLT_TYPE* inputData, int core_mode); // 网络的前向传播
void cnnbp(CNN* cnn, MY_FLT_TYPE* outputData, int core_mode); // 网络的后向传播
void nnbp(NN* nn, MY_FLT_TYPE* outputData, int core_mode); // 网络的后向传播
void cnnapplygrads(CNN* cnn, CNNOpts opts, MY_FLT_TYPE** inputData);
void nnapplygrads(NN* nn, NNOpts opts, MY_FLT_TYPE* inputData);
void cnnclear(CNN* cnn); // 将数据vyd清零
void nnclear(NN* nn); // 将数据vyd清零

int vecmaxIndex(MY_FLT_TYPE* vec, int veclength);// 返回向量最大数的序号

/*
	Pooling Function
	input 输入数据
	inputNum 输入数据数目
	mapSize 求平均的模块区域
	*/
void avgPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize); // 求平均值
void maxPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize); // 求最大值

/*
	单层全连接神经网络的处理
	nnSize是网络的大小
	*/
void nnff_oneLayer(MY_FLT_TYPE* output, MY_FLT_TYPE* input, MY_FLT_TYPE** wdata, MY_FLT_TYPE** connFlag, MY_FLT_TYPE* bas,
	nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn, const char* filename, MY_FLT_TYPE** inputdata); // 保存CNN网络中的相关数据

#endif
