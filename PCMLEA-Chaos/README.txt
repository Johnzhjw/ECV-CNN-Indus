DPCCMOEA is implemented using MPI parallel programming 
C and C++ are used

///////////////////////////////////////////////////////////////////
Input folders and files:
../Data_all/pareto_fronts	---	数据文件夹，包括各种Pareto最优前沿，用于计算IGD指标值，在运行过程中，每隔一定代数会进行计算
DATA_alg/SamplePoint		--- 用于数据初始化
DATA_alg/Weight		    	--- 用于数据初始化
strct_global_paras.testInstance.txt            --- 用于测试的MOP的名称及一些参数：变量数、目标数、其他（可选）

///////////////////////////////////////////////////////////////////
Output folders and files:
Group			--- 分组信息
OUTPUT			--- *.csv, 包含HV和IGD指标值，及优化耗时
PF				--- 算法得到的Pareto前沿
PS				--- 所得到的Pareto前沿所对应的解PS

///////////////////////////////////////////////////////////////////
Source files:
src/*.cpp *.h	--- 所有源码

src/CC_* 					--- 合作协同演化
src/control_* 				--- 主控制函数
src/global.h 				--- 全局头文件，所有变量与函数
src/global.cpp				--- 所有变量
src/control_main.cpp		--- 主循环
src/main.cpp				--- main函数
src/grouping_* 				--- 用于分组
src/indicator_* 			--- 用于计算指标值 - IGD 和 HV
src/MOP_* 					--- 各种测试函数 - DTLZ、WFG、LSMOP、MaOP、UF等基准测试集，及实际测试问题
src/MPI_* 					--- MPI 并行算法实现
src/optimizer_* 			--- 算子 - DE、GA、PSO
src/utility_* 				--- 工具函数

///////////////////////////////////////////////////////////////////
DATA_alg/para_file		--- select algorithm framework: DPCCMOLSEA or DPCCMOLSIA
							select test instances for testing --- refer ``strct_global_paras.testInstance.txt''
							optimizer selection
							quantum on or off
							QPSO --- quantum-enhanced PSO

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
Operating environment:
			Linux --- C/C++ with MPI
Compiling: 
			Makefile			--- make
The programme is only suitable for MPI environment.

Run:
			mpirun -np #CPUs ./app/DPNeuEvo.1.0.0

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
用make编译
用
	mpirun -np #cores ./app/DPNeuEvo.1.0.0
	或
	sh ./DPNeuEvo.sh
	运行 
#cores 替换为 所用CPU核的数目，即MPI进程数
