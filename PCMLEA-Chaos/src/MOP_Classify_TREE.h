#ifndef _MOP_CLASSIFY_TREE_
#define _MOP_CLASSIFY_TREE_

//////////////////////////////////////////////////////////////////////////
//Classify_TREE
#define NOBJ_CLASSIFY_TREE 2

extern int m_N_CLASS;
extern int m_N_MODEL;
extern int m_N_FEATURE;
extern int m_DIM_ClassifierTreeFunc;
extern double** mx_weights;

typedef  struct BiTNode {
	int* featureSubset;
	int* lLabelSet;
	int* rLabelSet;
	int  N_LABEL_left;
	int  N_LABEL_right;
	int  numFeature;
	int  model;
	int  label;
	int  level;
	struct BiTNode* parent, * lchild, * rchild;
	BiTNode() : featureSubset(NULL), lLabelSet(NULL), rLabelSet(NULL), N_LABEL_left(0),
		N_LABEL_right(0), numFeature(0), model(-1), label(-1), level(-1), parent(NULL), lchild(NULL),
		rchild(NULL)
	{
		//no other stuff
	}
} BiTNode, * BiTree;

void   f_Initialize_ClassifierTreeFunc(char prob[], int curN, int numN);
void   f_SetLimits_ClassifierTreeFunc(double* minLimit, double* maxLimit, int nx);
BiTree f_treeDecoding(int* codedTree);
int    f_freeBiTree(BiTree& T);
void   f_Fitness_ClassifierTreeFunc(double* individual, double* fitness, double* constrainV, int nx,
	int M);
int    f_CheckLimits_ClassifierTreeFunc(double* x, int nx);
void   f_freeMemoryTreeCLASS();
void   f_filter_ReliefF();
void   f_testAccuracy(double* individual, double* fitness);

#endif
