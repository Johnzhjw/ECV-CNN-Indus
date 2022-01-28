#ifndef __MOP_HANDLEDATA_H__
#define __MOP_HANDLEDATA_H__

#include "MOP_NN_FLT_TYPE.h"

enum HANDLEDATATYPE {
    HANDLEDATATYPE_STORE_ALL
};

typedef struct DataSet {
    int numSample;           // sample number
    int numFeature;          // feature number
    int numLabel;            // label number
    int numClass;            // label length
    MY_FLT_TYPE* tagClass;
    MY_FLT_TYPE** Data; //
    MY_FLT_TYPE* max_val;
    MY_FLT_TYPE* min_val;
    int* tag_row;
    int* tag_col;
    int nrow;
    int ncol;
    MY_FLT_TYPE** Label; //
} DataSet;

enum TAG_ROW_HANDLE_DATA {
    TAG_HANDLE_ROW_INVALID,
    TAG_HANDLE_ROW_VALID
};

enum TAG_FEATURE_HANDLE_DATA {
    TAG_HANDLE_DATA_INVALID,
    TAG_HANDLE_DATA_FEATURE,
    TAG_HANDLE_DATA_LABEL
};

void handleData_init(DataSet& curData, char fname[], int handle_type);
void handleData_read(DataSet& curData, char fname[], int handle_type);
void handleData_save(DataSet& curData, char fname[]);
void handleData_normalize(DataSet& curData, MY_FLT_TYPE* min_val, MY_FLT_TYPE* max_val);
void handleData_free(DataSet& curData);

#endif
