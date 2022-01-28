#include "MOP_HandleData.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void handleData_init(DataSet& curData, char fname[], int handle_type)
{
    curData.numSample = 0;
    curData.numFeature = 0;
    curData.numLabel = 0;
    curData.numClass = 0;
    curData.tagClass = NULL;
    curData.Data = NULL;
    curData.max_val = NULL;
    curData.min_val = NULL;
    curData.tag_row = NULL;
    curData.tag_col = NULL;
    curData.nrow = 0;
    curData.ncol = 0;
    curData.Label = NULL;
    //
    switch(handle_type) {
    case HANDLEDATATYPE_STORE_ALL:
        break;
    default:
        printf("%s(%d): Unknown data handle type - %d, exiting...\n",
               __FILE__, __LINE__, handle_type);
        break;
    }
    char tmp_delim[] = " ,\t\r\n";
    FILE* fpt = fopen(fname, "r");
    if(fpt) {
        int max_buf_size = 1000 * 20 + 1;
        char* buf = (char*)malloc(max_buf_size * sizeof(char));
        char* p;

        curData.nrow = 0;
        curData.ncol = 0;

        while(fgets(buf, max_buf_size, fpt)) {
            if(curData.ncol == 0) {
                for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                    curData.ncol++;
                }
            } else {
                int n = 0;
                for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                    n++;
                }
                if(n != curData.ncol) {
                    printf("%s(%d): For file %s, the numbers of featurs are not consistant for all rows, exiting...\n",
                           __FILE__, __LINE__,	fname);
                    exit(-111007);
                }
            }
            curData.nrow++;
        }

        free(buf);
        fclose(fpt);
    } else {
        printf("%s(%d): Open file %s error, exiting...\n",
               __FILE__, __LINE__, fname);
        exit(-111007);
    }

    curData.numSample = curData.nrow;
    curData.numFeature = curData.ncol - 1;
    curData.numLabel = 1;
    curData.tagClass = (MY_FLT_TYPE*)malloc(curData.numSample * sizeof(MY_FLT_TYPE));
    curData.Data = (MY_FLT_TYPE**)malloc(curData.numSample * sizeof(MY_FLT_TYPE*));
    curData.Label = (MY_FLT_TYPE**)malloc(curData.numSample * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < curData.numSample; i++) {
        curData.Data[i] = (MY_FLT_TYPE*)malloc(curData.numFeature * sizeof(MY_FLT_TYPE));
        curData.Label[i] = (MY_FLT_TYPE*)malloc(curData.numLabel * sizeof(MY_FLT_TYPE));
    }
    curData.max_val = (MY_FLT_TYPE*)malloc(curData.numFeature * sizeof(MY_FLT_TYPE));
    curData.min_val = (MY_FLT_TYPE*)malloc(curData.numFeature * sizeof(MY_FLT_TYPE));
    curData.tag_row = (int*)malloc(curData.nrow * sizeof(int));
    for(int i = 0; i < curData.nrow; i++) {
        curData.tag_row[i] = TAG_HANDLE_ROW_VALID;
    }
    curData.tag_col = (int*)malloc(curData.ncol * sizeof(int));
    for(int i = 0; i < curData.ncol; i++) {
        curData.tag_col[i] = TAG_HANDLE_DATA_FEATURE;
    }
    curData.tag_col[curData.ncol - 1] = TAG_HANDLE_DATA_LABEL;

    return;
}

void handleData_read(DataSet& curData, char fname[], int handle_type)
{
    handleData_init(curData, fname, handle_type);
    //
    char tmp_delim[] = " ,\t\r\n";
    FILE* fpt = fopen(fname, "r");
    if(fpt) {
        int max_buf_size = curData.ncol * 100 + 1;
        char* buf = (char*)malloc(max_buf_size * sizeof(char));
        char* p;
        int cur_real_r = 0;
        for(int cur_r = 0; cur_r < curData.nrow; cur_r++) {
            int cur_c = 0;
            int cur_real_c = 0;
            int cur_real_l = 0;
            if(curData.tag_row && curData.tag_row[cur_r] == TAG_HANDLE_ROW_INVALID) {
                if(!fgets(buf, max_buf_size, fpt)) {
                    printf("%s(%d): No more data for row %d, exiting...\n",
                           __FILE__, __LINE__, cur_r);
                    exit(-111006);
                }
                continue;
            }
            if(curData.tag_row && curData.tag_row[cur_r] != TAG_HANDLE_ROW_VALID) {
                printf("%s(%d): Unknown tag for this row %d~tag %d, exiting...\n",
                       __FILE__, __LINE__, cur_r, curData.tag_row[cur_r]);
                exit(-111000);
            }
            if(fgets(buf, max_buf_size, fpt)) {
                if(cur_real_r >= curData.numSample) {
                    printf("%s(%d): More valid data row - (row %d)~real row %d >= (nrow %d)~nrow real %d, exiting...\n",
                           __FILE__, __LINE__, cur_r, cur_real_r, curData.nrow, curData.numSample);
                    exit(-111000);
                }
                for(p = strtok(buf, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
                    double tmp_val;
                    if(cur_c >= curData.ncol) {
                        printf("%s(%d): More data items for this row - (row %d) col %d >= ncol %d - (data %s), exiting...\n",
                               __FILE__, __LINE__, cur_r, cur_c, curData.ncol, p);
                        exit(-111000);
                    }
                    if(curData.tag_col && curData.tag_col[cur_c] != TAG_HANDLE_DATA_INVALID) {
                        if(sscanf(p, "%lf", &tmp_val) != 1) {
                            printf("%s(%d): No more data for row %d col %d, exiting...\n",
                                   __FILE__, __LINE__, cur_r, cur_c);
                            exit(-111006);
                        }
                        if(curData.tag_col[cur_c] == TAG_HANDLE_DATA_FEATURE) {
                            if(cur_real_c >= curData.numFeature) {
                                printf("%s(%d): More valid data items for (row %d~ real row %d) col %d~ real col %d >= ncol %d~ ncol real %d - (%s), exiting...\n",
                                       __FILE__, __LINE__, cur_r, cur_real_r, cur_c, cur_real_c, curData.ncol, curData.numFeature, p);
                                exit(-111000);
                            }
                            if(curData.max_val && cur_real_r == 0) curData.max_val[cur_real_c] = tmp_val;
                            else if(curData.max_val && tmp_val > curData.max_val[cur_real_c]) curData.max_val[cur_real_c] = tmp_val;
                            if(curData.min_val && cur_real_r == 0) curData.min_val[cur_real_c] = tmp_val;
                            else if(curData.min_val && tmp_val < curData.min_val[cur_real_c]) curData.min_val[cur_real_c] = tmp_val;
                            curData.Data[cur_real_r][cur_real_c] = tmp_val;
                            cur_real_c++;
                        } else if(curData.tag_col[cur_c] == TAG_HANDLE_DATA_LABEL) {
                            if(cur_real_l >= curData.numLabel) {
                                printf("%s(%d): More valid label items for (row %d~ real row %d) col %d~ real l %d >= nl %d) - (%s), exiting...\n",
                                       __FILE__, __LINE__, cur_r, cur_real_r, cur_c, cur_real_l, curData.numLabel, p);
                                exit(-111000);
                            }
                            curData.Label[cur_real_r][cur_real_l] = tmp_val;
                            cur_real_l++;
                        } else {
                            printf("%s(%d): Unknown tag for this col - %d, exiting...\n",
                                   __FILE__, __LINE__, cur_c);
                            exit(-111000);
                        }
                    }
                    if(!curData.tag_col) {
                        if(curData.max_val && cur_real_r == 0) curData.max_val[cur_real_c] = tmp_val;
                        else if(curData.max_val && tmp_val > curData.max_val[cur_real_c]) curData.max_val[cur_real_c] = tmp_val;
                        if(curData.min_val && cur_real_r == 0) curData.min_val[cur_real_c] = tmp_val;
                        else if(curData.min_val && tmp_val < curData.min_val[cur_real_c]) curData.min_val[cur_real_c] = tmp_val;
                        curData.Data[cur_real_r][cur_real_c] = tmp_val;
                        cur_real_c++;
                    }
                    cur_c++;
                }
                if(cur_c != curData.ncol) {
                    printf("%s(%d): Number of data items is not consistent for (row %d) - col %d != ncol %d), exiting...\n",
                           __FILE__, __LINE__, cur_r, cur_c, curData.ncol);
                    exit(-111005);
                }
                if(cur_real_c != curData.numFeature) {
                    printf("%s(%d): Number of valid feature items is not consistent for (row %d real row %d) - real col %d != ncol real %d, exiting...\n",
                           __FILE__, __LINE__, cur_r, cur_real_r, cur_real_c, curData.numFeature);
                    exit(-111005);
                }
                if(cur_real_l != curData.numLabel) {
                    printf("%s(%d): Number of valid label items is not consistent for (row %d real row %d) - l %d != nl %d, exiting...\n",
                           __FILE__, __LINE__, cur_r, cur_real_r, cur_real_l, curData.numLabel);
                    exit(-111005);
                }
            } else {
                printf("%s(%d): No more data, exiting...\n", __FILE__, __LINE__);
                exit(-111006);
            }
            cur_real_r++;
        }
        if(cur_real_r != curData.numSample) {
            printf("%s(%d): Number of valid feature item rows is not consistent with the para - %d != %d), exiting...\n",
                   __FILE__, __LINE__, cur_real_r, curData.numSample);
            exit(-111005);
        }
        //
        curData.numClass = 0;
        for(int i = 0; i < curData.numSample; i++) {
            MY_FLT_TYPE curTag = curData.Label[i][0];
            int tmp_flag = 0;
            for(int j = 0; j < curData.numClass; j++) {
                if(curTag == curData.tagClass[j])
                    tmp_flag = 1;
            }
            if(!tmp_flag) {
                curData.tagClass[curData.numClass++] = curTag;
            }
        }
        //
        free(buf);
        fclose(fpt);
    } else {
        printf("%s(%d): Open file %s error, exiting...\n",
               __FILE__, __LINE__, fname);
        exit(-111007);
    }
    return;
}

void handleData_save(DataSet& curData, char fname[])
{
    FILE* fpt = fopen(fname, "w");
    if(fpt) {
        for(int i = 0; i < curData.numSample; i++) {
            for(int j = 0; j < curData.numFeature; j++) {
                fprintf(fpt, "%g,", curData.Data[i][j]);
            }
            fprintf(fpt, "%g\n", curData.Label[i][0]);
        }
        fclose(fpt);
    } else {
        printf("%s(%d): Open file %s error, exiting...\n",
               __FILE__, __LINE__, fname);
        exit(-111007);
    }
    return;
}

void handleData_normalize(DataSet& curData, MY_FLT_TYPE* min_val, MY_FLT_TYPE* max_val)
{
    for(int r = 0; r < curData.numSample; r++) {
        for(int c = 0; c < curData.numFeature; c++) {
            if(max_val[c] > min_val[c])
                curData.Data[r][c] = (curData.Data[r][c] - min_val[c]) / (max_val[c] - min_val[c]);
            else
                curData.Data[r][c] = 0;
        }
    }
    return;
}

void handleData_free(DataSet& curData)
{
    curData.numSample = 0;
    curData.numFeature = 0;
    curData.numLabel = 0;
    curData.numClass = 0;
    if(curData.tagClass) free(curData.tagClass);
    curData.tagClass = NULL;
    for(int i = 0; i < curData.numSample; i++) {
        if(curData.Data) free(curData.Data[i]);
        if(curData.Label) free(curData.Label[i]);
    }
    if(curData.Data) free(curData.Data);
    if(curData.Label) free(curData.Label);
    curData.Data = NULL;
    curData.Label = NULL;
    if(curData.max_val) free(curData.max_val);
    if(curData.min_val) free(curData.min_val);
    if(curData.tag_row) free(curData.tag_row);
    if(curData.tag_col) free(curData.tag_col);
    curData.max_val = NULL;
    curData.min_val = NULL;
    curData.tag_row = NULL;
    curData.tag_col = NULL;
    curData.nrow = 0;
    curData.ncol = 0;

    return;
}
