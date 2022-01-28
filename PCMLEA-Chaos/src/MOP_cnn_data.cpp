//<<<<<<< HEAD
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "MOP_cnn_data.h"

//英特尔处理器和其他低端机用户必须翻转头字节。
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    //printf("%d - %d - %d - %d\n", (int)ch1, (int)ch2, (int)ch3, (int)ch4);
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

ImgArr read_Img_IDX_FILE(const char* filename) // 读入图像
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("open file failed\n");
    assert(fp);

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number
    fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = ReverseInt(magic_number);
    //获取训练或测试image的个数number_of_images
    if(fread((char*)&number_of_images, sizeof(number_of_images), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    number_of_images = ReverseInt(number_of_images);
    //获取训练或测试图像的高度Heigh
    if(fread((char*)&n_rows, sizeof(n_rows), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    n_rows = ReverseInt(n_rows);
    //获取训练或测试图像的宽度Width
    if(fread((char*)&n_cols, sizeof(n_cols), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    n_cols = ReverseInt(n_cols);
    //获取第i幅图像，保存到vec中
    int i, r, c;

    // 图像数组的初始化
    ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
    imgarr->ImgNum = number_of_images;
    imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));

    for(i = 0; i < number_of_images; ++i) {
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < n_rows; ++r) {
            imgarr->ImgPtr[i].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            for(c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                if(fread((char*)&temp, sizeof(temp), 1, fp) != 1) {
                    printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                    exit(-1);
                }
                imgarr->ImgPtr[i].ImgData[r][c] = temp / (MY_FLT_TYPE)255.0;
            }
        }
    }

    fclose(fp);
    return imgarr;
}

ImgArr read_Img(const char* filename) // 读入图像
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("open file failed\n");
    assert(fp);

    // int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number
    //fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    //magic_number = ReverseInt(magic_number);
    //获取训练或测试image的个数number_of_images
    if(fread((char*)&number_of_images, sizeof(number_of_images), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    //number_of_images = ReverseInt(number_of_images);
    //获取训练或测试图像的高度Heigh
    if(fread((char*)&n_rows, sizeof(n_rows), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    //n_rows = ReverseInt(n_rows);
    //获取训练或测试图像的宽度Width
    if(fread((char*)&n_cols, sizeof(n_cols), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    //n_cols = ReverseInt(n_cols);
    //获取第i幅图像，保存到vec中
    int i, r, c;

    // 图像数组的初始化
    ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
    imgarr->ImgNum = number_of_images;
    imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));

    for(i = 0; i < number_of_images; ++i) {
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < n_rows; ++r) {
            imgarr->ImgPtr[i].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            for(c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                if(fread((char*)&temp, sizeof(temp), 1, fp) != 1) {
                    printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                    exit(-1);
                }
                imgarr->ImgPtr[i].ImgData[r][c] = temp / (MY_FLT_TYPE)255.0;
            }
        }
    }

    fclose(fp);
    return imgarr;
}

ImgArr read_Img_tmp(const char* filename, int nSamp, int nCh, int tHeight, int tWidth) // 读入图像
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("open file failed\n");
    assert(fp);

    // int magic_number = 0;
    int number_of_images = nSamp;
    int n_chns = nCh;
    int n_rows = tHeight;
    int n_cols = tWidth;

    //获取第i幅图像，保存到vec中
    int i, r;

    // 图像数组的初始化
    ImgArr imgarr = (ImgArr)malloc(n_chns * sizeof(MinstImgArr));
    for(int n = 0; n < n_chns; n++) {
        imgarr[n].ImgNum = number_of_images;
        imgarr[n].ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));
    }

    for(i = 0; i < number_of_images; ++i) {
        for(int n = 0; n < n_chns; n++) {
            imgarr[n].ImgPtr[i].r = n_rows;
            imgarr[n].ImgPtr[i].c = n_cols;
            imgarr[n].ImgPtr[i].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
            for(r = 0; r < imgarr[n].ImgPtr[i].r; ++r) {
                imgarr[n].ImgPtr[i].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            }
        }
        for(int n = 0; n < n_chns * n_rows * n_cols; ++n) {
            MY_FLT_TYPE temp = 0;
            if(fscanf(fp, "%lf", &temp) != 1) {
                printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                exit(-1);
            }
            int tmp_chn = n % n_chns;
            int tmp_rc = n / n_chns;
            int tmp_r = tmp_rc / n_cols;
            int tmp_c = tmp_rc % n_cols;
            imgarr[tmp_chn].ImgPtr[i].ImgData[tmp_r][tmp_c] = temp;
        }
    }

    fclose(fp);
    return imgarr;
}

ImgArr read_Img_table(const char* filetrain, const char* filetest,
                      int num_train, int num_test, int num_feature, int num_row, int num_col, int** pixel_index)
{
    int tmp_offset = 0;
    int number_of_images = num_train + num_test;
    int n_rows = num_row;
    int n_cols = num_col;
    // 图像数组的初始化
    ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
    imgarr->ImgNum = number_of_images;
    imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));
    //
    int i, r, c;
    MY_FLT_TYPE** tmp_ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
    for(r = 0; r < n_rows; r++) {
        tmp_ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
    }
    FILE* fp = NULL;
    //
    fp = fopen(filetrain, "r");
    if(fp == NULL)
        printf("%s(%d): open file failed\n", __FILE__, __LINE__);
    assert(fp);
    for(i = tmp_offset; i < tmp_offset + num_train; ++i) {
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < n_rows; ++r) {
            imgarr->ImgPtr[i].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            for(c = 0; c < n_cols; ++c) {
                double temp;
                if(r * n_cols + c < num_feature) {
                    if(fscanf(fp, "%lf", &temp) != 1) {
                        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                        exit(-1);
                    }
                } else {
                    temp = 0.5;
                }
                tmp_ImgData[r][c] = temp;
            }
        }
        if(pixel_index) {
            for(r = 0; r < n_rows; r++) {
                for(c = 0; c < n_cols; c++) {
                    if(r * n_cols + c < num_feature) {
                        int cur_r = pixel_index[r][c] / n_cols;
                        int cur_c = pixel_index[r][c] % n_cols;
                        imgarr->ImgPtr[i].ImgData[r][c] = tmp_ImgData[cur_r][cur_c];
                    } else {
                        imgarr->ImgPtr[i].ImgData[r][c] = 0.5;
                    }
                }
            }
        } else {
            for(r = 0; r < n_rows; r++) {
                for(c = 0; c < n_cols; c++) {
                    imgarr->ImgPtr[i].ImgData[r][c] = tmp_ImgData[r][c];
                }
            }
        }
    }
    fclose(fp);
    tmp_offset += num_train;
    //
    fp = fopen(filetest, "r");
    if(fp == NULL)
        printf("%s(%d): open file failed\n", __FILE__, __LINE__);
    assert(fp);
    for(i = tmp_offset; i < tmp_offset + num_test; ++i) {
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(r = 0; r < n_rows; ++r) {
            imgarr->ImgPtr[i].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            for(c = 0; c < n_cols; ++c) {
                double temp;
                if(r * n_cols + c < num_feature) {
                    if(fscanf(fp, "%lf", &temp) != 1) {
                        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
                        exit(-1);
                    }
                } else {
                    temp = 0.5;
                }
                tmp_ImgData[r][c] = temp;
            }
        }
        if(pixel_index) {
            for(r = 0; r < n_rows; r++) {
                for(c = 0; c < n_cols; c++) {
                    if(r * n_cols + c < num_feature) {
                        int cur_r = pixel_index[r][c] / n_cols;
                        int cur_c = pixel_index[r][c] % n_cols;
                        imgarr->ImgPtr[i].ImgData[r][c] = tmp_ImgData[cur_r][cur_c];
                    } else {
                        imgarr->ImgPtr[i].ImgData[r][c] = 0.5;
                    }
                }
            }
        } else {
            for(r = 0; r < n_rows; r++) {
                for(c = 0; c < n_cols; c++) {
                    imgarr->ImgPtr[i].ImgData[r][c] = tmp_ImgData[r][c];
                }
            }
        }
    }
    fclose(fp);
    //
    for(int i = 0; i < n_rows; i++) {
        free(tmp_ImgData[i]);
    }
    free(tmp_ImgData);
    //
    return imgarr;
}

void free_Img(ImgArr arr_img)
{
    int i, j;
    for(i = 0; i < arr_img->ImgNum; i++) {
        for(j = 0; j < arr_img->ImgPtr[i].r; j++) {
            free(arr_img->ImgPtr[i].ImgData[j]);
        }
        free(arr_img->ImgPtr[i].ImgData);
    }
    free(arr_img->ImgPtr);
    free(arr_img);
}

LabelArr read_Label_IDX_FILE(const char* filename, int len_label)// 读入图像
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("open file failed\n");
    assert(fp);
    int magic_number = 0;
    int number_of_labels = 0;
    int label_long = len_label;
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number
    fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = ReverseInt(magic_number);
    //获取训练或测试image的个数number_of_images
    if(fread((char*)&number_of_labels, sizeof(number_of_labels), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    number_of_labels = ReverseInt(number_of_labels);
    //if (fread((char*)&label_long, sizeof(label_long), 1, fp) != 1) {
    //    printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
    //    exit(-1);
    //}
    //label_long = ReverseInt(label_long);
    int i;// , l;
    // 图像标记数组的初始化
    LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
    labarr->LabelNum = number_of_labels;
    labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));
    for(i = 0; i < number_of_labels; ++i) {
        labarr->LabelPtr[i].l = label_long;
        labarr->LabelPtr[i].LabelData = (MY_FLT_TYPE*)calloc(label_long, sizeof(MY_FLT_TYPE));
        unsigned char temp = 0;
        if(fread((char*)&temp, sizeof(temp), 1, fp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }
        labarr->LabelPtr[i].LabelData[(int)temp] = 1.0;
    }
    fclose(fp);
    return labarr;
}

LabelArr read_Label(const char* filename)// 读入图像
{
    FILE* fp = NULL;
    fp = fopen(filename, "rb");
    if(fp == NULL)
        printf("open file failed\n");
    assert(fp);
    // int magic_number = 0;
    int number_of_labels = 0;
    int label_long = 3;
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number
    //fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    //magic_number = ReverseInt(magic_number);
    //获取训练或测试image的个数number_of_images
    if(fread((char*)&number_of_labels, sizeof(number_of_labels), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    //number_of_labels = ReverseInt(number_of_labels);
    if(fread((char*)&label_long, sizeof(label_long), 1, fp) != 1) {
        printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
        exit(-1);
    }
    //label_long = ReverseInt(label_long);
    int i;// , l;
    // 图像标记数组的初始化
    LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
    labarr->LabelNum = number_of_labels;
    labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));
    for(i = 0; i < number_of_labels; ++i) {
        labarr->LabelPtr[i].l = label_long;
        labarr->LabelPtr[i].LabelData = (MY_FLT_TYPE*)calloc(label_long, sizeof(MY_FLT_TYPE));
        unsigned char temp = 0;
        if(fread((char*)&temp, sizeof(temp), 1, fp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }
        labarr->LabelPtr[i].LabelData[(int)temp] = 1.0;
    }
    fclose(fp);
    return labarr;
}

LabelArr read_Lable_tabel(const char* filetrain, const char* filetest,
                          int num_train, int num_test, int num_class)
{
    int tmp_offset = 0;
    int number_of_labels = num_train + num_test;
    int n_labels = num_class;
    // 图像标记数组的初始化
    LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
    labarr->LabelNum = number_of_labels;
    labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));
    //
    int i;
    FILE* fp = NULL;
    //
    fp = fopen(filetrain, "r");
    if(fp == NULL)
        printf("%s(%d): open file failed\n", __FILE__, __LINE__);
    assert(fp);
    for(i = tmp_offset; i < tmp_offset + num_train; ++i) {
        labarr->LabelPtr[i].l = n_labels;
        labarr->LabelPtr[i].LabelData = (MY_FLT_TYPE*)calloc(n_labels, sizeof(MY_FLT_TYPE));
        int temp = 0;
        if(fscanf(fp, "%d", &temp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }
        labarr->LabelPtr[i].LabelData[temp] = 1.0;
    }
    fclose(fp);
    tmp_offset += num_train;
    //
    fp = fopen(filetest, "r");
    if(fp == NULL)
        printf("%s(%d): open file failed\n", __FILE__, __LINE__);
    assert(fp);
    for(i = tmp_offset; i < tmp_offset + num_test; ++i) {
        labarr->LabelPtr[i].l = n_labels;
        labarr->LabelPtr[i].LabelData = (MY_FLT_TYPE*)calloc(n_labels, sizeof(MY_FLT_TYPE));
        int temp = 0;
        if(fscanf(fp, "%d", &temp) != 1) {
            printf("%s(%d):ERROR!! --> calloc: no more data\n", __FILE__, __LINE__);
            exit(-1);
        }
        labarr->LabelPtr[i].LabelData[temp] = 1.0;
    }
    fclose(fp);
    //
    return labarr;
}

void free_Label(LabelArr arr_label)
{
    int i;
    for(i = 0; i < arr_label->LabelNum; i++) {
        free(arr_label->LabelPtr[i].LabelData);
    }
    free(arr_label->LabelPtr);
    free(arr_label);
}

void interpolate_Img(ImgArr& imgarr, LabelArr& labarr, ImgArr allimgs, LabelArr alllabels,
                     int* num4neighbor, int maxSize_neighborhood, int* flag_train, int num_train, int* flag_class)
{
    int num_imgs = allimgs->ImgNum;
    int num_class = alllabels->LabelPtr[0].l;
    int n_rows = allimgs->ImgPtr[0].r;
    int n_cols = allimgs->ImgPtr[0].c;
    //
    MY_FLT_TYPE** all_dists = (MY_FLT_TYPE**)malloc(num_imgs * sizeof(MY_FLT_TYPE*));
    for(int i = 0; i < num_imgs; i++) {
        all_dists[i] = (MY_FLT_TYPE*)calloc(num_imgs, sizeof(MY_FLT_TYPE));
    }
    int** all_indic = (int**)malloc(num_imgs * sizeof(int*));
    for(int i = 0; i < num_imgs; i++) {
        all_indic[i] = (int*)calloc(num_imgs, sizeof(int));
    }
    int* tmp_count_samples_per_class = (int*)calloc(num_class, sizeof(int));
    //////////////////////////////////////////////////////////////////////////
    MY_FLT_TYPE max_dist_tmp = (MY_FLT_TYPE)1e30;
    for(int i = 0; i < num_imgs; i++) {
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
        for(int lab = 1; lab < num_class; lab++) {
            if(tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
                tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        for(int j = i + 1; j < num_imgs; j++) {
            int cur_lab_j = 0;
            MY_FLT_TYPE tmp_fl_j = alllabels->LabelPtr[j].LabelData[0];
            for(int lab = 1; lab < num_class; lab++) {
                if(tmp_fl_j < alllabels->LabelPtr[j].LabelData[lab]) {
                    tmp_fl_j = alllabels->LabelPtr[j].LabelData[lab];
                    cur_lab_j = lab;
                }
            }
            if(cur_lab_i != cur_lab_j ||
               flag_train[i] <= 0 ||
               flag_train[j] <= 0 ||
               !flag_class[cur_lab_i]) {
                all_dists[i][j] = all_dists[j][i] = max_dist_tmp;
                continue;
            }
            //
            MY_FLT_TYPE cur_dist = 0;
            for(int r = 0; r < n_rows; r++) {
                for(int c = 0; c < n_cols; c++) {
                    cur_dist += (allimgs->ImgPtr[i].ImgData[r][c] - allimgs->ImgPtr[j].ImgData[r][c]) *
                                (allimgs->ImgPtr[i].ImgData[r][c] - allimgs->ImgPtr[j].ImgData[r][c]);
                }
            }
            if(cur_dist == 0)
                all_dists[i][j] = all_dists[j][i] = max_dist_tmp;
            else
                all_dists[i][j] = all_dists[j][i] = cur_dist;
        }
        all_dists[i][i] = max_dist_tmp;
    }
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < num_imgs; i++) {
        for(int j = 0; j < num_imgs; j++) {
            all_indic[i][j] = j;
        }
        if(flag_train[i] <= 0)
            continue;
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
        for(int lab = 1; lab < num_class; lab++) {
            if(tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
                tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        tmp_count_samples_per_class[cur_lab_i]++;
        if(!flag_class[cur_lab_i])
            continue;
        for(int j = 0; j < maxSize_neighborhood; j++) {
            for(int k = j + 1; k < num_imgs; k++) {
                if(all_dists[i][all_indic[i][j]] > all_dists[i][all_indic[i][k]]) {
                    int tmp_int = all_indic[i][j];
                    all_indic[i][j] = all_indic[i][k];
                    all_indic[i][k] = tmp_int;
                }
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
    labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
    int max_interpolated_imgs = 0;
    for(int iClass = 0; iClass < num_class; iClass++) {
        for(int i = 0; i < maxSize_neighborhood; i++) {
            max_interpolated_imgs += tmp_count_samples_per_class[iClass] * flag_class[iClass] * num4neighbor[i];
        }
    }
    max_interpolated_imgs *= num_train;
    imgarr->ImgNum = 0;
    imgarr->ImgPtr = (MinstImg*)calloc(num_imgs + max_interpolated_imgs, sizeof(MinstImg));
    labarr->LabelNum = 0;
    labarr->LabelPtr = (MinstLabel*)calloc(num_imgs + max_interpolated_imgs, sizeof(MinstLabel));
    int tmp_count = 0;
    for(int i = 0; i < num_imgs; i++) {
        if(flag_train[i] <= 0)
            continue;
        imgarr->ImgPtr[tmp_count].r = n_rows;
        imgarr->ImgPtr[tmp_count].c = n_cols;
        imgarr->ImgPtr[tmp_count].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
        for(int r = 0; r < n_rows; ++r) {
            imgarr->ImgPtr[tmp_count].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
            memcpy(imgarr->ImgPtr[tmp_count].ImgData[r], allimgs->ImgPtr[i].ImgData[r], n_cols * sizeof(MY_FLT_TYPE));
        }
        labarr->LabelPtr[tmp_count].l = num_class;
        labarr->LabelPtr[tmp_count].LabelData = (MY_FLT_TYPE*)malloc(num_class * sizeof(MY_FLT_TYPE));
        memcpy(labarr->LabelPtr[tmp_count].LabelData, alllabels->LabelPtr[i].LabelData, num_class * sizeof(MY_FLT_TYPE));
        tmp_count++;
    }
    for(int i = 0; i < num_imgs; i++) {
        if(flag_train[i] <= 0)
            continue;
        int cur_lab_i = 0;
        MY_FLT_TYPE tmp_fl_i = alllabels->LabelPtr[i].LabelData[0];
        for(int lab = 1; lab < num_class; lab++) {
            if(tmp_fl_i < alllabels->LabelPtr[i].LabelData[lab]) {
                tmp_fl_i = alllabels->LabelPtr[i].LabelData[lab];
                cur_lab_i = lab;
            }
        }
        if(!flag_class[cur_lab_i])
            continue;
        //
        int cur_i = i;
        for(int j = 0; j < maxSize_neighborhood; j++) {
            int cur_j = all_indic[cur_i][j];
            if(cur_i == cur_j)
                continue;
            if(flag_train[cur_j] <= 0)
                continue;
            //if(cur_i == all_indic[cur_j][j] && cur_i > cur_j)
            //    continue;
            int cur_lab_j = 0;
            MY_FLT_TYPE tmp_fl_j = alllabels->LabelPtr[cur_j].LabelData[0];
            for(int lab = 1; lab < num_class; lab++) {
                if(tmp_fl_j < alllabels->LabelPtr[cur_j].LabelData[lab]) {
                    tmp_fl_j = alllabels->LabelPtr[cur_j].LabelData[lab];
                    cur_lab_j = lab;
                }
            }
            if(!flag_class[cur_lab_i])
                continue;
            int cur_n = num4neighbor[j];
            for(int k = 1; k <= cur_n; k++) {
                MY_FLT_TYPE cur_w = (MY_FLT_TYPE)(rnd_uni_CNN(&rnd_uni_init_CNN) * (1 - 1.2e-7) + 1.2e-7); // (float)(k / (cur_n + 1.0));
                imgarr->ImgPtr[tmp_count].r = n_rows;
                imgarr->ImgPtr[tmp_count].c = n_cols;
                imgarr->ImgPtr[tmp_count].ImgData = (MY_FLT_TYPE**)malloc(n_rows * sizeof(MY_FLT_TYPE*));
                for(int r = 0; r < n_rows; ++r) {
                    imgarr->ImgPtr[tmp_count].ImgData[r] = (MY_FLT_TYPE*)malloc(n_cols * sizeof(MY_FLT_TYPE));
                    for(int c = 0; c < n_cols; ++c) {
                        imgarr->ImgPtr[tmp_count].ImgData[r][c] = allimgs->ImgPtr[cur_i].ImgData[r][c] * cur_w +
                                allimgs->ImgPtr[cur_j].ImgData[r][c] * (1 - cur_w);
                    }
                }
                labarr->LabelPtr[tmp_count].l = num_class;
                labarr->LabelPtr[tmp_count].LabelData = (MY_FLT_TYPE*)malloc(num_class * sizeof(MY_FLT_TYPE));
                memcpy(labarr->LabelPtr[tmp_count].LabelData, alllabels->LabelPtr[cur_i].LabelData, num_class * sizeof(MY_FLT_TYPE));
                tmp_count++;
            }
        }
    }
    imgarr->ImgNum = tmp_count;
    labarr->LabelNum = tmp_count;
    //////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < num_imgs; i++) {
        free(all_dists[i]);
        free(all_indic[i]);
    }
    free(all_dists);
    free(all_indic);
    free(tmp_count_samples_per_class);

    return;
}

char* intTochar(int i)// 将数字转换成字符串
{
    int itemp = i;
    int w = 0;
    while(itemp >= 10) {
        itemp = itemp / 10;
        w++;
    }
    char* ptr = (char*)malloc((w + 2) * sizeof(char));
    ptr[w + 1] = '\0';
    int r; // 余数
    while(i >= 10) {
        r = i % 10;
        i = i / 10;
        ptr[w] = (char)(r + 48);
        w--;
    }
    ptr[w] = (char)(i + 48);
    return ptr;
}

char* combine_strings(char* a, char* b) // 将两个字符串相连
{
    char* ptr;
    int lena = strlen(a), lenb = strlen(b);
    int i, l = 0;
    ptr = (char*)malloc((lena + lenb + 1) * sizeof(char));
    for(i = 0; i < lena; i++)
        ptr[l++] = a[i];
    for(i = 0; i < lenb; i++)
        ptr[l++] = b[i];
    ptr[l] = '\0';
    return (ptr);
}

void save_Img(ImgArr imgarr, const char filedir[512])
{
    int img_number = imgarr->ImgNum;

    int i, r;
    for(i = 0; i < img_number; i++) {
        //const char* filename = combine_strings(filedir, combine_strings(intTochar(i), ".gray"));
        char filename[1024];
        sprintf(filename, "%s%d.gray", filedir, i);
        FILE* fp = NULL;
        fp = fopen(filename, "wb");
        if(fp == NULL)
            printf("write file failed\n");
        assert(fp);

        for(r = 0; r < imgarr->ImgPtr[i].r; r++)
            fwrite(imgarr->ImgPtr[i].ImgData[r], sizeof(MY_FLT_TYPE), imgarr->ImgPtr[i].c, fp);

        fclose(fp);
    }
}

SetArrLabelIndexPtr getLabelIndex(LabelArr* arr_Label, int len_arr)
{
    SetArrLabelIndexPtr arr_index = (SetArrLabelIndexPtr)calloc(1, sizeof(SetArrLabelIndex));
    arr_index->ArrNum = len_arr;
    arr_index->ArrLabelIndexPtr = (ArrLabelIndex*)calloc(len_arr, sizeof(ArrLabelIndex));

    int i, j;
    for(i = 0; i < len_arr; i++) {
        int tmp = arr_Label[i]->LabelPtr[0].l;
        arr_index->ArrLabelIndexPtr[i].LabelNum = tmp;
        arr_index->ArrLabelIndexPtr[i].LabelIndexPtr = (LabelIndex*)calloc(tmp, sizeof(LabelIndex));
        for(j = 0; j < tmp; j++) {
            int tmp_l = arr_Label[i]->LabelNum;
            arr_index->ArrLabelIndexPtr[i].LabelIndexPtr[j].len = 0;
            arr_index->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData =
                (int*)calloc(tmp_l, sizeof(int));
        }
    }
    int k;
    for(i = 0; i < len_arr; i++) {
        for(j = 0; j < arr_Label[i]->LabelNum; j++) {
            int tmp = arr_Label[i]->LabelPtr[j].l;
            for(k = 0; k < tmp; k++) {
                int tmp_l = (int)arr_Label[i]->LabelPtr[j].LabelData[k];
                if(tmp_l > 0.0) {
                    int tmp_i = arr_index->ArrLabelIndexPtr[i].LabelIndexPtr[k].len;
                    arr_index->ArrLabelIndexPtr[i].LabelIndexPtr[k].IndexData[tmp_i] = j;
                    arr_index->ArrLabelIndexPtr[i].LabelIndexPtr[k].len++;
                }
            }
        }
    }

    return arr_index;
}

void freeSetArrLabelIndexPtr(SetArrLabelIndexPtr dataPtr)
{
    int i, j;
    for(i = 0; i < dataPtr->ArrNum; i++) {
        for(j = 0; j < dataPtr->ArrLabelIndexPtr[i].LabelNum; j++) {
            free(dataPtr->ArrLabelIndexPtr[i].LabelIndexPtr[j].IndexData);
        }
        free(dataPtr->ArrLabelIndexPtr[i].LabelIndexPtr);
    }
    free(dataPtr->ArrLabelIndexPtr);
    free(dataPtr);
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define IM1_LeNet 2147483563
#define IM2_LeNet 2147483399
#define AM_LeNet (1.0/IM1_LeNet)
#define IMM1_LeNet (IM1_LeNet-1)
#define IA1_LeNet 40014
#define IA2_LeNet 40692
#define IQ1_LeNet 53668
#define IQ2_LeNet 52774
#define IR1_LeNet 12211
#define IR2_LeNet 3791
#define NTAB_LeNet 32
#define NDIV_LeNet (1+IMM1_LeNet/NTAB_LeNet)
#define EPS_LeNet 1.2e-7
#define RNMX_LeNet (1.0-EPS_LeNet)

//the random generator in [0,1)
double rnd_uni_CNN(long* idum)
{
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB_LeNet];
    double temp;

    if(*idum <= 0) {
        if(-(*idum) < 1) *idum = 1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB_LeNet + 7; j >= 0; j--) {
            k = (*idum) / IQ1_LeNet;
            *idum = IA1_LeNet * (*idum - k * IQ1_LeNet) - k * IR1_LeNet;
            if(*idum < 0) *idum += IM1_LeNet;
            if(j < NTAB_LeNet) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1_LeNet;
    *idum = IA1_LeNet * (*idum - k * IQ1_LeNet) - k * IR1_LeNet;
    if(*idum < 0) *idum += IM1_LeNet;
    k = idum2 / IQ2_LeNet;
    idum2 = IA2_LeNet * (idum2 - k * IQ2_LeNet) - k * IR2_LeNet;
    if(idum2 < 0) idum2 += IM2_LeNet;
    j = iy / NDIV_LeNet;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy += IMM1_LeNet;   //printf("%lf\n", AM_CLASS*iy);
    if((temp = AM_LeNet * iy) > RNMX_LeNet) return RNMX_LeNet;
    else return temp;
}/*------End of rnd_uni_CLASS()--------------------------*/
int     seed_CNN = 237;
long    rnd_uni_init_CNN = -(long)seed_CNN;
//////////////////////////////////////////////////////////////////////////
int rnd_CNN(int low, int high)
{
    int res;
    if(low >= high) {
        res = low;
    } else {
        res = low + (int)(rnd_uni_CNN(&rnd_uni_init_CNN) * (high - low + 1));
        if(res > high) {
            res = high;
        }
    }
    return (res);
}
/* Fisher–Yates shuffle algorithm */
void shuffle_CNN(int* x, int size)
{
    int i, aux, k = 0;
    for(i = size - 1; i > 0; i--) {
        /* get a value between cero and i  */
        k = rnd_CNN(0, i);
        /* exchange of values */
        aux = x[i];
        x[i] = x[k];
        x[k] = aux;
    }
    //
    return;
}
double gaussrand_CNN(double a, double b)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = rnd_uni_CNN(&rnd_uni_init_CNN);
            double U2 = rnd_uni_CNN(&rnd_uni_init_CNN);

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

//== == == =
//#include <stdlib.h>
//#include <string.h>
//#include <stdio.h>
//#include <math.h>
//#include <assert.h>
//#include "minst.h"
//
////英特尔处理器和其他低端机用户必须翻转头字节。
//int ReverseInt(int i)
//{
//  unsigned char *split = (unsigned char*)&i;
//  return ((int)split[0]) << 24 | split[1] << 16 | split[2] << 8 | split[3];
//}
//
//ImgArr read_Img(const char* filename) // 读入图像
//{
//  FILE  *fp = NULL;
//  fp = fopen(filename, "rb");
//  if (fp == NULL)
//      printf("open file failed\n");
//  assert(fp);
//
//  int magic_number = 0;
//  int number_of_images = 0;
//  int n_rows = 0;
//  int n_cols = 0;
//  //从文件中读取sizeof(magic_number) 个字符到 &magic_number
//  fread((char*)&magic_number, sizeof(magic_number), 1, fp);
//  magic_number = ReverseInt(magic_number);
//  //获取训练或测试image的个数number_of_images
//  fread((char*)&number_of_images, sizeof(number_of_images), 1, fp);
//  number_of_images = ReverseInt(number_of_images);
//  //获取训练或测试图像的高度Heigh
//  fread((char*)&n_rows, sizeof(n_rows), 1, fp);
//  n_rows = ReverseInt(n_rows);
//  //获取训练或测试图像的宽度Width
//  fread((char*)&n_cols, sizeof(n_cols), 1, fp);
//  n_cols = ReverseInt(n_cols);
//  //获取第i幅图像，保存到vec中
//  int i, r, c;
//
//  // 图像数组的初始化
//  ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
//  imgarr->ImgNum = number_of_images;
//  imgarr->ImgPtr = (MinstImg*)malloc(number_of_images*sizeof(MinstImg));
//
//  for (i = 0; i < number_of_images; ++i)
//  {
//      imgarr->ImgPtr[i].r = n_rows;
//      imgarr->ImgPtr[i].c = n_cols;
//      imgarr->ImgPtr[i].ImgData = (float**)malloc(n_rows*sizeof(float*));
//      for (r = 0; r < n_rows; ++r)
//      {
//          imgarr->ImgPtr[i].ImgData[r] = (float*)malloc(n_cols*sizeof(float));
//          for (c = 0; c < n_cols; ++c)
//          {
//              unsigned char temp = 0;
//              fread((char*)&temp, sizeof(temp), 1, fp);
//              imgarr->ImgPtr[i].ImgData[r][c] = (float)temp / 255.0;
//          }
//      }
//  }
//
//  fclose(fp);
//  return imgarr;
//}
//
//LabelArr read_Lable(const char* filename)// 读入图像
//{
//  FILE  *fp = NULL;
//  fp = fopen(filename, "rb");
//  if (fp == NULL)
//      printf("open file failed\n");
//  assert(fp);
//
//  int magic_number = 0;
//  int number_of_labels = 0;
//  int label_long = 10;
//
//  //从文件中读取sizeof(magic_number) 个字符到 &magic_number
//  fread((char*)&magic_number, sizeof(magic_number), 1, fp);
//  magic_number = ReverseInt(magic_number);
//  //获取训练或测试image的个数number_of_images
//  fread((char*)&number_of_labels, sizeof(number_of_labels), 1, fp);
//  number_of_labels = ReverseInt(number_of_labels);
//
//  int i, l;
//
//  // 图像标记数组的初始化
//  LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
//  labarr->LabelNum = number_of_labels;
//  labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));
//
//  for (i = 0; i < number_of_labels; ++i)
//  {
//      labarr->LabelPtr[i].l = 10;
//      labarr->LabelPtr[i].LabelData = (float*)calloc(label_long, sizeof(float));
//      unsigned char temp = 0;
//      fread((char*)&temp, sizeof(temp), 1, fp);
//      labarr->LabelPtr[i].LabelData[(int)temp] = 1.0;
//  }
//
//  fclose(fp);
//  return labarr;
//}
//
//char* intTochar(int i)// 将数字转换成字符串
//{
//  int itemp = i;
//  int w = 0;
//  while (itemp >= 10){
//      itemp = itemp / 10;
//      w++;
//  }
//  char* ptr = (char*)malloc((w + 2)*sizeof(char));
//  ptr[w + 1] = '\0';
//  int r; // 余数
//  while (i >= 10){
//      r = i % 10;
//      i = i / 10;
//      ptr[w] = (char)(r + 48);
//      w--;
//  }
//  ptr[w] = (char)(i + 48);
//  return ptr;
//}
//
//char * combine_strings(char *a, char *b) // 将两个字符串相连
//{
//  char *ptr;
//  int lena = strlen(a), lenb = strlen(b);
//  int i, l = 0;
//  ptr = (char *)malloc((lena + lenb + 1) * sizeof(char));
//  for (i = 0; i < lena; i++)
//      ptr[l++] = a[i];
//  for (i = 0; i < lenb; i++)
//      ptr[l++] = b[i];
//  ptr[l] = '\0';
//  return(ptr);
//}
//
//void save_Img(ImgArr imgarr, char* filedir) // 将图像数据保存成文件
//{
//  int img_number = imgarr->ImgNum;
//
//  int i, r;
//  for (i = 0; i < img_number; i++){
//      const char* filename = combine_strings(filedir, combine_strings(intTochar(i), ".gray"));
//      FILE  *fp = NULL;
//      fp = fopen(filename, "wb");
//      if (fp == NULL)
//          printf("write file failed\n");
//      assert(fp);
//
//      for (r = 0; r < imgarr->ImgPtr[i].r; r++)
//          fwrite(imgarr->ImgPtr[i].ImgData[r], sizeof(float), imgarr->ImgPtr[i].c, fp);
//
//      fclose(fp);
//  }
//  //>> >> >> > 46993395e1691dc4e415b2299b9aa0ff7fc9e094
//}
//