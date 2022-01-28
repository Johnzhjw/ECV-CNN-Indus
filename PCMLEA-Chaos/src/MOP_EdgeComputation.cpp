#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MOP_EdgeComputation.h"

#define M_PI       3.14159265358979323846   // pi

//////////////////////////////////////////////////////////////////////////
int       ES_Num_cur = 0;//边缘服务器ES的数量为39个
const int Node_Num = RSU_SET_CARDINALITY_CUR;//节点RSU的数量为239个
const int Time_Per_ES_Works = 12;  // 每个ES工作的时间，假设时间不变，用于通过功率求能耗用的
const int Trans_Speed = 500000;//传输速率500Kb/s
const int Prop_Speed = 2 * pow(10, 8);//传播速率,int占4个字节，2*pow(10,8)不会溢出
//const int Group_Size = 1500;
const int Per_Core_HandleData_Speed = 10;//每个核单位时间处理数据包的个数
const int Per_CorePower_Max = 45;//单个核的最大工作阀值
const int Per_CoreNum_ES = 5;
const int Per_CoreNum_Node = 0;
const int ES_Core_Max = 32;//边缘服务器的核的最大值
const int P_Idle = 300;//服务器在空闲状态下的功率
const int P_Full = 495;//服务器在满状态下时的功率
const int Per_Core_Price = 50;//每个处理器的价格
const double Per_ES_Max_R = 2000;//每个ES的覆盖范围半径为
const double val_penalty = 1e6;

double TTrans_Speed[Node_Num];//rj向ei传输数据得传输时延
double Tprop1[Node_Num];//rj向ei传输数据得传播时延
double Tprop2[Node_Num];//rj向云服务器的传播时延
double Tqueue[Node_Num];//rj向ei传输数据得队列时间
double Tqueue1[Node_Num];//rj向ei传输数据得队列时间  扩展
int    which[Node_Num];//RSU分配到es还是云服务器，1分配到es，0到云
//int    Per_Rsu_Mark[Node_Num];

double Per_ES_HandleData_Speed;//每个ES处理数据包的速率
int   t321 = 0;
char  self_testInstName[1024];
int   self_nobj;
int   self_nvar;
int   self_position_parameters;

int    Have_Done_The_Num_Of_RSU[Node_Num];
//int    Which_RSU_In_Group_Is_ES[Node_Num];  // 每个ES在各自的类中是第几个
int    Core_Num_Per_ES_Array[Node_Num];  // 每个ES有多少核
double Load_Rate_Per_ES_Arrary[Node_Num];  // 每一类所有负载率的和

double All_Position_and_others[Node_Num][6];
double All_Distances_between_Nodes[Node_Num][Node_Num];
double Max_Distances_between_Nodes;
int    All_indeices_for_ES[Node_Num][Node_Num];
int    All_Sizes_for_clusters_ES[Node_Num];
int    All_Centroids_for_clusters_ES[Node_Num];
int    All_Nodes_ES_index[Node_Num];
double Distance_RSU_To_Cloud_Server[Node_Num];  // 所有rsu到云服务器的距离

double Cloud_Server_Position[] = { 116.552, 39.491 }; //定义一个向量，里面是云服务器的经纬度，此程序只有一个云服务器

//int    Which_RSU_Is[Node_Num];
double Return_Attribute[Node_Num];
//int    All_Sizes_for_clusters_ES[Node_Num];
double Handle_Task_Rate_Per_ES[Node_Num];

double *Total_Load_Rate_Per_ES;
int    *Which_RSU_Is_ES;
int    *Core_Num_Per_ES;  // 这是接每个ES处理器个数的指针
int    *WhichRSu_In_Group_Is_ES;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
double Caculate_Real_Distance(double* p1, double* p2)
{
    double dlon, dlat;  // dlon express the longitude differece of the two position, so does dlat
    double a, c, r;  // express a parameter of caculate real distance

    /* Use s1[] for converting from degree to radian */
    double s1[4];
    s1[0] = (p1[Latitude] * M_PI) / 180;
    s1[1] = (p1[Longitude] * M_PI) / 180;
    s1[2] = (p2[Latitude] * M_PI) / 180;
    s1[3] = (p2[Longitude] * M_PI) / 180;

    dlat = s1[0] - s1[2];  // latitude difference between p1 and p2
    //printf("dlat: %f \n ", dlat);
    dlon = s1[1] - s1[3];  // longitude difference between p1 and p2
    //printf("dlon: %f \n", dlon);
    a = pow(sin(dlat / 2), 2) + cos(s1[0]) * cos(s1[2]) * pow(sin(dlon / 2), 2);
    c = 2 * asin(sqrt(a));
    r = 6371000;  // the radius of earth
    return c * r;  // return the real distance of two positions
}

//double Caculate_Real_Distance(double* p1, double* p2)
//{
//    double dlon, dlat;  // dlon express the longitude differece of the two position, so does dlat
//    double a, c, r;  // express a parameter of caculate real distance
//
//    /* Use s1[] for converting from degree to radian */
//    double s1[4];
//    s1[0] = (p1[0] * M_PI) / 180;
//    s1[1] = (p1[1] * M_PI) / 180;
//    s1[2] = (p2[0] * M_PI) / 180;
//    s1[3] = (p2[1] * M_PI) / 180;
//
//    dlat = s1[0] - s1[2];  // latitude difference between p1 and p2
//    //printf("dlat: %f \n ", dlat);
//    dlon = s1[1] - s1[3];  // longitude difference between p1 and p2
//    //printf("dlon: %f \n", dlon);
//    a = pow(sin(dlat / 2), 2) + cos(s1[0]) * cos(s1[2]) * pow(sin(dlon / 2), 2);
//    c = 2 * atan2(sqrt(a), sqrt(1 - a));
//    r = 6371000;  // the radius of earth
//    return c * r;  // return the real distance of two positions
//}

//double Caculate_Real_Distance(double* p1, double* p2)
//{
//    double latA, latB, lonA, lonB;
//    //
//    latA = (p1[0] * M_PI) / 180;
//    latB = (p2[0] * M_PI) / 180;
//    lonA = (p1[1] * M_PI) / 180;
//    lonB = (p2[1] * M_PI) / 180;
//    //
//    double C = sin(latA) * sin(latB) * cos(lonA - lonB) + cos(latA) * cos(latB);
//    double dist = 6371000 * acos(C);
//    //
//    return dist;
//}

void Initialize_data_EdgeComputation(int curN, int numN)
{
    char file[1024];
    FILE* ifs;
    int max_buf_size = 1000 * 20 + 1;
    char tmp_delim[] = " ,\t\r\n";
    char* buff = (char*)calloc(max_buf_size, sizeof(char));
    char* p;
    int tmp_offset = 0;
    sprintf(file, "../Data_all/Data_EdgeComputation/%dall.csv", Node_Num);
    ifs = fopen(file, "r");
    if(!ifs) {
        printf("%s(%d): Open file %s error, exiting...\n",
               __FILE__, __LINE__, file);
        exit(-111007);
    }
    if(!fgets(buff, max_buf_size, ifs)) {
        printf("%s(%d): Not enough data for file %s, exiting...\n",
               __FILE__, __LINE__, file);
        exit(-111006);
    }
    while(fgets(buff, max_buf_size, ifs)) {
        int k = 0;
        for(p = strtok(buff, tmp_delim); p; p = strtok(NULL, tmp_delim)) {
            double tmp_val;
            if(sscanf(p, "%lf", &tmp_val) != 1) {
                printf("%s(%d): No more data for row %d file %s, exiting...\n",
                       __FILE__, __LINE__, tmp_offset, file);
                exit(-111006);
            }
            if(k >= 0 && k <= 1)
                All_Position_and_others[tmp_offset][k] = tmp_val;
            if(k == 4)
                All_Position_and_others[tmp_offset][Load_Rate] = tmp_val;
            k++;
        }
        if(k <= 4) {
            printf("%s(%d): Not enough data for row %d file %s, exiting...\n",
                   __FILE__, __LINE__, tmp_offset, file);
            exit(-111006);
        }
        tmp_offset++;
    }
    if(tmp_offset != Node_Num) {
        printf("%s(%d): The number of RSUs is not consistent withe the setting (%d != %d), exiting...\n",
               __FILE__, __LINE__, tmp_offset, Node_Num);
        exit(-111006);
    }
    fclose(ifs);
    free(buff);
    //
    //sprintf(file, "../Data_all/Data_EdgeComputation/All_positions");
    //ifs = fopen(file, "w");
    //if(!ifs) {
    //    printf("%s(%d): Open file %s error, exiting...\n",
    //           __FILE__, __LINE__, file);
    //    exit(-111007);
    //}
    //for(int i = 0; i < Node_Num; i++) {
    //    for(int j = 0; j < 2; j++) {
    //        fprintf(ifs, "%lf ", All_Position_and_others[i][j]);
    //    }
    //    fprintf(ifs, "\n");
    //}
    //fclose(ifs);
    //
    Max_Distances_between_Nodes = 0;
    for(int i = 0; i < Node_Num; i++) {
        for(int j = i + 1; j < Node_Num; j++) {
            All_Distances_between_Nodes[i][j] = Caculate_Real_Distance(All_Position_and_others[i],
                                                All_Position_and_others[j]);
            All_Distances_between_Nodes[j][i] = All_Distances_between_Nodes[i][j];
            if(Max_Distances_between_Nodes < All_Distances_between_Nodes[i][j])
                Max_Distances_between_Nodes = All_Distances_between_Nodes[i][j];
        }
        All_Distances_between_Nodes[i][i] = 0;
        //
        Distance_RSU_To_Cloud_Server[i] = Caculate_Real_Distance(All_Position_and_others[i],
                                          Cloud_Server_Position);
    }
    //
    //int ccc = 0;
    //for(int i = 0; i < ES_Num_cur; i++) {
    //    for(int j = 0; j < All_Position_and_others[i].size(); j++) {
    //        Per_Rsu_Mark[ccc] = All_Position_and_others[i][j][2];
    //        ccc++;
    //    }
    //}
}

void InitPara_EdgeComputation(char* instName, int numObj, int numVar, int posPara)
{
    strcpy(self_testInstName, instName);
    self_nobj = numObj;
    self_nvar = numVar;
    self_position_parameters = posPara;

    return;
}

void Fitness_EdgeComputation(double *individual, double *fitness, double *constrainV, int nx, int M)
{
    //ifstream in("../Data_all/Data_EdgeComputation/test.txt");
    //for(int i = 0; i < All_Position.size(); i++) {
    //    for(int j = 0; j < All_Position[i].size(); j++) {
    //        in >> All_Position[i][j][2];
    //    }
    //}

    //int k = 0;
    //double temp1 = 0;//指向具体数据
    //int temp2 = 0;//指向数据的下标
    //int temp3 = 0;
    //vector<int>temp8;
    //ifstream infile;
    //infile.open("../Data_all/Data_EdgeComputation/data.txt");
    //int temp;
    //while(!infile.eof()) {
    //    infile >> temp;
    //    temp8.push_back(temp);
    //}
    //infile.close();

    //for(int i = 0; i < Node_Num; i++) {
    //    temp1 = Per_Rsu_Mark[k];
    //    temp3 = temp2 = k;
    //    for(int j = 0; j < temp8[i]; j++) {
    //        if(Per_Rsu_Mark[k + j] > temp1) {
    //            temp1 = Per_Rsu_Mark[k + j];
    //            Per_Rsu_Mark[temp3] = 0;
    //            temp3 = temp2 = k + j;
    //        } else {
    //            Per_Rsu_Mark[k + j] = 0;
    //        }
    //    }
    //    Per_Rsu_Mark[temp2] = 1;
    //    temp3 = temp1 = temp2 = 0;
    //    k += temp8[i];
    //}
    //int oo = 0;
    //for(int i = 0; i < All_Position_and_others.size(); i++) {
    //    for(int j = 0; j < All_Position_and_others[i].size(); j++) {
    //        All_Position_and_others[i][j][2] = Per_Rsu_Mark[oo];
    //        oo++;

    //    }
    //}
    //
    double val_threshold = VAR_THRESHOLD_EdgeComputation;
    //
    ES_Num_cur = 0;
    for(int i = 0; i < Node_Num; i++) {
        All_Sizes_for_clusters_ES[i] = 0;
        if(individual[i] >= val_threshold) {
            All_Centroids_for_clusters_ES[ES_Num_cur++] = i;
            All_Position_and_others[i][Mark_ES] = 1;
        } else {
            All_Position_and_others[i][Mark_ES] = 0;
        }
    }
    if(ES_Num_cur == 0) {
        double add_penalty = val_penalty * Node_Num;
        fitness[0] = add_penalty;
        fitness[1] = add_penalty;
        fitness[2] = add_penalty; // The third goal
        fitness[3] = add_penalty; // The Fourth goal
        fitness[4] = add_penalty;
        fitness[5] = add_penalty;
        return;
    }
    for(int i = 0; i < ES_Num_cur; i++) {
        int cur_cen = All_Centroids_for_clusters_ES[i];
        All_indeices_for_ES[i][0] = cur_cen;
        All_Sizes_for_clusters_ES[i] = 1;
        All_Nodes_ES_index[cur_cen] = i;
    }
    for(int i = 0; i < Node_Num; i++) {
        if(All_Position_and_others[i][Mark_ES] == 1) continue;
        int cur_ind_cen = All_Centroids_for_clusters_ES[0];
        double cur_dist = All_Distances_between_Nodes[i][cur_ind_cen];
        int ind_min = 0;
        double val_min = cur_dist;
        for(int j = 1; j < ES_Num_cur; j++) {
            cur_ind_cen = All_Centroids_for_clusters_ES[j];
            cur_dist = All_Distances_between_Nodes[i][cur_ind_cen];
            if(val_min > cur_dist) {
                ind_min = j;
                val_min = cur_dist;
            }
        }
        All_indeices_for_ES[ind_min][All_Sizes_for_clusters_ES[ind_min]] = i;
        All_Sizes_for_clusters_ES[ind_min]++;
        All_Nodes_ES_index[i] = ind_min;
    }

    for(int i = 0; i < Node_Num; i++) {
        if(All_Position_and_others[i][Mark_ES] == 1) {
            All_Position_and_others[i][Core_Num] = Per_CoreNum_ES;
        } else {
            All_Position_and_others[i][Core_Num] = Per_CoreNum_Node;
        }
    }

    for(int i = 0; i < Node_Num; i++) {
        int cur_cen_ind = All_Nodes_ES_index[i];
        int cur_cen_ind_real = All_Centroids_for_clusters_ES[cur_cen_ind];
        All_Position_and_others[i][Real_Distance] = All_Distances_between_Nodes[i][cur_cen_ind_real];
    }
    //
    for(int i = 0; i < ES_Num_cur; i++) {
        double accumulate_load_rate = 0;  // 累加负载率变量
        for(int j = 0; j < All_Sizes_for_clusters_ES[i]; j++) {
            int cur_node_ind = All_indeices_for_ES[i][j];
            accumulate_load_rate += All_Position_and_others[cur_node_ind][Load_Rate];
        }
        Load_Rate_Per_ES_Arrary[i] = accumulate_load_rate;
    }

    for(int i = 0; i < ES_Num_cur; i++) {
        int cur_cen_ind_real = All_Centroids_for_clusters_ES[i];
        Core_Num_Per_ES_Array[i] = All_Position_and_others[cur_cen_ind_real][Core_Num];
    }

    for(int i = 0; i < ES_Num_cur; ++i) {
        if(Per_CorePower_Max > Load_Rate_Per_ES_Arrary[i]) {
            Handle_Task_Rate_Per_ES[i] = 1;
        } else {
            Handle_Task_Rate_Per_ES[i] = Per_CorePower_Max / Load_Rate_Per_ES_Arrary[i];
        }
    }

    double TES[Node_Num];//数据传输到云的ES的延迟
    double TCl[Node_Num];//数据直接传输到云的RSU的延迟

    double m1[Node_Num];// (Group_Size);//定义中间变量m1
    double m2[Node_Num];// (Group_Size, 1);//定义中间变量m2
    double m3[Node_Num];// (Group_Size, 1);//定义中间变量m3
    double m4[Node_Num];// (Group_Size, 1);//定义中间变量m4
    double m5[Node_Num];// (Group_Size, 1);//定义中间变量m5
    double m6[Node_Num];// (Group_Size);//定义中间变量m6
    double m7[Node_Num];// (Group_Size, 0);//定义中间变量m7

    for(int i = 0; i < Node_Num; i++) {
        TCl[i] = 0;
        m1[i] = 0;
        m2[i] = 1;
        m3[i] = 1;
        m4[i] = 1;
        m5[i] = 1;
        m6[i] = 0;
        m7[i] = 0;
    }

    for(int i = 0; i < ES_Num_cur; ++i) {
        m1[i] = Handle_Task_Rate_Per_ES[i] * Load_Rate_Per_ES_Arrary[i] / Per_Core_HandleData_Speed;
        for(int j = 0; j < Core_Num_Per_ES_Array[i]; ++j) {
            m2[i] *= m1[i];
            m3[i] *= j + 1;
        }
        for(int j = 0; j < Core_Num_Per_ES_Array[i]; ++j) {
            m4[i] = 1;
            m5[i] = 1;
            for(int l = 0; l < j; ++l) {
                m4[i] *= m1[i];
                m5[i] *= l + 1;
            }
            m6[i] += m4[i] / m5[i];
        }
        m7[i] = m6[i];
    }
    for(int i = 0; i < ES_Num_cur; ++i) {
        if(m1[i] / Core_Num_Per_ES_Array[i] >= 1) {
            printf("%s(%d): Error value m1[%d]/Core_Num_Per_ES_Array[%d] = %lf, not smaller than 1.\n",
                   __FILE__, __LINE__, i, i, m1[i]);
        }
        Tqueue[i] = m2[i] / m3[i] / (m7[i] * (1 - m1[i] / Core_Num_Per_ES_Array[i]) + m2[i] / m3[i]); //计算排列时间
        double tmp_d = Core_Num_Per_ES_Array[i] * Per_Core_HandleData_Speed -
                       Load_Rate_Per_ES_Arrary[i] * Handle_Task_Rate_Per_ES[i];
        if(tmp_d <= 0) {
            printf("%s(%d): Error value tmp_d[%d] = %lf, not greater than 0.\n",
                   __FILE__, __LINE__, i, tmp_d);
        }
        Tqueue[i] /= tmp_d;
        if(m1[i] / Core_Num_Per_ES_Array[i] >= 1 || tmp_d <= 0) {
            printf("%s(%d): Value may be wrong Tqueue[%d] = %lf, smaller than 0.\n",
                   __FILE__, __LINE__, i, Tqueue[i]);
        }
        if(Tqueue[i] < 0.) {
            printf("%s(%d): Value may be wrong Tqueue[%d] = %lf, smaller than 0.\n",
                   __FILE__, __LINE__, i, Tqueue[i]);
        }
    }
    for(int i = 0; i < ES_Num_cur; i++) {
        for(int j = 0; j < All_Sizes_for_clusters_ES[i]; j++) {
            int cur_ind = All_indeices_for_ES[i][j];
            Tqueue1[cur_ind] = Tqueue[i];
        }
    }

    for(int i = 0; i < Node_Num; ++i) {
        TTrans_Speed[i] = All_Position_and_others[i][Load_Rate] / Trans_Speed;
    }//ES到云服务器的传输时间

    int q = 0;
    for(int i = 0; i < Node_Num; i++) {
        Tprop2[i] = Distance_RSU_To_Cloud_Server[i] / Prop_Speed;
    }//计算rsu到云服务器的传播速率及总延迟
    int t = 0;
    for(int i = 0; i < Node_Num; i++) {
        Tprop1[i] = All_Position_and_others[i][Real_Distance] / Prop_Speed;
    }

    for(int i = 0; i < Node_Num; ++i) {
        TES[i] = TTrans_Speed[i] + Tprop1[i] + Tqueue1[i];
        TCl[i] = TTrans_Speed[i] + Tprop2[i];
        if(TCl[i] > TES[i]) {
            which[i] = 1;
        } else {
            which[i] = 0;
        }
    }//RSU卸载到云服务器还是边缘服务器

    double T1 = 0;
    for(int i = 0; i < Node_Num; ++i) {
        int cur_cen = All_Nodes_ES_index[i];
        T1 += TES[i] * Handle_Task_Rate_Per_ES[cur_cen] +
              TCl[i] * (1 - Handle_Task_Rate_Per_ES[cur_cen]);
    }

    double T = T1 / Node_Num * 10; //目标一

    //////////////////////////////////////////////////////////////////////////
    double Total_Load_Rate = 0;  // Total load rate of all RSU
    double Mean_Load_Rate;  // The average load rate
    double Sum_1_Thrid_Goal = 0;  // intermediate variable Sum1
    double Sum_2_Thrid_Goal = 0;  // intermediate variable Sum2
    double Fload;
    for(int i = 0; i < ES_Num_cur; ++i) {
        Sum_1_Thrid_Goal += Load_Rate_Per_ES_Arrary[i];
    }
    Mean_Load_Rate = Sum_1_Thrid_Goal / ES_Num_cur;
    for(int i = 0; i < ES_Num_cur; ++i) {
        Sum_2_Thrid_Goal += (Load_Rate_Per_ES_Arrary[i] - Mean_Load_Rate) * (Load_Rate_Per_ES_Arrary[i] - Mean_Load_Rate);

    }
    Fload = sqrt(Sum_2_Thrid_Goal / ES_Num_cur) / Mean_Load_Rate; //目标二*/

    //////////////////////////////////////////////////////////////////////////
    double Power_Per_ES[Node_Num];  // 定义边缘服务器在t个单位时间的功耗
    double Energy_Per_ES[Node_Num];  // 定义边缘服务器在t个单位时间的能耗
    double Energy_Total = 0;
    for(int i = 0; i < ES_Num_cur; i++) {
        if(Load_Rate_Per_ES_Arrary[i] < Per_CorePower_Max * Core_Num_Per_ES_Array[i]) {
            Power_Per_ES[i] = P_Idle + (P_Full - P_Idle) *
                              Load_Rate_Per_ES_Arrary[i] / Core_Num_Per_ES_Array[i] / Per_CorePower_Max;
        } else {
            Power_Per_ES[i] = P_Full;
        }
        Energy_Per_ES[i] = Power_Per_ES[i] * Time_Per_ES_Works;
        Energy_Total += Energy_Per_ES[i];
    }
    //目标三，求所有ES的总能耗最小*/

    //////////////////////////////////////////////////////////////////////////
    int If_Covered_By_ES = 0; // Mark The RSU If Coverd by ES
    double Sum_RSU_Cover_By_ES = 0;  // Caculated the sum of RSU which is Coverd by ES
    int If_Uncovered_By_ES = 0; // Mark The RSU If not Coverd by ES

    for(int i = 0; i < ES_Num_cur; i++) { // This RSU belong to which Group
        for(int j = 0; j < All_Sizes_for_clusters_ES[i]; j++) { // This RSU is belong to which one of this group
            int cur_node = All_indeices_for_ES[i][j];
            int flag_cover = 0;
            for(int k = 0; k < ES_Num_cur; k++) { // allow this RSU is caculated with every ES
                int cur_cen = All_Centroids_for_clusters_ES[k];
                double Real_Distance_RSU_To_Every_ES = All_Distances_between_Nodes[cur_node][cur_cen];  // 计算距离
                if(Real_Distance_RSU_To_Every_ES <= Per_ES_Max_R) { // If the RSU is covered
                    If_Covered_By_ES += 1;
                    flag_cover++;
                }
            }
            if(flag_cover == 0) {
                If_Uncovered_By_ES++;
            }
        }
    }
    // 目标四，让所有RSU被ES覆盖的总数最多

    //////////////////////////////////////////////////////////////////////////
    double distance_sum = 0;
    for(int i = 0; i < ES_Num_cur; i++) { // This RSU belong to which Group
        for(int j = 0; j < All_Sizes_for_clusters_ES[i]; j++) { // This RSU is belong to which one of this group
            int cur_node = All_indeices_for_ES[i][j];
            for(int k = 0; k < ES_Num_cur; k++) { // allow this RSU is caculated with every ES
                int cur_cen = All_Centroids_for_clusters_ES[k];
                double Real_Distance_RSU_To_Every_ES = All_Distances_between_Nodes[cur_node][cur_cen];  // 计算距离
                distance_sum += Real_Distance_RSU_To_Every_ES;
            }
        }
    }
    //目标五

    //////////////////////////////////////////////////////////////////////////
    double add_penalty = If_Uncovered_By_ES * val_penalty;
    fitness[0] = T + add_penalty;
    fitness[1] = Fload + add_penalty;
    fitness[2] = (Energy_Total / ES_Num_cur / Time_Per_ES_Works - P_Idle) / (P_Full - P_Idle) + add_penalty; // The third goal
    fitness[3] = 1 - (double)If_Covered_By_ES / Node_Num / Node_Num + add_penalty; // The Fourth goal
    fitness[4] = distance_sum / Max_Distances_between_Nodes / Node_Num / ES_Num_cur + add_penalty;
    fitness[5] = (double)ES_Num_cur / Node_Num + add_penalty;
    //
    return;
}

void SetLimits_EdgeComputation(double *minLimit, double *maxLimit, int dim)
{
    for(int i = 0; i < dim; i++) {
        minLimit[i] = 0.0;
        maxLimit[i] = 2.0 - 1e-6;
    }
}

int CheckLimits_EdgeComputation(double* Per_Rsu_Mark, int nx)
{
    for(int i = 0; i < Node_Num; i++) {
        if(Per_Rsu_Mark[i] < 0.0 || Per_Rsu_Mark[i] > 2.0 - 1e-6) {
            printf("error");
            return false;
        }
    }
    return true;
}

void Finalize_EdgeComputation()
{
    return;
}
