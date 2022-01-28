# include "global.h"

#define EPS (1.2e-7)

int isTheSame(double* obj1, double* obj2)
{
    int tag = 1;
    for(int i = 0; i < st_global_p.nObj; i++) {
        if(obj1[i] < obj2[i] || obj1[i] > obj2[i])
            tag = 0;
    }
    return tag;
}

int dominanceComparator(double* obj1, double* obj2)
{
    int i;

    /* Indicates if some objetive in solution 1 dominates the objetive in solution 2 */
    int dominates1 = 0;
    /* Indicates if some objetive in solution 2 dominates the objetive in solution 1 */
    int dominates2 = 0;
    int nEqual = 0;
    int result;
    double value1, value2;

    for(i = 0; i < st_global_p.nObj; i++) {
        value1 = obj1[i];
        value2 = obj2[i];
        if(value1 < value2) {
            result = -1;
        } else if(value1 > value2) {
            result = 1;
        } else {
            result = 0;
            nEqual++;
        }
        if(result == -1) {
            dominates1 = 1;
        }
        if(result == 1) {
            dominates2 = 1;
        }
    }

    if(nEqual == st_global_p.nObj)
        return 2;
    if(dominates1 == dominates2) {   /* non-dominated solutions */
        return 0;
    }
    if(dominates1 == 1) {   /* solution1 dominates */
        return 1;
    }
    return -1;  /* solucion2 dominates */
}

int check_dominance(double* obj1, double* obj2)
{
    int flag1 = 0;
    int flag2 = 0;
    for(int i = 0; i < st_global_p.nObj; ++i) {
        if(obj1[i] < obj2[i])
            flag1 = 1;
        else if(obj1[i] > obj2[i])
            flag2 = 1;
    }
    if(flag1 == 1 && flag2 == 0)
        return 1;
    else if(flag1 == 0 && flag2 == 1)
        return -1;
    else
        return 0;
}

int L_dominance(double* obj1, double* obj2)
{
    int n_b = 0;
    int n_e = 0;
    int n_w = 0;

    //double norm1 = 0.0;
    //double norm2 = 0.0;

    for(int i = 0; i < st_global_p.nObj; i++) {
        if(obj1[i] < obj2[i])
            n_b++;
        else if(obj1[i] > obj2[i])
            n_w++;
        else
            n_e++;

        // 		norm1+=obj1[i];
        // 		norm2+=obj2[i];
    }

    if(n_b - n_w > 0/*&&norm1<norm2*/)
        return 1;
    if(n_b - n_w < 0/*&&norm1>norm2*/)
        return -1;
    return 0;
}