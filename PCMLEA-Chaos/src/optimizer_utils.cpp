# include "global.h"
# include <math.h>

#define EPS 1.2e-7

int exponentialRankingSelection(int length, double pressure)
{
    double cp = pointer_gen_rand();
    int n = length;
    int i = n;
    double c = 1 - pressure;
    do {
        cp -= (c - 1) * pow(c, n - i) / (pow(c, n) - 1);
        i--;
    } while((cp > 0.0) && i > 0);
    return i;
}

int linearRankingSelection(int length, double pressure)  //prefers to select large i
{
    double cp = pointer_gen_rand();
    int u = length;
    int i = u;
    do {
        i--;
        cp -= (2 - pressure) / u + 2 * i * (pressure - 1) / (u * (u - 1));
    } while((cp > 0.0) && i > 0);
    return i;
}

void transform_to_Qbits(double* source, double* destination, int nIndiv)
{
    for(int i = 0; i < nIndiv; i++) {
        for(int j = 0; j < st_global_p.nDim; j++) {
            double val_quantum = (2.0 * source[i * st_global_p.nDim + j] - st_global_p.minLimit[j] - st_global_p.maxLimit[j]) /
                                 (st_global_p.maxLimit[j] - st_global_p.minLimit[j]);
            double new_quantum = sqrt(1.0 - val_quantum * val_quantum);
            if(flip_r((float)0.5)) new_quantum = -new_quantum;
            destination[i * 2 * st_global_p.nDim + j] = val_quantum;
            destination[i * 2 * st_global_p.nDim + st_global_p.nDim + j] = new_quantum;
        }
    }
    return;
}

void transform_fr_Qbits(double* source, double* destination, int nIndiv)
{
    for(int i = 0; i < nIndiv; i++) {
        for(int j = 0; j < st_global_p.nDim; j++) {
            double val_quantum = source[i * 2 * st_global_p.nDim + j];
            destination[i * st_global_p.nDim + j] = 0.5 * (st_global_p.maxLimit[j] * (1.0 + val_quantum) +
                                                    st_global_p.minLimit[j] * (1.0 - val_quantum));
        }
    }
    return;
}

void LeNet_delete(double* pTrail)
{
    int i;
    int num_trail = st_global_p.nDim;
    if(flip_r((float)0.9)) {
        do {
            i = rnd(0, st_global_p.nDim - 1);
            num_trail--;
        } while(num_trail &&
                ((i >= NUM_PARA_C1_M && i < NUM_PARA_C1) ||
                 (i >= NUM_PARA_C1 + NUM_PARA_C3_M && i < NUM_PARA_C1 + NUM_PARA_C3) ||
                 pTrail[i] == 0.0));
        pTrail[i] = 0.0;// -strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim + i];
    } else {
        int iMap;
        int t_flag;
        do {
            t_flag = true;
            iMap = rnd(0, NUM_PARA_C1_MAPS + NUM_PARA_C3_MAPS - 1);
            if(iMap < NUM_PARA_C1_MAPS) {
                for(i = 0; i < MAP_SIZE_C * MAP_SIZE_C; i++) {
                    if(pTrail[iMap * MAP_SIZE_C * MAP_SIZE_C + i] != 0.0) {
                        t_flag = false;
                    }
                }
            } else {
                int tmp = iMap - NUM_PARA_C1_MAPS;
                for(i = 0; i < MAP_SIZE_C * MAP_SIZE_C; i++) {
                    if(pTrail[NUM_PARA_C1 + tmp * MAP_SIZE_C * MAP_SIZE_C + i] != 0.0) {
                        t_flag = false;
                    }
                }
            }
            num_trail--;
        } while(num_trail && t_flag);
        if(iMap < NUM_PARA_C1_MAPS) {
            for(i = 0; i < MAP_SIZE_C * MAP_SIZE_C; i++) {
                pTrail[iMap * MAP_SIZE_C * MAP_SIZE_C + i] =
                    0.0;// -strct_pop_evo_info.var_offspring[iP*strct_global_paras.nDim + iMap*MAP_SIZE_C*MAP_SIZE_C + i];
            }
            pTrail[NUM_PARA_C1_M + iMap] =
                0.0;// -strct_pop_evo_info.var_offspring[iP*strct_global_paras.nDim + NUM_PARA_C1_M + iMap];
        } else {
            iMap -= NUM_PARA_C1_MAPS;
            for(i = 0; i < MAP_SIZE_C * MAP_SIZE_C; i++) {
                pTrail[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C + i] =
                    0.0;// -strct_pop_evo_info.var_offspring[iP*strct_global_paras.nDim + NUM_PARA_C1 + iMap*MAP_SIZE_C*MAP_SIZE_C + i];
            }
            t_flag = true;
            iMap = iMap % NUM_CHANNEL_C3_OUT;
            do {
                for(i = 0; i < MAP_SIZE_C * MAP_SIZE_C; i++) {
                    if(pTrail[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C + i] != 0) {
                        t_flag = false;
                    }
                }
                iMap += NUM_CHANNEL_C3_OUT;
            } while(iMap < NUM_PARA_C3_MAPS && t_flag);
            if(t_flag) {
                pTrail[NUM_PARA_C1 + NUM_PARA_C3_M + iMap % NUM_CHANNEL_C3_OUT] =
                    0.0;// -strct_pop_evo_info.var_offspring[iP*strct_global_paras.nDim + NUM_PARA_C1 + NUM_PARA_C3_M + iMap%NUM_CHANNEL_C3_OUT];
            }
        }
    }
}

void LeNet_xor(double* pCurrent, double* p1, double* p2, double* pTrail, float t_CR)
{
    //C1
    for(int iMap = 0; iMap < NUM_PARA_C1_MAPS; iMap++) {
        if(flip_r(t_CR)) {
            memcpy(&pTrail[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &pCurrent[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            pTrail[NUM_PARA_C1_M + iMap] =
                pCurrent[NUM_PARA_C1_M + iMap];
        } else if(flip_r(0.5)) {
            memcpy(&pTrail[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &p1[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            pTrail[NUM_PARA_C1_M + iMap] =
                p1[NUM_PARA_C1_M + iMap];
        } else {
            memcpy(&pTrail[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &p2[iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            pTrail[NUM_PARA_C1_M + iMap] =
                p2[NUM_PARA_C1_M + iMap];
        }
    }
    //C3
    double bias_C3[NUM_CHANNEL_C3_OUT];
    for(int iMap = 0; iMap < NUM_PARA_C3_MAPS; iMap++) {
        if(flip_r(t_CR)) {
            memcpy(&pTrail[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &pCurrent[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            bias_C3[iMap % NUM_CHANNEL_C3_OUT] =
                pCurrent[NUM_PARA_C1 + NUM_PARA_C3_M + iMap % NUM_CHANNEL_C3_OUT];
        } else if(flip_r(0.5)) {
            memcpy(&pTrail[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &p1[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            bias_C3[iMap % NUM_CHANNEL_C3_OUT] =
                p1[NUM_PARA_C1 + NUM_PARA_C3_M + iMap % NUM_CHANNEL_C3_OUT];
        } else {
            memcpy(&pTrail[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   &p2[NUM_PARA_C1 + iMap * MAP_SIZE_C * MAP_SIZE_C],
                   MAP_SIZE_C * MAP_SIZE_C * sizeof(double));
            bias_C3[iMap % NUM_CHANNEL_C3_OUT] =
                p2[NUM_PARA_C1 + NUM_PARA_C3_M + iMap % NUM_CHANNEL_C3_OUT];
        }
    }
    for(int i = 0; i < NUM_CHANNEL_C3_OUT; i++) {
        pTrail[NUM_PARA_C1 + NUM_PARA_C3_M + i] =
            bias_C3[i] / NUM_CHANNEL_C3_IN;
    }
    for(int i = NUM_PARA_C1 + NUM_PARA_C3; i < st_global_p.nDim; i++) {
        pTrail[i] = pCurrent[i];
        //if (flip_r((float)CRall[iP]))
        //	strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim + i] = p1[i];
        //else if (flip_r(0.5))
        //	strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim + i] = p2[i];
        //else
        //	strct_pop_evo_info.var_offspring[iP * strct_global_paras.nDim + i] = p3[i];
    }
}