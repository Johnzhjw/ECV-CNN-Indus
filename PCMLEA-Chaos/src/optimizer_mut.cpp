# include "global.h"
# include <math.h>
# include <assert.h>

#define EPS 1.2e-7

void realmutation(double* trail, double rate)
{
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    double eta_m = etam;
    int size_g;
    if(st_MPI_p.color_obj)
        size_g = st_grp_info_p.table_mine_size;
    else
        size_g = st_grp_info_p.table_mine_size;

    for(int j = 0; j < size_g; j++) {
        if(flip_r((float)rate)) {
            y = trail[j];
            if(st_MPI_p.color_obj) {
                yl = st_global_p.minLimit[st_grp_info_p.table_mine[j]];
                yu = st_global_p.maxLimit[st_grp_info_p.table_mine[j]];
            } else {
                yl = st_global_p.minLimit[st_grp_info_p.table_mine[j]];
                yu = st_global_p.maxLimit[st_grp_info_p.table_mine[j]];
            }
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = pointer_gen_rand();
            mut_pow = 1.0 / (eta_m + 1.0);
            if(rnd <= 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (eta_m + 1.0)));
                deltaq = pow(val, mut_pow) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (eta_m + 1.0)));
                deltaq = 1.0 - (pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            if(y < yl)
                y = yl;
            if(y > yu)
                y = yu;
            trail[j] = y;
        }
    }
    return;
}

void realmutation_whole_bin(double* indiv, double rate)
{
    int size_g;
    size_g = st_global_p.nDim;

    for(int j = 0; j < size_g; j++) {
        if(flip_r((float)rate)) {
            indiv[j] = st_global_p.maxLimit[st_grp_info_p.table_mine[j]] - indiv[j];
        }
    }
    return;
}

void binarymutation_whole_bin_Markov(double* indiv, double rate)
{
    int seleNum = 0;
    int unseNum = 0;
    //int th_nFeat = TH_N_FEATURE;
    int n_repetition = 2;// th_nFeat / 5;
    int n_repet_add = rnd(1, n_repetition);
    int n_repet_del = rnd(1, n_repetition);

    int* arrFeatSele = (int*)calloc(N_FEATURE, sizeof(int));
    int* arrFeatUnse = (int*)calloc(N_FEATURE, sizeof(int));

    int* arrINDEX = (int*)calloc(N_FEATURE, sizeof(int));
    for(int i = 0; i < N_FEATURE; i++) arrINDEX[i] = i;
    shuffle(arrINDEX, N_FEATURE);

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[rank_INDX[i]] == 0) {
            arrFeatUnse[unseNum++] = rank_INDX[i];
        }
    }

    double pressure = 1.5;

    int cur_unseNum = unseNum;
    while(n_repet_add--) {
        int tmp;
        tmp = cur_unseNum - 1 - linearRankingSelection(cur_unseNum, pressure);
        int count = -1;
        int theIND = -1;
        for(int j = 0; j < N_FEATURE; j++) {
            if(arrFeatUnse[j] >= 0) {
                count++;
            }
            if(count == tmp) {
                theIND = arrFeatUnse[j];
                arrFeatUnse[j] = -1;
                break;
            }
        }
        if(theIND == -1) {
            printf("%s: theIND not changed\n", AT);
        }
        indiv[theIND] = st_global_p.maxLimit[theIND] - indiv[theIND];
        if((int)indiv[theIND] == 0) {
            indiv[theIND] = rndreal(1.0, st_global_p.maxLimit[theIND]);
        }
        cur_unseNum--;
    }

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[rank_INDX[i]]) {
            arrFeatSele[seleNum++] = rank_INDX[i];
        }
    }
    int cur_seleNum = seleNum;
    while(n_repet_del--) {
        if(cur_seleNum) {
            int tmp;
            tmp = cur_seleNum - 1 - linearRankingSelection(cur_seleNum, pressure);
            int count = -1;
            int theIND = -1;
            for(int j = 0; j < seleNum; j++) {
                if(arrFeatSele[j] >= 0) {
                    count++;
                }
                if(count == tmp) {
                    theIND = arrFeatSele[j];
                    break;
                }
            }
            if(theIND == -1) {
                printf("%s: theIND not changed\n", AT);
            }
            tmp = theIND;

            int flag = 1;

            for(int iF = 0; iF < N_FEATURE; iF++) {
                if((int)indiv[arrINDEX[iF]] &&
                   arrINDEX[iF] != tmp &&
                   featureCorrelation2(tmp, N_FEATURE) >= featureCorrelation2(arrINDEX[iF], N_FEATURE) &&
                   featureCorrelation2(arrINDEX[iF], tmp) >= featureCorrelation2(arrINDEX[iF], N_FEATURE)) {
                    indiv[arrINDEX[iF]] = st_global_p.maxLimit[arrINDEX[iF]] - indiv[arrINDEX[iF]];
                    flag = 0;
                    for(int j = 0; j < seleNum; j++) {
                        if(arrFeatSele[j] == arrINDEX[iF])
                            arrFeatSele[j] = -1;
                    }
                    cur_seleNum--;
                }
            }
            if(flag && cur_seleNum > 1) {
                indiv[tmp] = st_global_p.maxLimit[tmp] - indiv[tmp];
                cur_seleNum--;
                for(int j = 0; j < seleNum; j++) {
                    if(arrFeatSele[j] == tmp)
                        arrFeatSele[j] = -1;
                }
            }
        }
    }

    free(arrFeatSele);
    free(arrFeatUnse);
    free(arrINDEX);

    return;
}

void randmutation_whole(double* indiv, double rate)
{
    double rnd;
    double y, yl, yu;
    double eta_m = etam;
    int size_g;
    size_g = st_global_p.nDim;

    for(int j = 0; j < size_g; j++) {
        if(flip_r((float)rate)) {
            y = indiv[j];
            yl = st_global_p.minLimit[j];
            yu = st_global_p.maxLimit[j];

            rnd = pointer_gen_rand();

            y = rnd * (yu - yl) + yl;

            indiv[j] = y;
        }
    }
    return;
}

void realmutation_whole(double* indiv, double rate)
{
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    double eta_m = etam;
    int size_g;
    size_g = st_global_p.nDim;

    for(int j = 0; j < size_g; j++) {
        if(flip_r((float)rate)) {
            y = indiv[j];
            yl = st_global_p.minLimit[j];
            yu = st_global_p.maxLimit[j];

            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = pointer_gen_rand();
            mut_pow = 1.0 / (eta_m + 1.0);
            if(rnd <= 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (eta_m + 1.0)));
                deltaq = pow(val, mut_pow) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (eta_m + 1.0)));
                deltaq = 1.0 - (pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            if(y < yl)
                y = yl;
            if(y > yu) {
                y = yu;
                //printf("%s\n", AT);
            }
            indiv[j] = y;
        }
    }
    return;
}

void localSearch(int iP)
{
}

void refinement(int iP)
{
    int i;
    int n = rnd(1, int(0.01 * st_global_p.nDim));
    int* inds = (int*)calloc(st_global_p.nDim, sizeof(int));
    {
        for(i = 0; i < st_global_p.nDim; i++) inds[i] = i;
        shuffle(inds, st_global_p.nDim);
    }
    realmutation_whole_fixed(&st_pop_evo_cur.var[iP * st_global_p.nDim], inds, n);
    memcpy(&st_pop_evo_offspring.var[iP * st_global_p.nDim], &st_pop_evo_cur.var[iP * st_global_p.nDim],
           st_global_p.nDim * sizeof(double));

    free(inds);

    return;
}

void realmutation_whole_fixed(double* indiv, int* inds, int n)
{
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    double eta_m = etam;
    int i, j;

    for(i = 0; i < n; i++) {
        j = inds[i];
        //		if (flip_r((float)rate))
        {
            y = indiv[j];
            yl = st_global_p.minLimit[j];
            yu = st_global_p.maxLimit[j];

            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = pointer_gen_rand();
            mut_pow = 1.0 / (eta_m + 1.0);
            if(rnd <= 0.5) {
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (eta_m + 1.0)));
                deltaq = pow(val, mut_pow) - 1.0;
            } else {
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (eta_m + 1.0)));
                deltaq = 1.0 - (pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            boundaryExceedingFixing(y, y, yl, yu);
            indiv[j] = y;
        }
    }
    return;
}

///////////////////////////////////////////////////////////
void adjustFeatureNum_hybrid(double* indiv)
{
    int count = 0;
    int* arrIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrIDX[count++] = i;
        }
    }

    if(count > th_nFeat) {
        int tmpSize = (int)(pointer_gen_rand() * th_nFeat) + 1;
        for(int i = count; i > tmpSize; i--) {
            int tmp = (int)(pointer_gen_rand() * i);
            int tmp2 = arrIDX[tmp];
            indiv[tmp2] = st_global_p.maxLimit[tmp2] - indiv[tmp2];
            arrIDX[tmp] = arrIDX[i - 1];
            arrIDX[i - 1] = tmp2;
        }
    }

    if(count == 0) {
        count = (int)(pointer_gen_rand() * th_nFeat) + 1;
        int i = 0;
        while(i < count) {
            int tmp = (int)(pointer_gen_rand() * st_global_p.nDim);
            int flag = 1;
            for(int j = 0; j < i; j++) {
                if(tmp == arrIDX[j])
                    flag = 0;
            }
            if(flag) {
                arrIDX[i++] = tmp;
            }
        }
        for(i = 0; i < count; i++) {
            indiv[arrIDX[i]] = st_global_p.maxLimit[arrIDX[i]] - indiv[arrIDX[i]];
            if((int)indiv[arrIDX[i]] == 0) {
                indiv[arrIDX[i]] = rndreal(1.0, st_global_p.maxLimit[arrIDX[i]]);
            }
        }
    }

    free(arrIDX);

    return;
}

void adjustFeatureNum_filter(double* indiv)
{
    int seleNum = 0;
    int unseNum = 0;
    int th_nFeat = TH_N_FEATURE;

    int* arrFeatSele = (int*)calloc(N_FEATURE, sizeof(int));
    int* arrFeatUnse = (int*)calloc(N_FEATURE, sizeof(int));

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[rank_INDX[i]]) {
            arrFeatSele[seleNum++] = rank_INDX[i];
        } else {
            arrFeatUnse[unseNum++] = rank_INDX[i];
        }
    }

    double pressure = 1.5;

    if(seleNum > th_nFeat) {
        int tmpSize = (int)(pointer_gen_rand() * th_nFeat) + 1;
        for(int i = seleNum; i > tmpSize; i--) {
            int tmp;
            tmp = linearRankingSelection(i, pressure);
            int count = -1;
            int theIND = -1;
            for(int j = 0; j < seleNum; j++) {
                if(arrFeatSele[j] >= 0) {
                    count++;
                }
                if(count == tmp) {
                    theIND = arrFeatSele[j];
                    arrFeatSele[j] = -1;
                    break;
                }
            }
            if(theIND == -1) {
                printf("%s: theIND not changed\n", AT);
            }
            indiv[theIND] = st_global_p.maxLimit[theIND] - indiv[theIND];
        }
    }
    if(seleNum == 0) {
        seleNum = (int)(pointer_gen_rand() * th_nFeat) + 1;
        for(int i = N_FEATURE; i > N_FEATURE - seleNum; i--) {
            int tmp;
            tmp = i - 1 - linearRankingSelection(i, pressure);
            int count = -1;
            int theIND = -1;
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrFeatUnse[j] >= 0) {
                    count++;
                }
                if(count == tmp) {
                    theIND = arrFeatUnse[j];
                    arrFeatUnse[j] = -1;
                    break;
                }
            }
            if(theIND == -1) {
                printf("%s: theIND not changed\n", AT);
            }
            indiv[theIND] = st_global_p.maxLimit[theIND] - indiv[theIND];
            if((int)indiv[theIND] == 0) {
                indiv[theIND] = rndreal(1.0, st_global_p.maxLimit[theIND]);
            }
        }
    }

    free(arrFeatSele);
    free(arrFeatUnse);

    return;
}

void adjustFeatureNum_Markov(double* indiv)
{
    int seleNum = 0;
    int unseNum = 0;
    int th_nFeat = TH_N_FEATURE;

    int* arrFeatSele = (int*)calloc(N_FEATURE, sizeof(int));
    int* arrFeatUnse = (int*)calloc(N_FEATURE, sizeof(int));

    int* arrINDEX = (int*)calloc(N_FEATURE, sizeof(int));
    for(int i = 0; i < N_FEATURE; i++) arrINDEX[i] = i;
    shuffle(arrINDEX, N_FEATURE);

    for(int i = 0; i < st_global_p.nDim; i++) {
        //printf("%d-%lf\n", rank_INDX[i], indiv[rank_INDX[i]]);
        if((int)indiv[rank_INDX[i]]) {
            arrFeatSele[seleNum++] = rank_INDX[i];
        } else {
            arrFeatUnse[unseNum++] = rank_INDX[i];
        }
    }

    double pressure = 1.5;
    int curSeleNum = seleNum;
    int tmp_seleNum = (int)(pointer_gen_rand() * th_nFeat) + 1;

    while(curSeleNum > tmp_seleNum) {
        if(st_MPI_p.mpi_rank == 0) {
            //printf("%d ", curSeleNum);
        }
        int tmp;
        tmp = curSeleNum - 1 - linearRankingSelection(curSeleNum, pressure);
        int count = -1;
        int theIND = -1;
        for(int j = 0; j < seleNum; j++) {
            if(arrFeatSele[j] >= 0) {
                count++;
            }
            if(count == tmp) {
                theIND = arrFeatSele[j];
                break;
            }
        }
        if(theIND == -1) {
            printf("%s: theIND not changed\n", AT);
        }
        tmp = theIND;

        int flag = 1;

        for(int iF = 0; iF < N_FEATURE; iF++) {
            if((int)indiv[arrINDEX[iF]] &&
               arrINDEX[iF] != tmp &&
               featureCorrelation2(tmp, N_FEATURE) >= featureCorrelation2(arrINDEX[iF], N_FEATURE) &&
               featureCorrelation2(arrINDEX[iF], tmp) >= featureCorrelation2(arrINDEX[iF], N_FEATURE)) {
                //if (strct_MPI_info.mpi_rank == 0)
                //{
                //	if (arrINDEX[iF] < 0 || arrINDEX[iF] >= N_FEATURE)
                //		printf("ERROR: %d ", arrINDEX[iF]);
                //}
                indiv[arrINDEX[iF]] = st_global_p.maxLimit[arrINDEX[iF]] - indiv[arrINDEX[iF]];
                curSeleNum--;
                for(int j = 0; j < seleNum; j++) {
                    if(arrFeatSele[j] == arrINDEX[iF])
                        arrFeatSele[j] = -1;
                }
                flag = 0;
            }
        }
        if(flag) {
            //if (strct_MPI_info.mpi_rank == 0)
            //{
            //	if (tmp < 0 || tmp >= N_FEATURE)
            //		printf("%d ", tmp);
            //}
            indiv[tmp] = st_global_p.maxLimit[tmp] - indiv[tmp];
            curSeleNum--;
            for(int j = 0; j < seleNum; j++) {
                if(arrFeatSele[j] == tmp)
                    arrFeatSele[j] = -1;
            }
        }
    }
    seleNum = curSeleNum;

    if(seleNum == 0) {
        memcpy(arrFeatUnse, rank_INDX, N_FEATURE * sizeof(int));
        //if (strct_MPI_info.mpi_rank == 0)
        //{
        //	printf("%d ", seleNum);
        //}

        seleNum = (int)(pointer_gen_rand() * th_nFeat) + 1;
        for(int i = N_FEATURE; i > N_FEATURE - seleNum; i--) {
            int tmp;
            tmp = i - 1 - linearRankingSelection(i, pressure);
            int count = -1;
            int theIND = -1;
            for(int j = 0; j < N_FEATURE; j++) {
                if(arrFeatUnse[j] >= 0) {
                    count++;
                }
                if(count == tmp) {
                    theIND = arrFeatUnse[j];
                    arrFeatUnse[j] = -1;
                    break;
                }
            }
            if(theIND == -1) {
                printf("%s: theIND not changed\n", AT);
            }
            indiv[theIND] = st_global_p.maxLimit[theIND] - indiv[theIND];
            if((int)indiv[theIND] == 0) {
                st_global_p.maxLimit[theIND] = rndreal(1.0, st_global_p.maxLimit[theIND]);
            }
        }
    }

    free(arrFeatSele);
    free(arrFeatUnse);
    free(arrINDEX);

    return;
}

void adjustFeatureNum_rand(double* indiv)
{
    int count = 0;
    int* arrIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrIDX[count++] = i;
        }
    }

    if(count > th_nFeat) {
        int tmpSize = (int)(pointer_gen_rand() * th_nFeat) + 1;
        for(int i = count; i > tmpSize; i--) {
            int tmp = (int)(pointer_gen_rand() * i);
            int tmp2 = arrIDX[tmp];
            indiv[tmp2] = st_global_p.maxLimit[tmp2] - indiv[tmp2];
            arrIDX[tmp] = arrIDX[i - 1];
            arrIDX[i - 1] = tmp2;
        }
    }

    if(count == 0) {
        count = (int)(pointer_gen_rand() * th_nFeat) + 1;
        int i = 0;
        while(i < count) {
            int tmp = (int)(pointer_gen_rand() * st_global_p.nDim);
            int flag = 1;
            for(int j = 0; j < i; j++) {
                if(tmp == arrIDX[j])
                    flag = 0;
            }
            if(flag) {
                arrIDX[i++] = tmp;
            }
        }
        for(i = 0; i < count; i++) {
            indiv[arrIDX[i]] = st_global_p.maxLimit[arrIDX[i]] - indiv[arrIDX[i]];
            if((int)indiv[arrIDX[i]] == 0) {
                indiv[arrIDX[i]] = rndreal(1.0, st_global_p.maxLimit[arrIDX[i]]);
            }
        }
    }

    free(arrIDX);

    return;
}

void adjustFeatureNum_rank_corr_tour(double* indiv)
{
    int* arrSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* arrUNSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;
    int depth = 10;
    int selectIND, candidIND;

    int selectNum = 0;
    int unselectNum = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrSelectIDX[selectNum++] = i;
        } else {
            arrUNSelectIDX[unselectNum++] = i;
        }
    }

    while(selectNum > th_nFeat) {   //////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        selectIND = rnd(0, selectNum - 1);
        for(int n = 0; n < depth; n++) {
            candidIND = rnd(0, selectNum - 1);
            if(filterWeights[arrSelectIDX[candidIND]][0] < filterWeights[arrSelectIDX[selectIND]][0]) {
                selectIND = candidIND;
            }
        }
        indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
        arrUNSelectIDX[unselectNum++] = arrSelectIDX[selectIND];
        arrSelectIDX[selectIND] = arrSelectIDX[--selectNum];
    }

    if(selectNum == 0) {
        selectNum = rnd(1, th_nFeat);
        int i = 0;
        while(i < selectNum) {
            selectIND = rnd(0, unselectNum - 1);
            for(int n = 0; n < depth * 50; n++) {
                candidIND = rnd(0, unselectNum - 1);
                if(filterWeights[arrUNSelectIDX[candidIND]][0] > filterWeights[arrUNSelectIDX[selectIND]][0]) {
                    selectIND = candidIND;
                }
            }
            indiv[arrUNSelectIDX[selectIND]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND]] - indiv[arrUNSelectIDX[selectIND]];
            if((int)indiv[arrUNSelectIDX[selectIND]] == 0) {
                indiv[arrUNSelectIDX[selectIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND]]);
            }
            arrSelectIDX[i++] = arrUNSelectIDX[selectIND];
            arrUNSelectIDX[selectIND] = arrUNSelectIDX[--unselectNum];
        }
    }

    if(selectNum >= 2) {
        int numReplace = selectNum / 9 + 1;
        int candidIND1, candidIND2;
        int selectIND1, selectIND2;
        depth = 5;
        for(int n = 0; n < numReplace; n++) {
            selectIND = rnd(0, selectNum - 1);
            do {
                selectIND1 = rnd(0, selectNum - 1);
            } while(selectIND == selectIND1);
            for(int nn = 0; nn < depth; nn++) {
                do {
                    candidIND1 = rnd(0, selectNum - 1);
                } while(selectIND == candidIND1);
                if(filterWeights[arrSelectIDX[candidIND1]][0] < filterWeights[arrSelectIDX[selectIND1]][0] &&
                   featureCorrelation2(arrSelectIDX[selectIND], arrSelectIDX[candidIND1]) > featureCorrelation2(arrSelectIDX[selectIND],
                           arrSelectIDX[selectIND1])) {
                    selectIND1 = candidIND1;
                }
            }

            selectIND2 = rnd(0, unselectNum - 1);
            for(int nn = 0; nn < depth * 10; nn++) {
                candidIND2 = rnd(0, unselectNum - 1);
                if(filterWeights[arrUNSelectIDX[candidIND2]][0] > filterWeights[arrUNSelectIDX[selectIND2]][0] &&
                   featureCorrelation2(arrSelectIDX[selectIND], arrUNSelectIDX[candidIND2]) < featureCorrelation2(arrSelectIDX[selectIND],
                           arrUNSelectIDX[selectIND2])) {
                    selectIND2 = candidIND2;
                }
            }

            if(flip_r((float)0.1) ||
               ((filterWeights[arrSelectIDX[selectIND1]][0] < filterWeights[arrUNSelectIDX[selectIND2]][0]) &&
                featureCorrelation2(arrSelectIDX[selectIND], arrSelectIDX[selectIND1]) > featureCorrelation2(arrSelectIDX[selectIND],
                        arrUNSelectIDX[selectIND2]))) {
                indiv[arrSelectIDX[selectIND1]] = st_global_p.maxLimit[arrSelectIDX[selectIND1]] - indiv[arrSelectIDX[selectIND1]];
                indiv[arrUNSelectIDX[selectIND2]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND2]] - indiv[arrUNSelectIDX[selectIND2]];
                if((int)indiv[arrUNSelectIDX[selectIND2]] == 0) {
                    indiv[arrUNSelectIDX[selectIND2]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND2]]);
                }
                int tmp = arrSelectIDX[selectIND1];
                arrSelectIDX[selectIND1] = arrUNSelectIDX[selectIND2];
                arrUNSelectIDX[selectIND2] = tmp;
            }
        }
    }

    free(arrSelectIDX);
    free(arrUNSelectIDX);

    return;
}

void adjustFeatureNum_rank_replace(double* indiv)
{
    int* arrSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* arrUNSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;
    //int depth = 10;
    int selectIND;

    int selectNum = 0;
    int unselectNum = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrSelectIDX[selectNum++] = i;
        } else {
            arrUNSelectIDX[unselectNum++] = i;
        }
    }

    while(selectNum > th_nFeat) {   //////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        selectIND = -1;
        double minV = INF_DOUBLE;
        for(int i = 0; i < selectNum; i++) {
            if(filterWeights[arrSelectIDX[i]][0] < minV) {
                minV = filterWeights[arrSelectIDX[i]][0];
                selectIND = i;
            }
        }
        indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
        arrUNSelectIDX[unselectNum++] = arrSelectIDX[selectIND];
        arrSelectIDX[selectIND] = arrSelectIDX[--selectNum];
    }

    if(selectNum == 0) {
        selectNum = rnd(1, th_nFeat);
        int n = 0;
        while(n < selectNum) {
            selectIND = -1;
            double maxV = -INF_DOUBLE;
            for(int i = 0; i < unselectNum; i++) {
                if(filterWeights[arrUNSelectIDX[i]][0] > maxV) {
                    maxV = filterWeights[arrUNSelectIDX[i]][0];
                    selectIND = i;
                }
            }
            indiv[arrUNSelectIDX[selectIND]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND]] - indiv[arrUNSelectIDX[selectIND]];
            if((int)indiv[arrUNSelectIDX[selectIND]] == 0) {
                indiv[arrUNSelectIDX[selectIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND]]);
            }
            arrSelectIDX[n++] = arrUNSelectIDX[selectIND];
            arrUNSelectIDX[selectIND] = arrUNSelectIDX[--unselectNum];
        }
    }

    {
        int selectIND1 = -1, selectIND2 = -1;
        double minV = INF_DOUBLE, maxV = -INF_DOUBLE;
        for(int i = 0; i < selectNum; i++) {
            if(filterWeights[arrSelectIDX[i]][0] < minV) {
                minV = filterWeights[arrSelectIDX[i]][0];
                selectIND1 = i;
            }
        }
        for(int i = 0; i < unselectNum; i++) {
            if(filterWeights[arrUNSelectIDX[i]][0] > maxV) {
                maxV = filterWeights[arrUNSelectIDX[i]][0];
                selectIND2 = i;
            }
        }
        indiv[arrSelectIDX[selectIND1]] = st_global_p.maxLimit[arrSelectIDX[selectIND1]] - indiv[arrSelectIDX[selectIND1]];
        indiv[arrUNSelectIDX[selectIND2]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND2]] - indiv[arrUNSelectIDX[selectIND2]];
        if((int)indiv[arrUNSelectIDX[selectIND2]] == 0) {
            indiv[arrUNSelectIDX[selectIND2]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND2]]);
        }
        int tmp = arrSelectIDX[selectIND1];
        arrSelectIDX[selectIND1] = arrUNSelectIDX[selectIND2];
        arrUNSelectIDX[selectIND2] = tmp;
    }

    free(arrSelectIDX);
    free(arrUNSelectIDX);

    return;
}

void adjustFeatureNum_corr_replace(double* indiv)
{
    int* arrSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* arrUNSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;
    //int depth = 10;
    int selectIND, candidIND;

    int selectNum = 0;
    int unselectNum = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrSelectIDX[selectNum++] = i;
        } else {
            arrUNSelectIDX[unselectNum++] = i;
        }
    }

    while(selectNum > th_nFeat) {   //////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        selectIND = -1;
        double minV = INF_DOUBLE;
        for(int i = 0; i < selectNum; i++) {
            if(filterWeights[arrSelectIDX[i]][0] < minV) {
                minV = filterWeights[arrSelectIDX[i]][0];
                selectIND = i;
            }
        }
        indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
        arrUNSelectIDX[unselectNum++] = arrSelectIDX[selectIND];
        arrSelectIDX[selectIND] = arrSelectIDX[--selectNum];
    }

    if(selectNum == 0) {
        selectNum = rnd(1, th_nFeat);
        int n = 0;
        while(n < selectNum) {
            selectIND = -1;
            double maxV = -INF_DOUBLE;
            for(int i = 0; i < unselectNum; i++) {
                if(filterWeights[arrUNSelectIDX[i]][0] > maxV) {
                    maxV = filterWeights[arrUNSelectIDX[i]][0];
                    selectIND = i;
                }
            }
            indiv[arrUNSelectIDX[selectIND]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND]] - indiv[arrUNSelectIDX[selectIND]];
            if((int)indiv[arrUNSelectIDX[selectIND]] == 0) {
                indiv[arrUNSelectIDX[selectIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND]]);
            }
            arrSelectIDX[n++] = arrUNSelectIDX[selectIND];
            arrUNSelectIDX[selectIND] = arrUNSelectIDX[--unselectNum];
        }
    }

    {
        int selectIND1 = -1, selectIND2 = -1;
        double minV = INF_DOUBLE, maxV = -INF_DOUBLE;
        for(int i = 0; i < selectNum; i++) {
            if(filterWeights[arrSelectIDX[i]][0] < minV) {
                minV = filterWeights[arrSelectIDX[i]][0];
                selectIND1 = i;
            }
        }
        candidIND = -1;
        for(int i = 0; i < selectNum; i++) {
            if(i != selectIND1) {
                if(candidIND == -1) {
                    candidIND = i;
                } else {
                    if(featureCorrelation2(arrSelectIDX[selectIND1], i) > featureCorrelation2(arrSelectIDX[selectIND1], candidIND)) {
                        candidIND = i;
                    }
                }
            }
        }
        for(int i = 0; i < unselectNum; i++) {
            if(filterWeights[arrUNSelectIDX[i]][0] > maxV) {
                maxV = filterWeights[arrUNSelectIDX[i]][0];
                selectIND2 = i;
            }
        }
        if(candidIND >= 0) {
            indiv[arrSelectIDX[candidIND]] = st_global_p.maxLimit[arrSelectIDX[candidIND]] - indiv[arrSelectIDX[candidIND]];
            indiv[arrUNSelectIDX[selectIND2]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND2]] - indiv[arrUNSelectIDX[selectIND2]];
            if((int)indiv[arrUNSelectIDX[selectIND2]] == 0) {
                indiv[arrUNSelectIDX[selectIND2]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND2]]);
            }
            int tmp = arrSelectIDX[candidIND];
            arrSelectIDX[candidIND] = arrUNSelectIDX[selectIND2];
            arrUNSelectIDX[selectIND2] = tmp;
        }
    }

    free(arrSelectIDX);
    free(arrUNSelectIDX);

    return;
}

void adjustFeatureNum_rank_corr_memetic(double* indiv)
{
    int* arrSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* arrUNSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int th_nFeat = TH_N_FEATURE;
    int depth = 10;
    int selectIND, candidIND;

    int selectNum = 0;
    int unselectNum = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrSelectIDX[selectNum++] = i;
        } else {
            arrUNSelectIDX[unselectNum++] = i;
        }
    }

    while(selectNum > th_nFeat) {   //////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        selectIND = rnd(0, selectNum - 1);
        for(int n = 0; n < depth; n++) {
            candidIND = rnd(0, selectNum - 1);
            if(filterWeights[arrSelectIDX[candidIND]][0] < filterWeights[arrSelectIDX[selectIND]][0]) {
                selectIND = candidIND;
            }
        }
        indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
        arrUNSelectIDX[unselectNum++] = arrSelectIDX[selectIND];
        arrSelectIDX[selectIND] = arrSelectIDX[--selectNum];
    }

    if(selectNum == 0) {
        selectNum = rnd(1, th_nFeat);
        int i = 0;
        while(i < selectNum) {
            selectIND = rnd(0, unselectNum - 1);
            for(int n = 0; n < depth * 50; n++) {
                candidIND = rnd(0, unselectNum - 1);
                if(filterWeights[arrUNSelectIDX[candidIND]][0] > filterWeights[arrUNSelectIDX[selectIND]][0]) {
                    selectIND = candidIND;
                }
            }
            indiv[arrUNSelectIDX[selectIND]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND]] - indiv[arrUNSelectIDX[selectIND]];
            if((int)indiv[arrUNSelectIDX[selectIND]] == 0) {
                indiv[arrUNSelectIDX[selectIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND]]);
            }
            arrSelectIDX[i++] = arrUNSelectIDX[selectIND];
            arrUNSelectIDX[selectIND] = arrUNSelectIDX[--unselectNum];
        }
    }

    {
    }

    free(arrSelectIDX);
    free(arrUNSelectIDX);

    return;
}

void adjustFeatureNum_cluster(double* indiv)
{
    int* arrSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    int* arrUNSelectIDX = (int*)calloc(st_global_p.nDim, sizeof(int));
    const int th_nFeat = TH_N_FEATURE;
    int depth = 10;
    int selectIND, candidIND;

    int selectNum = 0;
    int unselectNum = 0;

    for(int i = 0; i < st_global_p.nDim; i++) {
        if((int)indiv[i]) {
            arrSelectIDX[selectNum++] = i;
        } else {
            arrUNSelectIDX[unselectNum++] = i;
        }
    }

    int nInstance = N_sample_optimize;
    double* v_dist = (double*)calloc(st_global_p.nDim, sizeof(double));
    double d;

    for(int i = 0; i < st_global_p.nDim; i++) {
        v_dist[i] = 0.0;
    }
    for(int i = 0; i < selectNum; i++) {
        for(int j = i + 1; j < selectNum; j++) {
            d = 0.0;
            for(int k = 0; k < nInstance; k++) {
                d += (optimizeData[k][arrSelectIDX[i]] - optimizeData[k][arrSelectIDX[j]]) * (optimizeData[k][arrSelectIDX[i]] -
                        optimizeData[k][arrSelectIDX[j]]);
            }
            d = sqrt(d);
            v_dist[i] += d;
            v_dist[j] += d;
        }
    }
    double minDist;

    while(selectNum > th_nFeat) {   //////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        minDist = INF_DOUBLE;
        selectIND = -1;
        for(int i = 0; i < selectNum; i++) {
            if(v_dist[i] < minDist) {
                minDist = v_dist[i];
                selectIND = i;
            }
        }
        indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
        arrUNSelectIDX[unselectNum++] = arrSelectIDX[selectIND];
        arrSelectIDX[selectIND] = arrSelectIDX[--selectNum];
        v_dist[selectIND] = v_dist[selectNum];
    }

    if(selectNum == 0) {
        selectNum = rnd(1, th_nFeat);
        int i = 0;
        while(i < selectNum) {
            selectIND = rnd(0, unselectNum - 1);
            for(int n = 0; n < depth * 50; n++) {
                candidIND = rnd(0, unselectNum - 1);
                if(filterWeights[arrUNSelectIDX[candidIND]][0] > filterWeights[arrUNSelectIDX[selectIND]][0]) {
                    selectIND = candidIND;
                }
            }
            indiv[arrUNSelectIDX[selectIND]] = st_global_p.maxLimit[arrUNSelectIDX[selectIND]] - indiv[arrUNSelectIDX[selectIND]];
            if((int)indiv[arrUNSelectIDX[selectIND]] == 0) {
                indiv[arrUNSelectIDX[selectIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[selectIND]]);
            }
            arrSelectIDX[i++] = arrUNSelectIDX[selectIND];
            arrUNSelectIDX[selectIND] = arrUNSelectIDX[--unselectNum];
        }
    }

    {
        selectIND = rnd(0, selectNum - 1);
        candidIND = rnd(0, unselectNum - 1);
        if(st_ctrl_p.tag_selection[selectIND] < st_ctrl_p.tag_selection[candidIND]) {
            indiv[arrSelectIDX[selectIND]] = st_global_p.maxLimit[arrSelectIDX[selectIND]] - indiv[arrSelectIDX[selectIND]];
            indiv[arrUNSelectIDX[candidIND]] = st_global_p.maxLimit[arrUNSelectIDX[candidIND]] - indiv[arrUNSelectIDX[candidIND]];
            if((int)indiv[arrUNSelectIDX[candidIND]] == 0) {
                indiv[arrUNSelectIDX[candidIND]] = rndreal(1.0, st_global_p.maxLimit[arrUNSelectIDX[candidIND]]);
            }
        }
    }

    free(arrSelectIDX);
    free(arrUNSelectIDX);
    free(v_dist);

    return;
}
