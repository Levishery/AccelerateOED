import numpy as np
import time
from sampling import *
from mocu_comp import *
from MOCU import *
import numpy as np
import random


def find_Rand_seq(MOCU_matrix, save_f_inv, D_save, init_MOCU_val, K_max, w, N, h , M, T,
                  a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt):

    #aaa = np.zeros((N,N))
    Rand_seq = np.ones(update_cnt+1)*50.0
    it_temp_val = np.zeros(it_idx)
           
    Rand_seq[0] = init_MOCU_val


    Nnum = N*(N - 1) / 2
    i_set = np.zeros(update_cnt)
    j_set = np.zeros(update_cnt)
    ind_list = []
    for i in range(N):
        for j in range(i + 1, N):
            ind_list.append([[i, j]])
    random.shuffle(ind_list)

    for i in range(update_cnt):
        i_set[i] = np.asarray(ind_list[i])[0][0]
        j_set[i] = np.asarray(ind_list[i])[0][1]

    #print(i_set, j_set)
    for ij in range(1,update_cnt+1):
        flag = 0

        i = int(i_set[ij-1])
        j = int(j_set[ij-1])

        #print(i,j)


        f_inv = save_f_inv[i, j]

        if D_save[i, j] == 0.0:
            a_upper_bound_update[i, j] \
                = min(a_upper_bound_update[i, j], f_inv)
            a_upper_bound_update[j, i] \
                = min(a_upper_bound_update[i, j], f_inv)
            if f_inv > a_upper_bound_update[i, j]:
                flag = 1

        else:
            a_lower_bound_update[i, j] \
                = max(a_lower_bound_update[i, j], f_inv)
            a_lower_bound_update[j, i] \
                = max(a_lower_bound_update[i, j], f_inv)
            if f_inv < a_lower_bound_update[i, j]:
                flag = 1



        cnt = 0
        while Rand_seq[ij] > Rand_seq[ij - 1]:

            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)

            Rand_seq[ij] = np.mean(it_temp_val)
                    
            cnt = cnt + 1
            if cnt == 5:
                Rand_seq[ij] = Rand_seq[ij - 1]
                break





    return Rand_seq

