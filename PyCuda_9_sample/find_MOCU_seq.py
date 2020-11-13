import numpy as np
import time
from sampling import *
from mocu_comp import *
from MOCU import *
import numpy as np

def find_MOCU_seq(MOCU_matrix, save_f_inv, D_save, init_MOCU_val, K_max, w, N, h , M, T,
                  a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt, update_cnt_end):


    MOCU_seq = np.ones(update_cnt_end+1)*50.0
    it_temp_val = np.zeros(it_idx)
    it_temp_val_init = np.zeros(it_idx)

    MOCU_seq[0] = init_MOCU_val



    for ij in range(1,update_cnt_end+1):
        flag = 0
        min_ind = np.where(MOCU_matrix == np.min(MOCU_matrix[np.nonzero(MOCU_matrix)]))

        if len(min_ind[0]) == 1:
            min_i_MOCU = int(min_ind[0])
            min_j_MOCU = int(min_ind[1])
        else:
            min_i_MOCU = int(min_ind[0][0])
            min_j_MOCU = int(min_ind[1][0])

        #print(min_i_MOCU, min_j_MOCU)
        MOCU_matrix[min_i_MOCU, min_j_MOCU] = 0.0
        f_inv = save_f_inv[min_i_MOCU, min_j_MOCU]

        if D_save[min_i_MOCU, min_j_MOCU] == 0.0:
            a_upper_bound_update[min_i_MOCU, min_j_MOCU] \
                = min(a_upper_bound_update[min_i_MOCU, min_j_MOCU], f_inv)
            a_upper_bound_update[min_j_MOCU, min_i_MOCU] \
                = min(a_upper_bound_update[min_i_MOCU, min_j_MOCU], f_inv)
            if f_inv > a_upper_bound_update[min_i_MOCU, min_j_MOCU]:
                flag = 1

        else:
            a_lower_bound_update[min_i_MOCU, min_j_MOCU] \
                = max(a_lower_bound_update[min_i_MOCU, min_j_MOCU], f_inv)
            a_lower_bound_update[min_j_MOCU, min_i_MOCU] \
                = max(a_lower_bound_update[min_i_MOCU, min_j_MOCU], f_inv)
            if f_inv < a_lower_bound_update[min_i_MOCU, min_j_MOCU]:
                flag = 1


        cnt = 0
        while MOCU_seq[ij] > MOCU_seq[ij - 1]:
            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)
            MOCU_seq[ij] = np.median(it_temp_val)

            cnt = cnt + 1
            if cnt == 3:
                MOCU_seq[ij] = MOCU_seq[ij - 1]
                break


    return MOCU_seq