import numpy as np
from sampling import *
from mocu_comp import *



def task_module(w, N, h , M, a_lower_bound_update, a_upper_bound_update):
    a = np.zeros((N + 1, N + 1))

    a[0:N, 0:N] = sampling(N, a_lower_bound_update, a_upper_bound_update)

    c_lower = 0.0
    c_upper = 2.0
    c_0 = c_upper

    for ij in range(15):
        a[N, :] = c_0
        a[:, N] = a[N, :].transpose()
        a[N, N] = 0.0

        D = mocu_comp(w, h, N + 1, M, a)

        if D == 1:  # syn
            c_upper = (c_lower + c_upper) / 2.0
            c_0 = c_upper
        else:  # non-syn
            c_lower = c_upper
            c_upper = c_lower + 0.5 ** ij
            c_0 = c_upper

        if (D == 0) and (ij == 0):
            print("Opps! Non sync case exists!!!")

    a_save = (c_lower + c_upper) / 2.0

    return a_save