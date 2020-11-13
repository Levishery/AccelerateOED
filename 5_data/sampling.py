import numpy as np

def sampling(N, a_lower_bound, a_upper_bound):
    a = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            a[i,j] = a_lower_bound[i,j]+(a_upper_bound[i,j]-a_lower_bound[i,j])*np.random.random_sample()
            a[j,i] = a[i,j]
    return a