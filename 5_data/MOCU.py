

import pycuda.autoinit
import pycuda.driver as drv
#import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule
import numpy as np
import time
#from sampling import *
#from task_module import *
#from mocu_comp import *


mod = SourceModule("""

// This should be manually changed due to the technical issue in the PyCUDA.
// Well, yes, I am lazy...

#define N_global 10

__device__ int mocu_comp(double *w, double h, int N, int M, double* a)
{
    int D = 0;
    double tol,max_temp,min_temp;
    max_temp = -100.0;
    min_temp = 100.0;
    double pi_n = 3.14159265358979323846;

    double theta[N_global];
    double theta_old[N_global];
    double F[N_global],k1[N_global],k2[N_global],k3[N_global],k4[N_global];
    double diff_t[N_global];
    int i,j,k;
    double t = 0.0;
    double sum_temp;


    for (i=0;i<N;i++){
        theta[i] = 0.0;
        theta_old[i] = 0.0;
        F[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        diff_t[i] = 0.0;
    }

    for (k=0;k<M;k++){

        
        for (i=0;i<N;i++){

            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
              
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k1[i] = h*F[i];
            theta[i] = theta_old[i] + k1[i]/2.0;
          }
          
        

        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k2[i] = h*F[i];
            theta[i] = theta_old[i] + k2[i]/2.0;
          }
          

        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
         }
        for(i=0;i<N;i++){
            k3[i] = h*F[i];
            theta[i] = theta_old[i] + k3[i];
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }


        for(i=0;i<N;i++){        
            k4[i] = h*F[i];
            theta[i] = theta_old[i] + 1.0/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
          }


        for (i=0;i<N;i++){
            if ((M/2) < k)
            {
             diff_t[i] = (theta[i] - theta_old[i]);
            }

             if (theta[i] > 2.0*pi_n)
             {
          		theta[i] = theta[i] - 2.0*pi_n;
            }

             theta_old[i] = theta[i];  
        }

        if ((M/2) < k){
            for(i=0;i<N;i++){
                if (diff_t[i] > max_temp)
                {
                    max_temp  = diff_t[i];
                }

                if (diff_t[i] < min_temp)
                {
                    min_temp  = diff_t[i];
                }
            }

        }
      
        t = t+h;
      
    }

    
    tol = max_temp-min_temp;
    if (tol <= 0.001){
        D = 1;
    }
    
    return D;
}




__global__ void task(double *a, double *random_data, double *a_save, double *w, \
                     double h , int N, int M, double *a_lower_bound_update, \
                    double *a_upper_bound_update)
{
    const int i_c = blockDim.x*blockIdx.x + threadIdx.x;
    int i,j,ij;
    
    // every thread compute a sample of a matrix
    double a_new[N_global*N_global];
    for (i=0;i<N_global*N_global;i++){
            a_new[i] = 0.0;
    }

    int rand_ind, cnt0, cnt1;
    
    cnt0 = (i_c*(N-1)*N/2);
    cnt1 = 0;

    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++)
        {
            rand_ind = cnt0 + cnt1;
            a_new[j*(N+1)+i] = a_lower_bound_update[j*N+i]+ (a_upper_bound_update[j*(N)+i]-a_lower_bound_update[j*(N)+i])*random_data[rand_ind];
            a_new[i*(N+1)+j] = a_new[j*(N+1)+i];
            cnt1++;
        }
        

    }

    double c_lower = 0.0;
    double c_upper = 4.0;
    double c_0;
    c_0 = c_upper;
    int D;

    // Dichotomy approximation of MOCU(min force to keep them syn) for this sample
    for (ij=0;ij<18;ij++){
        for (i=0;i<N;i++){
          a_new[(i*(N+1))+N] = c_0;
          a_new[(N*(N+1))+i] = c_0;
        }


        D = mocu_comp(w, h, N+1, M, a_new);

        if (D == 1){  
            c_upper = (c_lower + c_upper) / 2.0;
            c_0 = c_upper;
          }
        else{  
            c_lower = c_upper;
            c_upper = c_lower + 2 * pow(0.5,ij);
            c_0 = c_upper;
          }

    }


    a_save[i_c] = (c_lower + c_upper) / 2.0;
    

}
"""
)

task = mod.get_function("task")

def MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update):

    blocks = 200
    block_size = np.int(K_max/blocks)

    # set the natural frequency of the additional oscillator to mean w
    w = np.append(w, np.mean(w))

    # set the natural frequency of the additional oscillator to mean w
    # w = np.append(w, 0)

    a_save = np.zeros(K_max).astype(np.float64)

    vec_a_lower = np.zeros(N*N).astype(np.float64)
    vec_a_upper = np.zeros(N*N).astype(np.float64)

    vec_a_lower = np.reshape(a_lower_bound_update, N*N)
    vec_a_upper = np.reshape(a_upper_bound_update, N*N)

    a = np.zeros((N+1)*(N+1)).astype(np.float64)
    theta = np.zeros(N+1).astype(np.float64)
    theta_old = np.zeros(N+1).astype(np.float64)

    rand_data = np.random.random(int((N-1)*N/2.0*K_max)).astype(np.float64)

    F = np.zeros(N+1).astype(np.float64)    
    k1 = np.zeros(N+1).astype(np.float64)
    k2 = np.zeros(N+1).astype(np.float64)
    k3 = np.zeros(N+1).astype(np.float64)
    k4 = np.zeros(N+1).astype(np.float64)
    diff_t = np.zeros((M)*(N+1)).astype(np.float64)




    task(drv.In(a), drv.In(rand_data), drv.Out(a_save), drv.In(w), 
        np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower), 
        drv.In(vec_a_upper), grid=(blocks,1), block=(block_size,1,1))
    

    #print(a_save)

    if min(a_save) == 0:
    	print("Non sync case exists")
    

    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max*0.005)
        uu = int(K_max*0.995)
        a_save = temp[ll-1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max*0.99)
        print('max a: %f' % a_star)
        if a_star > 7.9:
            raise ValueError("out of upper bound of binary search")
    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max)

    return MOCU_val