#--> Import python libraries
import numpy as np
#from matplotlib import pyplot as plt
import math
# from cheb import cheb
import scipy
from scipy import sparse
from scipy.sparse.linalg import inv
import statistics

# Save data as .mat file
import h5py
import hdf5storage
from scipy.io import loadmat,savemat
import time as runtime

# Import CUDA function
from cuda_functions import *

NX = 256 # For empty array initialization

# Physics-based SGS models stuff
s1_hat = np.zeros([NX,NX],dtype=np.complex128)
s2_hat = np.zeros([NX,NX],dtype=np.complex128)
s1     = np.zeros([NX,NX],dtype=np.complex128)
s2     = np.zeros([NX,NX],dtype=np.complex128)
s      = np.zeros([NX,NX],dtype=np.complex128)

s1_hat_gpu = gpuarray.to_gpu(np.reshape(s1_hat,NX*(NX),order='F'))
s2_hat_gpu = gpuarray.to_gpu(np.reshape(s2_hat,NX*(NX),order='F'))
s1_gpu = gpuarray.to_gpu(np.reshape(s1,NX*(NX),order='F'))
s2_gpu = gpuarray.to_gpu(np.reshape(s2,NX*(NX),order='F'))
s_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

def SMAG(vt_gpu, kx_gpu, w1_hat_gpu, psiCurrent_hat_gpu, NX, TPB, NB, HNX, HNB, Delta, plan, ev_coeff_gpu):
    S12_ker(s1_hat_gpu, s2_hat_gpu, kx_gpu, psiCurrent_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    cf.cufftExecZ2Z(plan, int(s1_hat_gpu.gpudata), int(s1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(s2_hat_gpu.gpudata), int(s2_gpu.gpudata), cf.CUFFT_INVERSE)
    S_ker(s_gpu, s1_gpu, s2_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    
    # Calculate eddy viscosity
    eddy_visc_ker(vt_gpu, s_gpu, block=(1,1,1), grid=(1,1,1))
    # vt = vt_gpu.get()
    # print(np.real(vt))
    # S = s_gpu.get()
    # print(np.mean(np.real(S)))

    norm_eddy_visc_ker(vt_gpu, ev_coeff_gpu, block=(1,1,1), grid=(1,1,1)) # Scalar operation

    return