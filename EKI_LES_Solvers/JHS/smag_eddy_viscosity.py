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
NX2 = NX*NX


meanvt    = np.zeros([1,1],dtype = np.complex128)
meanvb1_gpu = gpuarray.to_gpu(np.reshape(meanvt,1,order='F'))
meanvb2_gpu = gpuarray.to_gpu(np.reshape(meanvt,1,order='F'))


def SMAG(vt_gpu, vb_gpu, kx_gpu, w1_hat_gpu, psiCurrent_hat_gpu, diffu_hat_gpu, NX, TPB, NB, HNX, HNB, Delta, plan, ev_coeff_gpu,
s1_hat_gpu, s2_hat_gpu, s1_gpu, s2_gpu, s_gpu, hyperdiffu_hat_gpu, diffu_gpu, hyperdiffu_gpu, Fsmag, psi_gpu, meanvb1_gpu, meanvb2_gpu, CB_gpu):
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

    norm_eddy_visc_ker(vt_gpu, ev_coeff_gpu, block=(1,1,1), grid=(1,1,1)) # Scalar operation (vt_gpu[0] is ve)

    # Backscattering model
    diffusion(hyperdiffu_hat_gpu, kx_gpu, diffu_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(hyperdiffu_hat_gpu.gpudata), int(hyperdiffu_gpu.gpudata), cf.CUFFT_INVERSE)
    Fsmag_ker(vt_gpu, hyperdiffu_gpu, Fsmag, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(psiCurrent_hat_gpu.gpudata), int(psi_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(diffu_hat_gpu.gpudata), int(diffu_gpu.gpudata), cf.CUFFT_INVERSE)
    psi_gpu = psi_gpu/NX2
    diffu_gpu = diffu_gpu/NX2
    vb1 = -1*psi_gpu*Fsmag
    vb2 = psi_gpu*diffu_gpu

    eddy_visc_ker(meanvb1_gpu, vb1, block=(1,1,1), grid=(1,1,1))
    eddy_visc_ker(meanvb2_gpu, vb2, block=(1,1,1), grid=(1,1,1))

    vb_ker(vb_gpu, meanvb1_gpu, meanvb2_gpu, CB_gpu,  block=(1,1,1), grid=(1,1,1) )


    '''
    # Debug stuff
    vb = vb_gpu.get()
    meanvb1 = meanvb1_gpu.get()
    meanvb2 = meanvb2_gpu.get()

    vb1_cpu = vb1.get()
    vb2_cpu = vb2.get()

    Fsmag_cpu = Fsmag.get()


    print("vb:")
    print(vb)
    print("meanvb1:")
    print(meanvb1)
    print("meanvb2:")
    print(meanvb2)

    print(np.mean(vb1_cpu))
    print(np.mean(vb2_cpu))

    print(np.mean(Fsmag_cpu))
    vt = vt_gpu.get()
    print("vt:")
    print(vt)
    '''

    return