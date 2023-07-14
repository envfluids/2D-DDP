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

NX = 64 # For empty array initialization

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
sc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

# Variables for DSMAG
uu_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
uv_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vv_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

uuc_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
uvc_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vvc_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

uc_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vc_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

ucx_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vcx_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
ucy_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vcy_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

Sc1_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
Sc2_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
Sc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

hc1_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hc2_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hcc1_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hcc2_hat_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

hc1_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hc2_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hcc1_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
hcc2_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

uuc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
uvc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vvc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

uc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
vc_gpu = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

l11 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
l12 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
l22 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
m11 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
m12 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))
m22 = gpuarray.to_gpu(np.reshape(s,NX*(NX),order='F'))

dummy_scalar    = np.zeros([1,1],dtype = np.complex128)
a_gpu = gpuarray.to_gpu(np.reshape(dummy_scalar,1,order='F'))
b_gpu = gpuarray.to_gpu(np.reshape(dummy_scalar,1,order='F'))


# def SMAG(vt_gpu, kx_gpu, w1_hat_gpu, psiCurrent_hat_gpu, NX, TPB, NB, HNX, HNB, Delta, plan, ev_coeff_gpu):
#     S12_ker(s1_hat_gpu, s2_hat_gpu, kx_gpu, psiCurrent_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
#     cf.cufftExecZ2Z(plan, int(s1_hat_gpu.gpudata), int(s1_gpu.gpudata), cf.CUFFT_INVERSE)
#     cf.cufftExecZ2Z(plan, int(s2_hat_gpu.gpudata), int(s2_gpu.gpudata), cf.CUFFT_INVERSE)
#     Sc_ker(s_gpu, s1_gpu, s2_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    
#     # Calculate eddy viscosity
#     eddy_visc_ker(vt_gpu, s_gpu, block=(1,1,1), grid=(1,1,1))
#     # vt = vt_gpu.get()
#     # print(np.real(vt))
#     # S = s_gpu.get()
#     # print(np.mean(np.real(S)))

#     norm_eddy_visc_ker(vt_gpu, ev_coeff_gpu, block=(1,1,1), grid=(1,1,1)) # Scalar operation

#     return

def DSMAG(vt_gpu, kx_gpu, w1_hat_gpu, psiCurrent_hat_gpu, NX, dx, TPB, NB, HNX, HNB, Delta, plan, ev_coeff_gpu,
u1_gpu,v1_gpu,u1_hat_gpu,v1_hat_gpu,negRate_gpu):
    S12_ker(s1_hat_gpu, s2_hat_gpu, kx_gpu, psiCurrent_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    cf.cufftExecZ2Z(plan, int(s1_hat_gpu.gpudata), int(s1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(s2_hat_gpu.gpudata), int(s2_gpu.gpudata), cf.CUFFT_INVERSE)
    S_ker(s_gpu, s1_gpu, s2_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    Sc_ker(sc_gpu, s1_gpu, s2_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
   
    # Calculate CS coefficients
    compute_CS_smag(dx,NX,u1_gpu,v1_gpu,s1_gpu,s2_gpu,sc_gpu,kx_gpu,ev_coeff_gpu,u1_hat_gpu,v1_hat_gpu,negRate_gpu,plan)

    # Calculate eddy viscosity
    eddy_visc_ker(a_gpu, s_gpu, block=(1,1,1), grid=(1,1,1)) # Calculate |S|
    # vt_gpu = s_gpu
    # vt = vt_gpu.get()
    # print(np.real(vt))
    # S = s_gpu.get()
    # print(np.mean(np.real(S)))
    
    norm_eddy_visc_ker(vt_gpu, ev_coeff_gpu, a_gpu, block=(1,1,1), grid=(NX,NX,1)) # Scalar operation, cs*sqrt(|S|)
    

    return

def compute_CS_smag(dx,NX,u1_gpu,v1_gpu,s1_gpu,s2_gpu,s_gpu,kx_gpu,ev_gpu,u1_hat_gpu,v1_hat_gpu,negRate_gpu,plan):
    NX2 = NX*NX
    NX4 = NX2*NX2
    nxc = NX/2
   
    uu_gpu = u1_gpu*u1_gpu/NX4
    uv_gpu = u1_gpu*v1_gpu/NX4
    vv_gpu = v1_gpu*v1_gpu/NX4  

    cf.cufftExecZ2Z(plan, int(uu_gpu.gpudata), int(uu_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(uv_gpu.gpudata), int(uv_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(vv_gpu.gpudata), int(vv_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    
    spectralFilter_same_size(uu_hat_gpu,uuc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    spectralFilter_same_size(uv_hat_gpu,uvc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    spectralFilter_same_size(vv_hat_gpu,vvc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))

    spectralFilter_same_size(u1_hat_gpu,uc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    spectralFilter_same_size(v1_hat_gpu,vc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))

    firstDx(ucx_hat_gpu, kx_gpu, uc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    firstDy(ucy_hat_gpu, kx_gpu, uc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    firstDx(vcx_hat_gpu, kx_gpu, vc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    firstDy(vcy_hat_gpu, kx_gpu, vc_hat_gpu, block=(1,1,1), grid=(NX,NX,1))

    cf.cufftExecZ2Z(plan, int(ucx_hat_gpu.gpudata), int(Sc1_gpu.gpudata), cf.CUFFT_INVERSE) 
    Sc2_hat_gpu = 0.5*(ucy_hat_gpu + vcx_hat_gpu)
    cf.cufftExecZ2Z(plan, int(Sc2_hat_gpu.gpudata), int(Sc2_gpu.gpudata), cf.CUFFT_INVERSE) 
    Sc_ker(Sc_gpu, Sc1_gpu, Sc2_gpu, block=(1,1,1), grid=(NX,NX,1))
       
    hc1_gpu = s_gpu*s1_gpu/NX2
    hc2_gpu = s_gpu*s2_gpu/NX2
    cf.cufftExecZ2Z(plan, int(hc1_gpu.gpudata), int(hc1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(hc2_gpu.gpudata), int(hc2_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    spectralFilter_same_size(hc1_hat_gpu,hcc1_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    spectralFilter_same_size(hc2_hat_gpu,hcc2_hat_gpu, block=(1,1,1), grid=(NX,NX,1))
    cf.cufftExecZ2Z(plan, int(hcc1_hat_gpu.gpudata), int(hcc1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(hcc2_hat_gpu.gpudata), int(hcc2_gpu.gpudata), cf.CUFFT_INVERSE)
    

    cf.cufftExecZ2Z(plan, int(uuc_hat_gpu.gpudata), int(uuc_gpu.gpudata), cf.CUFFT_INVERSE) 
    cf.cufftExecZ2Z(plan, int(uvc_hat_gpu.gpudata), int(uvc_gpu.gpudata), cf.CUFFT_INVERSE) 
    cf.cufftExecZ2Z(plan, int(vvc_hat_gpu.gpudata), int(vvc_gpu.gpudata), cf.CUFFT_INVERSE) 
    cf.cufftExecZ2Z(plan, int(uc_hat_gpu.gpudata), int(uc_gpu.gpudata), cf.CUFFT_INVERSE) 
    cf.cufftExecZ2Z(plan, int(vc_hat_gpu.gpudata), int(vc_gpu.gpudata), cf.CUFFT_INVERSE) 

    l_ker(l11,uuc_gpu,uc_gpu,uc_gpu, block=(1,1,1), grid=(NX,NX,1))
    l_ker(l12,uvc_gpu,uc_gpu,vc_gpu, block=(1,1,1), grid=(NX,NX,1))
    l_ker(l22,vvc_gpu,vc_gpu,vc_gpu, block=(1,1,1), grid=(NX,NX,1))

    l11d = l11 - 1/2*(l11 + l22)
    l12d = l12
    l22d = l22 - 1/2*(l11 + l22)
    
    m_ker(m11,hcc1_gpu,Sc_gpu,Sc1_gpu, block=(1,1,1), grid=(NX,NX,1))
    m_ker(m12,hcc2_gpu,Sc_gpu,Sc2_gpu, block=(1,1,1), grid=(NX,NX,1))
    # m11 = 2.0*(hcc1_gpu/NX2-4.0*Sc_gpu*Sc1_gpu/NX2)
    # m12 = 2.0*(hcc2_gpu/NX2-4.0*Sc_gpu*Sc2_gpu/NX2)
    m22 = -m11
    
    aa = (l11d*m11 + 2.0*(l12d*m12) + l22d*m22)
    bb = (m11*m11 + 2.0*(m12*m12) + m22*m22)
    
    aa = 0.5*(aa+abs(aa))

    # eddy_visc_ker(a_gpu, aa, block=(1,1,1), grid=(1,1,1)) # Calculate |aa|
    # eddy_visc_ker(b_gpu, bb, block=(1,1,1), grid=(1,1,1)) # Calculate |bb|

    # scalar_division(a_gpu, b_gpu, ev_gpu, block=(1,1,1), grid=(1,1,1))
    
    division_ker(ev_gpu, aa, bb, block=(1,1,1), grid=(NX,NX,1))
    # scaleCs(ev_gpu, negRate_gpu, block=(1,1,1), grid=(NX,NX,1))

    # Debug testing
    # l11_r = np.real(np.reshape(l11.get(),[NX,NX],order='F'))
    # l12_r = np.real(np.reshape(l12.get(),[NX,NX],order='F'))
    # l22_r = np.real(np.reshape(l22.get(),[NX,NX],order='F'))
    # m11_r = np.real(np.reshape(m11.get(),[NX,NX],order='F'))
    # m12_r = np.real(np.reshape(m12.get(),[NX,NX],order='F'))
    # ev    = np.real(np.reshape(ev_gpu.get(),[NX,NX],order='F'))
    # uuc_r = np.real(np.reshape(uuc_gpu.get(),[NX,NX],order='F'))/NX2
    # aa_r = np.real(np.reshape(aa.get(),[NX,NX],order='F'))
    # bb_r = np.real(np.reshape(bb.get(),[NX,NX],order='F'))
    # savemat('test_Python.mat',dict([('l11Py',l11_r),('m11Py',m11_r),('l22Py',l22_r),('l12Py',l12_r),('m12Py',m12_r),('evPy',ev),('uucPy',uuc_r),('aaPy',aa_r),('bbPy',bb_r)]))
    

    return