# Solve 2D turbulence by Fourier-Fourier pseudo-spectral method
# Navier-Stokes equation is in the vorticity-stream function form

#--> Import pyCUDA
import pycuda.autoinit
import pycuda.driver as drv 
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from skcuda import cufft as cf
# import skcuda.linalg as skculinalg
# import cupy as cp

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

#import matplotlib.pyplot as plt
# Import NC saver
from savenc import *


# Tensorflow stuff
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# from keras import layers

# from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
#                           Flatten, concatenate, Activation, Reshape, \
#                           UpSampling2D,ZeroPadding2D
# from keras.layers import Dense
# from keras import Sequential
# import h5py
# import keras
# from pylab import plt
# from matplotlib import cm
# from scipy.io import loadmat,savemat
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ker = SourceModule("""
#include <pycuda-complex.hpp>
#include "cuComplex.h"
#include <cufft.h>
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val){
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparision to avoid hang in case of NaN (since NaN!=NaN)
		} while(assumed != old);
		return __longlong_as_double(old);	
}
#endif

const int NX = 4096;
const int NNX = 256; // Filter size
const int NX2 = 16777216; // NX^2
__device__ double dt = 1e-5;
__device__ double nu = 3.333333333e-6;
__device__ double alpha = 0.1;

__global__ void initialization_kernel(cufftDoubleComplex *u0, cufftDoubleComplex *v0, cufftDoubleComplex *w0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    v0[i*NX+j].x = -kx[j]*psi[j+i*NX].y;
    v0[i*NX+j].y = kx[j]*psi[j+i*NX].x;

    u0[i*NX+j].x = kx[i]*psi[j+i*NX].y;
    u0[i*NX+j].y = -kx[i]*psi[j+i*NX].x;

    w0[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].x;
    w0[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].y;
}

__global__ void iniW1_kernel(cufftDoubleComplex *w0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    w0[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].x;
    w0[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].y;
}

__global__ void UV_kernel(cufftDoubleComplex *u0, cufftDoubleComplex *v0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    v0[i*NX+j].x = -kx[j]*psi[j+i*NX].y;
    v0[i*NX+j].y = kx[j]*psi[j+i*NX].x;

    u0[i*NX+j].x = kx[i]*psi[j+i*NX].y;
    u0[i*NX+j].y = -kx[i]*psi[j+i*NX].x;
}

__global__ void convection2_kernel(cufftDoubleComplex *u, cufftDoubleComplex *w, \
    cufftDoubleComplex *convec)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    convec[i*NX+j].x = (u[i*NX+j].x / NX2) * (w[i*NX+j].x / NX2);
    convec[i*NX+j].y = 0.0;//u[i*NX+j].y * w[i*NX+j].y / NX2 / NX2;
}

__global__ void diffusion_kernel(cufftDoubleComplex *diffu, double *kx, cufftDoubleComplex *w1)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    diffu[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*w1[j+i*NX].x;
    diffu[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*w1[j+i*NX].y;
}

__global__ void convection3_kernel(cufftDoubleComplex *conu1, cufftDoubleComplex *conv1,\
    cufftDoubleComplex *conu0, cufftDoubleComplex *conv0, cufftDoubleComplex *convN, double *kx, cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = 0.5*dt*(kx[k]*(1.5*conu1[j*NX+k].y - 0.5*conu0[j*NX+k].y) + kx[j]*(1.5*conv1[j*NX+k].y - 0.5*conv0[j*NX+k].y)\
        + convN[j*NX+k].x);
    convec[j*NX+k].y = 0.5*dt*(kx[k]*(-1.5*conu1[j*NX+k].x + 0.5*conu0[j*NX+k].x) + kx[j]*(-1.5*conv1[j*NX+k].x + 0.5*conv0[j*NX+k].x)\
        + convN[j*NX+k].y);    
}

__global__ void convection4_kernel(cufftDoubleComplex *wx, cufftDoubleComplex *wy, double *kx,\
     cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    wx[i*NX+j].x = -kx[j]*w[j+i*NX].y;
    wx[i*NX+j].y = kx[j]*w[j+i*NX].x;

    wy[i*NX+j].x = -kx[i]*w[j+i*NX].y;
    wy[i*NX+j].y = kx[i]*w[j+i*NX].x;
}

__global__ void firstDx_kernel(cufftDoubleComplex *wx, double *kx, cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    wx[i*NX+j].x = -kx[j]*w[j+i*NX].y;
    wx[i*NX+j].y = kx[j]*w[j+i*NX].x;
}

__global__ void firstDy_kernel(cufftDoubleComplex *wy, double *kx,\
     cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    wy[i*NX+j].x = -kx[i]*w[j+i*NX].y;
    wy[i*NX+j].y = kx[i]*w[j+i*NX].x;
}

__global__ void convection5_kernel(cufftDoubleComplex *u1,cufftDoubleComplex *v1,cufftDoubleComplex *u0,cufftDoubleComplex *v0, \
    cufftDoubleComplex *w1x,cufftDoubleComplex *w1y,cufftDoubleComplex *w0x,cufftDoubleComplex *w0y,\
        cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = (-1.5*((u1[j*NX+k].x/NX2)*(w1x[j*NX+k].x/NX2) + (v1[j*NX+k].x/NX2)*(w1y[j*NX+k].x/NX2))\
        +0.5*((u0[j*NX+k].x/NX2)*(w0x[j*NX+k].x/NX2) + (v0[j*NX+k].x/NX2)*(w0y[j*NX+k].x/NX2)));
}

__global__ void convection6_kernel(cufftDoubleComplex *u1,cufftDoubleComplex *v1, cufftDoubleComplex *w1x,cufftDoubleComplex *w1y, cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = (u1[j*NX+k].x/NX2)*(w1x[j*NX+k].x/NX2) + (v1[j*NX+k].x/NX2)*(w1y[j*NX+k].x/NX2);
}

__global__ void RHS_kernel(cufftDoubleComplex *convec,cufftDoubleComplex *diffu,cufftDoubleComplex *w1,\
    cufftDoubleComplex *RHS)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    RHS[j*NX+k].x = convec[j*NX+k].x + 0.5*dt*nu*diffu[j*NX+k].x + w1[j*NX+k].x;
    RHS[j*NX+k].y = convec[j*NX+k].y + 0.5*dt*nu*diffu[j*NX+k].y + w1[j*NX+k].y;
}

__global__ void LHS_kernel(double *kx, cufftDoubleComplex *RHS, cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    w[i*NX+j].x = RHS[i*NX+j].x / (1.0+dt*alpha+0.5*dt*nu*((kx[j]*kx[j] + kx[i]*kx[i])));
    w[i*NX+j].y = RHS[i*NX+j].y / (1.0+dt*alpha+0.5*dt*nu*((kx[j]*kx[j] + kx[i]*kx[i])));

}
__global__ void psiTemp_kernel(cufftDoubleComplex *w, double *kx, cufftDoubleComplex *psiTemp)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    
    if (j==0 && i==0){
        psiTemp[i*NX+j].x = 0.0;
        psiTemp[i*NX+j].y = 0.0;
    }
    else{
        psiTemp[i*NX+j].x =  - w[i*NX+j].x / (kx[j]*kx[j] + kx[i]*kx[i]);
        psiTemp[i*NX+j].y =  - w[i*NX+j].y / (kx[j]*kx[j] + kx[i]*kx[i]);
    }
}

__global__ void update_kernel(cufftDoubleComplex *psiPrevious_hat_gpu,cufftDoubleComplex *psiCurrent_hat_gpu,\
        cufftDoubleComplex *psiTemp_gpu)
    {
        unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int k = blockIdx.y;
        psiPrevious_hat_gpu[j*NX+k].x = psiCurrent_hat_gpu[j*NX+k].x;
        psiPrevious_hat_gpu[j*NX+k].y = psiCurrent_hat_gpu[j*NX+k].y;
        psiCurrent_hat_gpu[j*NX+k].x  = psiTemp_gpu[j*NX+k].x;
        psiCurrent_hat_gpu[j*NX+k].y  = psiTemp_gpu[j*NX+k].y;
    }

__global__ void updateu_kernel(cufftDoubleComplex *u0_hat_gpu, cufftDoubleComplex *v0_hat_gpu,\
        cufftDoubleComplex *u1_hat_gpu,cufftDoubleComplex *v1_hat_gpu)
    {
        unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int j = blockIdx.y;
        u0_hat_gpu[i*NX+j].x = u1_hat_gpu[i*NX+j].x;
        u0_hat_gpu[i*NX+j].y = u1_hat_gpu[i*NX+j].y;
        v0_hat_gpu[i*NX+j].x = v1_hat_gpu[i*NX+j].x;
        v0_hat_gpu[i*NX+j].y = v1_hat_gpu[i*NX+j].y;
    }

__global__ void updatew_kernel(cufftDoubleComplex *w0_hat_gpu, cufftDoubleComplex *w1_hat_gpu)
    {
        unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int j = blockIdx.y;
        w0_hat_gpu[i*NX+j].x = w1_hat_gpu[i*NX+j].x;
        w0_hat_gpu[i*NX+j].y = w1_hat_gpu[i*NX+j].y;
    }

__global__ void DDP_kernel(cufftDoubleComplex *RHS, cufftDoubleComplex *pred)
{
    unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int k = blockIdx.z;

    RHS[j*NX+k].x = RHS[j*NX+k].x  + 1*dt*pred[j*NX+k].x;
    RHS[j*NX+k].y = RHS[j*NX+k].y  + 1*dt*pred[j*NX+k].y;
}

__global__ void spectralFilter_2D_kernel(cufftDoubleComplex *data, cufftDoubleComplex *data_F)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int HNNX = NNX/2;
    if (i<=HNNX && j<=HNNX){
        data_F[i*NNX+j].x = data[i*NX+j].x;
        data_F[i*NNX+j].y = data[i*NX+j].y;
    }
    else if (i>(NX-HNNX) && j<=HNNX){
        data_F[(NNX+i-NX)*NNX+j].x = data[i*NX+j].x;
        data_F[(NNX+i-NX)*NNX+j].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i<=HNNX){
        data_F[i*NNX+(NNX+j-NX)].x = data[i*NX+j].x;
        data_F[i*NNX+(NNX+j-NX)].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i>(NX-HNNX)){
        data_F[(NNX+i-NX)*NNX+(NNX+j-NX)].x = data[i*NX+j].x;
        data_F[(NNX+i-NX)*NNX+(NNX+j-NX)].y = data[i*NX+j].y;
    }
}

__global__ void addConvecC_kernel(cufftDoubleComplex *conu1, cufftDoubleComplex *conv1, double *kx, cufftDoubleComplex *convecC)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convecC[j*NX+k].x = -(kx[k]*conu1[j*NX+k].y  + kx[j]*conv1[j*NX+k].y);
    convecC[j*NX+k].y = (kx[k]*conu1[j*NX+k].x  + kx[j]*conv1[j*NX+k].x);
}

__global__ void GaussianFilter_2D_kernel(cufftDoubleComplex *u, cufftDoubleComplex *uG, double *Gk)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    uG[j*NX+k].x = Gk[j*NX+k]*u[j*NX+k].x;
    uG[j*NX+k].y = Gk[j*NX+k]*u[j*NX+k].y;
}


""")
initialization = ker.get_function("initialization_kernel")
iniW1 = ker.get_function("iniW1_kernel")
UV_ker = ker.get_function("UV_kernel")
convection2 = ker.get_function("convection2_kernel")
convection3 = ker.get_function("convection3_kernel")
convection4 = ker.get_function("convection4_kernel")
convection5 = ker.get_function("convection5_kernel")
convection6 = ker.get_function("convection6_kernel")
diffusion = ker.get_function("diffusion_kernel")
RHS_ker = ker.get_function("RHS_kernel")
LHS_ker = ker.get_function("LHS_kernel")
psiTemp_ker = ker.get_function("psiTemp_kernel")
update = ker.get_function("update_kernel")
updateu = ker.get_function("updateu_kernel")
updatew = ker.get_function("updatew_kernel")
DDP_ker = ker.get_function("DDP_kernel")
spectralFilter = ker.get_function("spectralFilter_2D_kernel")
GaussianFilter = ker.get_function("GaussianFilter_2D_kernel")
addConvecC = ker.get_function("addConvecC_kernel")
firstDx = ker.get_function("firstDx_kernel")
firstDy = ker.get_function("firstDy_kernel")




readTrue = 0
saveTrue = 1
NSAVE = 200 #10000
NNSAVE = 400 #100
maxit = NSAVE*NNSAVE

dt    = 1e-5
nu    = 3.333333333e-6
rho   = 1
Re    = 1.0/nu

Lx    = 2*math.pi
NX    = 4096
HNX   = 4096  # NX/2+1

# Neural network setup
# pool_size = 2
# drop_prob = 0.0
# conv_activation = 'relu'
# Nlat = NX
# Nlon = NX
# n_channels = 2
# def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

#     model = keras.Sequential([

#             ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon,Nlat,n_channels)),
#             layers.MaxPooling2D(pool_size=pool_size),
#             Dropout(drop_prob),
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.MaxPooling2D(pool_size=pool_size),
#             # end "encoder"
    
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.MaxPooling2D(pool_size=pool_size),

#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.MaxPooling2D(pool_size=pool_size),

#             # dense layers (flattening and reshaping happens automatically)
#             ] + [keras.layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +

#             [


#             # start "Decoder" (mirror of the encoder above)
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.UpSampling2D(size=pool_size),
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.UpSampling2D(size=pool_size),
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.UpSampling2D(size=pool_size),
#             Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
#             layers.UpSampling2D(size=pool_size),
#             layers.Convolution2D(1, kernel_size, padding='same', activation=None)
#             ]
#             )
#     optimizer= keras.optimizers.adam(lr=lr)


#     model.compile(loss='mean_squared_error', optimizer = optimizer)

#     return model

# params = {'conv_depth': 32, 'hidden_size': 500,
#               'kernel_size': 6, 'lr': 0.00001, 'n_hidden_layers': 0}

# model = build_model(**params)
# model.load_weights('./weights_cnn_KT_NX32') # load model weight from last time

#--> Computational variables
TPB = 1
NB  = NX//TPB
HNB = NB//2
HNB = HNX
TPBx = 1

## No mirror symmetry
HNB = NB
HNX = NX

dx    = Lx/NX
x     = np.linspace(0, Lx-dx, num=NX)
kx    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),np.arange((-NX/2+1),0,dtype=np.float64)))

# Create the tensor product mesh
[Y,X]       = np.meshgrid(x,x)
[Ky,Kx]     = np.meshgrid(kx,kx)
Ksq         = (Kx**2 + Ky**2)
Kabs        = np.sqrt(Ksq)
invKsq      = 1/Ksq
invKsq[0,0] = 0

# Coarse grid
NNX         = 256
kkx         = (2*math.pi/Lx)*np.concatenate((np.arange(0,NNX/2+1,dtype=np.float64),np.arange((-NNX/2+1),0,dtype=np.float64)))
[KKy,KKx]   = np.meshgrid(kkx,kkx)
KKsq        = (KKx**2 + KKy**2)

# Gaussian filter
Delta = 2*Lx/NNX
Gk = np.exp(-Ksq*Delta**2/24)

if readTrue == 0: # Initialization

    # Vorticity and stream function
    # nIni            = 2
    # psi             = 1*(np.cos(2*math.pi/(Lx/nIni)*X)+np.cos(2*math.pi/(Lx/nIni)*Y))
    
    # psi_hat         = np.zeros([NX,NX],dtype=np.complex128)
    # psi_hat[0,4]    = 1+1j#np.random.randn(1) + 1j*np.random.randn(1)
    # psi_hat[1,1]    = 1+1j#np.random.randn(1) + 1j*np.random.randn(1)
    # psi_hat[20,20]    = 1+1j
    # # psi_hat[3,0]    = np.random.randn(1) + 1j*np.random.randn(1)
    # psi             = np.real(np.fft.ifft2(psi_hat))
    # psi             = psi/np.max(psi)
    
    # psi_hat         = np.fft.fft2(psi)

    # w1              = np.random.randn(NX,NX)
    # w1              = w1/np.max(w1)
    # # w1              = np.zeros([NX,NX])

    # w1_hat          = np.fft.fft2(w1)
    # w1_hat[2,2]     = 1
    # psi_hat         = -w1_hat*invKsq

    # Williams initialization
    # fk = Ksq!=0
    # ckappa = np.zeros_like(Ksq)
    # ckappa[fk] = (np.sqrt(Ksq[fk])*(1. + (Ksq[fk]/36.)**2))**-0.5
    # psi_hat = np.random.randn(NX,NX)*ckappa +1j*np.random.randn(NX,NX)*ckappa
    # Psi = np.real(np.fft.ifft2(psi_hat))
    # Psi = Psi - Psi.mean()
    # psi_hat = np.fft.fft2(Psi)
    # u_K = (-1j*Ky)*psi_hat
    # v_K = (1j*Kx)*psi_hat
    # u = np.real(np.fft.ifft2(u_K))
    # v = np.real(np.fft.ifft2(v_K))
    # Ekin = 0.5*(u**2+v**2)
    # EK = np.fft.fft2(Ekin)
    # psi_hat = psi_hat/np.sqrt(2*EK.sum())

    # Initial condition
    kp = 10.0
    A  = 4*np.power(kp,(-5))/(3*np.pi)
    absK = np.sqrt(Kx*Kx+Ky*Ky)
    Ek = A*np.power(absK,4)*np.exp(-np.power(absK/kp,2))
    coef1 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
    coef2 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2

    perturb = np.zeros([NX,NX])
    perturb[0:NX//2+1, 0:NX//2+1] = coef1[0:NX//2+1, 0:NX//2+1]+coef2[0:NX//2+1, 0:NX//2+1]
    perturb[NX//2+1:, 0:NX//2+1] = coef2[NX//2-1:0:-1, 0:NX//2+1] - coef1[NX//2-1:0:-1, 0:NX//2+1]
    perturb[0:NX//2+1, NX//2+1:] = coef1[0:NX//2+1, NX//2-1:0:-1] - coef2[0:NX//2+1, NX//2-1:0:-1]
    perturb[NX//2+1:, NX//2+1:] = -(coef1[NX//2-1:0:-1, NX//2-1:0:-1] + coef2[NX//2-1:0:-1, NX//2-1:0:-1])
    perturb = np.exp(1j*perturb)

    w1_hat = np.sqrt(absK/np.pi*Ek)*perturb*np.power(NX,2)


    psi_hat         = -w1_hat*invKsq
    psiPrevious_hat = psi_hat.astype(np.complex128)
    psiCurrent_hat  = psi_hat.astype(np.complex128)
    time   = 0.0
    slnW = []
else:
    data_Poi = loadmat('data_end_LDC_Poi_gpu_Kraichnan.mat')
    
    # data_Poi = h5py.File('data_end_LDC_Poi.mat', 'r')
    psiCurrent_hat    = data_Poi['psiCurrent_hat']
    psiPrevious_hat   = data_Poi['psiPrevious_hat']

    # Add noise to create an ensemble of ics
    '''
    import random
    n = random.randint(1,10)
    Xi = random.uniform(1E-4, 1E-3) 
    Fk = -n*Xi*np.cos(n*Y)-n*Xi*np.cos(n*X)
    Fk = np.fft.fft2(Fk)
    print("n")
    print(n)
    print("Xi")
    print(Xi)
    psiCurrent_hat = psiCurrent_hat + (np.reshape(Fk,NX*(NX),order='F')).astype(np.complex128)
    psiPrevious_hat = psiPrevious_hat + (np.reshape(Fk,NX*(NX),order='F')).astype(np.complex128)
    '''

    # data_Poi = loadmat('data_filtered_NX32.mat')
    # psiCurrent_hat    = data_Poi['psiCurrent_hat'].T
    # psiPrevious_hat   = data_Poi['psiPrevious_hat'].T

    # data_Pi = h5py.File('slnPI.mat', 'r')
    # SGS    = data_Pi['slnPI']

    # time   = data_Poi['time']
    #slnW   = data_Poi['slnW']
    # time = np.asarray(time)
    # time = time[0]
    time = 0.0
    slnW = []

# Stochastic forcing
'''
kf = 9
b = 4
indx = []
indy = []
for ki in range(NX):
    for kj in range(NX//2):
        if (Kabs[ki,kj]<(kf+b) and Kabs[ki,kj]>(kf-b)):
            indx.append(ki)
            indy.append(kj)

AmF = 2*NX**2
R = 0.5
Fk = np.zeros([NX,NX],dtype=np.complex128)
Fk_hat = np.zeros([NX,NX],dtype=np.complex128)
for i in range(len(indx)):
    ki = indx[i]
    kj = indy[i]
    theta =  np.random.uniform(low=0.0, high=2*np.pi)
    Fk[ki,kj] = AmF*np.exp(1j*theta)
    
    if (ki != 0 and kj != 0):
        Fk[NX-ki, NX-kj] = np.conj(Fk[ki,kj])
    elif (ki==0):
        Fk[ki, NX-kj] = np.conj(Fk[ki,kj])
    elif (kj==0):
        Fk[NX-ki, kj] = np.conj(Fk[ki,kj])
'''
# Deterministic forcing
n = 25
Xi = 1
Fk = -n*Xi*np.cos(n*Y)-n*Xi*np.cos(n*X)
Fk = np.fft.fft2(Fk)

Force_gpu   = gpuarray.to_gpu(np.reshape(Fk,NX*(NX),order='F'))

# savemat('Fk.mat',dict([('FkPython',np.fft.ifft2(Fk))]))

# import matplotlib.pyplot as plt
# plt.contourf(np.fft.ifft2(Fk))
# plt.show()
# End forcing

print('Re = ', Re)
plan  = cf.cufftPlan2d(NX, NX, cf.CUFFT_Z2Z)

plan2 = cf.cufftPlan2d(NNX, NNX, cf.CUFFT_Z2Z)

slnU = np.zeros([NX,NNSAVE])
slnV = np.zeros([NX,NNSAVE])
slnPsi = np.zeros([NNX,NNX,NNSAVE])
slnWor = np.zeros([NNX,NNX,NNSAVE])
slnPsi_ss = np.zeros([NNX,NNX,NNSAVE])
slnWor_ss = np.zeros([NNX,NNX,NNSAVE])

#slnWorDNS = np.zeros([NX,NX,NNSAVE])
#slnPsiDNS = np.zeros([NX,NX,NNSAVE])
#slnUDNS = np.zeros([NX,NX,NNSAVE])
#slnVDNS = np.zeros([NX,NX,NNSAVE])
slnPI = np.zeros([NNX,NNX,NNSAVE])
slnuF = np.zeros([NNX,NNX,NNSAVE])
slnvF = np.zeros([NNX,NNX,NNSAVE])
slnStress1 = np.zeros([NNX,NNX,NNSAVE])
slnStress2 = np.zeros([NNX,NNX,NNSAVE])
slnStress3 = np.zeros([NNX,NNX,NNSAVE])
slnForce1 = np.zeros([NNX,NNX,NNSAVE])
slnForce2 = np.zeros([NNX,NNX,NNSAVE])

Energy = np.zeros([NNSAVE])
Enstrophy = np.zeros([NNSAVE])

#sampling_matrix_1 = np.zeros([21,NNX*NNX,NNSAVE],dtype=np.float64)
#sampling_matrix = np.zeros([21,NNX,NNX],dtype=np.float64)
#SGS = np.zeros([NNX*NNX,NNSAVE],dtype=np.float64)

onePython = np.zeros([NNSAVE])
count = 0

start_time = runtime.time()
for it in range(1,maxit+1):
    if it == 1:
        # On the first iteration
        w1_hat = np.zeros([NX,NX],dtype=np.complex128)
        u1_hat = np.zeros([NX,NX],dtype=np.complex128)
        v1_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0_hat = np.zeros([NX,NX],dtype=np.complex128)
        u0_hat = np.zeros([NX,NX],dtype=np.complex128)
        v0_hat = np.zeros([NX,NX],dtype=np.complex128)
        diffu_hat = np.zeros([NX,NX],dtype=np.complex128)
        diffu = np.zeros([NX,NX],dtype=np.complex128)
        convec = np.zeros([NX,NX],dtype=np.complex128)
        convec_hat = np.zeros([NX,NX],dtype=np.complex128)
        RHS = np.zeros([NX,NX],dtype=np.complex128)
        psiTemp = np.zeros([NX,NX],dtype=np.complex128)

        u1 = np.zeros([NX,NX],dtype=np.complex128)
        u0 = np.zeros([NX,NX],dtype=np.complex128)
        v1 = np.zeros([NX,NX],dtype=np.complex128)
        v0 = np.zeros([NX,NX],dtype=np.complex128)

        w1 = np.zeros([NX,NX],dtype=np.complex128)
        w0 = np.zeros([NX,NX],dtype=np.complex128)

        w1x_hat = np.zeros([NX,NX],dtype=np.complex128)
        w1y_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0x_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0y_hat = np.zeros([NX,NX],dtype=np.complex128)
        w1x = np.zeros([NX,NX],dtype=np.complex128)
        w1y = np.zeros([NX,NX],dtype=np.complex128)
        w0x = np.zeros([NX,NX],dtype=np.complex128)
        w0y = np.zeros([NX,NX],dtype=np.complex128)

        psi1 = np.zeros([NX,NX],dtype=np.complex128)

        conu1 = np.zeros([NX,NX],dtype=np.complex128)
        conv1 = np.zeros([NX,NX],dtype=np.complex128)
        conu0 = np.zeros([NX,NX],dtype=np.complex128)
        conv0 = np.zeros([NX,NX],dtype=np.complex128)
        conu1_hat = np.zeros([NX,NX],dtype=np.complex128)
        conv1_hat = np.zeros([NX,NX],dtype=np.complex128)
        conu0_hat = np.zeros([NX,NX],dtype=np.complex128)
        conv0_hat = np.zeros([NX,NX],dtype=np.complex128)

        convN_hat = np.zeros([NX,NX],dtype=np.complex128)

        K_filter = np.zeros([NNX,NNX],dtype = np.complex128)

        UU = np.zeros([NX,NX],dtype=np.complex128)
        UV = np.zeros([NX,NX],dtype=np.complex128)
        VV = np.zeros([NX,NX],dtype=np.complex128)
        UU_hat = np.zeros([NX,NX],dtype=np.complex128)
        UV_hat = np.zeros([NX,NX],dtype=np.complex128)
        VV_hat = np.zeros([NX,NX],dtype=np.complex128)
        UU_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UV_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        VV_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UU_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UV_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        VV_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)

        UUx_hat = np.zeros([NX,NX],dtype=np.complex128)
        UVx_hat = np.zeros([NX,NX],dtype=np.complex128)
        UVy_hat = np.zeros([NX,NX],dtype=np.complex128)
        VVy_hat = np.zeros([NX,NX],dtype=np.complex128)

        UUx_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UVx_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UVy_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        VVy_filter = np.zeros([NNX,NNX],dtype=np.complex128)

        UUx_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UVx_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        UVy_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        VVy_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)

        ux = np.zeros([NX,NX],dtype=np.complex128)
        vx = np.zeros([NX,NX],dtype=np.complex128)
        uy = np.zeros([NX,NX],dtype=np.complex128)
        vy = np.zeros([NX,NX],dtype=np.complex128)
        
        ux_hat = np.zeros([NX,NX],dtype=np.complex128)
        vx_hat = np.zeros([NX,NX],dtype=np.complex128)
        uy_hat = np.zeros([NX,NX],dtype=np.complex128)
        vy_hat = np.zeros([NX,NX],dtype=np.complex128)

        convec1 = np.zeros([NX,NX],dtype=np.complex128)
        convec2 = np.zeros([NX,NX],dtype=np.complex128)

        convec1_hat = np.zeros([NX,NX],dtype=np.complex128)
        convec2_hat = np.zeros([NX,NX],dtype=np.complex128)

        convec1_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        convec2_hat_filter = np.zeros([NNX,NNX],dtype=np.complex128)

        convec1_filter = np.zeros([NNX,NNX],dtype=np.complex128)
        convec2_filter = np.zeros([NNX,NNX],dtype=np.complex128)

        dummy = np.zeros([NX,NX],dtype=np.complex128)

        # GPU arrays

        w1_hat_gpu = gpuarray.to_gpu(np.reshape(w1_hat,NX*(NX),order='F'))
        u1_hat_gpu = gpuarray.to_gpu(np.reshape(u1_hat,NX*(NX),order='F'))
        v1_hat_gpu = gpuarray.to_gpu(np.reshape(v1_hat,NX*(NX),order='F'))
        w0_hat_gpu = gpuarray.to_gpu(np.reshape(w0_hat,NX*(NX),order='F'))
        u0_hat_gpu = gpuarray.to_gpu(np.reshape(u0_hat,NX*(NX),order='F'))
        v0_hat_gpu = gpuarray.to_gpu(np.reshape(v0_hat,NX*(NX),order='F'))
        w1_gpu = gpuarray.to_gpu(np.reshape(w1,NX*(NX),order='F'))
        w0_gpu = gpuarray.to_gpu(np.reshape(w0,NX*(NX),order='F'))
        conu1_gpu = gpuarray.to_gpu(np.reshape(conu1,NX*(NX),order='F'))
        conv1_gpu = gpuarray.to_gpu(np.reshape(conv1,NX*(NX),order='F'))
        conu0_gpu = gpuarray.to_gpu(np.reshape(conu0,NX*(NX),order='F'))
        conv0_gpu = gpuarray.to_gpu(np.reshape(conv0,NX*(NX),order='F'))
        conu1_hat_gpu = gpuarray.to_gpu(np.reshape(conu1_hat,NX*(NX),order='F'))
        conv1_hat_gpu = gpuarray.to_gpu(np.reshape(conv1_hat,NX*(NX),order='F'))
        conu0_hat_gpu = gpuarray.to_gpu(np.reshape(conu0_hat,NX*(NX),order='F'))
        conv0_hat_gpu = gpuarray.to_gpu(np.reshape(conv0_hat,NX*(NX),order='F'))
        convecN_hat_gpu = gpuarray.to_gpu(np.reshape(convN_hat,NX*(NX),order='F'))

        w1x_hat_gpu = gpuarray.to_gpu(np.reshape(w1x_hat,NX*(NX),order='F'))
        w1y_hat_gpu = gpuarray.to_gpu(np.reshape(w1y_hat,NX*(NX),order='F'))
        w0x_hat_gpu = gpuarray.to_gpu(np.reshape(w0x_hat,NX*(NX),order='F'))
        w0y_hat_gpu = gpuarray.to_gpu(np.reshape(w0y_hat,NX*(NX),order='F'))
        w1x_gpu = gpuarray.to_gpu(np.reshape(w1x,NX*(NX),order='F'))
        w1y_gpu = gpuarray.to_gpu(np.reshape(w1y,NX*(NX),order='F'))
        w0x_gpu = gpuarray.to_gpu(np.reshape(w0x,NX*(NX),order='F'))
        w0y_gpu = gpuarray.to_gpu(np.reshape(w0y,NX*(NX),order='F'))


        diffu_gpu = gpuarray.to_gpu(np.reshape(diffu,NX*(NX),order='F'))
        diffu_hat_gpu = gpuarray.to_gpu(np.reshape(diffu_hat,NX*(NX),order='F'))
        convec_gpu = gpuarray.to_gpu(np.reshape(convec,NX*(NX),order='F'))
        convec_hat_gpu = gpuarray.to_gpu(np.reshape(convec_hat,NX*(NX),order='F'))
        convecC_hat_gpu = gpuarray.to_gpu(np.reshape(convec_hat,NX*NX,order='F'))
        RHS_gpu = gpuarray.to_gpu(np.reshape(RHS,NX*(NX),order='F'))
        psiTemp_gpu = gpuarray.to_gpu(np.reshape(psiTemp,NX*(NX),order='F'))

        u1_gpu = gpuarray.to_gpu(np.reshape(u1,NX*(NX),order='F'))
        v1_gpu = gpuarray.to_gpu(np.reshape(v1,NX*(NX),order='F'))
        u0_gpu = gpuarray.to_gpu(np.reshape(u0,NX*(NX),order='F'))
        v0_gpu = gpuarray.to_gpu(np.reshape(v0,NX*(NX),order='F'))
        psi1_gpu = gpuarray.to_gpu(np.reshape(psi1,NX*(NX),order='F'))

        kx      = kx.astype(np.float64)
        kkx     = kkx.astype(np.float64)
        # KKx     = KKx.astype(dtype=np.complex128)
        # KKy     = KKy.astype(dtype=np.complex128)
        # KKsq    = KKsq.astype(dtype=np.complex128)
        # Kx      = Kx.astype(dtype=np.complex128)
        # Ky      = Ky.astype(dtype=np.complex128)
        # Ksq     = Ksq.astype(dtype=np.complex128)

        psiPrevious_hat_gpu = gpuarray.to_gpu(np.reshape(psiPrevious_hat,NX*(NX),order='F'))
        psiCurrent_hat_gpu  = gpuarray.to_gpu(np.reshape(psiCurrent_hat,NX*(NX),order='F'))
        kx_gpu              = gpuarray.to_gpu(kx)
        kkx_gpu             = gpuarray.to_gpu(kkx)
        # KKx_gpu             = gpuarray.to_gpu(np.reshape(0*KKx,NX*NX,order='F'))
        # KKy_gpu             = gpuarray.to_gpu(np.reshape(0*KKy,NX*NX,order='F'))
        # KKsq_gpu            = gpuarray.to_gpu(np.reshape(0*KKsq,NX*NX,order='F'))
        # Kx_gpu              = gpuarray.to_gpu(np.reshape(Kx,NX*NX,order='F'))
        # Ky_gpu              = gpuarray.to_gpu(np.reshape(Ky,NX*NX,order='F'))
        # Ksq_gpu             = gpuarray.to_gpu(np.reshape(Ksq,NX*NX,order='F'))

        psiHat_filter_gpu        = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))
        psi_filter_gpu           = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))

        w1Hat_filter_gpu        = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))
        w1_filter_gpu           = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))
        
        convecC_hat_filter_gpu  = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))
        convecC_filter_gpu      = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))

        convecN_hat_filter_gpu  = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))
        convecN_filter_gpu      = gpuarray.to_gpu(np.reshape(K_filter,NNX*NNX,order='F'))

        Gk_gpu      = gpuarray.to_gpu(np.reshape(Gk,NX*NX,order='F'))

        UU_gpu = gpuarray.to_gpu(np.reshape(UU,NX*NX,order='F'))
        UV_gpu = gpuarray.to_gpu(np.reshape(UV,NX*NX,order='F'))
        VV_gpu = gpuarray.to_gpu(np.reshape(VV,NX*NX,order='F'))
        UU_hat_gpu = gpuarray.to_gpu(np.reshape(UU_hat,NX*NX,order='F'))
        UV_hat_gpu = gpuarray.to_gpu(np.reshape(UV_hat,NX*NX,order='F'))
        VV_hat_gpu = gpuarray.to_gpu(np.reshape(VV_hat,NX*NX,order='F'))
        UU_filter_gpu = gpuarray.to_gpu(np.reshape(UU_filter,NNX*NNX,order='F'))
        UV_filter_gpu = gpuarray.to_gpu(np.reshape(UV_filter,NNX*NNX,order='F'))
        VV_filter_gpu = gpuarray.to_gpu(np.reshape(VV_filter,NNX*NNX,order='F'))
        UU_hat_filter_gpu = gpuarray.to_gpu(np.reshape(UU_hat_filter,NNX*NNX,order='F'))
        UV_hat_filter_gpu = gpuarray.to_gpu(np.reshape(UV_hat_filter,NNX*NNX,order='F'))
        VV_hat_filter_gpu = gpuarray.to_gpu(np.reshape(VV_hat_filter,NNX*NNX,order='F'))

        UUx_hat_gpu = gpuarray.to_gpu(np.reshape(UUx_hat,NX*NX,order='F'))
        UVx_hat_gpu = gpuarray.to_gpu(np.reshape(UVx_hat,NX*NX,order='F'))
        UVy_hat_gpu = gpuarray.to_gpu(np.reshape(UVy_hat,NX*NX,order='F'))
        VVy_hat_gpu = gpuarray.to_gpu(np.reshape(VVy_hat,NX*NX,order='F'))

        UUx_filter_gpu = gpuarray.to_gpu(np.reshape(UUx_filter,NNX*NNX,order='F'))
        UVx_filter_gpu = gpuarray.to_gpu(np.reshape(UVx_filter,NNX*NNX,order='F'))
        UVy_filter_gpu = gpuarray.to_gpu(np.reshape(UVy_filter,NNX*NNX,order='F'))
        VVy_filter_gpu = gpuarray.to_gpu(np.reshape(VVy_filter,NNX*NNX,order='F'))

        UUx_hat_filter_gpu = gpuarray.to_gpu(np.reshape(UUx_hat_filter,NNX*NNX,order='F'))
        UVx_hat_filter_gpu = gpuarray.to_gpu(np.reshape(UVx_hat_filter,NNX*NNX,order='F'))
        UVy_hat_filter_gpu = gpuarray.to_gpu(np.reshape(UVy_hat_filter,NNX*NNX,order='F'))
        VVy_hat_filter_gpu = gpuarray.to_gpu(np.reshape(VVy_hat_filter,NNX*NNX,order='F'))

        ux_gpu = gpuarray.to_gpu(np.reshape(ux,NX*NX,order='F'))
        vx_gpu = gpuarray.to_gpu(np.reshape(vx,NX*NX,order='F'))
        uy_gpu = gpuarray.to_gpu(np.reshape(uy,NX*NX,order='F'))
        vy_gpu = gpuarray.to_gpu(np.reshape(vy,NX*NX,order='F'))
        
        ux_hat_gpu = gpuarray.to_gpu(np.reshape(ux_hat,NX*NX,order='F'))
        vx_hat_gpu = gpuarray.to_gpu(np.reshape(vx_hat,NX*NX,order='F'))
        uy_hat_gpu = gpuarray.to_gpu(np.reshape(uy_hat,NX*NX,order='F'))
        vy_hat_gpu = gpuarray.to_gpu(np.reshape(vy_hat,NX*NX,order='F'))

        convec1_gpu = gpuarray.to_gpu(np.reshape(convec1,NX*NX,order='F'))
        convec2_gpu = gpuarray.to_gpu(np.reshape(convec2,NX*NX,order='F'))

        convec1_hat_gpu = gpuarray.to_gpu(np.reshape(convec1_hat,NX*NX,order='F'))
        convec2_hat_gpu = gpuarray.to_gpu(np.reshape(convec2_hat,NX*NX,order='F'))


        convec1_hat_filter_gpu = gpuarray.to_gpu(np.reshape(convec1_hat_filter,NNX*NNX,order='F'))
        convec2_hat_filter_gpu = gpuarray.to_gpu(np.reshape(convec2_hat_filter,NNX*NNX,order='F'))

        convec1_filter_gpu = gpuarray.to_gpu(np.reshape(convec1_filter,NNX*NNX,order='F'))
        convec2_filter_gpu = gpuarray.to_gpu(np.reshape(convec2_filter,NNX*NNX,order='F'))

        dummy_gpu = gpuarray.to_gpu(np.reshape(dummy,NX*NX,order='F'))

        # Start simulation

        initialization(u0_hat_gpu, v0_hat_gpu, w0_hat_gpu, kx_gpu, psiPrevious_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

        # Save Initial conditions
        w0_hat = np.reshape(w0_hat_gpu.get(),[NX,NX],order='F')
        psi0_hat = np.reshape(psiPrevious_hat_gpu.get(),[NX,NX],order='F')
        savemat('wDNS_ini.mat',dict([('w0_hat', w0_hat),('psi0_hat', psi0_hat)]))

        iniW1(w1_hat_gpu, kx_gpu, psiCurrent_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

        runtime2 = runtime.time()

    else:
        updateu(u0_hat_gpu, v0_hat_gpu, u1_hat_gpu, v1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    UV_ker(u1_hat_gpu, v1_hat_gpu, kx_gpu, psiCurrent_hat_gpu, 
        block=(TPB,1,1), grid=(HNB,HNX,1))

    # u1 = np.reshape(u1_hat_gpu.get(),[NX,NX],order='F')
    # v1 = np.reshape(v1_hat_gpu.get(),[NX,NX],order='F')
    # savemat('u1.mat',dict([('u1', u1),('v1', v1)]))

    diffusion(diffu_hat_gpu, kx_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    # Conservative convection form

    cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(u0_hat_gpu.gpudata), int(u0_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(v0_hat_gpu.gpudata), int(v0_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(w1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0_hat_gpu.gpudata), int(w0_gpu.gpudata), cf.CUFFT_INVERSE)

    convection2(u1_gpu, w1_gpu, conu1_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(v1_gpu, w1_gpu, conv1_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(u0_gpu, w0_gpu, conu0_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(v0_gpu, w0_gpu, conv0_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(conu1_gpu.gpudata), int(conu1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conv1_gpu.gpudata), int(conv1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conu0_gpu.gpudata), int(conu0_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conv0_gpu.gpudata), int(conv0_hat_gpu.gpudata), cf.CUFFT_FORWARD)

    # Non-conservative convection form
    convection4(w0x_hat_gpu, w0y_hat_gpu, kx_gpu, w0_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    convection4(w1x_hat_gpu, w1y_hat_gpu, kx_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(w1x_hat_gpu.gpudata), int(w1x_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w1y_hat_gpu.gpudata), int(w1y_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0x_hat_gpu.gpudata), int(w0x_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0y_hat_gpu.gpudata), int(w0y_gpu.gpudata), cf.CUFFT_INVERSE)

    convection5(u1_gpu, v1_gpu, u0_gpu, v0_gpu, w1x_gpu, w1y_gpu, w0x_gpu, w0y_gpu, convec_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(convec_gpu.gpudata), int(convecN_hat_gpu.gpudata), cf.CUFFT_FORWARD)


    # Convection = 0.5*(convec + convecN)
    convection3(conu1_hat_gpu, conv1_hat_gpu, conu0_hat_gpu, conv0_hat_gpu, convecN_hat_gpu, kx_gpu, convec_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    RHS_ker(convec_hat_gpu, diffu_hat_gpu, w1_hat_gpu, RHS_gpu, block=(TPB,1,1), grid=(NB,NX,1)) 

    # DD-P
    # Step n-1
    # cf.cufftExecZ2Z(plan, int(w0_hat_gpu.gpudata), int(w0_gpu.gpudata), cf.CUFFT_INVERSE)
    # cf.cufftExecZ2Z(plan, int(psiPrevious_hat_gpu.gpudata), int(psi1_gpu.gpudata), cf.CUFFT_INVERSE)
    # psiDDP = np.real(np.reshape(psi1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # wDDP = np.real(np.reshape(w1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # input_data=np.zeros([1, Nlon, Nlat, 2])
    # input_data[0,:,:,0] = psiDDP
    # input_data[0,:,:,1] = wDDP
    # SDEV_I=np.std(input_data.flatten())
    # MEAN_I=np.mean(input_data.flatten())
    # # SDEV_I = 3.5028
    # # MEAN_I = 0
    # input_normalized = (input_data - MEAN_I)/SDEV_I

    # prediction = model.predict(input_normalized)
    # SDEV_O     = 9.8945
    # MEAN_O     = 0.0
    # prediction0 = np.reshape(prediction,[NX,NX]) * SDEV_O + MEAN_O

    # Step n
    # cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(w1_gpu.gpudata), cf.CUFFT_INVERSE)
    # cf.cufftExecZ2Z(plan, int(psiCurrent_hat_gpu.gpudata), int(psi1_gpu.gpudata), cf.CUFFT_INVERSE)
    # psiDDP = np.real(np.reshape(psi1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # wDDP = np.real(np.reshape(w1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # input_data=np.zeros([1, Nlon, Nlat, 2])
    # input_data[0,:,:,0] = psiDDP
    # input_data[0,:,:,1] = wDDP
    # SDEV_I=np.std(input_data.flatten())
    # MEAN_I=np.mean(input_data.flatten())
    # # SDEV_I = 3.5028
    # # MEAN_I = 0
    # input_normalized = (input_data - MEAN_I)/SDEV_I

    # prediction = model.predict(input_normalized)
    # SDEV_O     = 5.3190
    # MEAN_O     = 0.0
    # prediction1 = np.reshape(prediction,[NX,NX]) * SDEV_O + MEAN_O

    # cf.cufftExecZ2Z(plan, int(diffu_hat_gpu.gpudata), int(diffu_gpu.gpudata), cf.CUFFT_INVERSE)
    # diffu = np.real(np.reshape(psi1_gpu.get(),[NX,NX],order='F'))/(NX*NX)


    ## Correct prediction
    # for i in range(NX):
    #     for j in range(NX):
    #         if diffu[i,j]*prediction[i,j] < 0:
    #             prediction[i,j] = 0    
    #         if np.abs(prediction[i,j]>0.3):
    #             prediction[i,j] = 0.3
    # prediction = -1.5*prediction1 + prediction0
    # prediction = -1*prediction1
    # prediction = np.complex128(prediction)

    # prediction = SGS[it+1,:,:]

    # pred_hat = np.fft.fft2(prediction)

    ## End Correct prediction

    # pred_gpu   = gpuarray.to_gpu(np.reshape(pred_hat,NX*(NX),order='F'))

    # print(pred_gpu.get())

    # DDP_ker(RHS_gpu, pred_gpu, block=(1,TPB,1), grid=(1,NB,NX))
    # print(np.shape(prediction))


    # End DD-P   

    # Stochastic forcing
    '''
    for i in range(len(indx)):
        ki = indx[i]
        kj = indy[i]
        theta =  np.random.uniform(low=0.0, high=2*np.pi)
        Fk_hat[ki,kj] = AmF*np.exp(1j*theta)
        Fk[ki,kj] = R*Fk[ki,kj] + (1-R**2)**(0.5)*Fk_hat[ki,kj]
        
        if (ki != 0 and kj != 0):
            Fk[NX-ki, NX-kj] = np.conj(Fk[ki,kj])
        elif (ki==0):
            Fk[ki, NX-kj] = np.conj(Fk[ki,kj])
        elif (kj==0):
            Fk[NX-ki, kj] = np.conj(Fk[ki,kj])

    Force_gpu   = gpuarray.to_gpu(np.reshape(Fk,NX*(NX),order='F'))
    '''
    # Deterministic forcing
    DDP_ker(RHS_gpu, Force_gpu, block=(1,TPB,1), grid=(1,NB,NX))
    
    # End forcing  
    
    updatew(w0_hat_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    LHS_ker(kx_gpu, RHS_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    psiTemp_ker(w1_hat_gpu, kx_gpu, psiTemp_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    # ## Debug
    # cf.cufftExecZ2Z(plan, int(psiTemp_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
    # u = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    
    # plt.contourf(X,Y,u)
    # plt.show()
    # ##

    update(psiPrevious_hat_gpu, psiCurrent_hat_gpu, psiTemp_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    time_LHS = runtime.time()-runtime2
    runtim2 = runtime.time()
 
    time = time + dt
    
    if np.mod(it, NSAVE) == 0:
        '''
        cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(dummy_gpu.gpudata), cf.CUFFT_INVERSE)
        u = np.real(np.reshape(dummy_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        slnUDNS[:,:,count] = u

        cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(dummy_gpu.gpudata), cf.CUFFT_INVERSE)
        u = np.real(np.reshape(dummy_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        slnVDNS[:,:,count] = u
        '''
        cf.cufftExecZ2Z(plan, int(w0_hat_gpu.gpudata), int(dummy_gpu.gpudata), cf.CUFFT_INVERSE)
        u = np.real(np.reshape(dummy_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        #slnWorDNS[:,:,count] = u

        cf.cufftExecZ2Z(plan, int(psiPrevious_hat_gpu.gpudata), int(dummy_gpu.gpudata), cf.CUFFT_INVERSE)
        psi = -np.real(np.reshape(dummy_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        #slnPsiDNS[:,:,count] = psi
        
        ener = u*psi
        enst = u*u

        Energy[count] = np.sum(ener)
        Enstrophy[count] = np.sum(enst)

        #onePython[count] = u[0,0]
        tempW = np.max(np.squeeze(u))
        # tempW = u[0,0]
        slnW.append(tempW)
        # savemat('data_LDC_vorPsi_slnW.mat', dict([('slnW', slnW), ('time', time)])) 
        print(it)
        print(tempW)
        print("--- %s seconds ---" % (runtime.time() - runtime2))
        runtime2 = runtime.time()
        
        # record velocities
        #cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
        #cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)

        #u = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        #v = np.real(np.reshape(v1_gpu.get(),[NX,NX],order='F'))/(NX*NX)

        #slnU[:,count] = u[:,0]
        #slnV[:,count] = v[:,0]


        # Save filtered variables
        # Apply spectral filter (Sharp spectral)
        ##GaussianFilter(psiPrevious_hat_gpu, dummy_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(psiPrevious_hat_gpu, psiHat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(psiHat_filter_gpu.gpudata), int(psi_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        psiF = np.reshape(np.real(psi_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        worF = np.real(np.fft.ifft2(-KKsq*np.fft.fft2(psiF)))
        slnPsi_ss[:,:,count] = -psiF
        slnWor_ss[:,:,count] = worF
        # Apply spectral filter (Gaussian)
        GaussianFilter(psiPrevious_hat_gpu, dummy_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(dummy_gpu, psiHat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(psiHat_filter_gpu.gpudata), int(psi_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        psiF = np.reshape(np.real(psi_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        worF = np.real(np.fft.ifft2(-KKsq*np.fft.fft2(psiF)))
        slnPsi[:,:,count] = -psiF
        slnWor[:,:,count] = worF


        # Obtain filtered variables, stress, force, and PI
        '''
        # ===============================================================================================================
        # Obtain PI = \hat{J} - J(\hat{}) 
        # ===============================================================================================================
        # Apply spectral filter
        GaussianFilter(psiPrevious_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(psiPrevious_hat_gpu, psiHat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(psiHat_filter_gpu.gpudata), int(psi_filter_gpu.gpudata), cf.CUFFT_INVERSE)

        # Conservative convection form
        addConvecC(conu1_hat_gpu, conv1_hat_gpu, kx_gpu, convecC_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(convecC_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(convecC_hat_gpu, convecC_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(convecC_hat_filter_gpu.gpudata), int(convecC_filter_gpu.gpudata), cf.CUFFT_INVERSE)     

        psiF = np.reshape(np.real(psi_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        worF = np.real(np.fft.ifft2(-KKsq*np.fft.fft2(psiF)))
        convectionF = np.reshape(np.real(convecC_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))

        u1F = np.real(np.fft.ifft2(-(1j*KKy)*np.fft.fft2(psiF)))
        v1F = np.real(np.fft.ifft2((1j*KKx)*np.fft.fft2(psiF)))
        conuF2 = 1j*KKx*np.fft.fft2(u1F*worF)
        convF2 = 1j*KKy*np.fft.fft2(v1F*worF)
        convectionF2 = np.real(np.fft.ifft2(conuF2+convF2))

        convecPI1 = convectionF - convectionF2

        # Non-conservative form

        convection6(u1_gpu, v1_gpu, w1x_gpu, w1y_gpu, convec_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan, int(convec_gpu.gpudata), int(convecN_hat_gpu.gpudata), cf.CUFFT_FORWARD)

        GaussianFilter(convecN_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(convecN_hat_gpu, convecN_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(convecN_hat_filter_gpu.gpudata), int(convecN_filter_gpu.gpudata), cf.CUFFT_INVERSE)     
        convectionF3 = np.reshape(np.real(convecN_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))

        conuF4 = u1F*np.real(np.fft.ifft2(1j*KKx*np.fft.fft2(worF)))
        convF4 = v1F*np.real(np.fft.ifft2(1j*KKy*np.fft.fft2(worF)))

        convectionF4 = conuF4 + convF4

        convecPI2 = convectionF3 - convectionF4

        convecPI = 0.5*(convecPI1 + convecPI2)
        

        slnPsi[:,:,count] = -psiF
        slnWor[:,:,count] = worF
        slnPI[:,:,count]  = convecPI

        # ===============================================================================================================
        # Obtain Stress
        # SGSStress1 = \bar{uu} - \bar{u}*\bar{u}
        # SGSStress2 = \bar{uv} - \bar{u}*\bar{v}
        # SGSStress3 = \bar{vv} - \bar{v}*\bar{v}
        # ===============================================================================================================

        # Calculate uu, uv, vv
        convection2(u1_gpu, u1_gpu, UU_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
        convection2(u1_gpu, v1_gpu, UV_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
        convection2(v1_gpu, v1_gpu, VV_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

        cf.cufftExecZ2Z(plan, int(UU_gpu.gpudata), int(UU_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        cf.cufftExecZ2Z(plan, int(UV_gpu.gpudata), int(UV_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        cf.cufftExecZ2Z(plan, int(VV_gpu.gpudata), int(VV_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        
        # Calculate \bar{uu}, \bar{uv}, \bar{vv}
        GaussianFilter(UU_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(UU_hat_gpu, UU_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(UV_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(UV_hat_gpu, UV_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(VV_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(VV_hat_gpu, VV_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))

        # Calculate \bar{u} -> u1F, \bar{v} -> v1F
        cf.cufftExecZ2Z(plan2, int(UU_hat_filter_gpu.gpudata), int(UU_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(UV_hat_filter_gpu.gpudata), int(UV_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(VV_hat_filter_gpu.gpudata), int(VV_filter_gpu.gpudata), cf.CUFFT_INVERSE)  

        UU_filter = np.reshape(np.real(UU_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        UV_filter = np.reshape(np.real(UV_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        VV_filter = np.reshape(np.real(VV_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))


        stress1 = UU_filter - u1F*u1F
        stress2 = UV_filter - u1F*v1F
        stress3 = VV_filter - v1F*v1F

        # ===============================================================================================================
        # Obtain Force
        # SGSForce1_conser = \bar{duu/dx + duv/dy} - (d\bar{u}\bar{u}dx + d\bar{u}\bar{v}dy)
        # SGSForce1_Noconser = \bar{udu/dx + vdu/dy} - (\bar{u}d\bar{u}dx + \bar{v}d\bar{u}dy)
        # SGSForce2_conser = \bar{duv/dx + dvv/dy} - (d\bar{u}\bar{v}dx + d\bar{v}\bar{v}dy)
        # SGSForce2_Noconser = \bar{udv/dx + vdv/dy} - (\bar{u}d\bar{v}dx + \bar{v}d\bar{v}dy)
        # ===============================================================================================================
        # Conservation form

        # Calculate duu/dx, duv/dx, duv/dy, dvv/dy
        # Calculate uu_hat, uv_hat, vv_hat
        cf.cufftExecZ2Z(plan, int(UU_gpu.gpudata), int(UU_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        cf.cufftExecZ2Z(plan, int(UV_gpu.gpudata), int(UV_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        cf.cufftExecZ2Z(plan, int(VV_gpu.gpudata), int(VV_hat_gpu.gpudata), cf.CUFFT_FORWARD)

        firstDx(UUx_hat_gpu, kx_gpu, UU_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        firstDy(UVy_hat_gpu, kx_gpu, UV_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        firstDx(UVx_hat_gpu, kx_gpu, UV_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))        
        firstDy(VVy_hat_gpu, kx_gpu, VV_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

        GaussianFilter(UUx_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(UUx_hat_gpu, UUx_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(UVy_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(UVy_hat_gpu, UVy_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(UVx_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(UVx_hat_gpu, UVx_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(VVy_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(VVy_hat_gpu, VVy_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))

        cf.cufftExecZ2Z(plan2, int(UUx_hat_filter_gpu.gpudata), int(UUx_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(UVy_hat_filter_gpu.gpudata), int(UVy_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(UVx_hat_filter_gpu.gpudata), int(UVx_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(VVy_hat_filter_gpu.gpudata), int(VVy_filter_gpu.gpudata), cf.CUFFT_INVERSE) 

        # Non-conservation form

        # Calculate du/dx, du/dy, dv/dx, dvdy
        firstDx(ux_hat_gpu, kx_gpu, u1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        firstDx(vx_hat_gpu, kx_gpu, v1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        firstDy(uy_hat_gpu, kx_gpu, u1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        firstDy(vy_hat_gpu, kx_gpu, v1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

        cf.cufftExecZ2Z(plan, int(ux_hat_gpu.gpudata), int(ux_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan, int(uy_hat_gpu.gpudata), int(uy_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan, int(vx_hat_gpu.gpudata), int(vx_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan, int(vy_hat_gpu.gpudata), int(vy_gpu.gpudata), cf.CUFFT_INVERSE) 

        # Calculate convecx = udu/dx + vdu/dy, convecy = udv/dx + vdv/dy
        convection6(u1_gpu, v1_gpu, ux_gpu, uy_gpu, convec1_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        convection6(u1_gpu, v1_gpu, vx_gpu, vy_gpu, convec2_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        
        cf.cufftExecZ2Z(plan, int(convec1_gpu.gpudata), int(convec1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        cf.cufftExecZ2Z(plan, int(convec2_gpu.gpudata), int(convec2_hat_gpu.gpudata), cf.CUFFT_FORWARD)
        GaussianFilter(convec1_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(convec1_hat_gpu, convec1_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        GaussianFilter(convec2_hat_gpu, Gk_gpu,  block=(TPB,1,1), grid=(NB,NX,1))
        spectralFilter(convec2_hat_gpu, convec2_hat_filter_gpu, block=(TPB,1,1), grid=(NB,NX,1))
        cf.cufftExecZ2Z(plan2, int(convec1_hat_filter_gpu.gpudata), int(convec1_filter_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan2, int(convec2_hat_filter_gpu.gpudata), int(convec2_filter_gpu.gpudata), cf.CUFFT_INVERSE)

        # Calculate force from conservative and non-conservative form

        UUx_filter = np.reshape(np.real(UUx_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        UVy_filter = np.reshape(np.real(UVy_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        UVx_filter = np.reshape(np.real(UVx_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        VVy_filter = np.reshape(np.real(VVy_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))

        convec1_filter = np.reshape(np.real(convec1_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))
        convec2_filter = np.reshape(np.real(convec2_filter_gpu.get()),[NNX,NNX],order='F')/(NNX*NNX*np.power((NX/NNX),2))


        # SGSForce1_conser = \bar{duu/dx + duv/dy} - (d\bar{u}\bar{u}dx + d\bar{u}\bar{v}dy)
        # SGSForce1_Noconser = \bar{udu/dx + vdu/dy} - (\bar{u}d\bar{u}dx + \bar{v}d\bar{u}dy)
        # SGSForce2_conser = \bar{duv/dx + dvv/dy} - (d\bar{u}\bar{v}dx + d\bar{v}\bar{v}dy)
        # SGSForce2_Noconser = \bar{udv/dx + vdv/dy} - (\bar{u}d\bar{v}dx + \bar{v}d\bar{v}dy)

        uux = np.real(np.fft.ifft2(1j*KKx*np.fft.fft2(u1F*u1F)))
        uvy = np.real(np.fft.ifft2(1j*KKy*np.fft.fft2(u1F*v1F)))
        uvx = np.real(np.fft.ifft2(1j*KKx*np.fft.fft2(u1F*v1F)))
        vvy = np.real(np.fft.ifft2(1j*KKy*np.fft.fft2(v1F*v1F)))

        Force1_conser = UUx_filter + UVy_filter - (uux + uvy)
        Force2_conser = UVx_filter + VVy_filter - (uvx + vvy)
        
        uxF = np.real(np.fft.ifft2(1j*KKx*np.fft.fft2(u1F)))
        uyF = np.real(np.fft.ifft2(1j*KKy*np.fft.fft2(u1F)))
        vxF = np.real(np.fft.ifft2(1j*KKx*np.fft.fft2(v1F)))
        vyF = np.real(np.fft.ifft2(1j*KKy*np.fft.fft2(v1F)))
        
        Force1_nonconser = convec1_filter - (u1F*uxF + v1F*uyF)
        Force2_nonconser = convec2_filter - (u1F*vxF + v1F*vyF)

        Force1 = 0.5*(Force1_conser + Force1_nonconser)
        Force2 = 0.5*(Force2_conser + Force2_nonconser)


        slnuF[:,:,count] = u1F
        slnvF[:,:,count] = v1F
        slnStress1[:,:,count] = stress1
        slnStress2[:,:,count] = stress2
        slnStress3[:,:,count] = stress3
        slnForce1[:,:,count] = Force1
        slnForce2[:,:,count] = Force2
        '''

        
        # Maulik's
        '''
        psiDDP = psiF
        wDDP = worF
        
        # Smagorinsky invariant
        psiDDP_hat = np.fft.fft2(psiDDP)
        S1 = np.real(np.fft.ifft2(-KKy*KKx*psiDDP_hat))
        S2 = 0.5*np.real(np.fft.ifft2(-(KKx*KKx - KKy*KKy)*psiDDP_hat))
        S  = 2*np.sqrt(S1*S1 + S2*S2)

        # Leith invariant
        w1DDP_hat = np.fft.fft2(worF)
        L1 = np.real(np.fft.ifft2(1j*KKx*w1DDP_hat))
        L2 = np.real(np.fft.ifft2(1j*KKy*w1DDP_hat))
        L  = np.sqrt(L1*L1 + L2*L2)

        # Diffusion 
        diffu = np.real(np.fft.ifft2(-KKsq*np.fft.fft2(worF)))

        # Interior points
        tempNX = NX
        NX = NNX
        sampling_matrix[0,:,:] = wDDP
        sampling_matrix[1,1:NX-1,1:NX-1] = wDDP[1:NX-1,2:]
        sampling_matrix[2,1:NX-1,1:NX-1] = wDDP[1:NX-1,0:NX-2]
        sampling_matrix[3,1:NX-1,1:NX-1] = wDDP[2:NX,1:NX-1]
        sampling_matrix[4,1:NX-1,1:NX-1] = wDDP[2:NX,2:]
        sampling_matrix[5,1:NX-1,1:NX-1] = wDDP[2:NX,0:NX-2]
        sampling_matrix[6,1:NX-1,1:NX-1] = wDDP[0:NX-2,1:NX-1]
        sampling_matrix[7,1:NX-1,1:NX-1] = wDDP[0:NX-2,2:]
        sampling_matrix[8,1:NX-1,1:NX-1] = wDDP[0:NX-2,0:NX-2]

        sampling_matrix[9,:,:] = psiDDP
        sampling_matrix[10,1:NX-1,1:NX-1] = psiDDP[1:NX-1,2:]
        sampling_matrix[11,1:NX-1,1:NX-1] = psiDDP[1:NX-1,0:NX-2]
        sampling_matrix[12,1:NX-1,1:NX-1] = psiDDP[2:NX,1:NX-1]
        sampling_matrix[13,1:NX-1,1:NX-1] = psiDDP[2:NX,2:]
        sampling_matrix[14,1:NX-1,1:NX-1] = psiDDP[2:NX,0:NX-2]
        sampling_matrix[15,1:NX-1,1:NX-1] = psiDDP[0:NX-2,1:NX-1]
        sampling_matrix[16,1:NX-1,1:NX-1] = psiDDP[0:NX-2,2:]
        sampling_matrix[17,1:NX-1,1:NX-1] = psiDDP[0:NX-2,0:NX-2]
 
        sampling_matrix[18,:,:] = S
        sampling_matrix[19,:,:] = L
        sampling_matrix[20,:,:] = diffu

        # Lateral (periodic boundaries)
        # North
        sampling_matrix[1,1:NX-1,NX-1] = wDDP[1:NX-1,0]
        sampling_matrix[2,1:NX-1,NX-1] = wDDP[1:NX-1,NX-2]
        sampling_matrix[3,1:NX-1,NX-1] = wDDP[2:,NX-1]
        sampling_matrix[4,1:NX-1,NX-1] = wDDP[2:,0]
        sampling_matrix[5,1:NX-1,NX-1] = wDDP[2:,NX-2]
        sampling_matrix[6,1:NX-1,NX-1] = wDDP[0:NX-2,NX-1]
        sampling_matrix[7,1:NX-1,NX-1] = wDDP[0:NX-2,0]
        sampling_matrix[8,1:NX-1,NX-1] = wDDP[0:NX-2,NX-2]

        sampling_matrix[10,1:NX-1,NX-1] = psiDDP[1:NX-1,0]
        sampling_matrix[11,1:NX-1,NX-1] = psiDDP[1:NX-1,NX-2]
        sampling_matrix[12,1:NX-1,NX-1] = psiDDP[2:,NX-1]
        sampling_matrix[13,1:NX-1,NX-1] = psiDDP[2:,0]
        sampling_matrix[14,1:NX-1,NX-1] = psiDDP[2:,NX-2]
        sampling_matrix[15,1:NX-1,NX-1] = psiDDP[0:NX-2,NX-1]
        sampling_matrix[16,1:NX-1,NX-1] = psiDDP[0:NX-2,0]
        sampling_matrix[17,1:NX-1,NX-1] = psiDDP[0:NX-2,NX-2]

        # South
        sampling_matrix[1,1:NX-1,0] = wDDP[1:NX-1,1]
        sampling_matrix[2,1:NX-1,0] = wDDP[1:NX-1,NX-1]
        sampling_matrix[3,1:NX-1,0] = wDDP[2:,0]
        sampling_matrix[4,1:NX-1,0] = wDDP[2:,1]
        sampling_matrix[5,1:NX-1,0] = wDDP[2:,NX-1]
        sampling_matrix[6,1:NX-1,0] = wDDP[0:NX-2,0]
        sampling_matrix[7,1:NX-1,0] = wDDP[0:NX-2,1]
        sampling_matrix[8,1:NX-1,0] = wDDP[0:NX-2,NX-1]

        sampling_matrix[10,1:NX-1,0] = psiDDP[1:NX-1,1]
        sampling_matrix[11,1:NX-1,0] = psiDDP[1:NX-1,NX-1]
        sampling_matrix[12,1:NX-1,0] = psiDDP[2:,0]
        sampling_matrix[13,1:NX-1,0] = psiDDP[2:,1]
        sampling_matrix[14,1:NX-1,0] = psiDDP[2:,NX-1]
        sampling_matrix[15,1:NX-1,0] = psiDDP[0:NX-2,0]
        sampling_matrix[16,1:NX-1,0] = psiDDP[0:NX-2,1]
        sampling_matrix[17,1:NX-1,0] = psiDDP[0:NX-2,NX-1]

        # West
        sampling_matrix[1,0,1:NX-1] = wDDP[0,2:]
        sampling_matrix[2,0,1:NX-1] = wDDP[0,0:NX-2]
        sampling_matrix[3,0,1:NX-1] = wDDP[1,1:NX-1]
        sampling_matrix[4,0,1:NX-1] = wDDP[1,2:]
        sampling_matrix[5,0,1:NX-1] = wDDP[1,0:NX-2]
        sampling_matrix[6,0,1:NX-1] = wDDP[NX-1,1:NX-1]
        sampling_matrix[7,0,1:NX-1] = wDDP[NX-1,2:]
        sampling_matrix[8,0,1:NX-1] = wDDP[NX-1,0:NX-2]

        sampling_matrix[10,0,1:NX-1] = psiDDP[0,2:]
        sampling_matrix[11,0,1:NX-1] = psiDDP[0,0:NX-2]
        sampling_matrix[12,0,1:NX-1] = psiDDP[1,1:NX-1]
        sampling_matrix[13,0,1:NX-1] = psiDDP[1,2:]
        sampling_matrix[14,0,1:NX-1] = psiDDP[1,0:NX-2]
        sampling_matrix[15,0,1:NX-1] = psiDDP[NX-1,1:NX-1]
        sampling_matrix[16,0,1:NX-1] = psiDDP[NX-1,2:]
        sampling_matrix[17,0,1:NX-1] = psiDDP[NX-1,0:NX-2]

        # East
        sampling_matrix[1,NX-1,1:NX-1] = wDDP[NX-1,2:]
        sampling_matrix[2,NX-1,1:NX-1] = wDDP[NX-1,0:NX-2]
        sampling_matrix[3,NX-1,1:NX-1] = wDDP[0,1:NX-1]
        sampling_matrix[4,NX-1,1:NX-1] = wDDP[0,2:]
        sampling_matrix[5,NX-1,1:NX-1] = wDDP[0,0:NX-2]
        sampling_matrix[6,NX-1,1:NX-1] = wDDP[NX-2,1:NX-1]
        sampling_matrix[7,NX-1,1:NX-1] = wDDP[NX-2,2:]
        sampling_matrix[8,NX-1,1:NX-1] = wDDP[NX-2,0:NX-2]

        sampling_matrix[10,NX-1,1:NX-1] = psiDDP[NX-1,2:]
        sampling_matrix[11,NX-1,1:NX-1] = psiDDP[NX-1,0:NX-2]
        sampling_matrix[12,NX-1,1:NX-1] = psiDDP[0,1:NX-1]
        sampling_matrix[13,NX-1,1:NX-1] = psiDDP[0,2:]
        sampling_matrix[14,NX-1,1:NX-1] = psiDDP[0,0:NX-2]
        sampling_matrix[15,NX-1,1:NX-1] = psiDDP[NX-2,1:NX-1]
        sampling_matrix[16,NX-1,1:NX-1] = psiDDP[NX-2,2:]
        sampling_matrix[17,NX-1,1:NX-1] = psiDDP[NX-2,0:NX-2]

        # North-East
        sampling_matrix[1,NX-1,NX-1] = wDDP[NX-1,0]
        sampling_matrix[2,NX-1,NX-1] = wDDP[NX-1,NX-2]
        sampling_matrix[3,NX-1,NX-1] = wDDP[0,NX-1]
        sampling_matrix[4,NX-1,NX-1] = wDDP[0,0]
        sampling_matrix[5,NX-1,NX-1] = wDDP[0,NX-2]
        sampling_matrix[6,NX-1,NX-1] = wDDP[NX-2,NX-1]
        sampling_matrix[7,NX-1,NX-1] = wDDP[NX-2,0]
        sampling_matrix[8,NX-1,NX-1] = wDDP[NX-2,NX-2]

        sampling_matrix[10,NX-1,NX-1] = psiDDP[NX-1,0]
        sampling_matrix[11,NX-1,NX-1] = psiDDP[NX-1,NX-2]
        sampling_matrix[12,NX-1,NX-1] = psiDDP[0,NX-1]
        sampling_matrix[13,NX-1,NX-1] = psiDDP[0,0]
        sampling_matrix[14,NX-1,NX-1] = psiDDP[0,NX-2]
        sampling_matrix[15,NX-1,NX-1] = psiDDP[NX-2,NX-1]
        sampling_matrix[16,NX-1,NX-1] = psiDDP[NX-2,0]
        sampling_matrix[17,NX-1,NX-1] = psiDDP[NX-2,NX-2]

        # Sorth-East
        sampling_matrix[1,NX-1,0] = wDDP[NX-1,1]
        sampling_matrix[2,NX-1,0] = wDDP[NX-1,NX-1]
        sampling_matrix[3,NX-1,0] = wDDP[0,0]
        sampling_matrix[4,NX-1,0] = wDDP[0,1]
        sampling_matrix[5,NX-1,0] = wDDP[0,NX-1]
        sampling_matrix[6,NX-1,0] = wDDP[NX-2,0]
        sampling_matrix[7,NX-1,0] = wDDP[NX-2,1]
        sampling_matrix[8,NX-1,0] = wDDP[NX-2,NX-1]

        sampling_matrix[10,NX-1,0] = psiDDP[NX-1,1]
        sampling_matrix[11,NX-1,0] = psiDDP[NX-1,NX-1]
        sampling_matrix[12,NX-1,0] = psiDDP[0,0]
        sampling_matrix[13,NX-1,0] = psiDDP[0,1]
        sampling_matrix[14,NX-1,0] = psiDDP[0,NX-1]
        sampling_matrix[15,NX-1,0] = psiDDP[NX-2,0]
        sampling_matrix[16,NX-1,0] = psiDDP[NX-2,1]
        sampling_matrix[17,NX-1,0] = psiDDP[NX-2,NX-1]

        # South-West
        sampling_matrix[1,0,0] = wDDP[0,1]
        sampling_matrix[2,0,0] = wDDP[0,NX-1]
        sampling_matrix[3,0,0] = wDDP[1,0]
        sampling_matrix[4,0,0] = wDDP[1,1]
        sampling_matrix[5,0,0] = wDDP[1,NX-1]
        sampling_matrix[6,0,0] = wDDP[NX-1,0]
        sampling_matrix[7,0,0] = wDDP[NX-1,1]
        sampling_matrix[8,0,0] = wDDP[NX-1,NX-1]

        sampling_matrix[10,0,0] = psiDDP[0,1]
        sampling_matrix[11,0,0] = psiDDP[0,NX-1]
        sampling_matrix[12,0,0] = psiDDP[1,0]
        sampling_matrix[13,0,0] = psiDDP[1,1]
        sampling_matrix[14,0,0] = psiDDP[1,NX-1]
        sampling_matrix[15,0,0] = psiDDP[NX-1,0]
        sampling_matrix[16,0,0] = psiDDP[NX-1,1]
        sampling_matrix[17,0,0] = psiDDP[NX-1,NX-1]

        # North-West
        sampling_matrix[1,0,NX-1] = wDDP[0,0]
        sampling_matrix[2,0,NX-1] = wDDP[0,NX-2]
        sampling_matrix[3,0,NX-1] = wDDP[1,NX-1]
        sampling_matrix[4,0,NX-1] = wDDP[1,0]
        sampling_matrix[5,0,NX-1] = wDDP[1,NX-2]
        sampling_matrix[6,0,NX-1] = wDDP[NX-1,NX-1]
        sampling_matrix[7,0,NX-1] = wDDP[NX-1,0]
        sampling_matrix[8,0,NX-1] = wDDP[NX-1,NX-2]

        sampling_matrix[10,0,NX-1] = psiDDP[0,0]
        sampling_matrix[11,0,NX-1] = psiDDP[0,NX-2]
        sampling_matrix[12,0,NX-1] = psiDDP[1,NX-1]
        sampling_matrix[13,0,NX-1] = psiDDP[1,0]
        sampling_matrix[14,0,NX-1] = psiDDP[1,NX-2]
        sampling_matrix[15,0,NX-1] = psiDDP[NX-1,NX-1]
        sampling_matrix[16,0,NX-1] = psiDDP[NX-1,0]
        sampling_matrix[17,0,NX-1] = psiDDP[NX-1,NX-2]

        sampling_matrix_1[:,:,count] = np.reshape(sampling_matrix,[21,NX*NX],order='F')
        SGS[:,count] = np.reshape(convecPI2,[NX*NX],order='F')
        '''
        count = count + 1
        #NX = tempNX

print("--- %s seconds ---(end iteration)" % (runtime.time() - start_time))

psiCurrent_hat = psiCurrent_hat_gpu.get()
psiPrevious_hat = psiPrevious_hat_gpu.get()

psi = psiCurrent_hat
if saveTrue == 1:
    savemat('data_end_LDC_Poi_gpu_Kraichnan.mat', dict([('psiCurrent_hat', psiCurrent_hat), ('psiPrevious_hat', psiPrevious_hat), ('time', time)
    , ('slnW', slnW)])) 
    #savemat('TKE.mat',dict([('slnU',slnU),('slnV',slnV)]))
    #savemat('onePython.mat',dict([('onePython',onePython)]))
    #psiCurrent_hat = np.fft.fft2(slnPsi[:,:,-1])
    #psiPrevious_hat = np.fft.fft2(slnPsi[:,:,-2])
    #savemat('data_filtered.mat', dict([('psiCurrent_hat',psiCurrent_hat),('psiPrevious_hat',psiPrevious_hat),('time',time),('dt',dt)]))
    savemat('Energy.mat',dict([('Energy',Energy),('Enstrophy',Enstrophy)]))

print("--- %s seconds ---" % (runtime.time() - start_time))

cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(u0_gpu.gpudata), cf.CUFFT_INVERSE)
cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)

# conjugateS_ker(w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
u0 = np.real(np.reshape(u0_gpu.get(),[NX,NX],order='F'))/(NX*NX)
u1 = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
v1 = np.real(np.reshape(v1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
Gk = np.real(np.reshape(Gk_gpu.get(),[NX,NX],order='F'))

# u1 = np.reshape(u1_hat_gpu.get(),[NX,NX],order='F')
# print(u[30,20] - u[30,44])
# u = (np.reshape(diffu_hat_gpu.get(),[NX,N+1],order='F'))
# u0 = np.real(prediction)
# print("--- %s seconds ---" % (runtime.time() - start_time))
savemat('w1.mat',dict([('w1Python', u0),('u1Python', u1),('v1Python', v1),('worFPython',worF),('Gk',Gk),('slnW',slnW)]))
print(time)

#import matplotlib.pyplot as plt
# plt.plot(y,u[0,:])
#plt.plot(slnW)
#plt.ylabel('u velocity(m/s)')
#plt.show()

# print(np.max(u[0,:]))

# # print(w1y_hat)
# print(u)

# meanU = np.mean(slnU[:,:],axis = 1,keepdims=True)
# meanV = np.mean(slnV[:,:],axis = 1,keepdims=True)
# Uprime = slnU - meanU
# Vprime = slnV - meanV
# TKE = Uprime**2 + Vprime**2
# TKE_hat = abs(np.fft.fft(TKE,axis=0))
# # plt.figure(1)
# # plt.loglog(kx[1:50],TKE_hat[1:50,-1])
# # plt.ylabel('TKE')
# plt.figure(1)
# plt.contourf(X,Y,u0)
# plt.figure(2)
# plt.contourf(X,Y,u1)
# plt.figure(3)
# plt.contourf(X,Y,v1)
#plt.plot(np.mean(u,axis=0),x)
# plt.show()

# Save data for training

'''
matfiledata = {}
matfiledata[u'slnWor'] = slnWor
# matfiledata[u'slnWorDNS'] = slnWorDNS
matfiledata[u'slnPsi'] = slnPsi
matfiledata[u'slnPI'] = slnPI
matfiledata[u'slnuF'] = slnuF
matfiledata[u'slnvF'] = slnvF
matfiledata[u'slnStress1'] = slnStress1
matfiledata[u'slnStress2'] = slnStress2
matfiledata[u'slnStress3'] = slnStress3
matfiledata[u'slnForce1'] = slnForce1
matfiledata[u'slnForce2'] = slnForce2
hdf5storage.write(matfiledata, '.', 'vorticity_field_KT_CNN.mat', matlab_compatible=True)
'''

'''
matfiledata = {}
#matfiledata[u'slnWor'] = slnWor
matfiledata[u'slnWorDNS'] = slnWorDNS
#matfiledata[u'slnUDNS'] = slnUDNS
#matfiledata[u'slnVDNS'] = slnVDNS
matfiledata[u'slnPsiDNS'] = slnPsiDNS
#matfiledata[u'slnmaxWor'] = slnW
hdf5storage.write(matfiledata, '.', 'DNS data.mat', matlab_compatible=True)
'''

matfiledata = {}
matfiledata[u'slnWor'] = slnWor
matfiledata[u'slnPsi'] = slnPsi
matfiledata[u'slnWor_ss'] = slnWor_ss
matfiledata[u'slnPsi_ss'] = slnPsi_ss
hdf5storage.write(matfiledata, '.', 'FDNS vor Psi.mat', matlab_compatible=True)

ddx = Lx/NNX
x = np.linspace(0, Lx-ddx, num=NNX)
t = np.linspace(0, (NNSAVE-1)*dt*NSAVE, num=NNSAVE)
savenc(slnPsi,slnWor, x, x, t, 'Gaussian')
savenc(slnPsi_ss,slnWor_ss, x, x, t, 'sharp')


'''
matfiledata = {}
matfiledata[u'slnuF'] = slnuF
matfiledata[u'slnvF'] = slnvF
hdf5storage.write(matfiledata, '.', 'FDNS U V.mat', matlab_compatible=True)

matfiledata = {}
matfiledata[u'slnPI'] = slnPI
hdf5storage.write(matfiledata, '.', 'FDNS PI.mat', matlab_compatible=True)

matfiledata = {}
matfiledata[u'slnStress1'] = slnStress1
matfiledata[u'slnStress2'] = slnStress2
matfiledata[u'slnStress3'] = slnStress3
hdf5storage.write(matfiledata, '.', 'FDNS Stress.mat', matlab_compatible=True)

matfiledata = {}
matfiledata[u'slnForce1'] = slnForce1
matfiledata[u'slnForce2'] = slnForce2
hdf5storage.write(matfiledata, '.', 'FDNS Force.mat', matlab_compatible=True)
'''

'''
matfiledata2 = {}
matfiledata2[u'Input'] = sampling_matrix_1
matfiledata2[u'Output'] = SGS
hdf5storage.write(matfiledata2, '.', 'vorticity_field_KT_ANN.mat', matlab_compatible=True)

with open('Input.npy', 'wb') as f:
    np.save(f, sampling_matrix_1)
with open('Output.npy', 'wb') as f:
    np.save(f, SGS)

'''
