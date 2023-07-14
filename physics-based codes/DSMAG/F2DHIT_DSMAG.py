# Solve 2D turbulence by Fourier-Fourier pseudo-spectral method
# Navier-Stokes equation is in the vorticity-stream function form
# Stream-function is negative stream-function

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

# Import SMAG eddy viscosity model
from dsmag_eddy_viscosity import *

# Import NC saver
from savenc import *

import os.path


def SMAG_solver(negRate, version_):
# cs = 0.2#cs# Smag coefficients
    readTrue = 0
    saveTrue = 1
    NSAVE = 1000
    NNSAVE = 1000
    maxit = NSAVE*NNSAVE

    dt    = 5e-4
    nu    = 5e-5
    rho   = 1
    Re    = 1.0/nu

    Lx    = 2*math.pi
    NX    = 64
    HNX   = 64  # NX/2+1

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
    NNX         = NX
    kkx         = (2*math.pi/Lx)*np.concatenate((np.arange(0,NNX/2+1,dtype=np.float64),np.arange((-NNX/2+1),0,dtype=np.float64)))
    [KKy,KKx]   = np.meshgrid(kkx,kkx)
    KKsq        = (KKx**2 + KKy**2)

    # Gaussian filter
    Delta = 2*Lx/NNX
    Gk = np.exp(-Ksq*Delta**2/24)

    # Smagorinsky coefficients
    # cs = 0.2
    # ev_coeff = (cs*Delta)*(cs*Delta)
    # ev_coeff_gpu = gpuarray.to_gpu(np.reshape(ev_coeff,1,order='F'))
    cs = 0.2
    ev_coeff = (cs*Delta)*(cs*Delta)*np.ones([NX,NX])
    ev_coeff_gpu = gpuarray.to_gpu(np.reshape(ev_coeff,NX*NX,order='F'))
    # negRate = 0.5
    negRate_gpu = gpuarray.to_gpu(np.reshape(negRate,1,order='F'))

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
        # kp = 10.0
        # A  = 4*np.power(kp,(-5))/(3*np.pi)
        # absK = np.sqrt(Kx*Kx+Ky*Ky)
        # Ek = A*np.power(absK,4)*np.exp(-np.power(absK/kp,2))
        # coef1 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
        # coef2 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2

        # perturb = np.zeros([NX,NX])
        # perturb[0:NX//2+1, 0:NX//2+1] = coef1[0:NX//2+1, 0:NX//2+1]+coef2[0:NX//2+1, 0:NX//2+1]
        # perturb[NX//2+1:, 0:NX//2+1] = coef2[NX//2-1:0:-1, 0:NX//2+1] - coef1[NX//2-1:0:-1, 0:NX//2+1]
        # perturb[0:NX//2+1, NX//2+1:] = coef1[0:NX//2+1, NX//2-1:0:-1] - coef2[0:NX//2+1, NX//2-1:0:-1]
        # perturb[NX//2+1:, NX//2+1:] = -(coef1[NX//2-1:0:-1, NX//2-1:0:-1] + coef2[NX//2-1:0:-1, NX//2-1:0:-1])
        # perturb = np.exp(1j*perturb)

        # w1_hat = np.sqrt(absK/np.pi*Ek)*perturb*np.power(NX,2)


        # psi_hat         = -w1_hat*invKsq
        # psiPrevious_hat = psi_hat.astype(np.complex128)
        # psiCurrent_hat  = psi_hat.astype(np.complex128)
        # time   = 0.0
        # slnW = []

        # LES
        ##time = 0.0
        data_Poi          = loadmat(os.path.dirname(__file__) + '/iniWor.mat')
        w1                = data_Poi['w1']
        w1_hat            = np.fft.fft2(w1)
        psiCurrent_hat    = (-invKsq*w1_hat).astype(np.complex128)
        psiPrevious_hat   = psiCurrent_hat

        time   = 0.0
        slnW = []

    else:
        data_Poi = loadmat('data_end_LDC_Poi_gpu_Kraichnan.mat')
        
        # data_Poi = h5py.File('data_end_LDC_Poi.mat', 'r')
        psiCurrent_hat    = data_Poi['psiCurrent_hat']
        psiPrevious_hat   = data_Poi['psiPrevious_hat']

        # data_Poi = loadmat('data_filtered_NX32.mat')
        # psiCurrent_hat    = data_Poi['psiCurrent_hat'].T
        # psiPrevious_hat   = data_Poi['psiPrevious_hat'].T

        # data_Pi = h5py.File('slnPI.mat', 'r')
        # SGS    = data_Pi['slnPI']

        # time   = data_Poi['time']
        #slnW   = data_Poi['slnW']
        # time = np.asarray(time)
        # time = time[0]
        time = 1
        slnW = []

    # Deterministic forcing
    n = 4
    Xi = 1
    Fk = -n*Xi*np.cos(n*Y)-n*Xi*np.cos(n*X)
    Fk = np.fft.fft2(Fk)
    # Fk = Gk*Fk

    Force_gpu   = gpuarray.to_gpu(np.reshape(Fk,NX*(NX),order='F'))

    print('Re = ', Re)
    plan  = cf.cufftPlan2d(NX, NX, cf.CUFFT_Z2Z)

    plan2 = cf.cufftPlan2d(NNX, NNX, cf.CUFFT_Z2Z)

    # slnU = np.zeros([NX,NNSAVE])
    # slnV = np.zeros([NX,NNSAVE])
    slnPsi = np.zeros([NNX,NNX,NNSAVE])
    slnWor = np.zeros([NNX,NNX,NNSAVE])
    slnCS = np.zeros([2,NNSAVE])


    #slnWorDNS = np.zeros([NX,NX,NNSAVE])
    #slnPsiDNS = np.zeros([NX,NX,NNSAVE])
    #slnUDNS = np.zeros([NX,NX,NNSAVE])
    #slnVDNS = np.zeros([NX,NX,NNSAVE])
    # slnPI = np.zeros([NNX,NNX,NNSAVE])
    # slnuF = np.zeros([NNX,NNX,NNSAVE])
    # slnvF = np.zeros([NNX,NNX,NNSAVE])
    # slnStress1 = np.zeros([NNX,NNX,NNSAVE])
    # slnStress2 = np.zeros([NNX,NNX,NNSAVE])
    # slnStress3 = np.zeros([NNX,NNX,NNSAVE])
    # slnForce1 = np.zeros([NNX,NNX,NNSAVE])
    # slnForce2 = np.zeros([NNX,NNX,NNSAVE])

    Energy = np.zeros([NNSAVE])
    Enstrophy = np.zeros([NNSAVE])

    #sampling_matrix_1 = np.zeros([21,NNX*NNX,NNSAVE],dtype=np.float64)
    #sampling_matrix = np.zeros([21,NNX,NNX],dtype=np.float64)
    #SGS = np.zeros([NNX*NNX,NNSAVE],dtype=np.float64)

    # onePython = np.zeros([NNSAVE])
    count = 0

    start_time = runtime.time()
    for it in range(maxit):
        if it == 0:
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

            # Eddy viscosity
            vt    = np.zeros([NX,NX],dtype = np.complex128)

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

            psiPrevious_hat_gpu = gpuarray.to_gpu(np.reshape(psiPrevious_hat,NX*(NX),order='F'))
            psiCurrent_hat_gpu  = gpuarray.to_gpu(np.reshape(psiCurrent_hat,NX*(NX),order='F'))
            kx_gpu              = gpuarray.to_gpu(kx)
            kkx_gpu             = gpuarray.to_gpu(kkx)

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

            # Eddy viscosity
            vt_gpu    = gpuarray.to_gpu(np.reshape(vt,NX*NX,order='F'))

            # Start simulation

            initialization(u0_hat_gpu, v0_hat_gpu, w0_hat_gpu, kx_gpu, psiPrevious_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

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

        # Calculate eddy viscosity
        DSMAG(vt_gpu, kx_gpu, w1_hat_gpu, psiCurrent_hat_gpu, NX, dx, TPB, NB, HNX, HNB, Delta, plan, ev_coeff_gpu,u1_gpu,v1_gpu,u1_hat_gpu,v1_hat_gpu,negRate_gpu)

        # Construct RHS
        RHS_ker(convec_hat_gpu, diffu_hat_gpu, vt_gpu, w1_hat_gpu, v1_hat_gpu, RHS_gpu, block=(TPB,1,1), grid=(NB,NX,1)) 

        # Deterministic forcing
        DDP_ker(RHS_gpu, Force_gpu, block=(1,TPB,1), grid=(1,NB,NX))
        # End forcing  
        
        updatew(w0_hat_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

        LHS_ker(kx_gpu, RHS_gpu, vt_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

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
            slnWor[:,:,count] = u

            cf.cufftExecZ2Z(plan, int(psiPrevious_hat_gpu.gpudata), int(dummy_gpu.gpudata), cf.CUFFT_INVERSE)
            psi = -np.real(np.reshape(dummy_gpu.get(),[NX,NX],order='F'))/(NX*NX)
            slnPsi[:,:,count] = psi
            
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

            print("Eddy viscosity:")
            vt = np.real(np.reshape(vt_gpu.get(),[NX,NX],order='F'))
            vt = np.mean(vt)
            print(vt)
            ccs = np.real(np.reshape(ev_coeff_gpu.get(),[NX,NX],order='F'))
            ccs = np.mean(np.sqrt(ccs)/dx)
            print("Cs:")
            print(ccs)

            slnCS[0,count] = ccs      
            slnCS[1,count] = vt

            # record velocities
            #cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
            #cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)

            #u = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
            #v = np.real(np.reshape(v1_gpu.get(),[NX,NX],order='F'))/(NX*NX)

            #slnU[:,count] = u[:,0]
            #slnV[:,count] = v[:,0]

            count = count + 1

        # Clear variables
        clearS_ker(vt_gpu, block=(1,1,1), grid=(1,1,1)) # Scalar operation

    print("--- %s seconds ---(end iteration)" % (runtime.time() - start_time))

    # psiCurrent_hat = psiCurrent_hat_gpu.get()
    # psiPrevious_hat = psiPrevious_hat_gpu.get()

    # psi = psiCurrent_hat
    # if saveTrue == 1:
    #     savemat('data_end_LDC_Poi_gpu_Kraichnan.mat', dict([('psiCurrent_hat', psiCurrent_hat), ('psiPrevious_hat', psiPrevious_hat), ('time', time)
    #     , ('slnW', slnW)])) 
    #     #savemat('TKE.mat',dict([('slnU',slnU),('slnV',slnV)]))
    #     #savemat('onePython.mat',dict([('onePython',onePython)]))
    #     #psiCurrent_hat = np.fft.fft2(slnPsi[:,:,-1])
    #     #psiPrevious_hat = np.fft.fft2(slnPsi[:,:,-2])
    #     #savemat('data_filtered.mat', dict([('psiCurrent_hat',psiCurrent_hat),('psiPrevious_hat',psiPrevious_hat),('time',time),('dt',dt)]))
    #     savemat('Energy.mat',dict([('Energy',Energy),('Enstrophy',Enstrophy)]))

    # print("--- %s seconds ---" % (runtime.time() - start_time))

    # cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(u0_gpu.gpudata), cf.CUFFT_INVERSE)
    # cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
    # cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)

    # conjugateS_ker(w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    # u0 = np.real(np.reshape(u0_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # u1 = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # v1 = np.real(np.reshape(v1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
    # Gk = np.real(np.reshape(Gk_gpu.get(),[NX,NX],order='F'))

    # u1 = np.reshape(u1_hat_gpu.get(),[NX,NX],order='F')
    # print(u[30,20] - u[30,44])
    # u = (np.reshape(diffu_hat_gpu.get(),[NX,N+1],order='F'))
    # u0 = np.real(prediction)
    # print("--- %s seconds ---" % (runtime.time() - start_time))
    # savemat('w1.mat',dict([('w1Python', u0),('u1Python', u1),('v1Python', v1),('slnW',slnW)]))
    print(time)
    savemat('CS.mat',dict([('CS', slnCS)]))

    print('Data size:')
    # slnWor.reshape([NX*NX, NNSAVE])
    print(np.shape(slnWor))
    # slnPsi.reshape([NX*NX, NNSAVE])
    print(np.shape(slnPsi))
    # X.reshape([NX*NX, 1])
    # Y.reshape([NX*NX, 1])
    t = np.linspace(0, time, num=NNSAVE)
    print(np.shape(t))
    savenc(slnPsi,slnWor, x, x, t, 'SMAG_' + str(version_))
    # Output
    # return slnWor, time
    return






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
    matfiledata[u'slnUDNS'] = slnUDNS
    matfiledata[u'slnVDNS'] = slnVDNS
    matfiledata[u'slnPsiDNS'] = slnPsiDNS
    #matfiledata[u'slnmaxWor'] = slnW
    hdf5storage.write(matfiledata, '.', 'DNS data.mat', matlab_compatible=True)
    '''

    '''
    matfiledata = {}
    matfiledata[u'slnWor'] = slnWor
    matfiledata[u'slnPsi'] = slnPsi
    hdf5storage.write(matfiledata, '.', 'FDNS vor Psi.mat', matlab_compatible=True)

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
