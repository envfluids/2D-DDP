#--> Import pyCUDA
import pycuda.autoinit
import pycuda.driver as drv 
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from skcuda import cufft as cf
# import skcuda.linalg as skculinalg
# import cupy as cp




ker = SourceModule("""
#include <pycuda-complex.hpp>
#include "cuComplex.h"
#include <cufft.h>
#include "math.h"
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
# define M_PI           3.14159265358979323846  /* pi */

const int NX = 64;
const int NNX = 64; // Filter size
const int NX2 = NX*NX; // NX^2

__device__ double dt = 5e-4;
__device__ double nu = 5e-5;
__device__ double alpha = 0.1;
__device__ double beta = 20.0;
//__device__ double Delta = 0.196349540849362; //2*Lx/NX

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

__global__ void RHS_kernel(cufftDoubleComplex *convec,cufftDoubleComplex *diffu,cufftDoubleComplex *vt,cufftDoubleComplex *w1,cufftDoubleComplex *v1,\
    cufftDoubleComplex *RHS)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    RHS[j*NX+k].x = convec[j*NX+k].x + 0.5*dt*(nu+vt[j*NX+k].x)*diffu[j*NX+k].x + w1[j*NX+k].x + dt*beta*v1[j*NX+k].x;
    RHS[j*NX+k].y = convec[j*NX+k].y + 0.5*dt*(nu+vt[j*NX+k].x)*diffu[j*NX+k].y + w1[j*NX+k].y + dt*beta*v1[j*NX+k].y;
}

__global__ void LHS_kernel(double *kx, cufftDoubleComplex *RHS, cufftDoubleComplex *vt, cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    w[i*NX+j].x = RHS[i*NX+j].x / (1.0+dt*alpha+0.5*dt*(nu+vt[i*NX+j].x)*((kx[j]*kx[j] + kx[i]*kx[i])));
    w[i*NX+j].y = RHS[i*NX+j].y / (1.0+dt*alpha+0.5*dt*(nu+vt[i*NX+j].x)*((kx[j]*kx[j] + kx[i]*kx[i])));

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

__global__ void spectralFilter_same_size_2D_kernel(cufftDoubleComplex *data, cufftDoubleComplex *data_F)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int HNNX = NNX/4;
    if (i<=HNNX && j<=HNNX){
        data_F[i*NNX+j].x = data[i*NX+j].x;
        data_F[i*NNX+j].y = data[i*NX+j].y;
    }
    else if (i>(NX-HNNX) && j<=HNNX){
        data_F[i*NX+j].x = data[i*NX+j].x;
        data_F[i*NX+j].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i<=HNNX){
        data_F[i*NX+j].x = data[i*NX+j].x;
        data_F[i*NX+j].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i>(NX-HNNX)){
        data_F[i*NX+j].x = data[i*NX+j].x;
        data_F[i*NX+j].y = data[i*NX+j].y;
    }
    else{
        data_F[i*NX+j].x = 0.0;
        data_F[i*NX+j].y = 0.0;
    }
}

__global__ void addConvecC_kernel(cufftDoubleComplex *conu1, cufftDoubleComplex *conv1, double *kx, cufftDoubleComplex *convecC)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convecC[j*NX+k].x = -(kx[k]*conu1[j*NX+k].y  + kx[j]*conv1[j*NX+k].y);
    convecC[j*NX+k].y = (kx[k]*conu1[j*NX+k].x  + kx[j]*conv1[j*NX+k].x);
}

__global__ void GaussianFilter_2D_kernel(cufftDoubleComplex *u, double *Gk)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    u[j*NX+k].x = Gk[j*NX+k]*u[j*NX+k].x;
    u[j*NX+k].y = Gk[j*NX+k]*u[j*NX+k].y;
}


__global__ void S12_kernel(cufftDoubleComplex *s1, cufftDoubleComplex *s2, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    s1[i*NX+j].x = -kx[j]*kx[i]*psi[j+i*NX].x;
    s1[i*NX+j].y = -kx[j]*kx[i]*psi[j+i*NX].y;

    s2[i*NX+j].x = 0.5*(kx[i]*kx[i] - kx[j]*kx[j])*psi[j+i*NX].x;
    s2[i*NX+j].y = 0.5*(kx[i]*kx[i] - kx[j]*kx[j])*psi[j+i*NX].y;
}

__global__ void S_kernel(cufftDoubleComplex *s, cufftDoubleComplex *s1,\
     cufftDoubleComplex *s2)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    s[i*NX+j].x = 4*(s1[i*NX+j].x * s1[i*NX+j].x + s2[i*NX+j].x * s2[i*NX+j].x)/NX2/NX2;
    s[i*NX+j].y = 0;
}

__global__ void Sc_kernel(cufftDoubleComplex *s, cufftDoubleComplex *s1,\
     cufftDoubleComplex *s2)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    s[i*NX+j].x = 2*sqrt(s1[i*NX+j].x * s1[i*NX+j].x + s2[i*NX+j].x * s2[i*NX+j].x)/NX2;
    s[i*NX+j].y = 0;
}

__global__ void eddy_visc_kernel(cufftDoubleComplex *vt, \
    cufftDoubleComplex *S)
{
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NX; j++){
            vt[0].x +=  S[i*NX+j].x;
        }
    }   
    vt[0].x /= NX2;
}

__global__ void norm_eddy_visc_kernel(cufftDoubleComplex *vt, double *ev, double *s)
{  
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    vt[i*NX+j].x = ev[i*NX+j] * sqrt(s[0]);
    vt[i*NX+j].y = 0; 
}

__global__ void scalar_division_kernel(cufftDoubleComplex *a, cufftDoubleComplex *b, double *c)
{  
    c[0] = a[0].x / b[0].x;
    
}

__global__ void l_kernel(cufftDoubleComplex *a, cufftDoubleComplex *b, cufftDoubleComplex *c, cufftDoubleComplex *d)
{  
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    a[i*NX+j].x = b[i*NX+j].x/NX2 - (c[i*NX+j].x/NX2)*(d[i*NX+j].x/NX2);
    a[i*NX+j].y = 0.0;
    
}

__global__ void m_kernel(cufftDoubleComplex *a, cufftDoubleComplex *b, cufftDoubleComplex *c, cufftDoubleComplex *d)
{  
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    a[i*NX+j].x = 2.0*(b[i*NX+j].x/NX2 - 4.0*(c[i*NX+j].x)*(d[i*NX+j].x/NX2));
    a[i*NX+j].y = 0.0;
    
}

__global__ void division_kernel(double *a, cufftDoubleComplex *b, cufftDoubleComplex *c)
{  
    //unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    //unsigned int j = blockIdx.y;
    //a[i*NX+j] = b[i*NX+j].x/c[i*NX+j].x;   

    double bb;
    double cc;

    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NX; j++){
            bb +=  b[i*NX+j].x;
            cc +=  c[i*NX+j].x;
        }
    }   
    
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NX; j++){
            a[i*NX+j] =  bb/cc;
        }
    }
    
}

__global__ void scaleCs_kernel(double *a, double *b)
{  
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    if (a[i*NX+j] < 0) {
        a[i*NX+j] = a[i*NX+j]*b[0];
        }
       
}

__global__ void clearS_kernel(cufftDoubleComplex *w)
{
    w[0].x = 0.0;
    w[0].y = 0.0;
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
spectralFilter_same_size = ker.get_function("spectralFilter_same_size_2D_kernel")
GaussianFilter = ker.get_function("GaussianFilter_2D_kernel")
addConvecC = ker.get_function("addConvecC_kernel")
firstDx = ker.get_function("firstDx_kernel")
firstDy = ker.get_function("firstDy_kernel")

S12_ker = ker.get_function("S12_kernel")
S_ker = ker.get_function("S_kernel")
Sc_ker = ker.get_function("Sc_kernel")
eddy_visc_ker = ker.get_function("eddy_visc_kernel")
norm_eddy_visc_ker = ker.get_function("norm_eddy_visc_kernel")

clearS_ker = ker.get_function("clearS_kernel")

scalar_division = ker.get_function("scalar_division_kernel")
l_ker = ker.get_function("l_kernel")
m_ker = ker.get_function("m_kernel")
division_ker = ker.get_function("division_kernel")
scaleCs = ker.get_function("scaleCs_kernel")
