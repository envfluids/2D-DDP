function cs = compute_Cl_DLeith_v2(dx,nx,psiCurrent_hat,w1_hat,Kx,Ky,Ksq, S)
% For DLeith, S is abs(\grad(\Omega))

kappa = 2;

nxc = nx/kappa;
JC = real(ifft2(-1i*Ky.*psiCurrent_hat)).*real(ifft2(1i*Kx.*w1_hat))+...
    real(ifft2(1i*Kx.*psiCurrent_hat)).*real(ifft2(1i*Ky.*w1_hat));
JCF_hat = spectralFilter_same_size(fft2(JC),nxc);
JCF = real(ifft2(JCF_hat));

psic_hat = spectralFilter_same_size(psiCurrent_hat,nxc);
w1c_hat = spectralFilter_same_size(w1_hat,nxc);

FJC = real(ifft2(-1i*Ky.*psic_hat)).*real(ifft2(1i*Kx.*w1c_hat))+...
    real(ifft2(1i*Kx.*psic_hat)).*real(ifft2(1i*Ky.*w1c_hat));

H = FJC - JCF;

diffu = real(ifft2(-Ksq.*w1_hat));
M2 = spectralFilter_same_size(fft2(S.*diffu),nxc);
M2 = real(ifft2(M2));

Sc_hat = spectralFilter_same_size(fft2(S),nxc);
Sc = real(ifft2(Sc_hat));
diffuc =  real(ifft2(-Ksq.*w1c_hat));
M1 = kappa^3*Sc.*diffuc;
M = M1 - M2;

aa = H.*M;
bb = M.*M;

aa = 0.5*(aa+abs(aa));
cs = abs(sum(aa(:))/sum(bb(:)));



end