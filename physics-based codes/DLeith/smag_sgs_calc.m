function [ve,CS] = smag_sgs_calc(w1_hat,psiCurrent_hat,Kx,Ky,Ksq,dx,CS)
% Strain rate
% S1 = real(ifft2(-Ky.*Kx.*psiCurrent_hat));
% S2 = 0.5*real(ifft2(-(Kx.^2 - Ky.^2).*psiCurrent_hat));
% S  = 2*sqrt(S1.^2 + S2.^2);

% Rotation rate
wx = real(ifft2((1i*Kx).*w1_hat));
wy = real(ifft2((1i*Ky).*w1_hat));
S = sqrt(wx.*wx + wy.*wy);

% SGS - Dynamic Smagorinsky
NX = size(S,1);
% cs1 = compute_CS_smag(dx,NX,real(ifft2(-1i*Ky.*psiCurrent_hat)),real(ifft2(1i*Kx.*psiCurrent_hat)),S1,S2,S,Kx,Ky);
% 
% cs2 = compute_CS_smag_v2(dx,NX,psiCurrent_hat,w1_hat,Kx,Ky,Ksq,S);

% disp(sqrt(cs1)/dx)
% disp(sqrt(cs2)/dx)
% SGS - Standard Smagorinsky
% cs = (0.17*dx)^2;
% S = sqrt(mean(S(:).^2));
% ve = cs1*S;

% Dynamic Leith
cl = compute_Cl_DLeith_v2(dx,NX,psiCurrent_hat,w1_hat,Kx,Ky,Ksq,S);
CS = [CS;(cl).^(1/3)/dx];
S = sqrt(mean(S(:).^2));
ve = cl*S;

% Debug
% ve = cs;

% SGSx = 1i*Kx.*fft2(ve.*real(ifft2(1i*Kx.*w1_hat)));
% SGSy = 1i*Ky.*fft2(ve.*real(ifft2(1i*Ky.*w1_hat)));
% 
% SGS_hat = SGSx+SGSy;
% SGS2 = SGS.^2;
% mean(SGS2(:))
% SGS_hat = fft2(SGS);
end