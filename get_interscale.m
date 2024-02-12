clear all,close all,clc
fileName = ['Case3p3/sharp.nc'];

C_JHS = 0.22;
CB_JHS = 0.951;
C_JHL = 0.29;
CB_JHL = 0.95;

Cs_Smag1 = 0.17;
Cs_Leith1 = 0.21;

  
source = fileName;
varname = '/Temp_data/OMEGA';
WW = ncread(source,varname);
varname = '/Temp_data/PI';
slnPI = ncread(source,varname);


ND = size(WW,1);

NX = size(WW,2);
Lx = 2*pi;

% Wavenumbers
dx = Lx/NX;
x = linspace(0,Lx-dx,NX);
kx = (2*pi/Lx)*[0:(NX/2) (-NX/2+1):-1];
[Kx,Ky] = meshgrid(kx,kx);
% Energy spectrum for all wavenumbers
Kabs = sqrt(Kx.^2 + Ky.^2);
Ksq = Kx.^2 + Ky.^2;
Ksq4 = Ksq.*Ksq;
Kabs(1) = 10^12;

interscale_Energy = zeros(ND,1);
interscale_Enstrophy = zeros(ND,1);

interscale_Energy_Smag = zeros(ND,1);
interscale_Enstrophy_Smag = zeros(ND,1);

interscale_Energy_Leith = zeros(ND,1);
interscale_Enstrophy_Leith = zeros(ND,1);

interscale_Energy_DSmag = zeros(ND,1);
interscale_Enstrophy_DSmag = zeros(ND,1);

interscale_Energy_DLeith = zeros(ND,1);
interscale_Enstrophy_DLeith = zeros(ND,1);

interscale_Energy_DSmagP = zeros(ND,1);
interscale_Enstrophy_DSmagP = zeros(ND,1);

interscale_Energy_DLeithP = zeros(ND,1);
interscale_Enstrophy_DLeithP = zeros(ND,1);

interscale_Energy_JHS = zeros(ND,1);
interscale_Enstrophy_JHS = zeros(ND,1);

interscale_Energy_JHL = zeros(ND,1);
interscale_Enstrophy_JHL = zeros(ND,1);


DCs = zeros(ND,1);
DCsP = zeros(ND,1);
DCl = zeros(ND,1);
DClP = zeros(ND,1);

% curve
PE_FDNS = zeros(NX/2,ND);
PE_Smag = zeros(NX/2,ND);
PE_Leith = zeros(NX/2,ND);
PE_JHS = zeros(NX/2,ND);
PE_JHL = zeros(NX/2,ND);

for it = 1:ND
    
    % LES
    wor = reshape(WW(it,:,:),NX,NX);
    wor_hat = fft2(wor);
    psi_hat = wor_hat./(Kabs.^2); 
    psi = real(ifft2(psi_hat));
    
    diffu_hat = -1*Ksq.*wor_hat;
    hyperdiffu_hat = Ksq4.*wor_hat;

    % FDNS
    PI = -reshape(slnPI(it,:,:),NX,NX);
    
    PI_hat = fft2(PI);
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_FDNS(:,it) = temp(1:NX/2);

    % Calculate interscale transfers
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;

    interscale_Energy(it) = mean(ene(:));
    interscale_Enstrophy(it) = mean(ens(:));
    
    % Smag
    S1 = real(ifft2(-Ky.*Kx.*psi_hat));
    S2 = 0.5*real(ifft2(-(Kx.^2 - Ky.^2).*psi_hat));
    S  = 2*sqrt((S1.^2 + S2.^2));
    cs = (Cs_Smag1*dx)^2;
    
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Smag(:,it) = temp(1:NX/2);
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;

    interscale_Energy_Smag(it) = mean(ene(:));
    interscale_Enstrophy_Smag(it) = mean(ens(:));
    
    % Leith
    w1x_hat = 1i*Kx.*wor_hat;
    w1y_hat = 1i*Ky.*wor_hat;
    S1 = real(ifft2(w1x_hat));
    S2 = real(ifft2(w1y_hat));
    S  = sqrt((S1.^2 + S2.^2));
    cs = (Cs_Leith1*dx)^3;
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Leith(:,it) = temp(1:NX/2);
    
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;
    
    interscale_Energy_Leith(it) = mean(ene(:));
    interscale_Enstrophy_Leith(it) = mean(ens(:));
    
    % DSmag
    S1 = real(ifft2(-Ky.*Kx.*psi_hat));
    S2 = 0.5*real(ifft2(-(Kx.^2 - Ky.^2).*psi_hat));
    S  = 2*sqrt((S1.^2 + S2.^2));
    Cs_Smag = compute_CS_smag(dx,NX,real(ifft2(1i*Ky.*psi_hat)),real(ifft2(-1i*Kx.*psi_hat)),S1,S2,S,Kx,Ky); 
    
    DCs(it) = Cs_Smag.^(1/2)/dx;
    
    cs = Cs_Smag;
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Smag(:,it) = temp(1:NX/2);
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;

    interscale_Energy_DSmag(it) = mean(ene(:));
    interscale_Enstrophy_DSmag(it) = mean(ens(:));
    
    % DLeith
    w1x_hat = 1i*Kx.*wor_hat;
    w1y_hat = 1i*Ky.*wor_hat;
    S1 = real(ifft2(w1x_hat));
    S2 = real(ifft2(w1y_hat));
    S  = sqrt((S1.^2 + S2.^2));
    Cs_Leith = compute_Cl_DLeith(dx,NX,-psi_hat,wor_hat,Kx,Ky,Ksq,S);
    DCl(it) = Cs_Leith^(1/3)/dx;
    cs = (Cs_Leith);
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Leith(:,it) = temp(1:NX/2);
    
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;
    
    interscale_Energy_DLeith(it) = mean(ene(:));
    interscale_Enstrophy_DLeith(it) = mean(ens(:));
    
    % DSmag_>0
    S1 = real(ifft2(-Ky.*Kx.*psi_hat));
    S2 = 0.5*real(ifft2(-(Kx.^2 - Ky.^2).*psi_hat));
    S  = 2*sqrt((S1.^2 + S2.^2));
    Cs_Smag = compute_CS_smagP(dx,NX,real(ifft2(1i*Ky.*psi_hat)),real(ifft2(-1i*Kx.*psi_hat)),S1,S2,S,Kx,Ky); 
    
    DCsP(it) = (Cs_Smag).^(0.5)/dx;
    
    cs = Cs_Smag;
    
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Smag(:,it) = temp(1:NX/2);
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;

    interscale_Energy_DSmagP(it) = mean(ene(:));
    interscale_Enstrophy_DSmagP(it) = mean(ens(:));
    
    % DLeith>0
    w1x_hat = 1i*Kx.*wor_hat;
    w1y_hat = 1i*Ky.*wor_hat;
    S1 = real(ifft2(w1x_hat));
    S2 = real(ifft2(w1y_hat));
    S  = sqrt((S1.^2 + S2.^2));
    
    Cs_Leith = compute_Cl_DLeithP(dx,NX,-psi_hat,wor_hat,Kx,Ky,Ksq,S);
    DClP(it) = Cs_Leith.^(1/3)/dx;
    
    cs = (Cs_Leith);
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    PI_hat = ve*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_Leith(:,it) = temp(1:NX/2);
    
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;
    
    interscale_Energy_DLeithP(it) = mean(ene(:));
    interscale_Enstrophy_DLeithP(it) = mean(ens(:));
    
%   JHS
    S1 = real(ifft2(-Ky.*Kx.*psi_hat));
    S2 = 0.5*real(ifft2(-(Kx.^2 - Ky.^2).*psi_hat));
    S  = 2*sqrt((S1.^2 + S2.^2));
    cs = (C_JHS*dx)^4;
    cB = CB_JHS;
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    Fsmag = -1*ve*real(ifft2(hyperdiffu_hat));
    vb1 = real(ifft2(psi_hat)).*Fsmag;
    vb2 = real(ifft2(psi_hat)).*real(ifft2(diffu_hat));
    vb = cB*mean(vb1(:))/mean(vb2(:));
    PI_hat = - ve*hyperdiffu_hat - vb*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_JHS(:,it) = temp(1:NX/2);
    
    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;
    
    interscale_Energy_JHS(it) = mean(ene(:));
    interscale_Enstrophy_JHS(it) = mean(ens(:));
    
    % JHL
    w1x_hat = 1i*Kx.*wor_hat;
    w1y_hat = 1i*Ky.*wor_hat;
    S1 = real(ifft2(w1x_hat));
    S2 = real(ifft2(w1y_hat));
    S  = sqrt((S1.^2 + S2.^2));
    cs = (C_JHL*dx)^5;
    cB = CB_JHL;
    S = sqrt(mean(S(:).^2));
    ve = cs*S;
    Fsmag = -1*ve*real(ifft2(hyperdiffu_hat));
    vb1 = real(ifft2(psi_hat)).*Fsmag;
    vb2 = real(ifft2(psi_hat)).*real(ifft2(diffu_hat));
    vb = cB*mean(vb1(:))/mean(vb2(:));
    PI_hat = -1*ve*hyperdiffu_hat - vb*diffu_hat;
    PI = real(ifft2(PI_hat));
    
    Pe = real(conj(psi_hat).*PI_hat);
    temp = angle_av(Pe,Kabs, NX);
    PE_JHL(:,it) = temp(1:NX/2);

    ene = 0.5*psi.*PI;
    ens = 0.5*wor.*PI;
    
    interscale_Energy_JHL(it) = mean(ene(:));
    interscale_Enstrophy_JHL(it) = mean(ens(:));
end

%%
% figure
% plot(mean(PE_FDNS,2),'k');
% hold on
% plot(mean(PE_Smag,2),'r--');
% plot(mean(PE_Leith,2),'b--');
% plot(mean(PE_JHS,2),'r');
% plot(mean(PE_JHL,2),'b');



save('interscale_FDNS.mat','interscale_Energy','interscale_Enstrophy','interscale_Energy_Smag','interscale_Enstrophy_Smag'....
,'interscale_Energy_Leith','interscale_Enstrophy_Leith','interscale_Energy_JHS','interscale_Enstrophy_JHS','interscale_Energy_JHL','interscale_Enstrophy_JHL');

% disp('Interscale Energy FDNS:')
% mean(interscale_Energy)
% std(interscale_Energy)
% disp('Interscale Enstrophy FDNS:')
% mean(interscale_Enstrophy)
% std(interscale_Enstrophy)
disp('Interscale Energy Smag:')
mean(interscale_Energy_Smag)
% std(interscale_Energy_Smag)
disp('Interscale Enstrophy Smag:')
mean(interscale_Enstrophy_Smag)
% std(interscale_Enstrophy_Smag)
% disp('Interscale Energy Leith:')
% mean(interscale_Energy_Leith)
% std(interscale_Energy_Leith)
% disp('Interscale Enstrophy Leith:')
% mean(interscale_Enstrophy_Leith)
% std(interscale_Enstrophy_Leith)
% disp('Interscale Energy JHS:')
% mean(interscale_Energy_JHS)
% std(interscale_Energy_JHS)
% disp('Interscale Enstrophy JHS:')
% mean(interscale_Enstrophy_JHS)
% std(interscale_Enstrophy_JHS)
% disp('Interscale Energy JHL:')
% mean(interscale_Energy_JHL)
% std(interscale_Energy_JHL)
% disp('Interscale Enstrophy JHL:')
% mean(interscale_Enstrophy_JHL)
% std(interscale_Enstrophy_JHL)


%%save(['TKE'],'TKE','tke')

