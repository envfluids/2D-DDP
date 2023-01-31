close all; clear variables; clc
% Solve lid-driven cavity by cheb-cheb method
% Navier-Stokes equation is in the vorticity-stream function form
% %% Preprocessing
%addpath('C:\Users\Yifei\Google Drive\LDC\MATLAB\YG\SC\');
% addpath('C:\Users\yy\Google Drive\LDC\MATLAB\YG');
addpath('/home/yifei/Desktop/matlab/P/long/Re10k/SC');
addpath('/home/yifei/Downloads/b2r');

% load('slnPI.mat');
% load('convectionF.mat');
tic
readTrue = 1;
saveTrue = 0;
NSAVE = 1; 
NNSAVE = 3000;
maxit = NSAVE*NNSAVE-1;
dt = 1e-3;
nu = 3.125e-5;%2*6.25e-4;%3.125e-1;
rho = 1;
Re = 1/nu;
% Create the Fourier wavenumbers
Lx = 2*pi;

% For dynamic Smagorinsky wavenumbers
% NX = 4;%256;%N*Lx/2;
% dx = Lx/NX;
% x = linspace(0,Lx-dx,NX);
% kx = (2*pi/Lx)*[0:(NX/2) (-NX/2+1):-1];
% [KKx,KKy] = meshgrid(kx,kx);

% True resolution
NX = 256;%N*Lx/2;
dx = Lx/NX;
x = linspace(0,Lx-dx,NX);
kx = (2*pi/Lx)*[0:(NX/2) (-NX/2+1):-1];
[Kx,Ky] = meshgrid(kx,kx);
Ksq = Kx.^2 + Ky.^2;
invKsq = 1./Ksq;
invKsq(1,1) = 0;
% Create the tensor product mesh
[X, Y] = meshgrid(x,x);


CS = [];
VE = [];
% Initialize
if readTrue==0
    % Initial condition
%     fk = find(Ksq ~= 0);
%     ckappa = zeros(size(Ksq));
%     ckappa(fk) = (sqrt(Ksq(fk)).*(1+(Ksq(fk)/36).^2)).^-0.5;
%     Psi_hat = (randn(NX,NX) + 1i*randn(NX,NX)).*ckappa;
%     Psi = ifft2(Psi_hat);
%     Psi = Psi - mean(Psi(:));
%     Psi_hat = fft2(Psi);
%     
%     u_K = (-1i*(Ky).*Psi_hat);
%     v_K = (1i*(Kx).*Psi_hat);
%     u = real(ifft2(u_K));
%     v = real(ifft2(v_K));
%     Ekin = 0.5*(u.^2 + v.^2);
%     EK = fft2(Ekin);
%     Psi_hat = Psi_hat./sqrt(2*sum(sum((EK))))*10;
% 
%     w1_hat = -Ksq.*Psi_hat;
    
%     w1_hat         = zeros(NX,NX);
%     w1_hat(5,1)    = 1*(randn(1) + 1i*randn(1));
%     w1_hat(2,2)    = 1*(randn(1) + 1i*randn(1));
%     w1_hat(10,10)    = 1*(randn(1) + 1i*randn(1));
%     w1_hat = rand(NX,NX);
%     psi_hat_rand = 1*randn(NX,NX);
%     w1_hat(1:10,1:10) = psi_hat_rand(1:10,1:10);
%     w1_hat(1,1) = 0;
%     w1 = real(ifft2(w1_hat));
%     w1 = w1/max(max(w1));
% %     w1 = rand(NX,NX);
%     w1 = w1./max(max(w1))*20;
%     w1_hat = fft2(w1);
    
% Initial condition

    NX_dns = 256;
    ii = NX_dns/NX;
    dx_dns = Lx/NX_dns;
    x_dns = linspace(0,Lx-dx,NX_dns);
    kx_dns = (2*pi/Lx)*[0:(NX_dns/2) (-NX_dns/2+1):-1];
    [Kx_dns,Ky_dns] = meshgrid(kx_dns,kx_dns);
    Ksq_dns = Kx_dns.^2 + Ky_dns.^2;
    
    

    kp    = 10;
    A     = 4*kp^(-5)/(3*pi);
    absK_dns  = sqrt(Kx_dns.^2 + Ky_dns.^2);
    Ek    = A*absK_dns.^4.*exp(-(absK_dns/kp).^2);
    
%     Ek = 25*absK_dns.^7/kp^8.*exp(-3.5*(absK_dns/kp).^2);
    
    shiftKx = fftshift(Kx_dns);
    shiftKy = fftshift(Ky_dns);
    coef1 = 2*pi*rand(NX_dns/2+1,NX_dns/2+1);
    coef2 = 2*pi*rand(NX_dns/2+1,NX_dns/2+1);
    
    perturb = zeros(NX_dns,NX_dns);
    perturb(1:NX_dns/2+1,1:NX_dns/2+1) = coef1(1:NX_dns/2+1,1:NX_dns/2+1) + coef2(1:NX_dns/2+1,1:NX_dns/2+1);
    perturb(1:NX_dns/2+1,NX_dns/2+2:end) = coef1(1:NX_dns/2+1,NX_dns/2:-1:2) - coef2(1:NX_dns/2+1,NX_dns/2:-1:2);
    perturb(NX_dns/2+2:end,1:NX_dns/2+1) = coef2(NX_dns/2:-1:2,1:NX_dns/2+1) - coef1(NX_dns/2:-1:2,1:NX_dns/2+1);
    perturb(NX_dns/2+2:end,NX_dns/2+2:end) = -(coef1(NX_dns/2:-1:2,NX_dns/2:-1:2) + coef2(NX_dns/2:-1:2,NX_dns/2:-1:2));
    perturb = exp(1i*perturb);
    
    w1_dns_hat = sqrt(absK_dns/pi.*Ek).*perturb*NX_dns*NX_dns;
    
    w1_dns = real(ifft2(w1_dns_hat));
    
    for i = 1:NX
        for j = 1:NX
            w1(i,j) = w1_dns(i*ii,j*ii);
        end
    end
%     w1 = w1/std(w1(:));
    w1_hat = fft2(w1);
    w1_dns = [];
    Kx_dns = [];
    Ky_dns = [];
    Ek = [];
    absk_dns = [];
    
    
    time = 0;
    slnW = [];

else
%     load(['Poislnvor', num2str(NX),'KT.mat']);
%     load(['data_filtered_NX',num2str(NX),'.mat']);
    load('Vorticity_Pi.mat','slnWor')
    w1 = reshape(slnWor(:,:,501),256,256).';
    w1_hat = fft2(w1);
    psiCurrent_hat = -invKsq.*w1_hat;
    psiPrevious_hat = psiCurrent_hat;
    time = 0;
    slnW = [];
    if (size(psiPrevious_hat,2) == 1)
        psiPrevious_hat = reshape(psiPrevious_hat,NX,NX);
        psiCurrent_hat = reshape(psiCurrent_hat,NX,NX);
    end
    if (size(psiPrevious_hat,1) == 1)
        psiPrevious_hat = reshape(psiPrevious_hat,NX,NX);
        psiCurrent_hat = reshape(psiCurrent_hat,NX,NX);
        psiPrevious_hat = psiPrevious_hat.';
        psiCurrent_hat = psiCurrent_hat.';
        slnW = slnW.';
    end
end

slnTKE = [];
slnU = [];
slnV = [];
slnEn = [];
slnWor = zeros(NX,NX,NNSAVE);


% slnPsi = zeros(NX,NX,NNSAVE);
% slnWor = slnPsi;
% slnPI  = slnPsi;
count  = 0;

Preproces = toc;
runtime = Preproces;
disp('Preprocessing time...');
disp(Preproces);
vidfile = VideoWriter(['Vorticity',num2str(Re),'KT']);
open(vidfile);
for it = 1:maxit
    
if it == 1
    % On the first iteration, w0 will be empty
%     disp('creating persistent variables in RHS')
    if time == 0
%         psiPrevious_hat = fft2(psi);
        psiPrevious_hat = -invKsq.*w1_hat;
        psiCurrent_hat  = psiPrevious_hat;
    end
    
    w1_hat = zeros((NX),(NX));
    u1_hat = zeros((NX),(NX));
    v1_hat = zeros((NX),(NX));
    w0_hat = zeros((NX),(NX));
    u0_hat = zeros((NX),(NX));
    v0_hat = zeros((NX),(NX));
    w1x_hat = zeros((NX),(NX));
    w1y_hat = zeros((NX),(NX));
    w0x_hat = zeros((NX),(NX));
    w0y_hat = zeros((NX),(NX));
    diffu_hat = zeros((NX),(NX));
    
    w0_hat = -Ksq.*psiPrevious_hat;
    u0_hat = -(1i*Ky).*psiPrevious_hat;
    v0_hat = (1i*Kx).*psiPrevious_hat;
    
    w1_hat = -Ksq.*psiCurrent_hat;
else
    
end

% 3 steps R-K
% Step 1
% convec_hat = convection_conserved(psiCurrent_hat, w1_hat, Kx, Ky);
% diffu_hat = -Ksq.*w1_hat;
% % SGS - standard Smagorinsky
% sgs = smag_sgs_calc(w1_hat,psiCurrent_hat,Kx,Ky,Ksq, dx);
% 
% w1_1_hat = w1_hat + dt*(-convec_hat+nu*diffu_hat+sgs);
% psi_1_hat = -w1_1_hat.*invKsq;
% % Step 2
% convec_hat = convection_conserved(psi_1_hat, w1_1_hat, Kx, Ky);
% diffu_hat = -Ksq.*w1_1_hat;
% sgs = smag_sgs_calc(w1_1_hat,psi_1_hat,Kx,Ky,Ksq, dx);
% 
% w1_2_hat = 0.75*w1_hat + 0.25*(w1_1_hat + dt*(-convec_hat+nu*diffu_hat+sgs));
% psi_2_hat = -w1_2_hat.*invKsq;
% % Step 3
% convec_hat = convection_conserved(psi_2_hat, w1_2_hat, Kx, Ky);
% diffu_hat = -Ksq.*w1_2_hat;
% sgs = smag_sgs_calc(w1_2_hat,psi_2_hat,Kx,Ky,Ksq, dx);
% 
% psiTemp = 1/3*w1_hat + 2/3*(w1_2_hat + dt*(-convec_hat+nu*diffu_hat+sgs));
 

% 2 Adam bash forth Crank Nicolson
convec1_hat = convection_conserved(psiCurrent_hat, w1_hat, Kx, Ky);
convec0_hat = convection_conserved(psiPrevious_hat, w0_hat, Kx, Ky);
diffu_hat = -Ksq.*w1_hat;
[ve,CS] = smag_sgs_calc(w1_hat,psiCurrent_hat,Kx,Ky,Ksq, dx, CS);
% ve = 0; % No SGS
RHS = w1_hat + dt*(-1.5*convec1_hat + 0.5*convec0_hat) + dt*0.5*(nu+ve)*diffu_hat;
RHS(1,1) = 0;
psiTemp = RHS./(1+0.5*dt*(nu+ve)*Ksq);

 % shifting
 w0_hat = w1_hat;
 w1_hat  = psiTemp;
 
 % Poisson equation for omega
 psiPrevious_hat = psiCurrent_hat;
 psiCurrent_hat = -w1_hat.*invKsq;
%  psiCurrent_hat(1,1) = 0;
 
 
    if (mod(it,NSAVE) == 0 || it == maxit)
        disp('Time=');
        disp(time);
        disp(['Iteration/' num2str(maxit)]);
        disp(it);
        disp('Runtime');
        disp(toc-runtime);
        runtime = toc;
%         figure
%         contourf(X,Y,real(ifft(reshape(u1_hat,NX,NX),[],2)));
%         tempW = max(max(real(ifft(reshape(w1_hat,NX,NX),[],2))))
        tempW = real(ifft2(reshape(w0_hat,NX,NX)));
        slnWor(:,:,count+1) = tempW;
%         slnEn = [slnEn;tempW(3,:).*tempW(3,:)];
        tempW = max(max(tempW));
        slnW = [slnW; tempW];
%         save(['Poislnvor', num2str(N), '.mat'],'psiCurrent_hat','psiPrevious_hat','time','slnW','-v7.3');
        u1_hat = -1i*Ky.*psiCurrent_hat;
        v1_hat = 1i*Kx.*psiCurrent_hat;
        tempu = real(ifft2(reshape(u1_hat,NX,NX)));
        tempv = real(ifft2(reshape(v1_hat,NX,NX)));
        
        slnU = [slnU;tempu(3,:)];
        slnV = [slnV;tempv(3,:)];
%         
%         Spectral Filtering
%         psiF = spectralFilter(psiCurrent_hat);
%         worF = spectralFilter(w1_hat);
%         
%         convectionF = spectralFilter(conu1+conv1);
%         u1F = spectralFilter(u1_hat);
%         v1F = spectralFilter(v1_hat);
%         w1xF = spectralFilter(w1x_hat);
%         w1yF = spectralFilter(w1y_hat);
%         convectionF2 = u1F.*w1xF + v1F.*w1yF;
%         
%         pi = convectionF - convectionF2;
%         

%         slnPsi(:,:,count) = psiF;
%         slnWor(:,:,count) = worF;
%         slnPI(:,:,count) = pi;
        
%         z = worF;
        
        im = sc(slnWor(:,:,count+1),[-tempW tempW],b2r(-40,40));
        writeVideo(vidfile, im);
        count  = count + 1;
        % Show Smagorinsky LES eddy viscosity
%         max(ve(:))
    end
    time = time + dt;
end
runtime = toc;
disp('Runtime');
disp(runtime);
close(vidfile)

%%
psi = real(ifft2(reshape(psiCurrent_hat,NX,NX)));
u1_hat = -1i*Ky.*psiCurrent_hat;
v1_hat = 1i*Kx.*psiCurrent_hat;
u1 = real(ifft2(reshape(u1_hat,NX,NX)));
v1 = real(ifft2(reshape(v1_hat,NX,NX)));
w1 = real(ifft2(reshape(w1_hat,NX,NX)));
contourf(X,Y,u1);
colormap('jet');
grid on
title('u');
figure
contourf(X,Y,v1);
colormap('jet');
grid on
title('v');
figure
contourf(X,Y,w1);
colormap('jet');
grid on
title('vorticity');
figure
contourf(X,Y,psi);
colormap('jet');
grid on
title('\Psi');
% figure
% % h1 = quiver(X,Y,u1,v1);
% % set(h1,'AutoScale','on', 'AutoScaleFactor', 5)
% starty = -1:0.01:1;
% startx = ones(size(starty))*Lx/2;
% h2=streamline(X,Y,u1,v1,startx,starty);
% set(h2,'LineWidth',1,'Color','k')
% axis([0 Lx -1 1]);


% Save data
if saveTrue
    save(['Poislnvor', num2str(NX), 'KT.mat'],'psiCurrent_hat',...
        'psiPrevious_hat','time','slnW','-v7.3');
    psiCurrent = real(ifft2(reshape(psiCurrent_hat,NX,NX)));
    psiPrevious = real(ifft2(reshape(psiPrevious_hat,NX,NX)));
    save(['Poislnvor_python', num2str(NX), 'KT.mat'],'psiCurrent','psiPrevious','time','slnW','-v7.3');
%     save(['vorticity_field_KT',num2str(Re),'.mat'],'slnPsi','slnWor','slnPI','-v7.3');
    onePLES = slnW;
    save('onePLES.mat','onePLES');
end
figure
plot(slnW)
slnWorDSMAG = slnWor;
save('w1LES_3s_Re32k_DSMAG.mat','slnWorDSMAG','-v7.3')
% uana = -0.5*3*uw*y.^2 + 0.5*uw;
% error = uana - u1(:,1);
% figure
% plot(y,error);
% norm(error)
% TKE spectrum
% meanU = mean(slnU);
% meanV = mean(slnV);
% meanU = ones(size(slnU,1),1)*meanU;
% meanV = ones(size(slnU,1),1)*meanV;
% uPrime = slnU - meanU;
% vPrime = slnV - meanV;
% TKE = 0.5*(uPrime.^2 + vPrime.^2);
% spectrum = mean(abs(fft((TKE),[],2)));
% figure
% loglog(kx(2:NX/2+1),spectrum(2:NX/2+1))
% save('spectrum_DY_LES.mat','spectrum');
% hold on
% loglog(kx(2:NX/2+1),kx(2:NX/2+1).^(-3))
% 
% enstrophy = abs(fft(mean(slnEn)));
% figure
% loglog(kx(1:NX/2+1),enstrophy(1:NX/2+1))
% hold on
% loglog(kx(1:NX/2+1),kx(1:NX/2+1).^(-3))
% 
% % yplus
% meanU = meanU(1,:);
% tauw = nu*mean(meanU)/(y(3)+1);
% utau = sqrt(abs(tauw));
% yplus = utau*(y(3)+1)/nu
%% Energy spectrum
% uprime = u1 - mean(u1(:));
% vprime = v1 - mean(v1(:));
% 
% TKE = sqrt(uprime.^2 + vprime.^2);
% spectrum = mean(abs(fft(TKE)),2);
% figure
% loglog(spectrum(2:NX/2))


%% Energy spectrum
% k = 1;
% for j = 1:NX
%     for i = 1:NX
%         data1d(k) = w1(i,j);
%         data1d(k+1) = 0;
%         k = k+2;
%     end
% end
% 
% for k = 1:2*NX*NX
%     data1d(k) = data1d(k)/(NX*NX);
% end
% 
% data1d = abs(fft(data1d));
% es = zeros(NX,NX);
% k = 1;
% for j = 1:NX
%     for i = 1:NX
%         kk = sqrt(kx(i)*kx(i) + kx(j)*kx(j));
%         es(i,j) = pi*(data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1))/kk;
%         k = k+2;
%     end
% end
% 
% for k = 1:NX*NX
%     eplot(k) = 0;
%     ic = 0;
%     for j = 1:NX
%         for i = 1:NX
%             kk = sqrt(kx(i)*kx(i) + kx(j)*kx(j));
%             if (kk>=k-0.5 && kk<=k+0.5)
%                 ic = ic+1;
%                 eplot(k) = eplot(k) + es(i,j);
%             end
%         end
%     end
% eplot(k) = eplot(k) / ic;    
% end
