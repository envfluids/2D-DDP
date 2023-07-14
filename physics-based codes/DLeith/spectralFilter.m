function uF = spectralFilter(u,NNX)
N = size(u,1)/2;
filterSize = NNX/2;
u = fftshift(u);
u_hat = fftshift(u(N-filterSize+2:N+1+filterSize,N-filterSize+2:N+1+filterSize))/(N/filterSize)^2;
u_hat = circshift(u_hat,1,1);
u_hat = circshift(u_hat,1,2);
uF = real(ifft2(u_hat));
% uF = u_hat;
end