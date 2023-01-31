function uF = spectralFilter_same_size(u,NNX)
N = size(u,1);
filterSize = NNX/2;
u(filterSize+2:N-filterSize+1,:)=0;
u(:,filterSize+2:N-filterSize+1)=0;
uF = real(ifft2(u));
% uF = u;
end