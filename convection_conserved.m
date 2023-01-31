function convec_hat = convection_conserved(psiCurrent_hat, w1_hat, Kx, Ky)
% Convservative form
u1_hat = -(1i*Ky).*psiCurrent_hat;
v1_hat = (1i*Kx).*psiCurrent_hat;
w1 = real(ifft2(w1_hat));
conu1 = 1i*Kx.*fft2((real(ifft2(u1_hat)).*w1));
conv1 = 1i*Ky.*fft2((real(ifft2(v1_hat)).*w1));
convec_hat = conu1 + conv1;

% Non-conservative form
w1x_hat = 1i*Kx.*w1_hat;
w1y_hat = 1i*Ky.*w1_hat;
conu1 = fft2(real(ifft2(u1_hat)).*real(ifft2(w1x_hat)));
conv1 = fft2(real(ifft2(v1_hat)).*real(ifft2(w1y_hat)));
convecN_hat = conu1 + conv1;

convec_hat = 0.5*(convec_hat + convecN_hat);


end