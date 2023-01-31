function cs = compute_CS_smag(dx,NX,u1,v1,S1,S2,S,KKx,KKy)
    nxc = NX/2;
    uu = u1.*u1;
    uv = u1.*v1;
    vv = v1.*v1;
    
    uuc = spectralFilter_same_size(fft2(uu),nxc);
    uvc = spectralFilter_same_size(fft2(uv),nxc);
    vvc = spectralFilter_same_size(fft2(vv),nxc);
    
    uc  = spectralFilter_same_size(fft2(u1),nxc);
    vc  = spectralFilter_same_size(fft2(v1),nxc);
    
    ucx = 1i*KKx.*uc;
    ucy = 1i*KKy.*uc;
    vcx = 1i*KKx.*vc;
    vcy = 1i*KKy.*vc;
    
    Sc1 = real(ifft2(ucx));
    Sc2 = 0.5*real(ifft2((ucy+vcx)));
    Sc  = 2*sqrt(Sc1.^2 + Sc2.^2);
    
    hc1 = S.*S1;
    hc2 = S.*S2;
    
    hcc1 = spectralFilter_same_size(fft2(hc1),nxc);
    hcc2 = spectralFilter_same_size(fft2(hc2),nxc);
    
    hcc1 = real(ifft2(hcc1));
    hcc2 = real(ifft2(hcc2));
    
    l11 = real(ifft2(uuc)) - real(ifft2(uc)).*real(ifft2(uc));
    l12 = real(ifft2(uvc)) - real(ifft2(uc)).*real(ifft2(vc));
    l22 = real(ifft2(vvc)) - real(ifft2(vc)).*real(ifft2(vc));
    
    l11d = l11 - 1/2*(l11 + l22);
    l12d = l12;
    l22d = l22 - 1/2*(l11 + l22);
    
    m11 = 2*(hcc1-2^2*Sc.*Sc1);
    m12 = 2*(hcc2-2^2*Sc.*Sc2);
    m22 = m11;
    
    aa = (l11.*m11 + 2.0*(l12.*m12) + l22.*m22);
    bb = (m11.*m11 + 2.0*(m12.*m12) + m22.*m22);
    
    aa = 0.5*(aa+abs(aa));
    cs = sum(aa(:))/sum(bb(:));
    
end