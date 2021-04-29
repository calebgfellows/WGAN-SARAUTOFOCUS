

## Creates a SAR projection operator ##

import numpy as np
from scipy.sparse import csr_matrix
import scipy
import math

def approx(mat,thr):

    maximum=np.max(np.max(np.abs(mat)));

    appmat=(np.abs(mat)>=(thr*maximum/100))*mat
    return appmat


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def form_SAR_projmtx(resol_pixsp_ratio,image_size,approx_level):
    a = resol_pixsp_ratio
    c = 3e8
    w0 = 2 * np.pi * 1e10 / a
    Tp = 4e-4
    fd = 1e12 / a
    wd = 2 * np.pi * fd
    B = fd * Tp
    rangeres = c / (2 * B)
    pixsp = rangeres / a
    D = pixsp * image_size
    r0 = D / 2
    fs = 2 * D * fd / c
    Tsamp = 1 / fs
    N = image_size
    Ns = int(Tp / Tsamp)
    Nang = Ns
    azres = rangeres
    angcoverage = 180 * c / (w0 * azres)
    ang = np.arange(-angcoverage/2, angcoverage/2, angcoverage/Nang)


    ang = ang[0:image_size]

    r = np.arange(-D / 2,D / 2,pixsp)
    r = (r + pixsp) / 2;


    t = np.arange(-Tp/2,Tp/2,Tsamp)
    t = t[0:Ns]
    w = 2 / c * (w0 + wd * t);

    x = np.linspace(D / 2, -D / 2 + pixsp, N)
    y = np.linspace(D / 2 - pixsp, -D / 2, N)

    [X, Y] = np.meshgrid(x, y);

    rr = np.array([(np.reshape(np.transpose(X),image_size*image_size)),(np.reshape(np.transpose(Y),image_size*image_size))])
    x = np.ceil(Ns * Nang * N * N * 0.07);
    T = np.zeros((Ns*Nang,N*N), dtype=complex)


    for ind in range(len(rr[0,:])):
        ycTtemp = np.zeros((Ns, Nang), dtype=complex);

        [ri, th] = cart2pol(rr[0,  ind], rr[1, ind])
        for k in range(Nang):
            phi1d = -w * (ri) * np.cos((th + ang[k] * np.pi / 180));
            x = phi1d * 1j

            yc1d = np.exp(phi1d *1j)
            yc1d_fft = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(np.conj(yc1d))));
            ycTtemp[:, k] = (np.transpose(yc1d_fft))

        Ttempcol = np.reshape(ycTtemp, (image_size*image_size), order='F')
        Ttempcol = approx(Ttempcol, approx_level);
        T[:, ind]=Ttempcol


    return T


