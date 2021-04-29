
## Generates and saves a phase-corrupted image ##


import numpy as np
import cv2
import os
from scipy.linalg import dft
import form_SAR_projection

def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x

def normal(x, mu, sigma):
    return (2. * np.pi * sigma ** 2.) ** -.5 * np.exp(-.5 * (x - mu) ** 2. / sigma ** 2.)

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)


    return W

def run(images, add, image_path):
    (k, m) = (images[0].shape)
    [scH, scW] = np.shape(images[0])
    imagenumber = len(images)

    I = scH * scW

    T = form_SAR_projection.form_SAR_projmtx(1, scH, 0)

    A = dft(np.sqrt(I))



    D1 = np.zeros((I,I),complex)


    for d in range(imagenumber):
        c = d + add
        print(c)
        scene = images[d]/255
        random = np.random.rand(m)
        scale_scene = cv2.convertScaleAbs(scene, alpha=(255.0))

        cv2.imwrite(os.path.join(image_path, ("Org\org%d.png") % c), scale_scene)
        scene_vec = np.reshape(scene, (I, 1), order='F')

        for i in range(k):
            D1[(i*m):(m*(i+1)), (i*m):(m*(i+1))] = A;


        C = np.matmul(D1, T);
        phase_hist = np.matmul(C, scene_vec)

        phase_hist_reshape = np.reshape(phase_hist, (m, k),  order='F');
        RR = np.abs(phase_hist_reshape)
        QQ = np.angle(phase_hist_reshape)

        x = np.arange(-1, 1 + 2 / (m - 1), 2 / (m - 1))
        y = 30 * (x ** 2)

        Y = max(y) - y

        for kk in range(m):
            QQ[:, kk] = QQ[:, kk] + Y[kk] # random[kk]


        phase_hist_reshape_err = RR * np.exp(1j * (QQ))
        phase_error = np.reshape(phase_hist_reshape_err, (k*m),  order='F')

        target_snr_db = 25
        powerimage = phase_error ** 2
        sig_avg_watts = np.mean(powerimage)

        sig_avg_db = 10 * np.log10(sig_avg_watts)
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), (len(powerimage)))
        output = phase_error + noise_volts

        phase_hist_reshape = np.reshape(output, (k,m),  order='F')




        pf_image2 = np.fft.ifft2(phase_hist_reshape)
        pf_image2 = np.flipud(np.swapaxes(pf_image2,0,1))
        pf_image2 = np.fft.fftshift(pf_image2,0)
        pf_image2 = np.abs(pf_image2)



        output = cv2.convertScaleAbs(pf_image2, alpha=(255.0))
        cv2.imwrite(os.path.join(image_path, ("Cor\cor%d.png") % c), output)

