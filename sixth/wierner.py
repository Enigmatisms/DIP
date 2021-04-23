#-*-coding: utf-8 -*-
"""
C++ 矩阵运算比较麻烦,特别是涉及到复数运算时
所以维纳滤波器部分就用python实现了
@author 何千越
@date 2021.4.23
"""

import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable as Var
from numpy import fft

def FourierTransform(img, show = True):
    fft_res = fft.fft2(img)
    if show == True:
        rl = fft_res.real
        im = fft_res.imag
        amp = np.sqrt(rl ** 2 + im ** 2)
        amp = np.log1p(amp)
        amp_min = np.min(amp)
        amp_max = np.max(amp)
        amp = (amp - amp_min) / (amp_max - amp_min)
        amp_show = np.zeros_like(amp)
        halfh = amp.shape[0] // 2
        halfw = amp.shape[1] // 2
        amp_show[:halfh, :halfw] = amp[halfh:, halfw:]
        amp_show[halfh:, :halfw] = amp[:halfh, halfw:]
        amp_show[:halfh, halfw:] = amp[halfh:, :halfw]
        amp_show[halfh:, halfw:] = amp[:halfh, :halfw]
        phase = np.arctan(im / rl)
        plt.subplot(1, 2, 1)
        plt.imshow(amp_show)
        plt.subplot(1, 2, 2)
        plt.imshow(phase)
        plt.show()
    return fft_res

def motionBlur2(img, T = 15, show = False):
    fft_res = FourierTransform(img, False)
    # fft_res = fft.fftshift(fft_res)
    H = getH(*img.shape, T)
    f = H * fft_res
    res = np.abs(fft.ifft2(f))
    if show:
        # res = np.abs(fft.fftshift(res))
        plt.imshow(res)
        plt.colorbar()
        plt.show()
    return res, H

def wholeNoise(img, T, show = False):
    motion_blur, H = motionBlur2(img)
    noise = np.random.normal(0, 10, motion_blur.shape)
    noise_amp = np.sum((noise ** 2) / (img.shape[0] * img.shape[1]))
    motion_blur += noise
    res = np.clip(motion_blur, 0, 255)
    return res, H, noise_amp

"""
    书上的积分表达无法得到正确结果
"""
def degradeFunction(u, v, a, b, T):
    val = (a * u + b * v)
    print(val)
    res = T / val * np.sin(val) * np.exp(-1j * val)
    res[val <= 1e-5] = T
    return res

def discreteDegrade(u, v, T):
    es = np.exp(1j * (u + v))
    h = np.zeros_like(u, dtype = np.complex128)
    for i in range(T):
        h += np.power(es, i)
    return 1. / T * h


def getH(height, width, T):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = 2 * np.pi * xx.astype(float) / float(width)
    yy = 2 * np.pi * yy.astype(float) / float(height)
    return discreteDegrade(xx, yy, T)

def wienerFilter(img, T):
    output = img.copy()
    noised, H = wholeNoise(output, T)
    cv.imwrite("./data/result/motion_blur_freq.jpg", noised)
    G = FourierTransform(noised, False)
    H_amp = np.abs(H) ** 2
    R = H_amp / (H_amp + 100) / H
    F = R * G
    res = np.abs(fft.ifft2(F))
    plt.subplot(1, 2, 1)
    plt.imshow(noised)
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.show()

"""
    思想：时域的均方误差 等价于 频域的均方误差，故最小二乘可以在频域做
    Torch 实现
"""
def constrainedMSEFiltering(img, T):
    output = img.copy()
    noised, H, noise_map = wholeNoise(output, T)
    G = FourierTransform(noised, False)
    H_amp = np.abs(H) ** 2
    H_inv = H.conj()

    G_ = torch.from_numpy(G)
    H_ = torch.from_numpy(H)
    Hamp = torch.from_numpy(H_amp)
    Hinv = torch.from_numpy(H_inv)
    L2 = torch.from_numpy(getLaplacianAmp(*img.shape))
    noise = torch.DoubleTensor([noise_map])

    r = Var(torch.DoubleTensor([0.1, ]), requires_grad = True)
    optimizer = optim.Adam([r, ], lr = 1e-2)
    h, w = img.shape

    epochs = 1000
    for i in range(100):
        F_hat = Hinv / (Hamp + r * L2) * G_
        lMat = G_ - H_ * F_hat
        diff = torch.sum(torch.abs(lMat) ** 2) / h / w
        loss = (noise - diff) ** 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Training epoch: %d / %d \tloss: "%(i, epochs), loss.detach().item())
    res = F_hat.detach().numpy()
    res = np.abs(fft.ifft2(res))
    cv.imwrite("./data/result/optimized.jpg", res)

def getLaplacianAmp(height, width):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    lap = xx ** 2 + yy ** 2
    return lap / np.max(lap)

if __name__ == "__main__":
    img = cv.imread("./data/lena.bmp", 0)
    # print(FourierTransform(np.array([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]), False))
    # wienerFilter(img, 15)
    constrainedMSEFiltering(img, 15)