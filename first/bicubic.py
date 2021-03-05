#-*-coding:utf-8-*-

import cv2 as cv
import numpy as np
import profile
import time

def getWeight(z, a = -0.5):
    res = []
    for i in range(-1, 3):
        x = abs(i - z)
        if x <= 1:
            res.append((a + 2) * x ** 3 - (a + 3) * x ** 2 + 1)
        elif x < 2:
            res.append(a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a)
        else:
            res.append(0.0)
    return np.array(res)

def biCubicSlow(img:np.array, ratio = 4):
    rows, cols = img.shape
    pad = np.zeros((rows + 3, cols + 3))
    pad[1:1 + rows, 1:1 + cols] = img
    new_rows = int(ratio * rows)
    new_cols = int(ratio * cols)
    result = np.zeros((new_rows, new_cols), dtype = np.float64)
    for i in range(new_rows):
        for j in range(new_cols):
            px = int(j / ratio)
            py = int(i / ratio)
            u = j / ratio - px
            v = i / ratio - py
            px += 1
            py += 1
            wx = getWeight(u)
            wy = getWeight(v)
            W = wy.reshape(-1, 1) @ wx.reshape(1, -1)
            crop = pad[py - 1:py + 3, px - 1:px + 3]
            result[i, j] = np.sum(W * crop)
    return result.astype(np.uint8)

if __name__ == "__main__":
    img = cv.imread('data/lena.bmp', 0)
    res = biCubicSlow(img)

    cv.imwrite('data/zoomed.bmp', res)
