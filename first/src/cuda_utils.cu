#include "../include/cuda_utils.h"

#define DEG2RAD 57.29578
#define R 25

typedef unsigned char uchar;

__global__ void linearInterp(const uchar* const src, uchar* dst, int cols, int k) {
    int i = blockIdx.y, j = blockIdx.x;
    int px = j / k, py = i / k;
    double u = double(j) / double(k) - double(px), v = double(i) / double(k) - double(py);
    int basey = py * cols;
    double res = src[basey + px] * (1 - u) * (1 - v);
    res += src[basey + px + 1] * u * (1 - v);
    res += src[basey + cols + px] * (1 - u) * v;
    res += src[basey + cols + px + 1] * u * v;
    dst[2048 * i + j] = uchar(res);
}

__global__ void imgRotate(const uchar* const src, uchar* dst, int rows, int cols, double angle) {
    angle /= DEG2RAD;
    int i = blockIdx.y, j = blockIdx.x;
    int cx = cols / 2, cy = rows / 2;
    double vx = j - cx, vy = i - cy;
    double cosa = cos(angle), sina = sin(angle);
    double nvx = cosa * vx - sina * vy, nvy = sina * vx + cosa * vy;
    int projx = cx + nvx, projy = cy + nvy;
    if (projx >= 0 && projx < cols && projy >= 0 && projy < rows){
        dst[projy * cols + projx] = src[i * cols + j];
    }
}

__global__ void copyMakeBorder(const uchar* const src, uchar* dst, int cols, int pad_cols, int t, int l) {
    int i = blockIdx.y, j = blockIdx.x;
    dst[(i + t) * pad_cols + j + l] = src[i * cols + j];
}

__global__ void imgShear(const uchar* const src, uchar* dst, int rows, int cols, double ratio) {
    int i = blockIdx.y, j = blockIdx.x;
    int cy = rows / 2;
    int dx = double(cy - i) * ratio;
    int base = i * cols, newx = j + dx;
    if (newx >= 0 && newx < cols) {
        dst[base + newx] = src[base + j];
    }
}

__global__ void nearestInterp(const unsigned char* const src, unsigned char* dst, int cols, int k) {
    int i = blockIdx.y, j = blockIdx.x;
    int px = j / k, py = i / k;
    double u = double(j) / double(k) - double(px), v = double(i) / double(k) - double(py);
    if (u < 0.5 && v < 0.5) {
        dst[i * cols * k + j] = src[py * (cols + 1) + px];
    }
    else if (u >= 0.5 && v < 0.5) {
        dst[i * cols * k + j] = src[py * (cols + 1) + px + 1];
    }
    else if (v >= 0.5 && u < 0.5) {
        dst[i * cols * k + j] = src[(py + 1) * (cols + 1) + px];
    }
    else {
        dst[i * cols * k + j] = src[(py + 1) * (cols + 1) + px + 1];
    }
}

__global__ void medianFilter(const uchar* const src, uchar* dst, int rows, int cols, int radius) {
    int y = blockIdx.y, x = blockIdx.x, cnt = 0;
    uchar* value = &dst[y * cols + x];
    if (*value != 0) return;
    int full_length = 2 * radius + 1;
    uchar *buf = (uchar *)malloc(full_length * full_length), med = 0;       // 这个malloc用得就很难受
    for (int i = -radius; i <= radius; i++) {
        int py = y + i;
        if (py >= 0 && py < rows) {
            py *= cols;
            for (int j = -radius; j <= radius; j++) {
                int px = x + j;
                if (px >= 0 && px < cols) {
                    buf[cnt] = src[py + px];
                    cnt++;
                }
            }
        }
    }
    halfBubbleSort<uchar>(buf, &med, cnt);
    *value = med;
    free(buf);
}

template<typename T>
__device__ void halfBubbleSort(T* buf, T* med, int size) {
    int half = int(size / 2) + 1, i = 0;
    for (; i < half; i++) {
        for (int j = i + 1; j < size; j++){
            if (buf[i] > buf[j]) {
                T tmp = buf[i];
                buf[i] = buf[j];
                buf[j] = tmp;
            }
        }
    }
    *med = buf[i - 1];
}
