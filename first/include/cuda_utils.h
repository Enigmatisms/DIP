#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/// @brief 求中值，从简，使用bubble sort
template<typename T>
__device__ void halfBubbleSort(T* buf, T* med, int size);

/// @brief 线性插值
__global__ void linearInterp(const unsigned char* const src, unsigned char* dst, int cols, int k = 4);

/// @brief padding 操作
/// @param pad_cols padding之后的列数
/// @param t 顶部padding
/// @param l 左部padding
__global__ void copyMakeBorder(const unsigned char* const src, unsigned char* dst, int cols, int pad_cols, int t = 0, int l = 0);

/// @brief 绕图像中心逆时针旋转 angle (单位为度)
__global__ void imgRotate(const unsigned char* const src, unsigned char* dst, int rows, int cols, double angle);

/// @brief 图像斜切变换
__global__ void imgShear(const unsigned char* const src, unsigned char* dst, int rows, int cols, double ratio = 0.333);

/// @brief 中值滤波（图像旋转 / 斜切会造成空缺，需要填补）
__global__ void medianFilter(const unsigned char* const src, unsigned char* dst, int rows, int cols, int radius = 2);

/// @brief 最临近插值
__global__ void nearestInterp(const unsigned char* const src, unsigned char* dst, int cols, int k = 4);

#endif //__CUDA_UTILS_H