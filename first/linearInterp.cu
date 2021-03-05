#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "include/utils.hpp"

// __device__ int ncols = 2048;

// ==================== 此段一直沿用之前做过的有关stixels的代码 ======================
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}
// ==================== ================================== ======================

__global__ void linearInterp(const uchar* const src, uchar* dst, int cols, int k = 4) {
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

int main(){
    cv::Mat img = cv::imread("../data/lena.bmp", 0), padded;
    cv::copyMakeBorder(img, padded, 0, 1, 0, 1, CV_HAL_BORDER_REPLICATE);
    int padded_size = padded.rows * padded.cols * sizeof(uchar),
        result_size = 2048 * 2048 * sizeof(uchar);
    uchar* padded_cu = (uchar*)malloc(padded_size);
    uchar* res_cu = (uchar *)malloc(result_size);
    uchar* res_data = (uchar *)malloc(result_size);
    dim3  grid(2048, 2048);
    printf("Start interpolation...\n");
    uint64_t start_t = getCurrentTime();
    CUDA_CHECK_RETURN(cudaMalloc((void **) &padded_cu, padded_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &res_cu, result_size));
    CUDA_CHECK_RETURN(cudaMemcpy(padded_cu, padded.data, padded_size, cudaMemcpyHostToDevice));

    linearInterp<<<grid, 1>>>(padded_cu, res_cu, padded.cols);

    CUDA_CHECK_RETURN(cudaMemcpy(res_data, res_cu, result_size, cudaMemcpyDeviceToHost));
    uint64_t end_t = getCurrentTime();
    printf("CUDA time elapsed: %lf\n ms", double(end_t - start_t) / 1e6);
    printf("Interpolation completed.\n");
    cv::Mat result(cv::Size(2048, 2048), CV_8UC1, (void *)res_data);
    cv::imwrite("../data/cu_zoomed.bmp", result);
    
    printf("Output completed.\n");
    free(res_data);

    CUDA_CHECK_RETURN(cudaFree(padded_cu));
    CUDA_CHECK_RETURN(cudaFree(res_cu));
    printf("Allocated memory freed.\n");
    return 0;
}
