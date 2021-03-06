#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "./include/cuda_utils.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

int main() {
    cv::Mat img = cv::imread("../data/lena.bmp", 0);
    cv::Mat trans_img(img.rows, img.cols, CV_8UC1);
    cv::Mat result(2048, 2048, CV_8UC1);
    int pad_row = img.rows + 1, pad_col = img.cols + 1,
        origin_size = img.rows * img.cols * sizeof(uchar),
        padded_size = pad_row * pad_col * sizeof(uchar),
        result_size = 2048 * 2048 * sizeof(uchar);
    uchar* origin_cu = (uchar *)malloc(origin_size);
    uchar* padded_cu = (uchar*)malloc(padded_size);
    uchar* res_cu = (uchar *)malloc(result_size);

    dim3  transform(512, 512);
    dim3  interp(2048, 2048);
    CUDA_CHECK_RETURN(cudaMalloc((void **) &origin_cu, origin_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &padded_cu, padded_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &res_cu, result_size));
    CUDA_CHECK_RETURN(cudaMemcpy(origin_cu, img.data, origin_size, cudaMemcpyHostToDevice));

    copyMakeBorder<<<transform, 1>>>(origin_cu, padded_cu, img.cols, pad_col);
    nearestInterp<<<interp, 1>>>(padded_cu, res_cu, img.cols);
    CUDA_CHECK_RETURN(cudaMemcpy(result.data, res_cu, result_size, cudaMemcpyDeviceToHost));
    
    cv::imwrite("../data/cu_nearest.bmp", result);

    CUDA_CHECK_RETURN(cudaFree(res_cu));
    CUDA_CHECK_RETURN(cudaFree(padded_cu));
    CUDA_CHECK_RETURN(cudaFree(origin_cu));

    return 0;
}