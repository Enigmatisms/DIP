#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "include/utils.hpp"
#include "include/cuda_utils.h"

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

int main(int argc, char* argv[]){
    std::string path = std::string("../data/");
    std::string name;
    if (argc < 2){
        std::cerr << "Too few arguments. Usage: ./Task <img index>\n";
        return -1;
    }
    if (atoi(argv[1]) == 0) {
        name = "elain1";
    }
    else {
        name = "lena";
    }
    cv::Mat img = cv::imread(path + name + ".bmp", 0);
    cv::Mat trans_img(img.rows, img.cols, CV_8UC1);
    cv::Mat result(2048, 2048, CV_8UC1);
    int pad_row = img.rows + 1, pad_col = img.cols + 1,
        origin_size = img.rows * img.cols * sizeof(uchar),
        padded_size = pad_row * pad_col * sizeof(uchar),
        result_size = 2048 * 2048 * sizeof(uchar);

    uchar* origin_cu = (uchar *)malloc(origin_size);
    uchar* trans_cu = (uchar *)malloc(origin_size);
    uchar* filter_cu = (uchar *)malloc(origin_size);
    uchar* padded_cu = (uchar*)malloc(padded_size);
    uchar* res_cu = (uchar *)malloc(result_size);

    dim3  transform(512, 512);
    dim3  interp(2048, 2048);
    printf("Start interpolation...\n");
    uint64_t start_t = getCurrentTime();
    CUDA_CHECK_RETURN(cudaMalloc((void **) &origin_cu, origin_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &trans_cu, origin_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &filter_cu, origin_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &padded_cu, padded_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &res_cu, result_size));
    CUDA_CHECK_RETURN(cudaMemcpy(origin_cu, img.data, origin_size, cudaMemcpyHostToDevice));

    imgShear<<<transform, 1>>>(origin_cu, trans_cu, img.rows, img.cols, 0.333);
    medianFilter<<<transform, 1>>>(trans_cu, filter_cu, img.rows, img.cols);
    copyMakeBorder<<<transform, 1>>>(filter_cu, padded_cu, img.cols, pad_col);
    nearestInterp<<<interp, 1>>>(padded_cu, res_cu, img.cols);

    CUDA_CHECK_RETURN(cudaMemcpy(trans_img.data, filter_cu, origin_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(result.data, res_cu, result_size, cudaMemcpyDeviceToHost));
    uint64_t end_t = getCurrentTime();
    printf("CUDA time elapsed: %lf\n ms", double(end_t - start_t) / 1e6);
    printf("Interpolation completed.\n");
    biCubicInterpZoom(trans_img, result);
    cv::imwrite(path + "transed.bmp", trans_img);
    cv::imwrite(path + name + "_bicubic_shear.bmp", result);
    
    printf("Output completed.\n");

    CUDA_CHECK_RETURN(cudaFree(padded_cu));
    CUDA_CHECK_RETURN(cudaFree(res_cu));
    CUDA_CHECK_RETURN(cudaFree(origin_cu));
    CUDA_CHECK_RETURN(cudaFree(filter_cu));
    CUDA_CHECK_RETURN(cudaFree(trans_cu));
    printf("Allocated memory freed.\n");
    return 0;
}
