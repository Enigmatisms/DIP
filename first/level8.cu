#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

__global__ void nLevelDisplay(const uchar* const src, uchar* dst, uchar and_code) {
    int x = blockIdx.x, y = blockIdx.y, col = 512;
    int index = y * col + x;
    if (and_code == 0xff) {
        dst[index] = src[index];
    }
    else {
        dst[index] = src[index] & and_code;
    }
}

uchar getAndCode(int level) {
    return uchar(~((1 << (level - 1)) - 1));
}

int main() {
    cv::Mat src = cv::imread("../data/lena.bmp", 0);
    printf("Image loaded: %d, %d\n", src.cols, src.rows);
    int size = sizeof(uchar) * src.cols * src.rows;
    uchar *src_cu, *dst_cu; 
    CUDA_CHECK_RETURN(cudaMalloc((void **) &src_cu, size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dst_cu, size));
    CUDA_CHECK_RETURN(cudaMemcpy(src_cu, src.data, size, cudaMemcpyHostToDevice));
    dim3 grid(src.rows, src.cols);
    for (int i = 1; i <= 8; i++) {
        uchar code = getAndCode(i);
        nLevelDisplay<<<grid, 1>>>(src_cu, dst_cu, code);
        cv::Mat dst(src.rows, src.cols, CV_8UC1);
        CUDA_CHECK_RETURN(cudaMemcpy(dst.data, dst_cu, size, cudaMemcpyDeviceToHost));
        std::string path = "../data/level_" + std::to_string(i) + ".bmp";
        cv::imwrite(path, dst);
    }
    CUDA_CHECK_RETURN(cudaFree(src_cu));
    CUDA_CHECK_RETURN(cudaFree(dst_cu));

    return 0;
}