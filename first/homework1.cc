/**
 * @date 2021.3.4 
 * @author hqy - sentinel
 * @note 数字图像处理第一次作业
 */

#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include "include/utils.hpp"

/// 求得图像均值以及方差
std::pair<double, double> getMeanVar(const cv::Mat& src, bool use_buildin = false) {
    if (use_buildin == false){      // 手写实现
        int img_sz = src.cols * src.rows;
        uchar* data = src.data;
        double mean = 0.0, var = 0.0;
        uint64_t start_t = getCurrentTime();
        for (size_t i = 0; i < img_sz; i++){
            mean += double(data[i]);
        }
        mean /= double(img_sz);
        for (size_t i = 0; i < img_sz; i++){
            var += std::pow(double(data[i]) - mean, 2);
        }
        var /= double(img_sz);
        return std::make_pair(mean, var);
    }
    else{           // opencv 调库两行
        cv::Scalar mean, var2;
        cv::meanStdDev(src, mean, var2);
        return std::make_pair(mean[0], std::pow(var2[0], 2));
    }
}

void imgCrop(const uchar* const data, int px, int py, int step, uchar* buf) {
    int cnt = 0;
    for (int i = 0; i < 4; i++){
        int offset = (py + i) * step + px;
        for (int j = 0; j < 4; j++, cnt++){
            buf[cnt] = data[offset + j];
        }
    }
}

template<typename T>
void getWeightVector(T* res, double z, double a = -0.5){
    for (int i = -1, cnt = 0; cnt < 4; i++, cnt++){
        double x = std::abs(i - z);
        if (x <= 1)
            res[cnt] = (a + 2) * std::pow(x, 3) - (a + 3) * std::pow(x, 2) + 1;
        else if (x < 2)
            res[cnt] = a * std::pow(x, 3) - 5 * a * std::pow(x, 2) + 8 * a * x - 4 * a;
    }
}

template<typename T = double>
uchar calcWeightSum(const T* const wx, const T* const wy, const uchar* const buf) {
    int cnt = 0;
    T res = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++, cnt++) {
            res += wx[j] * wy[i] * T(buf[cnt]);
        }
    }
    return uchar(res);
}

void linearInterpZoom(const cv::Mat& src, cv::Mat& dst, int k = 4) {
    cv::Mat padding;
    int nrows = src.rows * k, ncols = src.cols * k;
    dst = cv::Mat::zeros(cv::Size(ncols, nrows), CV_8UC1);
    cv::copyMakeBorder(src, padding, 1, 1, 1, 1, CV_HAL_BORDER_REPLICATE);      // opencv padding操作
    uint64_t start_t = getCurrentTime();
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < nrows; i++) {
        int base = i * ncols;
        for (size_t j = 0; j < ncols; j++) {
            ;
        }
    }
}

void biCubicInterpZoom(const cv::Mat& src, cv::Mat& dst, int k = 4) {
    cv::Mat padding;
    int nrows = src.rows * k, ncols = src.cols * k;
    dst = cv::Mat::zeros(cv::Size(ncols, nrows), CV_8UC1);
    cv::copyMakeBorder(src, padding, 1, 2, 1, 2, CV_HAL_BORDER_REPLICATE);      // opencv padding操作
    uint64_t start_t = getCurrentTime();
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < nrows; i++) {
        int base = i * ncols;
        for (size_t j = 0; j < ncols; j++) {
            int px = j / k, py = i / k;
            double u = double(j) / double(k) - double(px);
            double v = double(i) / double(k) - double(py);
            double wx[4] = {0, 0, 0, 0};
            double wy[4] = {0, 0, 0, 0};
            getWeightVector<double>(wx, u);
            getWeightVector<double>(wy, v);
            uchar crop[16];
            imgCrop(padding.data, px, py, padding.cols, crop);
            dst.data[base + j] = calcWeightSum<double>(wx, wy, crop);
        }
    }
    uint64_t end_t = getCurrentTime();
    printf("Time elapse: %lf ms\n", double(end_t - start_t) / 1e6);
}

void rotate(const cv::Mat& src, cv::Mat& dst, double angle) {
    dst.create(src.rows, src.cols, CV_8UC1);
}

void shear(const cv::Mat& src, cv::Mat& dst, double ratio) {
    dst.create(src.rows, src.cols, CV_8UC1);
}

int main() {
    cv::Mat img = cv::imread("../data/lena.bmp", 0);
    cv::Mat dst;
    biCubicInterpZoom(img, dst);
    cv::imshow("disp", dst);
    cv::waitKey(0);
    cv::imwrite("../data/czoomed.bmp", dst);
    return 0;
}