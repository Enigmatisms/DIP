#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

// 距离中心距离dist, d0是滤波器的cut off coeff, n0是有阶滤波器的阶数 
typedef double (*FilterCallBack)(double dist, double d0, int n0);
using Scalar = cv::Scalar_<int>;

/// @brief 输入8位，输出将会是2通道64位的
void imageDFT(const cv::Mat& src, cv::Mat& dst, cv::Scalar& mean, Scalar& info) {
    double minVal, maxVal;
    info[0] = src.cols;
    info[1] = src.rows;
    cv::minMaxIdx(src, &minVal, &maxVal);
    info[2] = minVal;
    info[3] = maxVal;
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 0, info[1], 0, info[0], cv::BORDER_CONSTANT);
    padded.convertTo(padded, CV_64FC1);
    mean = cv::mean(padded);             // 中心化过程
    padded -= mean;
    cv::dft(padded, dst);
}

void imageIDFT(const cv::Mat& padded, cv::Mat& dst, cv::Scalar mean, Scalar info) {
    cv::idft(padded, padded);
    padded += mean;
    cv::normalize(padded, padded, info[2], info[3], cv::NORM_MINMAX);
    padded.convertTo(dst, CV_8UC1);
    int origin_x = padded.cols - info[0], origin_y = padded.rows - info[1];
    dst = dst(cv::Rect(0, 0, origin_x, origin_y));
}

void displayDFT(const cv::Mat& src, Scalar info, cv::Scalar mean, std::string opath, bool use_disp = true) {
    cv::Mat disp;
    cv::log(1 + cv::abs(src), disp);
    cv::normalize(disp, disp, 0, 1, cv::NORM_MINMAX);
    int row = src.rows - info[1];
    int col = src.cols - info[0];
    cv::Mat crop_up, up_flip;
    cv::Mat crop_down, down_flip;
    disp(cv::Rect(0, 0, col, row)).copyTo(crop_up);
    disp(cv::Rect(0, row, col, row)).copyTo(crop_down);
    cv::flip(crop_up, up_flip, 1);
    cv::flip(crop_down, down_flip, 1);
    crop_up.copyTo(disp(cv::Rect(col, row, col, row)));
    crop_down.copyTo(disp(cv::Rect(col, 0, col, row)));
    up_flip.copyTo(disp(cv::Rect(0, row, col, row)));
    down_flip.copyTo(disp(cv::Rect(0, 0, col, row)));
    if (use_disp) {
        cv::imshow("dft", disp);
        cv::waitKey(0);
    }
    cv::Mat test_dst;
    cv::imwrite(opath, disp);
}

inline double BLPF(double dist, double d0, int n0) {
    return 1.0 / (1.0 + std::pow(dist / d0, 2 * n0));
}

inline double GLPF(double dist, double d0, int n0) {
    double coeff = 2.0 * std::pow(d0, 2);
    return std::exp(- std::pow(dist, 2) / coeff);
}

inline double BHPF(double dist, double d0, int n0) {
    return 1.0 - 1.0 / (1.0 + std::pow(dist / d0, 2 * n0));
}

inline double GHPF(double dist, double d0, int n0) {
    double coeff = 2.0 * std::pow(d0, 2);
    return 1 - std::exp(- std::pow(dist, 2) / coeff);
}

inline double laplacian(double dist, double d0, int n0) {
    return -4.0 * 3.14159265 * std::pow(dist, 2);
}

/// 由于opencv 输出的结果并不是中心化的频谱，所以需要进行一些特殊操作
void applyFreqFilter(const cv::Mat& dft_img, cv::Mat& dst, FilterCallBack filter, double d0, int n0) {
    int half_row = dft_img.rows / 2;
    dft_img.copyTo(dst);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < half_row; i++) {
        for (int j = 0; j < dft_img.cols; j++) {
            double dist = std::sqrt(std::pow(i, 2) + std::pow(j, 2));
            dst.at<double>(i, j) *= filter(dist, d0, n0);
        }
    }
    #pragma omp parallel for num_threads(8)
    for (int i = half_row; i < dst.rows; i++) {
        for (int j = 0; j < dft_img.cols; j++) {
            double dist = std::sqrt(std::pow(i - dst.rows, 2) + std::pow(j, 2));
            dst.at<double>(i, j) *= filter(dist, d0, n0);
        }
    }
}

int main() {
    cv::Mat img = cv::imread("../data/test3.pgm", 0), dft_img, dst;
    int padx = 0, pady = 0;
    cv::Scalar mean;
    Scalar info;
    imageDFT(img, dft_img, mean, info);
    displayDFT(dft_img, info, mean, "../data/dft_output.jpg", true);
    // dft_img(cv::Rect(50, 50, dft_img.cols - 50, dft_img.rows - 50)) = 0;
    imageIDFT(dft_img, dst, mean, info);
    cv::imshow("result", dst);
    cv::imshow("origin", img);
    cv::waitKey(0);
    return 0;
}