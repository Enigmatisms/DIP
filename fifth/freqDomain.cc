#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <vector>
#include <iostream>

// 距离中心距离dist, d0是滤波器的cut off coeff, n0是有阶滤波器的阶数 
typedef double (*FilterCallBack)(double dist, double d0, int n0);
using Scalar = cv::Scalar_<int>;

std::vector<std::string> names = {
    "test1.pgm", "test2.tif", "test3.pgm", "test4.tif", "test5.tif"
};

std::string opath_prefix = "../data/result/";

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
    cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX);
    disp.convertTo(disp, CV_8UC1);
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
    return 4.0 * 3.14159265 * std::pow(dist / d0, 2);
}

double calculateSpectrumRatio(const cv::Mat& origin, const cv::Mat& filtered) {
    cv::Mat noise = origin - filtered;
    double fil_val = 0.0, noise_val = 0.0;
    for (int i = 0; i < filtered.rows; i++) {
        const double* fptr = filtered.ptr<double>(i);
        const double* nptr = noise.ptr<double>(i);
        for (int j = 0; j < filtered.cols; j++) {
            double fval = *(fptr + j);
            double nval = *(nptr + j);
            fil_val += std::pow(fval, 2);
            noise_val += std::pow(nval, 2);
        }
    }
    printf("Preserved ratio: %lf\n", (fil_val) / (fil_val + noise_val));
    return fil_val / noise_val;     // 功率期望 比 噪声期望（面积归一化项被约掉了）
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
    double ratio = calculateSpectrumRatio(dft_img, dst);
    printf("Spectrum ratio is %lf\n", ratio);
}

void applyFilters(bool low_pass, bool composition = false) {
    std::map<std::string, FilterCallBack> mapping;
    if (low_pass)
        mapping = {
            {"_BLPF_20.jpg", BLPF},
            {"_GLPF_20.jpg", GLPF}
        };
    else 
        mapping = {
            {"_BHPF_20.jpg", BHPF},
            {"_GHPF_20.jpg", GHPF},
        };
    for (size_t i = 0; i < names.size(); i++) {
        int padx = 0, pady = 0;
        cv::Scalar mean;
        Scalar info;
        cv::Mat img = cv::imread("../data/" + names[i], 0), dft_img, dst;
        imageDFT(img, dft_img, mean, info);
        displayDFT(dft_img, info, mean, opath_prefix + "FT_test" + std::to_string(i + 1) + ".jpg", false);
        for (const std::pair<std::string, FilterCallBack>& pr: mapping) {
            std::cout << '\n' << names[i] << ": " << pr.first << " filter: \n";
            applyFreqFilter(dft_img, dst, pr.second, 20.0, 2);
            imageIDFT(dst, dst, mean, info);
            if (composition) {
                dst = img + dst - cv::mean(dst);
                cv::imwrite(opath_prefix + "sharp_test" + std::to_string(i + 1) + pr.first, dst);
            }
            else {
                cv::imwrite(opath_prefix + "test" + std::to_string(i + 1) + pr.first, dst);
            }
        }
    }
} 

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./Task <low_pass?> <composition?>";
    }
    applyFilters(atoi(argv[1]), atoi(argv[2]));
    return 0;
}