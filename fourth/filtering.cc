#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

#define PI 3.14159265

std::string opath = "../data/result/";
std::string path = "../data/";

void prob1() {
    std::vector<std::string> names = {
        "test1.pgm", "test2.tif"
    };
    std::vector<std::string> oname = {
        "test1", "test2"
    };
    std::vector<std::string> op_names = {
        "_gaussian_", "_median_", "_bilateral_"
    };
    for (size_t k = 0; k < names.size(); k++) {
        cv::Mat src = cv::imread(path + names[k], 0);
        cv::imshow("disp", src);
        cv::waitKey(0);
        for (size_t i = 0; i < op_names.size(); i++) {
            cv::Mat output;
            if (i == 0) {
                for (int j = 3; j < 9; j+=2) {
                    cv::medianBlur(src, output, j);
                    cv::imwrite(opath + oname[k] + op_names[i] + std::to_string(j) + ".jpg", output);
                }
            }
            else if (i == 1) {
                for (int j = 3; j < 9; j+=2) {
                    cv::GaussianBlur(src, output, cv::Size(j, j), 1.0, 1.0);
                    cv::imwrite(opath + oname[k] + op_names[i] + std::to_string(j) + ".jpg", output);
                }
            }
            else {
                for (int j = 3; j < 9; j+=2) {
                    cv::bilateralFilter(src, output, j, 9, 9);
                    cv::imwrite(opath + oname[k] + op_names[i] + std::to_string(j) + ".jpg", output);
                }
            }
        }
    }
}

cv::Mat getGaussianKernel(double sigma, int k_size) {
    assert(k_size & 1);
    int center = k_size / 2;
    cv::Mat dst(k_size, k_size, CV_64FC1);
    double coeff = 1.0 / 2.0 / std::pow(sigma, 2);
    // double mini = coeff / PI  * std::exp( - std::pow(center, 2) * coeff);
    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < k_size; i++) {
        for (int j = 0; j < k_size; j++) {
            double dist = (i - center) * (i - center) + (j - center) * (j - center);
            dst.at<double>(i, j) = coeff / PI  * std::exp( - dist * coeff);
        }
    }
    return dst;
}

void planeFiltering(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst) {
    int pad_sz = kernel.rows / 2;
    cv::Mat pad;
    cv::copyMakeBorder(src, pad, pad_sz, pad_sz, pad_sz, pad_sz, cv::BORDER_REFLECT);
    dst.create(src.rows, src.cols, CV_8UC1);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int now_row = i + pad_sz, now_col = j + pad_sz;
            double sum = 0;
            for (int x = -pad_sz; x <= pad_sz; x++) {
                for (int y = -pad_sz; y <= pad_sz; y++) {
                    sum += double(pad.at<uchar>(now_row + y, now_col + x)) * kernel.at<double>(pad_sz + y, pad_sz + x); 
                }
            }
            dst.at<uchar>(i, j) = uchar(sum);
        }
    }
}

void gaussianFiltering(const cv::Mat& src, cv::Mat& dst, double sig, int k_size) {
    cv::Mat kernel = getGaussianKernel(sig, k_size);
    planeFiltering(src, kernel, dst);
}

enum Ops {
    CANNY = 0,
    SOBEL_X = 1,
    SOBEL_Y = 2,
    LAPLACE = 3,
    UNSHARP = 4
};

void filterOps(cv::Mat& src, cv::Mat& dst, int ops) {
    if (ops == CANNY) {
        cv::Canny(src, dst, 4, 4);
        return;
    }
    cv::Mat kernel;
    if (ops == UNSHARP) {
        kernel = (cv::Mat_<double>(3, 3) << 
            0, -1, 0, 
            -1, 5, -1, 
            0, -1, 0
        );
        cv::Mat copy;
        cv::bilateralFilter(src, copy, 7, 12, 9);
        copy.copyTo(src);
    }
    else if (ops == SOBEL_X) {
        kernel = (cv::Mat_<double>(3, 3) << 
            -1, 0, 1, 
            -2, 0, 2, 
            -1, 0, 1
        );
    }
    else if (ops == SOBEL_Y) {
        kernel = (cv::Mat_<double>(3, 3) << 
            1, 2, 1, 
            0, 0, 0, 
            -1, -2, -1
        );
    }
    else {
        kernel = (cv::Mat_<double>(3, 3) << 
            0, -1, 0, 
            -1, 4, -1, 
            0, -1, 0
        );
    }
    planeFiltering(src, kernel, dst);
}

void prob2() {
    std::vector<std::string> names = {
        "test3.pgm", "test4.tif"
    };
    std::vector<std::string> oname = {
        "test3", "test4"
    };
    std::vector<std::string> op_names = {
        "_canny_", "_sobelx_", "_sobely_", "_laplace_", "_unsharp_"
    };
    for (size_t k = 0; k < names.size(); k++) {
        cv::Mat src = cv::imread(path + names[k], 0);
        for (size_t i = 0; i < op_names.size(); i++) {
            cv::Mat output;
            filterOps(src, output, i);
            cv::imwrite(opath + oname[k] + op_names[i] + ".jpg", output);
        }
    }
}

int main() {
    // prob1();
    // cv::Mat src = cv::imread(path + "test2.tif", 0), dst;
    // for (int i = 3; i < 9; i+= 2) {
    //     gaussianFiltering(src, dst, 1.5, i);
    //     cv::imshow("disp", dst);
    //     cv::imwrite(opath + "test2_hand_" + std::to_string(i) + ".jpg", dst);
    // }
    prob2();

    return 0;
}