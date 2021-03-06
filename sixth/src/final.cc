#include "../include/final.hpp"

void DeNoise::naiveDeNoise(const cv::Mat& src, cv::Mat& dst, int op) const{
    if (op == MEANF) {
        cv::Mat kernel = getMeanKernel(5);
        planeFiltering(src, kernel, dst);
    }
    else if (op == GEOMTRY) {
        arithmaticFiltering<USE_GEO>(src, dst, 5);
    }
    else if (op == HARMONIC) {
        arithmaticFiltering<USE_HAR>(src, dst, 5);
    }
    else if (op == MEDIANF) {
        cv::medianBlur(src, dst, 5);
    }
    else if (op == MAXF) {
        minMaxFiltering<USE_MAX>(src, dst, 5);
    }
    else if (op == MINF) {
        minMaxFiltering<USE_MIN>(src, dst, 5);
    }
    else if (op == MIDPOINT) {
        minMaxFiltering<USE_MID>(src, dst, 5);
    }
    else if (op == BILATERAL) {
        cv::bilateralFilter(src, dst, 5, 5, 5);
    }
    else {
        arithmaticFiltering<USE_INV>(src, dst, 5, 0.5);
    }   
}

void DeNoise::planeFiltering(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst) const{
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

template <int ARITHM>
void DeNoise::arithmaticFiltering(const cv::Mat& src, cv::Mat& dst, int ksize, int q) const{
    int pad_sz = ksize / 2;
    cv::Mat pad;
    cv::copyMakeBorder(src, pad, pad_sz, pad_sz, pad_sz, pad_sz, cv::BORDER_REFLECT);
    dst.create(src.rows, src.cols, CV_8UC1);
    if (ARITHM == USE_GEO) {
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                double prod = 1, ksz2 = std::pow(ksize, 2);
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x);
                        if (val > 0) 
                            prod *= double(val); 
                        else
                            ksz2 -= 1;
                    }
                }
                dst.at<uchar>(i, j) = uchar(std::pow(prod, 1. / ksz2));
            }
        }
    }
    else if (ARITHM == USE_HAR){
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                double sum = 0, ksz2 = std::pow(ksize, 2);
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x);
                        if (val > 0) 
                            sum += 1 / double(val);
                        else 
                            ksz2 -= 1;
                    }
                }
                dst.at<uchar>(i, j) = uchar(ksz2 / sum);
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                double sum_q_1 = 1e-5, sum_q = 1e-5;
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x);
                        if (val > 0) {
                            sum_q_1 += std::pow(double(val), q + 1); 
                            sum_q += std::pow(double(val), q); 
                        }
                    }
                }
                dst.at<uchar>(i, j) = uchar(sum_q_1 / sum_q);
            }
        }
    }
}

template <int SELECT>
void DeNoise::minMaxFiltering(const cv::Mat& src, cv::Mat& dst, int ksize) const{
    int pad_sz = ksize / 2;
    cv::Mat pad;
    cv::copyMakeBorder(src, pad, pad_sz, pad_sz, pad_sz, pad_sz, cv::BORDER_REFLECT);
    dst.create(src.rows, src.cols, CV_8UC1);
    if (SELECT == USE_MIN) {
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                uchar min_val = 255;
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x); 
                        if (val < min_val)
                            min_val = val;
                    }
                }
                dst.at<uchar>(i, j) = min_val;
            }
        }
    }
    else if (SELECT == USE_MAX){
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                uchar max_val = 0;
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x); 
                        if (val > max_val)
                            max_val = val;
                    }
                }
                dst.at<uchar>(i, j) = max_val;
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int now_row = i + pad_sz, now_col = j + pad_sz;
                uchar max_val = 0, min_val = 255;
                for (int x = -pad_sz; x <= pad_sz; x++) {
                    for (int y = -pad_sz; y <= pad_sz; y++) {
                        uchar val = pad.at<uchar>(now_row + y, now_col + x); 
                        if (val > max_val)
                            max_val = val;
                        if (val < min_val)
                            min_val = val;
                    }
                }
                dst.at<uchar>(i, j) = uchar(double(max_val) * 0.5 + double(min_val) * 0.5);
            }
        }
    }
}

void DeNoise::imgAddNoise(const cv::Mat& src, cv::Mat& dst, bool use_gauss, int mu, int sig) const{
    cv::RNG rng;
    cv::Mat noise(src.rows, src.cols, CV_64FC1);
    if (use_gauss == true) 
        rng.fill(noise, cv::RNG::NORMAL, mu, sig);
    else 
        rng.fill(noise, cv::RNG::UNIFORM, 0, 255);
    if (use_gauss) {
        cv::threshold(noise, noise, 0, 255, cv::THRESH_TOZERO);
        noise.convertTo(noise, CV_8UC1);
        dst = src + noise;
    }
    else {
        cv::Mat salt, peper;
        cv::threshold(noise, salt, 229.5, 255, cv::THRESH_BINARY);
        cv::threshold(noise, peper, 25.5, 255, cv::THRESH_BINARY_INV);
        salt.convertTo(salt, CV_8UC1);
        peper.convertTo(peper, CV_8UC1);
        dst = src + salt - peper;
    }
}

void DeNoise::motionBlur(const cv::Mat& src, cv::Mat& dst, int ksize) const{
    cv::Mat pad;
    cv::copyMakeBorder(src, pad, 0, ksize, 0, ksize, cv::BORDER_REPLICATE);
    dst.create(src.rows, src.cols, CV_8UC1);
    std::vector<int> summa(src.rows * src.cols);
    for (size_t i = 0; i < summa.size(); i++) 
        summa[i] = 0;
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int index = i * src.cols + j;
            for (int k = 0; k < ksize; k++) {
                int row = i + k, col = j + k;
                int idx = row * pad.cols + col;
                summa[index] += pad.data[idx];
            }
        }
        for (int j = 0; j < src.cols; j++) {
            int index = i * src.cols + j;
            dst.data[index] = uchar(summa[index] / ksize);
        }
    }
}