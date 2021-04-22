#ifndef __FINAL_HPP
#define __FINAL_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

/**
 * ============ 普通滤波 ===========
 * 算术均值滤波 
 * 几何均值滤波 (*)
 * 谐波均值滤波 (*)
 * 中值滤波
 * 最大滤波 (*)
 * 最小滤波 (*)
 * 中点滤波 (*)
 * 双边滤波
 * 逆谐波均值滤波 (*)
 * =========== 维纳 =============
 * 可能稍微麻烦一点
 */

#define MEANF       1       // 算术均值滤波 
#define GEOMTRY     2       // 几何均值滤波 (*)
#define HARMONIC    4       // 谐波均值滤波 (*)
#define MEDIANF     8       // 中值滤波
#define MAXF        16       // 最大滤波 (*)
#define MINF        32      // 最小滤波 (*)
#define MIDPOINT    64      // 中点滤波 (*)
#define BILATERAL   128      // 双边滤波
#define INVERSE     256     // 逆谐波均值滤波 (*)

enum FilterOps {
    USE_MIN = 0,
    USE_MAX = 1,
    USE_MID = 2,
    USE_GEO = 3,
    USE_HAR = 4,
    USE_INV = 5
};

class DeNoise {
public:
    DeNoise() {;}
    ~DeNoise() {;}
public:
    void naiveDeNoise(const cv::Mat& src, cv::Mat& dst, int op);
    void planeFiltering(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst);
    void imgAddNoise(const cv::Mat& src, cv::Mat& dst, bool use_gauss, int mu = 10, int sig = 20);

    template <int ARITHM>
    void arithmaticFiltering(const cv::Mat& src, cv::Mat& dst, int ksize, int q = 1);

    template <int SELECT>
    void minMaxFiltering(const cv::Mat& src, cv::Mat& dst, int ksize);
private:
    static cv::Mat getMeanKernel(int size) {
        return cv::Mat::ones(size, size, CV_64FC1) / double(size) / double(size);
    }
};


#endif  //__FINAL_HPP