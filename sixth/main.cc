/**
1.在测试图像上产生高斯噪声lena图-需能指定均值和方差；并用多种滤波器恢复图像，分析各自优缺点；
2.在测试图像lena图加入椒盐噪声（椒和盐噪声密度均是0.1）；用学过的滤波器恢复图像；在使用反谐波分析Q大于0和小于0的作用；

3.推导维纳滤波器并实现下边要求；
(a) 实现模糊滤波器如方程Eq. (5.6-11).
(b) 模糊lena图像：45度方向，T=1；
(c) 再模糊的lena图像中增加高斯噪声，均值= 0 ，方差=10 pixels 以产生模糊图像；
(d)分别利用方程 Eq. (5.8-6)和(5.9-4)，恢复图像；并分析算法的优缺点.
*/

#include "include/final.hpp"

// #define MEANF       1       // 算术均值滤波 
// #define GEOMTRY     2       // 几何均值滤波 (*)
// #define HARMONIC    4       // 谐波均值滤波 (*)
// #define MEDIANF     8       // 中值滤波
// #define MAXF        16       // 最大滤波 (*)
// #define MINF        32      // 最小滤波 (*)
// #define MIDPOINT    64      // 中点滤波 (*)
// #define BILATERAL   128      // 双边滤波
// #define INVERSE     256     // 逆谐波均值滤波 (*)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./Task <Kernel size>\n";
        return -1;
    }
    int ksz = atoi(argv[1]);
    DeNoise dns;
    cv::Mat img = cv::imread("../data/lena.bmp", 0), noised, res;

    std::vector<std::string> names = {
        "mean_", "geometry_", "harmonic_", "median_", "max_", "min_", "mid_", "bilateral_", "inv_"
    };
    std::vector<std::string> noise_names = {
        "saltpeper_", "guassian_"
    };
    for (size_t i = 0; i < noise_names.size(); i++) {
        dns.imgAddNoise(img, noised, i, 10, 30);
        cv::imwrite("../data/result/" + noise_names[i] + "noise.jpg", noised);
        int filter = 1;
        for (const std::string& name: names) {
            dns.naiveDeNoise(noised, res, filter);
            filter <<= 1;       // 左移一位
            std::string opath = "../data/result/" + noise_names[i] + name + std::to_string(ksz) + ".jpg";
            std::cout << "Image exported to'" << opath << "'\n";
            cv::imwrite(opath, res);
        }
    }
    return 0;
}