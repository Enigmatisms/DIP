#include "include/histOp.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./Task <img_name in folder 'data'>\n";
        return -1;
    }
    std::string prefix = "../data/";
    cv::Mat img = cv::imread(prefix + argv[1] + ".bmp", 0);
    cv::Mat img2 = cv::imread(prefix + argv[2] + ".bmp", 0);
    cv::imshow("origin", img);
    cv::Mat result, local, match;
    localHistEqualize2(img, local, 5);
    histEqualize(img, result);
    histMatching(img, img2, match);
    cv::imshow("disp", result);
    cv::imshow("disp2", local);
    cv::imshow("disp3", match);
    cv::waitKey(0);
    return 0;
}