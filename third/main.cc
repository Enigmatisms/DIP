#include "include/histOp.hpp"
#include <iostream>
#include <unistd.h>

std::string prefix = "../data/";
std::string oprefix = "../output/";

void ques1() {
    std::vector<std::string> names = {
        "lena"
    };
    for (std::string name : names) {
        for (int i = 4; i < 5; i++) {
            std::string path = prefix + name;
            if (i) {
                path += std::to_string(i);
            }
            path += ".bmp";
            if (access(path.c_str(), F_OK) != 0) continue;
            cv::Mat img = cv::imread(path, 0), res;
            // localHistEqualize2(img, res, 3);
            histEqualize(img, res);
            // histMatching(img, temp, res);
            // cv::threshold(img, res, 1, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
            std::string opath = oprefix + name + "_hist_" + std::to_string(i) + ".bmp";
            cv::imwrite(opath, res);
        }
    }
}

int main(int argc, char* argv[]) {
    ques1();
    return 0;
}