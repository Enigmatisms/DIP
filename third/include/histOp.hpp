#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <unordered_map>

void histEqualize(const cv::Mat& src, cv::Mat& dst) {
    int cnt[256];                       // 直方图
    memset(cnt, 0, 256 * sizeof(int));
    int total = src.cols * src.rows;
    double inv_all = 1.0 / double(total);
    for (int i = 0; i < total; i += src.cols) {
        for (int j = 0; j < src.cols; j++) {
            uchar res = src.data[i + j];
            cnt[res] ++;
        }
    }
    for (int i = 1; i < 256; i++) {     // 变为prefix sum
        cnt[i] += cnt[i - 1];
    }
    dst.create(cv::Size(src.cols, src.rows), CV_8UC1);
    
    for (int i = 0; i < total; i += src.cols) {
        #pragma omp parallel for num_threads(8)
        for (int j = 0; j < src.cols; j++) {
            uchar now = src.data[i + j];
            dst.data[i + j] = uchar(inv_all * double(cnt[now]) * 255);
        }
    }
}

void localHistEqualize(const cv::Mat& src, cv::Mat& dst, int sz) {
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, sz, sz, sz, sz, cv::BORDER_REFLECT);
    assert(sz > 0 && sz < 13);
    int total = std::pow(2 * sz + 1, 2);
    double inv_all = 1.0 / double(total);
    dst.create(src.rows, src.cols, CV_8UC1);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < src.rows; i++) {
        int pad_pos = (i + sz) * padded.cols;
        int dst_pos = i * dst.cols;
        for (int j = 0; j < src.cols; j++) {
            int pad_cur = pad_pos + j + sz;
            uchar med_val = padded.data[pad_cur];
            std::unordered_map<uchar, int> contain;
            for (int y = -sz; y <= sz; y++) {
                for (int x = -sz; x <= sz; x++) {
                    uchar now = padded.data[pad_cur + y * padded.cols + x];
                    std::unordered_map<uchar, int>::iterator it = contain.find(now);
                    if (it == contain.end()) {
                        contain[now] = 1;
                    }
                    else {
                        it->second ++;
                    }
                }
            }
            int med_cnt = 0;
            for (std::unordered_map<uchar, int>::const_iterator cit = contain.cbegin(); cit != contain.cend(); cit++) {
                if (cit->first <= med_val) {
                    med_cnt += cit->second;
                }
            }
            dst.data[dst_pos + j] = uchar(inv_all * double(med_cnt) * 255.0);
        }
    }
}

void localHistEqualize2(const cv::Mat& src, cv::Mat& dst, int sz) {
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, sz, sz, sz, sz, cv::BORDER_REFLECT);
    assert(sz > 0 && sz < 13);
    int total = std::pow(2 * sz + 1, 2);
    double inv_all = 1.0 / double(total);
    dst.create(src.rows, src.cols, CV_8UC1);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < src.rows; i++) {
        int pad_pos = (i + sz) * padded.cols;
        int dst_pos = i * dst.cols;
        for (int j = 0; j < src.cols; j++) {
            int pad_cur = pad_pos + j + sz;
            uchar med_val = padded.data[pad_cur];
            uchar cnt[256];                       // 直方图
            memset(cnt, 0, 256 * sizeof(uchar));
            for (int y = -sz; y <= sz; y++) {
                for (int x = -sz; x <= sz; x++) {
                    uchar now = padded.data[pad_cur + y * padded.cols + x];
                    cnt[now] ++;
                }
            }
            int med_cnt = 0;
            for (uchar k = 0; k < med_val; k++) {
                med_cnt += cnt[k];
            }
            dst.data[dst_pos + j] = uchar(inv_all * double(med_cnt) * 255.0);
        }
    }
}

void mapTable(const cv::Mat& src, uchar* table) {
    double cnt[256];
    memset(cnt, 0, 256 * sizeof(double));
    int total = src.cols * src.rows;
    double inv_all = 1.0 / double(total);
    for (int i = 0; i < total; i += src.cols) {
        for (int j = 0; j < src.cols; j++) {
            uchar res = src.data[i + j];
            cnt[res] ++;
        }
    }
    for (int i = 1; i < 256; i++) {
        cnt[i] += cnt[i - 1];
        table[i - 1] = uchar(inv_all * double(cnt[i - 1]) * 255);
    }
    table[255] = 255;
}

void histMatching(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst) {
    uchar map_rs[256];
    uchar map_gs[256];
    mapTable(src1, map_rs);
    mapTable(src2, map_gs);
    uchar inv_map[256];             // s --> g 逆映射
    memset(inv_map, 0, 256 * sizeof(uchar));
    for (int i = 0, cnt = 0; i < 256; i++) {
        while (cnt < map_gs[i]) {
            inv_map[cnt] = i;
            cnt++;
        }
        if (cnt >= 255) {
            break;
        }
    }
    dst.create(src1.rows, src1.cols, CV_8UC1);
    int total = src1.rows * src1.cols;
    for (int i = 0; i < total; i += dst.cols) {
        #pragma omp parallel for num_threads(8)
        for (int j = 0; j < dst.cols; j++) {
            uchar r = src1.data[i + j];
            uchar s = map_rs[r];
            dst.data[i + j] = inv_map[s];
        }
    }
}