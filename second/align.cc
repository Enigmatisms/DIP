/**
 * @brief 2D配准 闭式解算法
 * @author hqy
 * @date 2021.3.15
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

template<typename T>
void printMat(const T& mat, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size - 1; j++){
            std::cout << mat(i, j) << ", ";
        }
        std::cout << mat(i, size - 1) << std::endl;
    }
}

void hMatrixSolver(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& pts1,
    const std::vector<cv::KeyPoint>& pts2,
    Eigen::Matrix3d& H
) {
    int size = matches.size();
    Eigen::Matrix3Xd P(3, size);
    Eigen::Matrix3Xd Q(3, size);
    P.setZero();
    Q.setZero();
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < matches.size(); i++) {
        const cv::Point2f& _p1 = pts1[matches[i].queryIdx].pt;
        const cv::Point2f& _p2 = pts2[matches[i].trainIdx].pt;
        P.block<3, 1>(0, i) = Eigen::Vector3d(_p1.x, _p1.y, 1);
        Q.block<3, 1>(0, i) = Eigen::Vector3d(_p2.x, _p2.y, 1);
    }
    Eigen::Matrix3d PPT = P * P.transpose();
    Eigen::Matrix3d inv = (PPT).ldlt().solve(Eigen::Matrix3d::Identity());    // LDLT分解求逆矩阵 (PP^T)
    Eigen::Matrix3d res = PPT * inv;
    H = Q * P.transpose() * inv;
}

void featureExtract(
    const cv::Mat& src1,
    const cv::Mat& src2,
    cv::Mat& dst,
    Eigen::Matrix3d& H
){
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> pts1, pts2;
    #define POINT_NUM 64
    cv::Mat dscp1, dscp2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cv::Ptr<cv::FeatureDetector> det1 = cv::ORB::create(POINT_NUM);
            cv::Ptr<cv::DescriptorExtractor> des1 = cv::ORB::create(POINT_NUM);
            det1->detectAndCompute(src1, cv::noArray(), pts1, dscp1);
        }
        #pragma omp section
        {
            cv::Ptr<cv::FeatureDetector> det2 =  cv::ORB::create(POINT_NUM);
            cv::Ptr<cv::DescriptorExtractor> des2 =  cv::ORB::create(POINT_NUM);
            det2->detectAndCompute(src2, cv::noArray(), pts2, dscp2);
        }
    }
    std::cout << "Parallel ORB descriptors found.\n";
    cv::Ptr<cv::DescriptorMatcher> match = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT);
    match->match(dscp1, dscp2, matches);
    std::cout << "ORB descriptors matched.\n";
    std::vector<cv::Point2f> crn1, crn2;
    for (const cv::DMatch& mt: matches){
        crn1.push_back(pts1[mt.queryIdx].pt);
        crn2.push_back(pts2[mt.trainIdx].pt);
    }
    cv::Mat mask;
    
    cv::Mat cvH = cv::findHomography(crn1, crn2, mask, cv::RANSAC, 16);
    printf("Homography found. mask is %d, %d, %d, %d\n", mask.cols, mask.rows, mask.type(), CV_8UC1);
    printf("The original match num is %lu.\n", matches.size());
    std::vector<cv::DMatch> truth;
    for (size_t i = 0; i < matches.size(); i++){
        if (mask.at<uchar>(i) == 1){
            truth.push_back(matches[i]);
        }
    }
    cv::drawMatches(src1, pts1, src2, pts2, truth, dst);
    hMatrixSolver(truth, pts1, pts2, H);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%lf, ", cvH.at<double>(i, j));
        }
        printf("\n");
    }
}


/// 可视化
void alignmentVisualize(const cv::Mat& src, cv::Mat&dst, Eigen::Matrix3d H) {
    dst.create(src.rows, src.cols, CV_8UC3);
    #pragma for num_threads(8);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Eigen::Vector3d now(j, i, 1.0);
            Eigen::Vector3d trans = H * now;
            int px = trans[0], py = trans[1];
            if (px >= 0 && px < dst.cols && py >= 0 && py < dst.rows) {
                dst.at<cv::Vec3b>(py, px) = src.at<cv::Vec3b>(i, j);
            } 
        }
    }
}

int main() {
    cv::Mat src1 = cv::imread("../data/ImageA.jpg");
    cv::Mat src2 = cv::imread("../data/ImageB.jpg");
    cv::Mat gray1, gray2;
    cv::cvtColor(src1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(src2, gray2, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray1, gray1);
    cv::equalizeHist(gray2, gray2);
    cv::Mat aligned;
    cv::Mat feats;

    Eigen::Matrix3d H;

    featureExtract(gray1, gray2, feats, H);
    std::cout << "H matrix is:\n";
    printMat<Eigen::Matrix3d>(H, 3);
    cv::imwrite("../data/features.jpg", feats);

    alignmentVisualize(src1, aligned, H);
    cv::imwrite("../data/result.jpg", aligned);
    return 0;
}