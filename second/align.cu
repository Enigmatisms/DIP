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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

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
        double p1x = _p1.x, p1y = _p1.y, p2x = _p2.x, p2y = _p2.y;
        P.block<3, 1>(0, i) = Eigen::Vector3d(p1x, p1y, 1);
        Q.block<3, 1>(0, i) = Eigen::Vector3d(p2x, p2y, 1);
    }
    Eigen::Matrix3d inv = (P * P.transpose()).ldlt().solve(Eigen::Matrix3d::Identity());    // LDLT分解求逆矩阵 (PP^T)
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
    #define POINT_NUM 4096
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
    
    cv::findHomography(crn1, crn2, mask, cv::RANSAC, 16);
    printf("Homography found. mask is %d, %d, %d, %d\n", mask.cols, mask.rows, mask.type(), CV_8UC1);
    printf("The original match num is %lu.\n", matches.size());
    std::vector<cv::DMatch> truth;
    for (size_t i = 0; i < matches.size(); i++){
        if (mask.at<uchar>(i) == 1){
            truth.push_back(matches[i]);
        }
    }
    cv::drawMatches(src1, pts1, src2, pts2, truth, dst);
    hMatrixSolver(matches, pts1, pts2, H);
}


/// 可视化
__global__ void alignmentVisualize(const uchar* const src, uchar* dst, int cols, int rows, const double* const H) {
    int y = blockIdx.y, x = blockIdx.x;
    double now[3] = {x, y, 1.0};
    double trans[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            trans[i] += H[i * 3 + j] * now[j];
        }
    }
    int px = trans[0], py = trans[1];
    if (px >= 0 && px < cols && py >= 0 && py < rows) {
        dst[(px + py * cols) * 3 + 0] = src[3 * (x + y * cols) + 0];
        dst[(px + py * cols) * 3 + 1] = src[3 * (x + y * cols) + 1];
        dst[(px + py * cols) * 3 + 2] = src[3 * (x + y * cols) + 2];
    } 
}

int main() {
    cv::Mat src1 = cv::imread("../data/ImageA.jpg");
    cv::Mat src2 = cv::imread("../data/ImageB.jpg");
    cv::Mat aligned(src1.rows, src1.cols, CV_8UC3);
    cv::Mat feats;

    uchar *cu_align, *cu_img;
    Eigen::Matrix3d H;
    double* dev_H;
    size_t sz = sizeof(uchar) * src1.rows * src1.cols * src1.channels();
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_H, sizeof(double) * 9));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&cu_align, sz));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&cu_img, sz));

    featureExtract(src1, src2, feats, H);
    std::cout << "H matrix is:\n";
    printMat<Eigen::Matrix3d>(H, 3);
    cv::imwrite("../data/features.jpg", feats);

    CUDA_CHECK_RETURN(cudaMemcpy(dev_H, H.data(), sizeof(double) * 9, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_img, src1.data, sz, cudaMemcpyHostToDevice));
    dim3 grid(src1.cols, src1.rows);
    alignmentVisualize<<<grid, 1>>>(cu_img, cu_align, src1.cols, src1.rows, dev_H);

    CUDA_CHECK_RETURN(cudaMemcpy(aligned.data, cu_align, sz, cudaMemcpyDeviceToHost));
    cv::Mat result;
    cv::hconcat(aligned, src2, result);
    cv::imwrite("../data/result.jpg", result);

    CUDA_CHECK_RETURN(cudaFree(dev_H));
    CUDA_CHECK_RETURN(cudaFree(cu_align));
    CUDA_CHECK_RETURN(cudaFree(cu_img));
    return 0;
}