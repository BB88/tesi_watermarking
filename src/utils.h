//
// Created by miky on 16/08/15.
//

#ifndef TESI_WATERMARKING_UTILS_H
#define TESI_WATERMARKING_UTILS_H

#endif //TESI_WATERMARKING_UTILS_H

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "../src/dataset/tsukuba_dataset.h"
#include "../src/dataset/dataset.hpp"


using namespace cv;
namespace stereo_watermarking{
    void show_difference(cv::Mat img1,cv::Mat img2,std::string window);
    cv::Mat equalizeIntensity(const cv::Mat& inputImage);
    void sobel_filtering(cv::Mat src,const char* window_name);

    void histo (cv::Mat image, std::string window_name);
    void printRGB (cv::Mat image, int x, int y);
    int equi_histo(cv::Mat image, std::string window_name, cv::Mat &equi_image);
    void dft_magnitude(cv::Mat img,std::string window_name);


//    void viewPointCloudRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr, std::string title);
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizerRGB (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, std::string title);
//    void createPointCloudOpenCV (cv::Mat& img1, cv::Mat& img2,  cv::Mat& Q, cv::Mat& disp, cv::Mat& recons3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &point_cloud_ptr);
//    cv::datasets::FramePair rectifyImages(Mat& img1, Mat& img2, Mat& M1, Mat& D1, Mat& M2, Mat& D2, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Rect &roi1, Rect &roi2, float scale);
//    void generatePointCloud(cv::Mat disp, cv::Mat img_left,cv::Mat img_right, int frame_num);

    void writeMatToFile(cv::Mat& m, std::string filename);

    }