//
// Created by bene on 16/08/15.
//

#ifndef TESI_WATERMARKING_UTILS_H
#define TESI_WATERMARKING_UTILS_H

#endif //TESI_WATERMARKING_UTILS_H

//#include <pcl/common/common_headers.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include "../src/dataset/tsukuba_dataset.h"
#include "../src/dataset/dataset.hpp"


using namespace cv;
namespace stereo_watermarking{
    void show_difference(cv::Mat img1,cv::Mat img2,std::string window);
    void sobel_filtering(cv::Mat src,const char* window_name);

    void histo (cv::Mat image, std::string window_name);
    void printRGB (cv::Mat image, int x, int y);
    float MSE (int width,int height,double** A,double ** B);
    void dft_comparison(unsigned char* Image1, unsigned char* Image2, int dim ,  std::string img1_name, std::string img2_name );
    void show_double_mat(int width,int height,double** A,std::string window_name);
    void histo_equalizer(Mat img, std::string window_name);
    double similarity_measures(double* wat, double* retrieve_wat, int coeff_num, std::string first_element, std::string second_element);
    void show_ucharImage(unsigned char * image, int width, int height, string nameImage, int channels);
    void show_doubleImage(double * image, int width, int height, string nameImage);
    double*  not_blind_extraction(double* original_coeff,double* marked_coeff, int coeff_num, double power);
//    void writeToFile(double* m,int lenght, std::string filename);
    double  threshold_computation(double* original_coeff,int coeff_num, double power);
    void similarity_graph(int number_of_marks,int coeff_num,double* wat);
    double* compute_coeff_function(unsigned char* image, int dim,std::string filename);
//    void viewPointCloudRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr, std::string title);
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizerRGB (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, std::string title);
//    void createPointCloudOpenCV (cv::Mat& img1, cv::Mat& img2,  cv::Mat& Q, cv::Mat& disp, cv::Mat& recons3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &point_cloud_ptr);
//    cv::datasets::FramePair rectifyImages(Mat& img1, Mat& img2, Mat& M1, Mat& D1, Mat& M2, Mat& D2, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Rect &roi1, Rect &roi2, float scale);
//    void generatePointCloud(cv::Mat disp, cv::Mat img_left,cv::Mat img_right, int frame_num);
    void show_floatImage(float ** image, int width, int height, string nameImage);
    void show_doubleImage(double * image, int width, int height, string nameImage);
    void show_ucharImage(unsigned char * image, int width, int height, string nameImage);
    void writefloatMatToFile(float** m,int dim, std::string filepath);
    void writeMatToFile(double** m,int dim, std::string filepath);
    void writeToFile(double* m,int lenght, std::string filepath);
    void show_double_mat(int width,int height,double** A,std::string window_name);
    void compute_luminance(unsigned char* image, int dim, int flag, unsigned char **imr,unsigned char **img,
                                                unsigned char **imb, float **imyout, float **imc2,float **imc3);
    void save_ucharImage(unsigned char * image, int width, int height, string nameImage);
    Mat unsignedToMat(unsigned char * squared_image, Mat original_image,  int width, int height, int dim);

}