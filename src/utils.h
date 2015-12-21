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
    cv::Mat sobel_filtering(cv::Mat src);
    void show_double_mat(int width,int height,double** A,std::string window_name);
    void show_ucharImage(unsigned char * image, int width, int height, string nameImage, int channels);
    void show_doubleImage(double * image, int width, int height, string nameImage);
    void show_floatImage(float ** image, int width, int height, string nameImage);
    void show_doubleImage(double * image, int width, int height, string nameImage);
    void writefloatMatToFile(float** m,int dim, std::string filepath);
    void writeMatToFile(double** m,int dim, std::string filepath);
    void writeToFile(double* m,int lenght, std::string filepath);
    void show_double_mat(int width,int height,double** A,std::string window_name);
    void compute_luminance(unsigned char* image, int dim, int flag, unsigned char **imr,unsigned char **img,unsigned char **imb, float **imyout, float **imc2,float **imc3);
}