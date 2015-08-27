//
// Created by miky on 16/08/15.
//

#ifndef TESI_WATERMARKING_UTILS_H
#define TESI_WATERMARKING_UTILS_H

#endif //TESI_WATERMARKING_UTILS_H

namespace stereo_watermarking{
    void show_difference(cv::Mat img1,cv::Mat img2,std::string window);
    cv::Mat equalizeIntensity(const cv::Mat& inputImage);
    void sobel_filtering(cv::Mat src,const char* window_name);

    void histo (cv::Mat image, std::string window_name);
    void printRGB (cv::Mat image, int x, int y);
    int equi_histo(cv::Mat image, std::string window_name, cv::Mat &equi_image);
//    cv::Scalar getMQdepth( const cv::Mat& i1, const cv::Mat& i2);

}