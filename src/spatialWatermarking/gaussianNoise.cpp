//
// Created by bene on 02/10/15.
//

#include "gaussianNoise.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "../dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>

#include "../right_view_computation/right_view.h"


//#include <boost/algorithm/string.hpp>
#include <fstream>

using namespace cv;
using namespace std;

/**
 * gaussianNoiseStereoWatermarking(..)
 *
 * spatial stereo watermark embedding process
 *
 * @params left: left view to watermark
 * @params right: right view to watermark
 * @params noise: watermark
 * @params img_num: frame number
 * @return output: watermarked stereo frames
 */
vector<cv::Mat> spatialWatermarking::gaussianNoiseStereoWatermarking(cv::Mat left, cv::Mat right,cv::Mat noise, int img_num,std::string dispfolder){

    int height = left.rows; //height (480)
    int width = left.cols; // width (640)
    vector<cv::Mat> output;
    Mat left_w = left.clone();
    left_w += noise;
    std::ostringstream pathL;

    //load ground truth disparity
    // pathL << "./dataset/NTSD-200/disparity_maps/left/tsukuba_disparity_L_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts disparity
//    pathL << "./img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    pathL <<dispfolder<< "/norm_disp_left_to_right_" << img_num/10 << ".png";

    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat warped_mark = cv::Mat::zeros(left_w.rows, left_w.cols , CV_8UC3);
    int d;
    for (int j = 0; j < height; j++)
        for (int i = 0; i< width; i++){
            d = disp_left.at<uchar>(j,i);
            if ((i-d)>=0){
                warped_mark.at<Vec3b>(j, i-d) [0] =  noise.at<Vec3b>(j, i) [0];
                warped_mark.at<Vec3b>(j, i-d) [1] =  noise.at<Vec3b>(j, i) [1];
                warped_mark.at<Vec3b>(j, i-d) [2] =  noise.at<Vec3b>(j, i) [2];
            }
        }
    cv::Mat right_warp_w;
    right.copyTo(right_warp_w);
    right_warp_w += warped_mark;
    output.push_back(left_w);
    output.push_back(right_warp_w);
    return output;
}

/**
 * gaussianNoiseStereoDetection(..)
 *
 * spatial stereo watermark detection process
 *
 * @params left: marked left view
 * @params right: marked right view
 * @params noise: watermark to detect
 * @params img_num: frame number
 * @return correlations: detection correlation values
 */
vector<float> spatialWatermarking::gaussianNoiseStereoDetection(cv::Mat left_w, cv::Mat right_w, cv::Mat noise, int img_num,std::string dispfolder){

    int height = left_w.rows; //height (480)
    int width = right_w.cols; // width (640)
    vector<float> correlations;
    normalize(left_w, left_w,0, 255, CV_MINMAX, CV_8UC3);
    normalize(right_w, right_w,0, 255, CV_MINMAX, CV_8UC3);

    Mat left_correl;
    Mat m1, m2;
    left_w.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);
    matchTemplate(m1, m2, left_correl, CV_TM_CCOEFF_NORMED);
    for (int i = 0; i < left_correl.rows; i++)
    {
        for (int j = 0; j < left_correl.cols; j++)
        {
            cout << "correlation btw left watermarked and watermark " << (left_correl.at<float>(i,j));
            correlations.push_back(left_correl.at<float>(i,j));
        }
        cout << endl;
    }
    Mat right_correl;
    right_w.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);
    matchTemplate(m1, m2, right_correl, CV_TM_CCOEFF_NORMED);
    for (int i = 0; i < left_correl.rows; i++)
    {
        for (int j = 0; j < right_correl.cols; j++)
        {
            cout << "correlation btw right with not warped watermarked and watermark " << (right_correl.at<float>(i,j));
            correlations.push_back(right_correl.at<float>(i,j));
        }
        cout << endl;
    }

//    std::ostringstream pathL;
//    // load ground truth disparity
//    //pathL << "./dataset/NTSD-200/disparity_maps/left/tsukuba_disparity_L_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
//    // load graph cuts disparity
//    pathL << "./img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    std::ostringstream pathL;
    pathL <<dispfolder<< "/norm_disp_left_to_right_" << img_num/10 << ".png";

    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    std::ostringstream pathR;

    //load ground truth disparity
    //pathR << "./dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts rightToLeft disparity
    pathR <<dispfolder<< "/norm_disp_right_to_left_"  << img_num/10 << ".png";

    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);


//    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat warped_mark = cv::Mat::zeros(left_w.rows, left_w.cols , CV_8UC3);
    int d;
    for (int j = 0; j < height; j++)
        for (int i = 0; i< width; i++){
            d = disp_left.at<uchar>(j,i);
            if ((i-d)>=0){
                warped_mark.at<Vec3b>(j, i-d) [0] =  noise.at<Vec3b>(j, i) [0];
                warped_mark.at<Vec3b>(j, i-d) [1] =  noise.at<Vec3b>(j, i) [1];
                warped_mark.at<Vec3b>(j, i-d) [2] =  noise.at<Vec3b>(j, i) [2];
            }
        }
    Mat right_warped_correl;
    right_w.convertTo(m1, CV_32F);
    warped_mark.convertTo(m2, CV_32F);
    matchTemplate(m1, m2, right_warped_correl, CV_TM_CCOEFF_NORMED);
    for (int i = 0; i < right_warped_correl.rows; i++)
    {
        for (int j = 0; j < right_warped_correl.cols; j++)
        {
            cout << "correlation btw right with warped watermark and warped watermark " << (right_warped_correl.at<float>(i,j));
            correlations.push_back(right_warped_correl.at<float>(i,j));
        }
        cout << endl;
    }
//    std::ostringstream pathR;

    // load ground truth disparity
//    pathR << "./dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    // load graph cuts disparity
//    pathR << "./img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
//    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat occ_right = imread("./img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);

    Right_view rv;
    unsigned char * left_recon = rv.left_rnc_no_occ(right_w.data,disp_right ,width,height);
    cv::Mat left_reconstructed = cv::Mat::zeros(height, width , CV_8UC3);
    int count = 0;
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){
            left_reconstructed.at<Vec3b>(j,i) [0] = left_recon[count]; count++;
            left_reconstructed.at<Vec3b>(j,i) [1] = left_recon[count]; count++;
            left_reconstructed.at<Vec3b>(j,i) [2] = left_recon[count]; count++;
        }
    Mat left_rec_correl;
    left_reconstructed.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);
    matchTemplate(m1, m2, left_rec_correl, CV_TM_CCOEFF_NORMED);
    for (int i = 0; i < left_rec_correl.rows; i++)
    {
        for (int j = 0; j < left_rec_correl.cols; j++)
        {
            cout << "correlation btw left reconstructed with watermark and watermark " << (left_rec_correl.at<float>(i,j));
            correlations.push_back(left_rec_correl.at<float>(i,j));
        }
        cout << endl;
    }


    return correlations;
}