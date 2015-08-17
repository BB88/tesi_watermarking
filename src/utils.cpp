//
// Created by miky on 16/08/15.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include "utils.h"
#include <cv.h>
#include <highgui.h>

using namespace std;


void stereo_watermarking::show_difference(cv::Mat img1,cv::Mat img2){

    unsigned char *difference =  new unsigned char[img1.rows * img1.cols *3];
    unsigned char *img1_uchar =  img1.data;
    unsigned char *img2_uchar =  img2.data;

    for (int i=0;i<img1.rows * img1.cols *3;i++){
        difference[i] = img1_uchar[i] - img2_uchar[i];
    }

    cv::Mat difference_cv = cv::Mat::zeros(img1.rows, img1.cols , CV_8UC3);

    int count=0;
   for (int j = 0; j < img1.rows; j++)
        for (int i = 0; i < img1.cols; i++){
            difference_cv.at<cv::Vec3b>(j, i) [0] = difference[count]; count++;
            difference_cv.at<cv::Vec3b>(j, i) [1] = difference[count]; count++;
            difference_cv.at<cv::Vec3b>(j, i) [2] = difference[count]; count++;
        }
    cv::imshow("difference", difference_cv);
    cv::waitKey(0);
}

