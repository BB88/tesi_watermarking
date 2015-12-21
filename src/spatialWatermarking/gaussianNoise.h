//
// Created by miky on 02/10/15.
//

#ifndef TESI_WATERMARKING_GAUSSIANNOISE_H
#define TESI_WATERMARKING_GAUSSIANNOISE_H

#endif //TESI_WATERMARKING_GAUSSIANNOISE_H
#include <cv.h>
#include <cstdint>
#include <fstream>

using namespace cv;
using namespace std;

namespace spatialWatermarking{

    vector<cv::Mat> gaussianNoiseStereoWatermarking(cv::Mat left, cv::Mat right,cv::Mat noise, int img_num);
    vector<float> gaussianNoiseStereoDetection(cv::Mat left_w, cv::Mat right_w,cv::Mat noise, int img_num);

}