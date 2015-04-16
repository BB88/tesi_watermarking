//
// Created by bene on 10/04/15.
//

#ifndef TEST_TESI_STEREO_MATCHING_H
#define TEST_TESI_STEREO_MATCHING_H

#endif //TEST_TESI_STEREO_MATCHING_H


#include <opencv2/core/core.hpp>
#include <cv.h>
#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace stereomatching {
    using namespace cv;

    void display(Mat &img1, Mat &img2, Mat &disp);

    void stereo_matching(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& disp);

}

