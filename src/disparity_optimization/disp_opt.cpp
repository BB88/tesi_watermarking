//
// Created by miky on 18/06/15.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <highgui.h>
#include <cv.h>
#include <fstream>

#include "disp_opt.h"

using namespace std;
using namespace cv;


/**
 * disparity_normalization(..)
 *
 * normalize the disparity map computed by graph cuts
 *
 * @params kz_disp: disparity map computed by kolmogorov-zabih algorithm
 * @params dMin: minimum disparity value
 * @params dMax: maximum disparity value
 * @output wkz_disp: normalized disparity map
 */
void Disp_opt::disparity_normalization(cv::Mat kz_disp,int dMin,int dMax, cv::Mat &wkz_disp) {

    int d, c, dispSize;
    dispSize = dMax - dMin + 1;
    wkz_disp = cv::Mat::zeros(kz_disp.rows, kz_disp.cols, CV_8UC1);
    for (int j = 0; j < kz_disp.rows; j++) {
        for (int i = 0; i < kz_disp.cols; i++) {
            c = kz_disp.at<uchar>(j, i);
            if (c != 179) {
                d = (c - 255) * dispSize / -(255 - 64) + dMin + 1;
                wkz_disp.at<uchar>(j, i) = -d;
            } else {
                wkz_disp.at<uchar>(j, i) = 0;
            }
        }
    }
}

