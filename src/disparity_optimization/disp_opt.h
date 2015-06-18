//
// Created by bene on 18/06/15.
//

#include <iostream>
#include <opencv2/core/core.hpp>
//imwrite  imread
#include <highgui.h>
//cv::Mat etc
#include <cv.h>

#ifndef TESI_WATERMARKING_DIPARITY_OPTIMIZATION_H
#define TESI_WATERMARKING_DIPARITY_OPTIMIZATION_H


class Disp_opt {

public:
    void disparity_filtering(cv::Mat kz_disp);
    void disparity_normalization(cv::Mat g_disp);
    // to compare with the grouond truth occlusion map
    void occlusions_enhancing(cv::Mat f_disp);


};


#endif //TESI_WATERMARKING_DIPARITY_OPTIMIZATION_H
