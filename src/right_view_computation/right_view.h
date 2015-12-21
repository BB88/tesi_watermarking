//
// Created by bene on 05/06/15.
//

#ifndef TESI_WATERMARKING_RIGHT_VIEW_H
#define TESI_WATERMARKING_RIGHT_VIEW_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Right_view{

public:
     unsigned char* left_rnc_no_occ(unsigned char *right, cv::Mat disp, int width, int height);
};



#endif //TESI_WATERMARKING_RIGHT_VIEW_H
