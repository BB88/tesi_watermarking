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
    void right_reconstruction(cv::Mat left, cv::Mat disp);
    void left_reconstruction(cv::Mat right, cv::Mat disp);
    unsigned char * left_uchar_reconstruction(unsigned char *right_uchar,  unsigned char *disp_uchar,unsigned char* occ_map, int width, int height);
    string type2str(int type);
    unsigned char* right_uchar_reconstruction(unsigned char *marked_right, unsigned char *disp_uchar, unsigned char* occ_map, int width, int height);
    unsigned char* left_rnc(unsigned char *right, cv::Mat disp, cv::Mat occ_map, int width, int height,bool gt);


    unsigned char* left_rnc_no_occ(unsigned char *right, cv::Mat disp, int width, int height);
};



#endif //TESI_WATERMARKING_RIGHT_VIEW_H
