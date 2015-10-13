//
// Created by miky on 02/10/15.
//

#ifndef TESI_WATERMARKING_FREQUENCYWATERMARKING_H
#define TESI_WATERMARKING_FREQUENCYWATERMARKING_H

#endif //TESI_WATERMARKING_FREQUENCYWATERMARKING_H
#include <iostream>
#include <cv.h>

using namespace cv;
namespace FDTStereoWatermarking{

    void warpMarkWatermarking(Mat left, Mat right, int* watermark, int wsize, float power, std::string passwstr, std::string passwnum, bool gt);
    void warpRightWatermarking(int wsize, int tilesize, float power, bool clipping,
                                                      bool flagResyncAll, int tilelistsize, std::string passwstr,
                                                      std::string passwnum, bool gt);
    void videoWatermarking(Mat left, Mat right, int*watermark,int wsize, float power, std::string passwstr,
                                             std::string passwnum, bool gt, Mat &marked_left, Mat &markedRight);
    void videoDetection(Mat marked_left, Mat marked_right, int*watermark,int wsize, float power, std::string passwstr,
    std::string passwnum, int dim);
}