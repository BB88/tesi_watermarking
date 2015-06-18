//
// Created by bene on 18/06/15.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <highgui.h>
#include <cv.h>

#include "disp_opt.h"

using namespace std;
using namespace cv;

void Disp_opt::disparity_filtering(cv::Mat kz_disp) {

    cv::Mat output;
    //median filter
    cv::medianBlur(kz_disp, output, 7);
    imshow("Filtered disparity", output);
    //save filtered colored image
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", output);
    cv::Mat greyMat;
    //convert filtered colored disparity to greyscale
    cv::cvtColor(output, greyMat, CV_BGR2GRAY);
    //save filtered greyscale image
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/fg_disp.png", greyMat);
    imshow("Filtered greyscale disparity", greyMat);
    waitKey(25);

}

void Disp_opt::disparity_normalization(cv::Mat fg_disp) {

    int d, D , dMin, dMax, dispSize;
    dMin = -70;
    dMax = 0;
    dispSize = dMax - dMin + 1;
//    Mat disp = imread("/home/bene/Scrivania/disp/filt_grey_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Filtered greyscale disparity", fg_disp);
    cv::Mat n_disp = cv::Mat::zeros(fg_disp.rows, fg_disp.cols, CV_8UC1);
    // load ground_truth for comparison
    Mat gt = imread("/home/bene/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/left/frame_1.png",
                    CV_LOAD_IMAGE_GRAYSCALE);
    cv::imshow("Ground Truth",gt);
//    cout << "channels" << fg_disp.channels() << endl;
    for(int j=0;j< fg_disp.rows;j++) {
        for (int i = 0; i < fg_disp.cols; i++) {
            D = fg_disp.at<uchar>(j,i);
            d = - ((255 - D) * dispSize / (255 - 64) + dMin);
           // cout << d << endl;
            n_disp.at<uchar>(j,i) = d;
        }
    }
    imwrite("/home/bene/Scrivania/disp/orig_disp.png", n_disp);
    imshow("Normalized disparity", n_disp);
    waitKey(0);
}


void Disp_opt::occlusions_enhancing(cv::Mat f_disp) {

    namedWindow("Filtered disparity", CV_WINDOW_AUTOSIZE);
    imshow("Filtered disparity", f_disp);
    // Modify the pixels of disparity: occlusions are black, the rest is white
    for (int i = 0; i < f_disp.rows; i++) {
        for (int j = 0; j < f_disp.cols; j++) {
            if ((f_disp.at<Vec3b>(i, j)[0] == 255 && f_disp.at<Vec3b>(i, j)[1] == 255 &&
                    f_disp.at<Vec3b>(i, j)[2] == 0) ||
                (f_disp.at<Vec3b>(i, j)[0] < 100 && f_disp.at<Vec3b>(i, j)[1] < 100 &&
                        f_disp.at<Vec3b>(i, j)[2] < 100)) {
                f_disp.at<Vec3b>(i, j)[0] = 0;
                f_disp.at<Vec3b>(i, j)[1] = 0;
                f_disp.at<Vec3b>(i, j)[2] = 0;
            } else {
                f_disp.at<Vec3b>(i, j)[0] = 255;
                f_disp.at<Vec3b>(i, j)[1] = 255;
                f_disp.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    namedWindow("Modified pixel", CV_WINDOW_AUTOSIZE);
    imshow("Modified pixel", f_disp);
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/fbw_disp.png", f_disp);
    waitKey(0);
}