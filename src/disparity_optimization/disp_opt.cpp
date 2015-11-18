//
// Created by bene on 18/06/15.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <highgui.h>
#include <cv.h>
#include <fstream>

#include "disp_opt.h"

using namespace std;
using namespace cv;

void Disp_opt::disparity_filtering(cv::Mat kz_disp) {

    cv::Mat output;
    //median filter
    cv::medianBlur(kz_disp, output, 7);
    imshow("Filtered disparity", output);
    //save filtered colored image
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/f_disp_synt.png", output);
    cv::Mat greyMat;
    //convert filtered colored disparity to greyscale
    cv::cvtColor(output, greyMat, CV_BGR2GRAY);
    //save filtered greyscale image
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/fg_disp_synt.png", greyMat);
    imshow("Filtered greyscale disparity", greyMat);

}

// !! change the name of the normalized image you want to save !!

void Disp_opt::disparity_normalization(cv::Mat kz_disp,int dMin,int dMax, cv::Mat &wkz_disp) {


//    std::ofstream dispFile;
//    dispFile.open("/home/bene/Scrivania/dispMat2.txt");

//    cv::Mat kz_disp_gray;
//    cv::cvtColor(kz_disp,kz_disp_gray,CV_BGR2GRAY);

    int d, c, dispSize;
/* dMin = -77;
dMax = -19;*/
// dMin = -33;
// dMax = -8;
    dispSize = dMax - dMin + 1;
    wkz_disp = cv::Mat::zeros(kz_disp.rows, kz_disp.cols, CV_8UC1);
// load ground_truth for comparison
// Mat gt_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png",
// CV_LOAD_IMAGE_GRAYSCALE);
// cout << "channels" << fg_disp.channels() << endl;
    for (int j = 0; j < kz_disp.rows; j++) {
        for (int i = 0; i < kz_disp.cols; i++) {
            c = kz_disp.at<uchar>(j, i);
            if (c != 179) {
                d = (c - 255) * dispSize / -(255 - 64) + dMin + 1;
//                dispFile << d << "m" << " " << c << " " << -d << std::endl;
                wkz_disp.at<uchar>(j, i) = -d;
            } else {
                wkz_disp.at<uchar>(j, i) = 0;
//                dispFile << "X" << std::endl;
            }

// d = - ((255 - D) * dispSize / (255 - 64) + dMin);
// cout << d << endl;

        }
    }
//    dispFile.close();
}

void Disp_opt::occlusions_enhancing(cv::Mat f_disp) {

    namedWindow("Filtered disparity", CV_WINDOW_AUTOSIZE);
    imshow("Filtered disparity", f_disp);
    // Modify the pixels of disparity: occlusions are black, the rest is white

    //ho cambiato i con j ricontrollare che funzioni

    for (int j = 0; j < f_disp.rows; j++) {
        for (int i = 0; i < f_disp.cols; i++) {
            if ((f_disp.at<Vec3b>(j, i)[0] == 255 && f_disp.at<Vec3b>(j, i)[1] == 255 &&
                 f_disp.at<Vec3b>(j, i)[2] == 0) ||
                (f_disp.at<Vec3b>(j, i)[0] < 100 && f_disp.at<Vec3b>(j, i)[1] < 100 &&
                 f_disp.at<Vec3b>(j, i)[2] < 100)) {
                f_disp.at<Vec3b>(j, i)[0] = 0;
                f_disp.at<Vec3b>(j, i)[1] = 0;
                f_disp.at<Vec3b>(j, i)[2] = 0;
            } else {
                f_disp.at<Vec3b>(j, i)[0] = 255;
                f_disp.at<Vec3b>(j, i)[1] = 255;
                f_disp.at<Vec3b>(j, i)[2] = 255;
            }
        }
    }
    namedWindow("Modified pixel", CV_WINDOW_AUTOSIZE);
    imshow("Modified pixel", f_disp);
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/fbw_disp.png", f_disp);
}