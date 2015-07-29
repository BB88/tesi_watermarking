//
// Created by bene on 05/06/15.
//

#include "right_view.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;


    void Right_view::right_reconstruction(cv::Mat left, cv::Mat disp) {
        // read left image

        cv::imshow("Left",left);
        cv::imshow("Disp",disp);
       /* string ty =  type2str( disp.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), disp.cols, disp.rows );*/
        //create general right image
        int d, xr;
        // original right view
        cv::Mat o_right = cv::imread("/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/daylight/right/frame_1.png",
                                     CV_LOAD_IMAGE_COLOR);
        //Before changing
        cv::imshow("Original",o_right);
        /*
        string ty =  type2str( o_right.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), o_right.cols, o_right.rows);
        */
        // reconstructed right view
        cv::Mat n_right = cv::Mat::zeros(o_right.rows, o_right.cols, CV_8UC3);
        // change pixel value
        for(int j=0;j< left.rows;j++)
        {
            for (int i=0;i< left.cols;i++)
            {
                d = disp.at<uchar>(j,i);
//                cout<<d<<endl;
                xr = abs(i - d);
               // xr = i - d;
                // assign new values to reconstructed right view
                n_right.at<Vec3b>(j,xr) [0] = left.at<Vec3b>(j,i) [0];
                n_right.at<Vec3b>(j,xr) [1] = left.at<Vec3b>(j,i) [1];
                n_right.at<Vec3b>(j,xr) [2] = left.at<Vec3b>(j,i) [2];
            }
        }
        imwrite("/home/miky/ClionProjects/tesi_watermarking/img/nkz_right.png", n_right);
        //After changing
        cv::imshow("Reconstructed", n_right);
        waitKey(0);
    }

    string type2str(int type) {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }
