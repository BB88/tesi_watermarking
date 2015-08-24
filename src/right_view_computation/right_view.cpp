//
// Created by miky on 05/06/15.
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


       /* string ty =  type2str( disp.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), disp.cols, disp.rows );*/
        int d, xr;
        // original right view
        cv::Mat o_right = cv::imread("/home/miky/ClionProjects/tesi_watermarking/img/r.png", CV_LOAD_IMAGE_COLOR);
        /*
        string ty =  type2str( o_right.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), o_right.cols, o_right.rows);
        */
        //create general right image
        cv::Mat n_right = cv::Mat::zeros(left.rows, left.cols, CV_8UC3);
        // reconstructed right view
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
        /*imshow*/
/*
        cv::imshow("Left",left);
        cv::imshow("Disp",disp);
        cv::imshow("Original",o_right);
        cv::imshow("Reconstructed", n_right);
*/
    }


    void Right_view::left_reconstruction(cv::Mat right, cv::Mat disp) {


        int d, xl;
        //create general left image
        cv::Mat n_left = cv::Mat::zeros(right.rows, right.cols, CV_8UC3);
        // reconstructed right view
        // change pixel value
        for(int j=0;j< right.rows;j++){
            for (int i=right.cols-1; i>=0;i--){
                d = disp.at<uchar>(j,i);
                xl = i + d;
        // assign new values to reconstructed right view
                if(xl <= right.cols){
                    n_left.at<Vec3b>(j, xl) [0] = right.at<Vec3b>(j,i) [0];
                    n_left.at<Vec3b>(j, xl) [1] = right.at<Vec3b>(j,i) [1];
                    n_left.at<Vec3b>(j, xl) [2] = right.at<Vec3b>(j,i) [2];
                }
            }
        }
        imwrite("/home/miky/ClionProjects/tesi_watermarking/img/left_reconstructed.png", n_left);
        cv::Mat left = cv::imread("/home/miky/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
        /*imshow*/
  /*      cv::imshow("Left",right);
        cv::imshow("Disp",disp);
        cv::imshow("Original",left);*/
        cv::imshow("Reconstructed", n_left);
        waitKey(0);

    }

void Right_view::left_uchar_reconstruction(unsigned char *marked_right, unsigned char *disp_uchar, int width, int height) {


    int nc = 640 *3;
    unsigned char d = 0;
    //create general left image
    int rect_dim = 480*640*3;
    unsigned char *left_uchar = new unsigned char[rect_dim];
    for (int i = 0; i< rect_dim; i ++)
        left_uchar[i] = (unsigned char)0;
    // reconstructed right view
//     change pixel value
    cout<< static_cast<unsigned>(disp_uchar[(479*640)+639])<<endl;
    cout<< static_cast<unsigned>(marked_right[479*640*3 + (639*3)])<<endl;
//    int z = 639;
    for(int i = 0; i<480; i++){
        for (int j = (640-1)*3; j>=0 ; j-=3){
//            d = disp_uchar[(i*640)+z];
            d = disp_uchar[(i*640)+((j/3))];
//            z--;
//            if ((i*nc) + (j - 0 + static_cast<unsigned>(d)*3) < rect_dim && ){
            if ((i*nc) + (j +2 + static_cast<unsigned>(d)*3) < (i+1)*640*3 ){
            // assign new values to reconstructed left view
                left_uchar[ (i*nc) + (j + 0 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 0)];
                left_uchar[ (i*nc) + (j + 1 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 1)];
                left_uchar[ (i*nc) + (j + 2 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 2)];
//                left_uchar[ (i*nc) + (j*3 - 0 + static_cast<unsigned>(d)*3)] = (unsigned char) 127;
//                left_uchar[ (i*nc) + (j*3 - 1 + static_cast<unsigned>(d)*3)] = (unsigned char) 127;
//                left_uchar[ (i*nc) + (j*3 - 2 + static_cast<unsigned>(d)*3)] = (unsigned char) 127;
            }

        }
//        cout<<i<<endl;
    }
   cv::Mat left_marked = cv::Mat::zeros(480, 640, CV_8UC3);
    int count = 0;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            left_marked.at<Vec3b>(j,i) [0] = left_uchar[count]; count++;
            left_marked.at<Vec3b>(j,i) [1] = left_uchar[count]; count++;
            left_marked.at<Vec3b>(j,i) [2] = left_uchar[count]; count++;
        }
    imwrite("/home/miky/ClionProjects/tesi_watermarking/img/left_reconstructed_uchar.png", left_marked);
    imshow("left_marked", left_marked);
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
