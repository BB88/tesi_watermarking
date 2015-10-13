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


       /* string ty =  type2str( disp.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), disp.cols, disp.rows );*/
        int d, xr;
        // original right view
        cv::Mat o_right = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/r.png", CV_LOAD_IMAGE_COLOR);
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
        imwrite("/home/bene/ClionProjects/tesi_watermarking/img/nkz_right.png", n_right);
        /*imshow*/
/*
        cv::imshow("Left",left);
        cv::imshow("Disp",disp);
        cv::imshow("Original",o_right);
        cv::imshow("Reconstructed", n_right);
*/
    }

unsigned char* Right_view::left_uchar_reconstruction(unsigned char *marked_right, unsigned char *disp_uchar, unsigned char* occ_map, int width, int height) {

    int nc = width *3;
    unsigned char d = 0;
    unsigned char occ = 0;
    //create general left image
    int rect_dim = height*width*3;
    unsigned char *left_uchar = new unsigned char[rect_dim];
    for (int i = 0; i< rect_dim; i ++)
        left_uchar[i] = (unsigned char)0;
    for(int i = 0; i<height; i++){
        for (int j =0; j<width ; j++){
            d = disp_uchar[i*width + j];
            occ = occ_map[ i * width + j ];
            if(static_cast<int>(occ)!=0)
            if((j + static_cast<int>(d)) <= width )
                for(int k =0; k<3;k++){
                    left_uchar[(i*width + j+ static_cast<int>(d))*3 + k] =  marked_right[(i*width + j)*3 + k];
                }
        }
    }




//    for(int i = 0; i<height; i++){
//        for (int j = (width-1)*3; j>=0 ; j-=3){
//            d = disp_uchar[(i*width)+((j/3))];
//            occ = occ_map[(i*width)+((j/3))];
//            if (static_cast<unsigned>(occ)!=0){
//                if ((i*nc) + (j +2 + static_cast<unsigned>(d)*3) < (i+1)*width*3 ){
//            // assign new values to reconstructed left view
//                    left_uchar[ (i*nc) + (j + 0 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 0)];
//                    left_uchar[ (i*nc) + (j + 1 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 1)];
//                    left_uchar[ (i*nc) + (j + 2 + static_cast<unsigned>(d)*3)] = marked_right[ (i*nc) + (j + 2)];
//                }
//            }
//
//        }
//    }
   cv::Mat left_marked = cv::Mat::zeros(height, width, CV_8UC3);
    int count = 0;
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){
            left_marked.at<Vec3b>(j,i) [0] = left_uchar[count]; count++;
            left_marked.at<Vec3b>(j,i) [1] = left_uchar[count]; count++;
            left_marked.at<Vec3b>(j,i) [2] = left_uchar[count]; count++;
        }
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/left_reconstructed_uchar.png", left_marked);
    return left_uchar;

}
unsigned char* Right_view::right_uchar_reconstruction(unsigned char *marked_left, unsigned char *disp_uchar, unsigned char* occ_map, int width, int height) {

    unsigned char d = 0;
    unsigned char occ = 0;
    //create general left image
    int rect_dim = width*height*3;
    unsigned char *right_uchar = new unsigned char[rect_dim];
    for (int i = 0; i< rect_dim; i ++)
        right_uchar[i] = (unsigned char)0;
    for(int i = 0; i<height; i++){
        for (int j =0; j<width ; j++){
            d = disp_uchar[i*width + j];
            occ = occ_map[ i * width + j ];
            if(static_cast<int>(occ)!=0)
                if(j - static_cast<int>(d)>=0 )
                    for(int k =0; k<3;k++){
                        right_uchar[(i*width + j- static_cast<int>(d))*3 + k] =  marked_left[(i*width + j)*3 + k];
                    }
        }
    }
    cv::Mat right_marked = cv::Mat::zeros(height, width, CV_8UC3);
    int count = 0;
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){
            right_marked.at<Vec3b>(j,i) [0] = right_uchar[count]; count++;
            right_marked.at<Vec3b>(j,i) [1] = right_uchar[count]; count++;
            right_marked.at<Vec3b>(j,i) [2] = right_uchar[count]; count++;
        }
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/left_reconstructed_uchar.png", right_marked);
    return right_uchar;

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

unsigned char* Right_view::left_rnc(unsigned char *right, cv::Mat disp, cv::Mat occ_map, int width, int height, bool gt) {

        int nc = width;
        unsigned char d = 0;
        unsigned char occ = 0;
//      create general left image
        int dim = width*height*3;
        unsigned char * rcn_left = new unsigned char[dim];
        for (int i = 0; i<dim; i ++)
            rcn_left[i] = (unsigned char)0;
        for (int i=0;i<height;i++)
            for (int j= (width-1);j>=0;j--){
                d = disp.at<uchar>(i,j);

                if (gt )
                    occ = occ_map.at<uchar>(i,j);
                else occ = disp.at<uchar>(i,j);

                if(static_cast<int>(occ)!=0 && ((i*nc + j + static_cast<unsigned>(d))*3 + 2) < ((i+1)*nc*3))
                    for (int k=0; k<3; k++)
                        rcn_left[(i*nc + j + static_cast<int>(d))*3 + k ] = right[(i*nc + j)*3 + k];
            }
        return rcn_left;
    }
