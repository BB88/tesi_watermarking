//
// Created by bene on 02/10/15.
//

#include "frequencyWatermarking.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "../dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>

#include "../right_view_computation/right_view.h"
#include <fstream>

//includes watermarking
#include "../img_watermarking/watermarking.h"
#include "../img_watermarking/allocim.h"
#include "../utils.h"
#include "../img_watermarking/fft2d.h"
#include "../graphcuts/utils.h"
//#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace graph_cuts_utils;


/**
 * stereoWatermarking(..)
 *
 * frequency stereo watermark embedding process
 *
 * @params frameL: left view to watermark
 * @params frameR: right view to watermark
 * @params wsize: watermark size
 * @params power: watermark size
 * @params passwstr: string password
 * @params passwnum: alphanumeric password
 * @params watermark: 64 bit watermark sequence
 * @params img_num: frame number
 * @return output: watermarked stereo frames
 */
vector<cv::Mat> DFTStereoWatermarking::stereoWatermarking(cv::Mat frameL, cv::Mat frameR, int wsize, float power, std::string passwstr,
                                                 std::string passwnum, int* watermark, int img_num){

    Right_view rv;
    int dim = 512; //cropping size
    vector<cv::Mat> output;
    unsigned char *left_uchar = frameL.data;
    int squared_dim = dim * dim *3;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    int nc = 640;
    int nc_s = dim;
    int offset = 127;
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }

    Watermarking image_watermarking;
    image_watermarking.setParameters(watermark,wsize,power);
    image_watermarking.setPassword(passwstr,passwnum);
    float  **imidft_wat;
    imidft_wat = AllocIm::AllocImFloat(dim, dim);

    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,dim,dim,dim,imidft_wat);


    bool left_to_right = true;

    // prendo dmin e dmax e calcolo disp con kz
    std::string disp_data;
    std::vector<std::string> disprange;
    char sep = ' ';
    std::ifstream in("/home/bene/Scrivania/Tesi/dispRange.txt");
    if (in.is_open()) {
        int j=0;
        while (!in.eof()){
            if ( j == img_num ){
                getline(in, disp_data);
                for(size_t p=0, q=0; p!=disp_data.npos; p=q){
                    disprange.push_back(disp_data.substr(p+(p!=0), (q=disp_data.find(sep, p+1))-p-(p!=0)));
                }
            }
            getline(in, disp_data);
            j+=60;
        }
        in.close();
    }
    int dminl = atoi(disprange[0].c_str());
    int dmaxl = atoi(disprange[1].c_str());



    std::ostringstream pathL;

    //load ground truth disparity
    //  pathL << "./dataset/NTSD-200/disparity_maps/left/tsukuba_disparity_L_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts disparity
    pathL << "./img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";

    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    std::ostringstream pathR;

    //load ground truth disparity
    //pathR << "./dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts disparity
    pathR << "./img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";

    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat squared_lDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
        }
    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
        }

    unsigned char *right_uchar = frameR.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }

    unsigned char * recleft = rv.left_rnc_no_occ(squared_right,squared_rDisp ,dim,dim);
    float  **imidft_wat_rec;
    imidft_wat_rec = AllocIm::AllocImFloat(dim, dim);
    cv::Mat square_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    int count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            square_left_mat.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }
    cv::Mat rec_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            rec_left_mat.at<Vec3b>(j, i)[0] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[1] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[2] = recleft[count];
            count++;
        }
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++){
            if ( rec_left_mat.at<Vec3b>(j, i)[0]==0 && rec_left_mat.at<Vec3b>(j, i)[1]==0 && rec_left_mat.at<Vec3b>(j, i)[2]==0){
                rec_left_mat.at<Vec3b>(j, i)[0] = square_left_mat.at<Vec3b>(j, i)[0];
                rec_left_mat.at<Vec3b>(j, i)[1] = square_left_mat.at<Vec3b>(j, i)[1];
                rec_left_mat.at<Vec3b>(j, i)[2] = square_left_mat.at<Vec3b>(j, i)[2];
            }
        }
    unsigned char *squared_marked_left_rec = image_watermarking.insertWatermark(rec_left_mat.data,dim,dim,dim,imidft_wat_rec);

//    computing warped watermark   ********************
    float  **warp_mark;
    warp_mark = AllocIm::AllocImFloat(dim, dim);
    for (int i=0; i <dim; i++)
        for (int j=0;j<dim;j++)
            warp_mark[i][j] = 0.0;
    unsigned char d = 0;
    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            d = squared_lDisp.at<uchar>(i,j);
            int diff = j-static_cast<int>(d);
            if(static_cast<int>(d)!=0 && diff>=0)
                warp_mark[i][j-static_cast<int>(d)] = imidft_wat_rec[i][j];
        }
    unsigned char **imr;
    unsigned char **img;
    unsigned char **imb;
    float **imc2;
    float **imc3;
    imc2 = AllocIm::AllocImFloat(dim, dim);
    imc3 = AllocIm::AllocImFloat(dim, dim);
    imr = AllocIm::AllocImByte(dim, dim);
    img = AllocIm::AllocImByte(dim, dim);
    imb = AllocIm::AllocImByte(dim, dim);
    float ** right_lum;
    right_lum = AllocIm::AllocImFloat(dim, dim);
    stereo_watermarking::compute_luminance(squared_right,dim,1,imr,img,imb,right_lum,imc2,imc3);
//   compute marked right lum  ********************
    float ** marked_right_lum = AllocIm::AllocImFloat(dim, dim);

    for (int i = 0; i < nc_s; i++ )
        for (int j = 0; j < nc_s; j++) {
            marked_right_lum[i][j] = right_lum[i][j] + warp_mark[i][j];
        }
//    compute image from luminance   ********************
    unsigned char *marked_right = new unsigned char[squared_dim];
    stereo_watermarking::compute_luminance(marked_right,dim,-1,imr,img,imb,marked_right_lum,imc2,imc3);
// back to normal size *******************
    unsigned char *left_watermarked = new unsigned char [480*640*3];
    left_watermarked = frameL.data;
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < dim; j++) {
            for (int k =0; k<3;k++){
                left_watermarked[(i *nc + j + offset)*3 + k] = squared_marked_left[(i * nc_s + j)*3 + k];
            }
        }
    count = 0;
    cv::Mat left_wat_mat = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){

            left_wat_mat.at<Vec3b>(j, i) [0] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [1] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [2] = left_watermarked[count]; count++;
        }
    output.push_back(left_wat_mat);
    unsigned char *right_watermarked = new unsigned char [480*640*3];
    right_watermarked = frameR.data;
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < dim; j++) {
            for (int k =0; k<3;k++){
                right_watermarked[(i *nc + j + offset)*3 + k] = marked_right[(i * nc_s + j)*3 + k];
            }
        }
    count = 0;
    cv::Mat right_wat_mat = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){

            right_wat_mat.at<Vec3b>(j, i) [0] = right_watermarked[count]; count++;
            right_wat_mat.at<Vec3b>(j, i) [1] = right_watermarked[count]; count++;
            right_wat_mat.at<Vec3b>(j, i) [2] = right_watermarked[count]; count++;
        }
    output.push_back(right_wat_mat);
    return output;
}


/**
 * stereoDetection(..)
 *
 * frequency stereo watermark detection process
 *
 * @params markedL: marked left view
 * @params markedR: marked right view
 * @params wsize: watermark size
 * @params power: watermark size
 * @params passwstr: string password
 * @params passwnum: alphanumeric password
 * @params watermark: 64 bit watermark sequence
 * @params img_num: frame number
 * @return 1: detected in both frames; 2: detected only in the left view; 3: detected only in the right view
 * */
int DFTStereoWatermarking::stereoDetection(cv::Mat markedL, cv::Mat markedR, int wsize, float power, std::string passwstr,
                                            std::string passwnum, int* watermark,int img_num){


    Watermarking image_watermarking;
    image_watermarking.setParameters(watermark,wsize,power);
    image_watermarking.setPassword(passwstr,passwnum);
    int dim = 512;
    int offset = 127;
    int nc = 640;
    int nc_s = dim;
    int squared_dim = dim * dim *3;
    unsigned char *left_uchar = markedL.data;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }
    unsigned char *right_uchar = markedR.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }


    std::ostringstream pathL;
    //load ground truth disparity
//      pathL << "./dataset/NTSD-200/disparity_maps/left/tsukuba_disparity_L_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts disparity
    pathL << "./img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);


    std::ostringstream pathR;
    //load ground truth disparity
//    pathR << "./dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts disparity
    pathR << "./img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";

    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);


    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
        }

    Right_view rv;
    unsigned char * rcn_squared_left = rv.left_rnc_no_occ(squared_right,squared_rDisp,dim,dim);

    cv::Mat square_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    int count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {
            square_left_mat.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }
    cv::Mat rcn_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            rcn_left_mat.at<Vec3b>(j, i)[0] = rcn_squared_left[count];
            count++;
            rcn_left_mat.at<Vec3b>(j, i)[1] = rcn_squared_left[count];
            count++;
            rcn_left_mat.at<Vec3b>(j, i)[2] = rcn_squared_left[count];
            count++;
    }
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++){
            if ( rcn_left_mat.at<Vec3b>(j, i)[0]==0 && rcn_left_mat.at<Vec3b>(j, i)[1]==0 && rcn_left_mat.at<Vec3b>(j,
                                                                                                                                i)[2]==0){
                rcn_left_mat.at<Vec3b>(j, i)[0] = square_left_mat.at<Vec3b>(j, i)[0];
                rcn_left_mat.at<Vec3b>(j, i)[1] = square_left_mat.at<Vec3b>(j, i)[1];
                rcn_left_mat.at<Vec3b>(j, i)[2] = square_left_mat.at<Vec3b>(j, i)[2];
            }
    }


    bool left_det = image_watermarking.extractWatermark(squared_left,dim,dim, dim);
    bool rcnleft_det = image_watermarking.extractWatermark(rcn_left_mat.data,dim,dim,dim);
    if(left_det)
        if (rcnleft_det)
            return 1;
        else return 2;
    if(rcnleft_det)
        return 3;
    else return 0;
}
