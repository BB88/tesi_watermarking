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

    /*aggiunta per le dimensioni*/
    int watDim = 0;
    int height = frameL.rows; //height (480)
    int width = frameL.cols; // width (640)
    if (max(width,height)<256){
        std::cout<<"Frame too small"<<endl;
        // gestire l'uscita dalla funzione
    }
    if(max(width,height)<512){
        watDim = 256;
    } else if (max(width,height)<1024){
        watDim = 512;
    } else {
        watDim = 1024;
    }
    int n_cols = min(width, watDim);
    int n_rows = min(height, watDim);

    /*FIne aggiunta per le dimensioni*/

    // squared left construction
    Right_view rv;
//    int dim = 512; //cropping size      //MODIFICA BENE
    vector<cv::Mat> output;
    unsigned char *left_uchar = frameL.data;
//    int squared_dim = dim * dim *3;
    int squared_dim = watDim * watDim *3;      //MODIFICA BENE

    unsigned char *squared_left =  new unsigned char[squared_dim];
 //   int nc = 640;   //MODIFICA BENE
//    int nc_s = dim;  //MODIFICA BENE
//    int offset = 127;  //MODIFICA BENE
//    for (int i = 0; i < height; i++ )     //MODIFICA BENE
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
//            }
//        }
    /*aggiunta per le dimensioni*/
    int offset = abs(watDim - width) - 1;
    for (int i = 0; i < n_rows; i++ )
        for (int j = 0; j < n_cols; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * watDim + j)*3 + k] = left_uchar[(i *width + j + offset)*3 + k];
            }
        }
    /*FIne aggiunta per le dimensioni*/

    Watermarking image_watermarking;
    image_watermarking.setParameters(watermark,wsize,power);
    image_watermarking.setPassword(passwstr,passwnum);
    float  **imidft_wat;
//    imidft_wat = AllocIm::AllocImFloat(dim, dim); //MODIFICA BENE
    imidft_wat = AllocIm::AllocImFloat(watDim, watDim);

//    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,dim,dim,dim,imidft_wat); //MODIFICA BENE
    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,watDim,watDim,watDim,imidft_wat);// modificare la funzione in modo che prenda un solo parametro watdim

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
    //load graph cuts leftToRight disparity
    pathL << "./img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";

    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    std::ostringstream pathR;

    //load ground truth disparity
    //pathR << "./dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //load graph cuts rightToLeft disparity
    pathR << "./img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";

    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat squared_lDisp = cv::Mat::zeros(dim, dim, CV_8UC1);     //MODIFICA BENE
    /*aggiunta per le dimensioni*/
    cv::Mat squared_lDisp = cv::Mat::zeros(watDim, watDim, CV_8UC1);

    for (int i=0; i <n_rows; i++)
        for (int j=0;j < n_cols;j++){
            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
        }
    /*FIne aggiunta per le dimensioni*/


//    for (int i=0; i <480; i++)  //MODIFICA BENE
//        for (int j=0;j<dim;j++){
//            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
//        }



//    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);     //MODIFICA BENE
    /*aggiunta per le dimensioni*/
    cv::Mat squared_rDisp = cv::Mat::zeros(watDim, watDim, CV_8UC1);
    for (int i=0; i <n_rows; i++)
        for (int j=0;j<n_cols;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
        }
    /*FIne aggiunta per le dimensioni*/

//    for (int i=0; i <480; i++)  //MODIFICA BENE
//        for (int j=0;j<dim;j++){
//            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
//        }

    unsigned char *right_uchar = frameR.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    /*aggiunta per le dimensioni*/
    for (int i = 0; i < n_rows; i++ )
        for (int j = 0; j < n_cols; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * watDim + j)*3 + k] = right_uchar[(i *width + j + offset)*3 + k];
            }
        }
    /*FIne aggiunta per le dimensioni*/
//    for (int i = 0; i < 480; i++ )    //MODIFICA BENE
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
//            }
//        }
//    unsigned char * recleft = rv.left_rnc_no_occ(squared_right,squared_rDisp ,dim,dim);  //MODIFICA BENE
    unsigned char * recleft = rv.left_rnc_no_occ(squared_right,squared_rDisp ,watDim,watDim); //MODIFICA BENE
    float  **imidft_wat_rec;
//    imidft_wat_rec = AllocIm::AllocImFloat(dim, dim); //MODIFICA BENE
//    cv::Mat square_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);  //MODIFICA BENE
    imidft_wat_rec = AllocIm::AllocImFloat(watDim, watDim);   //MODIFICA BENE
    cv::Mat square_left_mat = cv::Mat::zeros(watDim, watDim, CV_8UC3);    //MODIFICA BENE
    int count=0;
    for (int j = 0; j < watDim; j++)
        for (int i = 0; i < watDim; i++) {

            square_left_mat.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }
    cv::Mat rec_left_mat = cv::Mat::zeros(watDim, watDim, CV_8UC3);
    count=0;
    for (int j = 0; j < watDim; j++)
        for (int i = 0; i < watDim; i++) {

            rec_left_mat.at<Vec3b>(j, i)[0] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[1] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[2] = recleft[count];
            count++;
        }
    for (int j = 0; j < watDim; j++)
        for (int i = 0; i < watDim; i++){
            if ( rec_left_mat.at<Vec3b>(j, i)[0]==0 && rec_left_mat.at<Vec3b>(j, i)[1]==0 && rec_left_mat.at<Vec3b>(j, i)[2]==0){
                rec_left_mat.at<Vec3b>(j, i)[0] = square_left_mat.at<Vec3b>(j, i)[0];
                rec_left_mat.at<Vec3b>(j, i)[1] = square_left_mat.at<Vec3b>(j, i)[1];
                rec_left_mat.at<Vec3b>(j, i)[2] = square_left_mat.at<Vec3b>(j, i)[2];
            }
        }
    unsigned char *squared_marked_left_rec = image_watermarking.insertWatermark(rec_left_mat.data,watDim,watDim,watDim,imidft_wat_rec);

//    computing warped watermark   ********************
    float  **warp_mark;
    warp_mark = AllocIm::AllocImFloat(watDim, watDim);
    for (int i=0; i <watDim; i++)
        for (int j=0;j<watDim;j++)
            warp_mark[i][j] = 0.0;
    unsigned char d = 0;
    for (int i=0; i < n_rows; i++)
        for (int j=0;j<n_cols;j++){
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
    imc2 = AllocIm::AllocImFloat(watDim, watDim);
    imc3 = AllocIm::AllocImFloat(watDim, watDim);
    imr = AllocIm::AllocImByte(watDim, watDim);
    img = AllocIm::AllocImByte(watDim, watDim);
    imb = AllocIm::AllocImByte(watDim, watDim);
    float ** right_lum;
    right_lum = AllocIm::AllocImFloat(watDim, watDim);
    stereo_watermarking::compute_luminance(squared_right,watDim,1,imr,img,imb,right_lum,imc2,imc3);
//   compute marked right lum  ********************
    float ** marked_right_lum = AllocIm::AllocImFloat(watDim, watDim);

    for (int i = 0; i < n_rows; i++ )
        for (int j = 0; j < n_cols; j++) {
            marked_right_lum[i][j] = right_lum[i][j] + warp_mark[i][j];
        }
//    compute image from luminance   ********************
    unsigned char *marked_right = new unsigned char[squared_dim];
    stereo_watermarking::compute_luminance(marked_right,watDim,-1,imr,img,imb,marked_right_lum,imc2,imc3);
// back to normal size *******************
    unsigned char *left_watermarked = new unsigned char [height*width*3];
    left_watermarked = frameL.data;
    for (int i = 0; i < n_rows; i++ )
        for (int j = 0; j < n_cols; j++) {
            for (int k =0; k<3;k++){
                left_watermarked[(i *width + j + offset)*3 + k] = squared_marked_left[(i *watDim  + j)*3 + k];
            }
        }
    count = 0;
    cv::Mat left_wat_mat = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){
            left_wat_mat.at<Vec3b>(j, i) [0] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [1] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [2] = left_watermarked[count]; count++;
        }
    output.push_back(left_wat_mat);
    unsigned char *right_watermarked = new unsigned char [height*width*3];
    right_watermarked = frameR.data;
    for (int i = 0; i < n_rows; i++ )
        for (int j = 0; j < n_cols; j++) {
            for (int k =0; k<3;k++){
                right_watermarked[(i *width + j + offset)*3 + k] = marked_right[(i * watDim + j)*3 + k];
            }
        }
    count = 0;
    cv::Mat right_wat_mat = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){

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
            if ( rcn_left_mat.at<Vec3b>(j, i)[0]==0 && rcn_left_mat.at<Vec3b>(j, i)[1]==0 && rcn_left_mat.at<Vec3b>(j,i)[2]==0){
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
