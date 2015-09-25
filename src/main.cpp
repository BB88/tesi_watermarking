#include <iostream>
#include <opencv2/core/core.hpp>
#include "dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <stdint.h>
#include "./disparity_computation/stereo_matching.h"
#include "./disparity_optimization/disp_opt.h"
#include "./disparity_optimization/occlusions_handler.h"
#include "./right_view_computation/right_view.h"
#include "disparity_optimization/disp_opt.h"
#include <limits>
#include <cstddef>
#include <iostream>
#include <fstream>

//includes watermarking
#include "./img_watermarking/watermarking.h"
#include "./img_watermarking/imgwat.h"
#include "./img_watermarking/allocim.h"

//grapfh cuts
#include "./graphcuts/image.h"
#include "./graphcuts/match.h"
#include "./graphcuts/utils.h"
#include "./graphcuts/io_png.h"

//libconfig
#include <libconfig.h++>
#include "./config/config.hpp"

#include "quality_metrics/quality_metrics.h"
#include "utils.h"

#include "img_watermarking/allocim.h"
#include "img_watermarking/fft2d.h"


using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;
using namespace graph_cuts_utils;





int main() {


    /*CONFIG SETTINGS*/

    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();

    int wsize = pars.wsize;
    int tilesize=pars.tilesize;
    float power=pars.power;
    bool clipping=pars.clipping;
    bool flagResyncAll=pars.flagResyncAll;
    int tilelistsize=pars.tilelistsize;


    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();

    bool masking=generalPars.masking;
    std::string passwstr=generalPars.passwstr;
    std::string passwnum=generalPars.passwnum;



    /* RUMORE GAUSSIANO  */

//
//    Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
//    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);
//
//    double m_NoiseStdDev=5;
////    double m_NoiseStdDev2=100;
//
//    Mat left_w = left.clone();
//    Mat right_w = right.clone();
//
////    Mat sqr_noise = cv::Mat::zeros(512 ,512 , CV_8UC3);
////    randn(sqr_noise,0,m_NoiseStdDev);
//    Mat noise = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);
//    randn(noise,0,m_NoiseStdDev);
//
////    for (int j = 0; j < 480; j++) // 640 - 512 - 1
////        for (int i = 0; i < 512; i++){
////            noise.at<Vec3b>(j, i+127) [0] = sqr_noise.at<Vec3b>(j,i) [0] ;
////            noise.at<Vec3b>(j, i+127) [1] = sqr_noise.at<Vec3b>(j,i) [1] ;
////            noise.at<Vec3b>(j, i+127) [2] = sqr_noise.at<Vec3b>(j,i) [2] ;
////
////        }
//
////    noise*=0.5; //watermark power
//
//    left_w += noise;
//    right_w += noise;
//
//
//    normalize(left_w, left_w,0, 255, CV_MINMAX, CV_8UC3);
//    normalize(right_w, right_w,0, 255, CV_MINMAX, CV_8UC3);
//
//    cv::imshow("left marked", left_w);
//    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png", left_w);
////    cv::imshow("right marked", right_w);
//    cv::waitKey(0);
//
////    stereo_watermarking::show_difference(left_w,left,"left sub noise");
//
//
//    Mat left_correl;
//    Mat m1, m2;
//    left_w.convertTo(m1, CV_32F);
//    noise.convertTo(m2, CV_32F);
//
//    matchTemplate(m1, m2, left_correl, CV_TM_CCOEFF_NORMED);
//
//    for (int i = 0; i < left_correl.rows; i++)
//    {
////        cout << "row " << i << endl;
//        for (int j = 0; j < left_correl.cols; j++)
//        {
//            cout << "correlation btw left watermarked and watermark " << (left_correl.at<float>(i,j));
//        } cout << endl; }
//
//    Mat right_correl;
//    right_w.convertTo(m1, CV_32F);
//    noise.convertTo(m2, CV_32F);
//
//    matchTemplate(m1, m2, right_correl, CV_TM_CCOEFF_NORMED);
//
//    for (int i = 0; i < left_correl.rows; i++)
//    {
////        cout << "row " << i << endl;
//        for (int j = 0; j < right_correl.cols; j++)
//        {
//            cout << "correlation btw right with not warped watermarked and watermark " << (right_correl.at<float>(i,j));
//        } cout << endl; }

//    Mat right_correl;
//    right_w.convertTo(m1, CV_32F);
//    noise.convertTo(m2, CV_32F);
//
//    matchTemplate(m1, m2, right_correl, CV_TM_CCOEFF_NORMED);
//
//    for (int i = 0; i < left_correl.rows; i++)
//    {
////        cout << "row " << i << endl;
//        for (int j = 0; j < right_correl.cols; j++)
//        {
//            cout << "correlation btw right with not warped watermarked and watermark " << (right_correl.at<float>(i,j));
//        } cout << endl; }
//
//
//
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat warped_mark = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);
//    int d;
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i< 640; i++){
//            d = disp.at<uchar>(j,i);
//            if ((i-d)>=0){
//              warped_mark.at<Vec3b>(j, i-d) [0] =  noise.at<Vec3b>(j, i) [0];
//              warped_mark.at<Vec3b>(j, i-d) [1] =  noise.at<Vec3b>(j, i) [1];
//              warped_mark.at<Vec3b>(j, i-d) [2] =  noise.at<Vec3b>(j, i) [2];
//            }
//        }
//
//
//    cv::Mat right_warp_w;
//    right.copyTo(right_warp_w);
//    right_warp_w += warped_mark;
//    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/right_warped_marked.png", right_warp_w);
//    cv::imshow("right warped marked", right_warp_w);
//
//    cv::waitKey(0);
//
//    Mat right_warped_correl;
//    right_warp_w.convertTo(m1, CV_32F);
//    noise.convertTo(m2, CV_32F);
//
//    matchTemplate(m1, m2, right_warped_correl, CV_TM_CCOEFF_NORMED);
//
//    for (int i = 0; i < right_warped_correl.rows; i++)
//    {
////        cout << "row " << i << endl;
//        for (int j = 0; j < right_warped_correl.cols; j++)
//        {
//            cout << "correlation btw right with warped watermark and watermark " << (right_warped_correl.at<float>(i,j));
//        } cout << endl; }
//
//
//    cv::Mat rdisp= imread("/home/bene/Scrivania/Tesi/frame_1.png",CV_LOAD_IMAGE_GRAYSCALE);
//    Right_view rv;
//    rv.left_uchar_reconstruction(right_warp_w.data,rdisp.data,480,640);
//    cv::Mat left_reconstructed = imread("/home/bene/ClionProjects/tesi_watermarking/img/left_reconstructed.png");
//
//    Mat left_rec_correl;
//    left_reconstructed.convertTo(m1, CV_32F);
//    noise.convertTo(m2, CV_32F);
//
//    matchTemplate(m1, m2, left_rec_correl, CV_TM_CCOEFF_NORMED);
//
//    for (int i = 0; i < left_rec_correl.rows; i++)
//    {
//        for (int j = 0; j < left_rec_correl.cols; j++)
//        {
//            cout << "correlation btw left reconstructed with watermark and watermark " << (left_rec_correl.at<float>(i,j));
//        } cout << endl; }

    //FINE RUMORE GAUSSIANO




    /*watermarking dft 256*256 + MSE    -> controllare watermarking.cpp : 256 e diag */


    cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
    unsigned char *left_uchar = left.data;
    int squared_dim = 256 * 256 *3;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    int nc = 640;
    int nc_s = 256;
    int offset = 127;
    for (int i = 0; i < nc_s; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }
    int dim = 256;
    Watermarking image_watermarking;
//    random binary watermark   ********************
    int watermark[64];
    for (int i = 0; i < 64; i++){
        int b = rand() % 2;
        watermark[i]=b;
    }
    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
    image_watermarking.setPassword(passwstr,passwnum);
    float  **imidft_wat;
    imidft_wat = AllocIm::AllocImFloat(256, 256);
    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,256,256,imidft_wat,false);
//     generate watermark's magnitude and phase   ********************
    stereo_watermarking::show_floatImage(imidft_wat,dim,dim,"mark");
    double  **imdft_mark;
    double  **imdftfase_mark;
    imdft_mark = AllocIm::AllocImDouble(dim, dim);
    imdftfase_mark = AllocIm::AllocImDouble(dim, dim);
    FFT2D::dft2d(imidft_wat, imdft_mark, imdftfase_mark, dim, dim);





    stereo_watermarking::writeMatToFile(imdft_mark,dim,"/home/bene/Scrivania/wat_mag.txt");
    stereo_watermarking::writeMatToFile(imdftfase_mark,dim,"/home/bene/Scrivania/wat_phase.txt");
//     generate squared disp and occ map   ********************
    cv::Mat disp_left = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_left = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_left.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat squared_lDisp = cv::Mat::zeros(256, 256, CV_8UC1);
    cv::Mat squared_lOcc = cv::Mat::zeros(256, 256, CV_8UC1);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++){
            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
            squared_lOcc.at<uchar>(i,j) = occ_left.at<uchar>(i,j+offset);
        }
//    computing warped watermark   ********************
    float  **warp_mark;
    warp_mark = AllocIm::AllocImFloat(256, 256);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            warp_mark[i][j] = 0.0;
    unsigned char d = 0;
    unsigned char occ = 0;
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++){
            d = squared_lDisp.at<uchar>(i,j);
            occ = squared_lOcc.at<uchar>(i,j);
            int diff = j-static_cast<int>(d);
            if(static_cast<int>(occ)!=0 && diff>=0)
                warp_mark[i][j-static_cast<int>(d)] = imidft_wat[i][j];
        }
    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
    double  **imdft_warp_mark;
    double  **imdftfase_warp_mark;
    imdft_warp_mark = AllocIm::AllocImDouble(dim, dim);
    imdftfase_warp_mark = AllocIm::AllocImDouble(dim, dim);
    FFT2D::dft2d(warp_mark, imdft_warp_mark, imdftfase_warp_mark, dim, dim);
    stereo_watermarking::writeMatToFile(imdft_warp_mark,dim,"/home/bene/Scrivania/warp_wat_mag.txt");
    stereo_watermarking::writeMatToFile(imdftfase_warp_mark,dim,"/home/bene/Scrivania/warp_wat_phase.txt");
//    marking right view with warped mark   ********************
    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
    unsigned char *right_uchar = right.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    for (int i = 0; i < nc_s; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }
//    dft watermarking   ********************
//    unsigned char *marked_right= image_watermarking.insertWatermark(squared_right,256,256,warp_mark,true);
//    spacial watermarking:compute right luminance   ********************
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
    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
//   compute marked right lum  ********************
    float ** marked_right_lum = AllocIm::AllocImFloat(dim, dim);
    for (int i = 0; i < nc_s; i ++ )
        for (int j = 0; j < nc_s; j++) {
            marked_right_lum[i][j] = right_lum[i][j] + power*warp_mark[i][j];
        }
    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");
//    insert wat in left view
    unsigned char **imrl;
    unsigned char **imgl;
    unsigned char **imbl;
    float **imc2l;
    float **imc3l;
    imc2l = AllocIm::AllocImFloat(dim, dim);
    imc3l = AllocIm::AllocImFloat(dim, dim);
    imrl = AllocIm::AllocImByte(dim, dim);
    imgl = AllocIm::AllocImByte(dim, dim);
    imbl = AllocIm::AllocImByte(dim, dim);
    float ** left_lum;
    left_lum = AllocIm::AllocImFloat(dim, dim);
    stereo_watermarking::compute_luminance(squared_left,dim,1,imrl,imgl,imbl,left_lum,imc2l,imc3l);
    float ** marked_left_lum = AllocIm::AllocImFloat(dim, dim);
    for (int i = 0; i < nc_s; i ++ )
        for (int j = 0; j < nc_s; j++) {
            marked_left_lum[i][j] = left_lum[i][j] + power*imidft_wat[i][j];
        }
    stereo_watermarking::show_floatImage(marked_left_lum,dim,dim,"marked_left_lum");
//    compute image from luminance   ********************
    unsigned char *marked_right = new unsigned char[squared_dim];
    stereo_watermarking::compute_luminance(marked_right,dim,-1,imr,img,imb,marked_right_lum,imc2,imc3);
    stereo_watermarking::show_ucharImage(marked_right,dim,dim,"marked_right");
//    left view reconstruction   ********************
    cv::Mat disp_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat squared_rDisp = cv::Mat::zeros(256, 256, CV_8UC1);
    cv::Mat squared_rOcc = cv::Mat::zeros(256, 256, CV_8UC1);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
            squared_rOcc.at<uchar>(i,j) = occ_right.at<uchar>(i,j+offset);
        }
    Right_view rv;
    unsigned char * rcn_squared_left = rv.left_rnc(marked_right,squared_rDisp,squared_rOcc,dim,dim);

//    imshow   ********************
//
//    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
//    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
//    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
//    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
//    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");
//    stereo_watermarking::show_ucharImage(marked_right,256,256,"marked_right");
    stereo_watermarking::show_ucharImage(rcn_squared_left,256,256,"rcn_squared_left");


//     detection   ********************
    bool left_det = image_watermarking.extractWatermark(squared_marked_left,dim,dim);
    bool right_det = image_watermarking.extractWatermark(marked_right,dim,dim);
    bool rcnleft_det = image_watermarking.extractWatermark(rcn_squared_left,dim,dim);
    cout<<" left_det    "<<left_det <<endl;
    cout<<" right_det   "<< right_det<<endl;
    cout<<"rcnleft_det  "<< rcnleft_det<<endl;


//    START COEFFICIENT ANALYSIS:  saving dft left coefficient   ********************
//    double *coeff_left = image_watermarking.getCoeff_dft();
//    int coeff_num = image_watermarking.getCoeff_number();
//    double *wat = new double[coeff_num];
//    wat = image_watermarking.getFinal_mark();
//    stereo_watermarking::writeToFile(wat,coeff_num,"/home/bene/Scrivania/wat.txt");
//    stereo_watermarking::writeToFile(coeff_left,coeff_num,"/home/bene/Scrivania/diag_coeff_left.txt");
//    decoding   ********************
//    bool left_detection = image_watermarking.extractWatermark(squared_marked_left, 256, 256);
//    cout<< "left_detection:    " << left_detection <<endl;
//    saving marked dft left coefficient   ********************
//    double *marked_coeff_left = image_watermarking.getMarked_coeff();
//    for (int i = 0; i < coeff_num; i++) {
//        marked_coeff_left[i] = marked_coeff_left[i]/coeff_left[i];
//    }
//    double *retrieve_left_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);
//    stereo_watermarking::writeMatToFile(retrieve_left_wat,coeff_num,"/home/bene/Scrivania/Tesi/retrieve_left_wat.txt");
//    stereo_watermarking::writeMatToFile(marked_coeff_left,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_left.txt");
//    stereo_watermarking::similarity_graph(100,coeff_num,wat);
//    constructing SQUARED RIGHT to compute dft analysis   ********************
//    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
//    unsigned char *right_uchar = right.data;
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *squared_right =  new unsigned char[squared_dim];
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
//    for (int i = 0; i < nc_s; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset - d_val)*3 + k];
//            }
//        }
//    unsigned char *squared_dft_marked_right = image_watermarking.insertWatermark(squared_right,256,256);
////    saving dft right coefficient   ********************
//    double *coeff_right = image_watermarking.getCoeff_dft();
//    stereo_watermarking::writeToFile(coeff_right,coeff_num,"/home/bene/Scrivania/diag_coeff_right.txt");
//    spatial extraction of the watermark   ********************


//    double *squared_mark =  new double[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark[i] = (double)squared_marked_left[i] - (double)squared_left[i];
//    }
//    unsigned char* mark_uchar = new unsigned char[squared_dim];
//    for (int i=0; i<squared_dim;i++ ){
//       mark_uchar[i] = (unsigned char) squared_mark[i];
//    }
//    stereo_watermarking::compute_magnitude_phase(mark_uchar,256,"mark");



//    double *retrieve_right_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);  // da modificare gli input
//    stereo_watermarking::writeMatToFile(marked_coeff_rec_left,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_rec_left.txt");

//    similarity   ********************
//    stereo_watermarking::similarity_measures(wat, wat, coeff_num,"inserted watermak", "inserted watermak");

//    similarity   ********************
//    double threshold = stereo_watermarking::threshold_computation(coeff_left, coeff_num, power);
//    double threshold = stereo_watermarking::threshold_computation(coeff_right, coeff_num, power);
//    cout<< "threshold:  "<<threshold<<endl;
//    stereo_watermarking::similarity_measures(wat, coeff_left, coeff_num,"inserted watermak", "coeff_left");
//    float correlation = stereo_watermarking::similarity_measures(wat, marked_coeff_left, coeff_num,"inserted watermak", "marked_coeff_left");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_right, coeff_num,"inserted watermak", "marked_coeff_right");
//    stereo_watermarking::similarity_measures(warp_mark_coeff, marked_coeff_right,coeff_num,"warp_mark_coeff", "marked_coeff_right");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_rec_left, coeff_num,"inserted watermak", "marked_coeff_rec_left");



//    stereo_watermarking::compute_coeff_function(squared_left,256,"coeff_left");
//    stereo_watermarking::compute_coeff_function(squared_right,256,"coeff_right");
//    stereo_watermarking::compute_coeff_function(squared_mark_uchar,256,"coeff_mark");
//    stereo_watermarking::compute_coeff_function(squared_warped_mark,256,"coeff_warped_mark");
//    stereo_watermarking::compute_coeff_function(squared_marked_left,256,"coeff_marked_left");
//    stereo_watermarking::compute_coeff_function(squared_marked_right,256,"coeff_marked_right");
//    stereo_watermarking::compute_coeff_function(squared_left_ric,256,"coeff_left_ric");




//    imshow   ********************
//    stereo_watermarking::show_ucharImage(squared_left, 256, 256, "squared left");
//    stereo_watermarking::show_ucharImage(mark_uchar, 256, 256, "mark_uchar");
//    stereo_watermarking::show_doubleImage(squared_mark, 256, 256, "squared_mark");
//    stereo_watermarking::show_ucharImage(squared_marked_left, 256, 256, "squared marked left");
//    stereo_watermarking::show_doubleImage(squared_mark, 256, 256, "squared_mark");
//    stereo_watermarking::show_ucharImage(squared_mark_uchar, 256, 256, "squared_mark_uchar");
//
//    stereo_watermarking::show_doubleImage(warped_mark, 640, 480, "mark_warped");
//    stereo_watermarking::show_ucharImage(squared_right, 256, 256, "squared right");
//    stereo_watermarking::show_ucharImage(marked_right, 640, 480, "marked_right");
//    stereo_watermarking::show_ucharImage(squared_marked_right, 256, 256, "squared_marked_right");
//    stereo_watermarking::show_ucharImage(squared_left_ric, 256, 256, "squared_left_ric");

//    show squared mark from reconstructed left   ********************
//    double *squared_mark_from_rec_left =  new double[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark_from_rec_left[i] = (double)squared_left[i] - (double)squared_left_ric[i]  ;
//    }
//    stereo_watermarking::show_doubleImage(squared_mark_from_rec_left, 256, 256, "squared_mark_from_rec_left");

//     dft check   ********************
/*
    stereo_watermarking::dft_comparison(squared_left,squared_right,256,"sinistra","destra");

    stereo_watermarking::dft_comparison(squared_left,squared_marked_left,256,"sinistra","sinistra_marchiata");

    stereo_watermarking::dft_comparison(right_squared.data,squared_marked_right,256,"destra", "destra_marchiata");

    stereo_watermarking::dft_comparison(squared_left, squared_left_ric,256,"sinistra" , "sinistra_marchiata_ricostruita");

    stereo_watermarking::dft_comparison(squared_marked_left, squared_left_ric,256,"sinistra_marchiata" , "sinistra_marchiata_ricostruita");
*/



   /* watermarking dft 512*512   -> controllare watermarking.cpp : 512 e diag */

//    cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
//    unsigned char *left_uchar = left.data;
//    int squared_dim = 512 * 512 *3;
//    unsigned char *squared_left =  new unsigned char[squared_dim];
//    int nc = 640 *3;
//    int nc_q = 512*3;
//    int index = 127 * 3;
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < nc_q; j++)
//            squared_left[(i * nc_q)+ j] = left_uchar[(i * nc) + (j + index)];
//    int count = 480 * 512 *3;
//    for (int i = count; i< squared_dim; i ++)
//        squared_left[i] = (unsigned char)0;
//
//    cv::Mat left_squared = cv::Mat::zeros(512, 512, CV_8UC3);
//    count = 0;
//    for (int j = 0; j < 512; j++)
//        for (int i = 0; i < 512; i++){
//            left_squared.at<Vec3b>(j,i) [0] = squared_left[count]; count++;
//            left_squared.at<Vec3b>(j,i) [1] = squared_left[count]; count++;
//            left_squared.at<Vec3b>(j,i) [2] = squared_left[count]; count++;
//        }
////
////    imshow("left_squared", left_squared);
////    waitKey(0);  /*left riprova: si vede*/
//    Watermarking image_watermarking;
////    //random binary watermark
//    int watermark[64];
//    for (int i = 0; i < 64; i++){
//        int b = rand() % 2;
//        watermark[i]=b;
//    }
//    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
//    image_watermarking.setPassword(passwstr,passwnum);
//    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,512,512);
//
//    cv::Mat left_squared_marked = cv::Mat::zeros(512, 512, CV_8UC3);
//    count = 0;
//    for (int j = 0; j < 512; j++)
//        for (int i = 0; i < 512; i++){
//            left_squared_marked.at<Vec3b>(j,i) [0] = squared_marked_left[count]; count++;
//            left_squared_marked.at<Vec3b>(j,i) [1] = squared_marked_left[count]; count++;
//            left_squared_marked.at<Vec3b>(j,i) [2] = squared_marked_left[count]; count++;
//        }
//    cv::Mat left_marked = cv::Mat::zeros(480, 640, CV_8UC3);
//    left.copyTo(left_marked);
//    count = 0;
//    for (int j = 0; j < 512; j++)
//        for (int i = 0; i < 512; i++){
//            left_marked.at<Vec3b>(j,i+127) [0] = left_squared_marked.at<Vec3b>(j,i) [0];
//            left_marked.at<Vec3b>(j,i+127) [1] = left_squared_marked.at<Vec3b>(j,i) [1];
//            left_marked.at<Vec3b>(j,i+127) [2] = left_squared_marked.at<Vec3b>(j,i) [2];
//
//        }
////    imshow("left_squared_marked", left_marked);
////    waitKey(0);   /*left riprova: si vede*/
//    unsigned char *squared_mark =  new unsigned char[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark[i] = squared_marked_left[i] - squared_left[i];
//    }
//  cv::Mat mark_squared = cv::Mat::zeros(512, 512, CV_8UC3);
//    count = 0;
//    for (int j = 0; j < 512; j++)
//        for (int i = 0; i < 512; i++){
//            mark_squared.at<Vec3b>(j,i) [0] = squared_mark[count]; count++;
//            mark_squared.at<Vec3b>(j,i) [1] = squared_mark[count]; count++;
//            mark_squared.at<Vec3b>(j,i) [2] = squared_mark[count]; count++;
//        }
///*
//    imshow("mark_squared", mark_squared);
//    waitKey(0);
//    */  /*marchio riprova: si vede*/
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *disp_uchar = disp.data;
//    cv::Mat occlusion = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *occ_uchar = occlusion.data;
//    int rect_dim = 480*640*3;
//    unsigned char *warped_mark = new unsigned char[rect_dim];
//    for (int i = 0; i< rect_dim; i ++)
//        warped_mark[i] = (unsigned char)0;
//    unsigned char d = 0;
//    unsigned char occ = 0;
//    int new_index = 127;
//    for (int i=0; i <480;i ++)
//        for(int j =0; j<512*3; j+=3){
//            d = disp_uchar[(i*640)+((j/3) + new_index)];
//            occ = occ_uchar[(i*640)+((j/3) + new_index)];
//            if(static_cast<unsigned>(occ)!= 0){
//              warped_mark[ (i*nc) + (index + j + 0 - static_cast<unsigned>(d)*3)] = squared_mark[ (i*nc_q) + (j + 0)];
//              warped_mark[ (i*nc) + (index + j + 1 - static_cast<unsigned>(d)*3)] = squared_mark[ (i*nc_q) + (j + 1)];
//              warped_mark[ (i*nc) + (index + j + 2 - static_cast<unsigned>(d)*3)] = squared_mark[ (i*nc_q) + (j + 2)];
//            }
//        }
//    cv::Mat mark_warped = cv::Mat::zeros(480, 640, CV_8UC3);
//    count = 0;
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 640; i++){
//            mark_warped.at<Vec3b>(j,i) [0] = warped_mark[count]; count++;
//            mark_warped.at<Vec3b>(j,i) [1] = warped_mark[count]; count++;
//            mark_warped.at<Vec3b>(j,i) [2] = warped_mark[count]; count++;
//        }
//    /*  imshow("mark_warped", mark_warped);
//      waitKey(0); */ /*marchio riprova: si vede*/
//    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);
//    unsigned char *right_uchar = right.data;
//    unsigned char *marked_right = new unsigned char[rect_dim];
//    for (int i=0;i<rect_dim;i++){
//        marked_right[i] = right_uchar[i] + warped_mark[i];
//    }
//    cv::Mat right_marked = cv::Mat::zeros(480, 640, CV_8UC3);
//    count = 0;
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 640; i++){
//            right_marked.at<Vec3b>(j,i) [0] = marked_right[count]; count++;
//            right_marked.at<Vec3b>(j,i) [1] = marked_right[count]; count++;
//            right_marked.at<Vec3b>(j,i) [2] = marked_right[count]; count++;
//        }
//    /* imshow("right_marked", right_marked);
//     waitKey(0); */ /*marchio shiftato riprova: si vede*/
//    cv::Mat right_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *right_disp_uchar = right_disp.data;
//    cv::Mat occ_map_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//    Right_view rv;
//    rv.left_uchar_reconstruction(marked_right, right_disp_uchar, occ_map_right.data,640,480);
//    cv::Mat left_reconstructed = imread("/home/bene/ClionProjects/tesi_watermarking//img/left_reconstructed_uchar.png",CV_LOAD_IMAGE_COLOR);



//    /*GENERAZIONE NUVOLA 3D*/
//
//    int frame_num=0; //serve per prendere i parametri dal file di testo ma per ora usiamo sempre il frame 0
//    cv::Mat nkz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::generatePointCloud(nkz_disp,left,right,frame_num);

//
//    /*FINE GENERAZIONE NUVOLA 3D*/



    /*decodifica di immagini 512*512*/

//    cv::Mat new_image_to_dec = cv::Mat::zeros(512, 512, CV_8UC3);
//
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 512; i++){
//            new_image_to_dec.at<Vec3b>(j,i) [0] = left_reconstructed.at<Vec3b>(j,i+new_index) [0];
//            new_image_to_dec.at<Vec3b>(j,i) [1] = left_reconstructed.at<Vec3b>(j,i+new_index) [1];
//            new_image_to_dec.at<Vec3b>(j,i) [2] = left_reconstructed.at<Vec3b>(j,i+new_index) [2];
//        }
//
//    cv::imshow("Left to dec", new_image_to_dec);
//     waitKey(0);
//
////    stereo_watermarking::show_difference(image_to_mark,left,"left marked");
////    stereo_watermarking::show_difference(right_new,right ,"right marked");
////    stereo_watermarking::show_difference(left,left_reconstructed,"left rec");
//
//
//    unsigned char *mark = new unsigned char[480*640*3];
//    for (int i = 0; i< 480*640*3; i ++)
//        mark[i] = (unsigned char)0;
//
//
//    for (int i=0; i <480;i ++)
//        for(int j =0; j<512*3; j+=3){
//            mark[ (i*nc) + (index + j + 0)] = squared_mark[ (i*nc_q) + (j + 0)];
//            mark[ (i*nc) + (index + j + 1)] = squared_mark[ (i*nc_q) + (j + 1)];
//            mark[ (i*nc) + (index + j + 2)] = squared_mark[ (i*nc_q) + (j + 2)];
//        }
//    unsigned char *left_w_uchar = left_w.data;
//    unsigned char *marked_new_left = new unsigned char[480*640*3];
//
//    bool wat = image_watermarking.extractWatermark(squared_marked_left, 512, 512);
//    cout<<wat;
//
//    for (int i=0;i<480*640*3;i++){
//        marked_new_left[i] = left_w_uchar[i] + mark[i];
//    }
//
//    unsigned char *squared_new_left_to_dec = new unsigned char[512*512*3];;
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < nc_q; j++)
//            squared_new_left_to_dec[(i * nc_q)+ j] = marked_new_left[(i * nc) + (j + index)];
//
//
//    bool wat = image_watermarking.extractWatermark(squared_new_left_to_dec, 512, 512);
//    cout<<wat;


    /*  kz_disp PARAMETERS */
/*

     *
     * lambda = 15.8
     * k = 79.12
     * dispMin dispMax = -77 -19

*/


//    /* GRAPH CUTS DISPARITY COMPUTATION*/
//
//    std::string img1_path =  "/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png";
//    std::string img2_path =  "/home/bene/ClionProjects/tesi_watermarking/img/right_warped_marked.png";
//
//
//    Match::Parameters params = { // Default parameters
//            Match::Parameters::L2, 1, // dataCost, denominator
//            8, -1, -1, // edgeThresh, lambda1, lambda2 (smoothness cost)
//            -1,        // K (occlusion cost)
//            4, false   // maxIter, bRandomizeEveryIteration
//    };
//    float K=-1, lambda=-1, lambda1=-1, lambda2=-1;
//    params.dataCost = Match::Parameters::L1;
////      params.dataCost = Match::Parameters::L2;
//
//    GeneralImage im1 = (GeneralImage)imLoad(IMAGE_GRAY, img1_path.c_str());
//    GeneralImage im2 = (GeneralImage)imLoad(IMAGE_GRAY, img2_path.c_str());
//    bool color = false;
//    if(graph_cuts_utils::isGray((RGBImage)im1) && graph_cuts_utils::isGray((RGBImage)im2)) {
//        color=false;
//        graph_cuts_utils::convert_gray(im1);
//        graph_cuts_utils::convert_gray(im2);
//    }
//
//    Match m(im1, im2, color);
////
////////    // Disparity
//    int dMin=-77, dMax=-19;
////
//    m.SetDispRange(dMin, dMax);
//
//    time_t seed = time(NULL);
//    srand((unsigned int)seed);
//
//    graph_cuts_utils::fix_parameters(m, params, K, lambda, lambda1, lambda2);
//
//    m.KZ2();
//
////        m.SaveXLeft(argv[5]);
//
//    m.SaveScaledXLeft("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_wat.png", false);
//
////    cv::Mat disp = imread("/home/bene/Scrivania/disp.png");
//    imshow("kz disp",disp);
//    waitKey(0);




    /*STEP 2: FILTER DISPARITY (OUTPUT OF KZ)*/

/*
    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/kz_disp.png");
    if (kz_disp.rows == 0){
        cout << "Empty image";
    } else {
        Disp_opt dp;
        dp.disparity_filtering(kz_disp);
    }
*/

    // path clion /home/bene/ClionProjects/tesi_watermarking/img/
    // path Scrivania /home/bene/Scrivania/

    /*STEP 3: NORMALIZE DISPARITY (OUTPUT OF KZ)*/

//    cv::Mat nkz_disp;
//    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_wat.png", CV_LOAD_IMAGE_GRAYSCALE);
//    if (kz_disp.rows == 0){
//        cout << "Empty image";
//    } else {
//        Disp_opt dp;
//        dp.disparity_normalization(kz_disp, nkz_disp);
//    }
//
//    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/norm_disp_from_wat.png",nkz_disp);


    /* QUALITY METRICS*/


//    cv::Mat disp_kz = imread( "/home/bene/ClionProjects/tesi_watermarking/img/nkz_right_dim_disp.png",CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::sobel_filtering(disp_kz,"sobel_disp");
//    cv::Mat sobel_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_disp.png",CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat disp_kz_wat = imread(  "/home/bene/ClionProjects/tesi_watermarking/img/norm_disp_from_wat.png",CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::sobel_filtering(disp_kz_wat,"sobel_disp_wat");
//    cv::Mat sobel_disp_wat  = imread( "/home/bene/ClionProjects/tesi_watermarking/img/sobel_disp_wat.png",CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat left_marked = imread("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png", CV_LOAD_IMAGE_COLOR);
//    stereo_watermarking::sobel_filtering(left_marked,"sobel_left_w");
//    Mat sobel_left_w = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_left_w.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat left2 = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
//    stereo_watermarking::sobel_filtering(left2,"sobel_left");
//    Mat sobel_left = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//
////    char* f1="/home/bene/ClionProjects/tesi_watermarking/img/left.png";
////    char* f2="/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png";
////
////    qm::compute_quality_metrics(f1,f2,8);
//
//    double mqdepth = qm::MQdepth(disp_kz, disp_kz_wat,sobel_disp,sobel_disp_wat , 8,false);
//    cout << "MQdepth : " << mqdepth << endl;
//
//    double mqcolor = qm::MQcolor(left2,left_marked, sobel_disp, sobel_left_w, sobel_left,8,false);
//    cout << "MQcolor : " << mqcolor << endl;



    /*ENHANCING OCCLUSIONS*/

/*
    cv::Mat f_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", CV_LOAD_IMAGE_COLOR);
    Disp_opt dp;
    dp.occlusions_enhancing(f_disp);
*/


    return 0;

}



//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf