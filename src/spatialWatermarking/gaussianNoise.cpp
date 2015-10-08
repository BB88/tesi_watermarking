//
// Created by bene on 02/10/15.
//

#include "gaussianNoise.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "../dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>
#include "../disparity_computation/stereo_matching.h"

#include "../right_view_computation/right_view.h"



void spatialWatermarking::gaussianNoiseStereoWatermarking(){

    Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);

    double m_NoiseStdDev=10;
//    double m_NoiseStdDev2=100;

    Mat left_w = left.clone();
    Mat right_w = right.clone();

//    Mat sqr_noise = cv::Mat::zeros(512 ,512 , CV_8UC3);
//    randn(sqr_noise,0,m_NoiseStdDev);
    Mat noise = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);
    randn(noise,0,m_NoiseStdDev);

//    for (int j = 0; j < 480; j++) // 640 - 512 - 1
//        for (int i = 0; i < 512; i++){
//            noise.at<Vec3b>(j, i+127) [0] = sqr_noise.at<Vec3b>(j,i) [0] ;
//            noise.at<Vec3b>(j, i+127) [1] = sqr_noise.at<Vec3b>(j,i) [1] ;
//            noise.at<Vec3b>(j, i+127) [2] = sqr_noise.at<Vec3b>(j,i) [2] ;
//
//        }

//    noise*=0.5; //watermark power

    left_w += noise;
    right_w += noise;


    normalize(left_w, left_w,0, 255, CV_MINMAX, CV_8UC3);
    normalize(right_w, right_w,0, 255, CV_MINMAX, CV_8UC3);

    cv::imshow("left marked", left_w);
    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png", left_w);
//    cv::imshow("right marked", right_w);
    cv::waitKey(0);

//    stereo_watermarking::show_difference(left_w,left,"left sub noise");


    Mat left_correl;
    Mat m1, m2;
    left_w.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);

    matchTemplate(m1, m2, left_correl, CV_TM_CCOEFF_NORMED);

    for (int i = 0; i < left_correl.rows; i++)
    {
//        cout << "row " << i << endl;
        for (int j = 0; j < left_correl.cols; j++)
        {
            cout << "correlation btw left watermarked and watermark " << (left_correl.at<float>(i,j));
        } cout << endl; }

    Mat right_correl;
    right_w.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);

    matchTemplate(m1, m2, right_correl, CV_TM_CCOEFF_NORMED);

    for (int i = 0; i < left_correl.rows; i++)
    {
//        cout << "row " << i << endl;
        for (int j = 0; j < right_correl.cols; j++)
        {
            cout << "correlation btw right with not warped watermarked and watermark " << (right_correl.at<float>(i,j));
        } cout << endl; }

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



    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat warped_mark = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);
    int d;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i< 640; i++){
            d = disp.at<uchar>(j,i);
            if ((i-d)>=0){
              warped_mark.at<Vec3b>(j, i-d) [0] =  noise.at<Vec3b>(j, i) [0];
              warped_mark.at<Vec3b>(j, i-d) [1] =  noise.at<Vec3b>(j, i) [1];
              warped_mark.at<Vec3b>(j, i-d) [2] =  noise.at<Vec3b>(j, i) [2];
            }
        }


    cv::Mat right_warp_w;
    right.copyTo(right_warp_w);
    right_warp_w += warped_mark;

    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/right_warped_marked.png", right_warp_w);
    cv::imshow("right warped marked", right_warp_w);

    cv::waitKey(0);

    Mat right_warped_correl;
    right_warp_w.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);

    matchTemplate(m1, m2, right_warped_correl, CV_TM_CCOEFF_NORMED);

    for (int i = 0; i < right_warped_correl.rows; i++)
    {
//        cout << "row " << i << endl;
        for (int j = 0; j < right_warped_correl.cols; j++)
        {
            cout << "correlation btw right with warped watermark and watermark " << (right_warped_correl.at<float>(i,j));
        } cout << endl; }


    cv::Mat rdisp= imread("/home/bene/Scrivania/Tesi/frame_1.png",CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);

    Right_view rv;
    unsigned char* left_recon = rv.left_rnc(right_warp_w.data, rdisp, occ_right, 640, 480 );

    cv::Mat left_reconstructed = cv::Mat::zeros(480, 640 , CV_8UC3);

    int count = 0;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            left_reconstructed.at<Vec3b>(j,i) [0] = left_recon[count]; count++;
            left_reconstructed.at<Vec3b>(j,i) [1] = left_recon[count]; count++;
            left_reconstructed.at<Vec3b>(j,i) [2] = left_recon[count]; count++;
        }


    cv::imshow("left_reconstructed", left_reconstructed);
    cv::waitKey(0);

    Mat left_rec_correl;
    left_reconstructed.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);

    matchTemplate(m1, m2, left_rec_correl, CV_TM_CCOEFF_NORMED);

    for (int i = 0; i < left_rec_correl.rows; i++)
    {
        for (int j = 0; j < left_rec_correl.cols; j++)
        {
            cout << "correlation btw left reconstructed with watermark and watermark " << (left_rec_correl.at<float>(i,j));
        } cout << endl; }


    cv::Mat synt_view = imread("/home/bene/ClionProjects/tesi_watermarking/img/synth_view_gauss_marked.png", CV_LOAD_IMAGE_COLOR);

    cv::Mat disp_synt = imread("/home/bene/ClionProjects/tesi_watermarking/img/norm_disp_syn.png", CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat squared_occ_synt = cv::Mat::zeros(480, 640, CV_8UC1);
    for (int i=0;i<480;i++)
        for (int j=0;j<640;j++){
            squared_occ_synt.at<uchar>(i,j) = 255;
        }

    unsigned char* left_recon_synt = rv.left_rnc(synt_view.data, disp_synt, squared_occ_synt, 640, 480 );

    cv::Mat left_synt_reconstructed = cv::Mat::zeros(480, 640 , CV_8UC3);

    count = 0;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            left_synt_reconstructed.at<Vec3b>(j,i) [0] = left_recon_synt[count]; count++;
            left_synt_reconstructed.at<Vec3b>(j,i) [1] = left_recon_synt[count]; count++;
            left_synt_reconstructed.at<Vec3b>(j,i) [2] = left_recon_synt[count]; count++;
        }

    cv::imshow("left_synt_reconstructed", left_synt_reconstructed);
    cv::waitKey(0);

    Mat left_synt_rec_correl;
    left_synt_reconstructed.convertTo(m1, CV_32F);
    noise.convertTo(m2, CV_32F);

    matchTemplate(m1, m2, left_synt_rec_correl, CV_TM_CCOEFF_NORMED);

    for (int i = 0; i < left_synt_rec_correl.rows; i++)
    {
        for (int j = 0; j < left_synt_rec_correl.cols; j++)
        {
            cout << "correlation btw left reconstructed from synthetized view and watermark " << (left_synt_rec_correl.at<float>(i,j));
        } cout << endl; }
}