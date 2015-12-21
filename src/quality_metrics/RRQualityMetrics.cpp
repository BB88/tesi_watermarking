//
// Created by miky on 06/10/15.
#include <iostream>
#include <opencv2/core/core.hpp>
#include <cv.h>
#include <highgui.h>

#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "../dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>

#include "RRQualityMetrics.h"

#include "../utils.h"

//quality metrics
#include "../quality_metrics/quality_metrics.h"

using namespace std;
using namespace cv;
using namespace stereo_watermarking;
using namespace qm;

/**
 * compute_metrics(..)
 *
 * compute the quality metrics of watermarked video
 *
 * @params video: path of the original non-marked stereo video
 * @params wat_video: path of the watermarked stereo video
 * @params disp_video_l: path of the video created with the left-to-right disparity maps
 * @params disp_video_r: path of the video created with the right-to-left disparity maps
 * @params wat_disp_video_l: path of the video created with the left-to-right disparity maps computed after the watermarking process
 * @params wat_disp_video_r: path of the video created with the right-to-left disparity maps computed after the watermarking process
 */
void RRQualityMetrics::compute_metrics(int step, std::string video,  std::string wat_video, std::string disp_video_l, std::string wat_disp_video_l, std::string disp_video_r,  std::string wat_disp_video_r ){

    VideoCapture capV(video.c_str());
    VideoCapture capVw(wat_video.c_str());
    VideoCapture capDl(disp_video_l.c_str());
    VideoCapture capDr(disp_video_r.c_str());
    VideoCapture capDwl(wat_disp_video_l.c_str());
    VideoCapture capDwr(wat_disp_video_r.c_str());

    if (!capV.isOpened())  // check if we succeeded
        return ;
    if (!capVw.isOpened())  // check if we succeeded
        return ;
    if (!capDl.isOpened())  // check if we succeeded
        return ;
    if (!capDwl.isOpened())  // check if we succeeded
        return ;
    if (!capDr.isOpened())  // check if we succeeded
        return ;
    if (!capDwr.isOpened())  // check if we succeeded
        return ;

    cv::Mat ref_left;
    cv::Mat wat_left;
    cv::Mat ref_disp_l;
    cv::Mat wat_disp_l;
    cv::Mat left_edge;
    cv::Mat wat_left_edge;
    cv::Mat depth_edge_l;
    cv::Mat wat_depth_edge_l;

    cv::Mat ref_right;
    cv::Mat wat_right;
    cv::Mat ref_disp_r;
    cv::Mat wat_disp_r;
    cv::Mat right_edge;
    cv::Mat wat_right_edge;
    cv::Mat depth_edge_r;
    cv::Mat wat_depth_edge_r;

    cv::Mat video_stereo;
    cv::Mat video_stereo_wat;

    int first_frame = 0;
    int last_frame = 1800;
    int count=0;
    double* MQcolor_array_l = new double[(last_frame-first_frame)/step];
    double* MQcolor_array_r = new double[(last_frame-first_frame)/step];
    double* MQdisp_array_l = new double[(last_frame-first_frame)/step];
    double* MQdisp_array_r = new double[(last_frame-first_frame)/step];
    ofstream fout_color_l("/home/miky/Scrivania/MQColor_left_gauss3.txt");
    ofstream fout_color_r("/home/miky/Scrivania/MQColor_right_gauss3.txt");
    ofstream fout_disp_l("/home/miky/Scrivania/MQDisp_left_gauss3.txt");
    ofstream fout_disp_r("/home/miky/Scrivania/MQDisp_right_gauss3.txt");
    for (int i = first_frame; i < last_frame; i++) {
        if(i % step == 0){
            capV >> video_stereo;
            capVw >> video_stereo_wat;
            capDl >> ref_disp_l;
            capDwl >> wat_disp_l;
            capDr >> ref_disp_r;
            capDwr >> wat_disp_r;
            video_stereo(Rect(0,0,640,480)).copyTo(ref_left);
            video_stereo(Rect(640,0,640,480)).copyTo(ref_right);
            video_stereo_wat(Rect(0,0,640,480)).copyTo(wat_left);
            video_stereo_wat(Rect(640,0,640,480)).copyTo(wat_right);
            left_edge =  stereo_watermarking::sobel_filtering(ref_left) ;
            right_edge =  stereo_watermarking::sobel_filtering(ref_right) ;
            wat_left_edge = stereo_watermarking::sobel_filtering(wat_left) ;
            wat_right_edge = stereo_watermarking::sobel_filtering(wat_right);
            depth_edge_l =  stereo_watermarking::sobel_filtering(ref_disp_l) ;
            depth_edge_r =  stereo_watermarking::sobel_filtering(ref_disp_r) ;
            wat_depth_edge_l =   stereo_watermarking::sobel_filtering(wat_disp_l) ;
            wat_depth_edge_r =   stereo_watermarking::sobel_filtering(wat_disp_r) ;

//            if(i != 960){
                MQdisp_array_l[i] = qm::MQdepth(ref_disp_l,wat_disp_l,depth_edge_l,wat_depth_edge_l,8,true);
                fout_disp_l << std::fixed << std::setprecision(8) <<MQdisp_array_l[i]<<"\t";
                fout_disp_l<<endl;
                MQcolor_array_l[i] = qm::MQcolor(ref_left,wat_left,depth_edge_l,wat_left_edge,left_edge,8,true);
                fout_color_l << std::fixed << std::setprecision(8) <<MQcolor_array_l[i]<<"\t";
                fout_color_l<<endl;
                MQdisp_array_r[i] = qm::MQdepth(ref_disp_r,wat_disp_r,depth_edge_r,wat_depth_edge_r,8,true);
                fout_disp_r << std::fixed << std::setprecision(8) <<MQdisp_array_r[i]<<"\t";
                fout_disp_r<<endl;
                MQcolor_array_r[i] = qm::MQcolor(ref_right,wat_right,depth_edge_r,wat_right_edge,right_edge,8,true);
                fout_color_r<< std::fixed << std::setprecision(8) <<MQcolor_array_r[i]<<"\t";
                fout_color_r<<endl;
                count++;
//            }
        }
        else {
                capV >> video_stereo;
                capVw >> video_stereo_wat;
        }
    }
    fout_color_l.close();
    fout_color_r.close();
    fout_disp_l.close();
    fout_disp_r.close();
    return;
}