//
// Created by bene on 06/10/15.
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


void RRQualityMetrics::compute_metrics(int step, std::string video,  std::string wat_video, std::string disp_video, std::string wat_disp_video){

    VideoCapture capV(video.c_str());
    VideoCapture capVw(wat_video.c_str());
    VideoCapture capD(disp_video.c_str());
    VideoCapture capDw(wat_disp_video.c_str());

    if (!capV.isOpened())  // check if we succeeded
        return ;
    if (!capVw.isOpened())  // check if we succeeded
        return ;
    if (!capD.isOpened())  // check if we succeeded
        return ;
    if (!capDw.isOpened())  // check if we succeeded
        return ;

    cv::Mat ref_left;
    cv::Mat wat_left;
    cv::Mat ref_disp;
    cv::Mat wat_disp;
    cv::Mat left_edge;
    cv::Mat wat_left_edge;
    cv::Mat depth_edge;
    cv::Mat wat_depth_edge;

    cv::Mat video_stereo;
    cv::Mat video_stereo_wat;


    int first_frame = 0;
    int last_frame = 1800;
    int count=0;
    double* MQcolor_array = new double[(last_frame-first_frame)/step];
    double* MQdisp_array = new double[(last_frame-first_frame)/step];

    for (int i = first_frame; i < last_frame; i++) {

        if(i % step == 0){


            capV >> video_stereo;
            capVw >> video_stereo_wat;

            capD >> ref_disp;
            capDw >> wat_disp;


            video_stereo(Rect(0,0,640,480)).copyTo(ref_left);
            video_stereo_wat(Rect(0,0,640,480)).copyTo(wat_left);
//
//
            left_edge =  stereo_watermarking::sobel_filtering(ref_left, "left_edge") ;

            wat_left_edge = stereo_watermarking::sobel_filtering(wat_left, "wat_left_edge") ;

            depth_edge =  stereo_watermarking::sobel_filtering(ref_disp, "depth_edge") ;

            wat_depth_edge =   stereo_watermarking::sobel_filtering(wat_disp, "wat_depth_edge") ;

            if(i != 660){

                MQdisp_array[i] = qm::MQdepth(ref_disp,wat_disp,depth_edge,wat_depth_edge,8,true);
                MQcolor_array[i] = qm::MQcolor(ref_left,wat_left,depth_edge,wat_left_edge,left_edge,8,true);
                count++;
            }

        }
        else {


            try
            {
//                std::cout << " capV >> video_stereo;" << std::endl;

                capV >> video_stereo;

            } catch (cv::Exception &exc)
            {
                std::cout << "cv exception: " << exc.what() << std::endl;
            } catch(...)
            {
                std::cout << "unknown exception" << std::endl;
            }



            try
            {
//                std::cout << " capVw >> video_stereo_wat;" << std::endl;

                capVw >> video_stereo_wat;

            } catch (cv::Exception &exc)
            {
                std::cout << "cv exception: " << exc.what() << std::endl;
            } catch(...)
            {
                std::cout << "unknown exception" << std::endl;
            }

        }

    }


    stereo_watermarking::writeToFile(MQcolor_array,count,"/home/bene/Scrivania/MQColor.txt");
    stereo_watermarking::writeToFile(MQdisp_array,count,"/home/bene/Scrivania/MQDisp.txt");


}