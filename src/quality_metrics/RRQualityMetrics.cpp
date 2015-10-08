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


void RRQualityMetrics::compute_metrics(){


    cv::Mat ref_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/left.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat wat_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/left_watermarked.png", CV_LOAD_IMAGE_COLOR);

    cv::Mat ref_disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_left_to_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat wat_disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_left_watermarked_to_right_watermarked.png", CV_LOAD_IMAGE_GRAYSCALE);

    stereo_watermarking::sobel_filtering(ref_left, "left_edge") ;
    cv::Mat left_edge = imread("/home/miky/ClionProjects/tesi_watermarking/img/left_edge.png", CV_LOAD_IMAGE_COLOR);

    stereo_watermarking::sobel_filtering(wat_left, "wat_left_edge") ;
    cv::Mat wat_left_edge = imread("/home/miky/ClionProjects/tesi_watermarking/img/wat_left_edge.png", CV_LOAD_IMAGE_COLOR);

    stereo_watermarking::sobel_filtering(ref_disp, "depth_edge") ;
    cv::Mat depth_edge = imread("/home/miky/ClionProjects/tesi_watermarking/img/depth_edge.png", CV_LOAD_IMAGE_COLOR);

    stereo_watermarking::sobel_filtering(wat_disp, "wat_depth_edge") ;
    cv::Mat wat_depth_edge = imread("/home/miky/ClionProjects/tesi_watermarking/img/wat_depth_edge.png", CV_LOAD_IMAGE_COLOR);

    qm::MQcolor(ref_left,wat_left,depth_edge,wat_left_edge,left_edge,8,true);
    qm::MQdepth(ref_disp,wat_disp,depth_edge,wat_depth_edge,8,true);


}