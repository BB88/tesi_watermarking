//
// Created by miky on 26/08/15.
//
#include <opencv/cv.h>
#include <opencv/highgui.h>

#ifndef TESI_WATERMARKING_QUALITY_METRICS_H
#define TESI_WATERMARKING_QUALITY_METRICS_H

#endif //TESI_WATERMARKING_QUALITY_METRICS_H

using namespace cv;
using namespace std;

namespace qm{
    double sigma(Mat & m, int i, int j, int block_size);
    double cov(Mat & m1, Mat & m2, int i, int j, int block_size);
    double eqm(Mat & img1, Mat & img2);
    double color_eqm(Mat & img1, Mat & img2);
    double psnr(Mat & img_src, Mat & img_compressed,int blocksize);
    double video_psnr(Mat & img_src, Mat & img_compressed);
    double avg_psnr(std::string origVideo,std::string compressVideo );
    double ssim(Mat & img_src, Mat & img_compressed, int block_size, bool show_progress );
    double MQdepth(Mat &depth_original, Mat &depth_wat, Mat & depth_edge, Mat & depth_w_edge, int block_size, bool show_progress );
    double MQcolor(Mat &color_original, Mat &color_wat, Mat & depth_edge, Mat &color_w_edge,Mat & color_edge, int block_size, bool show_progress );
    void compute_quality_metrics(char * file1, char * file2, int block_size);
}