#include <iostream>
#include <opencv2/core/core.hpp>
#include "dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>
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


using namespace std;
using namespace cv;
using namespace cv::datasets;

int main() {


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


    /*STEP 3: NORMALIZE DISPARITY (OUTPUT OF KZ)*/

/*

    cv::Mat fg_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/fg_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (fg_disp.rows == 0){
        cout << "Empty image";
    } else {
        Disp_opt dp;
        dp.disparity_normalization(fg_disp);
    }

*/


    /*STEP 4: RECONSTRUCT RIGHT VIEW*/


    cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking//img/l.png",
                          CV_LOAD_IMAGE_COLOR);
    // our disp
    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/n_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    // ground truth
/*    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/left/frame_1.png",
                          CV_LOAD_IMAGE_GRAYSCALE);*/
    Right_view rv;
    rv.right_reconstruction(left, disp);





    /*ENHANCING OCCLUSIONS*/

/*
    cv::Mat f_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", CV_LOAD_IMAGE_COLOR);
    Disp_opt dp;
    dp.occlusions_enhancing(f_disp);

*/







/*


    cv::Mat occluded = imread("/home/miky/ClionProjects/tesi_watermarking/img/filtered_bw.png");
    cv::Mat occluded_gt = imread("/home/miky/Scrivania/tsukuba_occlusion_L_00001.png");

    cout << occlusions_handler::getSimilarity(occluded,occluded_gt);
*/







     return 0;


}









//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf