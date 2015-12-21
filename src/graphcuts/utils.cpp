//
// Created by miky on 05/08/15.
//

#include "utils.h"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <cv.h>
#include <highgui.h>

#include "../disparity_optimization/disp_opt.h"

using namespace std;
using namespace cv;

static const int MAX_DENOM=1<<4;

//bool isGray(RGBImage im) {
//    const int xsize=imGetXSize(im), ysize=imGetYSize(im);
//    for(int y=0; y<ysize; y++)
//        for(int x=0; x<xsize; x++)
//            if(imRef(im,x,y).c[0] != imRef(im,x,y).c[1] ||
//               imRef(im,x,y).c[0] != imRef(im,x,y).c[2])
//                return false;
//    return true;
//}#include <cstdint>


/// Convert to gray level a color image (extract red channel)
void graph_cuts_utils::convert_gray(GeneralImage& im) {
    const int xsize=imGetXSize(im), ysize=imGetYSize(im);
    GrayImage g = (GrayImage)imNew(IMAGE_RGB, xsize, ysize);
    for(int y=0; y<ysize; y++)
        for(int x=0; x<xsize; x++)
            imRef(g,x,y) = imRef((RGBImage)im,x,y).c[0];
    imFree(im);
    im = (GeneralImage)g;
}




/// Is the color image actually gray?
bool graph_cuts_utils::isGray(RGBImage im) {
    const int xsize=imGetXSize(im), ysize=imGetYSize(im);
    for(int y=0; y<ysize; y++)
        for(int x=0; x<xsize; x++)
            if(imRef(im,x,y).c[0] != imRef(im,x,y).c[1] ||
               imRef(im,x,y).c[0] != imRef(im,x,y).c[2])
                return false;
    return true;
}



/// Store in \a params fractions approximating the last 3 parameters.
///
/// They have the same denominator (up to \c MAX_DENOM), chosen so that the sum
/// of relative errors is minimized.
void graph_cuts_utils::set_fractions(Match::Parameters& params,
                   float K, float lambda1, float lambda2) {
    float minError = std::numeric_limits<float>::max();
    for(int i=1; i<=MAX_DENOM; i++) {
        float e = 0;
        int numK=0, num1=0, num2=0;
        if(K>0)
            e += std::abs((numK=int(i*K+.5f))/(i*K) - 1.0f);
        if(lambda1>0)
            e += std::abs((num1=int(i*lambda1+.5f))/(i*lambda1) - 1.0f);
        if(lambda2>0)
            e += std::abs((num2=int(i*lambda2+.5f))/(i*lambda2) - 1.0f);
        if(e<minError) {
            minError = e;

            params.denominator = i;
            params.K = numK;
            params.lambda1 = num1;
            params.lambda2 = num2;
        }
    }
}

/// Make sure parameters K, lambda1 and lambda2 are non-negative.
///
/// - K may be computed automatically and lambda set to K/5.
/// - lambda1=3*lambda, lambda2=lambda
/// As the graph requires integer weights, use fractions and common denominator.
void graph_cuts_utils::fix_parameters(Match& m, Match::Parameters& params,
                    float& K, float& lambda, float& lambda1, float& lambda2) {
    if(K<0) { // Automatic computation of K
        m.SetParameters(&params);
        K = m.GetK();
    }
    if(lambda<0) // Set lambda to K/5
        lambda = K/5;
    if(lambda1<0) lambda1 = 3*lambda;
    if(lambda2<0) lambda2 = lambda;
    set_fractions(params, K, lambda1, lambda2);
    m.SetParameters(&params);
}

/**
 * kz_main(..)
 *
 * compute the disparity map with the graph cut kolmogorov zabih algorithm
 *
 * @params left_to_right: if true compute left-to-right disparity, if false compute right-to-left disparity
 * @params img1_name: left view name
 * @params img2_name: right view name
 * @params img1: left view
 * @params img2: right view
 * @params dmin: min disparity value
 * @params dmax: max disparity value
 * @return norm_disp: normalized disparity map
 */
cv::Mat graph_cuts_utils::kz_main(bool left_to_right, std::string img1_name, std::string img2_name, cv::Mat img1, cv::Mat img2, int dmin, int dmax ) {

/* kz_disp PARAMETERS */
/*
*
* lambda = 15.8
* k = 79.12
*/
    cv::cvtColor(img1, img1, CV_BGR2GRAY);
    cv::cvtColor(img2, img2, CV_BGR2GRAY);
    Match::Parameters params = { // Default parameters
            Match::Parameters::L2, 1, // dataCost, denominator
            8, -1, -1, // edgeThresh, lambda1, lambda2 (smoothness cost)
            -1, // K (occlusion cost)
            4, false // maxIter, bRandomizeEveryIteration
    };
    float K = -1, lambda = -1, lambda1 = -1, lambda2 = -1;
    params.dataCost = Match::Parameters::L1;
// params.dataCost = Match::Parameters::L2;

    GeneralImage im1 = (GeneralImage) imLoadFromMat(IMAGE_GRAY, img1);
    GeneralImage im2 = (GeneralImage) imLoadFromMat(IMAGE_GRAY, img2);
    bool color = false;
    if (graph_cuts_utils::isGray((RGBImage) im1) && graph_cuts_utils::isGray((RGBImage) im2)) {
//        color = false;
        graph_cuts_utils::convert_gray(im1);
        graph_cuts_utils::convert_gray(im2);
    }
    Match m1(im1, im2, color);
    Match m2(im2, im1, color);
    if (left_to_right)
        m1.SetDispRange(dmin, dmax);
    else m2.SetDispRange(dmin, dmax);
    time_t seed = time(NULL);
    srand((unsigned int) seed);
    std::stringstream path_disp;
    if (left_to_right)
        path_disp << "./disp_" << img1_name << "_" << "to_" << img2_name << ".png";
    else
        path_disp << "./disp_" << img2_name << "_" << "to_" << img1_name << ".png";
    if (left_to_right) {
        graph_cuts_utils::fix_parameters(m1, params, K, lambda, lambda1, lambda2);
        m1.KZ2();
        m1.SaveScaledXLeft(path_disp.str().c_str(), false);
    }
    else {
        graph_cuts_utils::fix_parameters(m2, params, K, lambda, lambda1, lambda2);
        m2.KZ2();
        m2.SaveScaledXLeft(path_disp.str().c_str(), true);
    }
    cv::Mat computed_disp = imread(path_disp.str().c_str(), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(computed_disp,computed_disp,CV_BGR2GRAY);
    cv::Mat norm_disp;
    Disp_opt opt;
    opt.disparity_normalization(computed_disp,dmin,dmax,norm_disp);
    return norm_disp;
    
}