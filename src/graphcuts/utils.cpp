//
// Created by miky on 05/08/15.
//

#include "utils.h"
#include <cmath>
static const int MAX_DENOM=1<<4;

//bool isGray(RGBImage im) {
//    const int xsize=imGetXSize(im), ysize=imGetYSize(im);
//    for(int y=0; y<ysize; y++)
//        for(int x=0; x<xsize; x++)
//            if(imRef(im,x,y).c[0] != imRef(im,x,y).c[1] ||
//               imRef(im,x,y).c[0] != imRef(im,x,y).c[2])
//                return false;
//    return true;
//}


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
