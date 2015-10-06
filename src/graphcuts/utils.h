//
// Created by miky on 05/08/15.
//

#ifndef TESI_WATERMARKING_UTILS_H
#define TESI_WATERMARKING_UTILS_H

#endif //TESI_WATERMARKING_UTILS_H

#include "image.h"
#include "match.h"
#include <iostream>
#include <limits>

using namespace std;

namespace graph_cuts_utils{

    void kz_main(bool left_to_right,std::string img1_name,  std::string img2_name);

    void convert_gray(GeneralImage& im);
    bool isGray(RGBImage im);
    void set_fractions(Match::Parameters& params,float K, float lambda1, float lambda2);
    void fix_parameters(Match& m, Match::Parameters& params, float& K, float& lambda, float& lambda1, float& lambda2);
}