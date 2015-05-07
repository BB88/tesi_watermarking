//
// Created by bene on 23/04/15.
//

#ifndef TESI_WATERMARKING_OCCLUSIONS_HANDLER_H
#define TESI_WATERMARKING_OCCLUSIONS_HANDLER_H

#endif //TESI_WATERMARKING_OCCLUSIONS_HANDLER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

namespace occlusions_handler {

    double getSimilarity(const Mat A, const Mat B);

    void occlusions_filler();


}