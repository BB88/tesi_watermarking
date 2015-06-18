//
// Created by bene on 23/04/15.
//

#include "occlusions_handler.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace occlusions_handler {

    double getSimilarity(const Mat A, const Mat B) {
        if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
            // Calculate the L2 relative error between the 2 images.
            double errorL2 = norm(A, B, CV_L2);
            // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
            double similarity = errorL2 / (double) (A.rows * A.cols);
            return similarity;
        }
        else {
            //cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
            return 100000000.0;  // Return a bad value
        }
    }




    }