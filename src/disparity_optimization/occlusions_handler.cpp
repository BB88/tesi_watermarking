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

    void occlusions_filler(){

        Mat image = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp2.png");
        if (image.cols == 0){
            cout << "Empty image";
        } else {
            int x =0;
            int y =0;
            Vec3b intensity = image.at<Vec3b>(y, x);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            cout << "blue " << static_cast<unsigned>(blue) << " " << "green " << static_cast<unsigned>(green)<< " " << "red " << static_cast<unsigned>(red);
        }

    }


}