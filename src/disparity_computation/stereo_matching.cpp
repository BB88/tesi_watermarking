//
// Created by bene on 10/04/15.
//

#include "stereo_matching.h"
#include <opencv2/core/core.hpp>
#include <cv.h>
#include <highgui.h>


using namespace std;
using namespace cv;

namespace stereomatching {

    void display(Mat &img1, Mat &img2, Mat &disp) {

    // namedWindow("left ", 1);
        imshow("left " , img1);
    // namedWindow("right", 1);
        imshow("right" , img2);
    // namedWindow("disparity", 0);
        imshow("disparity", disp);
        printf("press any key to continue...");

        fflush(stdout);
        waitKey();
        destroyAllWindows();

        printf("\n");

    }


    void stereo_matching(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& disp) {

        std::string tipo = "SGBM";

        Mat g1, g2;

///da provare
        cvtColor(img_left, g1, CV_BGR2GRAY);
        cvtColor(img_right, g2, CV_BGR2GRAY);

        StereoBM sbm;

        if (tipo == "BM") {


            int numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_left.rows / 8) + 15) & -16;

        //    sbm.state->roi1 = roi1;
        //    sbm.state->roi2 = roi2;
            sbm.state->preFilterCap = 63;
            sbm.state->SADWindowSize = 5;
            sbm.state->minDisparity = 1;
            sbm.state->numberOfDisparities = 32;
            sbm.state->textureThreshold = 9;
            sbm.state->uniquenessRatio = 12;
            sbm.state->speckleWindowSize = 0;
            sbm.state->speckleRange = 0;
            sbm.state->disp12MaxDiff = 1;
//
// sbm.state->SADWindowSize = 5;
// sbm.state->numberOfDisparities = 192;
// sbm.state->preFilterSize = 5;
// sbm.state->preFilterCap = 51;
// sbm.state->minDisparity = 25;
// sbm.state->textureThreshold = 223;
// sbm.state->uniquenessRatio = 0;
// sbm.state->speckleWindowSize = 0;
// sbm.state->speckleRange = 0;
// sbm.state->disp12MaxDiff = 0;

        }
        else if (tipo == "SGBM") {
            StereoSGBM sbm;
            sbm.SADWindowSize = 5;
            sbm.numberOfDisparities = 112;
            sbm.preFilterCap = 63;
            sbm.minDisparity = 0;
            sbm.uniquenessRatio = 10;
            sbm.speckleWindowSize = 0;
            sbm.speckleRange = 0;
            sbm.disp12MaxDiff = 1;
            sbm.fullDP = false;
            sbm.P1 = 8 * 3 * 5 * 5;
            sbm.P2 = 8 * 3 * 5 * 5;
            sbm(g1, g2, disp);


        }

        sbm(g1, g2, disp, CV_32F);


    }

}