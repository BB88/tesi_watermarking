//
// Created by miky on 05/06/15.
//

#include "right_view.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;


/**
 * left_rnc_no_occ(..)
 *
 * reconstruct the left view warping the right view according to the disparity map
 *
 * @params right: right view to warp
 * @params disp: right-to-left disparity map
 * @params width: view width
 * @params height: view height
 * @return rcn_left: reconstructed left view
 */
unsigned char* Right_view::left_rnc_no_occ(unsigned char *right, cv::Mat disp, int width, int height) {

    int nc = width;
    unsigned char d = 0;
    unsigned char occ = 0;
//      create general left image
    int dim = width*height*3;
    unsigned char * rcn_left = new unsigned char[dim];
    for (int i = 0; i<dim; i ++)
        rcn_left[i] = (unsigned char)0;
    for (int i=0;i<height;i++)
        for (int j= (width-1);j>=0;j--){
            d = disp.at<uchar>(i,j);

            if(static_cast<int>(d)!=0 && ((i*nc + j + static_cast<unsigned>(d))*3 + 2) < ((i+1)*nc*3))
                for (int k=0; k<3; k++)
                    rcn_left[(i*nc + j + static_cast<int>(d))*3 + k ] = right[(i*nc + j)*3 + k];
        }
    return rcn_left;
}