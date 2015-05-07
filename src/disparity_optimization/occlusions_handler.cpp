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

    void occlusions_enhancing(){

        Mat src1;
        src1 = imread("/home/bene/ClionProjects/tesi_watermarking/img/filtered_disp2.png", CV_LOAD_IMAGE_COLOR);
        namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
        imshow( "Original image", src1 );

        // 3 channel image with BGR color (type 8UC3)
        // the values can be stored in "int" or in "uchar". Here int is used.
/*        Vec3b intensity2 = src1.at<Vec3b>(10,15);
        int blue = intensity2.val[0];
        int green = intensity2.val[1];
        int red = intensity2.val[2];
        cout << "Intensity = " << endl << " " << blue << " " << green << " " << red << endl << endl;*/

// ******************* WRITE to Pixel intensity **********************
        // This is an example in OpenCV 2.4.6.0 documentation
/*        Mat H(10, 10, CV_64F);
        for(int i = 0; i < H.rows; i++)
            for(int j = 0; j < H.cols; j++)
                H.at<double>(i,j)=1./(i+j+1);
        cout<<H<<endl<<endl;*/

        // Modify the pixels of the BGR image
        for (int i=0; i<src1.rows; i++)
        {
            for (int j=0; j<src1.cols; j++)
            {
                if ((src1.at<Vec3b>(i,j)[0] == 255  &&  src1.at<Vec3b>(i,j)[1] == 255 &&  src1.at<Vec3b>(i,j)[2] == 0) || (src1.at<Vec3b>(i,j)[0] < 100  &&  src1.at<Vec3b>(i,j)[1] < 100 &&  src1.at<Vec3b>(i,j)[2] < 100) ){
                    src1.at<Vec3b>(i,j)[0] = 0;
                    src1.at<Vec3b>(i,j)[1] = 0;
                    src1.at<Vec3b>(i,j)[2] = 0;
                } else{
                    src1.at<Vec3b>(i,j)[0] = 255;
                    src1.at<Vec3b>(i,j)[1] = 255;
                    src1.at<Vec3b>(i,j)[2] = 255;
                }
            }
        }
        namedWindow( "Modify pixel", CV_WINDOW_AUTOSIZE );
        imshow( "Modify pixel", src1 );
        imwrite("/home/bene/ClionProjects/tesi_watermarking/img/filtered_bw.png", src1);
        waitKey(0);
    }


}