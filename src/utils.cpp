//
// Created by miky on 16/08/15.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include "utils.h"
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

void stereo_watermarking::sobel_filtering(cv::Mat src, const char* window_name){
    /* SOBEL */
    cv::Mat  src_gray;
    cv::Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Load an image

    if( src.empty() )
    { return ; }

    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Convert it to gray
    if ( src.channels() == 3)
        cvtColor( src, src_gray, COLOR_RGB2GRAY );
    else src.copyTo(src_gray);

    /// Create window
    namedWindow( window_name, WINDOW_AUTOSIZE );

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    std::ostringstream path ;
    path << "/home/miky/ClionProjects/tesi_watermarking/img/"<< window_name<<".png";
//    cout<<path.str();
    imwrite(path.str(),grad);
    imshow( window_name, grad );

    waitKey(0);
}
void stereo_watermarking::show_difference(cv::Mat img1,cv::Mat img2,std::string window){

    unsigned char *difference =  new unsigned char[img1.rows * img1.cols *3];
    unsigned char *img1_uchar =  img1.data;
    unsigned char *img2_uchar =  img2.data;

    for (int i=0;i<img1.rows * img1.cols *3;i++){
        difference[i] = abs(img1_uchar[i] - img2_uchar[i]);
    }

    cv::Mat difference_cv = cv::Mat::zeros(img1.rows, img1.cols , CV_8UC3);

    int count=0;
   for (int j = 0; j < img1.rows; j++)
        for (int i = 0; i < img1.cols; i++){
            difference_cv.at<cv::Vec3b>(j, i) [0] = difference[count]; count++;
            difference_cv.at<cv::Vec3b>(j, i) [1] = difference[count]; count++;
            difference_cv.at<cv::Vec3b>(j, i) [2] = difference[count]; count++;
        }
    cv::imshow(window.c_str(), difference_cv);
    cv::waitKey(0);
}

cv::Mat stereo_watermarking::equalizeIntensity(const cv::Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<cv::Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        merge(channels,ycrcb);
        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }

    return cv::Mat();
}
int stereo_watermarking::equi_histo(cv::Mat image, std::string window_name, cv::Mat &equi_image)
{
    Mat src;



    /// Load image
    image.copyTo(src);

    if( !src.data )
    { cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
        return -1;}

    /// Convert to grayscale
    cvtColor( src, src, CV_BGR2GRAY );

    /// Apply Histogram Equalization
    equalizeHist( src, equi_image );

    /// Display results
    namedWindow( window_name + " original", CV_WINDOW_AUTOSIZE );
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    imshow( window_name+ " original", src );
    imshow( window_name, equi_image );

    /// Wait until user exits the program
    waitKey(0);

    return 0;
}


void stereo_watermarking::printRGB (cv::Mat image, int x, int y){

    int r, g, b;
    b = image.at<Vec3b>(y,x)[0];
    g = image.at<Vec3b>(y,x)[1];
    r = image.at<Vec3b>(y,x)[2];
    cout<<endl;
    cout<<"red: " << r << " "<<"green: " << g << " "<<"blue: " << b <<endl;
}



void stereo_watermarking::histo (cv::Mat image, std::string window_name) {
    Mat src;

/// Load image
    image.copyTo(src);

/// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

/// Establish the number of bins
    int histSize = 256;

/// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

/// Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

// Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

/// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

/// Draw for each channel
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

/// Display
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    imshow(window_name, histImage);

    waitKey(0);
}


