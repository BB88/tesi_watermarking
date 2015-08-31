//
// Created by miky on 16/08/15.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include "utils.h"
#include <cv.h>
#include <highgui.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "dataset/tsukuba_dataset.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;

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


void stereo_watermarking::dft_magnitude(cv::Mat img,std::string window_name)
{
    // Read image from file
    // Make sure that the image is in grayscale
//    Mat img = imread("lena.JPG",0);

    cvtColor( img, img, CV_BGR2GRAY );

    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
    merge(planes, 2, complexI);
    dft(complexI, complexI);  // Applying DFT

    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

//    imshow("Input Image"  , img  );    // Show the result
    imshow(window_name, magI);
    waitKey(0);

    std::ostringstream path ;
    path << "/home/miky/Scrivania/"<< window_name<<".txt";
    stereo_watermarking::writeMatToFile(magI,path.str());
    return ;
}



//boost::shared_ptr<pcl::visualization::PCLVisualizer> stereo_watermarking::createVisualizerRGB (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, std::string title) {
//
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer(title));
//    viewer->setBackgroundColor (0.3, 0.3, 0.3);
//
//    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
//    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
//    viewer->addCoordinateSystem ( 1.0 );
//    viewer->initCameraParameters ();
//    //viewer->spin();
//    return (viewer);
//}
//void stereo_watermarking::viewPointCloudRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr, std::string title) {
//    //Create visualizer
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//    viewer = createVisualizerRGB( point_cloud_ptr, title);
//    viewer->resetCamera();
//    viewer->resetCameraViewpoint ("reconstruction");
////        viewer->resetCamera();
//
//    //Main loop
//    while ( !viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//    }
//    viewer->close();
//}
//
//void stereo_watermarking::createPointCloudOpenCV (Mat& img1, Mat& img2,  Mat& Q, Mat& disp, Mat& recons3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &point_cloud_ptr) {
//
//    cv::reprojectImageTo3D(disp, recons3D, Q, true);
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
//    for (int rows = 0; rows < recons3D.rows; ++rows) {
//
//        for (int cols = 0; cols < recons3D.cols; ++cols) {
//
//            cv::Point3f point = recons3D.at<cv::Point3f>(rows, cols);
//
//            pcl::PointXYZ pcl_point(point.x, point.y, point.z); // normal PointCloud
//            pcl::PointXYZRGB pcl_point_rgb;
//            pcl_point_rgb.x = point.x;    // rgb PointCloud
//            pcl_point_rgb.y = point.y;
//            pcl_point_rgb.z = point.z;
//            // image_left is the binocular_dense_stereo rectified image used in stere reconstruction
//            cv::Vec3b intensity = img1.at<cv::Vec3b>(rows, cols); //BGR
//
//            uint32_t rgb = (static_cast<uint32_t>(intensity[2]) << 16 | static_cast<uint32_t>(intensity[1]) << 8 | static_cast<uint32_t>(intensity[0]));
//
//            pcl_point_rgb.rgb = *reinterpret_cast<float *>(&rgb);
//
//            // filter erroneus points
//            if (pcl_point_rgb.z < 0)
//                point_cloud_ptr->push_back(pcl_point_rgb);
//        }
//
//
//    }
//
//    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
//    point_cloud_ptr->height = 1;
//
//
//}
//cv::datasets::FramePair stereo_watermarking::rectifyImages(Mat& img1, Mat& img2, Mat& M1, Mat& D1, Mat& M2, Mat& D2, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Rect &roi1, Rect &roi2, float scale){
//
//    Size img_size = img1.size();
//
//    M1 *= scale;
//    M2 *= scale;
//
//
//
//    // dopo Q: 0 o CV_CALIB_ZERO_DISPARITY
//    int flags = 0;
//    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, flags, -1, img_size, &roi1, &roi2 );
//
//    Mat map11, map12, map21, map22;
//    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
//    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
//
//    Mat img1r, img2r;
//    remap(img1, img1r, map11, map12, INTER_CUBIC);
//    remap(img2, img2r, map21, map22, INTER_CUBIC); // prima linear
//
//    cv::datasets::FramePair pair;
//
//    pair.frame_left = img1r;
//    pair.frame_right = img2r;
//
//    return pair;
//
//}
//void stereo_watermarking::generatePointCloud(cv::Mat disp, cv::Mat img_left,cv::Mat img_right, int frame_num){
//
//    string path("/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/");
//    Ptr<tsukuba_dataset> dataset = tsukuba_dataset::create();
//    dataset->load(path);
//
//    // load images data
//    Ptr<cv::datasets::tsukuba_datasetObj> data_stereo_img =
//            static_cast< Ptr<cv::datasets::tsukuba_datasetObj> >  (dataset->getTrain()[frame_num]);
//
//    // load images
//
////    cv::datasets::FramePair tuple_img = dataset->load_stereo_images(frame_num+1);
////    img_left = tuple_img.frame_left;
////    img_right = tuple_img.frame_right;
//
//    // init
//    Mat R1,R2,P1,P2,Q;
//    // zero distiorsions
//    Mat D_left = Mat::zeros(1, 5, CV_64F);
//    Mat D_right = Mat::zeros(1, 5, CV_64F);
//
//    // load K and R from dataset info
//    Mat M_left = Mat(data_stereo_img->k);
//    Mat M_right = Mat(data_stereo_img->k);
//
//    // Left image
//    Mat r_left = Mat(data_stereo_img->r);
//    Mat t_left = Mat(3, 1, CV_64FC1, &data_stereo_img->tl);
//
//    // Right image
//    Mat r_right = Mat(data_stereo_img->r);
//    Mat t_right = Mat(3, 1, CV_64FC1, &data_stereo_img->tr);
//
//    // rotation between left and right
//    // use ground truth rotation (img are already rectified
//    cv::Mat R = Mat::eye(3,3, CV_64F); //r_right*r_left.inv();
//    // translation between img2 and img1
////        cv::Mat T = t_left - (R.inv()*t_right );
//    // use ground truth translation
//    cv::Mat T = Mat::zeros(3, 1, CV_64F);
//    T.at<double>(0,0) = 10.;
//    Rect roi1,roi2;
//
//    cv::datasets::FramePair tuple_img_rect = stereo_watermarking::rectifyImages(img_left, img_right, M_left, D_left, M_right, D_right, R, T, R1, R2, P1, P2, Q, roi1, roi2, 1.f);
//    // get the rectified images
//    // img_left = tuple_img_rect.frame_left;
//    //        img_right = tuple_img_rect.frame_right;
////
////    Mat disp;
////    disp = dataset->load_disparity(frame_num+1);
//
////    float baseline = 10;
//
//    Mat depth_image(disp.size(), CV_32F);
//    Mat recons3D(disp.size(), CV_32FC3);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
//    stereo_watermarking::createPointCloudOpenCV(img_left, img_right, Q, disp, recons3D, point_cloud_ptr);
//
////    stereo_watermarking::depthFromDisparity (disp, M_left.at<double>(0,0), baseline, 0, depth_image, true);
////    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
////    stereo_watermarking::pointcloudFromDepthImage (depth_image, img_left, M_left, point_cloud_ptr);
//
//    stereo_watermarking::viewPointCloudRGB(point_cloud_ptr, "cloud ");
//}

void stereo_watermarking::writeMatToFile(cv::Mat& m, std::string filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

