//
// Created by miky on 16/08/15.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include "utils.h"
#include <cv.h>
#include <highgui.h>
#include <fstream>
//#include <pcl/common/common_headers.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include "dataset/tsukuba_dataset.h"
#include "./img_watermarking/watermarking.h"
#include "./img_watermarking/allocim.h"
#include "./img_watermarking/fft2d.h"
#include "./graphUtils/GraphUtils.h"

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
    cv::imwrite(path.str(),grad);
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




float stereo_watermarking::MSE(int width,int height,double** A,double ** B){
    float sum = 0.0;
    double difference;
    for(int x = 0; x < width;++x){
         for(int y = 0; y < height; ++y){
             difference = A[x][y] - B[x][y];
             sum = sum + difference*difference;
         }
    }
    sum = sum /(width*height);
return sum;
}

void stereo_watermarking::dft_comparison(unsigned char* Image1, unsigned char* Image2, int dim, std::string img1_name, std::string img2_name ){

    Watermarking image_watermarking;
    float   **imyout1;			// immagine
    double  **imdft1;		// immagine della DFT
    double  **imdftfase1;	// immagine della fase della DFT



    imyout1 = AllocIm::AllocImFloat(dim, dim);
    imdft1 = AllocIm::AllocImDouble(dim, dim);
    imdftfase1 = AllocIm::AllocImDouble(dim, dim);




// SE COLOUR
    unsigned char **imr1;	// matrici delle componenti RGB
    unsigned char **img1;
    unsigned char **imb1;

    float **imc21;			// matrice di crominanza c2
    float **imc31;

    imc21 = AllocIm::AllocImFloat(dim, dim);
    imc31 = AllocIm::AllocImFloat(dim, dim);
    imr1 = AllocIm::AllocImByte(dim, dim);
    img1 = AllocIm::AllocImByte(dim, dim);
    imb1 = AllocIm::AllocImByte(dim, dim);



    int offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr1[i][j] = Image1[offset];offset++;
            img1[i][j] = Image1[offset];offset++;
            imb1[i][j] = Image1[offset];offset++;
        }

    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    image_watermarking.rgb_to_crom(imr1, img1, imb1, dim, dim, 1, imyout1, imc21, imc31);

    FFT2D::dft2d(imyout1, imdft1, imdftfase1, dim, dim);

    float   **imyout2;			// immagine
    double  **imdft2;		// immagine della DFT
    double  **imdftfase2;	// immagine della fase della DFT



    imyout2 = AllocIm::AllocImFloat(dim, dim);
    imdft2 = AllocIm::AllocImDouble(dim, dim);
    imdftfase2 = AllocIm::AllocImDouble(dim, dim);




// SE COLOUR
    unsigned char **imr2;	// matrici delle componenti RGB
    unsigned char **img2;
    unsigned char **imb2;

    float **imc22;			// matrice di crominanza c2
    float **imc32;

    imc22 = AllocIm::AllocImFloat(dim, dim);
    imc32 = AllocIm::AllocImFloat(dim, dim);
    imr2 = AllocIm::AllocImByte(dim, dim);
    img2 = AllocIm::AllocImByte(dim, dim);
    imb2 = AllocIm::AllocImByte(dim, dim);



     offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr2[i][j] = Image2[offset];offset++;
            img2[i][j] = Image2[offset];offset++;
            imb2[i][j] = Image2[offset];offset++;
        }

    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    image_watermarking.rgb_to_crom(imr2, img2, imb2, dim, dim, 1, imyout2, imc22, imc32);

    FFT2D::dft2d(imyout2, imdft2, imdftfase2, dim, dim);

    float mse_mag = stereo_watermarking::MSE(dim,dim,imdft1,imdft2);
    float mse_ph = stereo_watermarking::MSE(dim,dim,imdftfase1,imdftfase2);


    std::cout<<std::setprecision (15) <<"immagini "<<img1_name<<" e "<<img2_name<<" : MSE magnitudine: "<< mse_mag << " MSE fase: "<<mse_ph<<endl;

    std::ostringstream filename1 ;
    filename1 << "magnitudine_"<< img1_name;
    stereo_watermarking::show_double_mat(dim,dim,imdft1,filename1.str());
    std::ostringstream filename2 ;
    filename2 << "magnitudine_"<< img2_name;
    stereo_watermarking::show_double_mat(dim,dim,imdft2,filename2.str());
    std::ostringstream filename3 ;
    filename3 << "fase_"<< img1_name;
    stereo_watermarking::show_double_mat(dim,dim,imdftfase1,filename3.str());
    std::ostringstream filename4 ;
    filename4 << "fase_"<< img2_name;
    stereo_watermarking::show_double_mat(dim,dim,imdftfase2,filename4.str());

}

double* stereo_watermarking::compute_coeff_function(unsigned char* image, int dim, std::string filename){

    Watermarking iw;

    float   **imyout4;			// immagine
    double  **imdft4;		// immagine della DFT
    double  **imdftfase4;	// immagine della fase della DFT

    imyout4 = AllocIm::AllocImFloat(dim, dim);
    imdft4 = AllocIm::AllocImDouble(dim, dim);
    imdftfase4 = AllocIm::AllocImDouble(dim, dim);


// SE COLOUR
    unsigned char **imr4;	// matrici delle componenti RGB
    unsigned char **img4;
    unsigned char **imb4;

    float **imc24;			// matrice di crominanza c2
    float **imc34;

    imc24 = AllocIm::AllocImFloat(dim, dim);
    imc34 = AllocIm::AllocImFloat(dim, dim);
    imr4 = AllocIm::AllocImByte(dim, dim);
    img4 = AllocIm::AllocImByte(dim, dim);
    imb4 = AllocIm::AllocImByte(dim, dim);

    int offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr4[i][j] = image[offset];offset++;
            img4[i][j] = image[offset];offset++;
            imb4[i][j] = image[offset];offset++;
        }

    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    iw.rgb_to_crom(imr4, img4, imb4, dim, dim, 1, imyout4, imc24, imc34);

    FFT2D::dft2d(imyout4, imdft4, imdftfase4, dim, dim);

    int lastValue = 128;
    int currDiag = 0;
    int loopFrom;
    int loopTo;
    int i;
    int row;
    int col;

    double * coeff_vector = new double [128*128/2];

    int j=0;
    do
    {
        if ( currDiag < dim/2 ) // if doing the upper-left triangular half
        {
            loopFrom = 0;
            loopTo = currDiag;
        }
        else // doing the bottom-right triangular half
        {
            loopFrom = currDiag - dim/2 + 1;
            loopTo = dim/2 - 1;
        }

        for ( i = loopFrom; i <= loopTo; i++ )
        {
            if ( currDiag % 2 == 0 ) // want to fill upwards
            {
                row = loopTo - i + loopFrom;
                col = i;
            }
            else // want to fill downwards
            {
                row = i;
                col = loopTo - i + loopFrom;
            }

            coeff_vector[j] = imdft4[ row ][ col ];
//            cout << row<<" "<<col<<endl;
            j++;
        }

        currDiag++;
    }
    while ( currDiag <= lastValue );

    std::ostringstream path ;
    path <<"/home/miky/Scrivania/"<< filename<<".txt";

    stereo_watermarking::writeToFile(coeff_vector,j,path.str());

    AllocIm::FreeIm(imc24) ;
    AllocIm::FreeIm(imc34) ;
    AllocIm::FreeIm(imr4);
    AllocIm::FreeIm(img4);
    AllocIm::FreeIm(imb4);

    AllocIm::FreeIm(imyout4);
    AllocIm::FreeIm(imdft4);
    AllocIm::FreeIm(imdftfase4);

    return coeff_vector;

}




/*
       cv::Mat right = imread("/home/miky/ClionProjects/tesi_watermarking/img/r.png", CV_LOAD_IMAGE_COLOR);
       unsigned char *right_uchar = right.data;
       unsigned char *squared_right =  new unsigned char[squared_dim];
       for (int i = 0; i < 256; i ++ )
           for (int j = 0; j < nc_q; j++)
               squared_right[(i * nc_q)+ j] = right_uchar[(i * nc) + (j + index - 21*3 )];

      unsigned char *squared_marked_right = image_watermarking.insertWatermark(squared_right,256,256);

       //salvare coefficienti dft
       double *coeff_right =image_watermarking.getCoeff_dft();


       float sum = 0.0;
       double difference;
       for (int i=0;i<coeff_num;i++){
           difference = coeff_left[i] - coeff_right[i];
           sum = sum + difference*difference;
       }
       sum = sum /(coeff_num);

       std::cout<<std::setprecision (15)<< "MSE coefficienti destra sinistra: "<<sum<<endl;
   */







//
//void stereo_watermarking::show_double_mat(int width,int height,double** A,std::string window_name){
//    cv::Mat mat =  cv::Mat::zeros(width, height, CV_32F);
//    for(int x = 0; x < width;++x){
//        for(int y = 0; y < height; ++y){
//            mat.at<float>(x,y)=A[x][y];
//        }
//    }
//    std::ostringstream path ;
//    path <<"/home/miky/Scrivania/images/dft/"<< window_name<<".png";
////    cout<<path.str();
//    cv::imwrite(path.str(),mat);
//    imshow(window_name,mat);
//    waitKey(0);
//    return;
//}
void stereo_watermarking::histo_equalizer(Mat img, std::string window_name){

    vector<Mat> channels;
    Mat img_hist_equalized;

    cvtColor(img, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

    split(img_hist_equalized,channels); //split the image into channels

    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

    merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

    cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

    //create windows
//    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
//    namedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);
    std::ostringstream path ;
    path <<"/home/miky/ClionProjects/tesi_watermarking/img/"<< window_name<<".png";
//    cout<<path.str();
    cv::imwrite(path.str(),img_hist_equalized);
    //show the image
//    imshow("Original Image", img);
//    imshow(window_name.c_str(), img_hist_equalized);
//
//    waitKey(0); //wait for key press

    destroyAllWindows();
}





//
//
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
//
//void stereo_watermarking::writeToFile(double* m,int lenght, std::string filename)
//{
//    ofstream fout(filename);
//
//    if(!fout)
//    {
//        cout<<"File Not Opened"<<endl;  return;
//    }
//
//    for(int i=0; i<lenght; i++)
//    {
//        fout<<m[i]<<"\t";
//        fout<<endl;
//    }
//
//    fout.close();
//}

double stereo_watermarking::similarity_measures(double* wat, double* retrieve_wat, int coeff_num, std::string first_element, std::string second_element){

    Mat wat_mat(1, coeff_num, CV_32F); // 1 è la riga
    for (int i = 0; i < wat_mat.cols; i++){
        wat_mat.at<float>(0,i) = (float)wat[i];
    }
  /*  Scalar     mean;
    Scalar     stddev;
    meanStdDev(wat_mat, mean, stddev);
    float myMAtMean = mean.val[0];
    float myMAtSvd = stddev.val[0];
    // cout <<"before :"<< setprecision(15)<<myMAtMean<<endl;
    // cout <<"before :" << setprecision(15)<<myMAtSvd<<endl;
    for (int i = 0; i < wat_mat.cols; i++){
        wat_mat.at<float>(0,i) = (wat_mat.at<float>(0,i) - myMAtMean)/myMAtSvd;
    }
    meanStdDev(wat_mat, mean, stddev);
    myMAtMean = mean.val[0];
    myMAtSvd = stddev.val[0];*/
    // cout <<"after :" << setprecision(15)<<myMAtMean<<endl;
    // cout  <<"after :"<< setprecision(15)<<myMAtSvd<<endl;

    Mat ret_wat_mat(1, coeff_num, CV_32F);
    //   cout<< wat_mat.cols<< " righe "<<wat_mat.rows<<endl;
    for (int i = 0; i < ret_wat_mat.cols; i++){
        ret_wat_mat.at<float>(0,i) = (float)retrieve_wat[i];
    }
    /*Scalar     ret_mean;
    Scalar     ret_stddev;
    meanStdDev(ret_wat_mat, ret_mean, ret_stddev);
    float ret_myMAtMean = ret_mean.val[0];
    float ret_myMAtSvd = ret_stddev.val[0];
    // cout <<"ret_before :"<< setprecision(15)<<ret_myMAtMean<<endl;
    // cout <<"ret_before :" << setprecision(15)<<ret_myMAtSvd<<endl;
    for (int i = 0; i < wat_mat.cols; i++){
        ret_wat_mat.at<float>(0,i) = (ret_wat_mat.at<float>(0,i) - ret_myMAtMean)/ret_myMAtSvd;
    }
    meanStdDev(ret_wat_mat, ret_mean, ret_stddev);
    ret_myMAtMean = ret_mean.val[0];
    ret_myMAtSvd = ret_stddev.val[0];*/
    // cout <<"ret_after :"<< setprecision(15)<<ret_myMAtMean<<endl;
    // cout <<"ret_after :" << setprecision(15)<<ret_myMAtSvd<<endl;
    Mat wat_corr;
    matchTemplate(ret_wat_mat,wat_mat, wat_corr, CV_TM_CCOEFF_NORMED);
    cout << "correlation btw " << first_element << " and "<< second_element << ":   " << (wat_corr.at<float>(0,0))<<endl;

  /*  double sim = 0.0;
    double den = 0.0;
    for (int i = 0; i < ret_wat_mat.cols; i++){
        sim += wat_mat.at<float>(0,i)*ret_wat_mat.at<float>(0,i);
        den += ret_wat_mat.at<float>(0,i)*ret_wat_mat.at<float>(0,i);
    }
    sim /= sqrt(den);*/
  //  cout <<"sim :   " << setprecision(15)<<sim<<endl; // max value 68
  //  return sim;


}
void stereo_watermarking::random_mark_detection(int number_of_marks, unsigned char* marked_image, int dim){
    static const char alpha_char[] = "abcdefghijklmnopqrstuvwxyz";
    static const char num_char [] =  "0123456789";
    bool * detection = new bool [number_of_marks];
    double * det = new double [number_of_marks];
    Watermarking image_watermarking;
    int wsize = 64;
    for (int i = 0; i < number_of_marks; i++){
        int mark[wsize];
        // generate another 64random bit watermark
        for (int i = 0; i < 64; i++) {
            int b = rand() % 2;
            mark[i] = b;
        }
        char *string_pswd = new char[16];
        char *num_pswd = new char[8];
        for (int i = 0; i < 16; i++) {
            string_pswd[i] = alpha_char[rand() % (sizeof(alpha_char) - 1)];
        }
        for (int i = 0; i < 8; i++) {
            num_pswd[i] = num_char[rand() % (sizeof(num_char) - 1)];
        }
        double power = 0.3;
        image_watermarking.setParameters(mark,wsize,power);
        image_watermarking.setPassword(string_pswd,num_pswd);
        detection[i] =  image_watermarking.extractWatermark(marked_image,dim ,dim, dim);
    }
    for (int i = 0; i < number_of_marks; i++){
        if(detection[i] == true ){
            det[i] = 1;
        } else
            det[i] = 0;

    }
    stereo_watermarking::writeToFile(det, number_of_marks, "/home/miky/Scrivania/detection.txt");

}






void stereo_watermarking::similarity_graph(int number_of_marks,int coeff_num,double* wat){

    static const char alpha_char[] = "abcdefghijklmnopqrstuvwxyz";
    static const char num_char [] =  "0123456789";
    Watermarking image_watermarking;
   // Point points[number_of_marks];
    float* sim_values = new float [number_of_marks+1];
    for (int k = 0; k < number_of_marks; k++) {
        double *retrieve_wat = new double[coeff_num];
        int mark[64];
        // generate another 64random bit watermark
        for (int i = 0; i < 64; i++) {
            int b = rand() % 2;
            mark[i] = b;
        }
        char *string_pswd = new char[16];
        char *num_pswd = new char[8];
        for (int i = 0; i < 16; i++) {
            string_pswd[i] = alpha_char[rand() % (sizeof(alpha_char) - 1)];
        }
        for (int i = 0; i < 8; i++) {
            num_pswd[i] = num_char[rand() % (sizeof(num_char) - 1)];
        }
        retrieve_wat = image_watermarking.marks_generator(mark, 64, string_pswd, num_pswd, coeff_num);
        // correlation and sim
        double sim = stereo_watermarking::similarity_measures(wat, retrieve_wat, coeff_num, "wat", "retrieve_wat");
        sim_values[k] = (float)sim;
      //  points[k]=  Point(k,sim);
    }
    sim_values[number_of_marks] = 67.517831195934;
    float tmp = 0.0;
    tmp = sim_values[27];
    sim_values[27] = sim_values[number_of_marks];
    sim_values[number_of_marks] = tmp;
    //drawFloatGraph(sim_values, number_of_marks, NULL, 0, 300,);
    //(floatArray, numFloats, bgImg, -25,25, w, h, "Yaw (in degrees)");
    showFloatGraph("Watermarks similarity", sim_values, number_of_marks+1, 0);
    // Create black empty images
   /* Mat simImage(number_of_marks, 1, CV_8UC3, Scalar(255, 255, 255));
    for (int i=1;i<number_of_marks;i++){
        line(simImage,points[i-1],points[i],Scalar(0,0,0),2,8,0);
    }
    namedWindow("similarity image",CV_WINDOW_AUTOSIZE);
    imshow("similarity image",simImage);
    waitKey(0);
    return;
*/
}

void stereo_watermarking::show_ucharImage(unsigned char * image, int width, int height, string nameImage, int channels){

    int count = 0;
    if( channels == 3) {
        cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++) {

                mat_image.at<Vec3b>(j, i)[0] = image[count];
                count++;
                mat_image.at<Vec3b>(j, i)[1] = image[count];
                count++;
                mat_image.at<Vec3b>(j, i)[2] = image[count];
                count++;

            }
        imshow(nameImage, mat_image);
    }else if (channels ==1){
        cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC1);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++) {
                mat_image.at<uchar>(j, i) = image[count];
                count++;
            }
        imshow(nameImage, mat_image);
    }
    else
        cout<<"error: wrong number of channels"<<endl;
    waitKey(0);
}

//void stereo_watermarking::show_doubleImage(double * image, int width, int height, string nameImage){
//
//    int count = 0;
//    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
//    for (int j = 0; j < height; j++)
//        for (int i = 0; i < width; i++){
//
//            mat_image.at<Vec3b>(j,i) [0] = image[count]; count++;
//            mat_image.at<Vec3b>(j,i) [1] = image[count]; count++;
//            mat_image.at<Vec3b>(j,i) [2] = image[count]; count++;
//
//        }
//    imshow(nameImage, mat_image);
//    waitKey(0);
//}

double* stereo_watermarking::not_blind_extraction(double* original_coeff, double* marked_coeff, int coeff_num, double power){

    double *retrieve_wat = new double[coeff_num];
    for (int offset = 0; offset < coeff_num; offset++) {
        //   retrieve_wat[offset] = (marked_coeff_left[offset] - coeff_left[offset]) / (coeff_left[offset]); // additiva-moltiplicativa
        retrieve_wat[offset] = (marked_coeff[offset] - original_coeff[offset]); //additiva
    }
    for (int offset = 0; offset < coeff_num; offset++) {
        retrieve_wat[offset] = retrieve_wat[offset]/power;
    }
    return retrieve_wat;
}

double stereo_watermarking::threshold_computation(double* original_coeff,int coeff_num, double power){
    Mat coeff_mat(1, coeff_num, CV_32F); // 1 è la riga
    for (int i = 0; i < coeff_mat.cols; i++){
        coeff_mat.at<float>(0,i) = (float)original_coeff[i];
    }
    cv:Scalar tmp = mean( coeff_mat );
    float coeff_mean = tmp.val[0];
    double thr = 0.0;
    thr = power/2 * coeff_mean;
    return thr;
}


void stereo_watermarking::compute_luminance(unsigned char* image, int dim, int flag, unsigned char **imr,unsigned char **img,
                                            unsigned char **imb, float **imyout, float **imc2,float **imc3){
    Watermarking image_watermarking;
    int offset = 0;
    if (flag == 1) {
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) {
                imr[i][j] = image[offset];offset++;
                img[i][j] = image[offset];offset++;
                imb[i][j] = image[offset];offset++;
            }
        image_watermarking.rgb_to_crom(imr, img, imb, dim, dim, 1, imyout, imc2, imc3);
    } else if(flag == -1) {
        image_watermarking.rgb_to_crom(imr, img, imb, dim, dim, -1, imyout, imc2, imc3);
        offset = 0;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
            {
                image[offset] = imr[i][j]; offset++;
                image[offset] = img[i][j]; offset++;
                image[offset] = imb[i][j]; offset++;
            }

    }
}

void stereo_watermarking::show_double_mat(int width,int height,double** A,std::string window_name){
    cv::Mat mat =  cv::Mat::zeros(width, height, CV_32F);
    for(int x = 0; x < width;++x){
        for(int y = 0; y < height; ++y){
            mat.at<float>(x,y)=A[x][y];
        }
    }
    std::ostringstream path ;
    path <<"/home/miky/Scrivania/images/dft/"<< window_name<<".png";
//    cout<<path.str();
    cv::imwrite(path.str(),mat);
    imshow(window_name,mat);
    waitKey(0);
    return;
}

void stereo_watermarking::writeToFile(double* m,int lenght, std::string filepath)
{
    ofstream fout(filepath);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<lenght; i++)
    {
        fout<<m[i]<<"\t";
        fout<<endl;
    }

    fout.close();
}

void stereo_watermarking::writeMatToFile(double** m,int dim, std::string filepath)
{
    ofstream fout(filepath);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<dim; i++) {
        for (int j = 0; j<dim;j++) {
            fout << m[i][j]<< "\t";
            fout << endl;
        }
    }
    fout.close();
}

void stereo_watermarking::writefloatMatToFile(float** m,int dim, std::string filepath)
{
    ofstream fout(filepath);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<dim; i++) {
        for (int j = 0; j<dim;j++) {
            fout << m[i][j]<< "\t";
            fout << endl;
        }
    }
    fout.close();
}

Mat stereo_watermarking::show_ucharImage(unsigned char * image, int width, int height, string nameImage){

    int count = 0;
    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){

            mat_image.at<Vec3b>(j,i) [0] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [1] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [2] = image[count]; count++;

        }
//    imshow(nameImage, mat_image);
//    waitKey(0);
    return mat_image;
}

Mat stereo_watermarking::unsignedToMat(unsigned char * squared_image, Mat original_image,  int width, int height, int dim){

    int offset = width - dim - 1;
    unsigned char *rect_image = new unsigned char [width*height*3];
    rect_image = original_image.data;
//    show_ucharImage(rect_image,width,height,"rect",3);
    for (int i = 0; i < height; i ++ )
        for (int j = 0; j < dim; j++) {
            for (int k =0; k<3;k++){
                rect_image[(i * width + j + offset)*3 + k] = squared_image[(i * dim + j)*3 + k];
            }
        }
    int count = 0;
    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){
            mat_image.at<Vec3b>(j,i) [0] = rect_image[count]; count++;
            mat_image.at<Vec3b>(j,i) [1] = rect_image[count]; count++;
            mat_image.at<Vec3b>(j,i) [2] = rect_image[count]; count++;
        }
    return mat_image;
}


void stereo_watermarking::save_ucharImage(unsigned char * image, int width, int height, string nameImage){

    int count = 0;
    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){

            mat_image.at<Vec3b>(j,i) [0] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [1] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [2] = image[count]; count++;

        }

    std::ostringstream path ;
    path <<"/home/miky/ClionProjects/tesi_watermarking/img/"<<nameImage<<".png";
    cv::imwrite(path.str(), mat_image);
}

void stereo_watermarking::show_doubleImage(double * image, int width, int height, string nameImage){

    int count = 0;
    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){

            mat_image.at<Vec3b>(j,i) [0] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [1] = image[count]; count++;
            mat_image.at<Vec3b>(j,i) [2] = image[count]; count++;

        }
    imshow(nameImage, mat_image);
    waitKey(0);
}


void stereo_watermarking::show_floatImage(float ** image, int width, int height, string nameImage){

    int count = 0;
    cv::Mat mat_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++){

            mat_image.at<Vec3b>(j,i) [0] = image[j][i];
            mat_image.at<Vec3b>(j,i) [1] = image[j][i];
            mat_image.at<Vec3b>(j,i) [2] = image[j][i];

        }
    imshow(nameImage, mat_image);
    waitKey(0);
}