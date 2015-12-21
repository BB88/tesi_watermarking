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

using namespace std;
using namespace cv;
using namespace cv::datasets;

/**
 * sobel_filtering(..)
 *
 * compute the sobel filtering of an image
 *
 * @params src: image to process
 * @return filtered image
 */
cv::Mat stereo_watermarking::sobel_filtering(cv::Mat src){
    cv::Mat  src_gray;
    cv::Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    if ( src.channels() == 3)
        cvtColor( src, src_gray, COLOR_RGB2GRAY );
    else src.copyTo(src_gray);
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    return grad;
}

/**
 * show_double_mat(..)
 *
 * show a double image
 *
 * @params A: image to show
 * @params width: image width
 * @params height: image height
 * @params window_name: image name for the window
 */
void stereo_watermarking::show_double_mat(int width,int height,double** A,std::string window_name){
    cv::Mat mat =  cv::Mat::zeros(width, height, CV_32F);
    for(int x = 0; x < width;++x){
        for(int y = 0; y < height; ++y){
            mat.at<float>(x,y)=A[x][y];
        }
    }
    std::ostringstream path ;
    path <<"/home/miky/Scrivania/images/dft/"<< window_name<<".png";
    cv::imwrite(path.str(),mat);
    imshow(window_name,mat);
    waitKey(0);
    return;
}

/**
 * writeToFile(..)
 *
 *write a double array to a .txt file
 *
 * @params m: array to write
 * @params lenght: array lenght
 * @params filepath: destination file
 */
void stereo_watermarking::writeToFile(double* m,int lenght, std::string filepath)
{
    ofstream fout(filepath);
    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }
    for(int i=0; i<lenght; i++)
    {
        fout << std::fixed << std::setprecision(8) <<m[i]<<"\t";
        fout<<endl;
    }
    fout.close();
}

/**
 * writeMatToFile(..)
 *
 *write a double matrice to a .txt file
 *
 * @params m: matrice to write
 * @params lenght: array lenght
 * @params filepath: destination file
 */
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

/**
 * writefloatMatToFile(..)
 *
 *write a float matrice to a .txt file
 *
 * @params m: matrice to write
 * @params lenght: array lenght
 * @params filepath: destination file
 */
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

/**
 * show_ucharImage(..)
 *
 * show an uchar image
 *
 * @params image: image to show
 * @params width: image width
 * @params height: image height
 * @params nameImage: image name for the window
 * @params channels: number of channel considered
 */
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

/**
 * show_doubleImage(..)
 *
 * show a double image
 *
 * @params image: image to show
 * @params width: image width
 * @params height: image height
 * @params nameImage: image name for the window
 */
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

/**
 * show_floatImage(..)
 *
 * show a float image
 *
 * @params image: image to show
 * @params width: image width
 * @params height: image height
 * @params nameImage: image name for the window
 */
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

/**
 * compute_luminance(..)
 *
 * compute the rgb decomposition of an image
 *
 * @params image: image to process
 * @params dim: image dimension (dimxdim)
 * @params flag: 1 if the image has 3 channels
 * @output imr:
 * @output img:
 * @output imb:
 * @output imyout:
 * @output imc2:
 * @output imc3:
 */
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