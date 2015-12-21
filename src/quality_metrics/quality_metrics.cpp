//
// Created by miky on 26/08/15.
//

#include "quality_metrics.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;
using namespace cv;

namespace qm
{
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 3255)
#define C3 (float) (0.015 * 255 * 0.03  * 255)


    // sigma on block_size
    double sigma(Mat & m, int i, int j, int block_size)
    {
        double sd = 0;

        Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
        Mat m_squared(block_size, block_size, CV_64F);

        multiply(m_tmp, m_tmp, m_squared);

        // E(x)
        double avg = mean(m_tmp)[0];
        // E(x²)
        double avg_2 = mean(m_squared)[0];


        sd = sqrt(avg_2 - avg * avg);

        return sd;
    }

    // Covariance
    double cov(Mat & m1, Mat & m2, int i, int j, int block_size)
    {
        Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
        Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
        Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));


        multiply(m1_tmp, m2_tmp, m3);

        double avg_ro 	= mean(m3)[0]; // E(XY)
        double avg_r 	= mean(m1_tmp)[0]; // E(X)
        double avg_o 	= mean(m2_tmp)[0]; // E(Y)


        double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

        return sd_ro;
    }

    // Mean squared error
    double eqm(Mat & img1, Mat & img2)
    {
        int i, j;
        double eqm = 0.0;
        int height = img1.rows;
        int width = img1.cols;

        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++)
                eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));

        eqm /= height * width;

        return eqm;
    }

    double color_eqm(Mat & img1, Mat & img2)
    {
        int i, j;
        double eqm = 0.0;
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        int height = img1.rows;
        int width = img1.cols;
       // cout << (double)img1.at<Vec3b>(i, j)[0] - (double)img2.at<Vec3b>(i, j)[0]<<endl;
        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++) {
                s0 += ((double)img1.at<Vec3b>(i, j)[0] - (double)img2.at<Vec3b>(i, j)[0]) * ((double)img1.at<Vec3b>(i, j)[0] - (double)img2.at<Vec3b>(i, j)[0]);
                s1 += ((double)img1.at<Vec3b>(i, j)[1] - (double)img2.at<Vec3b>(i, j)[1]) * ((double)img1.at<Vec3b>(i, j)[1] - img2.at<Vec3b>(i, j)[1]);
                s2 += (img1.at<Vec3b>(i, j)[2] - img2.at<Vec3b>(i, j)[2]) * (img1.at<Vec3b>(i, j)[2] - img2.at<Vec3b>(i, j)[2]);

            /*    cout << s0 <<endl;
                cout << s1 <<endl;
                cout << s2 <<endl;*/
            }
        eqm = s0 + s1 + s2;
        eqm /= 3 * height * width;

        return eqm;
    }




    /**
     *	Compute the PSNR between 2 images
     */
    double psnr(Mat & img_src, Mat & img_compressed, int blocksize)
    {
        int D = 255;
        return (10 * log10((D*D)/eqm(img_src, img_compressed)));
    }


    double video_psnr(Mat & img_src, Mat & img_compressed)
    {
        int D = 255;
     //   double value = img_src.at<Vec3b>(23,53)[0];
   //     std::cout<<value<<endl;
        return (10 * log10((D*D)/color_eqm(img_src,img_compressed)));
    }



    /**
      *	Compute the average PSNR between 2 videos
      */
    double avg_psnr(std::string origVideo,std::string compressVideo ){
        double avg_psnr = 0.0;
        VideoCapture capOrig(origVideo);
        if (!capOrig.isOpened()) {  // check if we succeeded
            cout << "Could not open the origVideo to read " << endl;
            return -1;
        }
        VideoCapture capCompr(compressVideo);
        if (!capCompr.isOpened()) {  // check if we succeeded
            cout << "Could not open the compressVideo to read " << endl;
            return -1;
        }
        int first_frame = 0;
        int last_frame = 1800;
        const int STEP = 60; // compare frame every STEP frame

        cv::Mat Origframe;
        cv::Mat Comprframe;

        for (int i = first_frame; i < last_frame; i++) {

            if(i%STEP==0){
                capOrig >> Origframe;
                capCompr >> Comprframe;
                if (Origframe.empty()) break;
                if (Comprframe.empty()) break;
                double psnr = video_psnr(Origframe, Comprframe);
                avg_psnr += psnr;

            }
           else {
                capOrig >> Origframe;
                capCompr >> Comprframe;
                if (Origframe.empty()) break;
                if (Comprframe.empty()) break;
                double psnr = video_psnr(Origframe, Comprframe);
            }
        }
        avg_psnr = avg_psnr/ (last_frame/STEP);
//        avg_psnr = avg_psnr/ last_frame;
        cout<< "average PSNR : "<< avg_psnr<<endl;
        return  avg_psnr;
    }


    /**
     * Compute the SSIM between 2 images
     */
    double ssim(Mat & img_src, Mat & img_compressed, int block_size, bool show_progress  )
    {
        double ssim = 0;

        int nbBlockPerHeight 	= img_src.rows / block_size;
        int nbBlockPerWidth 	= img_src.cols / block_size;

        for (int k = 0; k < nbBlockPerHeight; k++)
        {
            for (int l = 0; l < nbBlockPerWidth; l++)
            {
                int m = k * block_size;
                int n = l * block_size;

                double avg_o 	= mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double avg_r 	= mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double sigma_o 	= sigma(img_src, m, n, block_size);
                double sigma_r 	= sigma(img_compressed, m, n, block_size);
                double sigma_ro	= cov(img_src, img_compressed, m, n, block_size);

                ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));

            }
            // Progress
            if (show_progress)
                cout << "\r>>SSIM [" << (int) ((( (double)k) / nbBlockPerHeight) * 100) << "%]";
        }
        ssim /= nbBlockPerHeight * nbBlockPerWidth;

        if (show_progress)
        {
            cout << "\r>>SSIM [100%]" << endl;
            cout << "SSIM : " << ssim << endl;
        }

        return ssim;
    }

    void compute_quality_metrics(char * file1, char * file2, int block_size)
    {

        Mat img_src;
        Mat img_compressed;

        // Loading pictures
        img_src = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
        img_compressed = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);


        img_src.convertTo(img_src, CV_64F);
        img_compressed.convertTo(img_compressed, CV_64F);

        int height_o = img_src.rows;
        int height_r = img_compressed.rows;
        int width_o = img_src.cols;
        int width_r = img_compressed.cols;

        // Check pictures size
        if (height_o != height_r || width_o != width_r)
        {
            cout << "Images must have the same dimensions" << endl;
            return;
        }

        // Check if the block size is a multiple of height / width
        if (height_o % block_size != 0 || width_o % block_size != 0)
        {
            cout 	<< "WARNING : Image WIDTH and HEIGHT should be divisible by BLOCK_SIZE for the maximum accuracy" << endl
            << "HEIGHT : " 		<< height_o 	<< endl
            << "WIDTH : " 		<< width_o	<< endl
            << "BLOCK_SIZE : " 	<< block_size 	<< endl
            << endl;
        }

        double ssim_val = ssim(img_src, img_compressed, block_size,false);
        double psnr_val = psnr(img_src, img_compressed, block_size);

        cout << "SSIM : " << ssim_val << endl;
        cout << "PSNR : " << psnr_val << endl;
    }

    /**
     * MQdepth(..)
     *
     * compute the mean value of the RR quality metrics Qdepth
     *
     * @params depth_original: original disparity map
     * @params depth_wat: watermarked disparity map
     * @params depth_edge: edge of the disparity map
     * @params depth_w_edge: edge of the watermarked disparity map
     * @params block_size: size of the blocks to process
     * @params show_progress
     * @return mqd; mean of Qdepth metric     *
     *
     */
    double MQdepth(Mat & depth_original, Mat & depth_wat, Mat & depth_edge, Mat & depth_w_edge, int block_size, bool show_progress ){

            double mqd = 0;
            cv::Mat depth_o,depth_w,depth_e,depth_ew;
            depth_original.convertTo(depth_o, CV_64F);
            depth_wat.convertTo(depth_w, CV_64F);
            depth_edge.convertTo(depth_e, CV_64F);
            depth_w_edge.convertTo(depth_ew, CV_64F);
            int nbBlockPerHeight 	= depth_original.rows / block_size;
            int nbBlockPerWidth 	= depth_original.cols / block_size;
            for (int k = 0; k < nbBlockPerHeight; k++)
            {
                for (int l = 0; l < nbBlockPerWidth; l++)
                {
                    int m = k * block_size;
                    int n = l * block_size;
                    double avg_o 	= mean(depth_o(Range(k, k + block_size), Range(l, l + block_size)))[0];
                    double avg_r 	= mean(depth_w(Range(k, k + block_size), Range(l, l + block_size)))[0];
                    double sigma_o 	= sigma(depth_o, m, n, block_size);
                    double sigma_r 	= sigma(depth_w, m, n, block_size);
                    double sigma_ow = sigma(depth_e, m, n, block_size);
                    double sigma_rw = sigma(depth_ew, m, n, block_size);
                    double sigma_ro	= cov(depth_e, depth_ew, m, n, block_size); //only thing that change wrt the original SSIM formula
                    mqd += ((2*avg_o*avg_r+C1)/((avg_o*avg_o)+(avg_r*avg_r) + C1)) * ((2*sigma_o*sigma_r+C2)/((sigma_o*sigma_o)*(sigma_r*sigma_r)+C2)) * ((sigma_ro+C3)/(sigma_ow*sigma_rw+C3));

                }
                // Progress
//                if (show_progress)
//                    cout << "\r>>MQdepth [" << (int) ((( (double)k) / nbBlockPerHeight) * 100) << "%]";
            }
            mqd /= nbBlockPerHeight * nbBlockPerWidth;
//            if (show_progress)
//            {
////                cout << "\r>>MQdepth [100%]" << endl;
//                cout << "MQdepth : " << mqd << endl;
//            }

//            cout<<" depth return \"<<endl;
            return mqd;

    }

    /**
     * MQcolor(..)
     *
     * compute the mean value of the RR quality metrics Qcolor
     *
     * @params color_original: original view
     * @params color_wat: watermarked view
     * @params depth_edge: edge of the disparity map
     * @params color_w_edge: edge of the watermarked view
     * @params color_edge: edge of the original view
     * @params block_size: size of the blocks to process
     * @params show_progress
     * @return mqc; mean of Qcolor metric     *
     *
     */
    double MQcolor(Mat &color_original, Mat &color_wat, Mat & depth_edge, Mat &color_w_edge, Mat & color_edge, int block_size, bool show_progress ){

        double mqc = 0;
        cv::Mat color_o,color_w,color_e,depth_e,color_ew;
        color_original.convertTo(color_o, CV_64F);
        color_wat.convertTo(color_w, CV_64F);
        depth_edge.convertTo(depth_e, CV_64F);
        color_w_edge.convertTo(color_ew, CV_64F);
        color_edge.convertTo(color_e, CV_64F);

        int nbBlockPerHeight 	= color_original.rows / block_size;
        int nbBlockPerWidth 	= color_original.cols / block_size;

        for (int k = 0; k < nbBlockPerHeight; k++)
        {
            for (int l = 0; l < nbBlockPerWidth; l++)
            {
                int m = k * block_size;
                int n = l * block_size;
                double avg_o 	= mean(color_o(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double avg_r 	= mean(color_w(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double sigma_o 	= sigma(color_o, m, n, block_size);
                double sigma_r 	= sigma(color_w, m, n, block_size);
                double sigma_de = sigma(depth_e, m, n, block_size);
                double sigma_rwe = sigma(color_ew, m, n, block_size);
                double sigma_ro	= cov(depth_e, color_ew, m, n, block_size); //only thing that change wrt the original SSIM formula
                mqc += ((2*avg_o*avg_r+C1)/((pow(avg_o,2) + pow(avg_r,2) + C1))) * ((2*sigma_o*sigma_r+C2)/(pow(sigma_o,2)*pow(sigma_r,2)+C2)) * ((sigma_ro+C3)/(sigma_de*sigma_rwe+C3));

            }
            // Progress
//            if (show_progress)
//                cout << "\r>>MQcolor [" << (int) ((( (double)k) / nbBlockPerHeight) * 100) << "%]";
        }
        mqc /= nbBlockPerHeight * nbBlockPerWidth;
//        if (show_progress)
//        {
//            cout << "\r>>MQcolor [100%]" << endl;
//            cout << "MQcolor : " << mqc << endl;
//        }
        return mqc;

    }

}