#include <iostream>
#include <opencv2/core/core.hpp>
#include "dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include "./disparity_computation/stereo_matching.h"
#include "./disparity_optimization/disp_opt.h"
#include "./disparity_optimization/occlusions_handler.h"
#include "./right_view_computation/right_view.h"
#include "disparity_optimization/disp_opt.h"
#include <limits>
#include <cstddef>
#include <iostream>
#include <fstream>

//includes watermarking
#include "./img_watermarking/watermarking.h"

//libconfig
#include <libconfig.h++>
#include "./config/config.hpp"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;


int main() {


    //CONFIG SETTING//
    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();

    int wsize = pars.wsize;
    int tilesize=pars.tilesize;
    float power=pars.power;
    bool clipping=pars.clipping;
    bool flagResyncAll=pars.flagResyncAll;
    int tilelistsize=pars.tilelistsize;


    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();

    bool masking=generalPars.masking;
    std::string passwstr=generalPars.passwstr;
    std::string passwnum=generalPars.passwnum;


    /*  kz_disp PARAMETERS :
     *
     * lambda = 15.8
     * k = 79.12
     * dispMin dispMax = -77 -19   */



    /*STEP 2: FILTER DISPARITY (OUTPUT OF KZ)*/

/*
    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/kz_disp.png");
    if (kz_disp.rows == 0){
        cout << "Empty image";
    } else {
        Disp_opt dp;
        dp.disparity_filtering(kz_disp);
    }
*/


/*
    cv::Mat gkz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/kz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/kz_disp.png",CV_LOAD_IMAGE_COLOR);
    cv::imshow("Grey disp", gkz_disp);
    cv::imshow("Color disp", kz_disp);
*/



    /*STEP 3: NORMALIZE DISPARITY (OUTPUT OF KZ)*/

/*
    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/kz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (kz_disp.rows == 0){
        cout << "Empty image";
    } else {
        Disp_opt dp;
        dp.disparity_normalization(kz_disp);
    }
*/



    /*STEP 4: RECONSTRUCT RIGHT VIEW*/


    cv::Mat left = imread("/home/miky/ClionProjects/tesi_watermarking//img/l.png",
                          CV_LOAD_IMAGE_COLOR);
    // our disp
    cv::Mat disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    // ground truth
    // cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/left/frame_1.png",
    //                      CV_LOAD_IMAGE_GRAYSCALE);

    Right_view rv;
    rv.right_reconstruction(left, disp);



    /*ENHANCING OCCLUSIONS*/

/*
    cv::Mat f_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", CV_LOAD_IMAGE_COLOR);
    Disp_opt dp;
    dp.occlusions_enhancing(f_disp);
*/


/*

    */
/*DIFFERENCE BETWEEN GROUND TRUTH DISPARITY AND OUR DISPARITY *//*


    // ground truth disparity
    cv::Mat gt = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png",
                            CV_LOAD_IMAGE_GRAYSCALE);
    // our disparity
    cv::Mat nkz = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png",
                             CV_LOAD_IMAGE_GRAYSCALE);
    // difference matrix
    cv::Mat diff = cv::Mat::zeros(gt.rows, gt.cols, CV_8UC1);
    int count = 0;
    int min = 255, max = 0;
    for(int j=0;j< gt.rows;j++)
    {
        for (int i=0;i< gt.cols;i++)
        {
            int new_value;
            // handle occlusion points
            if (nkz.at<uchar>(j, i) != 0 ) {
                new_value = nkz.at<uchar>(j, i) - gt.at<uchar>(j, i);
                // if the disparity is equal, new_value = 0 -> black pixel
                diff.at<uchar>(j, i) = abs(new_value);
                if ( new_value < 0) count ++;
                if ( abs(new_value) < min ) min =  abs(new_value);
                if ( abs(new_value) > max ) max =  abs(new_value);
            } else {
                new_value = 255;
                diff.at<uchar>(j, i) = new_value;
            }
        }
    }
  //  cout << "Ground truth values that are bigger than our disparity values: " << count << endl;
  //  cout << "Min difference value: " << min << " Max difference value: " << max << endl;
    cv::imshow("Diff ", diff);
   // imwrite("/home/bene/ClionProjects/tesi_watermarking/img/diff_disp.png", diff);
    cv::Mat diff_disp = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/diff_disp.png",
                            CV_LOAD_IMAGE_GRAYSCALE);
    cv::imshow("Diff disp", diff_disp);
*/

            /* Found min and max value of difference disparity map */


/*
    cv::Mat gt = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/diff_disp.png",
                            CV_LOAD_IMAGE_GRAYSCALE);
    int min,max, d;
    min = 256;
    max = 0;
    for(int j=0;j< gt.rows;j++) {
        for (int i = 0; i < gt.cols; i++) {
            d = gt.at<uchar>(j,i);
            if ( d > max) max = d;
            if ( d < min ) min = d;
        }
    }
    cout << "Min: " << min << " Max: " << max;

    cv::imshow("Gt", gt);

   */




//    waitKey(5); // 300000 = 5 minutes




/*
    cv::Mat occluded = imread("/home/miky/ClionProjects/tesi_watermarking/img/filtered_bw.png");
    cv::Mat occluded_gt = imread("/home/miky/Scrivania/tsukuba_occlusion_L_00001.png");

    cout << occlusions_handler::getSimilarity(occluded,occluded_gt);
*/

    //CARICA IMMAGINE DA MARCHIARE
//    cv::Mat image_to_mark = cv::imread("/home/miky/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE );
    cv::Mat image_to_mark = cv::imread("/home/miky/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR );


    //IF GREYSCALE IMAGE
//    cv::Mat new_image = cv::Mat::zeros(512, 512, CV_8UC1);

//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 512; i++){
//            new_image.at<uchar>(j, i) = image_to_mark.at<uchar>(j, i);}
//

    //IF COLOUR IMAGE
    cv::Mat new_image = cv::Mat::zeros(512, 512, CV_8UC3);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            new_image.at<Vec3b>(j,i) [0] = image_to_mark.at<Vec3b>(j,i) [0];
            new_image.at<Vec3b>(j,i) [1] = image_to_mark.at<Vec3b>(j,i) [1];
            new_image.at<Vec3b>(j,i) [2] = image_to_mark.at<Vec3b>(j,i) [2];
        }


    unsigned char *squared_image = new_image.data; //SPERO SIA GIUSTO PER LE COLOUR

   /* algoritmo di watermarking  */

    unsigned char *output_img = new unsigned char[512 * 512 ];
//    memcpy(output_img, squared_image,512*512);

    Watermarking image_watermarking;

    //random binary watermark
    int watermark[wsize];
    for (int i = 0; i < wsize; i++){
        int b = rand() % 2;
        watermark[i]=b;
    }

    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
//    image_watermarking.setParameters(watermark,64,0,0.1,0,0,NULL,0);
    image_watermarking.setPassword(passwstr,passwnum);
    output_img = image_watermarking.insertWatermark(squared_image,512,512);



    int count=0;

    //IF GREYSCALE IMAGE
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 512; i++){
//            image_to_mark.at<uchar>(j, i) = output_img[count];
//            count ++;
//        }

    //IF COLOUR IMAGE
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            image_to_mark.at<Vec3b>(j,i) [0] = output_img[count]; count++;
            image_to_mark.at<Vec3b>(j,i) [1] = output_img[count]; count++;
            image_to_mark.at<Vec3b>(j,i) [2] = output_img[count]; count++;
        }



    rv.right_reconstruction(image_to_mark,disp );

//    int c=0;
//    int diversi=0;
//    for (int i=0;i<512;i++)
//        for(int j=0;j<512;j++) {
//            if (output_img[c] != squared_image[c])
//                diversi++;
//            c++;
//        }
//    cout<<diversi;



    return 0;


}









//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf