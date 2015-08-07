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
#include "./img_watermarking/imgwat.h"
#include "./img_watermarking/allocim.h"

//grapfh cuts
#include "./graphcuts/image.h"
#include "./graphcuts/match.h"
#include "./graphcuts/utils.h"
#include "./graphcuts/io_png.h"

//libconfig
#include <libconfig.h++>
#include "./config/config.hpp"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;
using namespace graph_cuts_utils;


int main() {


    /*CONFIG SETTINGS*/

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


    /* GRAPH CUTS DISPARITY COMPUTATION*/
/*

    std::ostringstream ss;

    std::string img1_path =  "/home/miky/ClionProjects/tesi_watermarking/img/l.png";
    std::string img2_path =  "/home/miky/ClionProjects/tesi_watermarking/img/r.png";
    Match::Parameters params = { // Default parameters
            Match::Parameters::L2, 1, // dataCost, denominator
            8, -1, -1, // edgeThresh, lambda1, lambda2 (smoothness cost)
            -1,        // K (occlusion cost)
            4, false   // maxIter, bRandomizeEveryIteration
    };
    float K=-1, lambda=-1, lambda1=-1, lambda2=-1;
    params.dataCost = Match::Parameters::L1;
//      params.dataCost = Match::Parameters::L2;

    GeneralImage im1 = (GeneralImage)imLoad(IMAGE_GRAY, img1_path.c_str());
    GeneralImage im2 = (GeneralImage)imLoad(IMAGE_GRAY, img2_path.c_str());
    bool color = false;
    if(graph_cuts_utils::isGray((RGBImage)im1) && graph_cuts_utils::isGray((RGBImage)im2)) {
        color=false;
        graph_cuts_utils::convert_gray(im1);
        graph_cuts_utils::convert_gray(im2);
    }

    Match m(im1, im2, color);
//
//////    // Disparity
    int dMin=-77, dMax=-19;
//
    m.SetDispRange(dMin, dMax);

    time_t seed = time(NULL);
    srand((unsigned int)seed);

    graph_cuts_utils::fix_parameters(m, params, K, lambda, lambda1, lambda2);

    m.KZ2();

//        m.SaveXLeft(argv[5]);

    m.SaveScaledXLeft("/home/miky/Scrivania/disp.png", false);

    cv::Mat disp = imread("/home/miky/Scrivania/disp.png");
    imshow("kz disp",disp);
    waitKey(0);
*/

 /*  kz_disp PARAMETERS */
/*

     *
     * lambda = 15.8
     * k = 79.12
     * dispMin dispMax = -77 -19

*/

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

    // path clion /home/bene/ClionProjects/tesi_watermarking/img/
    // path Scrivania /home/bene/Scrivania/

    /*STEP 3: NORMALIZE DISPARITY (OUTPUT OF KZ)*/

/*
    cv::Mat nkz_disp;
    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/wm_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (kz_disp.rows == 0){
        cout << "Empty image";
    } else {
        Disp_opt dp;
        dp.disparity_normalization(kz_disp, nkz_disp);
    }
*/


    /*DIFFERENCE BETWEEN GROUND TRUTH DISPARITY AND OUR DISPARITY */

/*
    // ground truth disparity
    cv::Mat gt = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp.png",
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
            if (nkz_disp.at<uchar>(j, i) != 0 ) {
                new_value = nkz_disp.at<uchar>(j, i) - gt.at<uchar>(j, i);
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
    // cout << "Ground truth values that are bigger than our disparity values: " << count << endl;
    // cout << "Min difference value: " << min << " Max difference value: " << max << endl;
    // cv::imshow("Diff disp", diff);
    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/diff_wmdisp.png", diff);
*/

    /*comparison between our disparity: the one with the mark and the one without */

/*

    cv::Mat nkz2 = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png",
                            CV_LOAD_IMAGE_GRAYSCALE);
    // our disparity
    cv::Mat wnkz = cv::imread("/home/bene/Scrivania/nwm_disp.png",
                             CV_LOAD_IMAGE_GRAYSCALE);
    // difference matrix
    cv::Mat diff2 = cv::Mat::zeros(gt.rows, gt.cols, CV_8UC1);
    int count2 = 0;
 //   int min = 255, max = 0;
    for(int j=0;j< gt.rows;j++)
    {
        for (int i=0;i< gt.cols;i++)
        {
            int new_value;
            // handle occlusion points
                new_value = nkz2.at<uchar>(j, i) - wnkz.at<uchar>(j, i);
                // if the disparity is equal, new_value = 0 -> black pixel
                diff2.at<uchar>(j, i) = abs(new_value);
                if ( new_value < 0) count ++;
                if ( abs(new_value) < min ) min =  abs(new_value);
                if ( abs(new_value) > max ) max =  abs(new_value);

        }
    }
    cv::imshow("Diff our disp", diff2);
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


    /*STEP 4: RECONSTRUCT RIGHT VIEW*/

/*
    cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);
    // our disp
    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    // ground truth
    // cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/gt_disp", CV_LOAD_IMAGE_GRAYSCALE);
    // watermarked left view
    // cv::Mat left_marked = imread("/home/bene/Scrivania/l_piccola.png", CV_LOAD_IMAGE_COLOR);
    Right_view rv;
    rv.right_reconstruction(left, disp);
*/


    /*ENHANCING OCCLUSIONS*/

/*
    cv::Mat f_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", CV_LOAD_IMAGE_COLOR);
    Disp_opt dp;
    dp.occlusions_enhancing(f_disp);
*/

//    waitKey(0); // 300000 = 5 minutes


    /* GET SIMILARITY BETWEEN OCCLUDED DISPARITIES (GT AND OURS) */

/*
    cv::Mat occluded = imread("/home/bene/ClionProjects/tesi_watermarking/img/filtered_bw.png");
    cv::Mat occluded_gt = imread("/home/bene/Scrivania/tsukuba_occlusion_L_00001.png");
    cout << occlusions_handler::getSimilarity(occluded,occluded_gt);
*/

    /*  BEGIN LEFT IMAGE WATERMARKING */

    cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);

   /* if greyscale image (disparity) */
/*
    cv::Mat image_to_mark = cv::imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE );
    left.copyTo(image_to_mark);
    cv::Mat new_image = cv::Mat::zeros(512, 512, CV_8UC1);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            new_image.at<uchar>(j, i) = image_to_mark.at<uchar>(j, i);}
*/
//     imshow("left", left);
   /* if colour image (left view) */
    cv::Mat image_to_mark = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);
    left.copyTo(image_to_mark);
    // resize left image to be squared
    cv::Mat new_image = cv::Mat::zeros(512, 512, CV_8UC3);
    int new_index = 640 - 512 - 1;
    for (int j = 0; j < 480; j++) // 640 - 512 - 1
        for (int i = 0; i < 512; i++){

            new_image.at<Vec3b>(j,i) [0] = image_to_mark.at<Vec3b>(j,new_index + i) [0];
            new_image.at<Vec3b>(j,i) [1] = image_to_mark.at<Vec3b>(j,new_index + i) [1];
            new_image.at<Vec3b>(j,i) [2] = image_to_mark.at<Vec3b>(j,new_index + i) [2];
        }
//    imshow("Left squared", new_image);
    unsigned char *squared_image = new_image.data;
    unsigned char *output_img = new unsigned char[512 * 512 ];
    // memcpy(output_img, squared_image,512*512);
    Watermarking image_watermarking;

    //random binary watermark
    int watermark[64];
    for (int i = 0; i < 64; i++){
        int b = rand() % 2;
        watermark[i]=b;
    }

    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
 //   image_watermarking.setParameters(watermark,64,0,0.8,0,0,NULL,0);
    image_watermarking.setPassword(passwstr,passwnum);
    output_img = image_watermarking.insertWatermark(squared_image,512,512);
    int count=0;

    /* if greyscale image (restore the original size of the image) */
/*
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            image_to_mark.at<uchar>(j, i) = output_img[count];
            count ++;
        }
*/

    /* if colour image (restore the original size of the image) */
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            image_to_mark.at<Vec3b>(j,new_index + i) [0] = output_img[count]; count++;
            image_to_mark.at<Vec3b>(j,new_index + i) [1] = output_img[count]; count++;
            image_to_mark.at<Vec3b>(j,new_index + i) [2] = output_img[count]; count++;
        }
    cv::imshow("left_marked", image_to_mark);
    waitKey(0);

    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png", image_to_mark);

    /* END LEFT IMAGE WATERMARKING */


    /* SHOW LEFT IMAGE WATERMARK*/


    cv::Mat mark = cv::Mat::zeros(left.rows, left.cols , CV_8UC3);;
    image_to_mark.copyTo(mark);

    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            mark.at<Vec3b>(j,i) [0] = abs(image_to_mark.at<Vec3b>(j,i)[0]-left.at<Vec3b>(j,i)[0]);
            mark.at<Vec3b>(j,i) [1] = abs(image_to_mark.at<Vec3b>(j,i)[1]-left.at<Vec3b>(j,i)[1]);
            mark.at<Vec3b>(j,i) [2] = abs(image_to_mark.at<Vec3b>(j,i)[2]-left.at<Vec3b>(j,i)[2]);
        }
    // splits the image into 3 channels and normalize to see watermark
    Mat channels[3];
    split(mark,channels);
    double mins, maxs;
    minMaxLoc(channels[0], &mins, &maxs);
    Mat chan0 = channels[0] *255 / maxs;
    Mat chan1 = channels[1] *255 / maxs;
    Mat chan2 = channels[2] *255 / maxs;
    cv::imshow("mark", chan0);
    cv::imshow("mark1", chan1);
    cv::imshow("mark2", chan2);
    waitKey(0);

    /* BEGIN RIGHT IMAGE WATERMARKING */

    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);

   /* THREE CHANNELS WATERMARKING */

/*
    cv::Mat mark;
    cv::absdiff(image_to_mark,left,mark);
    cv::Mat warped_mark = cv::Mat::zeros(480, 640, CV_8UC3);
 // left view watermark extraction and disparity-based modification
    double min_disp, max_disp;
    minMaxLoc(disp, &min_disp, &max_disp);
    int d;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            d = disp.at<uchar>(j,i);
            warped_mark.at<Vec3b>(j,i+d) [0] = mark.at<Vec3b>(j,i)[0];
            warped_mark.at<Vec3b>(j,i+d) [1] = mark.at<Vec3b>(j,i)[1];
            warped_mark.at<Vec3b>(j,i+d) [2] = mark.at<Vec3b>(j,i)[2];
        }
    Mat channelsW[3];
    split(mark,channelsW);
    double min, max;
    minMaxLoc(channelsW[0], &min, &max);
    cv::Mat wat_to_show0 = channelsW[0] *255 / max;
    cv::Mat wat_to_show1 = channelsW[1] *255 / max;
    cv::Mat wat_to_show2 = channelsW[2] *255 / max;
    cv::imshow("warped_mark0", wat_to_show0);
    cv::imshow("warped_mark1", wat_to_show1);
    cv::imshow("warped_mark2", wat_to_show2);
    waitKey(0);
    cv::Mat right_watermarked;
    right.copyTo(right_watermarked);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            right_watermarked.at<Vec3b>(j,i) [0] = right_watermarked.at<Vec3b>(j,i) [0] + warped_mark.at<Vec3b>(j,i)[0];
            right_watermarked.at<Vec3b>(j,i) [1] = right_watermarked.at<Vec3b>(j,i) [1] + warped_mark.at<Vec3b>(j,i)[1];
            right_watermarked.at<Vec3b>(j,i) [2] = right_watermarked.at<Vec3b>(j,i) [2] + warped_mark.at<Vec3b>(j,i)[2];
        }
    cv::imshow("right_watermarked", righ t_watermarked);
    waitKey(0);
    imwrite("/home/bene/Scrivania/right_marked.png", right_watermarked);
*/

    /* LUMINANCE WATERMARKING */

    unsigned char *watermarked_image;
    watermarked_image=image_to_mark.data; //copio i valori dell'immagine sinistra

    unsigned char *left_image;
    left_image=left.data;

    unsigned char *right_image;
    right_image=right.data;

    unsigned char **imrw;	// matrici delle componenti RGB
    unsigned char **imgw;
    unsigned char **imbw;
    float   **imyoutw;		// immagine
    float **imc2w;			// matrice di crominanza c2
    float **imc3w;

    imyoutw = AllocIm::AllocImFloat(480, 640);
    imc2w = AllocIm::AllocImFloat(480, 640);
    imc3w = AllocIm::AllocImFloat(480, 640);
    imrw = AllocIm::AllocImByte(480, 640);
    imgw = AllocIm::AllocImByte(480, 640);
    imbw = AllocIm::AllocImByte(480, 640);

    unsigned char **imrl;	// matrici delle componenti RGB
    unsigned char **imgl;
    unsigned char **imbl;
    float   **imyoutl;			// matrice di luminanza
    float **imc2l;			// matrice di crominanza c2
    float **imc3l;

    imyoutl = AllocIm::AllocImFloat(480, 640);
    imc2l = AllocIm::AllocImFloat(480, 640);
    imc3l = AllocIm::AllocImFloat(480, 640);
    imrl = AllocIm::AllocImByte(480, 640);
    imgl = AllocIm::AllocImByte(480, 640);
    imbl = AllocIm::AllocImByte(480, 640);


    unsigned char **imrr;	// matrici delle componenti RGB
    unsigned char **imgr;
    unsigned char **imbr;
    float   **imyoutr;			// matrice di luminanza
    float **imc2r;			// matrice di crominanza c2
    float **imc3r;

    imyoutr = AllocIm::AllocImFloat(480, 640);
    imc2r = AllocIm::AllocImFloat(480, 640);
    imc3r = AllocIm::AllocImFloat(480, 640);
    imrr = AllocIm::AllocImByte(480, 640);
    imgr = AllocIm::AllocImByte(480, 640);
    imbr = AllocIm::AllocImByte(480, 640);

    int offset = 0;
    for (int i=0; i<480; i++)
        for (int j=0; j<640; j++)
        {
            imrw[i][j] = watermarked_image[offset];offset++;
            imgw[i][j] = watermarked_image[offset];offset++;
            imbw[i][j] = watermarked_image[offset];offset++;
        }
    offset = 0;
    for (int i=0; i<480; i++)
        for (int j=0; j<640; j++)
        {
            imrl[i][j] = left_image[offset];offset++;
            imgl[i][j] = left_image[offset];offset++;
            imbl[i][j] = left_image[offset];offset++;
        }
    offset = 0;
    for (int i=0; i<480; i++)
        for (int j=0; j<640; j++)
        {
            imrr[i][j] = right_image[offset];offset++;
            imgr[i][j] = right_image[offset];offset++;
            imbr[i][j] = right_image[offset];offset++;
        }


    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    image_watermarking.rgb_to_crom(imrw, imgw, imbw, 480, 640, 1, imyoutw, imc2w, imc3w);
    image_watermarking.rgb_to_crom(imrl, imgl, imbl, 480, 640, 1, imyoutl, imc2l, imc3l);
    image_watermarking.rgb_to_crom(imrr, imgr, imbr, 480, 640, 1, imyoutr, imc2r, imc3r);

    float   **watermarkY;
    watermarkY = AllocIm::AllocImFloat(480, 640);
    for (int i=0;i<480;i++)
        for (int j=0;j<640;j++){
           watermarkY[i][j] = 0;
        }


    double min_disp, max_disp;
    minMaxLoc(disp, &min_disp, &max_disp);
    int d;

    for (int i=0;i<480;i++)
        for (int j=0;j<512;j++){
            d = disp.at<uchar>(i,j);
            watermarkY[i][j+d] = abs(imyoutw[i][j+ new_index]-imyoutl[i][j+new_index]);
        }

    for (int i=0;i<480;i++)
        for (int j=0;j<640;j++)
            imyoutr[i][j] = imyoutr[i][j] + 1*(watermarkY[i][j]);


    image_watermarking.rgb_to_crom(imrr, imgr, imbr, 480, 640, -1, imyoutr, imc2r, imc3r);


    offset = 0;
    for (int i=0; i<480; i++)
        for (int j=0; j<640; j++)
        {
            right_image[offset] = imrr[i][j]; offset++;
            right_image[offset] = imgr[i][j]; offset++;
            right_image[offset] = imbr[i][j]; offset++;
        }


    cv::Mat right_watermarked=cv::Mat::zeros(480, 640, CV_8UC3);;
    right.copyTo(right_watermarked);


    count = 0;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            right_watermarked.at<Vec3b>(j,i) [0] = right_image[count]; count++;
            right_watermarked.at<Vec3b>(j,i) [1] = right_image[count]; count++;
            right_watermarked.at<Vec3b>(j,i) [2] = right_image[count]; count++;
        }

    cv::imshow("right_watermarked", right_watermarked);
    cv::imwrite("/home/bene/ClionProjects/tesi_watermarking/img/right_marked.png", right_watermarked);
    waitKey(0);
    AllocIm::FreeIm(imc2w) ;
    AllocIm::FreeIm(imc3w) ;
    AllocIm::FreeIm(imrw);
    AllocIm::FreeIm(imgw);
    AllocIm::FreeIm(imbw);

    AllocIm::FreeIm(imyoutw);

    AllocIm::FreeIm(imc2l) ;
    AllocIm::FreeIm(imc3l) ;
    AllocIm::FreeIm(imrl);
    AllocIm::FreeIm(imgl);
    AllocIm::FreeIm(imbl);

    AllocIm::FreeIm(imyoutl);

    AllocIm::FreeIm(imc2r) ;
    AllocIm::FreeIm(imc3r) ;
    AllocIm::FreeIm(imrr);
    AllocIm::FreeIm(imgr);
    AllocIm::FreeIm(imbr);

    AllocIm::FreeIm(imyoutr);
    AllocIm::FreeIm(watermarkY);


    /* END RIGHT IMAGE WATERMARKING */

    /*  SHOW RIGHT IMAGE WATERMARK */

    cv::Mat right2 = imread("/home/bene/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat mark2 = cv::Mat::zeros(right2.rows, right2.cols , CV_8UC3);
    right_watermarked.copyTo(mark2);

    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            mark2.at<Vec3b>(j,i) [0] = abs(right_watermarked.at<Vec3b>(j,i)[0]-right2.at<Vec3b>(j,i)[0]);
            mark2.at<Vec3b>(j,i) [1] = abs(right_watermarked.at<Vec3b>(j,i)[1]-right2.at<Vec3b>(j,i)[1]);
            mark2.at<Vec3b>(j,i) [2] = abs(right_watermarked.at<Vec3b>(j,i)[2]-right2.at<Vec3b>(j,i)[2]);
        }
    double min, max;
    Mat channels2[3];
//    cv::absdiff(right_watermarked,right,mark2);
    split(mark2,channels2);
    minMaxLoc(channels2[0], &min, &max);

    Mat chan0d = channels2[0] *255 / max;
    Mat chan1d = channels2[1] *255 / max;
    Mat chan2d = channels2[2] *255 / max;
    cv::imshow("mark20", chan0d);
    cv::imshow("mark21", chan1d);
    cv::imshow("mark22", chan2d);
    waitKey(0);

    /* WATERMARK DETECTION*/



    cv::Mat new_image_to_dec = cv::Mat::zeros(512, 512, CV_8UC3);
    cv::Mat image_to_dec = imread("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png");

    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            new_image_to_dec.at<Vec3b>(j,i) [0] = image_to_dec.at<Vec3b>(j,new_index +i) [0];
            new_image_to_dec.at<Vec3b>(j,i) [1] = image_to_dec.at<Vec3b>(j,new_index +i) [1];
            new_image_to_dec.at<Vec3b>(j,i) [2] = image_to_dec.at<Vec3b>(j,new_index +i) [2];
        }

    cv::imshow("Left to dec", new_image_to_dec);
     waitKey(0);

    unsigned char *squared_image_to_dec = new_image_to_dec.data;
    bool wat = image_watermarking.extractWatermark(squared_image_to_dec, 512, 512);
    cout<<wat;


    cv::Mat right_image_to_dec = cv::Mat::zeros(512, 512, CV_8UC3);
    cv::Mat right_to_dec = imread("/home/bene/ClionProjects/tesi_watermarking/img/right_marked.png");

    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 512; i++){
            right_image_to_dec.at<Vec3b>(j,i) [0] = right_to_dec.at<Vec3b>(j,i) [0];
            right_image_to_dec.at<Vec3b>(j,i) [1] = right_to_dec.at<Vec3b>(j,i) [1];
            right_image_to_dec.at<Vec3b>(j,i) [2] = right_to_dec.at<Vec3b>(j,i) [2];
        }

    cv::imshow("Right to dec", right_image_to_dec);
    waitKey(0);
    unsigned char *squared_right_to_dec = right_image_to_dec.data;
    bool wat2 = image_watermarking.extractWatermark(squared_right_to_dec, 512, 512);
    cout<<wat2;


    return 0;

}



//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf