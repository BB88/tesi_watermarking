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


    cv::Mat left = imread("/home/miky/ClionProjects/tesi_watermarking//img/l.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat right = imread("/home/miky/ClionProjects/tesi_watermarking//img/r.png",CV_LOAD_IMAGE_COLOR);
//
//    cv::imshow("left_original", left);
//    cv::imshow("right_original", right);
//    waitKey(0);

    // our disp
//    cv::Mat disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    // ground truth
    // cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/left/frame_1.png",
    //                      CV_LOAD_IMAGE_GRAYSCALE);

//    Right_view rv;
//    rv.right_reconstruction(left, disp);



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
    cv::Mat image_to_mark;
    left.copyTo(image_to_mark);


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

    cv::imshow("left_marked", image_to_mark);
    waitKey(0);

//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 640; i++){
//            image_to_mark.at<Vec3b>(j,i) [0] = abs(image_to_mark.at<Vec3b>(j,i)[0]-left.at<Vec3b>(j,i)[0]);
//            image_to_mark.at<Vec3b>(j,i) [1] = abs(image_to_mark.at<Vec3b>(j,i)[1]-left.at<Vec3b>(j,i)[1]);
//            image_to_mark.at<Vec3b>(j,i) [2] = abs(image_to_mark.at<Vec3b>(j,i)[2]-left.at<Vec3b>(j,i)[2]);
//        }


/* splits the image into 3 channels and normalize to see watermark */

//    Mat channels[3];
//    split(mark,channels);
//    double min, max;
//    minMaxLoc(channels[0], &min, &max);
//    Mat chan0 = channels[0] *255 / max;
//    Mat chan1 = channels[1] *255 / max;
//    Mat chan2 = channels[2] *255 / max;
//    cv::imshow("mark", chan0);
//    cv::imshow("mark1", chan1);
//    cv::imshow("mark2", chan2);
//    waitKey(0);
/*********************************************/
//    rv.right_reconstruction(image_to_mark,disp );

//    int c=0;
//    int diversi=0;
//    for (int i=0;i<512;i++)
//        for(int j=0;j<512;j++) {
//            if (output_img[c] != squared_image[c])
//                diversi++;
//            c++;
//        }
//    cout<<diversi;

    cv::Mat disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);

 //////////////////MARCHIATURA VISTA DESTRA NEI TRE CANALI///////////////////

//    cv::Mat mark;
//    cv::absdiff(image_to_mark,left,mark);
//
//    cv::Mat warped_mark = cv::Mat::zeros(480, 640, CV_8UC3);


 //estraggo il marchio dalla vista sinistra e lo modifico in base alla disparità
//    double min_disp, max_disp;
//    minMaxLoc(disp, &min_disp, &max_disp);
//    int d;
//
//    for (int j = 0; j < (480); j++)
//        for (int i = 0; i < 640-max_disp; i++){
//            d = disp.at<uchar>(j,i);
//            warped_mark.at<Vec3b>(j,i+d) [0] = mark.at<Vec3b>(j,i)[0];
//            warped_mark.at<Vec3b>(j,i+d) [1] = mark.at<Vec3b>(j,i)[1];
//            warped_mark.at<Vec3b>(j,i+d) [2] = mark.at<Vec3b>(j,i)[2];
//
//        }
//
////    Mat channelsW[3];
////    split(warped_mark,channelsW);
////    cv::Mat wat_to_show0 = channelsW[0] *255 / max;
////    cv::Mat wat_to_show1 = channelsW[1] *255 / max;
////    cv::Mat wat_to_show2 = channelsW[2] *255 / max;
////    cv::imshow("warped_mark0", wat_to_show0);
////    cv::imshow("warped_mark1", wat_to_show1);
////    cv::imshow("warped_mark2", wat_to_show2);
////    waitKey(0);
//
//    cv::Mat right_watermarked;
//    right.copyTo(right_watermarked);
//
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 640; i++){
//            right_watermarked.at<Vec3b>(j,i) [0] = right_watermarked.at<Vec3b>(j,i) [0] + warped_mark.at<Vec3b>(j,i)[0];
//            right_watermarked.at<Vec3b>(j,i) [1] = right_watermarked.at<Vec3b>(j,i) [1] + warped_mark.at<Vec3b>(j,i)[1];
//            right_watermarked.at<Vec3b>(j,i) [2] = right_watermarked.at<Vec3b>(j,i) [2] + warped_mark.at<Vec3b>(j,i)[2];
//        }
//
//    cv::imshow("right_watermarked", right_watermarked);
//    waitKey(0);

    ///////MARCHIATURA VISTA DESTRA NELLA LUMINANZA///////////

    unsigned char *watermarked_image;
    watermarked_image=image_to_mark.data;

    unsigned char *left_image;
    left_image=left.data;

    unsigned char *right_image;
    right_image=right.data;

    unsigned char **imr;	// matrici delle componenti RGB
    unsigned char **img;
    unsigned char **imb;
    float   **imyout;		// immagine
    float **imc2;			// matrice di crominanza c2
    float **imc3;

    imyout = AllocIm::AllocImFloat(480, 640);
    imc2 = AllocIm::AllocImFloat(480, 640);
    imc3 = AllocIm::AllocImFloat(480, 640);
    imr = AllocIm::AllocImByte(480, 640);
    img = AllocIm::AllocImByte(480, 640);
    imb = AllocIm::AllocImByte(480, 640);

    unsigned char **imrl;	// matrici delle componenti RGB
    unsigned char **imgl;
    unsigned char **imbl;
    float   **imyoutl;			// immagine
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
    float   **imyoutr;			// immagine
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
            imr[i][j] = watermarked_image[offset];offset++;
            img[i][j] = watermarked_image[offset];offset++;
            imb[i][j] = watermarked_image[offset];offset++;
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
    image_watermarking.rgb_to_crom(imr, img, imb, 480, 640, 1, imyout, imc2, imc3);
    image_watermarking.rgb_to_crom(imrl, imgl, imbl, 480, 640, 1, imyoutl, imc2l, imc3l);
    image_watermarking.rgb_to_crom(imrr, imgr, imbr, 480, 640, 1, imyoutr, imc2r, imc3r);

    float   **watermarkY;
    watermarkY = AllocIm::AllocImFloat(480, 640);

    double min_disp, max_disp;
    minMaxLoc(disp, &min_disp, &max_disp);
    int d;

    for (int i=0;i<480;i++)
        for (int j=0;j<640-max_disp;j++){
            d = disp.at<uchar>(i,j);
            watermarkY[i][j+d] = abs(imyout[i][j]-imyoutl[i][j]);
        }

    for (int i=0;i<480;i++)
        for (int j=0;j<640;j++)
            imyoutr[i][j] = imyoutr[i][j]+watermarkY[i][j];


    image_watermarking.rgb_to_crom(imrr, imgr, imbr, 480, 640, -1, imyoutr, imc2r, imc3r);


    offset = 0;
    for (int i=0; i<480; i++)
        for (int j=0; j<640; j++)
        {
            right_image[offset] = imrr[i][j]; offset++;
            right_image[offset] = imgr[i][j]; offset++;
            right_image[offset] = imbr[i][j]; offset++;
        }


    cv::Mat right_watermarked;
    right.copyTo(right_watermarked);


    count = 0;
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){
            right_watermarked.at<Vec3b>(j,i) [0] = right_image[count]; count++;
            right_watermarked.at<Vec3b>(j,i) [1] = right_image[count]; count++;
            right_watermarked.at<Vec3b>(j,i) [2] = right_image[count]; count++;
        }



    cv::imshow("right_watermarked", right_watermarked);
    waitKey(0);

    AllocIm::FreeIm(imc2) ;
    AllocIm::FreeIm(imc3) ;
    AllocIm::FreeIm(imr);
    AllocIm::FreeIm(img);
    AllocIm::FreeIm(imb);

    AllocIm::FreeIm(imyout);

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

    ////watermark extraction////

//    ImgWat watermarking;
//    watermarking.setPassword(passwstr,passwnum);
//    watermarking.setParameters(watermark,wsize,0,power,false,false,NULL,0);
//
//
//    cv::Mat new_image_to_dec = cv::Mat::zeros(512, 512, CV_8UC3);
//    for (int j = 0; j < 480; j++)
//        for (int i = 0; i < 512; i++){
//            new_image_to_dec.at<Vec3b>(j,i) [0] = image_to_mark.at<Vec3b>(j,i) [0];
//            new_image_to_dec.at<Vec3b>(j,i) [1] = image_to_mark.at<Vec3b>(j,i) [1];
//            new_image_to_dec.at<Vec3b>(j,i) [2] = image_to_mark.at<Vec3b>(j,i) [2];
//        }
//
////    cv::imshow("to dec", new_image_to_dec);
////    waitKey(0);
//
//    unsigned char *squared_image_to_dec = new_image_to_dec.data; //SPERO SIA GIUSTO PER LE COLOUR
//
////    bool wat = watermarking.extractWatermark(squared_image_to_dec,512,512);
////    cout<<wat;
//
//    bool wat = image_watermarking.extractWatermark(squared_image_to_dec, 512, 512);
//    cout<<wat;


    double min, max;
    cv::Mat mark2;
    Mat channels2[3];
    cv::absdiff(right_watermarked,right,mark2);
    split(mark2,channels2);
    minMaxLoc(channels2[0], &min, &max);
    Mat chan0 = channels2[0] *255 / max;
    Mat chan1 = channels2[1] *255 / max;
    Mat chan2 = channels2[2] *255 / max;
    cv::imshow("mark20", chan0);
    cv::imshow("mark21", chan1);
    cv::imshow("mark22", chan2);
    waitKey(0);

    return 0;


}









//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf