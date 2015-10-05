//
// Created by miky on 02/10/15.
//

#include "frequencyWatermarking.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "../dataset/tsukuba_dataset.h"
#include <cv.h>
#include <cstdint>
#include <highgui.h>

#include "../disparity_computation/stereo_matching.h"
#include "../disparity_optimization/disp_opt.h"
#include "../disparity_optimization/occlusions_handler.h"
#include "../right_view_computation/right_view.h"
#include "../disparity_optimization/disp_opt.h"
#include <limits>
#include <cstddef>
#include <iostream>
#include <fstream>

//includes watermarking
#include "../img_watermarking/watermarking.h"
#include "../img_watermarking/allocim.h"



#include "../utils.h"
#include "../img_watermarking/fft2d.h"


using namespace std;
using namespace cv;
using namespace cv::datasets;




void FDTStereoWatermarking::warpMarkWatermarking(int wsize, int tilesize, float power, bool clipping,
                                                 bool flagResyncAll, int tilelistsize, std::string passwstr,
                                                 std::string passwnum){

    Right_view rv;
    cv::Mat left = imread("/home/miky/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);

    unsigned char *left_uchar = left.data;
    int squared_dim = 512 * 512 *3;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    int nc = 640;
    int nc_s = 512;
    int offset = 127;
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }
    int dim = 512;
    Watermarking image_watermarking;
//    random binary watermark   ********************
    int watermark[64];
    for (int i = 0; i < 64; i++){
        int b = rand() % 2;
        watermark[i]=b;
    }
    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
    image_watermarking.setPassword(passwstr,passwnum);
    float  **imidft_wat;
    imidft_wat = AllocIm::AllocImFloat(512, 512);
    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,512,512,512,imidft_wat,false);

//     generate watermark's magnitude and phase   ********************
//    stereo_watermarking::show_floatImage(imidft_wat,dim,dim,"mark");
//    stereo_watermarking::writefloatMatToFile(imidft_wat,dim,"/home/miky/Scrivania/wat_lum.txt");

//    double  **imdft_mark;
//    double  **imdftfase_mark;
//    imdft_mark = AllocIm::AllocImDouble(dim, dim);
//    imdftfase_mark = AllocIm::AllocImDouble(dim, dim);
//    FFT2D::dft2d(imidft_wat, imdft_mark, imdftfase_mark, dim, dim);

//    stereo_watermarking::writeMatToFile(imdft_mark,dim,"/home/miky/Scrivania/wat_mag.txt");
//    stereo_watermarking::writeMatToFile(imdftfase_mark,dim,"/home/miky/Scrivania/wat_phase.txt");

//     generate squared disp and occ map   ********************
    cv::Mat disp_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_left.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat squared_lDisp = cv::Mat::zeros(512, 512, CV_8UC1);
    cv::Mat squared_lOcc = cv::Mat::zeros(512, 512, CV_8UC1);
    for (int i=0;i<480;i++)
        for (int j=0;j<dim;j++){
            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
            squared_lOcc.at<uchar>(i,j) = occ_left.at<uchar>(i,j+offset);
        }

    cv::Mat disp_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat squared_rDisp = cv::Mat::zeros(512, 512, CV_8UC1);
    cv::Mat squared_rOcc = cv::Mat::zeros(512, 512, CV_8UC1);

    for (int i=0;i<480;i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
            squared_rOcc.at<uchar>(i,j) = occ_right.at<uchar>(i,j+offset);
        }

    cv::Mat right = imread("/home/miky/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
    unsigned char *right_uchar = right.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }

//  ricostruisco sinistra a partire da destra per creare il marchio giusto

    unsigned char * recleft = rv.left_rnc(squared_right,squared_rDisp,squared_rOcc,dim,dim);
    float  **imidft_wat_rec;
    imidft_wat_rec = AllocIm::AllocImFloat(512, 512);

// riempio la ricostruzione cosi la fase rimane invariata

    cv::Mat mat_image = cv::Mat::zeros(512, 512, CV_8UC3);
    int count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            mat_image.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            mat_image.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }
    cv::Mat mat_image4 = cv::Mat::zeros(512, 512, CV_8UC3);
    count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image4.at<Vec3b>(j, i)[0] = recleft[count];
            count++;
            mat_image4.at<Vec3b>(j, i)[1] = recleft[count];
            count++;
            mat_image4.at<Vec3b>(j, i)[2] = recleft[count];
            count++;
        }
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++){
            if ( mat_image4.at<Vec3b>(j,i)[0]==0 && mat_image4.at<Vec3b>(j,i)[1]==0 && mat_image4.at<Vec3b>(j,i)[2]==0){
                mat_image4.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
                mat_image4.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
                mat_image4.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
            }
        }

// ottengo il marchio generato con lasinistra ricostruita
    unsigned char *squared_marked_left_rec = image_watermarking.insertWatermark(mat_image4.data,512,512,512,imidft_wat_rec,false);

//    computing warped watermark   ********************
    float  **warp_mark;
    warp_mark = AllocIm::AllocImFloat(512, 512);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            warp_mark[i][j] = 0.0;
    unsigned char d = 0;
    unsigned char occ = 0;

    for (int i=0;i<480;i++)
        for (int j=0;j<dim;j++){
            d = squared_lDisp.at<uchar>(i,j);
            occ = squared_lOcc.at<uchar>(i,j);
            int diff = j-static_cast<int>(d);
            if(static_cast<int>(occ)!=0 && diff>=0)
                warp_mark[i][j-static_cast<int>(d)] = imidft_wat_rec[i][j];
        }
//    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
//    double  **imdft_warp_mark;
//    double  **imdftfase_warp_mark;
//    imdft_warp_mark = AllocIm::AllocImDouble(dim, dim);
//    imdftfase_warp_mark = AllocIm::AllocImDouble(dim, dim);
//    FFT2D::dft2d(warp_mark, imdft_warp_mark, imdftfase_warp_mark, dim, dim);
//    stereo_watermarking::writeMatToFile(imdft_warp_mark,dim,"/home/miky/Scrivania/warp_wat_mag.txt");
//    stereo_watermarking::writeMatToFile(imdftfase_warp_mark,dim,"/home/miky/Scrivania/warp_wat_phase.txt");
//    marking right view with warped mark   ********************


//    //    insert wat in left view
//    unsigned char **imrl;
//    unsigned char **imgl;
//    unsigned char **imbl;
//    float **imc2l;
//    float **imc3l;
//    imc2l = AllocIm::AllocImFloat(dim, dim);
//    imc3l = AllocIm::AllocImFloat(dim, dim);
//    imrl = AllocIm::AllocImByte(dim, dim);
//    imgl = AllocIm::AllocImByte(dim, dim);
//    imbl = AllocIm::AllocImByte(dim, dim);
//    float ** left_lum;
//    left_lum = AllocIm::AllocImFloat(dim, dim);
//    stereo_watermarking::compute_luminance(squared_left,dim,1,imrl,imgl,imbl,left_lum,imc2l,imc3l);
//    float ** marked_left_lum = AllocIm::AllocImFloat(dim, dim);
//    for (int i = 0; i < nc_s; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            marked_left_lum[i][j] = left_lum[i][j] + imidft_wat[i][j];
//        }
//    stereo_watermarking::show_floatImage(marked_left_lum,dim,dim,"marked_left_lum");
//    unsigned char *marked_left = new unsigned char[squared_dim];
//    stereo_watermarking::compute_luminance(marked_left,dim,-1,imrl,imgl,imbl,marked_left_lum,imc2l,imc3l);
//    dft watermarking   ********************
//    unsigned char *marked_right= image_watermarking.insertWatermark(squared_right,256,256,warp_mark,true);
//    spacial watermarking:compute right luminance   ********************
    unsigned char **imr;
    unsigned char **img;
    unsigned char **imb;
    float **imc2;
    float **imc3;
    imc2 = AllocIm::AllocImFloat(dim, dim);
    imc3 = AllocIm::AllocImFloat(dim, dim);
    imr = AllocIm::AllocImByte(dim, dim);
    img = AllocIm::AllocImByte(dim, dim);
    imb = AllocIm::AllocImByte(dim, dim);
    float ** right_lum;
    right_lum = AllocIm::AllocImFloat(dim, dim);
    stereo_watermarking::compute_luminance(squared_right,dim,1,imr,img,imb,right_lum,imc2,imc3);
//    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
//   compute marked right lum  ********************
    float ** marked_right_lum = AllocIm::AllocImFloat(dim, dim);

    for (int i = 0; i < nc_s; i ++ )
        for (int j = 0; j < nc_s; j++) {
            marked_right_lum[i][j] = right_lum[i][j] + warp_mark[i][j];
        }

//    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");

//    compute image from luminance   ********************
    unsigned char *marked_right = new unsigned char[squared_dim];
    stereo_watermarking::compute_luminance(marked_right,dim,-1,imr,img,imb,marked_right_lum,imc2,imc3);
//    stereo_watermarking::show_ucharImage(marked_right,dim,dim,"marked_right");
//    left view reconstruction   ********************


    unsigned char * rcn_squared_left = rv.left_rnc(marked_right,squared_rDisp,squared_rOcc,dim,dim);

    cv::Mat mat_image2 = cv::Mat::zeros(512, 512, CV_8UC3);
    count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image2.at<Vec3b>(j, i)[0] = rcn_squared_left[count];
            count++;
            mat_image2.at<Vec3b>(j, i)[1] = rcn_squared_left[count];
            count++;
            mat_image2.at<Vec3b>(j, i)[2] = rcn_squared_left[count];
            count++;
        }
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++){
            if ( mat_image2.at<Vec3b>(j,i)[0]==0 && mat_image2.at<Vec3b>(j,i)[1]==0 && mat_image2.at<Vec3b>(j,i)[2]==0){
                mat_image2.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
                mat_image2.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
                mat_image2.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
            }
        }
//    imshow("recontructed", mat_image2);
//    waitKey(0);

//    imshow   ********************
//
//    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
//    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
//    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
//    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
//    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");
//    stereo_watermarking::show_ucharImage(marked_right,256,256,"marked_right");
//    stereo_watermarking::show_ucharImage(rcn_squared_left,512,512,"rcn_squared_left");
//    stereo_watermarking::show_ucharImage(marked_left,256,256,"marked_left2");


//     detection   ********************
    bool left_det = image_watermarking.extractWatermark(squared_marked_left,dim,dim, dim);
    bool right_det = image_watermarking.extractWatermark(marked_right,dim,dim,dim);
    bool rcnleft_det = image_watermarking.extractWatermark(mat_image2.data,dim,dim,dim);
//    bool left_marked_det = image_watermarking.extractWatermark(marked_left,dim,dim,dim);


    cout<<" left_det    "<<left_det <<endl;
    cout<<" right_det   "<< right_det<<endl;
    cout<<"rcnleft_det  "<< rcnleft_det<<endl;
//    cout<<"left_marked_det  "<< left_marked_det<<endl;

// back to normal size *******************

    unsigned char *left_watermarked = new unsigned char [480*640*3];
    left_watermarked = left.data;

    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < 512; j++) {
            for (int k =0; k<3;k++){
                left_watermarked[(i *nc + j + offset)*3 + k] = squared_marked_left[(i * nc_s + j)*3 + k];
            }
        }
    stereo_watermarking::show_ucharImage(left_watermarked,640,480,"left_watermarked");
    stereo_watermarking::save_ucharImage(left_watermarked,640,480,"left_watermarked");

    unsigned char *right_watermarked = new unsigned char [480*640*3];
    right_watermarked = right.data;

    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < 512; j++) {
            for (int k =0; k<3;k++){
                right_watermarked[(i *nc + j + offset)*3 + k] = marked_right[(i * nc_s + j)*3 + k];
            }
        }
    stereo_watermarking::show_ucharImage(right_watermarked,640,480,"right_watermarked");
    stereo_watermarking::save_ucharImage(right_watermarked,640,480,"right_watermarked");

    cv::Mat synt_view = imread("/home/miky/ClionProjects/tesi_watermarking/img/synt.png", CV_LOAD_IMAGE_COLOR);

    unsigned char *synt_view_uchar = synt_view.data;
    unsigned char *squared_synt_view =  new unsigned char[squared_dim];
    for (int j = 0; j < 512*512*3; j++) {
        squared_synt_view[j]= 0;
    }
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_synt_view[(i * nc_s + j)*3 + k] = synt_view_uchar[(i *nc + j + offset)*3 + k];
            }
        }
    stereo_watermarking::show_ucharImage(squared_synt_view,512,512,"synt_view");

//    bool synt_view_det = image_watermarking.extractWatermark(squared_synt_view,dim,dim,dim);
//    cout<<" syn_det    "<<synt_view_det <<endl;

//    cv::Mat disp_synt = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_kz_syn.png", CV_LOAD_IMAGE_COLOR);
//    cv::Mat nkz_disp;
//    if (disp_synt.rows == 0){
//        cout << "Empty image";
//    } else {
//        Disp_opt dp;
//        dp.disparity_normalization(disp_synt, nkz_disp);
//    }
//    imwrite("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_syn.png",nkz_disp);

    cv::Mat kz_disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_syn.png", CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat squared_disp_synt = cv::Mat::zeros(512, 512, CV_8UC1);
    for (int i=0;i<480;i++)
        for (int j=0;j<dim;j++){
            squared_disp_synt.at<uchar>(i,j) = kz_disp.at<uchar>(i,j+offset);
        }

    cv::Mat squared_occ_synt = cv::Mat::zeros(512, 512, CV_8UC1);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++){
            squared_occ_synt.at<uchar>(i,j) = 255;
        }

    unsigned char * rcn_squared_left_synt = rv.left_rnc(squared_synt_view,squared_disp_synt,squared_occ_synt,dim,dim);
    stereo_watermarking::show_ucharImage(rcn_squared_left_synt,512,512,"rcn_synt_view");

    cv::Mat mat_image5 = cv::Mat::zeros(512, 512, CV_8UC3);
    count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image5.at<Vec3b>(j, i)[0] = rcn_squared_left_synt[count];
            count++;
            mat_image5.at<Vec3b>(j, i)[1] = rcn_squared_left_synt[count];
            count++;
            mat_image5.at<Vec3b>(j, i)[2] = rcn_squared_left_synt[count];
            count++;
        }
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++){
            if ( mat_image5.at<Vec3b>(j,i)[0]==0 && mat_image5.at<Vec3b>(j,i)[1]==0 && mat_image5.at<Vec3b>(j,i)[2]==0){
                mat_image5.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
                mat_image5.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
                mat_image5.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
            }
        }

    imshow("recont_synt_filled", mat_image5);
    waitKey(0);

    bool synt_view_det = image_watermarking.extractWatermark(mat_image5.data,dim,dim,dim);
    cout<<" syn_det    "<<synt_view_det <<endl;
 }


void FDTStereoWatermarking::warpRightWatermarking(int wsize, int tilesize, float power, bool clipping,
                                                      bool flagResyncAll, int tilelistsize, std::string passwstr,
                                                      std::string passwnum) {

    /*watermarking dft 512*256 + MSE    -> controllare watermarking.cpp : 256 e diag */

    cv::Mat left = imread("/home/miky/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);

    int dim = 512;

    unsigned char *left_uchar = left.data;
    int squared_dim = 512 * 512 *3;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    int nc = 640;
    int nc_s = 512;

    for (int k =0; k<squared_dim;k++){
        squared_left[k]=0;
    }

    int offset = 127;
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }

    Watermarking image_watermarking;
//    random binary watermark   ********************
    int watermark[64];
    for (int i = 0; i < 64; i++){
        int b = rand() % 2;
        watermark[i]=b;
    }
    image_watermarking.setParameters(watermark,wsize,tilesize,power,clipping,flagResyncAll,NULL,tilelistsize);
    image_watermarking.setPassword(passwstr,passwnum);
    float  **imidft_wat;
    imidft_wat = AllocIm::AllocImFloat(512, 512);
    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,512,512,512,imidft_wat,false);
    stereo_watermarking::show_ucharImage(squared_marked_left, 512, 512, "squared_marked_left",3);


    cv::Mat right = imread("/home/miky/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat occ_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_left.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/gt_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat right_disp = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat occ_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);

    Right_view rv;

    unsigned char *left_reconstructed =  rv.left_rnc(right.data,right_disp,occ_right,640,480);


    unsigned char *right_uchar = left_reconstructed;
    unsigned char *right_uchar_original = right.data;

    unsigned char *squared_right_to_mark =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_right_to_mark[k]=0;
    }
    unsigned char *squared_right_original =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_right_original[k]=0;
    }


    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
            for(int k = 0; k<3; k++)
                squared_right_to_mark[(i * nc_s + j)*3 + k] = right_uchar[(i * nc + j + offset)*3 + k];


    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
            for(int k = 0; k<3; k++)
                squared_right_original[(i * nc_s + j)*3 + k] = right_uchar_original[(i * nc + j + offset)*3 + k];


    float  **imidft_wat2;
    imidft_wat = AllocIm::AllocImFloat(512, 512);
    unsigned char *squared_marked_right = image_watermarking.insertWatermark(squared_right_to_mark,512,512,512,imidft_wat2,false);
    stereo_watermarking::show_ucharImage(squared_marked_right, 512, 512, "squared_marked_right",3);


    bool squared_marked_right_detection = image_watermarking.extractWatermark(squared_marked_right, 512, 512,512 );
    cout<< "squared_marked_right_detection:    " << squared_marked_right_detection <<endl;


    unsigned char *disp_uchar_left = disp.data;
    unsigned char *squared_disp_left =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_disp_left[k]=0;
    }
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
            squared_disp_left[(i * nc_s + j) ] = disp_uchar_left[(i * nc + j + offset) ];

    unsigned char *disp_uchar_right = right_disp.data;
    unsigned char *squared_disp_right =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_disp_right[k]=0;
    }
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
            squared_disp_right[(i * nc_s + j) ] = disp_uchar_right[(i * nc + j + offset) ];


    unsigned char *occ_uchar_left = occ_left.data;
    unsigned char *squared_occ_left =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_occ_left[k]=0;
    }
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
                squared_occ_left[(i * nc_s + j)] = occ_uchar_left[(i * nc + j + offset)];

    unsigned char *occ_uchar_right = occ_right.data;
    unsigned char *squared_occ_right =  new unsigned char[squared_dim];
    for (int k =0; k<squared_dim;k++){
        squared_occ_right[k]=0;
    }
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
    for (int i = 0; i < 480; i ++ )
        for (int j = 0; j < nc_s; j++)
            squared_occ_right[(i * nc_s + j)] = occ_uchar_right[(i * nc + j + offset)];

//
//    stereo_watermarking::show_ucharImage(squared_disp_left, 512, 512, "squared_disp_left",1);
//    stereo_watermarking::show_ucharImage(squared_occ_left, 512, 512, "squared_occ_left",1);

    unsigned char* squared_right_rec = rv.right_uchar_reconstruction(squared_marked_right,squared_disp_left,squared_occ_left,512,512);

//    stereo_watermarking::show_ucharImage(squared_right_rec, 512, 512, "squared_right_rec",3);



    cv::Mat mat_image = cv::Mat::zeros(512, 512, CV_8UC3);
    int count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image.at<Vec3b>(j, i)[0] = squared_right_rec[count];
            count++;
            mat_image.at<Vec3b>(j, i)[1] = squared_right_rec[count];
            count++;
            mat_image.at<Vec3b>(j, i)[2] = squared_right_rec[count];
            count++;

        }
    cv::Mat mat_image2 = cv::Mat::zeros(512, 512, CV_8UC3);
    count=0;
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++) {

            mat_image2.at<Vec3b>(j, i)[0] = squared_right_original[count];
            count++;
            mat_image2.at<Vec3b>(j, i)[1] = squared_right_original[count];
            count++;
            mat_image2.at<Vec3b>(j, i)[2] = squared_right_original[count];
            count++;
        }
    for (int j = 0; j < 512; j++)
        for (int i = 0; i < 512; i++){
            if ( mat_image.at<Vec3b>(j,i)[0]==0 && mat_image.at<Vec3b>(j,i)[1]==0 && mat_image.at<Vec3b>(j,i)[2]==0){
                mat_image.at<Vec3b>(j,i)[0] = mat_image2.at<Vec3b>(j,i)[0];
                mat_image.at<Vec3b>(j,i)[1] = mat_image2.at<Vec3b>(j,i)[1];
                mat_image.at<Vec3b>(j,i)[2] = mat_image2.at<Vec3b>(j,i)[2];
            }
        }
    namedWindow("Mat squared right", WINDOW_NORMAL);
    imshow("Mat squared right", mat_image);
    waitKey(0);

    stereo_watermarking::show_ucharImage(squared_right_rec, 512, 512, "squared_right_rec",3);
    bool right_detection = image_watermarking.extractWatermark(mat_image.data, 512, 512,512);
    cout<< "right_detection:    " << right_detection <<endl;


//    stereo_watermarking::show_ucharImage(squared_right_rec, 256, 256, "squared_right_rec",3);
    stereo_watermarking::show_ucharImage(squared_disp_right, 512, 512, "squared_disp_right",1);
    stereo_watermarking::show_ucharImage(squared_occ_right, 512, 512, "squared_occ_right",1);


    unsigned char* left_rec_from_right_marked = rv.left_uchar_reconstruction(mat_image.data,squared_disp_right,squared_occ_right,512,512);

    stereo_watermarking::show_ucharImage(left_rec_from_right_marked, 512, 512, "left_rec_from_right_marked",3);

    bool left_rec_detection = image_watermarking.extractWatermark(left_rec_from_right_marked, 512, 512,512);
    cout<< "left_rec_detection:    " << left_rec_detection <<endl;


}