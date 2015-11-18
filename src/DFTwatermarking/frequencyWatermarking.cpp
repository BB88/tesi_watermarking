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
#include "../right_view_computation/right_view.h"
#include <fstream>

//includes watermarking
#include "../img_watermarking/watermarking.h"
#include "../img_watermarking/allocim.h"
#include "../utils.h"
#include "../img_watermarking/fft2d.h"
#include "../graphcuts/utils.h"
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace graph_cuts_utils;



vector<cv::Mat> DFTStereoWatermarking::stereoWatermarking(cv::Mat frameL, cv::Mat frameR, int wsize, float power, std::string passwstr,
                                                 std::string passwnum, int* watermark, int img_num){


    Right_view rv;
    int dim = 512;

    vector<cv::Mat> output;

    unsigned char *left_uchar = frameL.data;
    int squared_dim = dim * dim *3;
    unsigned char *squared_left =  new unsigned char[squared_dim];
    int nc = 640;
    int nc_s = dim;
    int offset = 127;
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }

    Watermarking image_watermarking;

    image_watermarking.setParameters(watermark,wsize,power);
    image_watermarking.setPassword(passwstr,passwnum);

    float  **imidft_wat;
    imidft_wat = AllocIm::AllocImFloat(dim, dim);
    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,dim,dim,dim,imidft_wat,false);

    bool left_to_right = true;

    // prendo dmin e dmax e calcolo disp con kz
    std::string disp_data;
    std::vector<std::string> disprange;
    char sep = ' ';
    std::ifstream in("/home/miky/Scrivania/Tesi/dispRange.txt");
    if (in.is_open()) {
        int j=0;
        while (!in.eof()){
            if ( j == img_num ){
                getline(in, disp_data);
                for(size_t p=0, q=0; p!=disp_data.npos; p=q){
                    disprange.push_back(disp_data.substr(p+(p!=0), (q=disp_data.find(sep, p+1))-p-(p!=0)));
                }
            }
            getline(in, disp_data);
            j+=60;
        }
        in.close();
    }
    int dminl = atoi(disprange[0].c_str());
    int dmaxl = atoi(disprange[1].c_str());

    std::stringstream img1_name;
    img1_name <<"left_"<< img_num;
    std::stringstream img2_name;
    img2_name <<"right_"<< img_num;




    std::cout<<dminl<<" "<<dmaxl<<endl;

//    cv::Mat disp_left = graph_cuts_utils::kz_main(left_to_right,img1_name.str(),img2_name.str(),frameL,frameR,dminl,dmaxl);
//    cv::imshow ("disp_left". disp_left);
//    waitKey(0);


    //queste tre righe servono per prendere la disparità di ground truth
    std::ostringstream pathL;
    pathL << "/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/left/tsukuba_disparity_L_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
//    pathL << "/home/miky/ClionProjects/tesi_watermarking/img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    cv::Mat disp_left = imread(pathL.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);


    left_to_right = false;
//    cv::Mat disp_right = graph_cuts_utils::kz_main(left_to_right,"right","left",frameR,frameL);
    std::ostringstream pathR;
    pathR << "/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
//    pathR << "/home/miky/ClionProjects/tesi_watermarking/img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    int dminr = -dmaxl;
    int dmaxr = -dminl;


//    cv::Mat disp_right = graph_cuts_utils::kz_main(left_to_right,img1_name.str(),img2_name.str(),frameL,frameR,dminr,dmaxr);
//    cv::imshow ("disp_right". disp_right);
//    waitKey(0);

    cv::Mat squared_lDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
        }

    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);

    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
        }


    unsigned char *right_uchar = frameR.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];
    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }

//  ricostruisco sinistra a partire da destra per creare il marchio giusto
    unsigned char * recleft = rv.left_rnc_no_occ(squared_right,squared_rDisp ,dim,dim);

    float  **imidft_wat_rec;
    imidft_wat_rec = AllocIm::AllocImFloat(dim, dim);

// riempio la ricostruzione cosi la fase rimane invariata

    cv::Mat square_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    int count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            square_left_mat.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }
    cv::Mat rec_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);
    count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            rec_left_mat.at<Vec3b>(j, i)[0] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[1] = recleft[count];
            count++;
            rec_left_mat.at<Vec3b>(j, i)[2] = recleft[count];
            count++;
        }
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++){
            if ( rec_left_mat.at<Vec3b>(j, i)[0]==0 && rec_left_mat.at<Vec3b>(j, i)[1]==0 && rec_left_mat.at<Vec3b>(j,
                                                                                                                                i)[2]==0){
                rec_left_mat.at<Vec3b>(j, i)[0] = square_left_mat.at<Vec3b>(j, i)[0];
                rec_left_mat.at<Vec3b>(j, i)[1] = square_left_mat.at<Vec3b>(j, i)[1];
                rec_left_mat.at<Vec3b>(j, i)[2] = square_left_mat.at<Vec3b>(j, i)[2];
            }
        }

// ottengo il marchio generato con lasinistra ricostruita (imidft_wat)
    unsigned char *squared_marked_left_rec = image_watermarking.insertWatermark(rec_left_mat.data,dim,dim,dim,imidft_wat_rec,false);


//    computing warped watermark   ********************
    float  **warp_mark;
    warp_mark = AllocIm::AllocImFloat(dim, dim);
    for (int i=0; i <dim; i++)
        for (int j=0;j<dim;j++)
            warp_mark[i][j] = 0.0;
    unsigned char d = 0;

    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            d = squared_lDisp.at<uchar>(i,j);
            int diff = j-static_cast<int>(d);
            if(static_cast<int>(d)!=0 && diff>=0)
                warp_mark[i][j-static_cast<int>(d)] = imidft_wat_rec[i][j];
        }

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

    for (int i = 0; i < nc_s; i++ )
        for (int j = 0; j < nc_s; j++) {
            marked_right_lum[i][j] = right_lum[i][j] + warp_mark[i][j];
        }

//    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");

//    compute image from luminance   ********************
    unsigned char *marked_right = new unsigned char[squared_dim];
    stereo_watermarking::compute_luminance(marked_right,dim,-1,imr,img,imb,marked_right_lum,imc2,imc3);




// back to normal size *******************

    unsigned char *left_watermarked = new unsigned char [480*640*3];
    left_watermarked = frameL.data;

    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < dim; j++) {
            for (int k =0; k<3;k++){
                left_watermarked[(i *nc + j + offset)*3 + k] = squared_marked_left[(i * nc_s + j)*3 + k];
            }
        }

    count = 0;
    cv::Mat left_wat_mat = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){

            left_wat_mat.at<Vec3b>(j, i) [0] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [1] = left_watermarked[count]; count++;
            left_wat_mat.at<Vec3b>(j, i) [2] = left_watermarked[count]; count++;

        }
    output.push_back(left_wat_mat);
//    stereo_watermarking::show_ucharImage(left_watermarked,640,480,"left_watermarked",3);
//    stereo_watermarking::save_ucharImage(left_watermarked,640,480,"left_watermarked_new");

    unsigned char *right_watermarked = new unsigned char [480*640*3];
    right_watermarked = frameR.data;

    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < dim; j++) {
            for (int k =0; k<3;k++){
                right_watermarked[(i *nc + j + offset)*3 + k] = marked_right[(i * nc_s + j)*3 + k];
            }
        }

    count = 0;
    cv::Mat right_wat_mat = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int j = 0; j < 480; j++)
        for (int i = 0; i < 640; i++){

            right_wat_mat.at<Vec3b>(j, i) [0] = right_watermarked[count]; count++;
            right_wat_mat.at<Vec3b>(j, i) [1] = right_watermarked[count]; count++;
            right_wat_mat.at<Vec3b>(j, i) [2] = right_watermarked[count]; count++;

        }
    output.push_back(right_wat_mat);
//    stereo_watermarking::show_ucharImage(right_watermarked,640,480,"right_watermarked",3);
//    stereo_watermarking::save_ucharImage(right_watermarked,640,480,"right_watermarked_new");

    return output;

}


int DFTStereoWatermarking::stereoDetection(cv::Mat markedL, cv::Mat markedR, int wsize, float power, std::string passwstr,
                                            std::string passwnum, int* watermark,int img_num){


    Watermarking image_watermarking;

    image_watermarking.setParameters(watermark,wsize,power);
    image_watermarking.setPassword(passwstr,passwnum);
    int dim = 512;
    int offset = 127;
    int nc = 640;
    int nc_s = dim;
    int squared_dim = dim * dim *3;


    unsigned char *left_uchar = markedL.data;
    unsigned char *squared_left =  new unsigned char[squared_dim];

    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
            }
        }

    unsigned char *right_uchar = markedR.data;
    unsigned char *squared_right =  new unsigned char[squared_dim];

    for (int i = 0; i < 480; i++ )
        for (int j = 0; j < nc_s; j++) {
            for (int k =0; k<3;k++){
                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
            }
        }
    //    left view reconstruction   ********************
    bool left_to_right = false;
   //    cv::Mat disp_right = graph_cuts_utils::kz_main(left_to_right,"right","left",frameR,frameL);
    std::ostringstream pathR;
    //path per kz
    pathR << "/home/miky/ClionProjects/tesi_watermarking/img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << img_num/60 << ".png";
    //path per gt
//    pathR << "/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/disparity_maps/right/tsukuba_disparity_R_" << std::setw(5) << std::setfill('0') << img_num +1 << ".png";
    //path per quelle sitetizzate
//    pathR << "/home/miky/ClionProjects/tesi_watermarking/img/kz_disp_synt_75/norm_disp_75_synt_to_left_" << img_num +1 << ".png";
    cv::Mat disp_right = imread(pathR.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//    imshow("disp_right",disp_right);
//    waitKey(0);
    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);

    for (int i=0; i <480; i++)
        for (int j=0;j<dim;j++){
            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
        }


    Right_view rv;
    unsigned char * rcn_squared_left = rv.left_rnc_no_occ(squared_right,squared_rDisp,dim,dim);


    cv::Mat square_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);

    int count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            square_left_mat.at<Vec3b>(j, i)[0] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[1] = squared_left[count];
            count++;
            square_left_mat.at<Vec3b>(j, i)[2] = squared_left[count];
            count++;
        }

    cv::Mat rcn_left_mat = cv::Mat::zeros(dim, dim, CV_8UC3);

    count=0;
    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++) {

            rcn_left_mat.at<Vec3b>(j, i)[0] = rcn_squared_left[count];
            count++;
            rcn_left_mat.at<Vec3b>(j, i)[1] = rcn_squared_left[count];
            count++;
            rcn_left_mat.at<Vec3b>(j, i)[2] = rcn_squared_left[count];
            count++;
    }

    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++){
            if ( rcn_left_mat.at<Vec3b>(j, i)[0]==0 && rcn_left_mat.at<Vec3b>(j, i)[1]==0 && rcn_left_mat.at<Vec3b>(j,
                                                                                                                                i)[2]==0){
                rcn_left_mat.at<Vec3b>(j, i)[0] = square_left_mat.at<Vec3b>(j, i)[0];
                rcn_left_mat.at<Vec3b>(j, i)[1] = square_left_mat.at<Vec3b>(j, i)[1];
                rcn_left_mat.at<Vec3b>(j, i)[2] = square_left_mat.at<Vec3b>(j, i)[2];
            }
    }


////     detection   ********************
    std::string fileLikelihoodL  = "/home/miky/Scrivania/Tesi/LikelihoodL_03.txt";
    bool left_det = image_watermarking.extractWatermark(squared_left,dim,dim, dim, fileLikelihoodL);
//    bool right_det = image_watermarking.extractWatermark(squared_right,dim,dim,dim);
    std::string fileLikelihoodLr("/home/miky/Scrivania/Tesi/LikelihoodLr_03.txt");
    bool rcnleft_det = image_watermarking.extractWatermark(rcn_left_mat.data,dim,dim,dim, fileLikelihoodLr);


//    cout<<" left_det    "<<left_det <<endl;
//    cout<<" right_det   "<< right_det<<endl;
//    cout<<"rcnleft_det  "<< rcnleft_det<<endl<<endl;

    // 1: tutti e due, 2: sinistra, 3: sinistra ricostruita
    if(left_det)
        if (rcnleft_det)
            return 1;
        else return 2;
    if(rcnleft_det)
        return 3;
    else return 0;

}




//
//void DFTStereoWatermarking::warpMarkWatermarking(int wsize, float power, std::string passwstr,
//                                                 std::string passwnum, bool gt){
//
//    Right_view rv;
//    cv::Mat left = imread("/home/miky/ClionProjects/tesi_watermarking/img/left.png", CV_LOAD_IMAGE_COLOR);
//    int dim = 512;
//
//    unsigned char *left_uchar = left.data;
//    int squared_dim = dim * dim *3;
//    unsigned char *squared_left =  new unsigned char[squared_dim];
//    int nc = 640;
//    int nc_s = dim;
//    int offset = 127;
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_left[(i * nc_s + j)*3 + k] = left_uchar[(i *nc + j + offset)*3 + k];
//            }
//        }
//    Watermarking image_watermarking;
////    random binary watermark   ********************
//    int watermark[64];
//    for (int i = 0; i < 64; i++){
//        int b = rand() % 2;
//        watermark[i]=b;
//    }
//    image_watermarking.setParameters(watermark,wsize,power);
//    image_watermarking.setPassword(passwstr,passwnum);
//    float  **imidft_wat;
//    imidft_wat = AllocIm::AllocImFloat(dim, dim);
//    unsigned char *squared_marked_left = image_watermarking.insertWatermark(squared_left,dim,dim,dim,imidft_wat,false);
//    stereo_watermarking::show_ucharImage(squared_marked_left,dim,dim,"warp left",3);
//
////     generate watermark's magnitude and phase   ********************
////    stereo_watermarking::show_floatImage(imidft_wat,dim,dim,"mark");
////    stereo_watermarking::writefloatMatToFile(imidft_wat,dim,"/home/miky/Scrivania/wat_lum.txt");
//
////    double  **imdft_mark;
////    double  **imdftfase_mark;
////    imdft_mark = AllocIm::AllocImDouble(dim, dim);
////    imdftfase_mark = AllocIm::AllocImDouble(dim, dim);
////    FFT2D::dft2d(imidft_wat, imdft_mark, imdftfase_mark, dim, dim);
//
////    stereo_watermarking::writeMatToFile(imdft_mark,dim,"/home/miky/Scrivania/wat_mag.txt");
////    stereo_watermarking::writeMatToFile(imdftfase_mark,dim,"/home/miky/Scrivania/wat_phase.txt");
//
////     generate squared disp and occ map   ********************
//    cv::Mat disp_left;
//    if (gt)
//        disp_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//    else  disp_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_left_to_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//
//    cv::Mat occ_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat squared_lDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
//    cv::Mat squared_lOcc = cv::Mat::zeros(dim, dim, CV_8UC1);
//    for (int i=0;i<480;i++)
//        for (int j=0;j<dim;j++){
//            squared_lDisp.at<uchar>(i,j) = disp_left.at<uchar>(i,j+offset);
//            squared_lOcc.at<uchar>(i,j) = occ_left.at<uchar>(i,j+offset);
//        }
//
//    cv::Mat disp_right;
//    if (gt)
//        disp_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//    else  disp_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_right_to_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//
//    cv::Mat occ_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat squared_rDisp = cv::Mat::zeros(dim, dim, CV_8UC1);
//    cv::Mat squared_rOcc = cv::Mat::zeros(dim, dim, CV_8UC1);
//
//    for (int i=0;i<480;i++)
//        for (int j=0;j<dim;j++){
//            squared_rDisp.at<uchar>(i,j) = disp_right.at<uchar>(i,j+offset);
//            squared_rOcc.at<uchar>(i,j) = occ_right.at<uchar>(i,j+offset);
//        }
//
//    cv::Mat right = imread("/home/miky/ClionProjects/tesi_watermarking/img/right.png",CV_LOAD_IMAGE_COLOR);
//    unsigned char *right_uchar = right.data;
//    unsigned char *squared_right =  new unsigned char[squared_dim];
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset)*3 + k];
//            }
//        }
//
////  ricostruisco sinistra a partire da destra per creare il marchio giusto
//    unsigned char * recleft = rv.left_rnc(squared_right,squared_rDisp, squared_rOcc ,dim,dim,gt);
//
//    float  **imidft_wat_rec;
//    imidft_wat_rec = AllocIm::AllocImFloat(dim, dim);
//
//// riempio la ricostruzione cosi la fase rimane invariata
//
//    cv::Mat mat_image = cv::Mat::zeros(dim, dim, CV_8UC3);
//    int count=0;
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++) {
//
//            mat_image.at<Vec3b>(j, i)[0] = squared_left[count];
//            count++;
//            mat_image.at<Vec3b>(j, i)[1] = squared_left[count];
//            count++;
//            mat_image.at<Vec3b>(j, i)[2] = squared_left[count];
//            count++;
//        }
//    cv::Mat mat_image4 = cv::Mat::zeros(dim, dim, CV_8UC3);
//    count=0;
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++) {
//
//            mat_image4.at<Vec3b>(j, i)[0] = recleft[count];
//            count++;
//            mat_image4.at<Vec3b>(j, i)[1] = recleft[count];
//            count++;
//            mat_image4.at<Vec3b>(j, i)[2] = recleft[count];
//            count++;
//        }
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++){
//            if ( mat_image4.at<Vec3b>(j,i)[0]==0 && mat_image4.at<Vec3b>(j,i)[1]==0 && mat_image4.at<Vec3b>(j,i)[2]==0){
//                mat_image4.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
//                mat_image4.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
//                mat_image4.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
//            }
//        }
//
//// ottengo il marchio generato con lasinistra ricostruita (imidft_wat)
//    unsigned char *squared_marked_left_rec = image_watermarking.insertWatermark(mat_image4.data,dim,dim,dim,imidft_wat_rec,false);
//
////    computing warped watermark   ********************
//    float  **warp_mark;
//    warp_mark = AllocIm::AllocImFloat(dim, dim);
//    for (int i=0;i<dim;i++)
//        for (int j=0;j<dim;j++)
//            warp_mark[i][j] = 0.0;
//    unsigned char d = 0;
//    unsigned char occ = 0;
//
//    for (int i=0;i<480;i++)
//        for (int j=0;j<dim;j++){
//            d = squared_lDisp.at<uchar>(i,j);
//
//            if (gt)
//                occ = squared_lOcc.at<uchar>(i,j);
//            else  occ = squared_lDisp.at<uchar>(i,j);
//
//            int diff = j-static_cast<int>(d);
//            if(static_cast<int>(occ)!=0 && diff>=0)
//                warp_mark[i][j-static_cast<int>(d)] = imidft_wat_rec[i][j];
//        }
////    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
////    double  **imdft_warp_mark;
////    double  **imdftfase_warp_mark;
////    imdft_warp_mark = AllocIm::AllocImDouble(dim, dim);
////    imdftfase_warp_mark = AllocIm::AllocImDouble(dim, dim);
////    FFT2D::dft2d(warp_mark, imdft_warp_mark, imdftfase_warp_mark, dim, dim);
////    stereo_watermarking::writeMatToFile(imdft_warp_mark,dim,"/home/miky/Scrivania/warp_wat_mag.txt");
////    stereo_watermarking::writeMatToFile(imdftfase_warp_mark,dim,"/home/miky/Scrivania/warp_wat_phase.txt");
////    marking right view with warped mark   ********************
//
//
////    //    insert wat in left view
////    unsigned char **imrl;
////    unsigned char **imgl;
////    unsigned char **imbl;
////    float **imc2l;
////    float **imc3l;
////    imc2l = AllocIm::AllocImFloat(dim, dim);
////    imc3l = AllocIm::AllocImFloat(dim, dim);
////    imrl = AllocIm::AllocImByte(dim, dim);
////    imgl = AllocIm::AllocImByte(dim, dim);
////    imbl = AllocIm::AllocImByte(dim, dim);
////    float ** left_lum;
////    left_lum = AllocIm::AllocImFloat(dim, dim);
////    stereo_watermarking::compute_luminance(squared_left,dim,1,imrl,imgl,imbl,left_lum,imc2l,imc3l);
////    float ** marked_left_lum = AllocIm::AllocImFloat(dim, dim);
////    for (int i = 0; i < nc_s; i ++ )
////        for (int j = 0; j < nc_s; j++) {
////            marked_left_lum[i][j] = left_lum[i][j] + imidft_wat[i][j];
////        }
////    stereo_watermarking::show_floatImage(marked_left_lum,dim,dim,"marked_left_lum");
////    unsigned char *marked_left = new unsigned char[squared_dim];
////    stereo_watermarking::compute_luminance(marked_left,dim,-1,imrl,imgl,imbl,marked_left_lum,imc2l,imc3l);
////    dft watermarking   ********************
////    unsigned char *marked_right= image_watermarking.insertWatermark(squared_right,256,256,warp_mark,true);
////    spacial watermarking:compute right luminance   ********************
//    unsigned char **imr;
//    unsigned char **img;
//    unsigned char **imb;
//    float **imc2;
//    float **imc3;
//    imc2 = AllocIm::AllocImFloat(dim, dim);
//    imc3 = AllocIm::AllocImFloat(dim, dim);
//    imr = AllocIm::AllocImByte(dim, dim);
//    img = AllocIm::AllocImByte(dim, dim);
//    imb = AllocIm::AllocImByte(dim, dim);
//    float ** right_lum;
//    right_lum = AllocIm::AllocImFloat(dim, dim);
//    stereo_watermarking::compute_luminance(squared_right,dim,1,imr,img,imb,right_lum,imc2,imc3);
////    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
////   compute marked right lum  ********************
//    float ** marked_right_lum = AllocIm::AllocImFloat(dim, dim);
//
//    for (int i = 0; i < nc_s; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            marked_right_lum[i][j] = right_lum[i][j] + warp_mark[i][j];
//        }
//
////    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");
//
////    compute image from luminance   ********************
//    unsigned char *marked_right = new unsigned char[squared_dim];
//    stereo_watermarking::compute_luminance(marked_right,dim,-1,imr,img,imb,marked_right_lum,imc2,imc3);
//    stereo_watermarking::show_ucharImage(marked_right,dim,dim,"marked_right",3);
////    left view reconstruction   ********************
//
//    unsigned char * rcn_squared_left = rv.left_rnc(marked_right,squared_rDisp, squared_rOcc ,dim,dim,gt);
//
//    cv::Mat mat_image2 = cv::Mat::zeros(dim, dim, CV_8UC3);
//    count=0;
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++) {
//
//            mat_image2.at<Vec3b>(j, i)[0] = rcn_squared_left[count];
//            count++;
//            mat_image2.at<Vec3b>(j, i)[1] = rcn_squared_left[count];
//            count++;
//            mat_image2.at<Vec3b>(j, i)[2] = rcn_squared_left[count];
//            count++;
//        }
///*    imshow("before", mat_image2);*/
//    waitKey(0);
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++){
//            if ( mat_image2.at<Vec3b>(j,i)[0]==0 && mat_image2.at<Vec3b>(j,i)[1]==0 && mat_image2.at<Vec3b>(j,i)[2]==0){
//                mat_image2.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
//                mat_image2.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
//                mat_image2.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
//            }
//        }
//    /* imshow("recontructed", mat_image2);
//     waitKey(0);*/
//
////    imshow   ********************
////
////    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
////    stereo_watermarking::show_ucharImage(squared_right,dim,dim,"squared_right");
////    stereo_watermarking::show_floatImage(right_lum,dim,dim,"right_lum");
////    stereo_watermarking::show_floatImage(warp_mark,dim,dim,"warp_mark");
////    stereo_watermarking::show_floatImage(marked_right_lum,dim,dim,"marked_right_lum");
////    stereo_watermarking::show_ucharImage(marked_right,256,256,"marked_right");
////    stereo_watermarking::show_ucharImage(rcn_squared_left,dim,dim,"rcn_squared_left");
////    stereo_watermarking::show_ucharImage(marked_left,256,256,"marked_left2");
//
//
////     detection   ********************
//    bool left_det = image_watermarking.extractWatermark(squared_marked_left,dim,dim, dim);
//    bool right_det = image_watermarking.extractWatermark(marked_right,dim,dim,dim);
//    bool rcnleft_det = image_watermarking.extractWatermark(mat_image2.data,dim,dim,dim);
////    bool left_marked_det = image_watermarking.extractWatermark(marked_left,dim,dim,dim);
//
//
//    cout<<" left_det    "<<left_det <<endl;
//    cout<<" right_det   "<< right_det<<endl;
//    cout<<"rcnleft_det  "<< rcnleft_det<<endl;
////    cout<<"left_marked_det  "<< left_marked_det<<endl;
//
//// back to normal size *******************
//
//    unsigned char *left_watermarked = new unsigned char [480*640*3];
//    left_watermarked = left.data;
//
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < dim; j++) {
//            for (int k =0; k<3;k++){
//                left_watermarked[(i *nc + j + offset)*3 + k] = squared_marked_left[(i * nc_s + j)*3 + k];
//            }
//        }
//    stereo_watermarking::show_ucharImage(left_watermarked,640,480,"left_watermarked");
//    stereo_watermarking::save_ucharImage(left_watermarked,640,480,"left_watermarked");
//
//    unsigned char *right_watermarked = new unsigned char [480*640*3];
//    right_watermarked = right.data;
//
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < dim; j++) {
//            for (int k =0; k<3;k++){
//                right_watermarked[(i *nc + j + offset)*3 + k] = marked_right[(i * nc_s + j)*3 + k];
//            }
//        }
//    stereo_watermarking::show_ucharImage(right_watermarked,640,480,"right_watermarked");
//    stereo_watermarking::save_ucharImage(right_watermarked,640,480,"right_watermarked");
//    cv::Mat synt_view = imread("/home/miky/ClionProjects/tesi_watermarking/img/synt.png", CV_LOAD_IMAGE_COLOR);
//    unsigned char *synt_view_uchar = synt_view.data;
//    unsigned char *squared_synt_view =  new unsigned char[squared_dim];
//    for (int j = 0; j < dim*dim*3; j++) {
//        squared_synt_view[j]= 0;
//    }
//    for (int i = 0; i < 480; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_synt_view[(i * nc_s + j)*3 + k] = synt_view_uchar[(i *nc + j + offset)*3 + k];
//            }
//        }
//
//
////    bool synt_view_det = image_watermarking.extractWatermark(squared_synt_view,dim,dim,dim);
////    cout<<" syn_det    "<<synt_view_det <<endl;
//
////    cv::Mat disp_synt = imread("/home/miky/ClionProjects/tesi_watermarking/img/disp_kz_syn.png", CV_LOAD_IMAGE_COLOR);
////    cv::Mat nkz_disp;
////    if (disp_synt.rows == 0){
////        cout << "Empty image";
////    } else {
////        Disp_opt dp;
////        dp.disparity_normalization(disp_synt, nkz_disp);
////    }
////    imwrite("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_syn.png",nkz_disp);
//
//    cv::Mat norm_disp_syn = imread("/home/miky/ClionProjects/tesi_watermarking/img/norm_disp_syn.png", CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat norm_kzdisp = cv::Mat::zeros(480, 640, CV_8UC1);
//
//
//    cv::Mat squared_disp_synt = cv::Mat::zeros(dim, dim, CV_8UC1);
//    for (int i=0;i<480;i++)
//        for (int j=0;j<dim;j++){
//            squared_disp_synt.at<uchar>(i,j) = norm_disp_syn.at<uchar>(i,j+offset);
//        }
//
//
//    cv::Mat squared_occ_synt = cv::Mat::zeros(dim, dim, CV_8UC1);
//    for (int i=0;i<dim;i++)
//        for (int j=0;j<dim;j++){
//            squared_occ_synt.at<uchar>(i,j) = 255;
//        }
//
//    unsigned char * rcn_squared_left_synt = rv.left_rnc(squared_synt_view,squared_disp_synt,squared_occ_synt,dim,dim,gt);
//
////    stereo_watermarking::show_ucharImage(rcn_squared_left_synt,dim,dim,"rcn_synt_view");
//
//    cv::Mat mat_image5 = cv::Mat::zeros(dim, dim, CV_8UC3);
//    count=0;
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++) {
//            mat_image5.at<Vec3b>(j, i)[0] = rcn_squared_left_synt[count];
//            count++;
//            mat_image5.at<Vec3b>(j, i)[1] = rcn_squared_left_synt[count];
//            count++;
//            mat_image5.at<Vec3b>(j, i)[2] = rcn_squared_left_synt[count];
//            count++;
//        }
//    for (int j = 0; j < dim; j++)
//        for (int i = 0; i < dim; i++){
//            if ( mat_image5.at<Vec3b>(j,i)[0]==0 && mat_image5.at<Vec3b>(j,i)[1]==0 && mat_image5.at<Vec3b>(j,i)[2]==0){
//                mat_image5.at<Vec3b>(j,i)[0] = mat_image.at<Vec3b>(j,i)[0];
//                mat_image5.at<Vec3b>(j,i)[1] = mat_image.at<Vec3b>(j,i)[1];
//                mat_image5.at<Vec3b>(j,i)[2] = mat_image.at<Vec3b>(j,i)[2];
//            }
//        }
//
////    imshow("recont_synt_filled", mat_image5);
////    waitKey(0);
//
//    bool synt_view_det = image_watermarking.extractWatermark(mat_image5.data,dim,dim,dim);
//    cout<<" syn_det    "<<synt_view_det <<endl;
//}
//

