#include <iostream>
#include <opencv2/core/core.hpp>
#include "dataset/tsukuba_dataset.h"
#include <cv.h>
#include <highgui.h>

#include "./disparity_computation/stereo_matching.h"

#include <fstream>

//includes watermarking
#include "./img_watermarking/watermarking.h"
#include "./img_watermarking/imgwat.h"
#include "./img_watermarking/allocim.h"

//grapfh cuts
#include "./graphcuts/utils.h"

//quality metrics
#include "./quality_metrics/quality_metrics.h"
#include "./quality_metrics/RRQualityMetrics.h"

//libconfig
#include <libconfig.h++>
#include "./config/config.hpp"

#include "./spatialWatermarking/gaussianNoise.h"

#include "FDTwatermarking/frequencyWatermarking.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;
using namespace graph_cuts_utils;
using namespace qm;
using namespace RRQualityMetrics;
using namespace spatialWatermarking;




int main() {


    /*CONFIG SETTINGS*/


    bool coding = false;
    bool decoding = true;
    if (coding) {
        Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();
        int wsize = pars.wsize;
        float power = pars.power;
        Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();
        bool masking = generalPars.masking;
        std::string passwstr = generalPars.passwstr;
        std::string passwnum = generalPars.passwnum;
        //    random binary watermark   ********************
        int watermark[64];
        for (int i = 0; i < 64; i++) {
            int b = rand() % 2;
            watermark[i] = b;
        }
//        saving watermarking parameters
        string filepath = "/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/parameters.txt";
        ofstream par_file(filepath);
        if (!par_file) {
            cout << "File Not Opened" << endl;
        }
        for (int i = 0; i < 64; i++) {
            par_file << watermark[i] << " ";
            //     par_file<<endl;
        }
        par_file << endl;
        par_file << passwstr;
        par_file << endl;
        par_file << passwnum;
        par_file << endl;
        par_file << power;
        par_file.close();
        bool gt = true;
//        read video
        VideoCapture cap("/home/bene/ClionProjects/tesi_watermarking/img/output.mp4"); // open the default camera
        if (!cap.isOpened())  // check if we succeeded
            return -1;
        int frame_number = -1;
//        double cycle to process 100 frames at a time
        int first_frame = 0;
        int last_frame = 100;
        for (int i = 0; i < first_frame; i++) {
            frame_number++;
            Mat frame;
            cap >> frame;
        }
        for (int i = first_frame; i < last_frame; i++) {
            frame_number++;
            Mat frame;
            cap >> frame; // get a new frame from camera
            Mat marked_frame;
            frame.copyTo(marked_frame);
            FDTStereoWatermarking::leftWatermarking(frame, watermark, wsize, power, passwstr, passwnum, gt,
                                                    marked_frame);
            std::ostringstream path;
            path << "/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
            imwrite(path.str(), marked_frame);
        }
    }else if(decoding){     //detection
//    read parameters file
        string filepath = "/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/parameters.txt";
        string line;
        std::string passwstr;
        std::string passwnum;
        double power;
        ifstream myfile(filepath);
        getline(myfile, line);
        string wat = line;
        int wsize = wat.length() / 2;
        int watermark[wsize];
        stringstream stream(wat);
        int count = 0;
        while (1) {
            int n;
            stream >> n;
            if (!stream)
                break;
            watermark[count] = n;
            count++;
        }
        getline(myfile, line);
        passwstr = line;
        getline(myfile, line);
        passwnum = line;
        getline(myfile, line);
        string alpha = line;
        stringstream stream_alpha(alpha);
        double d;
        stream_alpha >> d;
        power = d;
//        read marked frames  ***********************
     //   VideoCapture cap("/home/bene/ClionProjects/tesi_watermarking/img/output_L_marked.mp4"); // open the default camera
        VideoCapture cap("/home/bene/ClionProjects/tesi_watermarking/video/output_L_marked_crf0.mp4"); // open the default camera
        if (!cap.isOpened())  // check if we succeeded
            return -1;
        int first_frame = 200;
        int last_frame = 300;
        for (int i = 0; i < first_frame; i++) {
            Mat frame;
            cap >> frame;
        }
        for (int i = first_frame; i < last_frame; i++) {
            Mat frame;
            cap >> frame; // get a new frame from camera
            //        imshow("frames", frame);
            //        waitKey(0);
            Mat marked_frame;
            frame.copyTo(marked_frame);
            bool left_det = FDTStereoWatermarking::leftDetection(marked_frame, watermark, 64, 0.3, passwstr, passwnum, 512);
            cout << "left_det " << left_det<<endl;
         }
    }

    // decoding one single frame
   /* cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/frame050.png",
                          CV_LOAD_IMAGE_COLOR);
    bool det = FDTStereoWatermarking::leftDetection(left, watermark, 64, 0.3, passwstr, passwnum, 512);
    cout << det;*/


  /*  cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/frame050.png",
                          CV_LOAD_IMAGE_COLOR);
    bool det = FDTStereoWatermarking::leftDetection(left, watermark, 64, 0.3, passwstr, passwnum, 512);
    cout << det;*/


/*   Mat edges;
   for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, CV_BGR2GRAY);

        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(30) >= 0) break;
    }*/

    //questo va ricontrollato
    //video processing
/*    CvCapture* capture = 0;
    IplImage *frame = 0;

    if (!(capture = cvCaptureFromFile("/home/bene/ClionProjects/tesi_watermarking/img/output.mp4")))
        printf("Cannot open video\n");

    cvNamedWindow("tsukuba", CV_WINDOW_AUTOSIZE);

    while (1) {

        frame = cvQueryFrame(capture);
        if (!frame)
            break;

        //non serve a nulla ma ti fa vedere che da un frame crea un'immagine e la mostra
        IplImage *temp = cvCreateImage(cvSize(frame->width/2, frame->height/2), frame->depth, frame->nChannels); // A new Image half size
        cvShowImage("temporary ", temp); // Display the frame

        cvResize(frame, temp, CV_INTER_CUBIC); // Resize
        cvShowImage("tsukuba", frame); // Display the frame
        cvReleaseImage(&temp);
        if (cvWaitKey(5000) == 27) // Escape key and wait, 5 sec per capture
            break;
    }

    cvReleaseImage(&frame);
    cvReleaseCapture(&capture);*/

//  spatialWatermarking::gaussianNoiseStereoWatermarking(gt);

//   FDTStereoWatermarking::warpMarkWatermarking(wsize, tilesize, power, clipping, flagResyncAll, tilelistsize, passwstr, passwnum, gt);

    //questo va ricontrollato
//   FDTStereoWatermarking::warpRightWatermarking(wsize, tilesize, power, clipping, flagResyncAll, tilelistsize, passwstr, passwnum,gt);

//   bool left_to_right = false;
//   graph_cuts_utils::kz_main(left_to_right,"left_watermarked","right_watermarked");

//    RRQualityMetrics::compute_metrics();



//////************WATERMARKING bene********************/////////////////////////

//    START COEFFICIENT ANALYSIS:  saving dft left coefficient   ********************
//    double *coeff_left = image_watermarking.getCoeff_dft();
//    int coeff_num = image_watermarking.getCoeff_number();
//    double *wat = new double[coeff_num];
//    wat = image_watermarking.getFinal_mark();
//    stereo_watermarking::writeToFile(wat,coeff_num,"/home/bene/Scrivania/wat.txt");
//    stereo_watermarking::writeToFile(coeff_left,coeff_num,"/home/bene/Scrivania/diag_coeff_left.txt");
//    decoding   ********************
//    bool left_detection = image_watermarking.extractWatermark(squared_marked_left, 256, 256);
//    cout<< "left_detection:    " << left_detection <<endl;
//    saving marked dft left coefficient   ********************
//    double *marked_coeff_left = image_watermarking.getMarked_coeff();
//    for (int i = 0; i < coeff_num; i++) {
//        marked_coeff_left[i] = marked_coeff_left[i]/coeff_left[i];
//    }
//    double *retrieve_left_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);
//    stereo_watermarking::writeMatToFile(retrieve_left_wat,coeff_num,"/home/bene/Scrivania/Tesi/retrieve_left_wat.txt");
//    stereo_watermarking::writeMatToFile(marked_coeff_left,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_left.txt");
//    stereo_watermarking::similarity_graph(100,coeff_num,wat);
//    constructing SQUARED RIGHT to compute dft analysis   ********************
//    cv::Mat right = imread("/home/bene/ClionProjects/tesi_watermarking/img/r.png",CV_LOAD_IMAGE_COLOR);
//    unsigned char *right_uchar = right.data;
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *squared_right =  new unsigned char[squared_dim];
//    cv::Mat right_squared = cv::Mat::zeros(256, 256, CV_8UC3);
//    unsigned char d_val = disp.at<uchar>(0,127);
//    for (int i = 0; i < nc_s; i ++ )
//        for (int j = 0; j < nc_s; j++) {
//            for (int k =0; k<3;k++){
//                squared_right[(i * nc_s + j)*3 + k] = right_uchar[(i *nc + j + offset - d_val)*3 + k];
//            }
//        }
//    unsigned char *squared_dft_marked_right = image_watermarking.insertWatermark(squared_right,256,256);
////    saving dft right coefficient   ********************
//    double *coeff_right = image_watermarking.getCoeff_dft();
//    stereo_watermarking::writeToFile(coeff_right,coeff_num,"/home/bene/Scrivania/diag_coeff_right.txt");
//    spatial extraction of the watermark   ********************


//    double *squared_mark =  new double[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark[i] = (double)squared_marked_left[i] - (double)squared_left[i];
//    }
//    unsigned char* mark_uchar = new unsigned char[squared_dim];
//    for (int i=0; i<squared_dim;i++ ){
//       mark_uchar[i] = (unsigned char) squared_mark[i];
//    }
//    stereo_watermarking::compute_magnitude_phase(mark_uchar,256,"mark");



//    double *retrieve_right_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);  // da modificare gli input
//    stereo_watermarking::writeMatToFile(marked_coeff_rec_left,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_rec_left.txt");

//    similarity   ********************
//    stereo_watermarking::similarity_measures(wat, wat, coeff_num,"inserted watermak", "inserted watermak");


    /* GRAPH CUTS DISPARITY COMPUTATION*/
//
//    std::string img1_path =  "/home/bene/ClionProjects/tesi_watermarking/img/l.png";
//    std::string img2_path =  "/home/bene/ClionProjects/tesi_watermarking/img/r.png";
//
//
//    Match::Parameters params = { // Default parameters
//            Match::Parameters::L2, 1, // dataCost, denominator
//            8, -1, -1, // edgeThresh, lambda1, lambda2 (smoothness cost)
//            -1,        // K (occlusion cost)
//            4, false   // maxIter, bRandomizeEveryIteration
//    };
//    float K=-1, lambda=-1, lambda1=-1, lambda2=-1;
//    params.dataCost = Match::Parameters::L1;
////      params.dataCost = Match::Parameters::L2;
//
//    GeneralImage im1 = (GeneralImage)imLoad(IMAGE_GRAY, img1_path.c_str());
//    GeneralImage im2 = (GeneralImage)imLoad(IMAGE_GRAY, img2_path.c_str());
//    bool color = false;
//    if(graph_cuts_utils::isGray((RGBImage)im1) && graph_cuts_utils::isGray((RGBImage)im2)) {
//        color=false;
//        graph_cuts_utils::convert_gray(im1);
//        graph_cuts_utils::convert_gray(im2);
//    }
//
//    Match m(im2, im1, color);
////
////////    // Disparity
//    int dMin=19, dMax=77;   //r-l
////    int dMin=-77, dMax=-19;  //l-r
////    int dMin=8, dMax=33;  // r-l syn
////
//    m.SetDispRange(dMin, dMax);
//
//    time_t seed = time(NULL);
//    srand((unsigned int)seed);
//
//    graph_cuts_utils::fix_parameters(m, params, K, lambda, lambda1, lambda2);
//
//    m.KZ2();
//
////        m.SaveXLeft(argv[5]);
//
//    m.SaveScaledXLeft("/home/bene/ClionProjects/tesi_watermarking/img/disp_rl_kz.png", true); //r -l
////    m.SaveScaledXLeft("/home/bene/ClionProjects/tesi_watermarking/img/disp_lr_kz.png", false);  //l-r
//
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_rl_kz.png");
//    imshow("kz disp",disp);
//    waitKey(0);
//////

    /*STEP 3: NORMALIZE DISPARITY (OUTPUT OF KZ)*/

//    cv::Mat nkz_disp;
//    cv::Mat kz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_rl_kz.png", CV_LOAD_IMAGE_GRAYSCALE);
//    if (kz_disp.rows == 0){
//        cout << "Empty image";
//    } else {
//        Disp_opt dp;
//        dp.disparity_normalization(kz_disp, nkz_disp);
//    }
////
//    imwrite("/home/bene/ClionProjects/tesi_watermarking/img/norm_disp_rl_kz.png",nkz_disp);
//    similarity   ********************
//    double threshold = stereo_watermarking::threshold_computation(coeff_left, coeff_num, power);
//    double threshold = stereo_watermarking::threshold_computation(coeff_right, coeff_num, power);
//    cout<< "threshold:  "<<threshold<<endl;
//    stereo_watermarking::similarity_measures(wat, coeff_left, coeff_num,"inserted watermak", "coeff_left");
//    float correlation = stereo_watermarking::similarity_measures(wat, marked_coeff_left, coeff_num,"inserted watermak", "marked_coeff_left");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_right, coeff_num,"inserted watermak", "marked_coeff_right");
//    stereo_watermarking::similarity_measures(warp_mark_coeff, marked_coeff_right,coeff_num,"warp_mark_coeff", "marked_coeff_right");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_rec_left, coeff_num,"inserted watermak", "marked_coeff_rec_left");



//    stereo_watermarking::compute_coeff_function(squared_left,256,"coeff_left");
//    stereo_watermarking::compute_coeff_function(squared_right,256,"coeff_right");
//    stereo_watermarking::compute_coeff_function(squared_mark_uchar,256,"coeff_mark");
//    stereo_watermarking::compute_coeff_function(squared_warped_mark,256,"coeff_warped_mark");
//    stereo_watermarking::compute_coeff_function(squared_marked_left,256,"coeff_marked_left");
//    stereo_watermarking::compute_coeff_function(squared_marked_right,256,"coeff_marked_right");
//    stereo_watermarking::compute_coeff_function(squared_left_ric,256,"coeff_left_ric");




//    imshow   ********************
//    stereo_watermarking::show_ucharImage(squared_left, 256, 256, "squared left");
//    stereo_watermarking::show_ucharImage(mark_uchar, 256, 256, "mark_uchar");
//    stereo_watermarking::show_doubleImage(squared_mark, 256, 256, "squared_mark");
//    stereo_watermarking::show_ucharImage(squared_marked_left, 256, 256, "squared marked left");
//    stereo_watermarking::show_doubleImage(squared_mark, 256, 256, "squared_mark");
//    stereo_watermarking::show_ucharImage(squared_mark_uchar, 256, 256, "squared_mark_uchar");
//
//    stereo_watermarking::show_doubleImage(warped_mark, 640, 480, "mark_warped");
//    stereo_watermarking::show_ucharImage(squared_right, 256, 256, "squared right");
//    stereo_watermarking::show_ucharImage(marked_right, 640, 480, "marked_right");
//    stereo_watermarking::show_ucharImage(squared_marked_right, 256, 256, "squared_marked_right");
//    stereo_watermarking::show_ucharImage(squared_left_ric, 256, 256, "squared_left_ric");

//    show squared mark from reconstructed left   ********************
//    double *squared_mark_from_rec_left =  new double[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark_from_rec_left[i] = (double)squared_left[i] - (double)squared_left_ric[i]  ;
//    }
//    stereo_watermarking::show_doubleImage(squared_mark_from_rec_left, 256, 256, "squared_mark_from_rec_left");

//     dft check   ********************
/*
    stereo_watermarking::dft_comparison(squared_left,squared_right,256,"sinistra","destra");
    stereo_watermarking::dft_comparison(squared_left,squared_marked_left,256,"sinistra","sinistra_marchiata");
    stereo_watermarking::dft_comparison(right_squared.data,squared_marked_right,256,"destra", "destra_marchiata");
    stereo_watermarking::dft_comparison(squared_left, squared_left_ric,256,"sinistra" , "sinistra_marchiata_ricostruita");
    stereo_watermarking::dft_comparison(squared_marked_left, squared_left_ric,256,"sinistra_marchiata" , "sinistra_marchiata_ricostruita");
*/

//    saving marked dft left coefficient   ********************
//    double *marked_coeff_right = image_watermarking.getMarked_coeff();
////    double *retrieve_right_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);  // da modificare gli input
////    stereo_watermarking::writeToFile(marked_coeff_right,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_right.txt");
////    reconstructing marked left   ********************
////    cv::Mat right_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_right.png", CV_LOAD_IMAGE_GRAYSCALE);
//    unsigned char *right_disp_uchar = right_disp.data;
////    cv::Mat occ_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/occ_right.png", CV_LOAD_IMAGE_GRAYSCALE);
////    Right_view rv;
//    unsigned char *left_reconstructed_uchar = rv.left_uchar_reconstruction(marked_right, right_disp_uchar, occ_right.data,640,480);
////    constrution of the squared reconstructed marked left   ********************
//    unsigned char *squared_left_ric =  new unsigned char[squared_dim];
//    cv::Mat left_squared_reconstructed = cv::Mat::zeros(256, 256, CV_8UC3);
//    for (int i = 0; i < nc_s; i ++ )
//        for (int j = 0; j < nc_s; j++)
//            for(int k = 0; k<3; k++)
//                squared_left_ric[(i * nc_s + j)*3 + k] = left_reconstructed_uchar[(i * nc + j + offset)*3 + k];
//    bool rec_left_detection = image_watermarking.extractWatermark(squared_left_ric, 256, 256);
//    cout<< "rec_left_detection:    " << rec_left_detection <<endl;
//    double *marked_coeff_rec_left = image_watermarking.getMarked_coeff();
//    for (int i = 0; i < coeff_num; i++) {
//        marked_coeff_rec_left[i] = marked_coeff_rec_left[i];///coeff_left[i];
//    }

//    double *retrieve_right_wat = stereo_watermarking::not_blind_extraction(coeff_left,marked_coeff_left,coeff_num,power);  // da modificare gli input
//    stereo_watermarking::writeToFile(marked_coeff_rec_left,coeff_num,"/home/bene/Scrivania/Tesi/marked_coeff_rec_left.txt");

//    similarity   ********************
  //  stereo_watermarking::similarity_measures(wat, wat, coeff_num,"inserted watermak", "inserted watermak");


//    double threshold = stereo_watermarking::threshold_computation(coeff_left, coeff_num, power);
//    cout<< "threshold:  "<<threshold<<endl;

  //  stereo_watermarking::similarity_measures(wat, coeff_left, coeff_num,"inserted watermak", "coeff_left");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_left, coeff_num,"inserted watermak", "marked_coeff_left");
 //   stereo_watermarking::similarity_measures(wat, marked_coeff_right, coeff_num,"inserted watermak", "marked_coeff_right");
//    stereo_watermarking::similarity_measures(marked_coeff_right, warp_mark_coeff, coeff_num,"marked_coeff_right", "warp_mark_coeff");
//    stereo_watermarking::similarity_measures(wat, marked_coeff_rec_left, coeff_num,"inserted watermak", "marked_coeff_rec_left");


//    double* coeff_vector_left = stereo_watermarking::compute_coeff_function(squared_left,256,"coeff_vector_left");
//    unsigned char* mark_squared = new unsigned char [coeff_num];
//    for (int i = 0; i < coeff_num; i++) {
//        mark_squared[i] = (unsigned char) squared_mark[i];
//    }
//    double* coeff_vector_right = stereo_watermarking::compute_coeff_function(squared_right_to_mark,256,"vector_mark");
//    for (int i = 0; i < 8382; i++) {
//        coeff_vector_mark[i] = coeff_vector_mark[i]/coeff_vector_left[i];
//    }
//    stereo_watermarking::writeToFile(coeff_vector_mark,8383,"/home/bene/Scrivania/coeff_vector_mark.txt");




//    imshow   ********************
//    stereo_watermarking::show_ucharImage(squared_left, 256, 256, "squared left",3);
//    stereo_watermarking::show_ucharImage(squared_marked_left, 256, 256, "squared marked left");
//    stereo_watermarking::show_doubleImage(squared_mark, 256, 256, "squared_mark");
//    stereo_watermarking::show_ucharImage(squared_mark_uchar, 256, 256, "squared_mark_uchar");

//    stereo_watermarking::show_doubleImage(warped_mark, 640, 480, "mark_warped");
//    stereo_watermarking::show_ucharImage(squared_right_to_mark, 256, 256, "squared right",3);
//    stereo_watermarking::show_ucharImage(marked_right, 640, 480, "marked_right");
//    stereo_watermarking::show_ucharImage(squared_marked_right, 256, 256, "squared_marked_right");
//    stereo_watermarking::show_ucharImage(squared_left_ric, 256, 256, "squared_left_ric");

//    show squared mark from reconstructed left   ********************
//    double *squared_mark_from_rec_left =  new double[squared_dim];
//    for (int i=0;i<squared_dim;i++){
//        squared_mark_from_rec_left[i] = (double)squared_left[i] - (double)squared_left_ric[i]  ;
//    }
//    stereo_watermarking::show_doubleImage(squared_mark_from_rec_left, 256, 256, "squared_mark_from_rec_left");

//     dft check   ********************
/*
    stereo_watermarking::dft_comparison(squared_left,squared_right_to_mark,256,"sinistra","destra");

    stereo_watermarking::dft_comparison(left_squared.data,left_squared_marked.data,256,"sinistra","sinistra_marchiata");

    stereo_watermarking::dft_comparison(right_squared.data,right_marked_squared.data,256,"destra", "destra_marchiata");

    stereo_watermarking::dft_comparison(left_squared.data, left_squared_reconstructed.data,256,"sinistra" , "sinistra_marchiata_ricostruita");

    stereo_watermarking::dft_comparison(left_squared_marked.data, left_squared_reconstructed.data,256,"sinistra_marchiata" , "sinistra_marchiata_ricostruita");
*/




//    /*GENERAZIONE NUVOLA 3D*/
//
//    int frame_num=0; //serve per prendere i parametri dal file di testo ma per ora usiamo sempre il frame 0
//    cv::Mat nkz_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/nkz_disp.png", CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::generatePointCloud(nkz_disp,left,right,frame_num);

//
//    /*FINE GENERAZIONE NUVOLA 3D*/








    /* GRAPH CUTS DISPARITY COMPUTATION*/
////
//    std::string img1_path =  "/home/bene/ClionProjects/tesi_watermarking/img/l.png";
//    std::string img2_path =  "/home/bene/ClionProjects/tesi_watermarking/img/r.png";
//
//
//    Match::Parameters params = { // Default parameters
//            Match::Parameters::L2, 1, // dataCost, denominator
//            8, -1, -1, // edgeThresh, lambda1, lambda2 (smoothness cost)
//            -1,        // K (occlusion cost)
//            4, false   // maxIter, bRandomizeEveryIteration
//    };
//    float K=-1, lambda=-1, lambda1=-1, lambda2=-1;
//    params.dataCost = Match::Parameters::L1;
////      params.dataCost = Match::Parameters::L2;
//
//    GeneralImage im1 = (GeneralImage)imLoad(IMAGE_GRAY, img1_path.c_str());
//    GeneralImage im2 = (GeneralImage)imLoad(IMAGE_GRAY, img2_path.c_str());
//    bool color = false;
//    if(graph_cuts_utils::isGray((RGBImage)im1) && graph_cuts_utils::isGray((RGBImage)im2)) {
//        color=false;
//        graph_cuts_utils::convert_gray(im1);
//        graph_cuts_utils::convert_gray(im2);
//    }
//
//    Match m(im2, im1, color);
////
////////    // Disparity
//    int dMin=19, dMax=77;
////    int dMin=-77, dMax=-19;
////    int dMin=8, dMax=33;
////
//    m.SetDispRange(dMin, dMax);
//
//    time_t seed = time(NULL);
//    srand((unsigned int)seed);
//
//    graph_cuts_utils::fix_parameters(m, params, K, lambda, lambda1, lambda2);
//
//    m.KZ2();
//
////        m.SaveXLeft(argv[5]);
//
//    m.SaveScaledXLeft("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_rl.png", true);
////    m.SaveScaledXLeft("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_syn.png", false);
//
//    cv::Mat disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_rl.png");
//    imshow("kz disp",disp);
//    waitKey(0);
////

    /*STEP 2: FILTER DISPARITY (OUTPUT OF KZ)*/


//    cv::Mat disp_synt = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp_kz_syn.png", CV_LOAD_IMAGE_COLOR);
//    if (disp_synt.rows == 0){
//        cout << "Empty image";
//    } else {
//        Disp_opt dp;
//        dp.disparity_filtering(disp_synt);
//    }


    // path clion /home/bene/ClionProjects/tesi_watermarking/img/
    // path Scrivania /home/bene/Scrivania/




    /* QUALITY METRICS*/


//    cv::Mat disp_kz = imread( "/home/bene/ClionProjects/tesi_watermarking/img/nkz_right_dim_disp.png",CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::sobel_filtering(disp_kz,"sobel_disp");
//    cv::Mat sobel_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_disp.png",CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat disp_kz_wat = imread(  "/home/bene/ClionProjects/tesi_watermarking/img/norm_disp_from_wat.png",CV_LOAD_IMAGE_GRAYSCALE);
//    stereo_watermarking::sobel_filtering(disp_kz_wat,"sobel_disp_wat");
//    cv::Mat sobel_disp_wat  = imread( "/home/bene/ClionProjects/tesi_watermarking/img/sobel_disp_wat.png",CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat left_marked = imread("/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png", CV_LOAD_IMAGE_COLOR);
//    stereo_watermarking::sobel_filtering(left_marked,"sobel_left_w");
//    Mat sobel_left_w = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_left_w.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//    cv::Mat left2 = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
//    stereo_watermarking::sobel_filtering(left2,"sobel_left");
//    Mat sobel_left = imread("/home/bene/ClionProjects/tesi_watermarking/img/sobel_left.png", CV_LOAD_IMAGE_GRAYSCALE);
//
////    char* f1="/home/bene/ClionProjects/tesi_watermarking/img/left.png";
////    char* f2="/home/bene/ClionProjects/tesi_watermarking/img/left_marked.png";
////
////    qm::compute_quality_metrics(f1,f2,8);
//
//    double mqdepth = qm::MQdepth(disp_kz, disp_kz_wat,sobel_disp,sobel_disp_wat , 8,false);
//    cout << "MQdepth : " << mqdepth << endl;
//
//    double mqcolor = qm::MQcolor(left2,left_marked, sobel_disp, sobel_left_w, sobel_left,8,false);
//    cout << "MQcolor : " << mqcolor << endl;



    /*ENHANCING OCCLUSIONS*/

/*
    cv::Mat f_disp = imread("/home/bene/ClionProjects/tesi_watermarking/img/f_disp.png", CV_LOAD_IMAGE_COLOR);
    Disp_opt dp;
    dp.occlusions_enhancing(f_disp);
*/


    return 0;

}



//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf