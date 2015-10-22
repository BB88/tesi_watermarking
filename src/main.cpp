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

#include "DFTwatermarking/frequencyWatermarking.h"
#include "utils.h"
#include "disparity_optimization/disp_opt.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;
using namespace graph_cuts_utils;
using namespace qm;
using namespace RRQualityMetrics;
using namespace spatialWatermarking;




int main() {


    //costruisce frame stereo

//    for (int i = 1; i<=1800;i++){
//
//        std::stringstream frameLpath;
//        std::stringstream frameRpath;
//        std::stringstream joinPath;
//        cv::Mat left;
//        cv::Mat right;
//        cv::Mat joinLR;
//
//        frameLpath << "/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/daylight/left/tsukuba_daylight_L_" << std::setw(5) << std::setfill('0') << i << ".png";
//        frameRpath << "/home/miky/ClionProjects/tesi_watermarking/dataset/NTSD-200/daylight/right/tsukuba_daylight_R_" << std::setw(5) << std::setfill('0') << i << ".png";
//
//        joinPath << "/home/miky/ClionProjects/tesi_watermarking/img/stereo_frames/stereo_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
//
//        left =  cv::Mat::zeros(480,640, CV_8UC3);
//        right = cv::Mat::zeros(480,640, CV_8UC3);
//
//        left = imread(frameLpath.str().c_str(),CV_LOAD_IMAGE_COLOR);
//        right = imread(frameRpath.str().c_str(),CV_LOAD_IMAGE_COLOR);
//
////        imshow("left",left);
////        imshow("right",right);
////        waitKey(0);
//        joinLR = cv::Mat::zeros(left.rows,left.cols, CV_8UC3);
//
//        hconcat(left,right,joinLR);
//
//        imwrite (joinPath.str(),joinLR);
//
//    }
// CODING MIKY

    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();

    int wsize = pars.wsize;
    float power = pars.power;

    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();

    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    //    random binary watermark   ********************
    int watermark[64];
    for (int i = 0; i < 64; i++) {
        int b = rand() % 2;
        watermark[i] = b;
    }

//    bool gt = true;
//        read video
    VideoCapture capStereo("/home/miky/ClionProjects/tesi_watermarking/img/stereo_video_crf1_g60.mp4"); // open the left camera
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }


    int step = 60;
    int first_frame = 0;
    int last_frame = 1800;

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    cv::Mat new_frameStereo;
    vector<cv::Mat> markedLR;

    // non funziona pare ci sia un bug in ffmpeg

//    std::string NAME = "/home/miky/ClionProjects/tesi_watermarking/img/stereo_video_marked_crf1_g60.mp4";
//    int ex = static_cast<int>(capStereo.get(CV_CAP_PROP_FOURCC));
//    // Transform from int to char via Bitwise operators
//    char EXT[] = { (char) (ex & 0XFF), (char) ((ex & 0XFF00) >> 8),
//                   (char) ((ex & 0XFF0000) >> 16), (char) ((ex & 0XFF000000)
//                                                           >> 24), 0 };
//    cout << "Input codec type: " << EXT << endl;
//    Size S = Size((int) capStereo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
//                  (int) capStereo.get(CV_CAP_PROP_FRAME_HEIGHT));
//
//
//
//    VideoWriter outputVideo(NAME, ex, capStereo.get(CV_CAP_PROP_FPS), S, true);
////    outputVideo.open(NAME, ex=-1, capStereo.get(CV_CAP_PROP_FPS), S, true);
//
//    if (!outputVideo.isOpened())
//    {
//        cout  << "Could not open the output video to write " << endl;
//        return -1;
//    }

//    for(int i = first_frame; i < last_frame; i++) //Show the image captured in the window and repeat
//    {
//        if(i%step==0){
//            capStereo >> frameStereo;
//            if (frameStereo.empty()) break;
//            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
//            frameStereo(Rect(640,0,640,480)).copyTo(frameR);
//            markedLR = DFTStereoWatermarking::stereoWatermarking(frameL,frameR,wsize,power,passwstr,passwnum,watermark, i);
//            hconcat(markedLR[0],markedLR[1],new_frameStereo);
//            std::ostringstream pathL;
//            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_06/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
//            imwrite(pathL.str(), new_frameStereo);
//        }
//        else {
//            capStereo >> frameStereo;
//            if (frameStereo.empty()) break;
//            std::ostringstream pathL;
//            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_06/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
//            imwrite(pathL.str(), frameStereo);
//
//        }
//    }




    for (int i = first_frame; i < last_frame; i++) {
        if(i%step==0){
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;

            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
            frameStereo(Rect(640,0,640,480)).copyTo(frameR);

//            DFTStereoWatermarking::stereoWatermarking(frameL,frameR,wsize,power,passwstr,passwnum,watermark,i);
//            imshow("left ", frameL);
//            imshow("right ", frameR);
//            waitKey(0);

            DFTStereoWatermarking::stereoDetection(frameL,frameR,wsize,power,passwstr,passwnum,watermark,i);
        }
        else {
                capStereo >> frameStereo;
                if (frameStereo.empty()) break;
             }
    }


//
//        DFTStereoWatermarking::videoWatermarking(frameL,frameR, watermark, wsize, power, passwstr, passwnum, gt,
//                                                 marked_frameL, marked_frameR);
//        std::ostringstream pathL;
//        pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/left/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
//        imwrite(pathL.str(), marked_frameL);
//        std::ostringstream pathR;
//        pathR << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/right/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
//        imwrite(pathR.str(), marked_frameR);



    /*CONFIG SETTINGS*/



//    bool coding = false;
//    bool decoding = false;
//    bool gotVideo = false;
//
//    if (coding) {
//
//        Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();
//        int wsize = pars.wsize;
//        float power = pars.power;
//        Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();
//        bool masking = generalPars.masking;
//        std::string passwstr = generalPars.passwstr;
//        std::string passwnum = generalPars.passwnum;
//        //    random binary watermark   ********************
//        int watermark[64];
//        for (int i = 0; i < 64; i++) {
//            int b = rand() % 2;
//            watermark[i] = b;
//        }
////        saving watermarking parameters
//        string filepath = "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/parameters.txt";
//        ofstream par_file(filepath);
//        if (!par_file) {
//            cout << "File Not Opened" << endl;
//        }
//        for (int i = 0; i < 64; i++) {
//            par_file << watermark[i] << " ";
//            //     par_file<<endl;
//        }
//        par_file << endl;
//        par_file << passwstr;
//        par_file << endl;
//        par_file << passwnum;
//        par_file << endl;
//        par_file << power;
//        par_file.close();
//
//        bool gt = true;
////        read video
//        VideoCapture capL("/home/miky/ClionProjects/tesi_watermarking/video/output.mp4"); // open the left camera
//        if (!capL.isOpened())  // check if we succeeded
//            return -1;
//        VideoCapture capR("/home/miky/ClionProjects/tesi_watermarking/video/outputRight.mp4"); // open the right camera
//        if (!capR.isOpened())  // check if we succeeded
//            return -1;
//        int frame_number = -1;
////        double cycle to process 100 frames at a time
//        int first_frame = 0;
//        int last_frame = 2;
//
//        //se voglio saltare i primi #first_frame
//        for (int i = 0; i < first_frame; i++) {
//            frame_number++;
//            Mat frameL;
//            capL >> frameL;
//            Mat frameR;
//            capR >> frameR;
//        }
//
//        for (int i = first_frame; i < last_frame; i++) {
//            frame_number++;
//            Mat frameL;
//            capL >> frameL;
//            Mat frameR;
//            capR >> frameR;
//
//            Mat marked_frameL;
//            Mat marked_frameR;
//
//            DFTStereoWatermarking::videoWatermarking(frameL,frameR, watermark, wsize, power, passwstr, passwnum, gt,
//                                                     marked_frameL, marked_frameR);
//            std::ostringstream pathL;
//            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/left/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
//            imwrite(pathL.str(), marked_frameL);
//            std::ostringstream pathR;
//            pathR << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/right/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
//            imwrite(pathR.str(), marked_frameR);
//        }
//    }
//    if(decoding){ //detection
//
//        if (!gotVideo){
//
//            Mat marked_left = imread("/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/left/frame_000.png",CV_LOAD_IMAGE_COLOR);
//            Mat marked_right = imread("/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/0.3/right/frame_000.png",CV_LOAD_IMAGE_COLOR);
//
//
//            string filepath = "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/parameters.txt";
//            string line;
//            std::string passwstr;
//            std::string passwnum;
//            double power;
//            ifstream myfile(filepath);
//            getline(myfile, line);
//            string wat = line;
//            int wsize = wat.length() / 2;
//            int watermark[wsize];
//            stringstream stream(wat);
//            int count = 0;
//            while (1) {
//                int n;
//                stream >> n;
//                if (!stream)
//                    break;
//                watermark[count] = n;
//                count++;
//            }
//            getline(myfile, line);
//            passwstr = line;
//            getline(myfile, line);
//            passwnum = line;
//            getline(myfile, line);
//            string alpha = line;
//            stringstream stream_alpha(alpha);
//            double d = 0.3;
//            stream_alpha >> d;
//            power = d;
////            DFTStereoWatermarking::warpMarkWatermarking(64,0.3, passwstr, passwnum,true);
//            DFTStereoWatermarking::videoDetection(marked_left, marked_right, watermark, 64, 0.3, passwstr, passwnum, 512);
//        }else {
////    read parameters file
//            string filepath = "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames/parameters.txt";
//            string line;
//            std::string passwstr;
//            std::string passwnum;
//            double power;
//            ifstream myfile(filepath);
//            getline(myfile, line);
//            string wat = line;
//            int wsize = wat.length() / 2;
//            int watermark[wsize];
//            stringstream stream(wat);
//            int count = 0;
//            while (1) {
//                int n;
//                stream >> n;
//                if (!stream)
//                    break;
//                watermark[count] = n;
//                count++;
//            }
//            getline(myfile, line);
//            passwstr = line;
//            getline(myfile, line);
//            passwnum = line;
//            getline(myfile, line);
//            string alpha = line;
//            stringstream stream_alpha(alpha);
//            double d;
//            stream_alpha >> d;
//            power = d;
////        read marked frames  ***********************
//            //   VideoCapture cap("/home/miky/ClionProjects/tesi_watermarking/img/output_L_marked.mp4"); // open the default camera
//            VideoCapture capL("/home/miky/ClionProjects/tesi_watermarking/video/output.mp4"); // open the default camera
//            if (!capL.isOpened())  // check if we succeeded
//                return -1;
//
//            VideoCapture capR("/home/miky/ClionProjects/tesi_watermarking/video/outputRight.mp4"); // open the default camera
//            if (!capR.isOpened())  // check if we succeeded
//                return -1;
//            int frame_number = -1;
//            int first_frame = 200;
//            int last_frame = 300;
//            for (int i = 0; i < first_frame; i++) {
//                frame_number++;
//                Mat marked_frameL;
//                capL >> marked_frameL;
//                Mat marked_frameR;
//                capR >> marked_frameR;
//            }
//            for (int i = first_frame; i < last_frame; i++) {
//                frame_number++;
//                Mat marked_frameL;
//                capL >> marked_frameL;
//                Mat marked_frameR;
//                capR >> marked_frameR;
//                DFTStereoWatermarking::videoDetection(marked_frameL, marked_frameR, watermark, 64, 0.3, passwstr,passwnum, 512);
//            }
//        }
//    }

    //questo ritrova tutto con disp NON di ground
//    DFTStereoWatermarking::warpMarkWatermarking(64,0.3, "flskdjsuyiajcens", "12578965" ,false);
    //  spatialWatermarking::gaussianNoiseStereoWatermarking(gt);


//   bool left_to_right = false;
//      graph_cuts_utils::kz_main(left_to_right,"left_watermarked","right_watermarked");

//    RRQualityMetrics::compute_metrics();


    return 0;

}



//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf