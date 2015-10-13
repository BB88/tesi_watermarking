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

    /*CONFIG SETTINGS*/



    bool coding = true;
    bool decoding = false;
    bool gotVideo = false;
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
        VideoCapture capL("/home/bene/ClionProjects/tesi_watermarking/video/output.mp4"); // open the left camera
        if (!capL.isOpened())  // check if we succeeded
            return -1;
        VideoCapture capR("/home/bene/ClionProjects/tesi_watermarking/video/outputRight.mp4"); // open the right camera
        if (!capR.isOpened())  // check if we succeeded
            return -1;
        int frame_number = -1;
//        double cycle to process 100 frames at a time
        int first_frame = 0;
        int last_frame = 2;
        cv::Mat left = imread("/home/bene/ClionProjects/tesi_watermarking/img/l.png", CV_LOAD_IMAGE_COLOR);
        imshow("original",left);
        waitKey(0);
        for (int i = 0; i < first_frame; i++) {
            frame_number++;
            Mat frameL;
            capL >> frameL;
            Mat frameR;
            capR >> frameR;
        }
        for (int i = first_frame; i < last_frame; i++) {
            frame_number++;
            Mat frameL;
            capL >> frameL;
            imshow("left",frameL);
            waitKey(0);
            Mat frameR;
            capR >> frameR;
            Mat marked_frameL;
            Mat marked_frameR;
            FDTStereoWatermarking::videoWatermarking(frameL,frameR, watermark, wsize, power, passwstr, passwnum, gt,
                                                     marked_frameL,marked_frameR);
            std::ostringstream pathL;
            pathL << "/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/left/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
            imwrite(pathL.str(), marked_frameL);
            std::ostringstream pathR;
            pathR << "/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/right/frame_" << std::setw(3) << std::setfill('0') <<frame_number << ".png";
            imwrite(pathR.str(), marked_frameR);
        }
    }
    if(decoding){ //detection
        if (!gotVideo){
            Mat marked_left = imread("/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/old/frame000.png",CV_LOAD_IMAGE_COLOR);
            Mat marked_right = imread("/home/bene/ClionProjects/tesi_watermarking/img/marked_frames/0.3/left/frame_000.png",CV_LOAD_IMAGE_COLOR);
            imshow("left",marked_left);
            imshow("right",marked_right);
            waitKey(0);
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
            double d = 0.3;
            stream_alpha >> d;
            power = d;
            FDTStereoWatermarking::warpMarkWatermarking(64,0.3, passwstr, passwnum,true);
//    FDTStereoWatermarking::videoDetection(marked_left, marked_right, watermark, 64, 0.3, passwstr, passwnum, 512);
        }else {
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
            VideoCapture capL("/home/bene/ClionProjects/tesi_watermarking/video/output.mp4"); // open the default camera
            if (!capL.isOpened())  // check if we succeeded
                return -1;
            VideoCapture capR(
                    "/home/bene/ClionProjects/tesi_watermarking/video/outputRight.mp4"); // open the default camera
            if (!capR.isOpened())  // check if we succeeded
                return -1;
            int frame_number = -1;
            int first_frame = 200;
            int last_frame = 300;
            for (int i = 0; i < first_frame; i++) {
                frame_number++;
                Mat marked_frameL;
                capL >> marked_frameL;
                Mat marked_frameR;
                capR >> marked_frameR;
            }
            for (int i = first_frame; i < last_frame; i++) {
                frame_number++;
                Mat marked_frameL;
                capL >> marked_frameL;
                Mat marked_frameR;
                capR >> marked_frameR;
                FDTStereoWatermarking::videoDetection(marked_frameL, marked_frameR, watermark, 64, 0.3, passwstr,
                                                      passwnum, 512);
            }
        }
    }

    return 0;

}



//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf