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
#include <bits/stream_iterator.h>
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

const int STEP = 60; //this is the watermarking step, meaning only one frame every STEP will be watermarked, in this case we are only marking the I frames

int stereovideoCoding(std::string videoPath ){

    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();

    int wsize = pars.wsize;
    float power = pars.power;

    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();

    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    //    random binary watermark generation and saving to the config file  ********************
    int watermark[64];
    for (int i = 0; i < 64; i++) {
        int b = rand() % 2;
        watermark[i] = b;
    }
    std::ifstream in("/home/miky/ClionProjects/tesi_watermarking/config/config.cfg");
    std::ofstream out("/home/miky/ClionProjects/tesi_watermarking/config/config.cfg.tmp");
    string data;
    string dataw;
    if (in.is_open() && out.is_open()) {
        while (!in.eof()) {
            getline(in, data);
            if (!data.find("watermark")) {
                dataw.append("watermark = \"");
                for (int i = 0; i < 64; i++) {
                    dataw.append(std::to_string(watermark[i]));
                }
                dataw.append("\";");
                out << dataw << "\n";
            }
            else out << data << "\n";

            if (0 != std::rename("/home/miky/ClionProjects/tesi_watermarking/config/config.cfg.tmp", "/home/miky/ClionProjects/tesi_watermarking/config/config.cfg"))
            {
                // Handle failure.
            }
        }
        in.close();
        out.close();
    }

    // load the video to watermark
    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }

    int first_frame = 0;
    int last_frame = 1800;

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    cv::Mat new_frameStereo;
    vector<cv::Mat> markedLR;

    //marking and saving stereo frames
    for(int i = first_frame; i < last_frame; i++)
    {
        if(i%STEP==0){
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
            frameStereo(Rect(640,0,640,480)).copyTo(frameR);
            markedLR = DFTStereoWatermarking::stereoWatermarking(frameL,frameR,wsize,power,passwstr,passwnum,watermark, i);
            hconcat(markedLR[0],markedLR[1],new_frameStereo);
            std::ostringstream pathL;
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_06/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), new_frameStereo);
        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            std::ostringstream pathL;
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_06/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), frameStereo);

        }
    }


}
int stereovideoDecoding(std::string videoPath){

    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();

    int wsize = pars.wsize;
    float power = pars.power;
    std::string watermark = pars.watermark;

    int mark[wsize];
    for(int i=0;i<wsize;i++){
        mark[i] = watermark.at(i)-48; //codifica ASCII dei caratteri
    }

    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();

    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }

    int first_frame = 0;
    int last_frame = 1800;

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;

    int decoded_both_frames = 0; //conto in quati frame è rilevato il marchio
    int decoded_one_frame = 0;

    for (int i = first_frame; i < last_frame; i++) {

    if(i%STEP==0){

        capStereo >> frameStereo;
        if (frameStereo.empty()) break;
        frameStereo(Rect(0,0,640,480)).copyTo(frameL);
        frameStereo(Rect(640,0,640,480)).copyTo(frameR);
        int det = DFTStereoWatermarking::stereoDetection(frameL,frameR,wsize,power,passwstr,passwnum,mark,i);
        if (det == 1)
            decoded_one_frame++;
        if (det == 2)
            decoded_both_frames++;
    }
    else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
         }
    }

    cout<<"processed stereo frames: "<<1800/60<<endl;
    cout<<"decoded in both frames: "<<decoded_both_frames<<endl;
    cout<<"decoded in one frame: "<<decoded_one_frame<<endl;


}
void videoMaker(){

}
int main() {

//    std::string videoPath = "/home/miky/ClionProjects/tesi_watermarking/img/stereo_video_crf1_g60.mp4";
//    stereovideoCoding(videoPath);

    std::string videoPath = "/home/miky/Scrivania/Tesi/marked_videos/stereo_marked_video_crf25_g60.mp4";
    stereovideoDecoding(videoPath);
    //costruisce frame stereo



    return 0;

}




//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf