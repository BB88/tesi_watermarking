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
#include <boost/algorithm/string.hpp>

#include "./roc/roc.h"

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
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_08/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), new_frameStereo);
        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            std::ostringstream pathL;
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_08/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
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
    int decoded_left_frame = 0;
    int decoded_right_frame = 0;

    for (int i = first_frame; i < last_frame; i++) {

    if(i%STEP==0){

        capStereo >> frameStereo;
        if (frameStereo.empty()) break;
        frameStereo(Rect(0,0,640,480)).copyTo(frameL);
        frameStereo(Rect(640,0,640,480)).copyTo(frameR);
        int det = DFTStereoWatermarking::stereoDetection(frameL,frameR,wsize,power,passwstr,passwnum,mark,i);
        switch (det){
            case (0): break;
            case (1): decoded_both_frames++;break;
            case (2): decoded_left_frame++;break;
            case (3): decoded_right_frame++;break;
        }
    }
    else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
         }
    }

    cout<<"processed stereo frames: "<<1800/60<<endl;
    cout<<"decoded in both frames: "<<decoded_both_frames<<endl;
    cout<<"decoded in left frame: "<<decoded_left_frame<<endl;
    cout<<"decoded in right frame: "<<decoded_right_frame<<endl;


}
void videoMaker(){

}
void transparent_check(){

    std::string video = "/home/miky/ClionProjects/tesi_watermarking/video/video_left.mp4";
    std::string wat_video = "/home/miky/ClionProjects/tesi_watermarking/video/video_left_marked.mp4";
    std::string disp_video = "/home/miky/ClionProjects/tesi_watermarking/video/video_disp.mp4";
    std::string wat_disp_video = "/home/miky/ClionProjects/tesi_watermarking/video/video_disp_marked.mp4";

    RRQualityMetrics::compute_metrics(STEP, video, wat_video, disp_video, wat_disp_video );

}

void disparity_saving(){

    cv::Mat dispInL;
    cv::Mat dispInR;
    cv::Mat dispOutL = cv::Mat::zeros(480,640,CV_8UC1);
    cv::Mat dispOutR = cv::Mat::zeros(480,640,CV_8UC1);;
    Disp_opt opt;
    for (int i = 0; i<1800;i+=STEP){

        // prendo dmin e dmax e calcolo disp con kz
        std::string disp_data;
        std::vector<std::string> disprange;
        char sep = ' ';
        std::ifstream in("/home/miky/Scrivania/Tesi/dispRange.txt");
        if (in.is_open()) {
            int j=0;
            while (!in.eof()){
                if ( j == i ){
                    getline(in, disp_data);
                    for(size_t p=0, q=0; p!=disp_data.npos; p=q){
                        disprange.push_back(disp_data.substr(p+(p!=0), (q=disp_data.find(sep, p+1))-p-(p!=0)));
                    }
                    break;
                }
                getline(in, disp_data);
                j+=60;
            }
            in.close();
        }

        int dminl = atoi(disprange[0].c_str());
        int dmaxl = atoi(disprange[1].c_str());
        int dmaxr = -dminl;
        int dminr = -dmaxl;

        std::ostringstream pathInL;
        pathInL << "/home/miky/ClionProjects/tesi_watermarking/img/kz_disp_synt/disp_synt_" << i <<"_to_left_"<<i<< ".png";
        std::ostringstream pathInR;
        pathInR << "/home/miky/ClionProjects/tesi_watermarking/img/kz_disp_synt/disp_synt_" << i <<"_to_left_"<<i<< ".png";

        dispInL = imread(pathInL.str().c_str(),CV_LOAD_IMAGE_COLOR);
        dispInR = imread(pathInR.str().c_str(),CV_LOAD_IMAGE_COLOR);
//          imshow("dlin",dispInL);
//          imshow("drin",dispInR);
//          waitKey(0);

        cv::cvtColor(dispInL,dispInL,CV_BGR2GRAY);
        cv::cvtColor(dispInR,dispInR,CV_BGR2GRAY);

        opt.disparity_normalization(dispInL,dminl,dmaxl,dispOutL);
        opt.disparity_normalization(dispInR,dminl,dmaxl,dispOutR);

//          imshow("dl",dispOutL);
//          imshow("dr",dispOutR);
//          waitKey(0);
//
        std::ostringstream pathOutL;
        pathOutL << "/home/miky/ClionProjects/tesi_watermarking/img/kz_norm_from_video/left_" << std::setw(2) << std::setfill('0') << i/60 << ".png";

        std::ostringstream pathOutR;
        pathOutR <<  "/home/miky/ClionProjects/tesi_watermarking/img/kz_norm_from_video/right_" << std::setw(2) << std::setfill('0') << i/60 << ".png";

//          cout<<pathOutL.str()<<endl<<pathOutR.str()<<endl;
        imwrite(pathOutL.str(),dispOutL);
        imwrite(pathOutR.str(),dispOutR);


    }

}
void synthetized_decoding(){

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


    int first_frame = 0;
    int last_frame = 30;

    cv::Mat frameL;
    cv::Mat frameSynt;

    int decoded_both_frames = 0; //conto in quati frame è rilevato il marchio
    int decoded_left_frame = 0;
    int decoded_right_frame = 0;

    for (int i = first_frame; i < last_frame; i++) {

        std::ostringstream pathL;
        pathL << "/home/miky/ClionProjects/tesi_watermarking/img/VS/left/left_" << std::setw(5) << std::setfill('0') << i +1 << ".png";
        frameL = imread(pathL.str().c_str(), CV_LOAD_IMAGE_COLOR);
//        imshow("frameL",frameL);
//        waitKey(0);

        std::ostringstream pathSynt;
        pathSynt << "/home/miky/ClionProjects/tesi_watermarking/img/VS/synth_view_75/synth_view"<<i+1<<".png";
        frameSynt = imread(pathSynt.str().c_str(), CV_LOAD_IMAGE_COLOR);

//        imshow("frameSynt",frameSynt);
//        waitKey(0);

        int det = DFTStereoWatermarking::stereoDetection(frameL,frameSynt,wsize,power,passwstr,passwnum,mark,i);

        switch (det){
            case (0): break;
            case (1): decoded_both_frames++;break;
            case (2): decoded_left_frame++;break;
            case (3): decoded_right_frame++;break;
        }


    }

    cout<<"decoded in both frames: "<<decoded_both_frames<<endl;
    cout<<"decoded in left frame: "<<decoded_left_frame<<endl;
    cout<<"decoded in right frame: "<<decoded_right_frame<<endl;


}


void spatialMarking(std::string videoPath,cv::Mat noise){

    // load the video to watermark
    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return ;
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

            markedLR = spatialWatermarking::gaussianNoiseStereoWatermarking(frameL,frameR,noise,true,i);

            hconcat(markedLR[0],markedLR[1],new_frameStereo);
//            imshow("stereo", new_frameStereo);
//            waitKey(0);

            std::ostringstream pathL;
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_gaussian_1_kz/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), new_frameStereo);

        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            std::ostringstream pathL;
            pathL << "/home/miky/ClionProjects/tesi_watermarking/img/marked_frames_gaussian_1_kz/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), frameStereo);

        }
    }
}
void spatialDecoding(std::string videoPath,cv::Mat noise){

    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return ;
    }

    int first_frame = 0;
    int last_frame = 1800;

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;

    int decoded_both_frames = 0; //conto in quati frame è rilevato il marchio
    int decoded_left_frame = 0;
    int decoded_right_frame = 0;

    ofstream fout("/home/miky/Scrivania/Tesi/gaussDetection3_30.txt");

    for (int i = first_frame; i < last_frame; i++) {

        if(i%STEP==0){

            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
            frameStereo(Rect(640,0,640,480)).copyTo(frameR);
            vector<float> det = spatialWatermarking::gaussianNoiseStereoDetection(frameL,frameR,noise,true,i);
            fout<<det[0]<<"\t1"<<endl<<det[1]<<"\t0"<<endl<<det[2]<<"\t1"<<endl<<det[3]<<"\t1"<<endl;
//            switch (det){
//                case (0): break;
//                case (1): decoded_both_frames++;break;
//                case (2): decoded_left_frame++;break;
//                case (3): decoded_right_frame++;break;
//            }
        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
        }
    }

    fout.close();

}

void disparity_computation(){


    cv::Mat dispInR;
    cv::Mat dispOutR = cv::Mat::zeros(480,640,CV_8UC1);


    Disp_opt opt;

    bool left_to_right = false;

    for (int i = 0; i<30;i++){

        // prendo dmin e dmax e calcolo disp con kz
        std::string disp_data;
        std::vector<std::string> disprange;
        char sep = ' ';
        std::ifstream in("./dispRange.txt");
        if (in.is_open()) {
            int j=0;
            while (!in.eof()){
                if ( j == i ){
                    getline(in, disp_data);
                    for(size_t p=0, q=0; p!=disp_data.npos; p=q){
                        disprange.push_back(disp_data.substr(p+(p!=0), (q=disp_data.find(sep, p+1))-p-(p!=0)));
                    }
                    break;
                }
                getline(in, disp_data);
                j++;
            }
            in.close();
        }

        std::ostringstream pathL;
        pathL << "./VS/left/left_"<< std::setw(5) << std::setfill('0') << i+1 << ".png";
        cv::Mat frameL = imread(pathL.str(),CV_LOAD_IMAGE_COLOR);

        int dminl_50 = atoi(disprange[0].c_str())/2;
        int dmaxl_50 = atoi(disprange[1].c_str())/2;

        int dmaxr_50 = -dminl_50;
        int dminr_50 = -dmaxl_50;

        std::cout<<dminr_50<<endl<<dmaxr_50<<endl;

        std::ostringstream pathSynt_50;
        pathSynt_50 << "./VS/synth_view_50/synth_view"<< i+1 << ".png";
        cv::Mat synt_50 = imread(pathSynt_50.str(),CV_LOAD_IMAGE_COLOR);

        graph_cuts_utils::kz_main(left_to_right,"left","synt",frameL, synt_50 ,dminr_50,dmaxr_50);
        cv::Mat disp_right_50 = imread("./disp_synt_to_left.png",CV_LOAD_IMAGE_COLOR);

        std::ostringstream pathInR_50;
        pathInR_50 << "./norm_disparities/norm_disp_50_synt_to_left_"<< i+1 << ".png";

        cv::cvtColor(disp_right_50,disp_right_50,CV_BGR2GRAY);
        opt.disparity_normalization(disp_right_50,dminl_50,dmaxl_50,dispOutR);

//        imshow(" frameL",frameL);
//        imshow("synt_50 ",synt_50);
//        imshow("dispOutR ",dispOutR);
//        waitKey(0);

        imwrite(pathInR_50.str(),dispOutR);

        int dminl_25 = atoi(disprange[0].c_str())/4;
        int dmaxl_25 = atoi(disprange[1].c_str())/4;

        int dmaxr_25 = -dminl_25;
        int dminr_25 = -dmaxl_25;

        std::ostringstream pathSynt_25;
        pathSynt_25 << "./VS/synth_view_25/synth_view"<< i+1 << ".png";
        cv::Mat synt_25 = imread(pathSynt_25.str(),CV_LOAD_IMAGE_COLOR);

        graph_cuts_utils::kz_main(left_to_right,"left","synt",frameL, synt_25 ,dminr_25,dmaxr_25);
        cv::Mat disp_right_25 = imread("./disp_synt_to_left.png",CV_LOAD_IMAGE_COLOR);

        std::ostringstream pathInR_25;
        pathInR_25 << "./norm_disparities/norm_disp_25_synt_to_left_"<< i+1 << ".png";

        cv::cvtColor(disp_right_25,disp_right_25,CV_BGR2GRAY);
        opt.disparity_normalization(disp_right_25,dminl_25,dmaxl_25,dispOutR);

        imwrite(pathInR_25.str(),dispOutR);

        int dminl_75 = atoi(disprange[0].c_str())*3/4;
        int dmaxl_75 = atoi(disprange[1].c_str())*3/4;

        int dmaxr_75 = -dminl_75;
        int dminr_75 = -dmaxl_75;

        std::ostringstream pathSynt_75;
        pathSynt_75 << "./VS/synth_view_75/synth_view"<< i+1 << ".png";
        cv::Mat synt_75 = imread(pathSynt_75.str(),CV_LOAD_IMAGE_COLOR);

        graph_cuts_utils::kz_main(left_to_right,"left","synt",frameL, synt_75 ,dminr_75,dmaxr_75);
        cv::Mat disp_right_75 = imread("./disp_synt_to_left.png",CV_LOAD_IMAGE_COLOR);

        std::ostringstream pathInR_75;
        pathInR_75 << "./norm_disparities/norm_disp_75_synt_to_left_"<< i+1 << ".png";

        cv::cvtColor(disp_right_75,disp_right_75,CV_BGR2GRAY);
        opt.disparity_normalization(disp_right_75,dminl_75,dmaxl_75,dispOutR);

        imwrite(pathInR_75.str(),dispOutR);
    }

}

int main() {

//    double m_NoiseStdDev=1;
//
//    Mat noise = cv::Mat::zeros(480, 640 , CV_8UC3);
//    randn(noise,0,m_NoiseStdDev);


//    std::string videoPath = "/home/miky/ClionProjects/tesi_watermarking/img/stereo_video_crf1_g60.mp4";
//    stereovideoCoding(videoPath);

//    spatialMarking(videoPath,noise);


//    std::string videoPath = "/home/miky/Scrivania/Tesi/marked_videos/marked_video_gaussian_3_crf30_g60.mp4";
//    stereovideoDecoding(videoPath);

//    spatialDecoding(videoPath,noise);

    //RR metrics
//   transparent_check();

    //ROC curve
    ROC roc("/home/miky/Scrivania/Tesi/gaussDetection3_30.txt");
    // the format of the output file is
    // column 0 -> False positive points in curve
    // column 1 -> True positive points in curve
    roc.writeToFile("/home/miky/Scrivania/Tesi/ROC_gauss_3_30.txt");

//    synthetized_decoding();

//    disparity_computation();

    return 0;

}




//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf