#include <iostream>
#include <opencv2/core/core.hpp>
#include "dataset/tsukuba_dataset.h"
#include <cv.h>
#include <highgui.h>


#include <fstream>

//includes watermarking
#include "./img_watermarking/watermarking.h"
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
//#include <boost/algorithm/string.hpp>

#include "./roc/roc.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace libconfig;
using namespace graph_cuts_utils;
using namespace qm;
using namespace RRQualityMetrics;
using namespace spatialWatermarking;

const int STEP = 10; //this is the watermarking step, meaning only one frame every STEP will be watermarked and decoded, in this case we are only marking the I frames

void showhelpinfo(char *s)
{
    cout<<"Usage:   "<<s<<" [-option] [input video/videos]  [output folder] "<<endl;
    cout<<"option:  "<<"-h  show help information"<<endl;
    cout<<"         "<<"-fe call frequency watermarking function"<<endl;
    cout<<"         "<<"-fd call frequency detection function"<<endl;
    cout<<"         "<<"-se call spatial watermarking function"<<endl;
    cout<<"         "<<"-sd call spatial detection function"<<endl;
    cout<<"         "<<"-d call disparity computation function"<<endl;
    cout<<"         "<<"-qm call quality metric computation"<<endl;
    cout<<"         "<<"-p compute PSNR"<<endl;
    cout<<"         "<<"-fm make stereo frames from two videos"<<endl;
}
/**
 * stereovideoCoding(..)
 *
 * frequency watermark embedding function, takes a stereo video sequence and generate the marked frames in the specified folder
 *
 * @param videoPath: path of the stereo video sequence to watermark
 * @return -1 if couldn't open the video sequence
 *
 */
int stereovideoCoding(std::string videoPath ,std::string folder){

    //load the watermark configuration parameters, specified in the .cfg file
    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();
    int wsize = pars.wsize;
    float power = pars.power;
    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();
    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    //    random binary watermark generation and saving to the .cfg file
    srand(time(NULL));
    int watermark[64];
    for (int i = 0; i < 64; i++) {
        int b = rand() % 2;
        watermark[i] = b;
    }

    std::ifstream in("../config/config.cfg");
    std::ofstream out("../config/config.cfg.tmp");

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
            if (0 != std::rename("../config/config.cfg.tmp", "../config/config.cfg"))

            {
                // Handle failure.
            }
        }
        in.close();
        out.close();
    }
    cout<<"cfg trovato"<<endl;

    // load the video to watermark
    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }
    //range of frames to mark
    int first_frame = 0;
    int last_frame = 3000000; //attenzione, mettere un numero altissimo, altrimenti il video si interrompe prima, se nn sappiamo di quanti frame è composto

    //marking and saving stereo frames
    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    cv::Mat new_frameStereo;
    vector<cv::Mat> markedLR;
    for(int i = first_frame; i < last_frame; i++)
    // forse è meglio mettere un while, cosi non c'e' bisogno di last_frame che serviva solo per
//        poter marchiare un tot di frame alla volta  visto che il processo era troppo lento

    {
        if(i%STEP==0){
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            /*aggiunta per le dimensioni*/
            frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
            frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
            /*fine aggiunta per le dimensioni*/
//                       frameStereo(Rect(0,0,640,480)).copyTo(frameL);
//                   frameStereo(Rect(640,0,640,480)).copyTo(frameR);
                   markedLR = DFTStereoWatermarking::stereoWatermarking(frameL,frameR,wsize,power,passwstr,passwnum,watermark, i);
                   hconcat(markedLR[0],markedLR[1],new_frameStereo);
                   std::ostringstream pathL;
                   pathL << folder << "/stereo_marked_frame_" << std::setw(3) << std::setfill('0') << i << ".png";

                   imwrite(pathL.str(), new_frameStereo);
               }
               else {
                   capStereo >> frameStereo;
                   if (frameStereo.empty()) break;
                   std::ostringstream pathL;
                   pathL<< folder <<  "/stereo_marked_frame_" << std::setw(3) << std::setfill('0') << i << ".png";
                   imwrite(pathL.str(), frameStereo);
        }
    }
}

/**
 * stereovideoDecoding(..)
 *
 * frequency watermark detection function
 *
 * @param videoPath: path of the stereo video sequence to decode
 * @return -1 if couldn't open the video sequence
 *
 */

int stereovideoDecoding(std::string videoPath){

    //load the watermark configuration parameters, specified in the .cfg file
    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();
    int wsize = pars.wsize;
    float power = pars.power;
    std::string watermark = pars.watermark;
    //convert the watermark string from ASCII to numbers
    int mark[wsize];
    for(int i=0;i<wsize;i++){
        mark[i] = watermark.at(i)-48;
    }
    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();
    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    // load the video to watermark
    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }

    //range of marked frames
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
        /*aggiunta per le dimensioni*/
        frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
        frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
        /*fine aggiunta per le dimensioni*/
//        frameStereo(Rect(0,0,640,480)).copyTo(frameL);
//        frameStereo(Rect(640,0,640,480)).copyTo(frameR);
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
    cout<<"decoded in both frames: "<<decoded_both_frames<<endl;
    cout<<"decoded in left frame: "<<decoded_left_frame<<endl;
    cout<<"decoded in right frame: "<<decoded_right_frame<<endl;

}


/**
 * synthetized_DFT_decoding(..)
 *
 * open the synthetized views and look for the watermark in the frequency domain
 */

void synthetized_DFT_decoding(){

    //load the watermark configuration parameters, specified in the .cfg file
    Watermarking_config::set_parameters_params pars = Watermarking_config::ConfigLoader::get_instance().loadSetParametersConfiguration();
    int wsize = pars.wsize;
    float power = pars.power;
    std::string watermark = pars.watermark;
    //convert the watermark string from ASCII to numbers
    int mark[wsize];
    for(int i=0;i<wsize;i++){
        mark[i] = watermark.at(i)-48;
    }
    Watermarking_config::general_params generalPars = Watermarking_config::ConfigLoader::get_instance().loadGeneralParamsConfiguration();
    std::string passwstr = generalPars.passwstr;
    std::string passwnum = generalPars.passwnum;

    int first_frame = 0;
    int last_frame = 30;
    cv::Mat frameL;
    cv::Mat frameSynt;
    int decoded_both_frames = 0;
    int decoded_left_frame = 0;
    int decoded_right_frame = 0;
    for (int i = first_frame; i < last_frame; i++) {
        std::ostringstream pathL;
        pathL << "./img/VS/left/left_" << std::setw(5) << std::setfill('0') << i +1 << ".png";
        frameL = imread(pathL.str().c_str(), CV_LOAD_IMAGE_COLOR);
        std::ostringstream pathSynt;
        pathSynt << "./img/VS/synth_view_75/synth_view"<<i+1<<".png";
        frameSynt = imread(pathSynt.str().c_str(), CV_LOAD_IMAGE_COLOR);
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

/**
 * synthetized_spatial_decoding(..)
 *
 * open the synthetized views and look for the watermark in the spatioal domain
 *
 * @params noise: image of the Gaussian noise watermark
 *
 */
void synthetized_spatial_decoding(cv::Mat noise){

    int first_frame = 0;
    int last_frame = 30;
    cv::Mat frameL;
    cv::Mat frameStereo;
    cv::Mat frameSynt;
    ofstream fout("./gaussDetection3_075_synt.txt");
    for (int i = first_frame; i < last_frame; i++) {
        std::ostringstream pathL;
        pathL << "./img/marked_frames_gaussian_3_kz/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i*60 << ".png";
        frameStereo = imread(pathL.str().c_str(), CV_LOAD_IMAGE_COLOR);
        /*aggiunta per le dimensioni*/
        frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
        /*fine aggiunta per le dimensioni*/
//        frameStereo(Rect(0,0,640,480)).copyTo(frameL);
        std::ostringstream pathSynt;
        pathSynt << "./img/gauss3_viewSyn/0.75/synth_view"<<i+1<<".png";
        frameSynt = imread(pathSynt.str().c_str(), CV_LOAD_IMAGE_COLOR);
        vector<float> det = spatialWatermarking::gaussianNoiseStereoDetection(frameL,frameSynt,noise,i);
        fout<<det[0]<<"\t1"<<endl<<det[1]<<"\t0"<<endl<<det[2]<<"\t1"<<endl<<det[3]<<"\t1"<<endl;
        }
    fout.close();
}

/**
 * spatialMarking(..)
 *
 * spatial watermark embedding function, takes a stereo video sequence and generate the marked frames in the specified folder
 *
 * @param videoPath: path of the stereo video sequence to watermark
 * @params noise: image of the Gaussian noise watermark
 * @return -1 if couldn't open the video sequence
 */
int spatialMarking(std::string videoPath,cv::Mat noise,std::string folder){

    // load the video to watermark
    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1 ;
    }

    //range of frames to mark
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
            /*aggiunta per le dimensioni*/
            frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
            frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
            /*fine aggiunta per le dimensioni*/
//            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
//            frameStereo(Rect(640,0,640,480)).copyTo(frameR);
            markedLR = spatialWatermarking::gaussianNoiseStereoWatermarking(frameL,frameR,noise,i);
            hconcat(markedLR[0],markedLR[1],new_frameStereo);
            std::ostringstream pathL;
            pathL << folder << "/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";
            imwrite(pathL.str(), new_frameStereo);
        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            std::ostringstream pathL;
            pathL << folder <<"/stereo_marked_frame_" << std::setw(5) << std::setfill('0') << i << ".png";

            imwrite(pathL.str(), frameStereo);
        }
    }
}

/**
 * spatialDecoding(..)
 *
 * spaytial watermark detection function
 *
 * @param videoPath: path of the stereo video sequence to decode
 * @params noise: image of the Gaussian noise watermark
 * @return -1 if couldn't open the video sequence
 *
 */

int spatialDecoding(std::string videoPath,cv::Mat noise){

    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }

    //range of marked frames
    int first_frame = 0;
    int last_frame = 1800;
    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    ofstream fout("./gauss3_30_detection.txt",std::ios_base::app);
    for (int i = first_frame; i < last_frame; i++) {
        if(i%STEP==0){
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
            /*aggiunta per le dimensioni*/
            frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
            frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
            /*fine aggiunta per le dimensioni*/
           /* frameStereo(Rect(0,0,640,480)).copyTo(frameL);
            frameStereo(Rect(640,0,640,480)).copyTo(frameR);*/
            vector<float> det = spatialWatermarking::gaussianNoiseStereoDetection(frameL,frameR,noise,i);
            //write a .txt file with the correlation values and the corresponding detection, in order to subsequentially compute the ROC function with Matlab
            fout<<det[0]<<"\t1"<<endl<<det[1]<<"\t0"<<endl<<det[2]<<"\t1"<<endl<<det[3]<<"\t1"<<endl;
        }
        else {
            capStereo >> frameStereo;
            if (frameStereo.empty()) break;
        }
    }
    fout.close();
    //ROC curve
    ROC roc("./gauss3_30_detection.txt");
//     the format of the output file is
//     column 0 -> False positive points in curve
//     column 1 -> True positive points in curve
    roc.writeToFile("./ROC_30_detection.txt");

}

/**
 * disparity_computation()
 *
 * compute the disparity maps using graph cuts algorithm and saving to the specified folder
 *
 *  @param videoPath: path of the stereo video sequence
 *  @return -1 if couldn't open the video sequence
 */

int  disparity_computation(std::string videoPath,std::string folder){

    VideoCapture capStereo(videoPath);
    if (!capStereo.isOpened()) {  // check if we succeeded
        cout << "Could not open the output video to read " << endl;
        return -1;
    }
    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    //frame range of which compute the disparities
    int first_frame = 0;
    int last_frame = 400;
    cv::Mat dispOutL = cv::Mat::zeros(1080,1920,CV_8UC1);
    cv::Mat dispOutR = cv::Mat::zeros(1080,1920,CV_8UC1);
    Disp_opt opt;
    for (int i = first_frame; i<last_frame;i++){
        if (i%STEP==0){
            capStereo >> frameStereo;
            /*aggiunta per le dimensioni*/
            frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
            frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
            /*fine aggiunta per le dimensioni*/
//            frameStereo(Rect(0,0,640,480)).copyTo(frameL);
//            frameStereo(Rect(640,0,640,480)).copyTo(frameR);
            std::string disp_data;
            std::vector<std::string> disprange;
            char sep = ' ';
            std::ifstream in("/home/miky/Scrivania/new_dataset/BasketdispRange2.txt");
            if (in.is_open()) {
                int j=0;
                while (!in.eof()){
                    if ( j == i/STEP ){
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

            int dminl = atoi(disprange[0].c_str());
            int dmaxl = atoi(disprange[1].c_str());
            int dmaxr = -dminl;
            int dminr = -dmaxl;

            graph_cuts_utils::kz_main(true,"left","right",frameL, frameR ,dminl,dmaxl);
            cv::Mat disp_left = imread("./disp_left_to_right.png",CV_LOAD_IMAGE_COLOR);
            std::ostringstream pathInL;
            pathInL << folder << "/norm_disp_left_to_right_"<< i/STEP << ".png";

            cv::cvtColor(disp_left,disp_left,CV_BGR2GRAY);
            opt.disparity_normalization(disp_left,dminl,dmaxl,dispOutL);
            imwrite(pathInL.str(),dispOutL);

            graph_cuts_utils::kz_main(false,"left","right",frameL, frameR ,dminr,dmaxr);
            cv::Mat disp_right = imread("./disp_right_to_left.png",CV_LOAD_IMAGE_COLOR);
            std::ostringstream pathInR;
            pathInR << folder <<  "/norm_disp_right_to_left_"<< i/STEP << ".png";

            cv::cvtColor(disp_right,disp_right,CV_BGR2GRAY);
            opt.disparity_normalization(disp_right,dminl,dmaxl,dispOutR);
            imwrite(pathInR.str(),dispOutR);

        }
        else {
            capStereo >> frameStereo;
        }
    }
}

/**
 * uniqueness_spatial_test(..)
 *
 * generate 100 different watermarks and try to find them in the stereo sequence
 *
 * @param videoPath: path of the marked stereo video sequence
 * @return -1 if couldn't open the video sequence
 */
int uniqueness_spatial_test(std::string videoPath){

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

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;
    capStereo >> frameStereo;
    /*aggiunta per le dimensioni*/
    frameStereo(Rect(0,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameL);
    frameStereo(Rect(frameStereo.cols/2,0,frameStereo.cols/2,frameStereo.rows)).copyTo(frameR);
    /*fine aggiunta per le dimensioni*/
 /*   frameStereo(Rect(0,0,640,480)).copyTo(frameL);
    frameStereo(Rect(640,0,640,480)).copyTo(frameR);*/
    int det = DFTStereoWatermarking::stereoDetection(frameL,frameR,wsize,power,passwstr,passwnum,mark,0);
    static const char alpha_char[] = "abcdefghijklmnopqrstuvwxyz";
    static const char num_char [] =  "0123456789";
    for (int i = 0; i < 100; i++) {

        char *string_pswd = new char[16];
        char *num_pswd = new char[8];
        for (int i = 0; i < 16; i++) {
            string_pswd[i] = alpha_char[rand() % (sizeof(alpha_char) - 1)];
        }
        for (int i = 0; i < 8; i++) {
            num_pswd[i] = num_char[rand() % (sizeof(num_char) - 1)];
        }
        srand(time(NULL));
        int watermark[64];
        for (int i = 0; i < 64; i++) {
            int b = rand() % 2;
            watermark[i] = b;
        }
        int det = DFTStereoWatermarking::stereoDetection(frameL,frameR,wsize,power,string_pswd,num_pswd,watermark,0);
    }
}

int make_stereo_frames(std::string videoPathL, std::string videoPathR, std::string folder){

    cv::Mat frameStereo;
    cv::Mat frameL;
    cv::Mat frameR;

    VideoCapture capStereoL(videoPathL);
    if (!capStereoL.isOpened()) {  // check if we succeeded
        cout << "Could not open the input L video to read " << endl;
        return -1;
    }
    VideoCapture capStereoR(videoPathR);
    if (!capStereoR.isOpened()) {  // check if we succeeded
        cout << "Could not open the input R video to read " << endl;
        return -1;
    }


    for(int i=0;i<400;i++)
    {

        capStereoL >> frameL;
        if (frameL.empty()) break;
        capStereoR >> frameR;
        if (frameR.empty()) break;
        hconcat(frameL,frameR,frameStereo);

        std::ostringstream pathS;
        pathS << folder << "/stereo_frame_" << i << ".png";
        imwrite(pathS.str(), frameStereo);

    }

}

int main(int argc, char* argv[]) {

    if(argc == 1)
    {
        showhelpinfo(argv[0]);
        exit(1);
    }

    string videoPath = argv[2];
    const char* tmp = argv[1];
    const char* folder;

    if (strcmp(tmp,"-h")==0){showhelpinfo(argv[0]);}
    if (strcmp(tmp,"-fe")==0){
        cout<<"frequency watermarking process---"<<endl;
        folder = argv[3];
        stereovideoCoding(videoPath, folder);
    }
    if (strcmp(tmp,"-fd")==0){
        cout<<"frequency detection process---"<<endl;
        stereovideoDecoding(videoPath);
    }
    if (strcmp(tmp,"-se")==0){
        cout<<"spatial watermarking process---"<<endl;
        double m_NoiseStdDev=1;
        Mat noise = cv::Mat::zeros(480, 640 , CV_8UC3);
        randn(noise,0,m_NoiseStdDev);
        noise *= 1; //watermark power
        folder = argv[3];
        spatialMarking(videoPath,noise,folder);
    }
    if (strcmp(tmp,"-sd")==0){
        cout<<"spatial detection process---"<<endl;
        double m_NoiseStdDev=1;
        Mat noise = cv::Mat::zeros(480, 640 , CV_8UC3);
        randn(noise,0,m_NoiseStdDev);
        noise *= 1; //watermark power
        spatialDecoding(videoPath, noise);
    }
    if (strcmp(tmp,"-qm")==0){
        cout<<"quality metrics computation---"<<endl;
        std::string video = "./video/video.mp4";
        std::string wat_video = "./video/marked_gauss_3_crf1.mp4";
        std::string disp_video_l = "./video/video_disp_l.mp4";
        std::string disp_video_r = "./video/video_disp_r.mp4";
        std::string wat_disp_video_l = "./video/disp_lr_g3.mp4";
        std::string wat_disp_video_r = "./video/disp_rl_g3.mp4";
        RRQualityMetrics::compute_metrics(STEP, video, wat_video, disp_video_l, wat_disp_video_l,disp_video_r, wat_disp_video_r );
    }
    if (strcmp(tmp,"-d")==0){
        cout<<"disparity computation---"<<endl;
        folder = argv[3];
        disparity_computation(videoPath,folder);
    }
    if (strcmp(tmp,"-fm")==0){
        cout<<"creating stereo frames---"<<endl;
        string videoPath2 = argv[3];
        folder = argv[4];
        make_stereo_frames(videoPath,videoPath2,folder);

    }
//    if(strcmp(tmp,"-p") == 0){
//        cout<<"PSNR computation---"<<endl;
//        qm::avg_psnr(videoPath1, videopath2);
//    }



//    string videoPath = "/home/miky/Scrivania/new_dataset/basket.mp4";
//    stereovideoCoding(videoPath);



//
//    if(argc == 1)
//    {
//        showhelpinfo(argv[0]);
//        exit(1);
//    }
//
//    const char* tmp = argv[1];
//
//    if (strcmp(tmp,"-h")==0){showhelpinfo(argv[0]);}
//    if (strcmp(tmp,"-fe")==0){
//        cout<<"frequency watermarking process---"<<endl;
//        string videoPath = argv[2];
//        stereovideoCoding(videoPath);
//    }
//    if (strcmp(tmp,"-fd")==0){
//        cout<<"frequency detection process---"<<endl;
//        string videoPath = argv[2];
//        stereovideoDecoding(videoPath);
//    }
//    if (strcmp(tmp,"-se")==0){
//        cout<<"spatial watermarking process---"<<endl;
//        string videoPath = argv[2];
//        double m_NoiseStdDev=1;
//        Mat noise = cv::Mat::zeros(480, 640 , CV_8UC3);
//        randn(noise,0,m_NoiseStdDev);
//        noise *= 1; //watermark power
//        spatialMarking(videoPath,noise);
//    }
//    if (strcmp(tmp,"-sd")==0){
//        cout<<"spatial detection process---"<<endl;
//        string videoPath = argv[2];
//        double m_NoiseStdDev=1;
//        Mat noise = cv::Mat::zeros(480, 640 , CV_8UC3);
//        randn(noise,0,m_NoiseStdDev);
//        noise *= 1; //watermark power
//        spatialDecoding(videoPath, noise);
//    }
//    if (strcmp(tmp,"-qm")==0){
//        cout<<"quality metrics computation---"<<endl;
//        std::string video = "./video/video.mp4";
//        std::string wat_video = "./video/marked_gauss_3_crf1.mp4";
//        std::string disp_video_l = "./video/video_disp_l.mp4";
//        std::string disp_video_r = "./video/video_disp_r.mp4";
//        std::string wat_disp_video_l = "./video/disp_lr_g3.mp4";
//        std::string wat_disp_video_r = "./video/disp_rl_g3.mp4";
//        RRQualityMetrics::compute_metrics(STEP, video, wat_video, disp_video_l, wat_disp_video_l,disp_video_r, wat_disp_video_r );
//    }
//    if (strcmp(tmp,"-d")==0){
//        cout<<"disparity computation---"<<endl;
//        string videoPath = argv[2];
//        disparity_computation(videoPath);
//    }
//    if(strcmp(tmp,"-p") == 0){
//        cout<<"PSNR computation---"<<endl;
//        string videoPath1 = argv[2];
//        string videoPath2 = argv[3];
//        qm::avg_psnr(videoPath1, videoPath2);
//    }


}

