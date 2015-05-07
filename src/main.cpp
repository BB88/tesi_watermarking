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
#include "./disparity_computation/sift_computation.h"
#include "./disparity_optimization/occlusions_handler.h"


using namespace std;
using namespace cv;
using namespace cv::datasets;

int main() {

 /*   string path("/home/bene/ClionProjects/tesi_watermarking/dataset/NTSD-200/");

    Ptr<tsukuba_dataset> dataset = tsukuba_dataset::create();
    dataset->load(path);

    unsigned int dataset_size = (unsigned int)dataset->getTrain().size();

    cout << dataset_size;
    Ptr<cv::datasets::tsukuba_datasetObj> data_stereo_img =
            static_cast< Ptr<cv::datasets::tsukuba_datasetObj> > (dataset->getTrain()[0]);
    cv::Mat img_left, img_right, disp;
    cv::datasets::FramePair tuple_img = dataset->load_stereo_images(1);
    img_left = tuple_img.frame_left;
    img_right = tuple_img.frame_right;

    stereomatching::stereo_matching(img_left, img_right, disp);
    stereomatching::display(img_left, img_right, disp);*/

//    sift_computation::sift_compute();

    occlusions_handler::occlusions_filler();

// cv::initModule_nonfree();
/* Mat image = imread("/home/bene/ClionProjects/tesi_watermarking/img/disp2.png");
 if (image.cols == 0){
  cout << "Empty image";
 } else {
     Mat output;
     cv::medianBlur(image, output, 7);
     imwrite("/home/bene/ClionProjects/tesi_watermarking/img/filtered_disp2.png", output);
     imshow("Output", output);
     waitKey();
 }*/
     return 0;


}






//   shortcuts:   https://www.jetbrains.com/clion/documentation/docs/CLion_ReferenceCard.pdf