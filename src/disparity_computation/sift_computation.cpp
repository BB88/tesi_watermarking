//
// Created by bene on 15/04/15.
//

#include "sift_computation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>

using namespace std;
using namespace cv;

namespace sift_computation {
    int sift_compute() {
       /* string path("/home/bene/Scrivania/frame_1.jpg");
        const cv::Mat input = cv::imread(path); //Load as grayscale

        cv::SiftFeatureDetector detector;
        std::vector<cv::KeyPoint> keypoints;
        detector.detect(input, keypoints);

        // Add results to image and save.
        cv::Mat output;
        cv::drawKeypoints(input, keypoints, output);
        imshow("immagine con i  sift", output);
        waitKey();
        return 0;
*/


        Mat image = imread("/home/bene/Scrivania/frame_1.jpg");

        // Create smart pointer for SIFT feature detector.
        Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
        vector<KeyPoint> keypoints;

        // Detect the keypoints
        featureDetector->detect(image, keypoints); // NOTE: featureDetector is a pointer hence the '->'.

        //Similarly, we create a smart pointer to the SIFT extractor.
        Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");

        // Compute the 128 dimension SIFT descriptor at each keypoint.
        // Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
        Mat descriptors;
        featureExtractor->compute(image, keypoints, descriptors);

        // If you would like to draw the detected keypoint just to check
        Mat outputImage;
        Scalar keypointColor = Scalar(255, 0, 0);     // Blue keypoints.
        drawKeypoints(image, keypoints, outputImage, keypointColor, DrawMatchesFlags::DEFAULT);

        namedWindow("Output");
        imshow("Output", outputImage);

        waitKey(); // Keep window there until user presses 'q' to quit.

        return 0;


    }

}