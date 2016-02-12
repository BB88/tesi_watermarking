///**
// * @file main.cpp
// * @brief Epipolar quasi-Euclidean rectification of images
// * @author Pascal Monasse
// *
// * Copyright (c) 2014 Pascal Monasse
// * All rights reserved.
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU Lesser General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU Lesser General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//#include "src/libIO/io_png.h"
//#include "src/siftMatch/demo_lib_sift.h"
//#include "src/libOrsa/fundamental_model.hpp"
//#include "src/libRectification/rectify.h"
//#include "src/libTransform/map_image.h"
//#include "src/libMatch/imgmatch.h"
//#include <cv.h>
//
//#include <algorithm>
//#include <iostream>
//#include <cstdlib>
//#include <cfloat>
//#include <ctime>
//
//typedef LWImage<float> FImage;
//
//static const int ITER_RANSAC=10000;
//static const int SPLINE_ORDER=5;
//
//
///// Load float image from PNG file.
//static bool load(const char* nameFile, FImage& im)
//{
//    size_t sw,sh,sc;
//    im.data = io_png_read_f32(nameFile, &sw,&sh,&sc);
//    if(! im.data) {
//        std::cerr << "Unable to load image file " << nameFile << std::endl;
//        return false;
//    }
//
//    im.w = static_cast<int>(sw);
//    im.h = static_cast<int>(sh);
//    im.comps = static_cast<int>(sc);
//    return true;
//}
//
///// Lexicographical ordering of matches. Used to remove duplicates.
//static bool operator<(const imgMatch& m1, const imgMatch& m2)
//{
//    if(m1.x1 < m2.x1) return true;
//    if(m1.x1 > m2.x1) return false;
//
//    if(m1.y1 < m2.y1) return true;
//    if(m1.y1 > m2.y1) return false;
//
//    if(m1.x2 < m2.x2) return true;
//    if(m1.x2 > m2.x2) return false;
//
//    return (m1.y2 < m2.y2);
//}
//
//static bool operator==(const imgMatch& m1, const imgMatch& m2)
//{
//    return (m1.x1==m2.x1 && m1.y1==m2.y1 &&
//            m1.x2==m2.x2 && m1.y2==m2.y2);
//}
//
///// Convert to gray
//float* gray(const FImage& im) {
//    const float DIV = 1.0f/im.comps;
//    const int sz = im.w*im.h;
//    float* data = new float[sz];
//    for(int i=sz-1; i>=0; i--) {
//        float v = im.data[i];
//        for(int j=1; j<im.comps; j++)
//            v += im.data[i+j*sz];
//        data[i] = v*DIV;
//    }
//    return data;
//}
//
///// Find SIFT point correspondences between both images.
//void sift_correspondences(const FImage& im1, const FImage& im2,
//                                 matchingslist& m) {
//    siftPar param;
//    default_sift_parameters(param);
//    param.DoubleImSize=0;
//
//    float* data;
//
//    data = im1.data;
//    if(im1.comps!=1) data = gray(im1);
//    keypointslist keyp1, keyp2;
//    compute_sift_keypoints(data, keyp1, im1.w, im1.h, param);
////    std::cout<< "sift: im1: " << keyp1.size() <<std::flush;
//    if(im1.comps!=1) delete [] data;
//
//    data = im2.data;
//    if(im2.comps!=1) data = gray(im2);
//    compute_sift_keypoints(data, keyp2, im2.w, im2.h, param);
////    std::cout<<      " im2: " << keyp2.size() <<std::flush;
//    if(im2.comps!=1) delete [] data;
//
//    matchingslist matchings;
//    compute_sift_matches(keyp1, keyp2, m, param);
//  //  std::cout<< " matches: " << m.size() <<std::endl;
//}
//
///// Compute and print min and max disparity
//void printDisparity(const std::vector<imgMatch>& match) {
//    std::vector<imgMatch>::const_iterator it=match.begin();
//    double min=DBL_MAX, max=-DBL_MAX;
//    for(; it != match.end(); ++it) {
//        double xl=it->x1;
//        double xr=it->x2;
//        xr -= xl;
//        if(xr < min)
//            min = xr;
//        if(xr > max)
//            max = xr;
//    }
//    std::cout << "Disparity: "
//              << (int)floor(min) << " " << (int)ceil(max) << std::endl;
//
//}
//
///// Rectification of two images
//int dispRange(std::string left, std::string right  ) {
//
//    char* argv[3];
//    argv[1] = (char*)left.c_str();
//    argv[2] = (char*)right.c_str();
//
//    FImage im1, im2;
//    if(! load(argv[1], im1) ||
//       ! load(argv[2], im2))
//        return 1;
//
//    // (1) SIFT
//    std::vector<imgMatch> m;
//    sift_correspondences(im1,im2,m);
//    std::sort(m.begin(), m.end());
//    std::vector<imgMatch>::iterator end = std::unique(m.begin(), m.end());
//    if(end != m.end()) {
//        m.erase(end, m.end());
//    }
//    printDisparity(m);
//    free(im1.data);
//    free(im2.data);
//
//
//    return 0;
//}
//
////int main(){
////    dispRange();
////
////}