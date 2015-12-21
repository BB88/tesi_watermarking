#ifndef RECTIFY_H
#define RECTIFY_H

#include "/home/bene/ClionProjects/rectify-quasi-euclidean_20140626/src/libNumerics/homography.h"
#include "/home/bene/ClionProjects/rectify-quasi-euclidean_20140626/src/libMatch/imgmatch.h"
#include <vector>

std::pair<float,float> compRectif(int w, int h, const std::vector<imgMatch>& m,
                                  libNumerics::Homography& Hl,
                                  libNumerics::Homography& Hr);

#endif
