#ifndef RECTIFY_H
#define RECTIFY_H

#include "../libNumerics/homography.h"
#include "../libMatch/imgmatch.h"
#include <vector>

std::pair<float,float> compRectif(int w, int h, const std::vector<imgMatch>& m,
                                  libNumerics::Homography& Hl,
                                  libNumerics::Homography& Hr);

#endif
