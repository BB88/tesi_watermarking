//
// Created by bene on 19/10/15.
//

#ifndef PROJECT_IMGMATCH_H
#define PROJECT_IMGMATCH_H

#include <vector>

struct imgMatch {
    imgMatch() {}
    imgMatch(float ix1, float iy1, float ix2, float iy2)
            : x1(ix1), y1(iy1), x2(ix2), y2(iy2) {}
    float x1, y1, x2, y2;
};

bool loadMatch(const char* nameFile, std::vector<imgMatch>& match);
bool saveMatch(const char* nameFile, const std::vector<imgMatch>& match);


#endif //PROJECT_IMGMATCH_H
