#include "rectify.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cfloat>
#include <cmath>

/// Compute and print min and max disparity
void printDisparity(const std::vector<imgMatch>& match,
                    const libNumerics::Homography& Hl,
                    const libNumerics::Homography& Hr)
{
    std::vector<imgMatch>::const_iterator it=match.begin();
    double min=DBL_MAX, max=-DBL_MAX;
    for(; it != match.end(); ++it) {
        double xl=it->x1, yl=it->y1;
        Hl(xl,yl);
        double xr=it->x2, yr=it->y2;
        Hr(xr,yr);
        xr -= xl;
        if(xr < min)
            min = xr;
        if(xr > max)
            max = xr;
    }
    std::cout << "Disparity: "
              << (int)floor(min) << " " << (int)ceil(max) << std::endl;
}

/// Usage: libRectification match.txt w h Hl Hr
/// Take as input a set of good matches @match.txt and the image dimensions,
/// @w and @h, and output the homographies to apply to left (@Hl) and right
/// (@Hr) images in the form of \f$3\times 3\f$ matrices, stored in Matlab
/// format.
int main(int argc, char** argv)
{
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " match.txt w h Hl Hr" <<std::endl;
        return 1;
    }

    std::vector<imgMatch> match;
    if(! loadMatch(argv[1],match)) {
        std::cerr << "Failed reading " << argv[1] << std::endl;
        return 1;
    }

    int w=0,h=0;
    if(! (std::istringstream(argv[2]) >> w).eof()) w=0;
    if(! (std::istringstream(argv[3]) >> h).eof()) h=0;
    if(w <=0 || h <= 0) {
        std::cerr << "Wrong dimensions of image" << std::endl;
        return 1;
    }

    libNumerics::Homography Hl, Hr;
    std::pair<float,float> e = compRectif(w, h, match, Hl, Hr);
    std::cout << "Initial rectification error: " <<e.first <<" pix" <<std::endl;
    std::cout << "Final rectification error: " << e.second <<" pix" <<std::endl;
    printDisparity(match, Hl, Hr);

    // Output files
    std::ofstream f1(argv[4]), f2(argv[5]);
    if((f1 << Hl.mat() << std::endl).fail()) {
        std::cerr << "Error writing file " << argv[4] << std::endl;
        return 1;
    }
    if((f2 << Hr.mat() << std::endl).fail()) {
        std::cerr << "Error writing file " << argv[5] << std::endl;
        return 1;
    }

    return 0;
}
