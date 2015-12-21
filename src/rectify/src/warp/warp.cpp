/**
 * @file warp.cpp
 * @brief Inverse warping of image from disparity map
 * @author Pascal Monasse
 * 
 * Copyright (c) 2014 Pascal Monasse
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "libIO/io_png.h"
#include "libIO/io_tiff.h"
#include "libIO/nan.h"
#include "libTransform/spline.h"

#include <iostream>
#include <cstdlib>

/// Spline order for interpolation.
static const int SPLINE_ORDER=5;

typedef LWImage<float> FImage;
typedef float* (*ReadImage)(const char*, size_t*, size_t*, size_t*);

/// Read TIFF image file (used as proxy of io_tiff_read_f32_gray).
static float* io_tiff_read_f32(const char* n, size_t* w, size_t* h, size_t* c) {
    *c = 1;
    return io_tiff_read_f32_gray(n, w, h);
}

/// Load float image from file.
static bool load(const char* nameFile, FImage& im, ReadImage readImage) {
    size_t sw,sh,sc;
    im.data = readImage(nameFile, &sw,&sh,&sc);
    if(! im.data) {
        std::cerr << "Unable to load image file " << nameFile << std::endl;
        return false;
    }

    im.w = static_cast<int>(sw);
    im.h = static_cast<int>(sh);
    im.comps = static_cast<int>(sc);
    return true;
}

/// Save float image into PNG file.
static bool save(const char* nameFile, FImage& im) {
    // Put in white invalid pixels
    for(int i=im.comps*im.w*im.h-1; i>=0; i--)
        if(! is_number(im.data[i]))
            im.data[i] = 255.0f;
    if(io_png_write_f32(nameFile, im.data, im.w, im.h, im.comps) != 0) {
        std::cerr << "Error writing file " << nameFile << std::endl;
        return false;
    }
    return true;
}

/// Warp the image, return number of pixels falling outside (should be 0).
static int warp(FImage disp, FImage in, FImage& out) {
    if(! prepare_spline(in, SPLINE_ORDER)) {
        std::cerr << "Unable to prepare interpolation image" << std::endl;
        std::exit(1);
    }
    int n=0;
    const int step=out.stepComp();
    float pix[3] = {0,0,0};
    const int stepIn = (in.comps==1)? 0: 1;
    for(int i=0; i<out.h; i++)
        for(int j=0; j<out.w; j++) {
            float d = *disp.pixel(j,i);
            float x = j+0.5f+d;
            float y = i+0.5f;
            if(is_number(d)) {
                if(interpolate_spline(in, SPLINE_ORDER, x, y, pix)) {
                    for(int c=0; c<3; c++)
                        out.pixel(j,i)[c*step] = pix[c*stepIn];
                } else {
                    ++n;
                    d = NaN;
                }
            }
            if(! is_number(d)) {
                out.pixel(j,i)[0*step] = 0;
                out.pixel(j,i)[1*step] = 255;
                out.pixel(j,i)[2*step] = 255;
            }
        }
    return n;
}

/// Inverse warping of an image from disparity map
int main(int argc, char* argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " disp.tif im2.png out.png" << std::endl;
        return 1;
    }

    FImage disp, in;
    if(! load(argv[1], disp, io_tiff_read_f32)) {
        std::cerr << "Expecting TIFF" << std::endl;
        return 1;
    }
    if(! load(argv[2], in, io_png_read_f32)) {
        std::cerr << "Expecting PNG" << std::endl;
        return 1;
    }

    FImage out = alloc_image<float>(disp.w, disp.h, 3);
    int n = warp(disp, in, out);
    
    if(n>0)
        std::cerr << "Warning: " << n << " pixels have invalid disparity"
                  << std::endl;

    if(! save(argv[3], out))
        return 1;

    free(disp.data);
    free(in.data);
    free(out.data);
    return 0;
}
