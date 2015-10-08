//
// Created by miky on 02/10/15.
//

#ifndef TESI_WATERMARKING_FREQUENCYWATERMARKING_H
#define TESI_WATERMARKING_FREQUENCYWATERMARKING_H

#endif //TESI_WATERMARKING_FREQUENCYWATERMARKING_H
#include <iostream>

namespace FDTStereoWatermarking{

    void warpMarkWatermarking(int* watermark, int wsize, float power, std::string passwstr, std::string passwnum);

    void warpRightWatermarking(int wsize, int tilesize, float power, bool clipping,
                                                      bool flagResyncAll, int tilelistsize, std::string passwstr,
                                                      std::string passwnum);
}