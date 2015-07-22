//
// Created by miky on 22/07/15.
//
#include <stdio.h>
#include <string.h>
#include <iostream>


// Local headers
#include "imgwat.h"
#include "bch.h"
#include "fft2d.h"
#include "allocim.h"

// Standard headers
#include <assert.h>

#include "watermarking.h"

///DA METTERE NEL CONFIG
// definitions
const double Watermarking::M0 = 0.7071067811865;
const double Watermarking::EPS = 0.001;
const double Watermarking::EPS2 = 0.035;
const int Watermarking::DIMFILT2 = 3;
const int Watermarking::DIMFILT = 7;

const int Watermarking::NIT = 50;  // numero iterazioni per la stima dei parametri alfa
//   e beta della weibull su ogni zona

const int Watermarking::NUM_PARAMS = 4;  // param. di marchiatura (prima diag., numero diag., potenza di marchiatura)
const int Watermarking::MAXZONE = 16;  // numero di zone in cui viene suddivisa la parte
// dello spettro sottoposta a marchiatura per farne
// l'analisi statistica.
const int Watermarking::WINDOW = 9;

using namespace AllocIm;
using namespace std;

void Watermarking::setParameters(int *w, int wsize, int tsize, float pwr, bool useClipping,
                           bool resynchronization, int *tilelist, int tilelistsize)
{
    // list of wsize (watermark)
    memcpy(watermark, w, sizeof(int) * wsize);

    wsize = wsize;
    tilesize = tsize;
    power = pwr;
    clipping = useClipping;
    flagResyncAll = resynchronization;

    assert(tilelistsize <= 33);
    memcpy(tiles, tilelist, sizeof(int) * tilelistsize);
}

void Watermarking::setPassword(std::string passwStr, std::string passwNum)
{
    passwstr = passwStr;
    passwnum = passwNum;
}


unsigned char * Watermarking::insertWatermark(unsigned char *image, int w, int h)
{
    bool flagOk;

    unsigned char *output_img = new unsigned char[w * h * 3];
    memcpy(output_img, image, w * h * 3);

    const char *passw_str = passwstr.c_str();
    const char *passw_num = passwnum.c_str();

    int result = WatCod(output_img, w, h, passw_str, passw_num, watermark, wsize, power, clipping, tilesize, tiles, &ntiles);

    if (result == -3)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Invalid number of bits!\nValid watermarks have 32 or 64 bits.");
//        // invalid 'nbit'
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid number of bits!\nValid watermarks have 32 or 64 bits."));
        flagOk = false;
    }
    else if (result == -2)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Invalid size of the tile. Valid size are 256, 512 or 1024.");
//        // invalid 'size'
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid size of the tile. Valid size are 256, 512 or 1024."));
        flagOk = false;
    }
    else if (result == -1)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - The watermark power is out-of-range.");
//        // the watermark power is out-of-range
//        QMessageBox::warning(NULL, tr(MSG_TITLE), tr("The watermark power is out-of-range."));
        flagOk = false;
    }
    else if (result == 0)
    {
        // OK!
        flagOk = true;
    }
    else if (result == 1)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Image too big to be processed!.");
//        // image too big!
//        QMessageBox::warning(NULL, tr(MSG_TITLE), tr("Image too big to be processed!"));
        flagOk = false;
    }
    else if (result == 2)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Invalid tile!!.");
//        // Invalid tile
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid tile!!"));
        flagOk = false;
    }

    if (flagOk)
        return output_img;
    else
    {
        delete [] output_img;
        return NULL;
    }
}

int Watermarking::WatCod(unsigned char *ImageOut, int width, int height, const char *passw_str, const char *passw_num,
                         int *watermark, int wsize, float power, bool flagClipping, int tilesize, int *tiles,
                         int *ntiles)
{

    /*
     * FFT2D
     * zone alias coefficient selection
     * generate_mark();
     * watermark embedding : addmark + antizone
     * IFFT2D
     *
     *
     *
     *
     */


}




void Watermarking::generate_mark(int *watermark,int wsize)
{
/*
 * BCH coding
 */
    int bch_wm[200]; //bch coding of the watermark

    if (wsize == 64)
    {
        m_BCH = 7;			// order of the Galois Field GF(2^m)
        t_BCH = 10;			// Error correcting capability
        length_BCH = 127;	// length of the BCH code
    }

    if (wsize == 32)
    {
        m_BCH = 6;			// order of the Galois Field GF(2^m)
        t_BCH = 5;			// Error correcting capability
        length_BCH = 59;	// length of the BCH code
    }

    if ((wsize != 64)&&(wsize != 32))
    {
        return ;
//        return -3;	// Incorrect 'nbit'
    }
//
//    // LOG
//    fprintf(flog, " - BCH: m=%d t=%d length=%d\n\n",m_BCH,t_BCH,length_BCH);
//
//    fprintf(flog, " - Marchio: ");
//
//    for (int ii=0; ii < wsize; ii++)
//        fprintf(flog,"%d",watermark[ii]);

    BCH::encode_bch(m_BCH,length_BCH,t_BCH,watermark,bch_wm);

//    // LOG
//    for (int ii=0; ii < length_BCH; ii++)
//        fprintf(flog, "%d",bit[ii]);
//
//    fprintf(flog, "\n\n");

}


