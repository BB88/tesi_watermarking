//
// Created by bene on 20/07/15.
//

#ifndef TESI_WATERMARKING_ALLOCIM_H
#define TESI_WATERMARKING_ALLOCIM_H

#endif //TESI_WATERMARKING_ALLOCIM_H


/*!
 *
 * Copyright (C) 2005
 * Communication and Image Laboratory,
 * Dept. of Electronics and Telecommunications
 * Florence University
 *
 *
 * This file is part of <em>LCI Tools</em>, a collection of
 * image processing tools for virtual restoration of artworks and
 * image watermarking.
 *
 * \file
 *
 *
 * \brief Image Allocation.
 *
 */



namespace AllocIm
{
    unsigned char **AllocImByte(int nr, int nc);
    int **AllocImInt(int nr, int nc);
    float **AllocImFloat(int nr, int nc);
    double **AllocImDouble(int nr, int nc);

    void FreeIm(unsigned char **punt);
    void FreeIm(int **punt);
    void FreeIm(float **punt);
    void FreeIm(double **punt);

} // namespace AllocIm

