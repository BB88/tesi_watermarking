//
// Created by bene on 20/07/15.
//
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


// Standard headers
#include "allocim.h"


/**************************************************************

       (from) ImLib.c

    Subroutine:     AllocIm / FreeIm / OpenIm / ReadIm / WriteIm
    Class:          Public Domain
    Year:           1992
    Owner:          Franco Bartolini
    Status:         Outline
    Version:        1.0
    Date:           16/12/92
    Revision:

	NOTE: Re-coded to adapt for C++ (by Massimiliano Corsini)

Utility:
    AllocIm
          sottoprogramma che alloca una immagine di nr righe e nc colonne
          l'immagine e` gestita da un puntatore che indirizza un vettore
          che contiene gli indirizzi di tutte le righe:
          nr, nc    (interi) numero righe e colonne
          size      (unsigned int) dimensione di ogni elemento dell'immagine
            valore ritornato -
          **punt     punt[0] punta all'area di memoria che contiene tuttal'immagine
                     memorizzata per righe
                     punt[i] punta alla riga i-ma
    FreeIm
          sottoprogramma che libera lo spazio allocato ad una immagine
          punt      (void **) immagine da liberare

********************************************************************/

unsigned char **AllocIm::AllocImByte(int nr, int nc)
{
    int r;
    unsigned char **punt = new unsigned char * [nr * sizeof(unsigned char *)];

    punt[0] = new unsigned char [nr * nc * sizeof(int)];

    for(r = 1; r < nr; r++)
        punt[r] = punt[r - 1] + nc * sizeof(int);

    return punt;
}

int **AllocIm::AllocImInt(int nr, int nc)
{
    int r;
    unsigned char **punt = new unsigned char * [nr * sizeof(unsigned char *)];

    punt[0] = new unsigned char [nr * nc * sizeof(int)];

    for(r = 1; r < nr; r++)
        punt[r] = punt[r - 1] + nc * sizeof(int);

    return reinterpret_cast<int **>(punt);
}

float **AllocIm::AllocImFloat(int nr, int nc)
{
    int r;
    unsigned char **punt = new unsigned char * [nr * sizeof(unsigned char *)];

    punt[0] = new unsigned char [nr * nc * sizeof(float)];

    for(r = 1; r < nr; r++)
        punt[r] = punt[r - 1] + nc * sizeof(float);

    return reinterpret_cast<float **>(punt);
}

double **AllocIm::AllocImDouble(int nr, int nc)
{
    int r;
    unsigned char **punt = new unsigned char * [nr * sizeof(unsigned char *)];

    punt[0] = new unsigned char [nr * nc * sizeof(double)];

    for(r = 1; r < nr; r++)
        punt[r] = punt[r - 1] + nc * sizeof(double);

    return reinterpret_cast<double **>(punt);
}

void AllocIm::FreeIm(unsigned char **punt)
{
    delete [] punt[0];
    delete [] punt;
}

void AllocIm::FreeIm(int **punt)
{
    delete [] punt[0];
    delete [] punt;
}

void AllocIm::FreeIm(float **punt)
{
    delete [] punt[0];
    delete [] punt;
}

void AllocIm::FreeIm(double **punt)
{
    delete [] punt[0];
    delete [] punt;
}
