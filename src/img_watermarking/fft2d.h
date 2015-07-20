//
// Created by miky on 20/07/15.
//

#ifndef TESI_WATERMARKING_FFT2D_H
#define TESI_WATERMARKING_FFT2D_H

#endif //TESI_WATERMARKING_FFT2D_H

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
 * \brief Bi-Dimensional Fast Fourier Transform.
 *
 */


namespace FFT2D
{
    /**
     * Bi-dimensional Fast Fourier Transform.
     *
     * \note For further details see the comments inside the code.
     */
    void fft(double *xr, double *xi, int num, int ind);

    /**
     * Bi-dimensional Inverse Discrete Fourier Transform.
     *
     * \note For further details see the comments inside the code.
     */
    void idft2d(double **imdft, double **imdftfase, float **imidft, int dimx, int dimy);

    /**
     * Bi-dimensional Discrete Fourier Transform.
     *
     * \note For further details see the comments inside the code.
     */
    void dft2d(float **imin, double **imout, double **imfase, int dimx, int dimy);

    // For further details see the comments inside the code.
    void conv(double **a, double **b, int _dimx, int _dimy, int ind);

} // namespace BCH

#endif  /* FFT2D_H */
