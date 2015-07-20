//
// Created by miky on 20/07/15.
//

#ifndef TESI_WATERMARKING_BCH_H
#define TESI_WATERMARKING_BCH_H

#endif //TESI_WATERMARKING_BCH_H
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
 * \brief BCH code routines.
 *
 */


namespace BCH
{
    /**
     * BCH Encoding.
     *
     * \note For further details see the comments inside the code.
     */
    void encode_bch(int m, int length, int t, int *data, int *recd);

    /**
     * BCH Decoding.
     *
     * \note For further details see the comments inside the code.
     */
    bool decode_bch(int m, int length, int t, int *recd);

} // namespace BCH



