//
// Created by miky on 20/07/15.
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
 * \brief BCH code routines (implementation).
 *
 */

// Local headers
#include "bch.h"

// Standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

/***************************************************************/
//
//	Title: Encoder/Decoder for binary BCH codes in C (Version 3.1)
//	Author: Morelos-Zaragoza,	Copyright (c) 1994-1997
//	Date: August 1994
//	Revised: June 13, 1997
//
//  Modified by Corsini Massimiliano, 5-10 April 2001
//


void BCH::encode_bch(int m, int length, int t, int *data, int *recd)
{
    int n;				// n = 2^m - 1 --> size of the multiplicative group of GF(2^m)
    int k;				// n - deg(g(x)) = dimension (no. of information bits/codeword) of the code
    int d;				// d = 2*t + 1 = designed mi. distance = no. of consecutive roots of g(x)+1
    int p[21];			// coefficients of a primitive polynomial used to generate GF(2^m)
    int alpha_to[1024]; // log table of GF(2^m)
    int index_of[1024]; // antilog table of GF(2^m)
    int g[512];			// coefficients of the generator polynomial, g(x)

    /*
        STEP 1:	Get precomputed coefficients p[] of p(x)
        ------------------------------------------------

        'm' must be between 2 and 20
    */

    int i;

    for (i=1; i<m; i++) p[i] = 0;

    p[0] = p[m] = 1;
    if (m == 2)			p[1] = 1;
    else if (m == 3)	p[1] = 1;
    else if (m == 4)	p[1] = 1;
    else if (m == 5)	p[2] = 1;
    else if (m == 6)	p[1] = 1;
    else if (m == 7)	p[1] = 1;
    else if (m == 8)	p[4] = p[5] = p[6] = 1;
    else if (m == 9)	p[4] = 1;
    else if (m == 10)	p[3] = 1;
    else if (m == 11)	p[2] = 1;
    else if (m == 12)	p[3] = p[4] = p[7] = 1;
    else if (m == 13)	p[1] = p[3] = p[4] = 1;
    else if (m == 14)	p[1] = p[11] = p[12] = 1;
    else if (m == 15)	p[1] = 1;
    else if (m == 16)	p[2] = p[3] = p[5] = 1;
    else if (m == 17)	p[3] = 1;
    else if (m == 18)	p[7] = 1;
    else if (m == 19)	p[1] = p[5] = p[6] = 1;
    else if (m == 20)	p[3] = 1;

    // Computation of n

    n = 1;
    for (i=0; i<=m; i++) n *= 2;
    n = n/2 - 1;

    /*

        STEP 2:	Generate field GF(2**m) from the irreducible
                polynomial p(X) with coefficients in p[0]..p[m].
        --------------------------------------------------------

        Lookup tables:
            index->polynomial form: alpha_to[] contains j=alpha^i;
            polynomial form -> index form:	index_of[j=alpha^i] = i

        alpha=2 is the primitive element of GF(2**m)

    */

    int mask;

    mask = 1;
    alpha_to[m] = 0;
    for (i = 0; i < m; i++)
    {
        alpha_to[i] = mask;
        index_of[alpha_to[i]] = i;
        if (p[i] != 0)
            alpha_to[m] ^= mask;
        mask <<= 1;
    }

    index_of[alpha_to[m]] = m;
    mask >>= 1;

    for (i = m + 1; i < n; i++)
    {
        if (alpha_to[i - 1] >= mask)
            alpha_to[i] = alpha_to[m] ^ ((alpha_to[i - 1] ^ mask) << 1);
        else
            alpha_to[i] = alpha_to[i - 1] << 1;

        index_of[alpha_to[i]] = i;
    }

    index_of[0] = -1;

    /*
        STEP 3: Compute the generator polynomial of a
                binary BCH code.
        ----------------------------------------------

        Fist generate the cycle sets modulo (2^m-1),
        cycle[][] =  (i, 2*i, 4*i, ..., 2^l*i).
        Then determine those cycle sets that contain integers in the
        set of (d-1) consecutive integers {1..(d-1)}.
        The generator polynomial is calculated as the product of
        linear factors of the form (x+alpha^i), for every i in
        the above cycle sets.
    */

    int	ii, jj, ll, kaux;
    int	test, aux, nocycles, root, noterms, rdncy;
    int cycle[1024][21], size[1024], min[1024], zeros[1024];

    // Generate cycle sets modulo n, n = (2^m-1)

    cycle[0][0] = 0;
    size[0] = 1;
    cycle[1][0] = 1;
    size[1] = 1;
    jj = 1;				// cycle set index

    do
    {
        // Generate the jj-th cycle set

        ii = 0;
        do
        {
            ii++;
            cycle[jj][ii] = (cycle[jj][ii - 1] * 2) % n;
            size[jj]++;
            aux = (cycle[jj][ii] * 2) % n;
        } while (aux != cycle[jj][0]);

        // Next cycle set representative

        ll = 0;
        do
        {
            ll++;
            test = 0;
            for (ii = 1; ((ii <= jj) && (!test)); ii++)
                // Examine previous cycle sets
                for (kaux = 0; ((kaux < size[ii]) && (!test)); kaux++)
                    if (ll == cycle[ii][kaux])
                        test = 1;
        } while ((test) && (ll < (n - 1)));

        if (!(test))
        {
            jj++;	// next cycle set index
            cycle[jj][0] = ll;
            size[jj] = 1;
        }

    } while (ll < (n - 1));
    nocycles = jj;		// number of cycle sets modulo n

    d = 2 * t + 1;

    // Search for roots 1, 2, ..., d-1 in cycle sets
    kaux = 0;
    rdncy = 0;
    for (ii = 1; ii <= nocycles; ii++)
    {
        min[kaux] = 0;
        test = 0;

        for (jj = 0; ((jj < size[ii]) && (!test)); jj++)
            for (root = 1; ((root < d) && (!test)); root++)
                if (root == cycle[ii][jj])
                {
                    test = 1;
                    min[kaux] = ii;
                }

        if (min[kaux])
        {
            rdncy += size[min[kaux]];
            kaux++;
        }
    }
    noterms = kaux;
    kaux = 1;
    for (ii = 0; ii < noterms; ii++)
        for (jj = 0; jj < size[min[ii]]; jj++)
        {
            zeros[kaux] = cycle[min[ii]][jj];
            kaux++;
        }

    k = length - rdncy;

    if (k<0)
    {
        // Questo si verifica quanto 't' � troppo alto, ossia
        // i bit sono insufficienti per realizzare la correzione
        // richiesta

        exit(-2); // Exit with error code 2.
    }

    // Compute the generator polynomial
    g[0] = alpha_to[zeros[1]];
    g[1] = 1;	// g(x) = (X + zeros[1]) initially
    for (ii = 2; ii <= rdncy; ii++)
    {
        g[ii] = 1;

        for (jj = ii - 1; jj > 0; jj--)
            if (g[jj] != 0)
                g[jj] = g[jj - 1] ^ alpha_to[(index_of[g[jj]] + zeros[ii]) % n];
            else
                g[jj] = g[jj - 1];

        g[0] = alpha_to[(index_of[g[0]] + zeros[ii]) % n];
    }


    /*
        STEP 4 : Start encoding...
        --------------------------

        Compute redundacy bb[], the coefficients of b(x). The redundancy
        polynomial b(x) is the remainder after dividing x^(length-k)*data(x)
        by the generator polynomial g(x).
    */

    int j;
    int feedback;
    int bb[512];		// coefficients of redundancy polynomial
    //   x^(length-k) i(x) modulo g(x)

    for (i = 0; i < length - k; i++) bb[i] = 0;

    for (i = k - 1; i >= 0; i--)
    {
        feedback = data[i] ^ bb[length - k - 1];

        if (feedback != 0)
        {
            for (j = length - k - 1; j > 0; j--)
                if (g[j] != 0)
                    bb[j] = bb[j - 1] ^ feedback;
                else
                    bb[j] = bb[j - 1];

            bb[0] = g[0] && feedback;
        }
        else
        {
            for (j = length - k - 1; j > 0; j--)
                bb[j] = bb[j - 1];

            bb[0] = 0;
        }
    }

    /*
     * recd[] are the coefficients of c(x) = x**(length-k)*data(x) + b(x)
     */

    for (i = 0; i < length - k; i++)
        recd[i] = bb[i];

    for (i = 0; i < k; i++)
        recd[i + length - k] = data[i];
}

/*
	decode_bch(..)
	--------------

	Simon Rockliff's implementation of Berlekamp's algorithm.

	Assume we have received bits in recd[i], i=0..(n-1).

	Compute the 2*t syndromes by substituting alpha^i into rec(X) and
	evaluating, storing the syndromes in s[i], i=1..2t (leave s[0] zero) .
	Then we use the Berlekamp algorithm to find the error location polynomial
	elp[i].

	If the degree of the elp is >t, then we cannot correct all the errors, and
	we have detected an uncorrectable error pattern. We output the information
	bits uncorrected.

	If the degree of elp is <=t, we substitute alpha^i , i=1..n into the elp
	to get the roots, hence the inverse roots, the error location numbers.
	This step is usually called "Chien's search".

	If the number of errors located is not equal the degree of the elp, then
	the decoder assumes that there are more than t errors and cannot correct
	them, only detect them. We output the information bits uncorrected.

*/

bool BCH::decode_bch(int m, int length, int t, int *recd)
{
    int	i, j, u, q, t2, count = 0, syn_error = 0;
    int	elp[126][124], vettd[1026], l[1026], u_lu[1026], s[1025];
    int	roots[200], loc[200], reg[201];

    int n;				// n = 2^m - 1 --> size of the multiplicative group of GF(2^m)
    int k;				// n - deg(g(x)) = dimension (no. of information bits/codeword) of the code
    int d;				// d = 2*t + 1 = designed mi. distance = no. of consecutive roots of g(x)+1
    int p[21];			// coefficients of a primitive polynomial used to generate GF(2^m)
    int alpha_to[1024]; // log table of GF(2^m)
    int index_of[1024]; // antilog table of GF(2^m)
    int g[512];			// coefficients of the generator polynomial, g(x)

    bool flagDecode;

    /*
        STEP 1:	Get precomputed coefficients p[] of p(x)
        ------------------------------------------------

        'm' must be between 2 and 20
    */

    for (i=1; i<m; i++) p[i] = 0;

    p[0] = p[m] = 1;
    if (m == 2)			p[1] = 1;
    else if (m == 3)	p[1] = 1;
    else if (m == 4)	p[1] = 1;
    else if (m == 5)	p[2] = 1;
    else if (m == 6)	p[1] = 1;
    else if (m == 7)	p[1] = 1;
    else if (m == 8)	p[4] = p[5] = p[6] = 1;
    else if (m == 9)	p[4] = 1;
    else if (m == 10)	p[3] = 1;
    else if (m == 11)	p[2] = 1;
    else if (m == 12)	p[3] = p[4] = p[7] = 1;
    else if (m == 13)	p[1] = p[3] = p[4] = 1;
    else if (m == 14)	p[1] = p[11] = p[12] = 1;
    else if (m == 15)	p[1] = 1;
    else if (m == 16)	p[2] = p[3] = p[5] = 1;
    else if (m == 17)	p[3] = 1;
    else if (m == 18)	p[7] = 1;
    else if (m == 19)	p[1] = p[5] = p[6] = 1;
    else if (m == 20)	p[3] = 1;

    // Computation of n

    n = 1;
    for (i=0; i<=m; i++) n *= 2;
    n = n/2 - 1;

    /*

        STEP 2:	Generate field GF(2**m) from the irreducible
                polynomial p(X) with coefficients in p[0]..p[m].
        --------------------------------------------------------

        Lookup tables:
            index->polynomial form: alpha_to[] contains j=alpha^i;
            polynomial form -> index form:	index_of[j=alpha^i] = i

        alpha=2 is the primitive element of GF(2**m)

    */

    int mask;

    mask = 1;
    alpha_to[m] = 0;
    for (i = 0; i < m; i++)
    {
        alpha_to[i] = mask;
        index_of[alpha_to[i]] = i;
        if (p[i] != 0)
            alpha_to[m] ^= mask;
        mask <<= 1;
    }

    index_of[alpha_to[m]] = m;
    mask >>= 1;

    for (i = m + 1; i < n; i++)
    {
        if (alpha_to[i - 1] >= mask)
            alpha_to[i] = alpha_to[m] ^ ((alpha_to[i - 1] ^ mask) << 1);
        else
            alpha_to[i] = alpha_to[i - 1] << 1;

        index_of[alpha_to[i]] = i;
    }

    index_of[0] = -1;

    /*
        STEP 3: Compute the generator polynomial of a
                binary BCH code.
        ----------------------------------------------

        Fist generate the cycle sets modulo (2^m-1),
        cycle[][] =  (i, 2*i, 4*i, ..., 2^l*i).
        Then determine those cycle sets that contain integers in the
        set of (d-1) consecutive integers {1..(d-1)}.
        The generator polynomial is calculated as the product of
        linear factors of the form (x+alpha^i), for every i in
        the above cycle sets.
    */

    int	ii, jj, ll, kaux;
    int	test, aux, nocycles, root, noterms, rdncy;
    int cycle[1024][21], size[1024], min[1024], zeros[1024];

    // Generate cycle sets modulo n, n = (2^m-1)

    cycle[0][0] = 0;
    size[0] = 1;
    cycle[1][0] = 1;
    size[1] = 1;
    jj = 1;				// cycle set index

    do
    {
        // Generate the jj-th cycle set

        ii = 0;
        do
        {
            ii++;
            cycle[jj][ii] = (cycle[jj][ii - 1] * 2) % n;
            size[jj]++;
            aux = (cycle[jj][ii] * 2) % n;
        } while (aux != cycle[jj][0]);

        // Next cycle set representative

        ll = 0;
        do
        {
            ll++;
            test = 0;
            for (ii = 1; ((ii <= jj) && (!test)); ii++)
                // Examine previous cycle sets
                for (kaux = 0; ((kaux < size[ii]) && (!test)); kaux++)
                    if (ll == cycle[ii][kaux])
                        test = 1;
        } while ((test) && (ll < (n - 1)));

        if (!(test))
        {
            jj++;	// next cycle set index
            cycle[jj][0] = ll;
            size[jj] = 1;
        }

    } while (ll < (n - 1));
    nocycles = jj;		// number of cycle sets modulo n

    d = 2 * t + 1;

    // Search for roots 1, 2, ..., d-1 in cycle sets
    kaux = 0;
    rdncy = 0;
    for (ii = 1; ii <= nocycles; ii++)
    {
        min[kaux] = 0;
        test = 0;

        for (jj = 0; ((jj < size[ii]) && (!test)); jj++)
            for (root = 1; ((root < d) && (!test)); root++)
                if (root == cycle[ii][jj])
                {
                    test = 1;
                    min[kaux] = ii;
                }

        if (min[kaux])
        {
            rdncy += size[min[kaux]];
            kaux++;
        }
    }
    noterms = kaux;
    kaux = 1;
    for (ii = 0; ii < noterms; ii++)
        for (jj = 0; jj < size[min[ii]]; jj++)
        {
            zeros[kaux] = cycle[min[ii]][jj];
            kaux++;
        }

    k = length - rdncy;

    if (k<0)
    {
        // Questo si verifica quanto 't' � troppo alto, ossia
        // i bit sono insufficienti per realizzare la correzione
        // richiesta

        printf("[BCH ERROR] Parameters invalid! [BCH ERROR]\n");
        exit(0);
    }

    // Compute the generator polynomial
    g[0] = alpha_to[zeros[1]];
    g[1] = 1;	// g(x) = (X + zeros[1]) initially
    for (ii = 2; ii <= rdncy; ii++)
    {
        g[ii] = 1;

        for (jj = ii - 1; jj > 0; jj--)
            if (g[jj] != 0)
                g[jj] = g[jj - 1] ^ alpha_to[(index_of[g[jj]] + zeros[ii]) % n];
            else
                g[jj] = g[jj - 1];

        g[0] = alpha_to[(index_of[g[0]] + zeros[ii]) % n];
    }

    /*
        STEP 4: Decoding starts...
        --------------------------
    */

    t2 = 2 * t;

    // first form the syndromes
    for (i = 1; i <= t2; i++)
    {
        s[i] = 0;
        for (j = 0; j < length; j++)
            if (recd[j] != 0)
                s[i] ^= alpha_to[(i * j) % n];
        if (s[i] != 0)
            syn_error = 1; // set error flag if non-zero syndrome

        // convert syndrome from polynomial form to index form
        s[i] = index_of[s[i]];
    }

    flagDecode = true;

    if (syn_error)
    {	/* if there are errors, try to correct them

		   Compute the error location polynomial via the Berlekamp
		   iterative algorithm. Following the terminology of Lin and
		   Costello's book :   vettd[u] is the 'mu'th discrepancy, where
		   u='mu'+1 and 'mu' (the Greek letter!) is the step number
		   ranging from -1 to 2*t (see L&C),  l[u] is the degree of
		   the elp at that step, and u_l[u] is the difference between
		   the step number and the degree of the elp.
		 */

        // initialise table entries
        vettd[0] = 0;		// index form
        vettd[1] = s[1];	// index form
        elp[0][0] = 0;		// index form
        elp[1][0] = 1;		// polynomial form

        for (i = 1; i < t2; i++)
        {
            elp[0][i] = -1;	// index form
            elp[1][i] = 0;	// polynomial form
        }

        l[0] = 0;
        l[1] = 0;
        u_lu[0] = -1;
        u_lu[1] = 0;
        u = 0;

        do
        {
            u++;
            if (vettd[u] == -1)
            {
                l[u + 1] = l[u];
                for (i = 0; i <= l[u]; i++)
                {
                    elp[u + 1][i] = elp[u][i];
                    elp[u][i] = index_of[elp[u][i]];
                }

            }
            else // search for words with greatest u_lu[q]
            {	 //   for which vettd[q]!=0
                q = u - 1;
                while ((vettd[q] == -1) && (q > 0)) q--;

                // have found first non-zero vettd[q]
                if (q > 0)
                {
                    j = q;
                    do
                    {
                        j--;
                        if ((vettd[j] != -1) && (u_lu[q] < u_lu[j]))
                            q = j;
                    } while (j > 0);
                }

                /*
                 * have now found q such that vettd[u]!=0 and
                 * u_lu[q] is maximum
                 */

                // store degree of new elp polynomial
                if (l[u] > l[q] + u - q)
                    l[u + 1] = l[u];
                else
                    l[u + 1] = l[q] + u - q;

                // form new elp(x)

                for (i = 0; i < t2; i++)
                    elp[u + 1][i] = 0;

                for (i = 0; i <= l[q]; i++)
                    if (elp[q][i] != -1)
                        elp[u + 1][i + u - q] =
                                alpha_to[(vettd[u] + n - vettd[q] + elp[q][i]) % n];

                for (i = 0; i <= l[u]; i++)
                {
                    elp[u + 1][i] ^= elp[u][i];
                    elp[u][i] = index_of[elp[u][i]];
                }
            }
            u_lu[u + 1] = u - l[u + 1];

            // form (u+1)th discrepancy
            if (u < t2)
            {
                // no discrepancy computed on last iteration
                if (s[u + 1] != -1)
                    vettd[u + 1] = alpha_to[s[u + 1]];
                else
                    vettd[u + 1] = 0;
                for (i = 1; i <= l[u + 1]; i++)
                    if ((s[u + 1 - i] != -1) && (elp[u + 1][i] != 0))
                        vettd[u + 1] ^= alpha_to[(s[u + 1 - i]
                                                  + index_of[elp[u + 1][i]]) % n];

                // put vettd[u+1] into index form
                vettd[u + 1] = index_of[vettd[u + 1]];
            }
        } while ((u < t2) && (l[u + 1] <= t));

        u++;
        if (l[u] <= t)
        {
            /* Can correct errors */

            // put elp into index form

            for (i = 0; i <= l[u]; i++)
                elp[u][i] = index_of[elp[u][i]];

            // Chien search: find roots of the error location polynomial
            for (i = 1; i <= l[u]; i++)
                reg[i] = elp[u][i];
            count = 0;
            for (i = 1; i <= n; i++)
            {
                q = 1;
                for (j = 1; j <= l[u]; j++)
                    if (reg[j] != -1) {
                        reg[j] = (reg[j] + j) % n;
                        q ^= alpha_to[reg[j]];
                    }
                if (!q)
                {	/* store root and error
					 * location number indices */
                    roots[count] = i;
                    loc[count] = n - i;
                    count++;
                }
            }

            if (count == l[u])
                // no. roots = degree of elp hence <= t errors
                for (i = 0; i < l[u]; i++)
                    recd[loc[i]] ^= 1;
            else	// elp has degree >t hence cannot solve
                flagDecode = false;
        }
        else
            flagDecode = false;
    }

    return flagDecode;
}

