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
#include "fft2d.h"
#include "allocim.h"

// Standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI     3.14159265358979323846

namespace FFT2D
{
    static const int N = 2048;
    static const int DFT = -1;
    static const int IDFT = 1;

/*
	fft(..)
	-------

	Routine che calcola la FFT (diretta o inversa) di una
	sequenza contenuta nei vettori xr[] (parte reale) e
	xi[] (parte immaginaria).
	*xr = sequenza di ingresso
	*xi = sequenza di uscita
	num = lunghezza sequenza
	ind = -1	DFT
	ind =  1	IDFT
*/

    void fft(double *xr, double *xi, int num, int ind)
    {
        double ux,u1x,tx,uy,u1y,ty,s;
        int nv2,i,ip,j,k,l,l1,l2,l3;
        double m;

        m=((double)num);
        for(i=1;i<17;++i)
        {
            m=m/2.0;
            if(m==1.0) l3=i;
        }

        nv2=num/2;

        j=1;
        for(i=1;i<num;++i)
        {
            if(i<j)
            {
                tx=xr[j-1];
                xr[j-1]=xr[i-1];
                xr[i-1]=tx;
                ty=xi[j-1];
                xi[j-1]=xi[i-1];
                xi[i-1]=ty;

                k=nv2;
                if(k<j)
                {
                    while(k<j)
                    {
                        j=j-k;
                        k=k/2;
                    }
                }
                j=j+k;
            }
            else
            {
                k=nv2;
                if(k<j)
                {
                    while(k<j)
                    {
                        j=j-k;
                        k=k/2;
                    }
                }
                j=j+k;
            }
        }
        for(l=1;l < (l3+1);++l)
        {
            l1=1;
            for(i=1;i<(l+1);++i)
                l1=l1*2;

            l2=l1/2;
            ux=1.0;
            uy=0.0;
            u1x=cos(PI/l2);
            u1y=ind*sin(PI/l2);
            for(j=1;j<(l2+1);++j)
            {
                for(i=j;i<(num+1);i+=l1)
                {
                    ip=i+l2;

                    tx=xr[ip-1]*ux-xi[ip-1]*uy;
                    ty=xi[ip-1]*ux+xr[ip-1]*uy;

                    xr[ip-1]=xr[i-1]-tx;
                    xi[ip-1]=xi[i-1]-ty;

                    xr[i-1]=xr[i-1]+tx;
                    xi[i-1]=xi[i-1]+ty;
                }
                s=ux;
                ux=ux*u1x-uy*u1y;
                uy=uy*u1x+s*u1y;
            }
        }

        if (ind==DFT)
            for(i=0;i<num;i++)
            {
                xr[i]/=num;
                xi[i]/=num;
            }
    }


/*
	idft2d(..)
	----------

    Calcola l'IDFT2d.

	flag = 1   l'ingresso e l'uscita sono immagini
	flag = -1  l'ingresso e l'uscita sono matrici di float
*/
    void idft2d(double **imdft, double **imdftfase, float **imidft, int dimx, int dimy)
    {
        double  **x,**y,R[N],I[N];
        int     i,j,dim;

        dim = dimx*dimy;

        // Allocazione di memoria matrice coefficienti DFT reali

        x = AllocIm::AllocImDouble(dimx, dimy);

        // Allocazione di memoria matrice coefficienti DFT complessi

        y = AllocIm::AllocImDouble(dimx, dimy);


        // Riempimento matrice x

        for(i=0;i<dimx;i++)
        {
            for(j=0;j<dimy;j++)
            {
                x[i][j]=(double)imdft[i][j];
            }
        }


        // Riempimento matrice y

        for(i=0;i<dimx;i++)
        {
            for(j=0;j<dimy;j++)
            {
                y[i][j]=(double)imdftfase[i][j];
            }
        }


        // Conversione coefficienti complessi modulo/fase, in coefficienti
        // complessi reali/immaginari

        conv(x,y,dimx,dimy,1);


        // Antitrasformata primo passaggio riga per riga

        for(i=0;i<dimx;i++)
        {
            fft(x[i],y[i],dimy,1) ;
        }

        // Antitrasformata secondo passaggio (colonna per colonna)

        for(j=0;j<dimy;j++)
        {
            for(i=0;i<dimx;i++)
            {
                R[i]=x[i][j];
                I[i]=y[i][j];
            }
            fft(R,I,dimx,1);
            for(i=0;i<dimx;i++)
            {
                x[i][j]=R[i];
                y[i][j]=I[i];
            }
        }


        // riempimento matrice di uscita reale

        for(i=0;i<dimx;i++)
            for(j=0;j<dimy;j++)
                imidft[i][j] = (float)x[i][j];

        // A questo punto sarebbe interessante un controllo
        // sul vettore dei complessi, e vedere se sono nulli

        /* ... da fare in futuro ... */

        /*********************** visualizzazione elementi ********/

        /* ... da fare in futuro ... */


        // Si liberano le aree di memoria
        AllocIm::FreeIm(x);
        AllocIm::FreeIm(y);
    }


/*

	dft2d(..)
	---------

	Calclo della DFT2d:

	flag = 1 -> l'ingresso e' un'immagine
	flag = -1-> l'ingresso e' un file di float

*/

    void dft2d(float **imin, double **imout, double **imfase, int dimx, int dimy)
    {
        double	**x,**y,R[N],I[N];
        int		i,j,dim;

        dim=dimx*dimy;


        // Allocazione di memoria matrice parte reale coefficienti DFT

        x = AllocIm::AllocImDouble(dimx, dimy);

        // Allocazione di memoria matrice parte immaginaria coefficienti DFT

        y = AllocIm::AllocImDouble(dimx, dimy);


        // L'immagine viene copiata su  x

        for(i=0;i<dimx;i++)
        {
            for(j=0;j<dimy;j++)
            {
                x[i][j]=(double)imin[i][j];
                y[i][j]=0.0;
            }
        }


        // Trasformata primo passaggio; eseguo la trasformata per tutte
        // le righe

        for(i=0;i<dimx;i++)
        {
            fft(x[i],y[i],dimy,-1);
        }

        // Trasformata secondo passaggio (colonna per colonna)

        for(j=0;j<dimy;j++)
        {
            for(i=0;i<dimx;i++)
            {
                R[i]=x[i][j] ;
                I[i]=y[i][j] ;
            }
            fft(R,I,dimx,-1);
            for(i=0;i<dimx;i++)
            {
                x[i][j]=R[i] ;
                y[i][j]=I[i] ;
            }
        }


        // trasformazione numero complesso, parte reale e immaginaria, in
        // modulo fase

        conv(x,y,dimx,dimy,-1) ;


        // riempimento matrice del modulo e matrice della fase

        for(i=0;i<dimx;i++)
        {
            for(j=0;j<dimy;j++)
            {
                imout[i][j]=x[i][j] ;
                imfase[i][j]=y[i][j] ;
            }
        }


        // Si liberano le aree di memoria
        AllocIm::FreeIm(x);
        AllocIm::FreeIm(y);
    }

/*
	conv(..)
	--------

	Routine per trasformazione numero complesso, parte reale/immaginaria,
	in complesso modulo/fase

	ind=-1 re/im -> mod/fase
	ind=1  mod/fase -> re/im
*/

    void conv(double **a, double **b, int dimx, int dimy, int ind)
    {
        double t1,t2;
        int i,j;

        if (ind==-1)
        {
            for(i=0;i<dimx;i++)
            {
                for(j=0;j<dimy;j++)
                {
                    if (a[i][j]>0.0)
                    {
                        t1=sqrt(a[i][j]*a[i][j]+b[i][j]*b[i][j]) ;
                        t2=atan(b[i][j]/a[i][j]);
                        a[i][j]=t1;
                        b[i][j]=t2;
                    }
                    else if (a[i][j]<0.0)
                    {
                        t1=sqrt(a[i][j]*a[i][j]+b[i][j]*b[i][j]) ;
                        t2=atan(b[i][j]/a[i][j])+PI;
                        a[i][j]=t1;
                        b[i][j]=t2;
                    }
                    else
                    {
                        if(b[i][j]>=0.0) t1=b[i][j];
                        else t1=-b[i][j];
                        if (b[i][j] >=0.0) t2=PI/2.0;
                        else t2=-PI/2.0;
                    }
                }
            }
        }
        else
        {
            for(i=0;i<dimx;i++)
            {
                for(j=0;j<dimy;j++)
                {
                    t1=a[i][j]*cos(b[i][j]);
                    t2=a[i][j]*sin(b[i][j]);
                    a[i][j]=t1;
                    b[i][j]=t2;
                }
            }
        }

    }

}  // namespace FFT
