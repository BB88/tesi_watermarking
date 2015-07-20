//
// Created by miky on 20/07/15.
//

#ifndef TESI_WATERMARKING_IMGWAT_H
#define TESI_WATERMARKING_IMGWAT_H

#endif //TESI_WATERMARKING_IMGWAT_H

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
 * \brief Image Watermarking Plugin.
 *
 */


// Qt headers
#include <qobject.h>
#include <qplugin.h>

// Interface
#include "imgwatinterface.h"

// Standard headers
#include <math.h>

#define PI     3.14159265358979323846

/**
 * Image Watermarking Plugin.
 *
 * This Qt plugins implements the image watermarking algorithm
 * of the Laboratorio Comunicazioni ed Immagini, Florence University.
 * For details about the algorithm refer to ... .
 */
class ImgWat : public QObject,
                     public ImgWatInterface
{
    Q_OBJECT
            Q_INTERFACES(ImgWatInterface)

    static const double M0;
    static const double EPS;
    static const double EPS2;
    static const int DIMFILT2;
    static const int DIMFILT;

    static const int NIT;
    static const int NUM_PARAMS;
    static const int MAXZONE;
    static const int WINDOW;

    static const int N;
    static const int DFT;
    static const int IDFT;

    //! Maximum number of tiles.
    static const int MAX_TILES;

    static const char * MSG_TITLE;

    typedef __int64 LONG8BYTE;	// NOTE: __int64 HAS NO ANSI EQUIVALENT (!)

    // Struttura punto di interesse
    struct Point
    {
        float	Val;
        float	Valf;
        int		Riga;
        int		Col;
    };

    // Struttura Scacco per la sincronizzazione
    struct Scacco
    {
        double DeltaX;
        double DeltaY;
        double Alfa;
        double Beta;
        double Periodo;
    };

    // Struttura di un nodo che collegato ricorsivamente crea un albero
    struct tnode
    {
        int    flag;
        struct Point Pt;
        struct tnode *left;
        struct tnode *right;
        struct tnode *father;
    };

// private data members
private:

    // Some useful parameters for BCH encoding:
    //
    //	(127,64,21) (m=7,t=10)
    //	(128,64,17) (m=8,t=8)
    //	(132,64,19)	(m=8,t=9)
    //	(140,64,21) (m=8,t=10)
    //	....

    //! Order of the Galois Field GF(2^m)
    int m_BCH;

    //! Error correcting capability
    int t_BCH;

    //! Length of the BCH code
    int length_BCH;

    //! Variables for re-synchronization.
    double theta, omega, dx, dy;

    int cont[16];
    double z[5];

    LONG8BYTE semecorrente[4];		// seme corrente del generatore clcg
    LONG8BYTE semeiniziale[4];		// seme iniziale del generatore clcg

    //! Password (alphabetic field).
    QString passwstr;

    //! Password (numeric field).
    QString passwnum;

    //! Watermark power.
    float power;

    //! Clipping flag.
    bool clipping;

    //! Number of bits of the watermark.
    int nbits;

    //! Re-synchronization flag.
    bool flagResyncAll;

    //! Size of the single tile (squared).
    int tilesize;

    //! Number of tiles.
    int ntiles;

    /**
     * List of tiles to mark.
     *
     * \note The first element of this list indicates that all the tiles
     * have to be marked. For this reason the tiles list has <em>tiles + 1</em> elements.
     *
     * \note Max 32+1 tiles are considered (see WatCod for further details).
     */
    int tiles[33];

    //! Watermark (max 64 bits).
    int watermark[64];

// public methods
public:

    /**
     * Set watermarking parameters.
     *
     * \param w             Watermark bits
     * \param bits          Number of bits of the watermark
     * \param size          Size of the tiles
     * \param pwr           Watermark power
     * \param useClipping   Clipping flag
     * \param tilelist      List of tiles to watermark
     * \param tilenumber    Number of elements of the tiles list
     */
    void setParameters(int *w, int bits, int size, float pwr, bool useClipping,
                       bool synchronization, int *tilelist, int tilenumber);

    /**
     * Set password.
     */
    void setPassword(QString passwStr, QString passwNum);

    /**
     * Insert the watermark into the given image.
     *
     * This method wrap the original image watermarking encoding algorithm (WatCod).
     *
     * \note The input image is not modified.
     */
    unsigned char * insertWatermark(unsigned char *image, int w, int h);

    /**
     * Extract the watermark from the given image.
     *
     * This method wrap the original image watermarking decoding algorithm (WatDec).
     */
    bool extractWatermark(unsigned char *image, int w, int h);

// private methods (most of them are from  rout_1_2a.h)
private:

    // NOTE: For a description of these routines see the comments inside the code (!!)

    /**
     * Original Image Watermarking Encoding algorithm.
     *
     * For further details see the comments inside the code.
     */
    int WatCod(unsigned char *ImageIn , int nrImageIn, int ncImageIn,
               const char *campolett, const char *camponum, int *marchio, int size, int nbit,
               float power,bool flagClipping, int *vettoretile, int *numerotile);

    /**
     * Original Image Watermarking Decoding algorithm.
     *
     * For further details see the commentz inside the code.
     */
    int WatDec(unsigned char *ImageIn, int nrImageIn, int ncImageIn,
               const char *campolett, const char *camponum, int *bit, int size, int nbit,
               float power, double *datiuscita, unsigned char *buffimrisinc,
               int *vettoretile, bool flagRisincTotale );

    // da rout_1_2a.h
    void PicRoutfloat(float **img_orig, int nr, int nc,
                      float **img_mark, float **img_map_flt, float **impic);

    void resizefloat(float **im_in, float **im_out, int r, int c,int rq, int cq, int ind);

    void DecimVarfloat(float **imc1, int nr, int nc, int win, float **img_map_flt);

    void rgb_to_crom(unsigned char **imr, unsigned char **img, unsigned char **imb,
                     int nr, int nc, int flag, float ** imc1, float **imc2, float ** imc3);

    double sottrvetzone(double *vet, double *mark, int camp);

    double valmedzone(double *vet,int num_camp);

    double valquadzone(double *vet, int num_camp);

    void codmarchio(const char *campolett, const char *camponum, LONG8BYTE *s);

    void inizializza(LONG8BYTE *s);

    double generatore();

    void addmark(double *buff, double *mark, int num_camp, double peso);

    void reverse(char *s);

    void itoa(LONG8BYTE n,char *s);

    double* zone(double **imdft, int nr, int nc, int diag0,
                 int ndiag, int detect, int *elem);

    void antizone(double **imdft,int nr, int nc, int diag0, int ndiag, double *buff);

    double dgamma(double x);

    void mlfunc(double *buff,int nrfile,int niteraz);

    void syncro(float **imc1, float **im_M, int nr, int nc, int periodo, int ampiezza);

    int decsyncro(double **imdft, int nre, int nce, int periodo);

    void resyncro(float **imgin, int nr, int nc, int n1, int n2, float **imgout);

    int Detect(float **ImPtr,int nr, int nc,struct	Scacco *GrigliaPtr, int period);

    double Media(float** Im_in,float** Im_out,int nr,int nc);

    double Threshold(float **Im,int nc,int nr,double mean);

    tnode *MaxPointS(float **Im,float **ImN,int nc, int nr,double Conf);

    tnode *addtree(struct tnode *p,float Val,int Riga,int Col);

    tnode *talloc(void);

    int EstraiMax(struct tnode* p,struct Point *PtrPt);

    void Controllo(struct tnode* p,int nr,int nc,
                   struct Point* PtrPtR,struct Point* PtrPtL);

    void TrovaDelta(float **ImPtr,struct Point *BPtrL,struct Point *BPtrR,
                    struct Scacco *GPtr, int nr,int nc,int Dim, int period);

    void Trasformazione(struct Scacco *GPtr,double	FMx1,double FMy1,
                        double FMx2,double FMy2);

    void FreeTree(struct tnode *p);

    void Marchia(float **Im,float  **Im_f, int nr ,int nc,int na,int nT);

    void decoale(double **imr, int nre, int nce, int d1, int nd,
                 LONG8BYTE *seed, double alpha,int *bit, int nbit);

    inline double sind(double angle)
    {
        return sin((angle*PI/180.0));
    }

    inline double cosd(double angle)
    {
        return cos((angle*PI/180.0));
    }

// accessors
public:

    int * watermarkBits();
};

#endif  /* IMGWATPLUGIN_H */
