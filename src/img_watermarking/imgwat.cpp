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
 * \brief Image Watermarking Plugin (implementation).
 *
 */
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

//// Qt headers
//#include <QMessageBox>

// definitions
const double ImgWat::M0 = 0.7071067811865;
const double ImgWat::EPS = 0.001;
const double ImgWat::EPS2 = 0.035;
const int ImgWat::DIMFILT2 = 3;
const int ImgWat::DIMFILT = 7;

const int ImgWat::NIT = 50;  // numero iterazioni per la stima dei parametri alfa
//   e beta della weibull su ogni zona

const int ImgWat::NUM_PARAMS = 4;  // param. di marchiatura (prima diag., numero diag., potenza di marchiatura)  
const int ImgWat::MAXZONE = 16;  // numero di zone in cui viene suddivisa la parte
// dello spettro sottoposta a marchiatura per farne
// l'analisi statistica.
const int ImgWat::WINDOW = 9;

const char * ImgWat::MSG_TITLE = "Image Watermarking Plugin";

using namespace AllocIm;
using namespace std;

void ImgWat::setParameters(int *w, int bits, int size, float pwr, bool useClipping,
                                 bool resynchronization, int *tilelist, int tilenumber)
{
    // list of bits (watermark)
    memcpy(watermark, w, sizeof(int) * bits);

    nbits = bits;
    tilesize = size;
    power = pwr;
    clipping = useClipping;
    flagResyncAll = resynchronization;

    assert(tilenumber <= 33);
    memcpy(tiles, tilelist, sizeof(int) * tilenumber);
}

void ImgWat::setPassword(std::string passwStr, std::string passwNum)
{
    passwstr = passwStr;
    passwnum = passwNum;
}

int * ImgWat::watermarkBits()
{
    return &(watermark[0]);
}

unsigned char * ImgWat::insertWatermark(unsigned char *image, int w, int h)
{
    bool flagOk;

    unsigned char *output_img = new unsigned char[w * h * 3];
    memcpy(output_img, image, w * h * 3);

//    QByteArray barray = passwstr.toAscii();
    const char *passw_str = passwstr.c_str();
//    QByteArray barray2 = passwnum.toAscii();
    const char *passw_num = passwnum.c_str();

    int result = WatCod(output_img, h, w, passw_str, passw_num, watermark,
                        tilesize, nbits, power, clipping, tiles, &ntiles);

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

bool ImgWat::extractWatermark(unsigned char *image, int w, int h)
{
    bool flagOk;

//    QByteArray barray = passwstr.toAscii();
    const char *passw_str = passwstr.c_str();
//    QByteArray barray2 = passwnum.toAscii();
    const char *passw_num = passwnum.c_str();

    // image after resynchronization
    unsigned char *imrsinc = new unsigned char[w * h * 3];

    // resynchronization data for each tile... not used(!!) 
    // (see inside WatDec(.) for further details)
    double *datiuscita = new double[32000];

    int result = WatDec(image, h, w, passw_str, passw_num, watermark,
                        tilesize, nbits, power, datiuscita, imrsinc, tiles, flagResyncAll);


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
//        flagOk = false;
    }
    else if (result == 0)
    {
        // Watermark bits successfully recovered!!
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
        fprintf(flog, " - Invalid BCH code!!.");
//        // invalid BCH code (!!)
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid BCH code!!"));
        flagOk = false;
    }
    else if (result == 3)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Watermark not found!!.");
//        // watermark not found (!!)
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Watermark not found!!"));
        flagOk = false;
    }
    else if (result == 4)
    {
        // LOG
        FILE *flog = fopen("watcod.log","wt");;
        fprintf(flog, " - Invalid tile!!.");
//        // invalid watermark tile (!!)
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid tile!!"));
        flagOk = false;
    }

    return flagOk;
}


/*
    WatCod(..) esegue la marchiatura della luminanza di un'immagine.
	----------------------------------------------------------------

    [VECCHI COMMENTI]
	Dall'immagine, in formato r,g,b, si estrae la componente di 
	luminanza sulla quale si inserisce
	il sincronismo. Sull'immagine sincronizzata si calcola 
	la mappa di sensibilita'.  
	L'immagine viene estesa automaticamente a 512x512 pixel.
	I parametri di marchiatura (diagonali, ecc) sono assegnati 
	automaticamente.
	Il marchio viene generato in base a due stringhe (una letterale
	ed una numerica) che vengono date in ingresso, dalla linea
	di comando.
	Le stringhe vengono dapprima convertite in un vettore di 4 
	interi (che costituiscono i semi dei 4 lcg) dalla funzione
	codmarchio; successivamente, la funzione generamarchio 
	fornisce il codice random.
	[VECCHI COMMENTI - END]

	DESCRIZIONE INGRESSI: 

		BYTE* ImageIn ----> Immagine da marchiare
	
		int nrImageIn ----> Numero di righe (altezza) 
							dell'immagine da marchiare
	
		int ncImageIn ----> Numero di colonne (larghezza)
							dell'immagine da marchiare

	    char *campolett --> Stringa di caratteri (max 16)
							(i numeri e i segni che non fanno 
							 parte dell'alfabeto previsto vengono
							 codificati con lo 0)

		char *camponum ---> Stringa di caratteri (max 8) rappresentanti
							NUMERI (i numeri ed i caratteri che
							non fanno parte dell'alfabeto previsto
							vengono codificati con lo 0)

		int *marchio -----> E' il marchio da introdurre nell'immagine
							DEVE contenere soltanto valori 0 ed 1
							(attualmente il marchio DEVE essere 
							 costituito da 64 bit)

		int size	------> 256/512/1024 (dimensioni possibili)

		int nbit	------> 32/64 bit (bit del marchio; 32 consigliato per piccole dimensioni)

		float power	------> potenza del marchio (0.1 - 0.9, consigliato 0.4-0.6 (max 0.99999999) )

		bool flagClipping --> Clipping per diminuire visibilit� potenza (by Alessandro PIVA)


    NOTA: campolett + camponum = PASSWORD


	Codici di ritorno:
	
	   -3	--> Errore nei parametri ( 'nbit' non ammissibile !! )
	   -2	--> Errore nei parametri ( 'size' non ammissibile !! )
	   -1	--> Errore nei parametri ( 'power' out-of-range !! )
		0	--> OK!
		1	--> Errore! (Immagine troppo grande !!)
		2   --> Errore! (Si sta cercando di marchiare un tile che non c'� !!)


**********************************************************************/


int ImgWat::WatCod(unsigned char *ImageIn , int nrImageIn, int ncImageIn,
                         const char *campolett, const char *camponum,
                         int *marchio, int size, int nbit,
                         float power,bool flagClipping, int *vettoretile,
                         int *numerotile)
{

    int i,j;
    int marklen;			// lunghezza del marchio
    int nr, nc;				// righe e colonne imm. da marchiare
    int nre, nce;			// righe e colonne imm. estesa
    int flag;				// indica se c'e` stata estensione
    int d1;					// prima diagonale da marchiare
    int nd;					// numero diagonali da marchiare
    int dim2;				// dimensione immagine estesa
    int periodo;			// Periodo (in pixel) della griglia di 
    // sincronismo (consigliato: 6)
    int ampiezza;			// Ampiezza della griglia (cons. 2 o 3): 
    // e' il numero di livelli di grigio che 
    // verranno sommati o sottratti ai 
    // pixel dell'immagine per introdurre 
    // il sincronismo
    int cont=0;

    LONG8BYTE *seed;		// variabile per generare il marchio
    double *mark;			// vettore che contiene il marchio
    double alfa;			// potenza media del marchio
    double alfamax;			// potenza massima del marchio
    double mmedio;			// valor medio della maschera
    float **imridim;		// matrice imm. estesa
    float **imy;			// matrice di luminanza
    float **imc2;			// matrice di crominanza c2
    float **imc3;			// matrice di crominanza c3
    unsigned char **imr;	// matrici delle componenti RGB
    unsigned char **img;
    unsigned char **imb;
    float   **im_M;			// immagine + sincronismo
    float   **impic;		// immagine finale marchiata
    float   **img_map_flt;	// immagine maschera
    double  **imdft;		// immagine della DFT
    double  **imdftfase;	// immagine della fase della DFT
    float   **imidft;		// immagine della IDFT

    int n;
    int k,L;
    int bit[200];			// Max 200 bit 


    /******************MODIFICHE BY GIOVANNI FONDELLI********************/

    int nt;				// Numero totale dei tile
    int dtr, dtc;		// Dimensioni riga e colonna del tile
    int ntr, ntc;		// Numero dei tile per riga e colonna
    int ntx, nty;		// Variabili per il ciclo di marchiatura
    int resto;			// Variabile per il calcolo dei tile necessari
    float **imyout;		// Matrice di luminanza del tile
    float **impicout;   // Immagine finale marchiata
    int dimvr = 0;		// Dimensioni effettive dell'immagine
    int dimvc = 0;		// nel tile
    int presenza = 1;	// Flag presenza del tile
    int tileattuale;	// Numero tile esaminato
    int ntcopia;

    /********************************************************************/


    // START MODIFIED by CORMAX,	14/MAR/2001

    double *buffer=NULL;	// Al posto del 'buff globale'
    long offset;

    nr = nrImageIn;
    nc = ncImageIn;

    // LOG
    FILE *flog = fopen("watcod.log","wt");;
    fprintf(flog, " - Image dimensions: nr=%d nc=%d\n\n",nr,nc);


    // Codifica BCH del marchio 
    ////////////////////////////

    if (nbit == 64)
    {
        m_BCH = 7;			// order of the Galois Field GF(2^m)
        t_BCH = 10;			// Error correcting capability
        length_BCH = 127;	// length of the BCH code
    }

    if (nbit == 32)
    {
        m_BCH = 6;			// order of the Galois Field GF(2^m)
        t_BCH = 5;			// Error correcting capability
        length_BCH = 59;	// length of the BCH code
    }

    if ((nbit != 64)&&(nbit != 32))
    {
        return -3;	// Incorrect 'nbit'
    }

    // LOG
    fprintf(flog, " - BCH: m=%d t=%d length=%d\n\n",m_BCH,t_BCH,length_BCH);

    fprintf(flog, " - Marchio: ");

    for (int ii=0; ii < nbit; ii++)
        fprintf(flog,"%d",marchio[ii]);

    BCH::encode_bch(m_BCH,length_BCH,t_BCH,marchio,bit);

    // LOG
    for (int ii=0; ii < length_BCH; ii++)
        fprintf(flog, "%d",bit[ii]);

    fprintf(flog, "\n\n");

    // STOP MODIFIED by CORMAX		14/MAR/2001

    // parametri di marchiatura
    ////////////////////////////

    // NOTA: Se si vuole modificare questi parametri � necessario
    //       modificarli pure in WatDec(..), poich� devono essere
    //		 identici

    // Controllo sulle dimensioni del tile
    //////////////////////////////////////

    if (size<=256)
    {
        dim2 = 256;		// Dimensione 256x256
        d1 = 40;		// Diagonali..
        nd = 40;
    }

    if ((size>256)&&(size<=512))
    {
        dim2 = 512;		// Dimensione 512x512
        d1 = 80;		// Diagonali..
        nd = 74;
    }

    if ((size>512)&&(size<=1024))
    {
        dim2 = 1024;	// Dimensione 1024x1024
        d1 = 160;		// Diagonali..
        nd = 144;
    }

    alfa = power;	// Potenza del Marchio

    if ((alfa < 0.1)||(alfa > 0.9))
    {
        return -1;		// Power out-of-range
    }

    periodo = 6;
    ampiezza = 2;

    // LOG
    fprintf(flog, " - Params: dim2=%d d1=%d nd=%d periodo=%d ampiezza=%d power=%.2f\n\n",
            dim2,d1,nd,periodo,ampiezza,alfa);

    // Controllo degli errori sui parametri di ingresso
    ////////////////////////////////////////////////////


/******************MODIFICHE BY GIOVANNI FONDELLI********************/


    // Allocazione memoria per i semi del generatore e 
    // codifica delle stringhe-marchio

    seed = new LONG8BYTE [4];
    codmarchio(campolett, camponum, seed);

    // LOG
    fprintf(flog, " - Seed: %f %f %f %f\n\n",
            (LONG8BYTE)seed[0], (LONG8BYTE)seed[1], (LONG8BYTE)seed[2], (LONG8BYTE)seed[3]);

    dtr = size;    // Per adesso si suppone il tile quadrato
    dtc = size;

    // Allocazione della memoria
    /////////////////////////////

    imy = AllocImFloat(nr, nc);
    imc2 = AllocImFloat(nr, nc);
    imc3 = AllocImFloat(nr, nc);
    impicout = AllocImFloat(nr, nc);


/********************************************************************/


    // Suddivisione dell'immagini nelle 3 componenti fondamentali
    // (Added by CORMAX)

    imr = AllocImByte(nr, nc);
    img = AllocImByte(nr, nc);
    imb = AllocImByte(nr, nc);

    offset = 0;
    for (i=0; i<nr; i++)
        for (j=0; j<nc; j++)
        {
            imr[i][j] = ImageIn[offset];offset++;
            img[i][j] = ImageIn[offset];offset++;
            imb[i][j] = ImageIn[offset];offset++;
        }

    // Pre-elaborazioni, Estensioni, Sincronizzazione, FFT, etc
    ////////////////////////////////////////////////////////////

    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    rgb_to_crom(imr, img, imb, nr, nc, 1, imy, imc2, imc3);


    // LOG
    fprintf(flog, " - IMY (line 90): ");
    for (int ii = 0; ii < 30; ii++)
        fprintf(flog, "%.4f ", imy[90][ii]);

    fprintf(flog, "\n - IMY (line 150): ");
    for (int ii = 0; ii < 30; ii++)
        fprintf(flog, "%.4f ", imy[150][ii]);

    fprintf(flog,"\n\n");

    /******************MODIFICHE BY GIOVANNI FONDELLI********************/

    // Calcolo dei tile necessari per la codifica
    /////////////////////////////////////////////

    ntr=nr/dtr;
    resto=nr%dtr;
    if (resto>0)
        ntr=ntr+1;

    ntc=nc/dtc;
    resto=nc%dtc;
    if (resto>0)
        ntc=ntc+1;

    nt = ntr * ntc;

    ntcopia = nt;

    *numerotile = ntcopia;

    if (vettoretile[0]==1)    // Flag All tile attivato
    {
        for (i=1; i<=nt; i++) vettoretile[i]=1;
    }
    else
    {
        for (i=1; i<=32; i++)    // Controllo marchiatura tile inesistente
        {
            if ((vettoretile[i]==1)&&(i>nt))
                return 2;
        }
    }

    // LOG
    fprintf(flog, " - Tile Numbers: nt=%d ntr=%d ntc=%d\n\n",nt,ntr,ntc);


    // Inizio ciclo marchiatura
    ///////////////////////////

    for (ntx=1; ntx<=ntr; ntx++)
        for (nty=1; nty<=ntc; nty++)
        {
            tileattuale=nty+(ntc*(ntx-1));	// Calcolo del numero del tile
            // conoscendo la riga ntx, la
            // colonna nty e ntc

            if (vettoretile[tileattuale]==1)
                presenza=1;
            else if (vettoretile[tileattuale]==0)
                presenza=0;

            for (i=0; i<dtr; i++)
                for (j=0; j<dtc; j++)
                    if ((i+((ntx-1)*dtr)<nr)&&(j+((nty-1)*dtc)<nc))
                    {
                        dimvr=i+1;
                        dimvc=j+1;
                    }

            imyout = AllocImFloat(dimvr, dimvc);
            im_M = AllocImFloat(dimvr, dimvc);
            impic = AllocImFloat(dimvr, dimvc);
            imdft = AllocImDouble(dimvr, dimvc);
            imdftfase = AllocImDouble(dimvr, dimvc);
            imidft = AllocImFloat(dimvr, dimvc);
            img_map_flt = AllocImFloat(dimvr, dimvc);

            for (i=0; i<dimvr; i++)
                for (j=0; j<dimvc; j++)
                    imyout[i][j] =
                            static_cast<float>(imy[i+((ntx-1)*dtr)][j+((nty-1)*dtc)]);

            if (presenza==1)
            {
                // LOG
                fprintf(flog, " - Syncro: dimvr=%d dimvc=%d periodo=%d ampiezza=%d\n\n",
                        dimvr,dimvc,periodo,ampiezza);

                // Si introduce il sincronismo sull'immagine luminanza del tile
                syncro(imyout, im_M, dimvr, dimvc, periodo, ampiezza);

                // Si calcola la maschera di sensibilita' del tile
                DecimVarfloat(imyout, dimvr, dimvc, WINDOW, img_map_flt);

                // Si controllano le dimensioni dell'immagine per effettuare
                // un'eventuale estensione e si calcola la
                // FFT dell'immagine (o dell'immagine estesa)


                // Si controlla se l'immagine ha dimensioni pi� piccole di un tile
                //////////////////////////////////////////////////////////////////

                if ((dimvr==dim2)&&(dimvc==dim2))
                {
                    nre = dimvr;
                    nce = dimvc;

                    FFT2D::dft2d(im_M, imdft, imdftfase, nre, nce);

                    flag = 0;  // non c'e` stata estensione
                }
                else
                {
                    nre = dim2;
                    nce = dim2;

                    // Allocazione memoria per l'immagine estesa, per calcolarne la DFT

                    FreeIm(imdft);
                    FreeIm(imdftfase);
                    FreeIm(imidft);

                    imridim= AllocImFloat(nre, nce);
                    imdft = AllocImDouble(nre, nce);
                    imdftfase = AllocImDouble(nre, nce);
                    imidft = AllocImFloat(nre, nce);

                    resizefloat(im_M, imridim, dimvr, dimvc, nre, nce, 1);

                    FFT2D::dft2d( imridim, imdft, imdftfase, nre, nce);

                    flag = 1;    // c'e` stata estensione
                }

                // LOG
                fprintf(flog, " - Estensione: flag=%d\n\n",flag);


                // Si calcola il valor medio della mappa 
                // (la mappa ha le stesse dimensioni del tile)
                //////////////////////////////////////////////

                mmedio = 0;

                for(i = 0; i < dimvr; i++)
                    for(j = 0; j < dimvc; j++)
                        mmedio += (double)img_map_flt[i][j];

                mmedio = mmedio/(double)(dimvr * dimvc);
                mmedio = 1.0 - mmedio;

                // Si calcola il valore massimo di alfa
                alfamax = alfa/mmedio;

                // LOG
                fprintf(flog, " -  Valore medio mappa: mmedio=%f alfamax=%f\n\n",
                        mmedio,alfamax);


                /********************************************************************/


                /***************************************************************/
                /******************* PROCESSO DI MARCHIATURA *******************/
                /***************************************************************/

                // Selezione dei coefficienti da marchiare
                buffer = zone(imdft, nre, nce, d1, nd, 0, &marklen);

                // Creazione del marchio (con distribuzione uniforme tra [-1,1])
                mark = new double [marklen];

                inizializza(seed);

                for(i = 0; i < marklen; i++)
                    mark[i] = 2.0 * (generatore() - 0.5);

                // LOG
                fprintf(flog, " - Mark (marklen=%d): ",marklen);
                for (int ii = 0; ii < 30; ii++)
                    fprintf(flog, "%.4f ", mark[ii]);

                fprintf(flog,"\n\n");

                // Effettua la MODULAZIONE
                ///////////////////////////

                n=0;
                L=marklen/length_BCH;
                for (k=length_BCH-1; k>=0; k--)
                {
                    if (bit[k]==0)
                    {
                        if (k==0)
                        {
                            for (i=n; i<marklen;mark[i]*=-1,i++);
                        }
                        else
                            for (i=n; i<n+L;mark[i]*=-1,i++);
                        n+=L;
                    }
                    else
                        n+=L;
                }


                // Modifica dei coefficienti col marchio scelto

                addmark(buffer, mark, marklen, alfamax);


                // Reinserimento dei coefficienti marchiati

                antizone(imdft, nre, nce, d1, nd, buffer);


                // IDFT dell'immagine marchiata

                FFT2D::idft2d(imdft, imdftfase, imidft, nre, nce);


                /******************MODIFICHE BY GIOVANNI FONDELLI********************/


                // Pesatura tra l'immagine marchiata I' e la maschera di sensibilita'

                for(i=0;i<dimvr;i++)
                    for(j=0;j<dimvc;j++)
                        img_map_flt[i][j] = 255.0f*img_map_flt[i][j];

                PicRoutfloat(imyout, dimvr, dimvc, imidft, img_map_flt, impic);

                // Confronto fra immagine originale e imm. marchiata finale

                // by Alessandro PIVA  (CLIPPING)

                if (flagClipping)
                {
                    for(i = 0; i < dimvr; i++)
                        for(j = 0; j < dimvc; j++)
                        {
                            if(imyout[i][j] - impic[i][j] > 5.0f)
                            {
                                impic[i][j] = imyout[i][j] - 5.0f;
                                cont++;
                            }
                            else if (imyout[i][j] - impic[i][j] < -5.0f)
                            {
                                impic[i][j] = imyout[i][j] + 5.0f;
                                cont++;
                            }
                        }
                }
            }  // Fine del controllo if presenza==1


            // Si ricostruisce l'immagine finale marchiata dai singoli tile
            ///////////////////////////////////////////////////////////////

            if (presenza==1)
            {
                for (i=0; i<dimvr; i++)
                    for (j=0; j<dimvc; j++)
                        impicout[i+((ntx-1)*dtr)][j+((nty-1)*dtc)]=
                                static_cast<float>(impic[i][j]);
            }
            else
            {
                for (i=0; i<dimvr; i++)
                    for (j=0; j<dimvc; j++)
                        impicout[i+((ntx-1)*dtr)][j+((nty-1)*dtc)]=
                                static_cast<float>(imyout[i][j]);
            }

            FreeIm(imyout);
            FreeIm(img_map_flt);
            FreeIm(im_M);
            FreeIm(impic);
            FreeIm(imdft);
            FreeIm(imdftfase);
            FreeIm(imidft);

            if(flag == 1)
                FreeIm(imridim);

        }

    // Fine ciclo marchiatura
    ///////////////////////////////////////////////


    /********************************************************************/


    // LOG
    fprintf(flog, " - IMYw (line 90): ");
    for (int ii=0; ii < 30; ii++)
        fprintf(flog, "%.4f ",impicout[90][ii]);

    fprintf(flog, "\n - IMYw (line 150): ");
    for (int ii=0; ii<30; ii++)
        fprintf(flog, "%.4f ",impicout[150][ii]);

    fprintf(flog,"\n\n");

    // Reinserimento della luminanza marchiata nell'immagine
    rgb_to_crom(imr, img, imb, nr, nc, -1, impicout, imc2, imc3);

    offset = 0;
    for (i=0; i<nr; i++)
        for (j=0; j<nc; j++)
        {
            ImageIn[offset] = imr[i][j]; offset++;
            ImageIn[offset] = img[i][j]; offset++;
            ImageIn[offset] = imb[i][j]; offset++;
        }

    if (buffer != NULL)
        delete [] buffer;

    delete [] seed;
    delete [] mark;
    FreeIm(imy) ;
    FreeIm(imc2) ;
    FreeIm(imc3) ;
    FreeIm(impicout) ;
    FreeIm(imr);
    FreeIm(img);
    FreeIm(imb);

    // LOG
    fclose(flog);

    return 0;    // OK!
}

/*
	decoale(..)
	-----------

	Rivela il marchio in un'immagine di dimensioni potenze d
	del 2 utilizzando il criterio di Neyman-Pearson; si calcola il 
	rapporto di verosimiglianza logaritmico :

		L(x)=f(x|m*)/f(x|0) e la soglia.
    
	Beta e alfa sono stimati a posteriori sull'immagine analizzata.

	La funzione ritorna L(x) - la soglia  
*/

void ImgWat::decoale(double **imr, int nre, int nce, int d1, int nd,
                           LONG8BYTE *seed, double alpha,int *bit, int nbit)
{
    int i,k,n;
    int	tot;
    int marklen;			// lunghezza del marchio
    double  *mark;			// vettore marchio
    double  aweib[MAXZONE];
    double  bweib[MAXZONE];	// vettori alfa e beta per ogni zona
    double  v1,v2;			// var. di appoggio per il
    //   calcolo della soglia
    double  *aw,*bw;
    double	soma,somb;
    int L;

    double *appbuff = NULL;

    // La funzione zone raggruppa i coeff. marchiati in MAX zone 
    // e restituisce la lunghezza del marchio (ossia il numero 
    // di coefficienti selezionati)

    appbuff = zone(imr, nre, nce, d1, nd, 1, &marklen);

    // Studio della statistica delle MAX zone (si calcolano i beta 
    // e gli alfa dell'immagine con il criterio della Massima Verosimiglianza)

    tot = 0;

    for(k=0; k<MAXZONE; k++)
    {
        // studio della statistica della zona i-esima

        mlfunc(appbuff,cont[k],NIT);

        appbuff += cont[k];
        tot += cont[k];

        aweib[k]=z[2];
        bweib[k]=z[1];
    }


    // Si genera il marchio seed  e si calcola L(X) e soglia

    mark = new double [marklen];
    aw = new double [marklen];
    bw = new double [marklen];

    appbuff -= tot;

    n=0;
    for (k=0; k<MAXZONE; k++)
    {
        for (i=n;i<n+cont[k];i++)
        {
            aw[i]=aweib[k];
            if (aw[i]==0) aw[i]=0.000001;
            bw[i]=bweib[k];
        }

        n+=cont[k];
    }

    inizializza(seed);

    for(i = 0; i < marklen; i++)
        mark[i] = 2.0 * (generatore() - 0.5);

    // Calcolo della soglia del rivelatore 

    /*
        ...

        // ADDED BY CORMAX
        //
        // LA SOGLIA NON SERVE, ESSENDO UNA MARCHIATURA A LETTURA!!

    */

    // DECODIFICA (LETTURA)
    ////////////////////////

    n=0;
    L=marklen/nbit;
    soma=somb=0.0;
    for (k=nbit-1;k>0;k--)
    {
        v1=v2=0;

        for (i=n;i<n+L;i++)
        {
            v1-=pow((appbuff[i]/(aw[i]*(1.0+alpha*mark[i]))),bw[i]);
            v2-=pow((appbuff[i]/(aw[i]*(1.0-alpha*mark[i]))),bw[i]);
        }

        if (v1>v2)
        {
            bit[k]=1;
        }
        else bit[k]=0;

        n+=L;
    }

    // Ultimo bit
    v1=v2=0;
    for (i=n;i<marklen;i++)
    {
        v1-=pow((appbuff[i]/(aw[i]*(1.0+alpha*mark[i]))),bw[i]);
        v2-=pow((appbuff[i]/(aw[i]*(1.0-alpha*mark[i]))),bw[i]);
    }

    if (v1>v2)
    {
        bit[k]=1;
    }
    else bit[k]=0;

    delete [] mark;
    delete [] aw;
    delete [] bw;

    if (appbuff != NULL)
        delete [] appbuff;
}


/*******************************************************************


	WatDec(..)
	----------

    [VECCHI COMMENTI]
	Effettua la decodifica del marchio su un'immagine a colori 
	supposta marchiata.
  	Dall'immagine  supposta marchiata si estrae la componente
	luminanza, e la si estende ad una dimensione fissata utile per 
	la FFT (es. 512x512).
	Poi si ricercano i picchi in frequenza che descrivono lo spettro
	della griglia di sincronismo, e si risincronizza l'immagine, 
	riportandola nelle condizioni geometriche che aveva all'uscita
	del processo di marchiatura.
	[VECCHI COMMENTI - END]

	Ingressi:
		
		BYTE* ImageIn	-->	Immagine marchiata

		int nrImageIn	-->	Altezza dell'Immagine marchiata

		int ncImageIn	--> Larghezza dell'Immagine marchiata

		char* campolett	--> Password (parte caratteri, max 16)

		char* camponum	--> Password (parte numeri, max 8)

		int* bit		--> Marchio sotto forma di bit

		int size		--> Dimensione immagine (ammissibili: 256/512/1024)

		int nbit		--> Numero di bit del marchio (ammissibili: 32/64)

		float power		--> Potenza del marchio (0.1-0.9, consigliato: 0.4-0.6)
							N.B: questo parametro non pu� superare 1.0 per ipotesi 
						 	     matematiche relative alla costruzione dell'algoritmo 
							     di marchiatura


	Codici di ritorno:
	
	   -3	--> Errore nei parametri ( 'nbit' non ammissibile !! )
	   -2	--> Errore nei parametri ( 'size' non ammissibile !! )
	   -1	--> Errore nei parametri ( 'power' out-of-range !! )
		0	--> OK!
		1	--> Errore! (Immagine troppo grande !!)
		2	--> Errore! (Codice BCH non valido !!)
		3	--> Errore! (Marchio non Rilevato !!)
		4   --> Errore! (Si sta cercando di decodificare un tile che non c'� !!)


*******************************************************************/

int ImgWat::WatDec(unsigned char *ImageIn, int nrImageIn, int ncImageIn,
                         const char *campolett, const char *camponum,
                         int *bit, int size, int nbit,
                         float power, double *datiuscita, unsigned char *buffimrisinc,
                         int *vettoretile, bool flagRisincTotale )
{
    int nr, nc;				// righe e colonne imm. marchiata
    int nre, nce;			// righe e colonne imm. estesa
    int dim2;				// dimensione immagine estesa
    int d1;					// prima diagonale marchiata
    int nd;					// numero diagonali marchiate
    int periodo;			// Periodo (in pixel) della griglia di 
    // sincronismo (consigliato: 6)
    int i,j;
    double alpha;			// potenza media del marchio cercato
    unsigned char **imr;	// immagini delle tre componenti RGB
    unsigned char **img;	//
    unsigned char **imb;	//
    float **imy;			// matrice luminanza
    float **imc2;			// matrice crominanza c2
    float **imc3;			// matrice crominanza c3
    float **imridim;		// matrice luminanza estesa
    float **imrisinc;		// matrice luminanza risincronizzata
    double **imdft;			// matrice contenente il modulo della DFT 
    double **imdftfase;		// matrice contenente la fase della DFT
    double **im1dft;		// matrice contenente il modulo della DFT
    double **im1dftfase;	// matrici contenente la fase della DFT
    LONG8BYTE *seed;		// variabile per generare il marchio

    // Added by CORMAX
    int BitLetti[200];		// Max 200 bit da leggere
    int dec=0;
    long offset;


    /******************MODIFICHE BY GIOVANNI FONDELLI********************/

    int nt;					// Numero totale dei tile     
    int dtr, dtc;			// Dimensioni riga e colonna del tile
    int ntr,ntc;			// Numero dei tile per riga e colonna
    int ntx, nty;			// Variabili per il ciclo di decodifica
    int resto;				// Variabile per il calcolo dei tile necessari
    int decappoggio = 0;	// Controllo per vedere se nei singoli
    // tile � presente il sincronismo
    float **imyout;			// Matrice di luminanza del tile	
    double **imdftout;		// Matrice dft del tile ridimensionato
    double **imdftoutfase;
    int dimvr = 0;			// Dimensioni effettive dell'immagine
    int dimvc = 0;			// nel tile
    const int maxtile=30100;// Massimo numero di tile previsto	
    double **im1dftrisinc;	// Matrice contenente la somma dei moduli
    // delle dft delle immagini risincronizzate
    float **imrisincout;	// Matrice luminanza immagine risincronizzata	
    double datiout[maxtile];// Vettore contenente i dati dei tile
    int presenza = 1;		// Flag presenza del tile
    int tileattuale;		// Numero tile esaminato
    int ciclo;				// Variabile per risincronizzazione totale
    float **imrisinctotale;	// Matrice luminanza immagine totale risinc.
    int tileusati=0;		// Conteggio del numero di tile
    // effettivamente elaborati nel ciclo	
    int nouniforme=0;		// Controllo ver vedere se ho una zona uniforme


    /********************************************************************/


    nr = nrImageIn;
    nc = ncImageIn;


    // Parametri per decodifica BCH del marchio 
    ///////////////////////////////////////////

    if (nbit == 64)
    {
        m_BCH = 7;			// order of the Galois Field GF(2^m)
        t_BCH = 10;			// Error correcting capability
        length_BCH = 127;	// length of the BCH code
    }

    if (nbit == 32)
    {
        m_BCH = 6;			// order of the Galois Field GF(2^m)
        t_BCH = 5;			// Error correcting capability
        length_BCH = 59;	// length of the BCH code
    }

    if ((nbit != 64)&&(nbit != 32))
    {
        return -3;	// Incorrect 'nbit'
    }

    int ii;
    FILE *flog;
    flog = fopen("watdec.log", "wt");

    // LOG
    fprintf(flog, " - Image dimensions: nr=%d nc=%d\n\n",nr,nc);
    fprintf(flog, " - BCH: m=%d t=%d length=%d\n\n",m_BCH,t_BCH,length_BCH);

    // Parametri di marchiatura 
    //////////////////////////// 

    // NOTA: Se si vuole modificare questi parametri � necessario
    //       modificarli pure in WatDec(..), poich� devono essere
    //		 identici

    // Controllo sulle dimensioni del tile
    //////////////////////////////////////

    if (size<=256)
    {
        dim2 = 256;		// Dimensione 256x256
        d1 = 40;		// Diagonali..
        nd = 40;
    }

    if ((size>256)&&(size<=512))
    {
        dim2 = 512;		// Dimensione 512x512
        d1 = 80;		// Diagonali..
        nd = 74;
    }

    if ((size>512)&&(size<=1024))
    {
        dim2 = 1024;	// Dimensione 1024x1024
        d1 = 160;		// Diagonali..
        nd = 144;
    }


    /********************************************************************/


    alpha = power;	// Potenza del Marchio

    if ((alpha < 0.1)||(alpha > 0.9))
    {
        return -1;		// Power out-of-range
    }


    periodo = 6;


    // LOG
    fprintf(flog, " - Params: dim2=%d d1=%d nd=%d periodo=%d power=%f\n\n",
            dim2,d1,nd,periodo,alpha);

    //  Controllo errori sui parametri di ingresso
    ///////////////////////////////////////////////


    /******************MODIFICHE BY GIOVANNI FONDELLI********************/


    // Controllo sul flag che decide il rivelatore: se e' stato fornito 
    // un flag sbagliato, si puo' correggere

    // PRE-PROCESSING
    //////////////////

    nre = dim2;
    nce = dim2;

    dtr = size;	// Per adesso si suppone il tile quadrato
    dtc = size;


    // Allocazione della memoria
    ////////////////////////////

    imy = AllocImFloat(nr, nc);
    imc2 = AllocImFloat(nr, nc);
    imc3 = AllocImFloat(nr, nc);
    imrisinc = AllocImFloat(nre, nce);
    imridim = AllocImFloat(nre, nce);
    imdft = AllocImDouble(nre, nce);
    imdftfase = AllocImDouble(nre, nce);
    imdftout = AllocImDouble(nre, nce);
    imdftoutfase = AllocImDouble(nre, nce);
    im1dft = AllocImDouble(nre, nce);
    im1dftfase = AllocImDouble(nre, nce);
    im1dftrisinc = AllocImDouble(nre, nce);
    imrisincout = AllocImFloat(nr, nc);


    /********************************************************************/


    // Separazione delle componenti rgb		(added by CORMAX)

    imr = AllocImByte(nr, nc);
    img = AllocImByte(nr, nc);
    imb = AllocImByte(nr, nc);

    offset = 0;
    for (i=0; i<nr; i++)
        for (j=0; j<nc; j++)
        {
            imr[i][j] = ImageIn[offset];offset++;
            img[i][j] = ImageIn[offset];offset++;
            imb[i][j] = ImageIn[offset];offset++;
        }

    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    rgb_to_crom(imr, img, imb, nr, nc, 1, imy, imc2, imc3);

    // LOG
    fprintf(flog, " - IMY (line 90): ");
    for (ii=0; ii<30; ii++)
        fprintf(flog, "%.0f ", imy[90][ii]);

    for (ii=0; ii < 30;ii++)
        fprintf(flog, "%.0f ", imy[150][ii]);

    fprintf(flog, "\n\n");


    /******************MODIFICHE BY GIOVANNI FONDELLI********************/


    // Calcolo dei tile necessari per la decodifica
    ////////////////////////////////////////////////
    ntr=nr/dtr;
    resto=nr%dtr;
    if (resto>0) ntr=ntr+1;

    ntc=nc/dtc;
    resto=nc%dtc;
    if (resto>0) ntc=ntc+1;

    nt=ntr*ntc;

    // Inizializzazioni
    ///////////////////

    for (i=0; i<maxtile; i++) datiout[i]=0.0;

    for (i=0; i<nre; i++)
        for (j=0; j<nce; j++)
            imdft[i][j]=0.0;

    for (i=0; i<nre; i++)
        for (j=0; j<nce; j++)
            imdftout[i][j]=0.0;

    for (i=0; i<nre; i++)
        for (j=0; j<nce; j++)
            im1dftrisinc[i][j]=0.0;


    // Nel 29990� elemento del vettore datiuscita
    // viene memorizzato il numero di tile totali
    // mentre nel 29994� elemento viene memorizzato
    // se viene decodificato un unico tile
    //////////////////////////////////////
    datiout[29990]=nt;

    if(vettoretile[0]==1)		// Flag All tile attivato
    {
        for (i=1; i<=nt; i++) vettoretile[i]=1;
    }
    else
    {

        for (i=1; i<=32; i++)	// Controllo decodifica tile inesistente
        {
            if ((vettoretile[i]==1)&&(i>nt))
            {
                for (i=0; i<maxtile; i++) datiuscita[i] = datiout[i];
                return 4;
            }
        }
    }

    for (i=1; i<=nt; i++)		// Calcolo tile effettivamente usati
    {
        if (vettoretile[i]==1) tileusati++;
    }

    datiout[30021]=tileusati;

    // Codifica delle stringhe-marchio per la generazione dei semi dei 4 lcg
    ////////////////////////////////////////////////////////////////////////

    seed = new LONG8BYTE [4];
    codmarchio(campolett, camponum, seed);

    // LOG
    fprintf(flog, " - Seed: %f %f %f %f\n\n",
            (LONG8BYTE)seed[0], (LONG8BYTE)seed[1], (LONG8BYTE)seed[2], (LONG8BYTE)seed[3]);

    // Inizio ciclo decodifica
    //////////////////////////

    for (ntx=1; ntx<=ntr; ntx++)
        for (nty=1; nty<=ntc; nty++)
        {
            tileattuale=nty+(ntc*(ntx-1));

            if (vettoretile[tileattuale]==1)
                presenza=1;
            else if (vettoretile[tileattuale]==0)
                presenza=0;

            dec=0;

            nouniforme=0;

            if (presenza == 1)
            {
                // Nel primo elemento del vettore datiuscita
                // viene memorizzato il tile attuale esaminato
                //////////////////////////////////////////////
                datiout[0] = tileattuale;

                for (i=0; i<dtr; i++)
                    for (j=0; j<dtc; j++)
                        if ((i+((ntx-1)*dtr)<nr)&&(j+((nty-1)*dtc)<nc))
                        {
                            dimvr=i+1;
                            dimvc=j+1;
                        }

                imyout = AllocImFloat(dimvr, dimvc);

                for (i=0; i<dimvr; i++)
                    for (j=0; j<dimvc; j++)
                        imyout[i][j]=(float)imy[i+((ntx-1)*dtr)][j+((nty-1)*dtc)];

                resizefloat(imyout, imridim, dimvr, dimvc, nre, nce, 1);


                // RIVELAZIONE DIRETTA DEL MARCHIO SENZA 
                // RECUPERO SINCRONISMO O RIVELAZIONE DI PRIMO LIVELLO
                //////////////////////////////////////////////////////

                // Si fa la DFT dell'immagine estesa per recuperare il marchio
                //////////////////////////////////////////////////////////////

                FFT2D::dft2d(imridim, imdftout, imdftoutfase, nre, nce);

                // Si memorizza la somma delle dft dei tile
                ///////////////////////////////////////////////

                for (i=0; i<nre; i++)
                    for (j=0; j<nce; j++)
                        imdft[i][j] += (double)imdftout[i][j];

                decoale(imdftout, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

                for (i=0; i<200; i++)
                    nouniforme += BitLetti[i];

                datiout[30020] = nouniforme;

                if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti)) &&
                    (nouniforme>0))
                {
                    // THE WATERMARK HAS BEEN DETECTED
                    FreeIm(imy);
                    FreeIm(imc2);
                    FreeIm(imc3);
                    FreeIm(imyout);
                    FreeIm(imrisinc);
                    FreeIm(imridim);
                    FreeIm(imdft);
                    FreeIm(imdftfase);
                    FreeIm(imdftout);
                    FreeIm(imdftoutfase);
                    FreeIm(im1dft);
                    FreeIm(im1dftfase);
                    FreeIm(im1dftrisinc);
                    FreeIm(imr);
                    FreeIm(img);
                    FreeIm(imb);

                    // NOTA: I bit decodificati vengono salvati alla fine
                    //		 di BitLetti, ossia a partire dalla posizione
                    //		 (BCH length)-(information message)

                    int offs;
                    offs = length_BCH - nbit;
                    for (i=0; i<nbit; i++)
                        bit[i] = BitLetti[i+offs];

                    for (i=0; i<maxtile; i++)
                        datiuscita[i] = datiout[i];

                    // Si salva l'immagine risincronizzata
                    //////////////////////////////////////

                    offset = 0;
                    for (i=0; i<nr; i++)
                        for (j=0; j<nc; j++)
                        {
                            buffimrisinc[offset] = imrisincout[i][j]; offset++;
                            buffimrisinc[offset] = imrisincout[i][j]; offset++;
                            buffimrisinc[offset] = imrisincout[i][j]; offset++;
                        }

                    FreeIm(imrisincout);

                    // LOG
                    fclose(flog);

                    return 0;	// OK! Marchio rivelato
                }
                else
                {
                    // Il codice BCH non � valido, si prova ad effettuare
                    // il recupero del sincronismo dell'immagine, ossia:
                    // RIVELAZIONE DIRETTA DEL MARCHIO CON RECUPERO SINCRONISMO
                    // O RIVELAZIONE DI SECONDO LIVELLO

                    // Recupero dei picchi in frequenza che indicano la griglia di sincronismo

                    // Inizializzo i valori di rotazione/zoom, in modo che
                    // se desyncro non trova il sincronismo, non si abbiano
                    // valori nulli per dx e dy

                    theta=omega=0.0;
                    dx=dy=1.0;

                    dec = decsyncro(imdftout, nre, nce, periodo);

                    // LOG
                    fprintf(flog, "  - decsyncro: theta=%.4f omega=%.4f dx=%.4f dy=%.4f\n\n",
                            theta, omega, dx ,dy);

                    // Nel vettore datiuscita vengono memorizzati
                    // i dati relativi ai parametri di sincronismo
                    // per i tile
                    /////////////
                    datiout[tileattuale*2*nt] = (double)theta;
                    datiout[(tileattuale*2*nt)+1] = (double)dx;
                    datiout[(tileattuale*2*nt)+2] = (double)omega;
                    datiout[(tileattuale*2*nt)+3] = (double)dy;

                    if (dec==1)
                    {
                        decappoggio=dec;

                        // Dal secondo elemento
                        // del vettore datiuscita viene memorizzato
                        // se in quel tile ho risincronizzazione
                        //////////////////////////////////////
                        datiout[tileattuale] = 1;

                        // Nel 29992� elemento del vettore datiuscita
                        // viene memorizzato se viene risincronizzato
                        // il tile
                        //////////
                        datiout[29992] = 1;


                        // Risincronizzazione dell'immagine
                        ///////////////////////////////////

                        resyncro(imyout, dimvr, dimvc, dim2, dim2, imrisinc);

                        // Si ricostruisce l'immagine risincronizzata
                        /////////////////////////////////////////////

                        for (i=0; i<dim2; i++)
                            for (j=0; j<dim2; j++)
                                if(((i+(ntx-1)*dtr)<nr)&&((j+(nty-1)*dtc)<nc))
                                    imrisincout[i+((ntx-1)*dtr)][j+((nty-1)*dtc)]=
                                            static_cast<float>(imrisinc[i][j]);


                        FFT2D::dft2d(imrisinc, im1dft, im1dftfase, nre, nce);

                        // Si fa la somma delle dft dei tile risincronizzati
                        ////////////////////////////////////////////////////

                        for (i=0; i<nre; i++)
                            for (j=0; j<nce; j++)
                                im1dftrisinc[i][j] += static_cast<float>(im1dft[i][j]);

                        decoale(im1dft, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

                        nouniforme=0;

                        for (i=0; i<200; i++)
                            nouniforme += BitLetti[i];

                        datiout[30020] = nouniforme;

                        if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti)) &&
                            (nouniforme>0))
                        {
                            // WATERMARK HAS BEEN DETECTED 
                            FreeIm(imy);
                            FreeIm(imc2);
                            FreeIm(imc3);
                            FreeIm(imyout);
                            FreeIm(imrisinc);
                            FreeIm(imridim);
                            FreeIm(imdft);
                            FreeIm(imdftfase);
                            FreeIm(imdftout);
                            FreeIm(imdftoutfase);
                            FreeIm(im1dft);
                            FreeIm(im1dftfase);
                            FreeIm(im1dftrisinc);
                            FreeIm(imr);
                            FreeIm(img);
                            FreeIm(imb);

                            // NOTA: I bit decodificati vengono salvati alla fine
                            //		 di BitLetti, ossia a partire dalla posizione
                            //		 (BCH length)-(information message)

                            int offs;
                            offs = length_BCH - nbit;
                            for (i=0; i<nbit; i++)
                                bit[i] = BitLetti[i+offs];

                            for (i=0; i<maxtile; i++)
                                datiuscita[i] = datiout[i];

                            // Si salva l'immagine risincronizzata
                            //////////////////////////////////////

                            offset = 0;
                            for (i=0; i<nr; i++)
                                for (j=0; j<nc; j++)
                                {
                                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                                }

                            FreeIm(imrisincout);

                            // LOG
                            fclose(flog);

                            return 0;	// OK!
                        }

                    }  // Fine if dec=1

                }  // Fine else

                FreeIm(imyout);

            } // Fine if presenza=1 

        }  // Fine ciclo decodifica dei tile


    // Si controlla se il sincronismo non � presente nei tile
    /////////////////////////////////////////////////////////

    if (decappoggio!=1)
    {
        FreeIm(imy);
        FreeIm(imc2);
        FreeIm(imc3);
        FreeIm(imrisinc);
        FreeIm(imridim);
        FreeIm(imdft);
        FreeIm(imdftfase);
        FreeIm(imdftout);
        FreeIm(imdftoutfase);
        FreeIm(im1dft);
        FreeIm(im1dftfase);
        FreeIm(im1dftrisinc);
        FreeIm(imr);
        FreeIm(img);
        FreeIm(imb);

        for (i=0; i<maxtile; i++)
            datiuscita[i] = datiout[i];

        // Si salva l'immagine risincronizzata
        //////////////////////////////////////

        offset = 0;
        for (i=0; i<nr; i++)
            for (j=0; j<nc; j++)
            {
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
            }

        FreeIm(imrisincout);

        return 3;	// Marchio non presente!!
    }


    if (tileusati>1)
    {
        // Nel 29991� elemento del vettore datiuscita
        // viene memorizzato se � stata fatta la somma
        // dei coefficenti delle dft
        ////////////////////////////
        datiout[29991] = 1;

        decoale(imdft, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

        nouniforme = 0;

        for (i=0; i<200; i++)
            nouniforme += BitLetti[i];

        datiout[30020]=nouniforme;

        if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti)) &&
            (nouniforme>0))
        {
            // THE WATERMARK HAS BEEN DETECTED
            FreeIm(imy);
            FreeIm(imc2);
            FreeIm(imc3);
            FreeIm(imyout);
            FreeIm(imrisinc);
            FreeIm(imridim);
            FreeIm(imdft);
            FreeIm(imdftfase);
            FreeIm(imdftout);
            FreeIm(imdftoutfase);
            FreeIm(im1dft);
            FreeIm(im1dftfase);
            FreeIm(im1dftrisinc);
            FreeIm(imr);
            FreeIm(img);
            FreeIm(imb);

            // NOTA: I bit decodificati vengono salvati alla fine
            //		 di BitLetti, ossia a partire dalla posizione
            //		 (BCH length)-(information message)

            int offs;
            offs = length_BCH - nbit;
            for (i=0; i<nbit; i++)
                bit[i] = BitLetti[i+offs];

            for (i=0; i<maxtile; i++)
                datiuscita[i] = datiout[i];

            // Si salva l'immagine risincronizzata
            //////////////////////////////////////
            offset = 0;
            for (i=0; i<nr; i++)
                for (j=0; j<nc; j++)
                {
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                }

            FreeIm(imrisincout);

            // LOG
            fclose(flog);

            return 0;	// OK!
        }

        // Nel 29993� elemento del vettore datiuscita
        // viene memorizzato se � stata fatta la somma
        // dei coefficenti delle dft delle imrisinc	
        //////////////////////////////////////////////
        datiout[29993] = 1;

        decoale(im1dftrisinc, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

        nouniforme = 0;

        for (i=0; i<200; i++)
            nouniforme += BitLetti[i];

        datiout[30020] = nouniforme;

        if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti)) &&
            (nouniforme>0))
        {
            // THE WATERMARK HAS BEEN DETECTED
            FreeIm(imy);
            FreeIm(imc2);
            FreeIm(imc3);
            FreeIm(imyout);
            FreeIm(imrisinc);
            FreeIm(imridim);
            FreeIm(imdft);
            FreeIm(imdftfase);
            FreeIm(imdftout);
            FreeIm(imdftoutfase);
            FreeIm(im1dft);
            FreeIm(im1dftfase);
            FreeIm(im1dftrisinc);
            FreeIm(imr);
            FreeIm(img);
            FreeIm(imb);

            // NOTA: I bit decodificati vengono salvati alla fine
            //		 di BitLetti, ossia a partire dalla posizione
            //		 (BCH length)-(information message)

            int offs;
            offs = length_BCH - nbit;
            for (i=0; i<nbit; i++)
                bit[i] = BitLetti[i+offs];

            for (i=0; i<maxtile; i++)
                datiuscita[i] = datiout[i];

            // Si salva l'immagine risincronizzata
            //////////////////////////////////////
            offset = 0;
            for (i=0; i<nr; i++)
                for (j=0; j<nc; j++)
                {
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                    buffimrisinc[offset] = imrisincout[i][j]; offset++;
                }

            FreeIm(imrisincout);

            // LOG
            fclose(flog);

            return 0;	// OK!
        }
    }  // Fine if (tileusati>1)


    if (flagRisincTotale)
    {
        // Nel 29995� elemento del vettore datiuscita
        // viene memorizzato se � stata fatta 
        // la risincronizzazione totale	
        ///////////////////////////////
        datiout[29995] = 1;

        imrisinctotale = AllocImFloat(nr, nc);


        // Nuovo ciclo decodifica
        /////////////////////////

        for (ciclo=1; ciclo<=nt; ciclo++)
        {
            // Inizializzazioni
            ///////////////////

            for (i=0; i<nre; i++)
                for (j=0; j<nce; j++)
                    imdft[i][j]=0.0;

            for (i=0; i<nre; i++)
                for (j=0; j<nce; j++)
                    imdftout[i][j]=0.0;

            if (vettoretile[ciclo] == 1)
            {
                for (i=0; i<nre; i++)
                    for (j=0; j<nce; j++)
                        imdft[i][j] = 0.0;

                datiout[30001] = 0;

                // Nel 30002� elemento del vettore datiuscita
                // viene memorizzato il ciclo della risincronizzazione totale
                /////////////////////////////////////////////////////////////
                datiout[30002] = ciclo;

                theta=(double) datiout[ciclo*2*nt];
                dx=(double) datiout[(ciclo*2*nt)+1];
                omega=(double) datiout[(ciclo*2*nt)+2];
                dy=(double) datiout[(ciclo*2*nt)+3];

                datiout[29996] = (double)theta;
                datiout[29997] = (double)dx;
                datiout[29998] = (double)omega;
                datiout[29999] = (double)dy;

                // Risincronizzazione dell'immagine totale
                //////////////////////////////////////////

                resyncro(imy, nr, nc, nr, nc, imrisinctotale);

                for (ntx=1; ntx<=ntr; ntx++)
                    for (nty=1; nty<=ntc; nty++)
                    {
                        for (i=0; i<nre; i++)
                            for (j=0; j<nce; j++)
                                imdftout[i][j] = 0.0;

                        for (i=0; i<nre; i++)
                            for (j=0; j<nce; j++)
                                imridim[i][j] = 0;

                        tileattuale=nty+(ntc*(ntx-1));

                        // Nel 30000� elemento del vettore datiuscita
                        // viene memorizzato il tile attuale esaminato
                        //////////////////////////////////////////////
                        datiout[30000] = tileattuale;


                        for (i=0; i<dtr; i++)
                            for (j=0; j<dtc; j++)
                                if ((i+((ntx-1)*dtr)<nr)&&(j+((nty-1)*dtc)<nc))
                                {
                                    dimvr=i+1;
                                    dimvc=j+1;
                                }

                        imyout = AllocImFloat(dimvr, dimvc);

                        for (i=0; i<dimvr; i++)
                            for (j=0; j<dimvc; j++)
                                imyout[i][j] =
                                        static_cast<float>(imrisinctotale[i+((ntx-1)*dtr)][j+((nty-1)*dtc)]);

                        resizefloat(imyout, imridim, dimvr, dimvc, nre, nce, 1);


                        // RIVELAZIONE DIRETTA DEL MARCHIO SENZA 
                        // RECUPERO SINCRONISMO O RIVELAZIONE DI PRIMO LIVELLO
                        //////////////////////////////////////////////////////

                        // Si fa la DFT dell'immagine estesa per recuperare il marchio
                        //////////////////////////////////////////////////////////////
                        FFT2D::dft2d(imridim, imdftout, imdftoutfase, nre, nce);


                        // Si memorizza la somma delle dft dei tile
                        ///////////////////////////////////////////
                        for (i=0; i<nre; i++)
                            for (j=0; j<nce; j++)
                                imdft[i][j] += static_cast<double>(imdftout[i][j]);

                        decoale(imdftout, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

                        nouniforme = 0;

                        for (i=0; i<200; i++)
                            nouniforme += BitLetti[i];

                        datiout[30020] = nouniforme;

                        if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti)) &&
                            (nouniforme>0))
                        {
                            // THE WATERMARK HAS BEEN DETECTED
                            FreeIm(imy);
                            FreeIm(imc2);
                            FreeIm(imc3);
                            FreeIm(imyout);
                            FreeIm(imrisinc);
                            FreeIm(imridim);
                            FreeIm(imdft);
                            FreeIm(imdftfase);
                            FreeIm(imdftout);
                            FreeIm(imdftoutfase);
                            FreeIm(im1dft);
                            FreeIm(im1dftfase);
                            FreeIm(im1dftrisinc);
                            FreeIm(imr);
                            FreeIm(img);
                            FreeIm(imb);

                            // NOTA: I bit decodificati vengono salvati alla fine
                            //		 di BitLetti, ossia a partire dalla posizione
                            //		 (BCH length)-(information message)

                            int offs;
                            offs = length_BCH - nbit;
                            for (i=0; i<nbit; i++)
                                bit[i] = BitLetti[i+offs];

                            for (i=0; i<maxtile; i++)
                                datiuscita[i] = datiout[i];

                            // Si salva l'immagine risincronizzata
                            //////////////////////////////////////
                            offset = 0;
                            for (i=0; i<nr; i++)
                                for (j=0; j<nc; j++)
                                {
                                    buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                                    buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                                    buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                                }

                            FreeIm(imrisinctotale);

                            // LOG
                            fclose(flog);

                            return 0;	// OK! Marchio rivelato
                        }

                        FreeIm(imyout);

                    }  // Fine ciclo decodifica dei tile

                // Nel 30001� elemento del vettore datiuscita
                // viene memorizzato se � stata fatta la somma
                // dei coefficenti delle dft nella risincronizzazione totale
                ////////////////////////////////////////////////////////////
                datiout[30001] = 1;

                decoale(imdft, nre, nce, d1, nd, seed, alpha,BitLetti, length_BCH);

                nouniforme=0;

                for (i=0; i<200; i++)
                    nouniforme += BitLetti[i];

                datiout[30020] = nouniforme;

                if ((BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti))&&
                    (nouniforme>0))
                {
                    // THE WATERMARK HAS BEEN DETECTED
                    FreeIm(imy);
                    FreeIm(imc2);
                    FreeIm(imc3);
                    FreeIm(imyout);
                    FreeIm(imrisinc);
                    FreeIm(imridim);
                    FreeIm(imdft);
                    FreeIm(imdftfase);
                    FreeIm(imdftout);
                    FreeIm(imdftoutfase);
                    FreeIm(im1dft);
                    FreeIm(im1dftfase);
                    FreeIm(im1dftrisinc);
                    FreeIm(imr);
                    FreeIm(img);
                    FreeIm(imb);

                    // NOTA: I bit decodificati vengono salvati alla fine
                    //		 di BitLetti, ossia a partire dalla posizione
                    //		 (BCH length)-(information message)

                    int offs;
                    offs = length_BCH - nbit;
                    for (i=0; i<nbit; i++)
                        bit[i] = BitLetti[i+offs];

                    for (i=0; i<maxtile; i++)
                        datiuscita[i] = datiout[i];

                    // Si salva l'immagine risincronizzata
                    //////////////////////////////////////
                    offset = 0;
                    for (i=0; i<nr; i++)
                        for (j=0; j<nc; j++)
                        {
                            buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                            buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                            buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                        }

                    FreeIm(imrisinctotale);

                    return 0;	// OK!
                }
            }  // Fine ciclo risinc. totale
        }  // Fine if (vettoretile[ciclo]==1)
    }  // Fine if (flagRisincTotale)

    FreeIm(imy);
    FreeIm(imc2);
    FreeIm(imc3);
    FreeIm(imrisinc);
    FreeIm(imridim);
    FreeIm(imdft);
    FreeIm(imdftfase);
    FreeIm(imdftout);
    FreeIm(imdftoutfase);
    FreeIm(im1dft);
    FreeIm(im1dftfase);
    FreeIm(im1dftrisinc);
    FreeIm(imr);
    FreeIm(img);
    FreeIm(imb);

    for (i=0; i<maxtile; i++)
        datiuscita[i] = datiout[i];

    if (datiout[29995]==1)
    {
        // Si salva l'immagine risincronizzata
        //////////////////////////////////////
        offset = 0;
        for (i=0; i<nr; i++)
            for (j=0; j<nc; j++)
            {
                buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
                buffimrisinc[offset] = imrisinctotale[i][j]; offset++;
            }

        FreeIm(imrisinctotale);
    }
    else
    {
        // Si salva l'immagine risincronizzata
        //////////////////////////////////////
        offset = 0;
        for (i=0; i<nr; i++)
            for (j=0; j<nc; j++)
            {
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
                buffimrisinc[offset] = imrisincout[i][j]; offset++;
            }

        FreeIm(imrisincout);
    }

    // LOG
    fclose(flog);

    return 2;	// Codice BCH non valido!!
}


//
//		Copyright (C) 2005 - Laboratorio Comunicazione ed Immagini (LCI)
//		----------------------------------------------------------------
//
//		* Image Watermarking DLL *
//
//		Version : "DLL per Laboratorio"
//
//
//							developed by 
//											Franco Bartolini
//											Mauro Barni
//											Roberto Caldelli 
//											Alessandro Piva
//											Alessia De Rosa
//                                          Massimiliano Corsini
//                                          Giovanni Fondelli
//


/***************************************************************/
//

/*
	PicRoutfloat(..)
	----------------

	Effettua la pesatura tra l'immagine originale (in float) 
	e l'immagine marchiata(in float) in base ai valori della 
	maschera di sensibilita'.

	Restituisce l'immagine marchiata finale (unsigned char)
		
	Argomenti: 

		img_orig:    nome dell'immagine originale;
		nr:          numero righe dell'immagine;
		nc:          numero colonne dell'immagine;
		img_mark:    nome dell'immagine marchiata;
		nomeimgmap:  nome dell'immagine mappa;
		nomeimgout:  nome dell'immagine marchiata finale;
*/

void ImgWat::PicRoutfloat(float **img_orig, int nr, int nc,
                                float **img_mark, float **img_map_flt, float **impic)
{
    int r, c;
    double max;

    // cerco il valore massimo della mappa

    max = 255.0;

    // Adesso modifico l'immagine finale secondo i valori di
    // mappatura

    for(r=0;r<nr;r++)
        for(c=0;c<nc;c++)
            impic[r][c] = (float)((double)img_orig[r][c]*((double)img_map_flt[r][c]/max) +
                                  (double)img_mark[r][c]*(max -(double)img_map_flt[r][c])/max);

}


/*
	resizefloat(..)
	---------------

	Routine che converte immagine (formato float) di dimensioni 
	qualsiasi r e c in immagine di dimensioni rq e cq (potenze del due)
	e viceversa.

	se ind = 1 estensione da rxc a rqxcq
	se ind != 1 ritaglio da rqxcq a rxc 
*/

void ImgWat::resizefloat(float **im_in, float **im_out, int r, int c,
                               int rq, int cq, int ind)
{
    int  i, j, nc, nr, nro, nco;
    double	x;

    nr=r;
    nc=c;
    nro=rq;
    nco=cq;

    x=0;
    for (i=0;i<nr;i++)
    {
        for (j=0;j<nc;x+=im_in[i][j],j++);
    }
    x/=(nr*nc);


    if (ind==1)
    {
        for(i =0; i < nro; i++)
        {
            for(j =0; j < nco; j++)
            {
                im_out[i][j] = (i < nr && j < nc ? im_in[i][j] :(float)x);
            }
        }
    }

}

/*
	DecimVarfloat(..)
	-----------------
	
	E' la routine per il calcolo delle varianza locale di
	blocchi dell'immagine. Calcola la varianza di ogni
	blocco shiftando di un pixel per volta la finestra, il massimo delle
	varianze cosi' calcolate e scrive una matrice con le varianze locali
	normalizzate rispetto a tale massimo.
	riceve in ingresso un file di float e restituisce la maschera in float

	Argomenti:
		nr = numero righe immagine
		nc = numero colonne immagine 
		win = dimensione della finestra scorrevole dell'immagine decimata (numero DISPARI)
*/

void ImgWat::DecimVarfloat(float **imc1, int nr, int nc,
                                 int win, float **img_map_flt)
{
    int         i,j,r,c,x,y;
    int         win2,ja,jb;
    double      add_med, sottr_med, add_vqm, sottr_vqm, var_max, wintot,mmedio;
    double      **vlocal, **vnorm, **mlocal, **vqm;
    double      aux;


    vlocal = AllocImDouble(nr, nc);  // matrice delle varianze locali
    vnorm = AllocImDouble(nr, nc);  // matrice delle varianze locali normalizzate
    mlocal= AllocImDouble(nr, nc);  // matrice delle medie locali
    vqm = AllocImDouble(nr, nc);  // matrice dei vqm locali


    // Calcolo la matrice delle varianze locali dell'immagine decimata
    win2 = win/2;
    wintot = win*win;

    for(r=0;r<nr;r++)
        for(c=0;c<nc;c++)
        {
            mlocal[r][c] = 0.0;	// Inizializzazione
            vqm[r][c] = 0.0;
            vlocal[r][c] = 0.0;
        }

    // Ciclo per pixel [r][0]
    for(r=0;r<nr;r++)
    {
        for(x = -win2;x <= win2;x++)
        {
            i=r+x;
            // Rendo l'imm simmetrica quando la finestra
            if(i>=0 && i<nr) 	// esce dai bordi 
            {

            }
            else
            {
                i=i<0 ? -i : (2*r-i);
            }
            for(y = -win2;y <= win2;y++)
            {
                j=y;
                j=j>0 ? j : -j;
                mlocal[r][0] += (double)imc1[i][j]/wintot;
                vqm[r][0] += (double)(imc1[i][j]*imc1[i][j])/wintot;
            }
        }
    }

    // Ciclo per tutti gli altri pixel
    for(r=0;r<nr;r++)
    {
        for(c=1;c<nc;c++)
        {
            add_med = sottr_med = add_vqm = sottr_vqm = 0.0;

            jb=c-win2-1;
            // jb: j Before, indice colonna pixel da sottrarre
            jb=jb>0 ? jb : -jb;
            ja=c+win2;
            // ja: j After, indice colonna pixel da aggiungere
            ja=ja<nc ? ja : (2*c-ja);

            for(x = -win2;x <= win2;x++)
            {
                i=r+x;
                // Rendo l'immagine simmetrica quando la finestra
                if(i>=0 && i<nr)	// esce dai bordi
                {

                }
                else
                {
                    i=i<0 ? -i : (2*r-i);
                }
                add_med += (double)imc1[i][ja];
                sottr_med += (double)imc1[i][jb];
                add_vqm += (double)(imc1[i][ja]*imc1[i][ja]);
                sottr_vqm += (double)(imc1[i][jb]*imc1[i][jb]);
            }
            mlocal[r][c] = mlocal[r][c-1] + add_med/wintot - sottr_med/wintot;
            vqm[r][c] = vqm[r][c-1] + add_vqm/wintot  - sottr_vqm/wintot;
        }
    }

    //  Cerco il max. delle varianze per normalizzare rispetto a questo
    var_max	 = 0.0;
    for(r=0;r<nr;r++)
        for(c=0;c<nc;c++)
        {
            aux = ((double)vqm[r][c]-(double)mlocal[r][c]*(double)mlocal[r][c]);
            if(aux < 0.0)
                aux = 0.0;

            vlocal[r][c] = (double)sqrt(aux);


            var_max = vlocal[r][c] > var_max ? vlocal[r][c] : var_max;
        }

    for(r=0;r<nr;r++)
        for(c=0;c<nc;c++)
        {
            vnorm[r][c] = vlocal[r][c]/var_max*2.0;
            if (vnorm[r][c] < 0.1)
                vnorm[r][c]=0.1;
            else if (vnorm[r][c] > 1.0)
                vnorm[r][c]=1.0;

        }


    // modifico l'immagine finale secondo i valori di vnorm[r][c]
    mmedio=0;
    for(r=0;r<nr;r++)
    {
        for(c=0;c<nc;c++)
        {
            vnorm[r][c] = 1.0-vnorm[r][c];
            img_map_flt[r][c] = (float)vnorm[r][c];
        }
    }

    // Libero aree di memoria
    FreeIm(vlocal);
    FreeIm(vqm);
    FreeIm(mlocal);
    FreeIm(vnorm);
}


/*
	rgb_to_crom (..)
	----------------
	
	Converte un'immagine in formato r, g, b, in un'immagine in 
	formato c1, c2, c3 dove c1 rappresenta la luminanza dell'immagine
	(calcolata come semplice somma tra le componenti r, g, b) e c2 e c3 
	sono le componenti di crominanza). 
	
	Se il flag fornito in ingresso e' 1 si ha la trasformazione 
	rgb->crom, se e' -1 si ha crom->rgb.

	La trasformazione rgb->crom da' in uscita tre file (*.flt) di float
	(evitando cosi' la propagazione di errori di arrotondamento ad unsigned char)
*/

void ImgWat::rgb_to_crom(unsigned char **imr, unsigned char **img,
                               unsigned char **imb, int nr, int nc, int flag,
                               float ** imc1, float **imc2, float ** imc3)
{
    int i,j;
    double red, green, blue;
    double c1, c2, c3;


    if(flag == 1)   /* rgb -> crom */
    {

        for(i = 0; i< nr; i++)
        {
            for(j = 0; j < nc; j++)
            {
                red = (double)imr[i][j];
                green = (double)img[i][j];
                blue = (double)imb[i][j];

                c1 = red + green + blue;

                if(c1 == 0)
                {
                    c2 = 0;
                    c3 = 0;
                }
                else
                {
                    c2 = blue / c1;
                    c3 = (2.0 * red + blue)/(2.0 * c1);
                    c1 /= 3.0;
                    c2 = c2 * 255.0;
                    c3 = c3 * 255.0;
                }

                if(c1 > 255.0)
                    c1 = 255.0;
                else if(c1 < 0.0)
                    c1 = 0.0;

                if(c2 > 255.0)
                    c2 = 255.0;
                else if(c2 < 0.0)
                    c2 = 0.0;

                if(c3 > 255.0)
                    c3 = 255.0;
                else if(c3 < 0.0)
                    c3 = 0.0;

                imc1[i][j] = (float)c1;
                imc2[i][j] = (float)c2;
                imc3[i][j] = (float)c3;
            }
        }
    }
    else if(flag == -1)
    {
        for(i = 0; i < nr; i++)
        {
            for(j = 0; j < nc; j++)
            {
                c1 = (double)imc1[i][j];
                c2 = (double)imc2[i][j];
                c3 = (double)imc3[i][j];

                c1 /= 85.0;
                c2 /= 255.0;
                c3 /= 255.0;

                red = c1 * (c3 - 0.5 * c2);
                green = c1 * (1.0 - c3 - (c2 / 2.0));
                blue = c1 * c2;

                red = red * 255.0;
                green = green * 255.0;
                blue = blue * 255.0;

                if(red > 255.0)
                {
                    red = 255.0;
                }
                else if(red < 0.0) red = 0.0;

                if(green > 255.0)
                {
                    green = 255.0;
                }
                else if(green < 0.0) green = 0.0;

                if(blue > 255.0)
                {
                    blue = 255.0;
                }
                else if(blue < 0.0) blue = 0.0;

                imr[i][j] = (unsigned char)(red + 0.5);
                img[i][j] = (unsigned char)(green + 0.5);
                imb[i][j] = (unsigned char)(blue + 0.5);
            }
        }
    }

}


/*
	sottrvetzone(..)
	----------------

	unknown function...

*/
double ImgWat::sottrvetzone(double *vet, double *mark, int camp)
{
    int m;
    double corr;

    m = camp;				// lunghezza del marchio
    corr = 0.0;

    // effettua la correlazione 
    int i;
    for(i = 0; i < m; i++)
    {
        corr += vet[i]*mark[i];
    }

    corr= corr/m;

    return(corr);
}


/*
	valmedzone(..)
	--------------
	estrae i coeff. dft marchiati e calcola la media del 
	valor assoluto, ponendo il risultato in uscita

	Argomenti:
		vet:      nome vettore di coeff. dft marchiati estratti;
		num_camp:  lunghezza del marchio;
*/
double ImgWat::valmedzone(double *vet,int num_camp)
{
    int m;
    int i;
    double media;

    m = num_camp;		// lunghezza del marchio
    media = 0.0;


    //	effettua la media
    for(i=0; i < m; i++)
    {
        if(vet[i]>0.0)
            media += vet[i];
        else
            media -= vet[i];
    }

    return(media);
}


/*
	valquadzone(..)	
	---------------

	Estrae i coeff. dft marchiati e calcola il
	media quadratica del valor assoluto, ponendo
	il risultato in uscita

	Argomenti:
		vet:      nome vettore di coeff. dft marchiati estratti;
		num_camp:  lunghezza del marchio;
*/
double ImgWat::valquadzone(double *vet, int num_camp)
{
    int m;
    int i;
    double mediaquad;

    m = num_camp;		// lunghezza del marchio
    mediaquad = 0.0;

    for(i=0; i < m; i++)
    {
        mediaquad += vet[i]*vet[i];
    }

    return(mediaquad);
}


/*
	codmarchio(..)
	-------------

	Codmarchio riceve in ingresso due stringhe, una di caratteri 
	e una di numeri, e le converte nei 4 semi dei generatori clcg
	di numeri pseudo-casuali. Restituisce in uscita il puntatore i
	al vettore con i 4 semi.
*/
void ImgWat::codmarchio(const char *campolett, const char *camponum, LONG8BYTE *s)
{
    int i, j;
    int *cl, *cn;    // vettori che contengono la codifica dei caratteri
    int **S;         // matrice con combinazione dei vettori cl e cn

    // Allocazione aree di memoria
    cl = new int [16];
    cn = new int [8];
    S = AllocImInt(4, 6);

    for(i = 0; i < 16; i ++)
    {
        switch (campolett[i])
        {
            case ' ': 		cl[i] = 0;
                break;

            case 'A': case 'a':	cl[i] = 1;
                break;

            case 'B': case 'b':	cl[i] = 2;
                break;

            case 'C': case 'c':     cl[i] = 3;
                break;

            case 'D': case 'd':     cl[i] = 4;
                break;

            case 'E': case 'e':     cl[i] = 5;
                break;

            case 'F': case 'f':     cl[i] = 6;
                break;

            case 'G': case 'g':     cl[i] = 7;
                break;

            case 'H': case 'h':     cl[i] = 8;
                break;

            case 'I': case 'i':     cl[i] = 9;
                break;

            case 'J': case 'j':     cl[i] = 10;
                break;

            case 'K': case 'k':     cl[i] = 11;
                break;

            case 'L': case 'l':     cl[i] = 12;
                break;

            case 'M': case 'm':     cl[i] = 13;
                break;

            case 'N': case 'n':     cl[i] = 14;
                break;

            case 'O': case 'o':     cl[i] = 15;
                break;

            case 'P': case 'p':     cl[i] = 16;
                break;

            case 'Q': case 'q':     cl[i] = 17;
                break;

            case 'R': case 'r':     cl[i] = 18;
                break;

            case 'S': case 's':     cl[i] = 19;
                break;

            case 'T': case 't':     cl[i] = 20;
                break;

            case 'U': case 'u':     cl[i] = 21;
                break;

            case 'V': case 'v':     cl[i] = 22;
                break;

            case 'W': case 'w':     cl[i] = 23;
                break;

            case 'X': case 'x':     cl[i] = 24;
                break;

            case 'Y': case 'y':     cl[i] = 25;
                break;

            case 'Z': case 'z':     cl[i] = 26;
                break;

            case '.': 	        cl[i] = 27;
                break;

            case '-':               cl[i] = 28;
                break;

            case '&':               cl[i] = 29;
                break;

            case '/':               cl[i] = 30;
                break;

            case '@':               cl[i] = 31;
                break;

            default: 		cl[i] = 0;
                break;
        }
    }

    for(i = 0; i < 8; i++)
    {
        switch (camponum[i])
        {
            case '0':		cn[i] = 0;
                break;

            case '1':               cn[i] = 1;
                break;

            case '2':               cn[i] = 2;
                break;

            case '3':               cn[i] = 3;
                break;

            case '4':               cn[i] = 4;
                break;

            case '5':               cn[i] = 5;
                break;

            case '6':               cn[i] = 6;
                break;

            case '7':               cn[i] = 7;
                break;

            case '8':               cn[i] = 8;
                break;

            case '9':               cn[i] = 9;
                break;

            case '.':               cn[i] = 10;
                break;

            case '/':               cn[i] = 11;
                break;

            case ',':               cn[i] = 12;
                break;

            case '$':               cn[i] = 13;
                break;

/*			case 'lira':                cn[i] = 14;
                                                break;
*/
            case ' ':               cn[i] = 15;
                break;

            default: 		cn[i] = 0;
                break;
        }
    }


    for(i = 0; i < 4; i ++)
    {
        for(j = 0; j < 4; j ++) S[i][j] = cl[i + 4 * j];

        for(j = 0; j < 2; j ++) S[i][j + 4] = cn[i + 4 * j];
    }

    for(i = 0; i < 4; i ++)
    {
        s[i] = S[i][0] + S[i][1] * (int)pow(2, 5) + S[i][2] * (int)pow(2, 10) + S[i][3] * (int)pow(2, 15) + S[i][4] * (int)pow(2, 20) + S[i][5] * (int)pow(2, 24) + 1;
    }

    FreeIm(S);
    delete [] cl;
    delete [] cn;
}


/*
	inizializzaza(..) e generatore()
	--------------------------------

	Funzioni per la generazione di sequenze pseudo-casuali
	di numeri reali uniformemente distribuiti tra [0,1]
*/

void ImgWat::inizializza(LONG8BYTE *s)
{
    int j;

    for(j = 0; j < 4; j ++)
    {
        semeiniziale[j] = s[j];
        semecorrente[j] = semeiniziale[j];
    }
}


double ImgWat::generatore()
{
    LONG8BYTE 	s;
    double	u;
    double	q;

    LONG8BYTE a[] = {45991, 207707, 138556, 49689};
    LONG8BYTE m[] = {2147483647, 2147483543, 2147483423, 2147483323};

    u = 0.0;

    s = semecorrente[0];
    s = (s * a[0]) % m[0];

    semecorrente[0] = s;
    u = u + 4.65661287524579692e-10 * s;
    q = 4.65661287524579692e-10;

    s = semecorrente[1];
    s = (s * a[1]) % m[1];

    semecorrente[1] = s;
    u = u - 4.65661310075985993e-10 * s;
    if(u < 0)
        u = u + 1.0;

    s = semecorrente[2];
    s = (s * a[2]) % m[2];

    semecorrente[2] = s;
    u = u + 4.65661336096842131e-10 * s;
    if(u >= 1.0)
        u = u - 1.0;

    s = semecorrente[3];
    s = (s * a[3]) % m[3];

    semecorrente[3] = s;
    u = u - 4.65661357780891134e-10 * s;
    if(u < 0)
        u = u + 1.0;
    return u;
}


/*
	addmark(..)
	-----------

    ADDMARK somma il marchio ai coefficienti dft

	Argomenti:
		dft: vettore di coeff. dft;
		mark: vettore dei coefficienti del marchio;
		num_camp:  lunghezza del marchio;
		peso:      coefficiente alfa di peso.
*/

void ImgWat::addmark(double *buff, double *mark, int num_camp, double peso)
{
    int n;
    int i;
    double alfa;

    n = num_camp;	// lunghezza del vettore
    alfa = peso;	// peso con cui sommo il marchio

    // aggiorna il valore di dft
    for(i=0; i<n; i++)
        buff[i] = buff[i]*(1.0 + alfa*mark[i]);
}


/*
                               ZONE 

	Questa funzione raggruppa i coefficienti DFT (appartenenti
	alle 16 parti in cui viene suddivisa la zona dell'immagine 
	che viene marchiata) nel vettore buff.
*/

double* ImgWat::zone(double **imdft, int nr, int nc, int diag0, int ndiag,
                           int detect, int *elem)
{
    int       m,i,j,d1,nd,max;
    int       elementi;
    double	  *ptr0, *ptr1, *ptr2, *ptr3, *ptr4, *ptr5, *ptr6, *ptr7;
    double	  *ptr8, *ptr9, *ptr10, *ptr11, *ptr12, *ptr13, *ptr14, *ptr15;
    double	  *buff;

    d1=diag0;
    nd=ndiag;

    // Calcolo dell' ordine dell' ultima diagonale

    max=d1+(nd-1);


    // Conteggio dei coeff. di ogni zona

    for(i=0; i<MAXZONE; i++) cont[i]=0;

    elementi = 0;

    for(m=d1;m<=max;m++)
    {
        for(i=1;i<m;i++)
        {
            if(m>=d1 && m<(d1+(max-d1)/2))
            {
                if (i>0 && i<(m/4))
                {
                    cont[0]++;
                    elementi++;
                    cont[8]++;
                    elementi++;
                }
                if (i>=(m/4) && i<(m/2))
                {
                    cont[1]++;
                    elementi++;
                    cont[9]++;
                    elementi++;
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    cont[2]++;
                    elementi++;
                    cont[10]++;
                    elementi++;
                }
                if (i>=((3*m)/4) && i<m)
                {
                    cont[3]++;
                    elementi++;
                    cont[11]++;
                    elementi++;
                }
            }

            if(m>=(d1+(max-d1)/2) && m<=max)
            {
                if (i>0 && i<(m/4))
                {
                    cont[4]++;
                    elementi++;
                    cont[12]++;
                    elementi++;
                }
                if (i>=(m/4) && i<(m/2))
                {
                    cont[5]++;
                    elementi++;
                    cont[13]++;
                    elementi++;
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    cont[6]++;
                    elementi++;
                    cont[14]++;
                    elementi++;
                }
                if (i>=((3*m)/4) && i<m)
                {
                    cont[7]++;
                    elementi++;
                    cont[15]++;
                    elementi++;
                }
            }
        }
    }


    // Copio i coeff.DFT delle diverse zone su un unico vettore

    buff = new double [elementi];

    ptr0 = buff;
    ptr1 = ptr0 + cont[0];
    ptr2 = ptr1 + cont[1];
    ptr3 = ptr2 + cont[2];
    ptr4 = ptr3 + cont[3];
    ptr5 = ptr4 + cont[4];
    ptr6 = ptr5 + cont[5];
    ptr7 = ptr6 + cont[6];
    ptr8 = ptr7 + cont[7];
    ptr9 = ptr8 + cont[8];
    ptr10 = ptr9 + cont[9];
    ptr11 = ptr10 + cont[10];
    ptr12 = ptr11 + cont[11];
    ptr13 = ptr12 + cont[12];
    ptr14 = ptr13 + cont[13];
    ptr15 = ptr14 + cont[14];

    for(m=d1;m<=max;m++)
    {
        for(i=1;i<m;i++)
        {
            if(m>=d1 && m<(d1+(max-d1)/2))
            {
                if (i>0 && i<(m/4))
                {
                    j=m-i;
                    *ptr0++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr8++ = imdft[i][j];
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    *ptr1++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr9++ = imdft[i][j];
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    *ptr2++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr10++ = imdft[i][j];
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    *ptr3++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr11++ = imdft[i][j];
                }
            }

            if(m>=(d1+(max-d1)/2) && m<=max)
            {
                if (i>0 && i<(m/4))
                {
                    j=m-i;
                    *ptr4++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr12++ = imdft[i][j];
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    *ptr5++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr13++ = imdft[i][j];
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    *ptr6++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr14++ = imdft[i][j];
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    *ptr7++ = imdft[i][j];
                    j=nc-1-m+i;
                    *ptr15++ = imdft[i][j];
                }
            }
        }
    }


    *elem = elementi;

    return buff;
}

/*
	antizone(..)
	------------
		 
	Questa funzione rimette i coefficienti marchiati al loro posto
*/


void ImgWat::antizone(double **imdft,int nr, int nc, int diag0, int ndiag, double *buff)
{
    int m,i,j,d1,nd,max,c[MAXZONE];

    d1=diag0;
    nd=ndiag;

    // Calcolo dell' ordine dell' ultima diagonale

    max=d1+(nd-1);


    // Costruzione del vett. contatore per il reinserimento dei coeff. marchiati

    c[0]=0;
    for(i=1; i<MAXZONE; i++)
        c[i] = c[i-1]+cont[i-1];

    // Reinserimento dei coeff. marchiati nella dft dell'immagine

    for(m=d1;m<=max;m++)
    {
        for(i=1;i<m;i++)
        {
            if(m>=d1 && m<(d1+(max-d1)/2))
            {
                if (i>0 && i<(m/4))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[0]];
                    imdft[nr-i][nc-j]=buff[c[0]];
                    c[0]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[8]];
                    imdft[nr-i][nc-j]=buff[c[8]];
                    c[8]++;
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[1]];
                    imdft[nr-i][nc-j]=buff[c[1]];
                    c[1]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[9]];
                    imdft[nr-i][nc-j]=buff[c[9]];
                    c[9]++;
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[2]];
                    imdft[nr-i][nc-j]=buff[c[2]];
                    c[2]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[10]];
                    imdft[nr-i][nc-j]=buff[c[10]];
                    c[10]++;
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    imdft[i][j]=buff[c[3]];
                    imdft[nr-i][nc-j]=buff[c[3]];
                    c[3]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[11]];
                    imdft[nr-i][nc-j]=buff[c[11]];
                    c[11]++;
                }
            }

            if(m>=(d1+(max-d1)/2) && m<=max)
            {
                if (i>0 && i<(m/4))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[4]];
                    imdft[nr-i][nc-j]=buff[c[4]];
                    c[4]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[12]];
                    imdft[nr-i][nc-j]=buff[c[12]];
                    c[12]++;
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[5]];
                    imdft[nr-i][nc-j]=buff[c[5]];
                    c[5]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[13]];
                    imdft[nr-i][nc-j]=buff[c[13]];
                    c[13]++;
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    imdft[i][j]=buff[c[6]];
                    imdft[nr-i][nc-j]=buff[c[6]];
                    c[6]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[14]];
                    imdft[nr-i][nc-j]=buff[c[14]];
                    c[14]++;
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    imdft[i][j]=buff[c[7]];
                    imdft[nr-i][nc-j]=buff[c[7]];
                    c[7]++;
                    j=nc-1-m+i;
                    imdft[i][j]=buff[c[15]];
                    imdft[nr-i][nc-j]=buff[c[15]];
                    c[15]++;
                }
            }
        }
    }

}

/* 
	dgamma(..)
	----------

	Gamma function in double precision

	Added by CORMAX,	14/MAR/2001

*/

double ImgWat::dgamma(double x)
{
    int k, n;
    double w, y;

    n = x < 1.5 ? -((int) (2.5 - x)) : (int) (x - 1.5);
    w = x - (n + 2);
    y = ((((((((((((-1.99542863674e-7 * w + 1.337767384067e-6) * w -
                   2.591225267689e-6) * w - 1.7545539395205e-5) * w +
                 1.45596568617526e-4) * w - 3.60837876648255e-4) * w -
               8.04329819255744e-4) * w + 0.008023273027855346) * w -
             0.017645244547851414) * w - 0.024552490005641278) * w +
           0.19109110138763841) * w - 0.233093736421782878) * w -
         0.422784335098466784) * w + 0.99999999999999999;
    if (n > 0) {
        w = x - 1;
        for (k = 2; k <= n; k++) {
            w *= x - k;
        }
    } else {
        w = 1;
        for (k = 0; k > n; k--) {
            y *= x - k;
        }
    }
    return w / y;
}



/*
	mlfunc(..)
	----------

	Risolve numericamente l'equazione del criterio ML 
	per calcolare i parametri della p.d.f. Weibull
*/

void ImgWat::mlfunc(double *buff,int nrfile,int niteraz)
{
    int i,k,nr,ny;
    double a,b,c,beta0,beta1,beta,dbeta,prec,alfa,media,var,pu;
    double *y;


    nr=nrfile;
    beta0=0.7;
    beta1=2.3;
    ny=niteraz;

    dbeta=beta1-beta0;
    prec=dbeta/ny;

    y = new double [ny];

    // Calcolo elementi della matrice di uscita

    for(k=0;k<ny;k++)
    {
        a=b=c=0.0;
        beta=beta0+prec*k;

        for(i=0;i<nr;i++)
        {
            a+=pow(buff[i],beta);
            b+=pow(buff[i],beta)*log(buff[i]);
            if (buff[i] == 0.0)
            {
                c+= -23.03;
            }
            else c+=log(buff[i]);
        }

        if(a<0.000001)
        {
            a=0.000001;
        }

        y[k]=c/nr-b/a+1.0/beta;
    }


    // Calcolo dei valori minimi e dei corrispondenti alfa e beta
    // per ogni riga

    k=0;
    z[0]=fabs(y[k]);
    z[1]=beta0;
    for(k=1;k<ny;k++)
    {
        if(fabs(y[k])<z[0])
        {
            z[0]=fabs(y[k]);
            z[1]=beta0+prec*k;
        }
    }
    alfa=0.0;
    for(i=0;i<nr;i++)
        alfa+=pow((buff[i]),(z[1]));
    alfa=alfa/(double)nr;
    z[2]=pow(alfa,(1.0/(z[1])));


    /* OCCHIO!! BISOGNA INFORMARSI SU QUESTA FUNZIONE Gamma(..)

        ANCHE SE z[3] e z[4] NON VENGONO MAI UTILIZZATI!!!!

    */

    pu = dgamma(1.0+1.0/z[1]);

    media=z[2]*pu;

    pu = dgamma(1.0+2.0/z[1]);

    var=pow(z[2],2.0)*pu-pow(media,2.0);
    z[3]=media;
    z[4]=var;

    delete [] y;
}


/*
	syncro(..)
	----------

	Esegue la marchiatura dell'immagine con una scacchiera 
	di periodo e ampiezza che vengono fornite come 
	parametri di ingresso.

	L'immagine di ingresso e' in formato float.    
*/

void ImgWat::syncro(float **imc1, float **im_M, int nr, int nc,
                          int periodo, int ampiezza)
{
    // Si inserisce la griglia di sinusoidi
    Marchia(imc1,im_M,nr,nc,ampiezza,periodo);
}

/*
	decsyncro(..) 
	-------------
	
	Rileva la griglia di sinusoidi.

	Riceve come parametri la dft dell'immagine estesa,
	le dimensioni dell'immagine	estesa e il periodo delle sinusoidi.    
*/

int ImgWat::decsyncro(double **imdft, int nre, int nce, int periodo)
{
    int  i, j, Det;
    float **imdft1;
    struct Scacco *ParametersPtr, Parameters;

    imdft1 = AllocImFloat(nre, nce);

    for(i=0;i<nre;i++)
        for(j=0;j<nce;j++)
        {
            imdft1[i][j] = (float)imdft[i][j];
        }

    // By CORMAX, RICONTROLLARE QUESTA INIZIALIZZAZIONE...

    // � necessaria una inizializzazione 
    ParametersPtr=&Parameters;
    ParametersPtr->Alfa=ParametersPtr->Beta=0.0;
    ParametersPtr->DeltaX=ParametersPtr->DeltaY=100.0;
    ParametersPtr->Periodo=(double)periodo;

    /* Si rileva la griglia */

    Det = Detect(imdft1, nre, nce, ParametersPtr,periodo);
    if(Det == 1000)
    {
        return 1000;
    }

    theta = ParametersPtr->Alfa;
    omega = ParametersPtr->Beta;
    dx = (ParametersPtr->DeltaX)/100.0;
    dy = (ParametersPtr->DeltaY)/100.0;

    FreeIm(imdft1) ;

    return 1;
}

/*
	resyncro(..)
	------------

	risincronizza un'immagine
*/

void ImgWat::resyncro(float **imgin, int nr, int nc,
                            int n1, int n2, float **imgout)
{
    int i, j, mi, li;
    double lf, mf;
    double a, b, c, d, app, app1;
    double uno, due, tre, quattro;


    app1 = dx;
    dx = dy;
    dy = app1;

    app = theta;
    theta = omega;
    omega = app;

    // Calcolo coefficienti della trasformazione geometrica
    a = dx*cosd(theta)*cosd(omega) - dy*sind(theta)*sind(omega);
    b = dx*cosd(theta)*sind(omega) + dy*sind(theta)*cosd(omega);
    c = -dx*sind(theta)*cosd(omega) - dy*cosd(theta)*sind(omega);
    d = -dx*sind(theta)*sind(omega) + dy*cosd(theta)*cosd(omega);

    int ic,jc;

    for(i = 0; i < n1; i ++)
        for(j = 0; j < n2; j++)
        {
            // Si ruota rispetto al centro dell'immagine originale
            ic = i - (nr-1)/2;
            jc = j - (nc-1)/2;

            lf = a*(double)ic + b*(double)jc;
            mf = c*(double)ic + d*(double)jc;

            li = (int)lf;
            mi = (int)mf;

            // Rimappo le coordinate sull'immagine in ingresso
            li += (nr-1)/2;
            mi += (nc-1)/2;

            // Scrivo sull'immagine ESTESA !!
            if((li < 0) || (mi < 0) || (li >= (nr-1) ) || (mi >= (nc-1) ))
            {
                imgout[i][j] = 0;
            }
            else
            {
                uno = (double)imgin[li+1][mi];
                due = (double)imgin[li+1][mi+1];
                tre = (double)imgin[li][mi+1];
                quattro = (double)imgin[li][mi];

                // Vedi (li,mi)
                lf += (nr-1)/2;
                mf += (nc-1)/2;

                // Bilinear Filtering
                if((li == nr-1) && (mi == nc-1))
                    imgout[i][j] = (float)quattro;
                else if(li == nr-1)
                    imgout[i][j] = (float)((quattro*(1.0 - (mf - (int)mf)) + tre*(mf - (int)mf)));
                else if(mi == nc-1)
                    imgout[i][j] = (float)((quattro*(1.0 - (lf - (int)lf)) + uno*(lf - (int)lf)));
                else
                {
                    imgout[i][j] = (float)(((uno*(1.0 - (mf - (int)mf)) + due*(mf - (int)mf))*(lf - (int)lf) +
                                            (quattro*(1.0 - (mf - (int)mf)) + tre*(mf - (int)mf))*(1.0 - (lf - (int)lf))));
                }

                // imgout[i][j] = imgin[li][mi];   <-- NO BILINEAR


                // MODIFICATO by CORMAX,	13/04/2001
                //
                // Aggiunto le successive due linee di controllo
                // sui valori di imgout[][]
                if (imgout[i][j] < 0.0) imgout[i][j] = 0.0;
                if (imgout[i][j] > 255.0) imgout[i][j] = 255.0;
            }
        }
}


/*

	Detect(..)
	----------

	Detect costruisce e utilizza una matrice d'appoggio che contiene 
	i coefficient del modulo dell'immagine, ma opportunamente filtrati.
	Fornisce a Decoder.c i parametri cercati.

*/

int ImgWat::Detect(float **ImPtr,int nr, int nc,
                         struct Scacco *GrigliaPtr, int period)
{
    int Dim, Est;
    double media,Conf;
    struct Point PtR,*PtrPtR,PtL,*PtrPtL;
    struct tnode *root;
    float **Im_F_out;

    PtrPtL=&PtL;
    PtrPtR=&PtR;


    Im_F_out = AllocImFloat(nr, nc); //Alloca memoria per matr. filtr.

    media=Media(ImPtr,Im_F_out,nr,nc);// "Filtra" i coeff. e ne calcola la media totale
    Conf=Threshold(Im_F_out,nc,nr,media);//Calcola la soglia per determinare i punti "salienti"
    root=MaxPointS(ImPtr,Im_F_out,nc,nr,Conf);//Costruisce albero ordinato
    Est = EstraiMax(root,PtrPtR);//Rileva il primo Massimo

    if(Est == 1000)
    {
        return 1000;
    }

    Est = EstraiMax(root,PtrPtL);//Rileva il secondo Massimo
    if(Est == 1000)
    {
        return 1000;
    }

    // Controlla che le frequenze rilevate rispondano ai requisiti
    // e in caso negativo ne ricerca un'altro
    Controllo(root,nr,nc,PtrPtR,PtrPtL);

    // Decide se si � in presenza di pattern con stripes o con reticolo
    if(3*(PtrPtL->Val)<(PtrPtR->Val))
    {
        Dim=1;
    }
    else
    {
        Dim=2;
    }

    // In base alle frequenze rilevate e al tipo di pattern 
    // determina i parametri
    TrovaDelta(ImPtr,PtrPtL,PtrPtR,GrigliaPtr,nr,nc,Dim,period);
    FreeTree(root);
    FreeIm(Im_F_out) ;

    return 1;
}

/*
	Media(..)
	---------

	Si costruisce una matrice Im_out contenete dei valori legati ai coefficienti del 
	modulo dell'immagine originale Im_in.
	Ogni elemento � uguale al coefficiente di modulo corrispondente diviso per 
	la media dei valori appartenenti all'intorno dello stesso.
	L'intorno � definito tramite  una finestra di dimensioni  DIMFILT x DIMFILT.
	La funzione restituisce la media complessiva della matrice Im_out.
*/

double ImgWat::Media(float** Im_in,float** Im_out,int nr,int nc)
{
    int		index,rr,cc,j,Norm, k,h;
    float	ColumnWindowSum,WindowSum,*G,Value;
    double	Mean;


    // Creo una vettore G di dimensioni DIMFILT ho quindi un numero 
    // di colonne pari al numero di colonne della finestra utilizzata 
    // per "filtrare" i coefficienti. 

    // Alloco memoria per il vettore
    G = new float [DIMFILT];

    // Inizializzazione delle variabili a zero
    Mean=0;
    index=0;
    Norm=0;
    WindowSum=0;
    for(rr=0;rr<DIMFILT;rr++) G[rr]=0;

    /*
        Si utilizza il vettore G organizzato come una struttura circolare
        anzich� la finestra in modo tale da	sfruttare le operazioni gi� fatte 
        precedentemente.Esso sar� costruito in modo tale che l'elemento i-esimo 
        � uguale alla somma degli elementi 	appartenenti all'ipotetica finestra 
        nella colonna i-esima. La somma su tutta la finestra � uguale alla somma 
        di tutti gli elementi del vettore, o, ancora meglio, � uguale alla Somma
        della finestra precedente pi� la somma della nuova colonna che subentra,
        meno la somma di quella che non	deve pi� essere considerata. 
        Al fine di ottenere la media ogni elemento della matrice viene infine
        normalizzato rispetto alle dimensioni della finestra 
    */

    for(cc=-DIMFILT;cc<0;cc++)		// Porta a "regime" il sistema
    {
        ColumnWindowSum=0;
        for(j=-DIMFILT2;j<DIMFILT2+1;j++)
        {
            ColumnWindowSum+=Im_in[(j+nr)%nr][(cc+nc)%nc];	// Somma degli elementi
            // della colonna
        }
        G[index]=ColumnWindowSum;		// Memorizza la somma della colonna
        WindowSum+=G[index];			// Aggiunge la nuova somma a quella totale
        index=(index+1)%DIMFILT;		// Incrementa l'indice del vettore circolare
        WindowSum-=G[index];			// Elimina la somma della colonna che uscir� alla
        //   successiva iterazione
    }

    k = 0;
    h = 0;
    for(rr=0;rr<nr/2;rr++)
    {
        for(cc=0;cc<nc;cc++)
        {
            ColumnWindowSum=0;
            h++;
            for(j=-DIMFILT2;j<DIMFILT2+1;j++)
            {
                k++;
                ColumnWindowSum+=Im_in[(rr+j+nr)%nr][cc];
            }

            G[index]=ColumnWindowSum;
            WindowSum+=G[index];

            Value=Im_in[rr][cc]*DIMFILT*DIMFILT/WindowSum;	// Ricavo l'elemento da
            Im_out[rr][cc]=Value;							// memorizzare in Im_out
            if(((double)rr/nr)>EPS2 || (double)(fabs((cc+nc/2)%nc-nc/2.0)/nc)>EPS2)
            {
                Mean+=Value;	// Non considero nella somma complessiva quei valori
                Norm++;			// troppo vicini agli assi.
            }

            index=(index+1)%DIMFILT;
            WindowSum-=G[index];

        }
    }

    delete [] G;	// Rilascio memoria utilizzata da G

    Mean=Mean/(double)Norm;		// Calcolo la media rispetto agli elementi effettivamente considerati
    return(Mean);
}


/*
	Threshold(..)
	-------------

	Threshold calcola la varianza della matrice fornita e in base alla media mean fornisce
	un valore di soglia.
*/

double ImgWat::Threshold(float **Im,int nc,int nr,double mean)
{
    int rr,cc,Norm;
    double VarTot,Soglia;

    Norm=0;
    VarTot=0;
    for(rr=0;rr<nr/2;rr++)
    {
        for(cc=0;cc<nc;cc++)
        {
            if(((double)rr/(double)nr)>EPS2 || (double)(fabs((cc+nc/2)%nc-nc/2.)/nc)>EPS2)
            {
                VarTot+=sqrt(pow((double)Im[rr][cc]-mean,2.0));
                Norm++;
            }
        }
    }
    Soglia=mean+7.0*VarTot/(double)Norm;
    return(Soglia);
}


/*
	MaxPointS(..)
	-------------

	MaxPointS fornisce il puntatore alla radice di un 
	albero ordinato secondo il valore del modulo dell'immagine 
	originale costituito dagli elementi che superano i 
	requisiti della soglia
*/

ImgWat::tnode * ImgWat::MaxPointS(float **Im,float **ImN,int nc, int nr,double Conf)
{
    struct tnode *root;
    int rr,cc;
    float Val;

    root=NULL;
    root=addtree(root,0,0,0);// Costruisce radice albero
    for(rr=0;rr<nr/2;rr++)
    {
        for(cc=0;cc<nc;cc++)
        {
            if(((Val=ImN[rr][cc])>Conf)&& ((((double)rr/(double)nr)>EPS2) && (fabs((cc+nc/2)%nc-nc/2.0)/nc)>EPS2))
            {
                root=addtree(root,Im[rr][cc],rr,cc); // Aggiunge foglie all'albero
            }
        }
    }

    return root;
}


/*
	addtree(..)
	-----------

	Costruisce effettivamente l'albero che pu� essere percorso nei 
	due versi: 	Radice -:- Foglie  o Foglie -:-Radice.
	L'ordine � da Sinistra verso Destra (Dal minore a maggiore)
*/

ImgWat::tnode *ImgWat::addtree(struct tnode *p,float Val,int Riga,int Col)
{
    if(p==NULL)
    {
        p=talloc();
        p->Pt.Val=Val;
        p->Pt.Riga=Riga;
        p->Pt.Col=Col;
        p->left=p->right=p->father=NULL;
        p->flag=0;
    }
    else if(Val<(p->Pt.Val))
    {
        p->left = addtree(p->left,Val,Riga,Col);
        (p->left)->father=p;
    }
    else
    {
        p->right = addtree(p->right,Val,Riga,Col);
        (p->right)->father=p;

    }
    return p;
}

/*
	talloc(..)
	----------

	Alloca memoria per ogni foglia dell'albero.
*/
ImgWat::tnode *ImgWat::talloc(void)
{
    return new tnode;
}

/*
	EstraiMax(..)
	-------------

	Copia in una struttura punto gli i campi della foglia con valore massimo, che non sia ancora
	stata scelta.    p->flag indica lo stato del nodo: 
		0 : Se nodo da attraversare verso destra, se foglia da copiare.
		1 : Nodo da copiare; a destra il ramo � secco cio� gi� utilizzato, da non riattraversare. 
		2 : Indica che l'attraversamento deve essere verso sinistra, a destra il ramo � secco.
		3 : Ramo secco.
*/
int ImgWat::EstraiMax(struct tnode* p,struct Point *PtrPt)
{
    if(p->flag == 3)						// Il ramo � secco
    {
        if(p->father!=NULL)
        {
            ((p->father)->flag)++;		// Ricerco a partire dal nodo padre
            EstraiMax(p->father,PtrPt);	// a cui ho aumentato lo stato

        }
        else
        {
            return 1000;
        }								// l'attraversamento dell'albero non ha dato esito
    }
    else if (p->flag == 2)				// Devo attraversare a sinistra se � possibile
    {
        if(p->left!=NULL)
        {
            EstraiMax(p->left,PtrPt);
        }
        else
        {
            (p->flag)++	;					// Non � possibile l'attraversamento a sx, quindi
            if(p->father!=NULL)				// aumento lo stato del nodo,
            {
                ((p->father)->flag)++;		// del nodo padre e
                EstraiMax(p->father,PtrPt);	// ricerco a partire dal nodo padre
            }
            else
            {
                return 1000;
            }
        }
    }
    else if (p->flag == 1)			// Memorizzo nodo
    {
        PtrPt->Riga=p->Pt.Riga;
        PtrPt->Col=p->Pt.Col;
        PtrPt->Val=p->Pt.Val;
        if(p->left!=NULL)			// Setto stato opportuno per il nodo
        {
            p->flag=2;
        }
        else
        {
            p->flag=3;
            if(p->father!=NULL)
            {
                ((p->father)->flag)++;// Aumento lo stato del nodo padre
            }
        }
    }
    else if(p->flag==0)
    {
        if(p->right!=NULL)					// Nodo  da attraversare
        {
            EstraiMax(p->right,PtrPt);
        }
        else								// Foglia da copiare
        {
            PtrPt->Riga=p->Pt.Riga;
            PtrPt->Col=p->Pt.Col;
            PtrPt->Val=p->Pt.Val;
            if(p->left!=NULL)				// Setto stato opportuno per la foglia
            {
                p->flag=2;
            }
            else
            {
                p->flag=3;
                if(p->father!=NULL)
                {
                    ((p->father)->flag)++;	// Aumento lo stato del nodo padre
                }
            }
        }
    }
    else
    {
        // Se si � qui c'� qualche errore
        exit(EXIT_FAILURE);
    }

    return 1;
}

/*
	Controllo(..)
	-------------

	Controlla se i punti individuati sono troppo vicini
*/
void ImgWat::Controllo(struct tnode* p,int nr,int nc,
                             struct Point* PtrPtR,struct Point* PtrPtL)
{
    int rrL,ccL,rrR,ccR;

    rrL=PtrPtL->Riga;
    ccL=PtrPtL->Col;
    rrR=PtrPtR->Riga;
    ccR=PtrPtR->Col;

    if(((fabs(((double)(rrL-rrR))/(double)nr)<EPS2) && ((fabs(((ccR+nc/2)%nc-nc/2.0)-((ccL+nc/2)%nc-nc/2.0))/nc)<EPS2)))
    {
        EstraiMax(p,PtrPtL);
        Controllo(p,nr,nc,PtrPtR,PtrPtL);
    }
}


/*
	TrovaDelta(..)
	--------------

	Determina le frequenze relative alle righe e alle colonne 
	individuate utilizzando una interpolazione parabolica. 
	Nel caso del Reticolo chiamando Trasformazione() ottiene i 
	parametri cercati, nel caso delle Stripes li ottiene direttamente. 
*/
void ImgWat::TrovaDelta(float **ImPtr,struct Point *BPtrL,struct Point *BPtrR,
                              struct Scacco *GPtr, int nr,int nc,int Dim, int period)
{
    double	FMx1,FMy1,FMx2,FMy2,tmp;
    double	Freq_cL,Freq_rU,Freq_c,Freq_r,Freq_cR,Freq_rB;
    int		cc,rr,ccL,ccR,rrU,rrB;

    cc=(BPtrR->Col);
    rr=(BPtrR->Riga);
    if(cc!=0)	ccL=cc-1;	else 	ccL=nc-1;
    if(cc!=nc)	ccR=cc+1;	else	ccR=0;
    if(rr!=0)	rrB=rr-1;	else	rrB=nr-1;
    if(rr!=nr)	rrU=rr+1;	else	rrU=0;

    Freq_c=((cc+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna
    Freq_r=((rr+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga
    Freq_cL=((ccL+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna adiacente a sinistra
    Freq_rB=((rrB+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga adiacente in basso
    Freq_cR=((ccR+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna adiacente a destra
    Freq_rU=((rrU+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga adiacente in in alto


    // Interpolazione parabolica separatamente per ogni dimensione

    FMx1=((Freq_c+Freq_cR)*(ImPtr[rr][ccL])-2*(Freq_cL+Freq_cR)*(ImPtr[rr][cc])
          +(Freq_c+Freq_cL)*(ImPtr[rr][ccR]))
         /(2*((ImPtr[rr][ccL])+(ImPtr[rr][ccR])-2*(ImPtr[rr][cc])));
    FMy1=((Freq_r+Freq_rU)*(ImPtr[rrB][cc])-2*(Freq_rB+Freq_rU)*(ImPtr[rr][cc])
          +(Freq_r+Freq_rB)*(ImPtr[rrU][cc]))
         /(2*((ImPtr[rrB][cc])+(ImPtr[rrU][cc])-2*(ImPtr[rr][cc])));


    if(Dim==1)	// Se Stripes
    {
        GPtr->DeltaX=100.0/((double)period * sqrt(pow(FMx1,2.0)+pow(FMy1,2.0)));
        GPtr->DeltaY=GPtr->DeltaX;
        GPtr->Alfa=0.0;
        GPtr->Beta=-((180.0 * atan2(FMy1,FMx1)/PI)-45.0);

    }
    else		// Se Reticolo
    {

        cc=(BPtrL->Col);
        rr=(BPtrL->Riga);

        if(cc!=0)	ccL=cc-1;	else 	ccL=nc-1;
        if(cc!=nc)	ccR=cc+1;	else	ccR=0;
        if(rr!=0)	rrB=rr-1;	else	rrB=nr-1;
        if(rr!=nr)	rrU=rr+1;	else	rrU=0;

        Freq_c=((cc+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna
        Freq_r=((rr+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga
        Freq_cL=((ccL+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna adiacente a sinistra
        Freq_rB=((rrB+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga adiacente in basso
        Freq_cR=((ccR+nc/2)%nc-nc/2.0)/nc;	//Frequenza della colonna adiacente a destra
        Freq_rU=((rrU+nr/2)%nr-nr/2.0)/nr;	//Frequenza della riga adiacente in in alto

        // Interpolazione parabolica separatamente per ogni dimensione

        FMx2=((Freq_c+Freq_cR)*(ImPtr[rr][ccL])-2*(Freq_cL+Freq_cR)*(ImPtr[rr][cc])
              +(Freq_c+Freq_cL)*(ImPtr[rr][ccR]))
             /(2*((ImPtr[rr][ccL])+(ImPtr[rr][ccR])-2*(ImPtr[rr][cc])));
        FMy2=((Freq_r+Freq_rU)*(ImPtr[rrB][cc])-2*(Freq_rB+Freq_rU)*(ImPtr[rr][cc])
              +(Freq_r+Freq_rB)*(ImPtr[rrU][cc]))
             /(2*((ImPtr[rrB][cc])+(ImPtr[rrU][cc])-2*(ImPtr[rr][cc])));

        if(FMx1<FMx2)
        {
            tmp=FMx1;
            FMx1=FMx2;
            FMx2=tmp;
            tmp=FMy1;
            FMy1=FMy2;
            FMy2=tmp;
        }
        Trasformazione(GPtr,FMx1,FMy1,FMx2,FMy2);	// Trovo i parametri
    }

}


/*
	Trasformazione(..)
	------------------

	Ricava i parametri della prima Rotazione con angolo Alfa ,  
	del ridimensinamento percentuale DeltaX e DeltaY
	e dell'ultima Rotazione con angolo Beta. 
*/
void ImgWat::Trasformazione(struct Scacco *GPtr,
                                  double FMx1, double FMy1, double FMx2, double FMy2)
{
    double d1,d2,d3,d4,Alfa,DeltaX,DeltaY,Pinc,Zs,Zd,temp;
    double	Beta,DiagDecQ,DiagCreQ,D2mC2,D2pC2,Periodo;

    d1=FMx1-FMx2;
    d3=FMx1+FMx2;
    d2=FMy1-FMy2;
    d4=FMy1+FMy2;
    DiagDecQ=pow(FMx1,2.0)+pow(FMy1,2.0);	// Quadrato della Diagonale Decrescente
    DiagCreQ=pow(FMx2,2.0)+pow(FMy2,2.0);	// Quadrato della Diagonale Crescente
    D2mC2=DiagDecQ-DiagCreQ;
    D2pC2=DiagDecQ+DiagCreQ;
    Pinc=FMx1*FMx2+FMy1*FMy2;	// Prodotto Incrociato
    Periodo=GPtr->Periodo;


    /* Determinazione di Alfa,DeltaX e DeltaY */

    if(fabs(D2mC2)<EPS && fabs(Pinc)<EPS)
    {
        Alfa=0.0;	// Caso in cui perdo un grado di libert� in rotazione
        DeltaX=(sqrt(D2pC2))*Periodo;
        DeltaY=DeltaX;
    }
    else
    {
        Alfa=atan2(D2mC2/2.0,-Pinc)/2.0;
        Zs=D2pC2;
        Zd=sqrt(D2mC2*D2mC2+4.0*Pinc*Pinc);

        DeltaX=sqrt(Zs+Zd)*Periodo;
        DeltaY=sqrt(Zs-Zd)*Periodo;
    }

    //	Visualizzazione di Alfa fra -90� e + 90�

    if(Alfa>PI/4.0)
    {
        Alfa=Alfa-PI/2.0;
        temp=DeltaX;
        DeltaX=DeltaY;
        DeltaY=temp;
    }
    else if(Alfa<-PI/4.0)
    {
        Alfa=Alfa+PI/2.0;
        temp=DeltaX;
        DeltaX=DeltaY;
        DeltaY=temp;
    }

    // Determinazione di Beta
    Beta=atan2(-d1*sin(Alfa)+d3*cos(Alfa),-d2*sin(Alfa)+d4*cos(Alfa));

    // Memorizzazione dei parametri
    GPtr->Alfa=-180.0*Alfa/PI;
    GPtr->Beta=-180.0*Beta/PI;
    GPtr->DeltaX=200.0/DeltaX;
    GPtr->DeltaY=200.0/DeltaY;
}


/*
	FreeTree(..)
	------------

	Libera la memoria allocata per l'albero con precedenta 
	alle foglie 
*/
void ImgWat::FreeTree(struct tnode *p)
{
    if (p==NULL)
    {
    }
    else
    {
        FreeTree(p->left);
        FreeTree(p->right);
        delete p;
        p = NULL; // just in case...
    }
}


/*
	Marchia(..)
	-----------

	Sovrappone la scacchiera all'immagine.
*/
void ImgWat::Marchia(float **Im,float **Im_f, int nr ,int nc,int na,int nT)
{
    int rr,cc,Periodo;
    float tmp,val;

    val=(float)na;
    Periodo=nT;

    // MODIFICATO by CORMAX,	13/04/2001
    //
    // Tolto MINVALUE, MAXVALUE. 
    // Il controllo sui valori di Im_f[][] avviene sulle ultime 
    // due linee prima della chiusura dei cicli for .

    for(rr=0;rr<nr;rr++)
    {
        for(cc=0;cc<nc;cc++)
        {
            tmp = Im[rr][cc];
            if(((cc%Periodo<(Periodo/2))&&(rr%Periodo<(Periodo/2)))||
               ((cc%Periodo>(Periodo/2)-1)&&(rr%Periodo>(Periodo/2)-1)))
            {
                Im_f[rr][cc] = (float)(tmp+val);
            }
            else
            {
                Im_f[rr][cc] = (float)(tmp-val);
            }

            if (Im_f[rr][cc] < 0.0) Im_f[rr][cc] = 0.0;
            if (Im_f[rr][cc] > 255.0) Im_f[rr][cc] = 255.0;
        }
    }
}


/********************************************************************/

//Q_EXPORT_PLUGIN(ImgWat);

