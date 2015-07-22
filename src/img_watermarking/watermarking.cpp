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
    float   **imyout;			// immagine
    double  **imdft;		// immagine della DFT
    double  **imdftfase;	// immagine della fase della DFT
    float   **imidft;		// immagine della IDFT

    imyout = AllocImFloat(512, 512);
    imdft = AllocImDouble(512, 512);
    imdftfase = AllocImDouble(512, 512);
    imidft = AllocImFloat(512, 512);

    int count=0;
    for (int i=0; i<512; i++)
        for (int j=0; j<512; j++){
            imyout[i][j] =
                    static_cast<float>(ImageOut[count]);
            count++;
        }



    FFT2D::dft2d(imyout, imdft, imdftfase, 512, 512);


    int coefficient_number;
    double *coefficient_vector =NULL;

//    if ((size>256)&&(size<=512))
//    {
//        dim2 = 512;		// Dimensione 512x512
    int diag0 = 80;		// Diagonali..
    int ndiag = 74;
//    }

    coefficient_vector = zones_to_watermark(imdft, 512, 512, diag0, ndiag, 0, &coefficient_number);

    double * mark;
    mark = new double[coefficient_number];

    generate_mark(watermark,wsize,passw_str,passw_num,coefficient_number,mark);

    addmark(coefficient_vector,mark,coefficient_number,power);

    antizone(imdft, 512, 512, diag0, ndiag, coefficient_vector);


    FFT2D::idft2d(imdft, imdftfase, imidft, 512, 512);

    /*
     * FFT2D
     *
     * int coefficient_number;
     *
     * zone alias coefficient selection
     *
        double * mark;
        mark = new double[coefficient_number];

       generate_mark();
     * watermark embedding : addmark + antizone
     * IFFT2D
     *
     *
     *
     *
     */


}



void Watermarking::generate_mark(int *watermark,int wsize, const char *passw_str, const char *passw_num, int marklen, double* mark)
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


    /*
     * seed generation
     */

    LONG8BYTE *seed;		// variabile per generare il marchio


    seed = new LONG8BYTE [4];
    seed_generator(passw_str,passw_num,seed);

    seed_initialization(seed);

    for(int i = 0; i < marklen; i++)
        mark[i] = 2.0 * ( pseudo_random_generator() - 0.5);

    // mark modulation
    ///////////////////////////

    int n=0;
    int L=marklen/length_BCH;
    for (int k=length_BCH-1; k>=0; k--)
    {
        if (bch_wm[k]==0)
        {
            if (k==0)
            {
                for (int i=n; i<marklen;i++)
                    mark[i]*=-1;
            }
            else
                for (int i=n; i<n+L;i++)
                    mark[i]*=-1;
            n+=L;
        }
        else
            n+=L;
    }

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

void Watermarking::addmark(double *buff, double *mark, int num_camp, double peso)
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
	antizone(..)
	------------

	Questa funzione rimette i coefficienti marchiati al loro posto
*/


void Watermarking::antizone(double **imdft,int nr, int nc, int diag0, int ndiag, double *buff)
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
	inizializzaza(..) e generatore()
	--------------------------------

	Funzioni per la generazione di sequenze pseudo-casuali
	di numeri reali uniformemente distribuiti tra [0,1]
*/

void Watermarking::seed_initialization(LONG8BYTE *s)
{
    int j;

    for(j = 0; j < 4; j ++)
    {
        init_seed[j] = s[j];
        current_seed[j] = current_seed[j];
    }
}


double Watermarking::pseudo_random_generator()
{
    LONG8BYTE 	s;
    double	u;
    double	q;

    LONG8BYTE a[] = {45991, 207707, 138556, 49689};
    LONG8BYTE m[] = {2147483647, 2147483543, 2147483423, 2147483323};

    u = 0.0;

    s = current_seed[0];
    s = (s * a[0]) % m[0];

    current_seed[0] = s;
    u = u + 4.65661287524579692e-10 * s;
    q = 4.65661287524579692e-10;

    s = current_seed[1];
    s = (s * a[1]) % m[1];

    current_seed[1] = s;
    u = u - 4.65661310075985993e-10 * s;
    if(u < 0)
        u = u + 1.0;

    s = current_seed[2];
    s = (s * a[2]) % m[2];

    current_seed[2] = s;
    u = u + 4.65661336096842131e-10 * s;
    if(u >= 1.0)
        u = u - 1.0;

    s = current_seed[3];
    s = (s * a[3]) % m[3];

    current_seed[3] = s;
    u = u - 4.65661357780891134e-10 * s;
    if(u < 0)
        u = u + 1.0;
    return u;
}


/*
	codmarchio(..)
	-------------

	Codmarchio riceve in ingresso due stringhe, una di caratteri
	e una di numeri, e le converte nei 4 semi dei generatori clcg
	di numeri pseudo-casuali. Restituisce in uscita il puntatore i
	al vettore con i 4 semi.
*/
void Watermarking::seed_generator(const char *passw_str, const char *passw_num, LONG8BYTE *s )
{

    int *string_coding, *number_coding;    // vettori che contengono la codifica dei caratteri
    int **S;         // matrice con combinazione dei vettori cl e cn

    // Allocazione aree di memoria
    string_coding = new int [16];
    number_coding = new int [8];
    S = AllocImInt(4, 6);

    for(int i = 0; i < 16; i ++)
    {
        switch (passw_str[i])
        {
            case ' ': 		string_coding[i] = 0;
                break;

            case 'A': case 'a':	string_coding[i] = 1;
                break;

            case 'B': case 'b':	string_coding[i] = 2;
                break;

            case 'C': case 'c':     string_coding[i] = 3;
                break;

            case 'D': case 'd':     string_coding[i] = 4;
                break;

            case 'E': case 'e':     string_coding[i] = 5;
                break;

            case 'F': case 'f':     string_coding[i] = 6;
                break;

            case 'G': case 'g':     string_coding[i] = 7;
                break;

            case 'H': case 'h':     string_coding[i] = 8;
                break;

            case 'I': case 'i':     string_coding[i] = 9;
                break;

            case 'J': case 'j':     string_coding[i] = 10;
                break;

            case 'K': case 'k':     string_coding[i] = 11;
                break;

            case 'L': case 'l':     string_coding[i] = 12;
                break;

            case 'M': case 'm':     string_coding[i] = 13;
                break;

            case 'N': case 'n':     string_coding[i] = 14;
                break;

            case 'O': case 'o':     string_coding[i] = 15;
                break;

            case 'P': case 'p':     string_coding[i] = 16;
                break;

            case 'Q': case 'q':     string_coding[i] = 17;
                break;

            case 'R': case 'r':     string_coding[i] = 18;
                break;

            case 'S': case 's':     string_coding[i] = 19;
                break;

            case 'T': case 't':     string_coding[i] = 20;
                break;

            case 'U': case 'u':     string_coding[i] = 21;
                break;

            case 'V': case 'v':     string_coding[i] = 22;
                break;

            case 'W': case 'w':     string_coding[i] = 23;
                break;

            case 'X': case 'x':     string_coding[i] = 24;
                break;

            case 'Y': case 'y':     string_coding[i] = 25;
                break;

            case 'Z': case 'z':     string_coding[i] = 26;
                break;

            case '.': 	        string_coding[i] = 27;
                break;

            case '-':               string_coding[i] = 28;
                break;

            case '&':               string_coding[i] = 29;
                break;

            case '/':               string_coding[i] = 30;
                break;

            case '@':               string_coding[i] = 31;
                break;

            default: 		string_coding[i] = 0;
                break;
        }
    }

    for(int i = 0; i < 8; i++)
    {
        switch (passw_num[i])
        {
            case '0':		number_coding[i] = 0;
                break;

            case '1':               number_coding[i] = 1;
                break;

            case '2':               number_coding[i] = 2;
                break;

            case '3':               number_coding[i] = 3;
                break;

            case '4':               number_coding[i] = 4;
                break;

            case '5':               number_coding[i] = 5;
                break;

            case '6':               number_coding[i] = 6;
                break;

            case '7':               number_coding[i] = 7;
                break;

            case '8':               number_coding[i] = 8;
                break;

            case '9':               number_coding[i] = 9;
                break;

            case '.':               number_coding[i] = 10;
                break;

            case '/':               number_coding[i] = 11;
                break;

            case ',':               number_coding[i] = 12;
                break;

            case '$':               number_coding[i] = 13;
                break;

/*			case 'lira':                cn[i] = 14;
                                                break;
*/
            case ' ':               number_coding[i] = 15;
                break;

            default: 		number_coding[i] = 0;
                break;
        }
    }


    for(int i = 0; i < 4; i ++)
    {
        for(int j = 0; j < 4; j ++)
            S[i][j] = string_coding[i + 4 * j];

        for(int j = 0; j < 2; j ++)
            S[i][j + 4] = number_coding[i + 4 * j];
    }

    for(int i = 0; i < 4; i ++)
    {
        s[i] = S[i][0] + S[i][1] * (int)pow(2, 5) + S[i][2] * (int)pow(2, 10) + S[i][3] * (int)pow(2, 15) + S[i][4] * (int)pow(2, 20) + S[i][5] * (int)pow(2, 24) + 1;
    }

    FreeIm(S);
    delete [] string_coding;
    delete [] number_coding;
}

/*
                               ZONE

	Questa funzione raggruppa i coefficienti DFT (appartenenti
	alle 16 parti in cui viene suddivisa la zona dell'immagine
	che viene marchiata) nel vettore buff.
*/

double* Watermarking::zones_to_watermark(double **imdft, int height, int width, int diag0, int ndiag,
                     int detect, int *coefficient_number)
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
                    j=height-1-m+i;
                    *ptr8++ = imdft[i][j];
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    *ptr1++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr9++ = imdft[i][j];
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    *ptr2++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr10++ = imdft[i][j];
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    *ptr3++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr11++ = imdft[i][j];
                }
            }

            if(m>=(d1+(max-d1)/2) && m<=max)
            {
                if (i>0 && i<(m/4))
                {
                    j=m-i;
                    *ptr4++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr12++ = imdft[i][j];
                }
                if (i>=(m/4) && i<(m/2))
                {
                    j=m-i;
                    *ptr5++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr13++ = imdft[i][j];
                }
                if (i>=(m/2) && i<((3*m)/4))
                {
                    j=m-i;
                    *ptr6++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr14++ = imdft[i][j];
                }
                if (i>=((3*m)/4) && i<m)
                {
                    j=m-i;
                    *ptr7++ = imdft[i][j];
                    j=height-1-m+i;
                    *ptr15++ = imdft[i][j];
                }
            }
        }
    }


    *coefficient_number = elementi;

    return buff;
}