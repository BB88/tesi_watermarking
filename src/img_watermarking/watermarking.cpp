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
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <highgui.h>

#include </home/miky/ClionProjects/tesi_watermarking/src/utils.h>


using namespace cv;

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

void Watermarking::setParameters(int *w, int watsize, int tsize, float pwr, bool useClipping,
                           bool resynchronization, int *tilelist, int tilelistsize)
{
    // list of wsize (watermark)
    memcpy(watermark, w, sizeof(int) * watsize);

    wsize = watsize;
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


unsigned char * Watermarking::insertWatermark(unsigned char *imageIn, int w, int h,int dim,float** imidft_wat,bool warp_flag)
{
    bool flagOk;

// SE IMMAGINE GREY
//    unsigned char *output_img = new unsigned char[w * h];
//    memcpy(output_img, image, w * h);

// SE COLOUR
    unsigned char *imageOut = new unsigned char[w * h*3];
    memcpy(imageOut, imageIn, w * h*3);

    const char *passw_str = passwstr.c_str();
    const char *passw_num = passwnum.c_str();


    int result = -1;

    if(warp_flag == false)
        result = WatCod(imageOut, w, h, passw_str, passw_num, watermark, wsize, power, imidft_wat,dim);
    else
        result = warpedWatCod(imageOut, w, h, passw_str, passw_num, watermark, wsize, power, imidft_wat);

    return imageOut;


}

int Watermarking::WatCod(unsigned char *ImageOut, int width, int height, const char *passw_str, const char *passw_num,
                         int *watermark, int wsize, float power, float** imidft_wat, int dim)
{
    int diag0;
    int ndiag;
    if (dim==256){
        diag0 = 30;  //40
        ndiag = 40;
    }else if (dim==512){
        diag0 = 80;
        ndiag = 74;
    }
    float   **imyout;
    double  **imdft;
    double  **imdftfase;
    float   **imidft;

    imyout = AllocImFloat(dim, dim);
    imdft = AllocImDouble(dim, dim);
    imdftfase = AllocImDouble(dim, dim);
    imidft = AllocImFloat(dim, dim);

    unsigned char **imr;
    unsigned char **img;
    unsigned char **imb;

    float **imc2;
    float **imc3;

    imc2 = AllocImFloat(dim, dim);
    imc3 = AllocImFloat(dim, dim);
    imr = AllocImByte(dim, dim);
    img = AllocImByte(dim, dim);
    imb = AllocImByte(dim, dim);

    int offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr[i][j] = ImageOut[offset];offset++;
            img[i][j] = ImageOut[offset];offset++;
            imb[i][j] = ImageOut[offset];offset++;
        }

    rgb_to_crom(imr, img, imb, dim, dim, 1, imyout, imc2, imc3);
//    stereo_watermarking::show_floatImage(imyout,dim,dim,"left_lum");
//    dft computation: magnitude and phase
    FFT2D::dft2d(imyout, imdft, imdftfase, dim, dim);
    int coefficient_number;
    double *coefficient_vector = NULL;
//    coefficients extraction
    coefficient_vector = zones_to_watermark(imdft, dim, dim, diag0, ndiag, 0, &coefficient_number);
//    stereo_watermarking::writeToFile(coefficient_vector,coefficient_number,"/home/miky/Scrivania/wm_coff_mark.txt");
//    saving coefficients
    coeff_dft = new double [coefficient_number];
    for (int k = 0; k < coefficient_number; k++ )
        coeff_dft[k] = coefficient_vector[k];
    coeff_number = coefficient_number;
//    mark generation
    double * mark;
    mark = new double[coefficient_number];
    generate_mark(watermark,wsize,passw_str,passw_num,coefficient_number, mark,false);
    //mette marchio in range (0,1)
//     for(int i=0;i<coefficient_number;i++){
//       mark[i]+=1;
//       mark[i]/=2;
//      }
//    saving mark
    final_mark = new double [coefficient_number];
    for (int k = 0; k < coefficient_number; k++ ) {
        final_mark[k] = mark[k];
    }
//    generate dft of the watermark
    double** dft_wat = AllocImDouble(dim, dim);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            dft_wat[i][j] = 0.0;
    antizone(dft_wat, dim, dim, diag0, ndiag, mark);
//    generate magnitude and phase
    double  **imdft_wat;
    double  **imdftfase_wat;
    imdft_wat = AllocImDouble(dim, dim);
    imdftfase_wat = AllocImDouble(dim, dim);
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            if (dft_wat[i][j]<0){
                imdft_wat[i][j] = power*abs(dft_wat[i][j])*imdft[i][j];
                imdftfase_wat[i][j] = imdftfase[i][j] + PI;
            } else if (dft_wat[i][j]>0){
                imdft_wat[i][j] = power*abs(dft_wat[i][j])*imdft[i][j];
                imdftfase_wat[i][j] = imdftfase[i][j] + 0.0;
            } else {
                imdft_wat[i][j] = 0.0;
                imdftfase_wat[i][j] = 0.0;
            }
    FFT2D::idft2d(imdft_wat, imdftfase_wat, imidft_wat, dim, dim);
//    stereo_watermarking::writefloatMatToFile(imidft_wat,dim,"/home/miky/Scrivania/wat_lum.txt");
//    add mark to coefficients: 1 if add_mult
    addmark(coefficient_vector, mark, coefficient_number, power,1);
//    put back the marked coefficients
    antizone(imdft, dim, dim, diag0, ndiag, coefficient_vector);
//    idft->back to the luminance
    FFT2D::idft2d(imdft, imdftfase, imidft, dim, dim);
//    stereo_watermarking::show_floatImage(imidft,256,256,"squared_marked_left");
//    back to chrominance
    rgb_to_crom(imr, img, imb, dim, dim, -1, imidft, imc2, imc3);
//    back to image
    offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            ImageOut[offset] = imr[i][j]; offset++;
            ImageOut[offset] = img[i][j]; offset++;
            ImageOut[offset] = imb[i][j]; offset++;
        }

//    stereo_watermarking::show_ucharImage(ImageOut,dim,dim,"marked_left");
    AllocIm::FreeIm(imc2) ;
    AllocIm::FreeIm(imc3) ;
    AllocIm::FreeIm(imr);
    AllocIm::FreeIm(img);
    AllocIm::FreeIm(imb);

    AllocIm::FreeIm(imyout);
    AllocIm::FreeIm(imdft);
    AllocIm::FreeIm(imdftfase);
    AllocIm::FreeIm(imidft);

}
int Watermarking::warpedWatCod(unsigned char *ImageOut, int width, int height, const char *passw_str, const char *passw_num, int *watermark, int wsize, float power, float** imidft_wat)
{
    int dim = width;
    int diag0;
    int ndiag;
    if (dim==256){
        diag0 = 30;  //40
        ndiag = 40;
    }else if (dim==512){
        diag0 = 80;
        ndiag = 74;
    }
    float   **imyout;
    double  **imdft;
    double  **imdftfase;
    float   **imidft;

    imyout = AllocImFloat(dim, dim);
    imdft = AllocImDouble(dim, dim);
    imdftfase = AllocImDouble(dim, dim);
    imidft = AllocImFloat(dim, dim);

    unsigned char **imr;
    unsigned char **img;
    unsigned char **imb;

    float **imc2;
    float **imc3;

    imc2 = AllocImFloat(dim, dim);
    imc3 = AllocImFloat(dim, dim);
    imr = AllocImByte(dim, dim);
    img = AllocImByte(dim, dim);
    imb = AllocImByte(dim, dim);

    int offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr[i][j] = ImageOut[offset];offset++;
            img[i][j] = ImageOut[offset];offset++;
            imb[i][j] = ImageOut[offset];offset++;
        }

    rgb_to_crom(imr, img, imb, dim, dim, 1, imyout, imc2, imc3);
    stereo_watermarking::writefloatMatToFile(imyout,256,"/home/miky/Scrivania/right_lum.txt");
//    dft computation: magnitude and phase
    FFT2D::dft2d(imyout, imdft, imdftfase, dim, dim);
    int coefficient_number;
    double *coefficient_vector = NULL;
//    coefficients extraction
    coefficient_vector = zones_to_watermark(imdft, dim, dim, diag0, ndiag, 0, &coefficient_number);
//    stereo_watermarking::writeToFile(coefficient_vector,coefficient_number,"/home/miky/Scrivania/wm_coff_mark.txt");
//    compute magnitude and phase of the watermark
    double  **imdft_wat;
    double  **imdftfase_wat;
    imdft_wat = AllocImDouble(dim, dim);
    imdftfase_wat = AllocImDouble(dim, dim);
    FFT2D::dft2d(imidft_wat, imdft_wat, imdftfase_wat, dim, dim);
    double *mark = new double[coefficient_number];
    mark = zones_to_watermark(imdft, dim, dim, diag0, ndiag, 0, &coefficient_number);

//    add mark to coefficients: 1 if add_mult
    addmark(coefficient_vector, mark, coefficient_number, power,1);
//    put back the marked coefficients
    antizone(imdft, dim, dim, diag0, ndiag, coefficient_vector);
//    idft->back to the luminance
    FFT2D::idft2d(imdft, imdftfase, imidft, dim, dim);
//    stereo_watermarking::writefloatMatToFile(imidft,256,"/home/miky/Scrivania/marked_right_lum.txt");
    stereo_watermarking::show_floatImage(imidft,dim,dim,"marked_right_dft");
//    back to chrominance
    rgb_to_crom(imr, img, imb, dim, dim, -1, imidft, imc2, imc3);
//    back to image
    offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            ImageOut[offset] = imr[i][j]; offset++;
            ImageOut[offset] = img[i][j]; offset++;
            ImageOut[offset] = imb[i][j]; offset++;
        }
    AllocIm::FreeIm(imc2) ;
    AllocIm::FreeIm(imc3) ;
    AllocIm::FreeIm(imr);
    AllocIm::FreeIm(img);
    AllocIm::FreeIm(imb);

    AllocIm::FreeIm(imyout);
    AllocIm::FreeIm(imdft);
    AllocIm::FreeIm(imdftfase);
    AllocIm::FreeIm(imidft);

    AllocIm::FreeIm(imdft_wat);
    AllocIm::FreeIm(imdftfase_wat);
}



double* Watermarking::marks_generator(int *watermark,int wsize, const char *passw_str, const char *passw_num, int coefficient_number){

    double * mark = new double[coefficient_number];
    Watermarking::generate_mark(watermark,wsize,passw_str,passw_num,coefficient_number, mark,false);
    return mark;
}



void Watermarking::generate_mark(int *watermark,int wsize, const char *passw_str, const char *passw_num, int marklen, double* mark, bool detection)
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


    /*for (int i=0;i<200;i++)
        cout<<bch_wm[i]<<" ";
    cout<<"\n";
*/
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

void Watermarking::addmark(double *buff, double *mark, int coeff_number, double power, int add_mult )
{
    double alfa;
    codemark = new double [coeff_number];
    int count = 0;
    if (add_mult == 1) {
        for (int i = 0; i < coeff_number; i++) {
            codemark[i] = buff[i] * alfa * mark[i];
            buff[i] = buff[i] * (1.0 + power * mark[i]);  // additivo moltiplicativo
        }
    } else { //additivo
        for (int i = 0; i < coeff_number; i++) {
            if ((buff[i] + alfa * mark[i]) >= 0)
                buff[i] = buff[i] + alfa * mark[i];
            else {
                buff[i] = buff[i];
                count++;
            }
        }
    }
    cout << "#not marked coeff :   "<< count<<endl;
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
        current_seed[j] = init_seed[j];
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

void Watermarking::DecimVarfloat(float **imc1, int nr, int nc,
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

void Watermarking::PicRoutfloat(float **img_orig, int nr, int nc,
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




//////////watermark estraction//////////////////


bool Watermarking::extractWatermark(unsigned char *image, int w, int h,int dim)
{
    bool flagOk = false;

    const char *passw_str = passwstr.c_str();
    const char *passw_num = passwnum.c_str();

    // image after resynchronization
    unsigned char *imrsinc = new unsigned char[w * h * 3];

    // resynchronization data for each tile... not used(!!)
    // (see inside WatDec(.) for further details)
    double *datiuscita = new double[32000];
//    cout<<fixed<<power<<endl;
    int result = WatDec(image, h, w, passw_str, passw_num, watermark, tilesize, wsize, power, datiuscita, imrsinc, tiles, flagResyncAll,dim);


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
        fprintf(flog, " - Invalid size of the tile. Valid size are 256, 256 or 1024.");

//        // invalid 'size'
//        QMessageBox::warning(NULL, tr(MSG_TITLE),
//                             tr("Invalid size of the tile. Valid size are 256, 256 or 1024."));
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






int Watermarking::WatDec(unsigned char *ImageIn, int nrImageIn, int ncImageIn,
                   const char *campolett, const char *camponum,
                   int *bit, int size, int nbit,
                   float power, double *datiuscita, unsigned char *buffimrisinc,
                   int *vettoretile, bool flagRisincTotale,int dim )
{

    int diag0;
    int ndiag;
    if (dim == 256){
        diag0 = 30;
        ndiag = 40;
    }
    else if(dim == 512){
        diag0 = 80;
        ndiag = 74;
    }
//    int diag0 = 30;		// Diagonali..
//    int ndiag = 40;		// numero diagonali marchiate


    float **imy;			// matrice luminanza
    float **imc2;			// matrice crominanza c2
    float **imc3;			// matrice crominanza c3

    unsigned char **imr;	// immagini delle tre componenti RGB
    unsigned char **img;	//
    unsigned char **imb;	//



    double **imdft;			// matrice contenente il modulo della DFT
    double **imdftfase;		// matrice contenente la fase della DFT
    double **im1dft;		// matrice contenente il modulo della DFT
    double **im1dftfase;	// matrici contenente la fase della DFT
    LONG8BYTE *seed;		// variabile per generare il marchio

    float **imyout;			// Matrice di luminanza del tile

    imyout = AllocImFloat(dim, dim);

    double **imdftout;		// Matrice dft del tile ridimensionato
    double **imdftoutfase;

    imdftout = AllocImDouble(dim, dim);
    imdftoutfase = AllocImDouble(dim, dim);

    imr = AllocImByte(dim, dim);
    img = AllocImByte(dim, dim);
    imb = AllocImByte(dim, dim);
    imy = AllocImFloat(dim, dim);
    imc2 = AllocImFloat(dim, dim);
    imc3 = AllocImFloat(dim, dim);

    int offset = 0;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
        {
            imr[i][j] = ImageIn[offset];offset++;
            img[i][j] = ImageIn[offset];offset++;
            imb[i][j] = ImageIn[offset];offset++;
        }


    // Si calcolano le componenti di luminanza e crominanza dell'immagine
    rgb_to_crom(imr, img, imb, dim, dim, 1, imyout, imc2, imc3);


//    int coefficient_number;
//    double * mark;
//    mark = new double[coefficient_number];
//    generate_mark(watermark,wsize,campolett,camponum,coefficient_number, mark, true);

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

    seed = new LONG8BYTE [4];
    seed_generator(campolett,camponum,seed);

//    for (int i=0; i<256; i++)
//        for (int j=0; j<256; j++)
//            imdftout[i][j]=0.0;

    FFT2D::dft2d(imyout, imdftout, imdftoutfase, dim, dim);
/*    for (int i = 0; i<256 ; i++)
        for (int j = 0; j <256 ; j++)
              if (imdftout[i][j]<0.0){
                  cout<< "valore negativo"<<endl;
              }*/

    // Added by CORMAX
    int BitLetti[200];		// Max 200 bit da leggere
    int dec=0;

/*
//
//    int nouniforme=0;		// Controllo ver vedere se ho una zona uniforme




//    double *coefficient_vector = NULL;
//    coefficient_vector = zones_to_watermark(imdftout, 256, 256, diag0, ndiag, 0, &coefficient_number);
//
*/

    decoale(imdftout, dim, dim, diag0, ndiag, seed, power ,BitLetti, length_BCH);

  /*  for (int i=0;i<200;i++)
        cout<<BitLetti[i]<<" ";
    cout<<"\n";

*/

//    for (int i=0; i<200; i++)
//        nouniforme += BitLetti[i];
   bool res= BCH::decode_bch(m_BCH,length_BCH,t_BCH,BitLetti);

    if (res)
    {
 /*       // THE WATERMARK HAS BEEN DETECTED
//        FreeIm(imy);
//        FreeIm(imc2);
//        FreeIm(imc3);
//        FreeIm(imyout);
//
//        FreeIm(imdft);
//        FreeIm(imdftfase);
//        FreeIm(imdftout);
//        FreeIm(imdftoutfase);
//        FreeIm(im1dft);
//        FreeIm(im1dftfase);


        // NOTA: I bit decodificati vengono salvati alla fine
        //		 di BitLetti, ossia a partire dalla posizione
        //		 (BCH length)-(information message)

//        int offs;
//        offs = length_BCH - nbit;
//        for (int i=0; i<nbit; i++)
//            bit[i] = BitLetti[i+offs];


//                    // LOG
//                    fclose(flog);*/

        return 0;	// OK! Marchio rivelato
    }


    FreeIm(imyout);

    return -5;
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

void Watermarking::decoale(double **imr, int nre, int nce, int d1, int nd,
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

    appbuff = zones_to_watermark(imr, nre, nce, d1, nd, 1, &marklen);
//    marked_coeff = new double [marklen];
//    for (int k = 0; k < marklen; k++ )
//        marked_coeff[k] = appbuff[k];
////    stereo_watermarking::writeMatToFile(marked_coeff,marklen,"/home/miky/Scrivania/Tesi/dec_marked_coeff.txt");
//    marked_coeff_number = marklen;


//
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

    seed_initialization(seed);

    for(i = 0; i < marklen; i++)
        mark[i] = 2.0 * (pseudo_random_generator() - 0.5);




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


/*
	mlfunc(..)
	----------

	Risolve numericamente l'equazione del criterio ML
	per calcolare i parametri della p.d.f. Weibull
*/

void Watermarking::mlfunc(double *buff,int nrfile,int niteraz)
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
	dgamma(..)
	----------

	Gamma function in double precision

	Added by CORMAX,	14/MAR/2001

*/

double Watermarking::dgamma(double x)
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

void Watermarking::rgb_to_crom(unsigned char **imr, unsigned char **img,
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


