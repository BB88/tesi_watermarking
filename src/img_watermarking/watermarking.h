//
// Created by miky on 22/07/15.
//

#ifndef TESI_WATERMARKING_WATERMARKING_H
#define TESI_WATERMARKING_WATERMARKING_H

#endif //TESI_WATERMARKING_WATERMARKING_H
#include <math.h>

#define PI  3.14159265358979323846


class Watermarking
//       : public QObject,
//                     public ImgWatInterface
{
//    Q_OBJECT
//            Q_INTERFACES(ImgWatInterface)

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

    static const char *MSG_TITLE;

    typedef __int64_t LONG8BYTE;    // NOTE: __int64 HAS NO ANSI EQUIVALENT (!)

    // Struttura punto di interesse
    struct Point {
        float Val;
        float Valf;
        int Riga;
        int Col;
    };

    // Struttura Scacco per la sincronizzazione
    struct Scacco {
        double DeltaX;
        double DeltaY;
        double Alfa;
        double Beta;
        double Periodo;
    };

    // Struttura di un nodo che collegato ricorsivamente crea un albero
    struct tnode {
        int flag;
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

    LONG8BYTE current_seed[4];        // seme corrente del generatore clcg
    LONG8BYTE init_seed[4];        // seme iniziale del generatore clcg

    //! Password (alphabetic field).
    std::string passwstr;

    //! Password (numeric field).
    std::string passwnum;

    //! Watermark power.
    float power;

    //! Clipping flag.
    bool clipping;

    //! Number of bits of the watermark.
    int wsize;

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
     * \param wsize          Number of bits of the watermark
     * \param tsize          Size of the tiles
     * \param pwr           Watermark power
     * \param useClipping   Clipping flag
     * \param tilelist      List of tiles to watermark
     * \param tilelistsize    Number of elements of the tiles list
     */
    void setParameters(int *w, int watsize, int tsize, float pwr, bool useClipping,
                       bool synchronization, int *tilelist, int tilelistsize);

    /**
     * Set password.
     */
    void setPassword(std::string passwStr, std::string passwNum);

    /**
     * Insert the watermark into the given image.
     *
     * This method wrap the original image watermarking encoding algorithm (WatCod).
     *
     * \note The input image is not modified.
     */
    unsigned char *insertWatermark(unsigned char *image, int w, int h);

    /**
     * Extract the watermark from the given image.
     *
     * This method wrap the original image watermarking decoding algorithm (WatDec).
     */
    bool extractWatermark(unsigned char *image, int w, int h);


    void rgb_to_crom(unsigned char **imr, unsigned char **img,
                     unsigned char **imb, int nr, int nc, int flag,
                     float ** imc1, float **imc2, float ** imc3);

    void DecimVarfloat(float **imc1, int nr, int nc,
                       int win, float **img_map_flt);

    void PicRoutfloat(float **img_orig, int nr, int nc,
                      float **img_mark, float **img_map_flt, float **impic);

private:

    int WatCod(unsigned char *ImageOut , int width, int height,
               const char *passw_str, const char *passw_num, int *watermark, int wsize, float power, bool flagClipping, int tilesize, int *tiles, int *ntiles);


    void seed_generator(const char *passw_str, const char *passw_num, LONG8BYTE *s );
    void generate_mark(int *watermark,int wsize, const char *passw_str, const char *passw_num, int coefficient_number,double* mark,bool detection) ;
    void seed_initialization(LONG8BYTE *s);
    double pseudo_random_generator();
    double* zones_to_watermark(double **imdft, int height, int width, int diag0, int ndiag, int detect, int *coefficient_number);


    void addmark(double *buff, double *mark, int num_camp, double peso);


    void antizone(double **imdft,int nr, int nc, int diag0, int ndiag, double *buff);

    int WatDec(unsigned char *ImageIn, int nrImageIn, int ncImageIn,
                             const char *campolett, const char *camponum,
                             int *bit, int size, int nbit,
                             float power, double *datiuscita, unsigned char *buffimrisinc,
                             int *vettoretile, bool flagRisincTotale );
    void decoale(double **imr, int nre, int nce, int d1, int nd,
                               LONG8BYTE *seed, double alpha,int *bit, int nbit);

    void mlfunc(double *buff,int nrfile,int niteraz);
    double dgamma(double x);

//    void rgb_to_crom(unsigned char **imr, unsigned char **img,
//                                   unsigned char **imb, int nr, int nc, int flag,
//                                   float ** imc1, float **imc2, float ** imc3);

};