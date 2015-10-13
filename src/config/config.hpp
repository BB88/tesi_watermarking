#ifndef TESI_WATERMARKING_CONFIG_H
#define TESI_WATERMARKING_CONFIG_H


#include <libconfig.h++>
#include <iostream>

using namespace std;

namespace Watermarking_config {

    struct general_params{
        std::string passwstr;
        std::string passwnum;
        bool masking;
    };

    struct set_parameters_params {
        int wsize ;
        int tilesize;
        float power;
        bool clipping;
        bool flagResyncAll;
        int tilelistsize;
    };

    class ConfigLoader {

    private:

        ConfigLoader();
        virtual ~ConfigLoader(){}

    private:
        libconfig::Config cfg;

    public:

        static ConfigLoader& get_instance() {
            // l'unica istanza della classe viene creata alla prima chiamata di get_instance()
            // e verrà distrutta solo all'uscita dal programma
            static ConfigLoader instance;
            return instance;
        }

        set_parameters_params loadSetParametersConfiguration();
        general_params loadGeneralParamsConfiguration();


        // C++ 11
        // =======
        // We can use the better technique of deleting the methods
        // we don't want.
        ConfigLoader(ConfigLoader const&)               = delete;
        void operator=(ConfigLoader const&)  = delete;
    };



}
#endif //TESI_WATERMARKING_CONFIG_H