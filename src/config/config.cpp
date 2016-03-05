
#include "config.hpp"
#include <iostream>

using namespace std;

namespace Watermarking_config {

    // ecco il costruttore privato in modo che l'utente non possa istanziare direttamante

    ConfigLoader::ConfigLoader(std::string configPath) {

//            libconfig::Config cfg;
            // Read the file. If there is an error, report it and exit.
            try
            {

                cfg.readFile(configPath.c_str());

            }
            catch(const libconfig::FileIOException &fioex)
            {
                std::cerr << "I/O error while reading file." << std::endl;
                exit(EXIT_FAILURE);
            }
            catch(const libconfig::ParseException &pex)
            {
                std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                        << " - " << pex.getError() << std::endl;
                exit(EXIT_FAILURE);
            }
            std::cout << "Settings loaded "<<std::endl;
    };


    set_parameters_params ConfigLoader::loadSetParametersConfiguration() {

        set_parameters_params pars;
        try
        {

            const libconfig::Setting & root = cfg.getRoot();
            const libconfig::Setting & ConfigSettings  = root["set_parameters_params"];

            pars.wsize = (int) ConfigSettings["wsize"];
            pars.power = (float) ConfigSettings["power"];
            pars.watermark = (std::string) ConfigSettings["watermark"].c_str();


        }
        catch(const libconfig::SettingNotFoundException &nfex)
        {
            std::cout << nfex.what() << " Config" << std::endl;

        }
        return pars;
    }


    general_params ConfigLoader::loadGeneralParamsConfiguration() {
        general_params pars;
        try
        {

            const libconfig::Setting & root = cfg.getRoot();
            const libconfig::Setting & ConfigSettings  = root["general_params"];

            pars.masking = (bool) ConfigSettings["masking"];
            pars.passwstr = (std::string) ConfigSettings["passwstr"].c_str();
            pars.passwnum = (std::string) ConfigSettings["passwnum"].c_str();


        }
        catch(const libconfig::SettingNotFoundException &nfex)
        {
            std::cout << nfex.what() << " Config" << std::endl;

        }
        return pars;
    }

}



//
//int main() {
//    std::cout << singleton::get_instance().method() << std::endl;
//
//    return 0;
//
//}