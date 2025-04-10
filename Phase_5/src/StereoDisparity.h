#ifndef STEREO_DISPARITY_H
#define STEREO_DISPARITY_H

#include <vector>
#include <OpenCL/opencl.h>

// Structure pour représenter une image
struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

// Fonctions de chargement et sauvegarde d'images
Image load_image(const char* filename);
void save_disparity(const char* filename, const Image& img);

// Pré-calcul des valeurs de fenêtre pour les deux images
void precomputeWindowValues(const Image& img, std::vector<double>& means, std::vector<double>& stdDevs, int win_size);

// Version CPU de l'algorithme de disparité stéréo
Image computeDisparityCPU(const Image& left, const Image& right, int max_disp, int win_size);

// Version OpenCL de l'algorithme de disparité stéréo
std::vector<unsigned char> computeStereoDisparity(cl_context context, cl_command_queue commandQueue,
                                                   cl_mem imageBuffer0, cl_mem imageBuffer1,
                                                   int width, int height, int channels);

// Fonction pour afficher les résultats de profilage
void printProfilingInfo(const std::string& kernelName, double executionTime, size_t globalWorkSize);

// Fonction pour calculer la disparité stéréo avec CPU
std::vector<unsigned char> computeStereoDisparityCPU(unsigned char* image0, unsigned char* image1,
                                                    int width, int height, int channels);

#endif