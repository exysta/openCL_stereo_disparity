#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <OpenCL/opencl.h>
#include <vector>

// Fonction pour convertir une image RGB en niveaux de gris avec OpenCL
std::vector<unsigned char> convertToGrayscaleGPU(cl_context context, cl_command_queue commandQueue,
                                                 unsigned char* inputImage, int width, int height, int channels);

// Fonction pour redimensionner une image en niveaux de gris avec OpenCL
std::vector<unsigned char> resizeGrayscaleGPU(cl_context context, cl_command_queue commandQueue,
                                              const std::vector<unsigned char>& inputImage, 
                                              int src_width, int src_height,
                                              int dst_width, int dst_height);

#endif // IMAGE_PROCESSING_H
