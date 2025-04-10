#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"
#include "StereoDisparity.h"

// Fonction pour convertir une image RGB en niveaux de gris avec OpenCL
std::vector<unsigned char> convertToGrayscaleGPU(cl_context context, cl_command_queue commandQueue,
                                                 unsigned char* inputImage, int width, int height, int channels) {
    cl_int err;
    
    // Lire le fichier kernel
    std::ifstream kernelFile("src/kernels/image_processing_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Impossible d'ouvrir le fichier du kernel de traitement d'image." << std::endl;
        return std::vector<unsigned char>();
    }
    
    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();
    
    // Créer programme et compiler
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur compilation programme de traitement d'image." << std::endl;
        
        // Récupérer le device associé à la commandQueue
        cl_device_id device;
        err = clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Erreur lors de la récupération du device." << std::endl;
            clReleaseProgram(program);
            return std::vector<unsigned char>();
        }
        
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << buildLog.data() << std::endl;
        
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer kernel
    cl_kernel grayscaleKernel = clCreateKernel(program, "rgb_to_grayscale", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création du kernel de conversion en niveaux de gris." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer les buffers
    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, inputSize, nullptr, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création des buffers pour la conversion en niveaux de gris." << std::endl;
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Copier les données d'entrée
    err = clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, inputSize, inputImage, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la copie des données d'entrée pour la conversion en niveaux de gris." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Configurer les arguments du kernel
    err  = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(grayscaleKernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(grayscaleKernel, 4, sizeof(int), &channels);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la configuration des arguments du kernel de conversion en niveaux de gris." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer un événement pour le profilage
    cl_event grayscaleEvent;
    
    // Exécuter le kernel
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    
    // Déterminer la taille optimale du work-group
    cl_device_id device;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    size_t wg_size = 16; // Valeur par défaut
    while (wg_size * wg_size > max_work_group_size) {
        wg_size /= 2;
    }
    
    size_t localSize[2];
    localSize[0] = (width % wg_size == 0) ? wg_size : 1;
    localSize[1] = (height % wg_size == 0) ? wg_size : 1;
    
    err = clEnqueueNDRangeKernel(commandQueue, grayscaleKernel, 2, nullptr, globalSize, localSize, 0, nullptr, &grayscaleEvent);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de l'exécution du kernel de conversion en niveaux de gris." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Attendre que le kernel termine
    clFinish(commandQueue);
    
    // Récupérer les informations de profilage
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double grayscaleTime = (time_end - time_start) / 1000000.0; // Conversion en millisecondes
    
    std::cout << "Temps d'exécution du kernel de conversion en niveaux de gris: " << grayscaleTime << " ms" << std::endl;
    
    // Libérer l'événement
    clReleaseEvent(grayscaleEvent);
    
    // Lire les résultats
    std::vector<unsigned char> outputData(width * height);
    err = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, outputSize, outputData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la lecture des résultats de la conversion en niveaux de gris." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Libérer les ressources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(grayscaleKernel);
    clReleaseProgram(program);
    
    return outputData;
}

// Fonction pour redimensionner une image en niveaux de gris avec OpenCL
std::vector<unsigned char> resizeGrayscaleGPU(cl_context context, cl_command_queue commandQueue,
                                              const std::vector<unsigned char>& inputImage, 
                                              int src_width, int src_height,
                                              int dst_width, int dst_height) {
    cl_int err;
    
    // Lire le fichier kernel
    std::ifstream kernelFile("src/kernels/image_processing_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Impossible d'ouvrir le fichier du kernel de traitement d'image." << std::endl;
        return std::vector<unsigned char>();
    }
    
    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();
    
    // Créer programme et compiler
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur compilation programme de traitement d'image." << std::endl;
        
        // Récupérer le device associé à la commandQueue
        cl_device_id device;
        err = clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Erreur lors de la récupération du device." << std::endl;
            clReleaseProgram(program);
            return std::vector<unsigned char>();
        }
        
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << buildLog.data() << std::endl;
        
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer kernel
    cl_kernel resizeKernel = clCreateKernel(program, "resize_grayscale", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création du kernel de redimensionnement." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer les buffers
    size_t inputSize = src_width * src_height * sizeof(unsigned char);
    size_t outputSize = dst_width * dst_height * sizeof(unsigned char);
    
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, inputSize, nullptr, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création des buffers pour le redimensionnement." << std::endl;
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Copier les données d'entrée
    err = clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, inputSize, inputImage.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la copie des données d'entrée pour le redimensionnement." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Configurer les arguments du kernel
    err  = clSetKernelArg(resizeKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(resizeKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(resizeKernel, 2, sizeof(int), &src_width);
    err |= clSetKernelArg(resizeKernel, 3, sizeof(int), &src_height);
    err |= clSetKernelArg(resizeKernel, 4, sizeof(int), &dst_width);
    err |= clSetKernelArg(resizeKernel, 5, sizeof(int), &dst_height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la configuration des arguments du kernel de redimensionnement." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Créer un événement pour le profilage
    cl_event resizeEvent;
    
    // Exécuter le kernel
    size_t globalSize[2] = { (size_t)dst_width, (size_t)dst_height };
    
    // Déterminer la taille optimale du work-group
    cl_device_id device;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    size_t wg_size = 16; // Valeur par défaut
    while (wg_size * wg_size > max_work_group_size) {
        wg_size /= 2;
    }
    
    size_t localSize[2];
    localSize[0] = (dst_width % wg_size == 0) ? wg_size : 1;
    localSize[1] = (dst_height % wg_size == 0) ? wg_size : 1;
    
    err = clEnqueueNDRangeKernel(commandQueue, resizeKernel, 2, nullptr, globalSize, localSize, 0, nullptr, &resizeEvent);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de l'exécution du kernel de redimensionnement." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Attendre que le kernel termine
    clFinish(commandQueue);
    
    // Récupérer les informations de profilage
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double resizeTime = (time_end - time_start) / 1000000.0; // Conversion en millisecondes
    
    std::cout << "Temps d'exécution du kernel de redimensionnement: " << resizeTime << " ms" << std::endl;
    
    // Libérer l'événement
    clReleaseEvent(resizeEvent);
    
    // Lire les résultats
    std::vector<unsigned char> outputData(dst_width * dst_height);
    err = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, outputSize, outputData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la lecture des résultats du redimensionnement." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Libérer les ressources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(resizeKernel);
    clReleaseProgram(program);
    
    return outputData;
}
