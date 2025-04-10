#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "StereoDisparity.h"
#include "stb_image.h"
#include "stb_image_write.h"


// Fonction pour charger une image
Image load_image(const char* filename) {
    Image img;
    int channels;
    unsigned char* data = stbi_load(filename, &img.width, &img.height, &channels, 1);
    
    if(!data) {
        std::cerr << "Erreur lors du chargement de l'image: " << filename << std::endl;
        exit(1);
    }
    
    img.data.assign(data, data + img.width * img.height);
    stbi_image_free(data);
    return img;
}

// Fonction pour sauvegarder l'image de disparité
void save_disparity(const char* filename, const Image& img) {
    stbi_write_png(filename, img.width, img.height, 1, img.data.data(), img.width);
}

// Pré-calcul des valeurs de fenêtre pour les deux images
void precomputeWindowValues(const Image& img, std::vector<double>& means, std::vector<double>& stdDevs, int win_size) {
    int width = img.width;
    int height = img.height;
    
    means.resize(width * height);
    stdDevs.resize(width * height);
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            double sum = 0;
            double sumSq = 0;
            int count = 0;
            
            for(int wy = std::max(0, y - win_size); wy <= std::min(height - 1, y + win_size); wy++) {
                for(int wx = std::max(0, x - win_size); wx <= std::min(width - 1, x + win_size); wx++) {
                    double val = img.at(wx, wy);
                    sum += val;
                    sumSq += val * val;
                    count++;
                }
            }
            
            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double stdDev = sqrt(std::max(0.0, variance));
            
            means[y * width + x] = mean;
            stdDevs[y * width + x] = stdDev;
        }
    }
}

// Version CPU de l'algorithme de disparité stéréo
Image computeDisparityCPU(const Image& left, const Image& right, int max_disp, int win_size) {
    int width = left.width;
    int height = left.height;
    Image disparity{width, height, std::vector<unsigned char>(width * height, 0)};
    
    // Pré-calcul des moyennes et écarts-types des fenêtres
    std::vector<double> leftMeans, leftStdDevs, rightMeans, rightStdDevs;
    precomputeWindowValues(left, leftMeans, leftStdDevs, win_size);
    precomputeWindowValues(right, rightMeans, rightStdDevs, win_size);
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float max_zncc = -1.0;  // ZNCC range is [-1,1]
            int best_d = 0;
            
            double meanL = leftMeans[y * width + x];
            double stdDevL = leftStdDevs[y * width + x];
            
            // Skip computation if standard deviation is too small (uniform area)
            if(stdDevL < 1.0) continue;

            for(int d = 0; d <= std::min(max_disp, x); d++) {
                int xR = x - d;
                
                double meanR = rightMeans[y * width + xR];
                double stdDevR = rightStdDevs[y * width + xR];
                
                // Skip if right window has uniform texture
                if(stdDevR < 1.0) continue;
                
                // Calculate ZNCC directly
                double numerator = 0;
                int validPoints = 0;
                
                for(int wy = std::max(0, y - win_size); wy <= std::min(height - 1, y + win_size); wy++) {
                    // Process window rows in continuous memory blocks when possible
                    int wxStart = std::max(0, x - win_size);
                    int wxEnd = std::min(width - 1, x + win_size);
                    int wxRStart = wxStart - d;
                    
                    // Adjust for window boundaries
                    if(wxRStart < 0) {
                        wxStart += (0 - wxRStart);
                        wxRStart = 0;
                    }
                    
                    int wxREnd = wxEnd - d;
                    if(wxREnd >= width) {
                        wxEnd -= (wxREnd - width + 1);
                        wxREnd = width - 1;
                    }
                    
                    for(int wx = wxStart; wx <= wxEnd; wx++) {
                        int wxR = wx - d;
                        double diffL = left.at(wx, wy) - meanL;
                        double diffR = right.at(wxR, wy) - meanR;
                        numerator += diffL * diffR;
                        validPoints++;
                    }
                }
                
                if(validPoints > 0) {
                    double zncc = numerator / (validPoints * stdDevL * stdDevR);
                    if(zncc > max_zncc) {
                        max_zncc = zncc;
                        best_d = d;
                    }
                }
            }
            
            // Normalize disparity value to [0,255]
            disparity.data[y * width + x] = static_cast<unsigned char>((best_d * 255) / max_disp);
        }
    }
    
    return disparity;
}

// Fonction pour afficher les résultats de profilage de manière formatée
void printProfilingInfo(const std::string& kernelName, double executionTime, size_t globalWorkSize) {
    std::cout << "\n====== Profilage du kernel '" << kernelName << "' ======" << std::endl;
    std::cout << "Temps d'exécution: " << std::fixed << std::setprecision(3) << executionTime << " ms" << std::endl;
    std::cout << "Taille globale du travail: " << globalWorkSize << " éléments" << std::endl;
    std::cout << "Temps moyen par élément: " << std::fixed << std::setprecision(6) << (executionTime / globalWorkSize) << " ms" << std::endl;
    std::cout << "Débit: " << std::fixed << std::setprecision(2) << (globalWorkSize / executionTime * 1000) << " éléments/s" << std::endl;
    std::cout << "===========================================" << std::endl;
}

// Version OpenCL de l'algorithme de disparité stéréo
std::vector<unsigned char> computeStereoDisparity(cl_context context, cl_command_queue commandQueue, cl_mem imageBuffer0, cl_mem imageBuffer1, int width, int height, int channels) {
    cl_int err;

    // Lire le fichier kernel
    std::ifstream kernelFile("src/kernels/disparity_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Impossible d'ouvrir le fichier du kernel." << std::endl;
        return std::vector<unsigned char>();
    }

    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();

    // Créer programme et compiler
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        std::cerr << "Erreur compilation programme." << std::endl;
        
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

    // Créer kernels
    cl_kernel disparityKernel = clCreateKernel(program, "disparity", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création du kernel de disparité." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    cl_kernel smoothingKernel = clCreateKernel(program, "smooth_disparity", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création du kernel de lissage." << std::endl;
        clReleaseKernel(disparityKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Créer des buffers pour les résultats
    cl_mem disparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    cl_mem smoothedDisparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    
    // Afficher la taille des buffers
    std::cout << "\nTaille des buffers:" << std::endl;
    std::cout << "Buffer image gauche: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Buffer image droite: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Buffer disparité: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Buffer disparité lissée: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Taille totale des buffers: " << (4 * width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la création du buffer de disparité." << std::endl;
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Paramètres kernel de disparité
    err  = clSetKernelArg(disparityKernel, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(disparityKernel, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(disparityKernel, 2, sizeof(cl_mem), &disparityBuffer);
    err |= clSetKernelArg(disparityKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(disparityKernel, 4, sizeof(int), &height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la configuration des arguments du kernel de disparité." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Paramètres kernel de lissage
    err  = clSetKernelArg(smoothingKernel, 0, sizeof(cl_mem), &disparityBuffer);
    err |= clSetKernelArg(smoothingKernel, 1, sizeof(cl_mem), &smoothedDisparityBuffer);
    err |= clSetKernelArg(smoothingKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(smoothingKernel, 3, sizeof(int), &height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de la configuration des arguments du kernel de lissage." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Créer un événement pour le profilage du kernel de disparité
    cl_event disparityEvent;
    
    // Exécuter le kernel de disparité
    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };
    
    // Déterminer la taille optimale du work-group
    cl_device_id device;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    
    // Récupérer les informations sur le device
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    // Récupérer les dimensions maximales de work-item
    size_t max_work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, nullptr);
    
    // Calculer une taille de work-group qui respecte les contraintes
    size_t wg_size_x = 8; // Valeur par défaut plus petite
    size_t wg_size_y = 8;
    
    // S'assurer que les tailles respectent les contraintes du device
    wg_size_x = std::min(wg_size_x, max_work_item_sizes[0]);
    wg_size_y = std::min(wg_size_y, max_work_item_sizes[1]);
    
    // S'assurer que le produit ne dépasse pas max_work_group_size
    while (wg_size_x * wg_size_y > max_work_group_size) {
        if (wg_size_x > wg_size_y) {
            wg_size_x /= 2;
        } else {
            wg_size_y /= 2;
        }
    }
    
    // S'assurer que les tailles sont des diviseurs de la taille globale
    if (width % wg_size_x != 0) {
        wg_size_x = 1; // Utiliser 1 si pas divisible
    }
    
    if (height % wg_size_y != 0) {
        wg_size_y = 1; // Utiliser 1 si pas divisible
    }
    
    size_t localWorkSize[2] = { wg_size_x, wg_size_y };
    
    std::cout << "\nExécution du kernel de disparité avec work-group size: " << wg_size_x << "x" << wg_size_y << std::endl;
    std::cout << "Taille globale: " << width << "x" << height << std::endl;
    std::cout << "Taille maximale de work-group: " << max_work_group_size << std::endl;
    std::cout << "Tailles maximales de work-item: [" << max_work_item_sizes[0] << ", " << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]" << std::endl;
    
    // Essayer d'abord avec la taille de work-group calculée
    err = clEnqueueNDRangeKernel(commandQueue, disparityKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &disparityEvent);
    
    // Si ça échoue, essayer avec NULL pour laisser OpenCL décider de la taille du work-group
    if (err != CL_SUCCESS) {
        std::cout << "Erreur avec la taille de work-group personnalisée, utilisation de la taille par défaut..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, disparityKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &disparityEvent);
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de l'exécution du kernel de disparité: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Attendre que le kernel termine
    clFinish(commandQueue);
    
    // Récupérer les informations de profilage
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(disparityEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(disparityEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double disparityTime = (time_end - time_start) / 1000000.0; // Conversion en millisecondes
    
    // Afficher les informations de profilage
    printProfilingInfo("disparityKernel", disparityTime, width * height);
    
    // Libérer l'événement
    clReleaseEvent(disparityEvent);

    // Créer un événement pour le profilage du kernel de lissage
    cl_event smoothingEvent;
    
    // Exécuter le kernel de lissage
    std::cout << "\nExécution du kernel de lissage..." << std::endl;
    // Essayer d'abord avec la taille de work-group calculée
    err = clEnqueueNDRangeKernel(commandQueue, smoothingKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &smoothingEvent);
    
    // Si ça échoue, essayer avec NULL pour laisser OpenCL décider de la taille du work-group
    if (err != CL_SUCCESS) {
        std::cout << "Erreur avec la taille de work-group personnalisée pour le lissage, utilisation de la taille par défaut..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, smoothingKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &smoothingEvent);
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur lors de l'exécution du kernel de lissage: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Attendre que le kernel termine
    clFinish(commandQueue);
    
    // Récupérer les informations de profilage
    clGetEventProfilingInfo(smoothingEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(smoothingEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double smoothingTime = (time_end - time_start) / 1000000.0; // Conversion en millisecondes
    
    // Afficher les informations de profilage
    printProfilingInfo("smoothingKernel", smoothingTime, width * height);
    
    // Libérer l'événement
    clReleaseEvent(smoothingEvent);

    // Créer un événement pour le profilage de la lecture du buffer
    cl_event readEvent;
    
    // Lire les résultats
    std::vector<unsigned char> disparityMap(width * height);
    err = clEnqueueReadBuffer(commandQueue, smoothedDisparityBuffer, CL_TRUE, 0, width * height * sizeof(unsigned char), disparityMap.data(), 0, NULL, &readEvent);
    
    // Attendre que la lecture termine
    clFinish(commandQueue);
    
    // Récupérer les informations de profilage
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double readTime = (time_end - time_start) / 1000000.0; // Conversion en millisecondes
    
    std::cout << "\nTemps de lecture du buffer de résultat: " << readTime << " ms" << std::endl;
    
    // Libérer l'événement
    clReleaseEvent(readEvent);

    // Libérer les ressources
    clReleaseMemObject(disparityBuffer);
    clReleaseMemObject(smoothedDisparityBuffer);
    clReleaseKernel(disparityKernel);
    clReleaseKernel(smoothingKernel);
    clReleaseProgram(program);
    
    std::cout << "\nCalcul de disparité OpenCL terminé avec succès." << std::endl;
    
    return disparityMap;
}