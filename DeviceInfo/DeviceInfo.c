#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cl_int err;
    cl_uint numPlatforms;

    // Récupérer le nombre de plateformes OpenCL
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        printf("Aucune plateforme OpenCL trouvée.\n");
        return EXIT_FAILURE;
    }

    // Allouer de la mémoire pour les plateformes et récupérer les ID
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur lors de la récupération des plateformes.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    // Choisir la première plateforme
    cl_platform_id platform = platforms[0];

    // Récupérer le premier device de type par défaut
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur lors de la récupération du device.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    // 1. CL_DEVICE_LOCAL_MEM_TYPE
    cl_device_local_mem_type localMemType;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n",
               (localMemType == CL_LOCAL) ? "Local" :
               (localMemType == CL_GLOBAL) ? "Global" : "Inconnu");
    }

    // 2. CL_DEVICE_LOCAL_MEM_SIZE
    cl_ulong localMemSize;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_LOCAL_MEM_SIZE: %llu bytes\n", localMemSize);
    }

    // 3. CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint computeUnits;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", computeUnits);
    }

    // 4. CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint maxClockFreq;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", maxClockFreq);
    }

    // 5. CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    cl_ulong maxConstBufferSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstBufferSize), &maxConstBufferSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %llu bytes\n", maxConstBufferSize);
    }

    // 6. CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t maxWorkGroupSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %zu\n", maxWorkGroupSize);
    }

    // 7. CL_DEVICE_MAX_WORK_ITEM_SIZES
    // On suppose ici un maximum sur 3 dimensions (le standard OpenCL définit 3 dimensions au maximum)
    size_t workItemSizes[3];
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItemSizes), workItemSizes, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
        for (int i = 0; i < 3; i++) {
            printf("%zu ", workItemSizes[i]);
        }
        printf("\n");
    }

    // Libération de la mémoire
    free(platforms);
    return EXIT_SUCCESS;
}