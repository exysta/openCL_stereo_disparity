#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cl_int err;
    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        printf("Aucune plateforme OpenCL trouvée.\n");
        return EXIT_FAILURE;
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur lors de la récupération des plateformes.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_platform_id platform = platforms[0];

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur lors de la récupération du device.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_device_local_mem_type localMemType;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n",
               (localMemType == CL_LOCAL) ? "Local" :
               (localMemType == CL_GLOBAL) ? "Global" : "Inconnu");
    }

    cl_ulong localMemSize;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_LOCAL_MEM_SIZE: %llu bytes\n", localMemSize);
    }

    cl_uint computeUnits;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", computeUnits);
    }

    cl_uint maxClockFreq;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", maxClockFreq);
    }

    cl_ulong maxConstBufferSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstBufferSize), &maxConstBufferSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %llu bytes\n", maxConstBufferSize);
    }

    size_t maxWorkGroupSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    if (err == CL_SUCCESS) {
        printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %zu\n", maxWorkGroupSize);
    }

    cl_uint maxDimensions;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxDimensions), &maxDimensions, NULL);
    if (err == CL_SUCCESS) {
        size_t *workItemSizes = malloc(sizeof(size_t) * maxDimensions);
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDimensions, workItemSizes, NULL);
        if (err == CL_SUCCESS) {
            printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
            for (cl_uint i = 0; i < maxDimensions; i++) {
                printf("%zu ", workItemSizes[i]);
            }
            printf("\n");
        }
        free(workItemSizes);
    }
    free(platforms);
    return EXIT_SUCCESS;
}