#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

#define ARRAY_SIZE  13  // "Hello World!" + '\0'

// Code source du kernel OpenCL sous forme de chaîne de caractères.
const char *kernelSource = 
"__kernel void hello(__global char* output) {      \n"
"   int id = get_global_id(0);                      \n"
"   if (id == 0) {                                  \n"
"       output[0] = 'H';                           \n"
"       output[1] = 'e';                           \n"
"       output[2] = 'l';                           \n"
"       output[3] = 'l';                           \n"
"       output[4] = 'o';                           \n"
"       output[5] = ' ';                           \n"
"       output[6] = 'W';                           \n"
"       output[7] = 'o';                           \n"
"       output[8] = 'r';                           \n"
"       output[9] = 'l';                           \n"
"       output[10] = 'd';                          \n"
"       output[11] = '!';                          \n"
"       output[12] = '\\0';                        \n"
"   }                                             \n"
"}                                                 \n";

int main() {
    cl_int err;

    // 1. Obtenir la plateforme OpenCL
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la récupération de la plateforme\n");
        return EXIT_FAILURE;
    }
    
    // 2. Obtenir un périphérique (GPU de préférence, sinon CPU)
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err != CL_SUCCESS) {
        // Si aucun GPU n'est trouvé, on essaie avec le CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if(err != CL_SUCCESS) {
            printf("Erreur lors de la récupération du périphérique\n");
            return EXIT_FAILURE;
        }
    }
    
    // 3. Créer un contexte OpenCL
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la création du contexte\n");
        return EXIT_FAILURE;
    }
    
    // 4. Créer une file de commandes
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la création de la file de commandes\n");
        return EXIT_FAILURE;
    }
    
    // 5. Créer le programme OpenCL à partir de la source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la création du programme\n");
        return EXIT_FAILURE;
    }
    
    // 6. Compiler (build) le programme
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        // En cas d'erreur, afficher le log de build
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Erreur de build:\n%s\n", log);
        free(log);
        return EXIT_FAILURE;
    }
    
    // 7. Créer le kernel
    cl_kernel kernel = clCreateKernel(program, "hello", &err);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la création du kernel\n");
        return EXIT_FAILURE;
    }
    
    // 8. Créer un buffer pour stocker le message (sur le device)
    char output[ARRAY_SIZE] = {0};
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * ARRAY_SIZE, output, &err);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la création du buffer\n");
        return EXIT_FAILURE;
    }
    
    // 9. Définir l'argument du kernel (le buffer)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la définition de l'argument du kernel\n");
        return EXIT_FAILURE;
    }
    
    // 10. Exécuter le kernel : ici, un seul work-item suffit
    size_t globalSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de l'exécution du kernel\n");
        return EXIT_FAILURE;
    }
    
    // 11. Lire le buffer résultat depuis le device vers le host
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(char) * ARRAY_SIZE, output, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Erreur lors de la lecture du buffer\n");
        return EXIT_FAILURE;
    }
    
    // 12. Afficher le message récupéré
    printf("%s\n", output);
    
    // 13. Libérer les ressources OpenCL
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return EXIT_SUCCESS;
}