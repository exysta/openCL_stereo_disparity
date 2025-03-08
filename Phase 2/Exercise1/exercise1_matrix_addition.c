#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <OpenCL/opencl.h>

#define MATRIX_SIZE 100
#define MATRIX_ELEMENTS (MATRIX_SIZE * MATRIX_SIZE)

// Function to initialize a matrix with random values
void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Function to print matrix (for debugging)
void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// C implementation of matrix addition
void add_Matrix(float *matrix_1, float *matrix_2, float *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = matrix_1[i] + matrix_2[i];
    }
}

// Function to measure time in microseconds
double get_time_in_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

// OpenCL kernel for matrix addition
const char *kernel_source = "\n" \
"__kernel void add_matrix(__global float* matrix_1, __global float* matrix_2, __global float* result, int size) {\n" \
"    int id = get_global_id(0);\n" \
"    if (id < size) {\n" \
"        result[id] = matrix_1[id] + matrix_2[id];\n" \
"    }\n" \
"}\n";

// Function to print platform info
void print_platform_info(cl_platform_id platform) {
    char platform_name[1024];
    char platform_vendor[1024];
    char platform_version[1024];
    
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
    
    printf("Platform Name: %s\n", platform_name);
    printf("Platform Vendor: %s\n", platform_vendor);
    printf("Platform Version: %s\n", platform_version);
}

int main() {
    cl_int err;
    // Allocate memory for matrices
    float *matrix_1 = (float *)malloc(MATRIX_ELEMENTS * sizeof(float));
    float *matrix_2 = (float *)malloc(MATRIX_ELEMENTS * sizeof(float));
    float *result_cpu = (float *)malloc(MATRIX_ELEMENTS * sizeof(float));
    float *result_gpu = (float *)malloc(MATRIX_ELEMENTS * sizeof(float));
    
    if (!matrix_1 || !matrix_2 || !result_cpu || !result_gpu) {
        printf("Memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    // Initialize matrices with random values
    initialize_matrix(matrix_1, MATRIX_ELEMENTS);
    initialize_matrix(matrix_2, MATRIX_ELEMENTS);
    
    // CPU implementation
    double start_time_cpu = get_time_in_microseconds();
    add_Matrix(matrix_1, matrix_2, result_cpu, MATRIX_ELEMENTS);
    double end_time_cpu = get_time_in_microseconds();
    double elapsed_time_cpu = end_time_cpu - start_time_cpu;
    
    printf("CPU Matrix Addition Time: %.2f microseconds\n", elapsed_time_cpu);
    
    // OpenCL implementation
    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform ID: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Print platform info
    print_platform_info(platform);
    
    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Try CPU if GPU is not available
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            printf("Error getting device ID: %d\n", err);
            return EXIT_FAILURE;
        }
    }
    
    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating context: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Create command queue with profiling enabled
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating command queue: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating program: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error building program: %s\n", log);
        free(log);
        return EXIT_FAILURE;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "add_matrix", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Create buffers
    cl_mem buffer_matrix_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                          MATRIX_ELEMENTS * sizeof(float), matrix_1, &err);
    cl_mem buffer_matrix_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                          MATRIX_ELEMENTS * sizeof(float), matrix_2, &err);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                        MATRIX_ELEMENTS * sizeof(float), NULL, &err);
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_matrix_1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_matrix_2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
    int size = MATRIX_ELEMENTS;
    err |= clSetKernelArg(kernel, 3, sizeof(int), &size);
    
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Execute kernel
    size_t global_work_size = MATRIX_ELEMENTS;
    cl_event event;
    
    double start_time_gpu = get_time_in_microseconds();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("Error enqueueing kernel: %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Wait for kernel to finish
    clFinish(queue);
    double end_time_gpu = get_time_in_microseconds();
    
    // Read result back
    err = clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, MATRIX_ELEMENTS * sizeof(float), 
                             result_gpu, 0, NULL, NULL);
    
    // Get profiling info
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double opencl_execution_time = (double)(time_end - time_start) / 1000.0; // Convert to microseconds
    
    printf("GPU Matrix Addition Time (Host Measured): %.2f microseconds\n", end_time_gpu - start_time_gpu);
    printf("GPU Matrix Addition Time (OpenCL Profiling): %.2f microseconds\n", opencl_execution_time);
    
    // Verify results (for debugging)
    int correct = 1;
    for (int i = 0; i < MATRIX_ELEMENTS; i++) {
        if (fabs(result_cpu[i] - result_gpu[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    printf("Results %s\n", correct ? "match" : "don't match");
    
    // Clean up
    clReleaseMemObject(buffer_matrix_1);
    clReleaseMemObject(buffer_matrix_2);
    clReleaseMemObject(buffer_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(matrix_1);
    free(matrix_2);
    free(result_cpu);
    free(result_gpu);
    
    return EXIT_SUCCESS;
}
