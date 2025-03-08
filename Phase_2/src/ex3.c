#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include "lodepng.h"

// Function to measure time in microseconds
double get_time_in_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

// Function to read an image using LodePNG
void readImage(const char* filename, unsigned char** data, int* width, int* height) {
    unsigned w, h;
    unsigned error = lodepng_decode32_file(data, &w, &h, filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(1);
    }
    *width = w;
    *height = h;
}

// Function to write a grayscale image to file
void writeImage(const char* filename, unsigned char* grayscale, int width, int height) {
    unsigned char* rgba = (unsigned char*)malloc(width * height * 4);
    if (!rgba) {
        printf("Memory allocation failed for RGBA conversion\n");
        return;
    }
    
    for (int i = 0; i < width * height; i++) {
        int idx = i * 4;
        rgba[idx + 0] = grayscale[i];     // R
        rgba[idx + 1] = grayscale[i];     // G
        rgba[idx + 2] = grayscale[i];     // B
        rgba[idx + 3] = 255;              // A
    }
    
    unsigned error = lodepng_encode32_file(filename, rgba, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }
    
    free(rgba);
}

// Function to check OpenCL errors
void checkError(cl_int err, const char* msg);

int main() {
    double total_start_time = get_time_in_microseconds();
    
    cl_int err;
    cl_uint          num_platforms;
    cl_platform_id  *platforms;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel resizeKernel, grayscaleKernel, filterKernel;
    cl_mem inputBuffer, outputBuffer, grayscaleBuffer;
    size_t globalWorkSize[2], localWorkSize[2];
    int width, height;
    unsigned char* imageData;
    
    double read_start = get_time_in_microseconds();

    // Read input image
    printf("Reading image...\n");
    readImage("../ressources/im0.png", &imageData, &width, &height);
    double read_end = get_time_in_microseconds();
    printf("Image loaded: %dx%d pixels (%.2f ms)\n", width, height, (read_end - read_start) / 1000.0);

    // PLATFORM
    // In this example we will only consider one platform
    //
    int num_max_platforms = 1;
    err = clGetPlatformIDs(num_max_platforms, NULL, &num_platforms);
    printf("Num platforms detected: %d\n", num_platforms);

    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_max_platforms, platforms, &num_platforms);

    if(num_platforms < 1)
    {
        printf("No platform detected, exit\n");
        exit(1);
    }

    //DEVICE (could be CL_DEVICE_TYPE_GPU)
    //
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");

    // Create buffers
    double buffer_start = get_time_in_microseconds();
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, imageData, &err);
    checkError(err, "clCreateBuffer (input)");
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (width / 4) * (height / 4) * 4, NULL, &err);
    checkError(err, "clCreateBuffer (output)");
    grayscaleBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height, NULL, &err);
    checkError(err, "clCreateBuffer (grayscale)");
    double buffer_end = get_time_in_microseconds();
    double buffer_time = (buffer_end - buffer_start) / 1000.0;

    // Load and build program
    FILE* kernelFile = fopen("./exercise3_kernels.cl", "r");
    if (!kernelFile) {
        fprintf(stderr, "Failed to open kernel file\n");
        exit(EXIT_FAILURE);
    }
    fseek(kernelFile, 0, SEEK_END);
    size_t kernelSize = ftell(kernelFile);
    rewind(kernelFile);
    char* kernelSource = (char*)malloc(kernelSize + 1);
    fread(kernelSource, 1, kernelSize, kernelFile);
    kernelSource[kernelSize] = '\0';
    fclose(kernelFile);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernelSize, &err);
    checkError(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    resizeKernel = clCreateKernel(program, "resizeImage", &err);
    checkError(err, "clCreateKernel (resize)");
    grayscaleKernel = clCreateKernel(program, "grayscaleImage", &err);
    checkError(err, "clCreateKernel (grayscale)");
    filterKernel = clCreateKernel(program, "applyFilter", &err);
    checkError(err, "clCreateKernel (filter)");

    // Set kernel arguments and execute kernels with profiling
    cl_event resizeEvent, grayscaleEvent, filterEvent;
    
    // Execute resize kernel
    printf("\nExecuting resize kernel...\n");
    err = clSetKernelArg(resizeKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(resizeKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(resizeKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(resizeKernel, 3, sizeof(int), &height);
    checkError(err, "Setting resize kernel arguments");
    
    globalWorkSize[0] = width / 4;
    globalWorkSize[1] = height / 4;
    // Ensure workgroup size divides global size evenly
    localWorkSize[0] = 8;
    localWorkSize[1] = 8;
    // Adjust global size to be multiple of local size
    globalWorkSize[0] = ((globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = ((globalWorkSize[1] + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
    
    err = clEnqueueNDRangeKernel(queue, resizeKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &resizeEvent);
    checkError(err, "Enqueueing resize kernel");
    clFinish(queue);
    
    // Execute grayscale kernel
    printf("\nExecuting grayscale kernel...\n");
    err = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &grayscaleBuffer);
    err |= clSetKernelArg(grayscaleKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(grayscaleKernel, 3, sizeof(int), &height);
    checkError(err, "Setting grayscale kernel arguments");
    
    err = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, NULL, globalWorkSize, localWorkSize, 1, &resizeEvent, &grayscaleEvent);
    checkError(err, "Enqueueing grayscale kernel");
    clFinish(queue);
    
    // Execute filter kernel
    printf("\nExecuting filter kernel...\n");
    cl_mem filteredBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (width / 4) * (height / 4), NULL, &err);
    checkError(err, "Creating filtered buffer");
    
    err = clSetKernelArg(filterKernel, 0, sizeof(cl_mem), &grayscaleBuffer);
    err |= clSetKernelArg(filterKernel, 1, sizeof(cl_mem), &filteredBuffer);
    err |= clSetKernelArg(filterKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(filterKernel, 3, sizeof(int), &height);
    checkError(err, "Setting filter kernel arguments");
    
    err = clEnqueueNDRangeKernel(queue, filterKernel, 2, NULL, globalWorkSize, localWorkSize, 1, &grayscaleEvent, &filterEvent);
    checkError(err, "Enqueueing filter kernel");
    clFinish(queue);
    
    // Read back the results
    double readback_start = get_time_in_microseconds();
    unsigned char* result = (unsigned char*)malloc((width / 4) * (height / 4));
    err = clEnqueueReadBuffer(queue, filteredBuffer, CL_TRUE, 0, (width / 4) * (height / 4), result, 1, &filterEvent, NULL);
    checkError(err, "Reading back results");
    double readback_end = get_time_in_microseconds();
    double readback_time = (readback_end - readback_start) / 1000.0;
    
    // Get profiling information
    cl_ulong queued_time, submit_time, start_time, end_time;
    double resize_time, grayscale_time, filter_time;
    
    // Resize kernel timing
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, NULL);
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit_time, NULL);
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    resize_time = (double)(end_time - start_time) * 1.0e-6;
    printf("\nResize kernel timing:\n");
    printf("  Queue to Submit: %.3f µs\n", (submit_time - queued_time) * 1.0e-3);
    printf("  Submit to Start: %.3f µs\n", (start_time - submit_time) * 1.0e-3);
    printf("  Start to End: %.3f µs\n", (end_time - start_time) * 1.0e-3);
    
    // Grayscale kernel timing
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, NULL);
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit_time, NULL);
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    grayscale_time = (double)(end_time - start_time) * 1.0e-6;
    printf("\nGrayscale kernel timing:\n");
    printf("  Queue to Submit: %.3f µs\n", (submit_time - queued_time) * 1.0e-3);
    printf("  Submit to Start: %.3f µs\n", (start_time - submit_time) * 1.0e-3);
    printf("  Start to End: %.3f µs\n", (end_time - start_time) * 1.0e-3);
    
    // Filter kernel timing
    clGetEventProfilingInfo(filterEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, NULL);
    clGetEventProfilingInfo(filterEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit_time, NULL);
    clGetEventProfilingInfo(filterEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(filterEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    filter_time = (double)(end_time - start_time) * 1.0e-6;
    printf("\nFilter kernel timing:\n");
    printf("  Queue to Submit: %.3f µs\n", (submit_time - queued_time) * 1.0e-3);
    printf("  Submit to Start: %.3f µs\n", (start_time - submit_time) * 1.0e-3);
    printf("  Start to End: %.3f µs\n", (end_time - start_time) * 1.0e-3);
    
    // Write the result
    printf("\nWriting output image %dx%d pixels...\n", width/4, height/4);
    writeImage("../output/image_0_bw.png", result, width / 4, height / 4);
    printf("Image processing complete!\n");
    
    double total_end_time = get_time_in_microseconds();
    double total_time = (total_end_time - total_start_time) / 1000.0; // Convert to milliseconds
    
    // Print profiling information
    printf("\n===== OpenCL Profiling Information =====\n");
    printf("Image read time: %.2f ms\n", (read_end - read_start) / 1000.0);
    printf("Buffer creation and data transfer time: %.3f µs\n", buffer_time * 1000.0);
    printf("Resize kernel time: %.3f µs\n", resize_time * 1000.0);
    printf("Grayscale kernel time: %.3f µs\n", grayscale_time * 1000.0);
    printf("Filter kernel time: %.3f µs\n", filter_time * 1000.0);
    printf("Result readback time: %.3f µs\n", readback_time * 1000.0);
    printf("Total kernel time: %.3f µs\n", (resize_time + grayscale_time + filter_time) * 1000.0);
    printf("Total execution time: %.2f ms\n", total_time);
    printf("====================================\n");
    
    // Cleanup
    free(result);
    clReleaseMemObject(filteredBuffer);

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(grayscaleBuffer);
    clReleaseKernel(resizeKernel);
    clReleaseKernel(grayscaleKernel);
    clReleaseKernel(filterKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(imageData);
    free(kernelSource);

    return 0;
}

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed with error code %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}