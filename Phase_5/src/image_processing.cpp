#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"
#include "StereoDisparity.h"

// Convert RGB image to grayscale using OpenCL
std::vector<unsigned char> convertToGrayscaleGPU(cl_context context, cl_command_queue queue,
                                                 unsigned char* input, int width, int height, int channels) {
    cl_int err;
    
    // Read kernel file
    std::ifstream kernelFile("src/kernels/image_processing_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open image processing kernel file." << std::endl;
        return std::vector<unsigned char>();
    }
    
    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();
    
    // Create program and compile
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Image processing program compilation error." << std::endl;
        
        // Get device associated with command queue
        cl_device_id device;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to retrieve device." << std::endl;
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
    
    // Create kernel
    cl_kernel grayscaleKernel = clCreateKernel(program, "rgb_to_grayscale", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create grayscale kernel." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Create buffers
    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, inputSize, nullptr, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffers for grayscale conversion." << std::endl;
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Copy input data
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, inputSize, input, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to copy input data for grayscale conversion." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Set kernel arguments
    err  = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(grayscaleKernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(grayscaleKernel, 4, sizeof(int), &channels);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set grayscale kernel arguments." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Create event for profiling
    cl_event grayscaleEvent;
    
    // Execute kernel
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    
    // Determine optimal work-group size
    cl_device_id device;
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    size_t wg_size = 16; // Default value
    while (wg_size * wg_size > max_work_group_size) {
        wg_size /= 2;
    }
    
    size_t localSize[2];
    localSize[0] = (width % wg_size == 0) ? wg_size : 1;
    localSize[1] = (height % wg_size == 0) ? wg_size : 1;
    
    err = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, nullptr, globalSize, localSize, 0, nullptr, &grayscaleEvent);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute grayscale kernel." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(queue);
    
    // Get profiling information
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(grayscaleEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double grayscaleTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    std::cout << "Grayscale kernel execution time: " << grayscaleTime << " ms" << std::endl;
    
    // Release event
    clReleaseEvent(grayscaleEvent);
    
    // Read results
    std::vector<unsigned char> outputData(width * height);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, outputSize, outputData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read grayscale results." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(grayscaleKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Release resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(grayscaleKernel);
    clReleaseProgram(program);
    
    return outputData;
}

// Resize grayscale image using OpenCL
std::vector<unsigned char> resizeGrayscaleGPU(cl_context context, cl_command_queue queue,
                                            const std::vector<unsigned char>& input,
                                            int src_w, int src_h, int dst_w, int dst_h) {
    cl_int err;
    
    // Read kernel file
    std::ifstream kernelFile("src/kernels/image_processing_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open image processing kernel file." << std::endl;
        return std::vector<unsigned char>();
    }
    
    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();
    
    // Create program and compile
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Image processing program compilation error." << std::endl;
        
        // Get device associated with command queue
        cl_device_id device;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to retrieve device." << std::endl;
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
    
    // Create kernel
    cl_kernel resizeKernel = clCreateKernel(program, "resize_grayscale", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create resize kernel." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Create buffers
    size_t inputSize = src_w * src_h * sizeof(unsigned char);
    size_t outputSize = dst_w * dst_h * sizeof(unsigned char);
    
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, inputSize, nullptr, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffers for resizing." << std::endl;
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Copy input data
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, inputSize, input.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to copy input data for resizing." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Set kernel arguments
    err  = clSetKernelArg(resizeKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(resizeKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(resizeKernel, 2, sizeof(int), &src_w);
    err |= clSetKernelArg(resizeKernel, 3, sizeof(int), &src_h);
    err |= clSetKernelArg(resizeKernel, 4, sizeof(int), &dst_w);
    err |= clSetKernelArg(resizeKernel, 5, sizeof(int), &dst_h);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set resize kernel arguments." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Create event for profiling
    cl_event resizeEvent;
    
    // Execute kernel
    size_t globalSize[2] = { (size_t)dst_w, (size_t)dst_h };
    
    // Determine optimal work-group size
    cl_device_id device;
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    size_t wg_size = 16; // Default value
    while (wg_size * wg_size > max_work_group_size) {
        wg_size /= 2;
    }
    
    size_t localSize[2];
    localSize[0] = (dst_w % wg_size == 0) ? wg_size : 1;
    localSize[1] = (dst_h % wg_size == 0) ? wg_size : 1;
    
    err = clEnqueueNDRangeKernel(queue, resizeKernel, 2, nullptr, globalSize, localSize, 0, nullptr, &resizeEvent);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute resize kernel." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(queue);
    
    // Get profiling information
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(resizeEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double resizeTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    std::cout << "Resize kernel execution time: " << resizeTime << " ms" << std::endl;
    
    // Release event
    clReleaseEvent(resizeEvent);
    
    // Read results
    std::vector<unsigned char> outputData(dst_w * dst_h);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, outputSize, outputData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read resize results." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(resizeKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Release resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(resizeKernel);
    clReleaseProgram(program);
    
    return outputData;
}
