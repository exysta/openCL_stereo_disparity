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


// Function to load an image
Image load_image(const char* filename) {
    Image img;
    int channels;
    unsigned char* data = stbi_load(filename, &img.width, &img.height, &channels, 1);
    
    if(!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        exit(1);
    }
    
    img.data.assign(data, data + img.width * img.height);
    stbi_image_free(data);
    return img;
}

// Function to save disparity image
void save_disparity(const char* filename, const Image& img) {
    stbi_write_png(filename, img.width, img.height, 1, img.data.data(), img.width);
}

// Pre-compute window values for both images
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

// CPU version of stereo disparity algorithm
Image computeDisparityCPU(const Image& left, const Image& right, int max_disp, int win_size) {
    int width = left.width;
    int height = left.height;
    Image disparity{width, height, std::vector<unsigned char>(width * height, 0)};
    
    // Pre-compute means and standard deviations of windows
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

// Function to print profiling information in a formatted way
void printProfilingInfo(const std::string& kernelName, double executionTime, size_t globalWorkSize) {
    std::cout << "\n====== Profiling of kernel '" << kernelName << "' ======" << std::endl;
    std::cout << "Execution time: " << std::fixed << std::setprecision(3) << executionTime << " ms" << std::endl;
    std::cout << "Global work size: " << globalWorkSize << " elements" << std::endl;
    std::cout << "Average time per element: " << std::fixed << std::setprecision(6) << (executionTime / globalWorkSize) << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (globalWorkSize / executionTime * 1000) << " elements/s" << std::endl;
    std::cout << "===========================================" << std::endl;
}

// OpenCL stereo disparity implementation
std::vector<unsigned char> computeStereoDisparity(cl_context context, cl_command_queue commandQueue, 
                                                  cl_mem imageBuffer0, cl_mem imageBuffer1, 
                                                  int width, int height, int channels) {
    cl_int err;

    // Read kernel file
    std::ifstream kernelFile("src/kernels/disparity_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Unable to open kernel file." << std::endl;
        return std::vector<unsigned char>();
    }

    std::string src((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = src.c_str();

    // Create program and compile
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        std::cerr << "Program compilation error." << std::endl;
        
        // Get device associated with commandQueue
        cl_device_id device;
        err = clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error retrieving device." << std::endl;
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

    // Create kernels
    cl_kernel disparityKernel = clCreateKernel(program, "disparity", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating disparity kernel." << std::endl;
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    cl_kernel smoothingKernel = clCreateKernel(program, "smooth_disparity", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating smoothing kernel." << std::endl;
        clReleaseKernel(disparityKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Create buffers for results
    cl_mem disparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    cl_mem smoothedDisparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    
    // Print buffer sizes
    std::cout << "\nBuffer sizes:" << std::endl;
    std::cout << "Left image buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Right image buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Left disparity buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Right disparity buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Cross-checked buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Occlusion filled buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Smoothed disparity buffer: " << (width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    std::cout << "Total buffer size: " << (7 * width * height * sizeof(unsigned char) / 1024.0) << " KB" << std::endl;
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating disparity buffer." << std::endl;
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Disparity kernel parameters
    err  = clSetKernelArg(disparityKernel, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(disparityKernel, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(disparityKernel, 2, sizeof(cl_mem), &disparityBuffer);
    err |= clSetKernelArg(disparityKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(disparityKernel, 4, sizeof(int), &height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error setting disparity kernel arguments." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Create additional kernels for cross-check and occlusion filling
    cl_kernel dispRightToLeftKernel = clCreateKernel(program, "disparity_right_to_left", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating right-to-left disparity kernel." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    cl_kernel crossCheckKernel = clCreateKernel(program, "cross_check", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating cross-check kernel." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseKernel(dispRightToLeftKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    cl_kernel fillOcclusionsKernel = clCreateKernel(program, "fill_occlusions", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating occlusion filling kernel." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseKernel(dispRightToLeftKernel);
        clReleaseKernel(crossCheckKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Create additional buffers for right-to-left disparity and intermediate steps
    cl_mem rightToLeftDispBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    cl_mem crossCheckedDispBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    cl_mem occlusionFilledBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating additional buffers." << std::endl;
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseMemObject(rightToLeftDispBuffer);
        clReleaseMemObject(crossCheckedDispBuffer);
        clReleaseMemObject(occlusionFilledBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseKernel(dispRightToLeftKernel);
        clReleaseKernel(crossCheckKernel);
        clReleaseKernel(fillOcclusionsKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }
    
    // Set kernel arguments for right-to-left disparity calculation
    err  = clSetKernelArg(dispRightToLeftKernel, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(dispRightToLeftKernel, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(dispRightToLeftKernel, 2, sizeof(cl_mem), &rightToLeftDispBuffer);
    err |= clSetKernelArg(dispRightToLeftKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(dispRightToLeftKernel, 4, sizeof(int), &height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error setting right-to-left disparity kernel arguments." << std::endl;
        // Release all resources
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseMemObject(rightToLeftDispBuffer);
        clReleaseMemObject(crossCheckedDispBuffer);
        clReleaseMemObject(occlusionFilledBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseKernel(dispRightToLeftKernel);
        clReleaseKernel(crossCheckKernel);
        clReleaseKernel(fillOcclusionsKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Smoothing kernel parameters
    err  = clSetKernelArg(smoothingKernel, 0, sizeof(cl_mem), &occlusionFilledBuffer); // Changed to use occlusion filled buffer
    err |= clSetKernelArg(smoothingKernel, 1, sizeof(cl_mem), &smoothedDisparityBuffer);
    err |= clSetKernelArg(smoothingKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(smoothingKernel, 3, sizeof(int), &height);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error setting smoothing kernel arguments." << std::endl;
        // Release all resources
        clReleaseMemObject(disparityBuffer);
        clReleaseMemObject(smoothedDisparityBuffer);
        clReleaseMemObject(rightToLeftDispBuffer);
        clReleaseMemObject(crossCheckedDispBuffer);
        clReleaseMemObject(occlusionFilledBuffer);
        clReleaseKernel(disparityKernel);
        clReleaseKernel(smoothingKernel);
        clReleaseKernel(dispRightToLeftKernel);
        clReleaseKernel(crossCheckKernel);
        clReleaseKernel(fillOcclusionsKernel);
        clReleaseProgram(program);
        return std::vector<unsigned char>();
    }

    // Create event for disparity kernel profiling
    cl_event disparityEvent;
    
    // Execute disparity kernel
    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };
    
    // Determine optimal work-group size
    cl_device_id device;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    
    // Get device information
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    // Get maximum work-item dimensions
    size_t max_work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, nullptr);
    
    // Calculate work-group size that respects constraints
    size_t wg_size_x = 8; // Default smaller value
    size_t wg_size_y = 8;
    
    // Ensure sizes respect device constraints
    wg_size_x = std::min(wg_size_x, max_work_item_sizes[0]);
    wg_size_y = std::min(wg_size_y, max_work_item_sizes[1]);
    
    // Ensure product does not exceed max_work_group_size
    while (wg_size_x * wg_size_y > max_work_group_size) {
        if (wg_size_x > wg_size_y) {
            wg_size_x /= 2;
        } else {
            wg_size_y /= 2;
        }
    }
    
    // Ensure sizes are divisors of global size
    if (width % wg_size_x != 0) {
        wg_size_x = 1; // Use 1 if not divisible
    }
    
    if (height % wg_size_y != 0) {
        wg_size_y = 1; // Use 1 if not divisible
    }
    
    size_t localWorkSize[2] = { wg_size_x, wg_size_y };
    
    std::cout << "\nExecuting disparity kernel with work-group size: " << wg_size_x << "x" << wg_size_y << std::endl;
    std::cout << "Global size: " << width << "x" << height << std::endl;
    std::cout << "Maximum work-group size: " << max_work_group_size << std::endl;
    std::cout << "Maximum work-item sizes: [" << max_work_item_sizes[0] << ", " << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]" << std::endl;
    
    // Try first with calculated work-group size
    err = clEnqueueNDRangeKernel(commandQueue, disparityKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &disparityEvent);
    
    // If fails, try with NULL to let OpenCL decide work-group size
    if (err != CL_SUCCESS) {
        std::cout << "Error with custom work-group size, using default..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, disparityKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &disparityEvent);
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing disparity kernel: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(commandQueue);
    
    // Get profiling information
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(disparityEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(disparityEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double disparityTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    // Print profiling information
    printProfilingInfo("disparityKernel", disparityTime, width * height);
    
    // Release event
    clReleaseEvent(disparityEvent);

    // Create event for right-to-left disparity profiling
    cl_event rightToLeftEvent;
    
    // Execute right-to-left disparity kernel
    std::cout << "\nExecuting right-to-left disparity kernel..." << std::endl;
    err = clEnqueueNDRangeKernel(commandQueue, dispRightToLeftKernel, 2, NULL, globalWorkSize, 
                                localWorkSize, 0, NULL, &rightToLeftEvent);
    
    // If fails, try with NULL to let OpenCL decide work-group size
    if (err != CL_SUCCESS) {
        std::cout << "Error with custom work-group size for right-to-left disparity, using default..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, dispRightToLeftKernel, 2, NULL, globalWorkSize, 
                                    NULL, 0, NULL, &rightToLeftEvent);
    }
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing right-to-left disparity kernel: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(commandQueue);
    
    // Get profiling information
    clGetEventProfilingInfo(rightToLeftEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(rightToLeftEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double rightToLeftTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    // Print profiling information
    printProfilingInfo("rightToLeftDisparityKernel", rightToLeftTime, width * height);
    
    // Release event
    clReleaseEvent(rightToLeftEvent);
    
    // Set up cross-check kernel parameters
    const int max_disp = 50; // Make sure this matches the value in the kernel
    const int cross_check_threshold = 1; // Threshold for cross-check (in pixel units)
    
    err  = clSetKernelArg(crossCheckKernel, 0, sizeof(cl_mem), &disparityBuffer);
    err |= clSetKernelArg(crossCheckKernel, 1, sizeof(cl_mem), &rightToLeftDispBuffer);
    err |= clSetKernelArg(crossCheckKernel, 2, sizeof(cl_mem), &crossCheckedDispBuffer);
    err |= clSetKernelArg(crossCheckKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(crossCheckKernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(crossCheckKernel, 5, sizeof(int), &max_disp);
    err |= clSetKernelArg(crossCheckKernel, 6, sizeof(int), &cross_check_threshold);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error setting cross-check kernel arguments: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Create event for cross-check profiling
    cl_event crossCheckEvent;
    
    // Execute cross-check kernel
    std::cout << "\nExecuting cross-check kernel..." << std::endl;
    err = clEnqueueNDRangeKernel(commandQueue, crossCheckKernel, 2, NULL, globalWorkSize, 
                                localWorkSize, 0, NULL, &crossCheckEvent);
    
    // If fails, try with NULL to let OpenCL decide work-group size
    if (err != CL_SUCCESS) {
        std::cout << "Error with custom work-group size for cross-check, using default..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, crossCheckKernel, 2, NULL, globalWorkSize, 
                                    NULL, 0, NULL, &crossCheckEvent);
    }
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing cross-check kernel: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(commandQueue);
    
    // Get profiling information
    clGetEventProfilingInfo(crossCheckEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(crossCheckEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double crossCheckTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    // Print profiling information
    printProfilingInfo("crossCheckKernel", crossCheckTime, width * height);
    
    // Release event
    clReleaseEvent(crossCheckEvent);
    
    // Save intermediate results (cross-checked disparity map)
    std::vector<unsigned char> crossCheckedMap(width * height);
    err = clEnqueueReadBuffer(commandQueue, crossCheckedDispBuffer, CL_TRUE, 0, 
                            width * height * sizeof(unsigned char), crossCheckedMap.data(), 
                            0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error reading cross-checked disparity map: " << err << std::endl;
    } else {
        // Save the cross-checked disparity map
        Image crossCheckedImage{width, height, crossCheckedMap};
        save_disparity("disparity_cross_checked.png", crossCheckedImage);
        std::cout << "Saved cross-checked disparity map to disparity_cross_checked.png" << std::endl;
    }
    
    // Occlusion filling - multiple iterations for better results
    cl_mem tempBuffers[2] = {crossCheckedDispBuffer, occlusionFilledBuffer};
    int current_src = 0;
    int current_dst = 1;
    
    const int num_fill_iterations = 4;
    
    for (int iter = 0; iter < num_fill_iterations; iter++) {
        // Set up fill_occlusions kernel parameters
        err  = clSetKernelArg(fillOcclusionsKernel, 0, sizeof(cl_mem), &tempBuffers[current_src]);
        err |= clSetKernelArg(fillOcclusionsKernel, 1, sizeof(cl_mem), &tempBuffers[current_dst]);
        err |= clSetKernelArg(fillOcclusionsKernel, 2, sizeof(int), &width);
        err |= clSetKernelArg(fillOcclusionsKernel, 3, sizeof(int), &height);
        err |= clSetKernelArg(fillOcclusionsKernel, 4, sizeof(int), &iter);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Error setting occlusion filling kernel arguments (iteration " << iter << "): " << err << std::endl;
            continue;
        }
        
        // Create event for occlusion filling profiling
        cl_event fillEvent;
        
        // Execute occlusion filling kernel
        std::cout << "\nExecuting occlusion filling kernel (iteration " << iter+1 << "/" << num_fill_iterations << ")..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, fillOcclusionsKernel, 2, NULL, globalWorkSize, 
                                    localWorkSize, 0, NULL, &fillEvent);
        
        // If fails, try with NULL to let OpenCL decide work-group size
        if (err != CL_SUCCESS) {
            std::cout << "Error with custom work-group size for occlusion filling, using default..." << std::endl;
            err = clEnqueueNDRangeKernel(commandQueue, fillOcclusionsKernel, 2, NULL, globalWorkSize, 
                                        NULL, 0, NULL, &fillEvent);
        }
        
        if (err != CL_SUCCESS) {
            std::cerr << "Error executing occlusion filling kernel: " << err << std::endl;
            return std::vector<unsigned char>();
        }
        
        // Wait for kernel to finish
        clFinish(commandQueue);
        
        // Get profiling information
        clGetEventProfilingInfo(fillEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(fillEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        double fillTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
        
        // Print profiling information
        printProfilingInfo("occlusionFillingKernel", fillTime, width * height);
        
        // Release event
        clReleaseEvent(fillEvent);
        
        // Swap source and destination buffers for the next iteration
        std::swap(current_src, current_dst);
    }
    
    // After all iterations, the result is in tempBuffers[current_src]
    // Save the occlusion-filled disparity map
    std::vector<unsigned char> occlusionFilledMap(width * height);
    err = clEnqueueReadBuffer(commandQueue, tempBuffers[current_src], CL_TRUE, 0, 
                            width * height * sizeof(unsigned char), occlusionFilledMap.data(), 
                            0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Error reading occlusion-filled disparity map: " << err << std::endl;
    } else {
        // Save the occlusion-filled disparity map
        Image occlusionFilledImage{width, height, occlusionFilledMap};
        save_disparity("disparity_occlusion_filled.png", occlusionFilledImage);
        std::cout << "Saved occlusion-filled disparity map to disparity_occlusion_filled.png" << std::endl;
    }
    
    // Make sure smoothing kernel uses the current source buffer
    err = clSetKernelArg(smoothingKernel, 0, sizeof(cl_mem), &tempBuffers[current_src]);
    if (err != CL_SUCCESS) {
        std::cerr << "Error updating smoothing kernel source buffer: " << err << std::endl;
    }

    // Create event for smoothing kernel profiling
    cl_event smoothingEvent;
    
    // Execute smoothing kernel
    std::cout << "\nExecuting smoothing kernel..." << std::endl;
    // Try first with calculated work-group size
    err = clEnqueueNDRangeKernel(commandQueue, smoothingKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &smoothingEvent);
    
    // If fails, try with NULL to let OpenCL decide work-group size
    if (err != CL_SUCCESS) {
        std::cout << "Error with custom work-group size for smoothing, using default..." << std::endl;
        err = clEnqueueNDRangeKernel(commandQueue, smoothingKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &smoothingEvent);
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Error executing smoothing kernel: " << err << std::endl;
        return std::vector<unsigned char>();
    }
    
    // Wait for kernel to finish
    clFinish(commandQueue);
    
    // Get profiling information
    clGetEventProfilingInfo(smoothingEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(smoothingEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double smoothingTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    // Print profiling information
    printProfilingInfo("smoothingKernel", smoothingTime, width * height);
    
    // Release event
    clReleaseEvent(smoothingEvent);

    // Create event for buffer read profiling
    cl_event readEvent;
    
    // Read results
    std::vector<unsigned char> disparityMap(width * height);
    err = clEnqueueReadBuffer(commandQueue, smoothedDisparityBuffer, CL_TRUE, 0, width * height * sizeof(unsigned char), disparityMap.data(), 0, NULL, &readEvent);
    
    // Wait for read to finish
    clFinish(commandQueue);
    
    // Get profiling information
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double readTime = (time_end - time_start) / 1000000.0; // Convert to milliseconds
    
    std::cout << "\nBuffer read time: " << readTime << " ms" << std::endl;
    
    // Release event
    clReleaseEvent(readEvent);

    // Release resources
    clReleaseMemObject(disparityBuffer);
    clReleaseMemObject(smoothedDisparityBuffer);
    clReleaseMemObject(rightToLeftDispBuffer);
    clReleaseMemObject(crossCheckedDispBuffer);
    clReleaseMemObject(occlusionFilledBuffer);
    clReleaseKernel(disparityKernel);
    clReleaseKernel(dispRightToLeftKernel);
    clReleaseKernel(crossCheckKernel);
    clReleaseKernel(fillOcclusionsKernel);
    clReleaseKernel(smoothingKernel);
    clReleaseProgram(program);
    
    std::cout << "\nOpenCL disparity calculation completed successfully." << std::endl;
    
    return disparityMap;
}