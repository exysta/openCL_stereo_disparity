#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "lodepng.h"

// OpenCL includes
#include <CL/cl.h>

// Structure to hold image data
struct Image {
    unsigned width, height;
    std::vector<unsigned char> data;
};

// Structure to track execution timing
struct ExecutionTiming {
    double readTime;
    double resizeTime;
    double grayscaleTime;
    double filterTime;
    double writeTime;
    double totalTime;
    double setupTime;  // Time for OpenCL setup
};

// Function to check OpenCL errors
void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation 
                 << "', error code: " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to read an OpenCL kernel from a file
std::string readKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to read an image using LodePNG (reused from Exercise 2)
Image readImage(const char* filename) {
    std::vector<unsigned char> image_data;
    Image image;
    unsigned width, height;
  
    // Decode the image
    unsigned error = lodepng::decode(image_data, width, height, filename);
    
    // Check for errors
    if (error) {
        std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return Image();  // Return an empty Image on error
    }
    
    image.data = image_data;
    image.height = height;
    image.width = width;
    
    std::cout << "Successfully read image: " << filename << " (" << width << "x" << height << ")" << std::endl;
    return image;
}

// Function to save a grayscale image using LodePNG (reused from Exercise 2)
void writeImage(const char* filename, const Image& grayscale) {
    // Convert the grayscale image to RGBA for saving
    std::vector<unsigned char> output_data(grayscale.width * grayscale.height * 4);
    
    for (unsigned y = 0; y < grayscale.height; y++) {
        for (unsigned x = 0; x < grayscale.width; x++) {
            unsigned gray_pos = y * grayscale.width + x;
            unsigned rgba_pos = (y * grayscale.width + x) * 4;
            
            // Set R, G, B to the grayscale value, and A to 255 (fully opaque)
            output_data[rgba_pos] = grayscale.data[gray_pos];
            output_data[rgba_pos + 1] = grayscale.data[gray_pos];
            output_data[rgba_pos + 2] = grayscale.data[gray_pos];
            output_data[rgba_pos + 3] = 255;
        }
    }
    
    // Encode and save the image
    unsigned error = lodepng::encode(filename, output_data, grayscale.width, grayscale.height);
    
    // Check for errors
    if (error) {
        std::cout << "Encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    } else {
        std::cout << "Successfully saved image: " << filename << std::endl;
    }
}

// Function to save an RGBA image using LodePNG (reused from Exercise 2)
void writeRGBAImage(const char* filename, const Image& image) {
    // Encode and save the image
    unsigned error = lodepng::encode(filename, image.data, image.width, image.height);
    
    // Check for errors
    if (error) {
        std::cout << "Encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    } else {
        std::cout << "Successfully saved image: " << filename << std::endl;
    }
}

// Function to initialize OpenCL
void initializeOpenCL(cl_context& context, cl_command_queue& queue, cl_device_id& device) {
    cl_int error;
    cl_platform_id platform;
    
    // Get platform
    error = clGetPlatformIDs(1, &platform, NULL);
    checkError(error, "Getting platform ID");
    
    // Get device
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (error != CL_SUCCESS) {
        // If no GPU is available, try CPU
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        checkError(error, "Getting device ID");
        std::cout << "Using CPU device (no GPU found)" << std::endl;
    } else {
        std::cout << "Using GPU device" << std::endl;
    }
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    checkError(error, "Creating context");
    
    // Create command queue with profiling enabled
    #ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(context, device, NULL, &error);
    #else
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    #endif
    
    checkError(error, "Creating command queue");
}

// Function to create and build a program from kernel source
cl_program buildProgram(cl_context context, cl_device_id device, const std::string& source) {
    cl_int error;
    
    // Create program
    const char* source_str = source.c_str();
    size_t source_size = source.length();
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &error);
    checkError(error, "Creating program");
    
    // Build program
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        // If there was a build error, print the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        
        std::cerr << "Error building program. Build log:" << std::endl;
        std::cerr << log.data() << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return program;
}

// OpenCL kernel for resizing an image
const char* resizeKernelSource = R"(
__kernel void resizeImage(
    __global const uchar4* input,
    __global uchar4* output,
    const unsigned int inputWidth,
    const unsigned int factor)
{
    // Get global position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int outputWidth = inputWidth / factor;
    
    // Calculate input position
    const int inputX = x * factor;
    const int inputY = y * factor;
    
    // Copy pixel from input to output
    output[y * outputWidth + x] = input[inputY * inputWidth + inputX];
}
)";

// OpenCL kernel for converting an image to grayscale
const char* grayscaleKernelSource = R"(
__kernel void convertToGrayscale(
    __global const uchar4* input,
    __global uchar* output,
    const unsigned int width)
{
    // Get global position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int index = y * width + x;
    
    // Get pixel
    uchar4 pixel = input[index];
    
    // Convert to grayscale using Y = 0.2126R + 0.7152G + 0.0722B
    float gray = 0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z;
    
    // Store in output
    output[index] = (uchar)gray;
}
)";

// OpenCL kernel for applying a 5x5 filter
const char* filterKernelSource = R"(
__kernel void applyFilter(
    __global const uchar* input,
    __global uchar* output,
    const unsigned int width,
    const unsigned int height)
{
    // Get global position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Skip boundary pixels
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) {
        // Just copy the input for boundary pixels
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    // Define the 5x5 Gaussian blur kernel
    const float kernel[25] = {
        1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f,
        4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
        6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f,
        4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
        1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f
    };
    
    // Apply the kernel
    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            // Calculate the position to sample from the input image
            int sampleX = x + kx;
            int sampleY = y + ky;
            
            // Get the sample value
            uchar sample = input[sampleY * width + sampleX];
            
            // Apply kernel weight
            sum += sample * kernel[(ky + 2) * 5 + (kx + 2)];
        }
    }
    
    // Store the result
    output[y * width + x] = (uchar)sum;
}
)";

// Function to resize an image using OpenCL
Image resizeImageOpenCL(
    const Image& original,
    cl_context context,
    cl_command_queue queue,
    cl_device_id device,
    cl_ulong& executionTime)
{
    cl_int error;
    
    // Calculate new dimensions
    unsigned int factor = 4;  // Resize by factor 4
    unsigned int newWidth = original.width / factor;
    unsigned int newHeight = original.height / factor;
    
    // Create result image
    Image resized;
    resized.width = newWidth;
    resized.height = newHeight;
    resized.data.resize(newWidth * newHeight * 4);  // 4 bytes per pixel for RGBA
    
    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        original.data.size() * sizeof(unsigned char), original.data.data(), &error);
    checkError(error, "Creating input buffer for resize");
    
    cl_mem outputBuffer = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        resized.data.size() * sizeof(unsigned char), NULL, &error);
    checkError(error, "Creating output buffer for resize");
    
    // Create and build program
    cl_program program = buildProgram(context, device, resizeKernelSource);
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "resizeImage", &error);
    checkError(error, "Creating resize kernel");
    
    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    error |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &original.width);
    error |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &factor);
    checkError(error, "Setting resize kernel arguments");
    
    // Define work sizes
    size_t globalSize[2] = {newWidth, newHeight};
    
    // Execute kernel
    cl_event event;
    error = clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
    checkError(error, "Enqueuing resize kernel");
    
    // Wait for kernel to complete
    error = clWaitForEvents(1, &event);
    checkError(error, "Waiting for resize kernel");
    
    // Get profiling information
    cl_ulong timeStart, timeEnd;
    error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                  sizeof(timeStart), &timeStart, NULL);
    error |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                                   sizeof(timeEnd), &timeEnd, NULL);
    checkError(error, "Getting profiling info for resize");
    
    executionTime = timeEnd - timeStart;
    
    // Read back the result
    error = clEnqueueReadBuffer(
        queue, outputBuffer, CL_TRUE, 0,
        resized.data.size() * sizeof(unsigned char), resized.data.data(),
        0, NULL, NULL);
    checkError(error, "Reading resize result");
    
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseEvent(event);
    
    std::cout << "Resized image to " << resized.width << "x" << resized.height 
              << " in " << executionTime / 1000000.0 << " ms" << std::endl;
    
    return resized;
}

// Function to convert image to grayscale using OpenCL
Image convertToGrayscaleOpenCL(
    const Image& color,
    cl_context context,
    cl_command_queue queue,
    cl_device_id device,
    cl_ulong& executionTime)
{
    cl_int error;
    
    // Create result image
    Image grayscale;
    grayscale.width = color.width;
    grayscale.height = color.height;
    grayscale.data.resize(grayscale.width * grayscale.height);  // 1 byte per pixel for grayscale
    
    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        color.data.size() * sizeof(unsigned char), color.data.data(), &error);
    checkError(error, "Creating input buffer for grayscale");
    
    cl_mem outputBuffer = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        grayscale.data.size() * sizeof(unsigned char), NULL, &error);
    checkError(error, "Creating output buffer for grayscale");
    
    // Create and build program
    cl_program program = buildProgram(context, device, grayscaleKernelSource);
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "convertToGrayscale", &error);
    checkError(error, "Creating grayscale kernel");
    
    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    error |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &color.width);
    checkError(error, "Setting grayscale kernel arguments");
    
    // Define work sizes
    size_t globalSize[2] = {color.width, color.height};
    
    // Execute kernel
    cl_event event;
    error = clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
    checkError(error, "Enqueuing grayscale kernel");
    
    // Wait for kernel to complete
    error = clWaitForEvents(1, &event);
    checkError(error, "Waiting for grayscale kernel");
    
    // Get profiling information
    cl_ulong timeStart, timeEnd;
    error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                  sizeof(timeStart), &timeStart, NULL);
    error |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                                   sizeof(timeEnd), &timeEnd, NULL);
    checkError(error, "Getting profiling info for grayscale");
    
    executionTime = timeEnd - timeStart;
    
    // Read back the result
    error = clEnqueueReadBuffer(
        queue, outputBuffer, CL_TRUE, 0,
        grayscale.data.size() * sizeof(unsigned char), grayscale.data.data(),
        0, NULL, NULL);
    checkError(error, "Reading grayscale result");
    
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseEvent(event);
    
    std::cout << "Converted image to grayscale in " 
              << executionTime / 1000000.0 << " ms" << std::endl;
    
    return grayscale;
}

// Function to apply filter using OpenCL
Image applyFilterOpenCL(
    const Image& grayscale,
    cl_context context,
    cl_command_queue queue,
    cl_device_id device,
    cl_ulong& executionTime)
{
    cl_int error;
    
    // Create result image
    Image filtered;
    filtered.width = grayscale.width;
    filtered.height = grayscale.height;
    filtered.data.resize(filtered.width * filtered.height);
    
    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        grayscale.data.size() * sizeof(unsigned char), grayscale.data.data(), &error);
    checkError(error, "Creating input buffer for filter");
    
    cl_mem outputBuffer = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        filtered.data.size() * sizeof(unsigned char), NULL, &error);
    checkError(error, "Creating output buffer for filter");
    
    // Create and build program
    cl_program program = buildProgram(context, device, filterKernelSource);
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "applyFilter", &error);
    checkError(error, "Creating filter kernel");
    
    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    error |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &filtered.width);
    error |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &filtered.height);
    checkError(error, "Setting filter kernel arguments");
    
    // Define work sizes
    size_t globalSize[2] = {filtered.width, filtered.height};
    
    // Execute kernel
    cl_event event;
    error = clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
    checkError(error, "Enqueuing filter kernel");
    
    // Wait for kernel to complete
    error = clWaitForEvents(1, &event);
    checkError(error, "Waiting for filter kernel");
    
    // Get profiling information
    cl_ulong timeStart, timeEnd;
    error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                  sizeof(timeStart), &timeStart, NULL);
    error |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                                   sizeof(timeEnd), &timeEnd, NULL);
    checkError(error, "Getting profiling info for filter");
    
    executionTime = timeEnd - timeStart;
    
    // Read back the result
    error = clEnqueueReadBuffer(
        queue, outputBuffer, CL_TRUE, 0,
        filtered.data.size() * sizeof(unsigned char), filtered.data.data(),
        0, NULL, NULL);
    checkError(error, "Reading filter result");
    
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseEvent(event);
    
    std::cout << "Applied 5x5 Gaussian blur filter in " 
              << executionTime / 1000000.0 << " ms" << std::endl;
    
    return filtered;
}

// Function to display profiling information
void displayProfilingInfo(const ExecutionTiming& timing, 
                         cl_ulong resizeTime, cl_ulong grayscaleTime, cl_ulong filterTime) {
    std::cout << "\n===== Profiling Information =====" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Setup time:           " << timing.setupTime << " seconds" << std::endl;
    std::cout << "Read image:           " << timing.readTime << " seconds" << std::endl;
    std::cout << "Resize image (OpenCL): " << resizeTime / 1000000000.0 << " seconds" << std::endl;
    std::cout << "Convert to grayscale (OpenCL): " << grayscaleTime / 1000000000.0 << " seconds" << std::endl;
    std::cout << "Apply filter (OpenCL): " << filterTime / 1000000000.0 << " seconds" << std::endl;
    std::cout << "Write image:          " << timing.writeTime << " seconds" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total execution time: " << timing.totalTime << " seconds" << std::endl;
    std::cout << "=================================" << std::endl;
}

int main() {
    ExecutionTiming timing;
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Read the image (reused from Exercise 2)
    auto start = std::chrono::high_resolution_clock::now();
    Image originalImage = readImage("../ressources/im0.png");
    auto end = std::chrono::high_resolution_clock::now();
    timing.readTime = std::chrono::duration<double>(end - start).count();
    
    if (originalImage.data.empty()) {
        std::cerr << "Failed to read the image." << std::endl;
        return 1;
    }
    
    // Setup OpenCL
    start = std::chrono::high_resolution_clock::now();
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    initializeOpenCL(context, queue, device);
    end = std::chrono::high_resolution_clock::now();
    timing.setupTime = std::chrono::duration<double>(end - start).count();
    
    // OpenCL profiling times (in nanoseconds)
    cl_ulong resizeTime, grayscaleTime, filterTime;
    
    // Resize the image using OpenCL
    Image resizedImage = resizeImageOpenCL(originalImage, context, queue, device, resizeTime);
    
    // Save the resized image for visualization
    writeRGBAImage("image_0_resized_opencl.png", resizedImage);
    
    // Convert to grayscale using OpenCL
    Image grayscaleImage = convertToGrayscaleOpenCL(resizedImage, context, queue, device, grayscaleTime);
    
    // Save the grayscale image
    writeImage("image_0_grayscale_opencl.png", grayscaleImage);
    
    // Apply the filter using OpenCL
    Image filteredImage = applyFilterOpenCL(grayscaleImage, context, queue, device, filterTime);
    
    // Write the final image
    start = std::chrono::high_resolution_clock::now();
    writeImage("image_0_bw_opencl.png", filteredImage);
    end = std::chrono::high_resolution_clock::now();
    timing.writeTime = std::chrono::duration<double>(end - start).count();
    
    // Calculate total execution time
    auto end_total = std::chrono::high_resolution_clock::now();
    timing.totalTime = std::chrono::duration<double>(end_total - start_total).count();
    
    // Display profiling information
    displayProfilingInfo(timing, resizeTime, grayscaleTime, filterTime);
    
    // Clean up OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}