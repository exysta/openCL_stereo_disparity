#include <CL/cl.h> // Using C API based on your code
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <cmath>    // For std::round, std::max
#include <chrono>   // For timing
#include <numeric>  // For std::accumulate in timing (optional)
#include <algorithm> // For std::max

// Custom OpenCL exception class (keep as is)
class OpenCLException : public std::runtime_error {
public:
    OpenCLException(const std::string& message, cl_int error)
        : std::runtime_error(message + ": Error code " + std::to_string(error)), error_code(error) {}

    cl_int getErrorCode() const { return error_code; }

private:
    cl_int error_code;
};

// Function to check OpenCL errors (keep as is)
void checkError(cl_int error, const std::string& message) {
    if (error != CL_SUCCESS) {
        throw OpenCLException(message, error);
    }
}

// Function to get and print device information (keep as is)
void printDeviceInfo(cl_device_id device) {
    // ... (implementation as before) ...
    cl_int error;
    cl_device_local_mem_type localMemType;
    cl_ulong localMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxClockFreq;
    cl_ulong maxConstantBufferSize;
    size_t maxWorkGroupSize;
    size_t maxWorkItemSizes[3];

    error = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    checkError(error, "Failed to get CL_DEVICE_LOCAL_MEM_TYPE");

    error = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    checkError(error, "Failed to get CL_DEVICE_LOCAL_MEM_SIZE");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    checkError(error, "Failed to get CL_DEVICE_MAX_COMPUTE_UNITS");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, NULL);
    checkError(error, "Failed to get CL_DEVICE_MAX_CLOCK_FREQUENCY");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    checkError(error, "Failed to get CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    checkError(error, "Failed to get CL_DEVICE_MAX_WORK_GROUP_SIZE");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);
    checkError(error, "Failed to get CL_DEVICE_MAX_WORK_ITEM_SIZES");

    std::cout << "==== GPU Device Information ====" << std::endl;
    std::cout << "CL_DEVICE_LOCAL_MEM_TYPE: " << (localMemType == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL") << std::endl;
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << localMemSize << " bytes" << std::endl;
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxComputeUnits << std::endl;
    std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << maxClockFreq << " MHz" << std::endl;
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << maxConstantBufferSize << " bytes" << std::endl;
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxWorkGroupSize << std::endl;
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: [" << maxWorkItemSizes[0] << ", "
              << maxWorkItemSizes[1] << ", " << maxWorkItemSizes[2] << "]" << std::endl;
    std::cout << "================================" << std::endl;
}

// Function to load PNG images (keep as is)
bool loadImages(const std::string& file1, const std::string& file2,
                std::vector<unsigned char>& im0_data, std::vector<unsigned char>& im1_data,
                unsigned& width0, unsigned& height0,
                unsigned& width1, unsigned& height1) {
    // ... (implementation as before) ...
    unsigned error;

    // Load im0.png
    error = lodepng::decode(im0_data, width0, height0, file1);
    if (error) {
        std::cerr << "Error loading " << file1 << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    // Load im1.png
    error = lodepng::decode(im1_data, width1, height1, file2);
    if (error) {
        std::cerr << "Error loading " << file2 << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    // Verify image dimensions
    if (width0 != width1 || height0 != height1) {
        std::cerr << "Error: Image dimensions do not match" << std::endl;
        return false;
    }

    // Print debug info
    std::cout << "Loaded " << file1 << ": " << width0 << " x " << height0 << " (RGBA)" << std::endl;
    std::cout << "Loaded " << file2 << ": " << width1 << " x " << height1 << " (RGBA)" << std::endl;

    return true;
}

// Function to save grayscale PNG (keep as is)
void saveGrayscalePNG(const std::string& filename,
                      const std::vector<unsigned char>& imageData,
                      unsigned width, unsigned height) {
    // ... (implementation as before) ...
     unsigned error = lodepng::encode(filename, imageData, width, height, LCT_GREY, 8);
    if (error) {
        throw std::runtime_error("PNG encoder error saving " + filename + ": " +
                                 std::string(lodepng_error_text(error)));
    }
     std::cout << "Saved grayscale image to " << filename << " (" << width << "x" << height << ")" << std::endl;
}

// Function to initialize OpenCL (keep as is)
bool initOpenCL(cl_platform_id& platform, cl_device_id& device,
                cl_context& context, cl_command_queue& queue) {
    // ... (implementation as before) ...
    try {
        cl_int error;
        cl_uint numPlatforms;

        // Get platform
        error = clGetPlatformIDs(0, nullptr, &numPlatforms);
        checkError(error, "Failed to get number of platforms");

        if (numPlatforms == 0) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return false;
        }

        std::vector<cl_platform_id> platforms(numPlatforms);
        error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        checkError(error, "Failed to get platform IDs");

        std::cout << "Available OpenCL platforms:\n";
        for (cl_uint i = 0; i < numPlatforms; ++i) {
            char platformName[128];
            error = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
            if(error != CL_SUCCESS) continue; // Basic error check
            std::cout << i + 1 << ": " << platformName << "\n";
        }

        // --- Simple Platform/Device Selection (Select first GPU or default) ---
        platform = nullptr;
        device = nullptr;
        for (cl_uint i = 0; i < numPlatforms; ++i) {
            cl_uint numDevices = 0;
            error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
            if (error != CL_SUCCESS || numDevices == 0) {
                 error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &numDevices); // Try default if no GPU
                 if (error != CL_SUCCESS || numDevices == 0) continue;
            }

            std::vector<cl_device_id> devices(numDevices);
            cl_device_type selected_type = CL_DEVICE_TYPE_GPU; // Prefer GPU
            error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
             if (error != CL_SUCCESS) {
                 selected_type = CL_DEVICE_TYPE_DEFAULT;
                 error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, numDevices, devices.data(), nullptr);
                 if (error != CL_SUCCESS) continue;
             }

            platform = platforms[i];
            device = devices[0]; // Select the first available device
            char platformName[128], deviceName[128];
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            std::cout << "Selected Platform: " << platformName << std::endl;
            std::cout << "Selected Device: " << deviceName << " (Type: " << (selected_type == CL_DEVICE_TYPE_GPU ? "GPU" : "Default") << ")" << std::endl;
            break;
        }

        if (!platform || !device) {
             std::cerr << "Error: No suitable OpenCL device found." << std::endl;
             return false;
        }
        // ---------------------------------------------------------------------


        // Print device information
        printDeviceInfo(device);

        // Create context
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        checkError(error, "Failed to create context");

        // Create command queue
        queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
        checkError(error, "Failed to create command queue");

        return true;
    }
    catch (const OpenCLException& e) {
        std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
        return false;
    }
}

// Function to create input/output image objects (keep as is)
bool createImageObjects(cl_context context,
                        unsigned in_width, unsigned in_height,   // Input dimensions
                        unsigned out_width, unsigned out_height, // Output dimensions (pre-calculated)
                        const std::vector<unsigned char>& im0_data,
                        const std::vector<unsigned char>& im1_data,
                        cl_mem& im0_image, cl_mem& im1_image,
                        cl_mem& im0_formated, cl_mem& im1_formated)
{
    // ... (implementation as before) ...
    cl_int error = CL_SUCCESS;
    im0_image = im1_image = im0_formated = im1_formated = nullptr;

    try {
        // Image formats
        const cl_image_format input_format = {CL_RGBA, CL_UNORM_INT8}; // Use UNORM_INT8 for read_imagef
        const cl_image_format output_format = {CL_R, CL_UNSIGNED_INT8}; // Grayscale output (uchar)

        // Input image descriptor
        cl_image_desc input_desc = {};
        input_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        input_desc.image_width = in_width;
        input_desc.image_height = in_height;

        // Create input images
        // Use CL_MEM_COPY_HOST_PTR if data is ready, otherwise alloc and enqueue write later
        im0_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &input_format, &input_desc,
                                  const_cast<unsigned char*>(im0_data.data()), &error);
        checkError(error, "Failed to create input im0_image");

        im1_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &input_format, &input_desc,
                                  const_cast<unsigned char*>(im1_data.data()), &error);
        checkError(error, "Failed to create input im1_image");

        // Output image descriptor (using pre-calculated dimensions)
        cl_image_desc output_desc = {};
        output_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        output_desc.image_width = out_width;
        output_desc.image_height = out_height;

        // Create output images (formatted grayscale)
        // Use CL_MEM_WRITE_ONLY if only kernel writes, CL_MEM_READ_WRITE if reading back to host
        im0_formated = clCreateImage(context, CL_MEM_READ_WRITE, // Kernel writes, host reads
                                     &output_format, &output_desc,
                                     nullptr, &error);
        checkError(error, "Failed to create output im0_formated");

        im1_formated = clCreateImage(context, CL_MEM_READ_WRITE, // Kernel writes, host reads
                                     &output_format, &output_desc,
                                     nullptr, &error);
        checkError(error, "Failed to create output im1_formated");

        std::cout << "Created OpenCL image objects. Input: " << in_width << "x" << in_height
                  << ", Formatted Output: " << out_width << "x" << out_height << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        // Cleanup any created resources on failure
        if (im0_image) clReleaseMemObject(im0_image);
        if (im1_image) clReleaseMemObject(im1_image);
        if (im0_formated) clReleaseMemObject(im0_formated);
        if (im1_formated) clReleaseMemObject(im1_formated);

        std::cerr << "Image object creation failed: " << e.what() << std::endl;
        return false;
    }
}

// Function to verify image objects by reading back a pixel (keep as is)
void verifyImageObjects(cl_command_queue queue, cl_mem im0_image, unsigned width, unsigned height) {
   // ... (implementation as before) ...
    if (!im0_image) {
        std::cerr << "Verification skipped: im0_image is null." << std::endl;
        return;
    }
    if (width == 0 || height == 0) {
         std::cerr << "Verification skipped: Invalid dimensions (0)." << std::endl;
         return;
    }
    try {
        cl_int error;
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {1, 1, 1};  // Read a single pixel
        float pixel_data[4]; // Read as float4 because input format is UNORM_INT8

        // Ensure commands are finished before reading
        error = clFinish(queue);
        checkError(error, "Failed to finish queue before verification read");

        error = clEnqueueReadImage(queue, im0_image, CL_TRUE, origin, region, 0, 0,
                                  pixel_data, 0, nullptr, nullptr);
        checkError(error, "Failed to read from im0_image for verification");

        // Convert float [0,1] back to approx uchar [0,255] for display
        std::cout << "Verification: First pixel from im0_image (GPU, read as float): R=" << static_cast<int>(pixel_data[0] * 255.0f)
                  << ", G=" << static_cast<int>(pixel_data[1] * 255.0f) << ", B=" << static_cast<int>(pixel_data[2] * 255.0f)
                  << ", A=" << static_cast<int>(pixel_data[3] * 255.0f) << std::endl;
    }
    catch (const OpenCLException& e) {
        std::cerr << "Verification failed: Could not read from im0_image: " << e.what() << std::endl;
    }
     catch (const std::exception& e) {
         std::cerr << "Verification failed (std::exception): " << e.what() << std::endl;
     }
}

// Cleanup function (keep as is)
void cleanup(cl_context context, cl_command_queue queue, cl_program program,
             cl_kernel kernel1, cl_kernel kernel2, cl_kernel kernel3,
             std::vector<cl_mem> buffers,
             std::vector<cl_mem> images)
{
    // ... (implementation as before) ...
    std::cout << "Cleaning up OpenCL resources..." << std::endl;
    cl_int err;

    // Flush and finish before releasing
    if (queue) {
        err = clFlush(queue); // Ensure commands are sent
        if(err != CL_SUCCESS) std::cerr << "Warning: clFlush failed during cleanup: " << err << std::endl;
        err = clFinish(queue); // Wait for completion
        if(err != CL_SUCCESS) std::cerr << "Warning: clFinish failed during cleanup: " << err << std::endl;
    }

    // Release kernels
    if (kernel1) clReleaseKernel(kernel1);
    if (kernel2) clReleaseKernel(kernel2);
    if (kernel3) clReleaseKernel(kernel3);

    // Release program
    if (program) clReleaseProgram(program);

    // Release buffers
    for (cl_mem buf : buffers) {
        if (buf) clReleaseMemObject(buf);
    }

    // Release images
    for (cl_mem img : images) {
        if (img) clReleaseMemObject(img);
    }

    // Release queue and context
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
     std::cout << "Cleanup complete." << std::endl;
}

// Function to load OpenCL kernels from file (keep as is)
std::string loadKernelSource(const std::string& filename) {
    // ... (implementation as before) ...
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    // Read file contents into string
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// Function to build program from kernel source (keep as is)
cl_program createAndBuildProgram(cl_context context, cl_device_id device, const std::string& kernelSource) {
    // ... (implementation as before) ...
    cl_int error;
    const char* source = kernelSource.c_str();
    size_t sourceLength = kernelSource.length();

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceLength, &error);
    checkError(error, "Failed to create program");

    // Build program
    std::cout << "Building OpenCL program..." << std::endl;
    // Add options useful for AMD iGPU, like relaxing math precision slightly if needed
    // Forcing CL 1.2 standard is often safe. Remove if issues arise.
    error = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-single-precision-constant -cl-mad-enable", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // If build failed, get build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize + 1); // +1 for null terminator
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        buildLog[logSize] = '\0'; // Ensure null termination

        std::cerr << "--------------------Build Log--------------------\n"
                  << buildLog.data()
                  << "\n-----------------------------------------------" << std::endl;
        clReleaseProgram(program);
        checkError(error, "Failed to build program");
    }
     std::cout << "Program built successfully." << std::endl;
    return program;
}

// Function to create multiple kernels from one program (keep as is)
bool createKernels(cl_program program,
                   cl_kernel& kernel1, const char* name1,
                   cl_kernel& kernel2, const char* name2,
                   cl_kernel& kernel3, const char* name3)
{
    // ... (implementation as before) ...
    cl_int error;
    kernel1 = kernel2 = kernel3 = nullptr;

    try {
        kernel1 = clCreateKernel(program, name1, &error);
        checkError(error, std::string("Failed to create kernel: ") + name1);

        kernel2 = clCreateKernel(program, name2, &error);
        checkError(error, std::string("Failed to create kernel: ") + name2);

        kernel3 = clCreateKernel(program, name3, &error);
        checkError(error, std::string("Failed to create kernel: ") + name3);

        std::cout << "Kernels created: '" << name1 << "', '" << name2 << "', '" << name3 << "'" << std::endl;
        return true;
    } catch (const OpenCLException& e) {
        std::cerr << "Kernel creation failed: " << e.what() << std::endl;
        if (kernel1) clReleaseKernel(kernel1);
        if (kernel2) clReleaseKernel(kernel2);
        // kernel3 will be null if it failed
        return false;
    }
}


int main(int argc, char* argv[]) {

    // --- Configuration ---
    const std::string left_image_path = "../ressources/im0.png";
    const std::string right_image_path = "../ressources/im1.png";
    const std::string kernel_file_path = "kernels.cl"; // Contains all kernels
    const std::string output_dir = "../output/";

    // --- Resize/Filter Configuration ---
    // Factor to resize by. >1.0 = Downscale, <1.0 = Upscale
    const float resizeFactor = 4.0f;
    // Radius for the Box Filter window (Radius 1 -> 3x3, Radius 2 -> 5x5 etc.)
    const int boxFilterRadius = 1;

    // --- Debug Configuration ---
    const bool saveFormattedDebugImages = true; // Set to false to skip saving intermediate images

    // --- ZNCC Configuration ---
    const int win_size = 4;    // Half-width of the ZNCC window (e.g., 4 for a 9x9 window)
    const int max_disp = 260;   // Maximum disparity to search
    // ---------------------

    // Variables for image data
    std::vector<unsigned char> im0_data_rgba, im1_data_rgba;
    unsigned width = 0, height = 0; // Input dimensions
    unsigned width1_tmp, height1_tmp; // For initial load check

    // OpenCL variables
    cl_int err;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    // Kernels (Using Box Filter kernel)
    cl_kernel resizeGrayscaleBoxKernel = nullptr; // Specific name
    cl_kernel precomputeStatsKernel = nullptr;
    cl_kernel computeDisparityKernel = nullptr;

    // OpenCL Memory Objects (Images)
    cl_mem im0_image_rgba = nullptr; // Input RGBA (UNORM_INT8)
    cl_mem im1_image_rgba = nullptr; // Input RGBA (UNORM_INT8)
    cl_mem im0_formated_uchar = nullptr; // Output of resize (uchar grayscale R)
    cl_mem im1_formated_uchar = nullptr; // Output of resize (uchar grayscale R)

    // OpenCL Memory Objects (Buffers)
    cl_mem left_float_buf = nullptr;   // Input for ZNCC (float grayscale)
    cl_mem right_float_buf = nullptr;  // Input for ZNCC (float grayscale)
    cl_mem left_means_buf = nullptr;
    cl_mem left_stddevs_buf = nullptr;
    cl_mem right_means_buf = nullptr;
    cl_mem right_stddevs_buf = nullptr;
    cl_mem disparity_buf = nullptr;     // Final output buffer

    // Keep track of resources for cleanup
    std::vector<cl_mem> cl_buffers;
    std::vector<cl_mem> cl_images;

    // Timers
    std::vector<double> timings_ms; // Store timings for different stages


    try {
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // --- 1. Load Images ---
        std::cout << "--- Loading Images ---" << std::endl;
        if (!loadImages(left_image_path, right_image_path,
                         im0_data_rgba, im1_data_rgba, width, height, width1_tmp, height1_tmp)) {
            return 1;
        }

        // --- 2. Initialize OpenCL ---
        std::cout << "\n--- Initializing OpenCL ---" << std::endl;
        if (!initOpenCL(platform, device, context, queue)) {
            return 1;
        }

        // --- 3. Load and Build Kernels ---
        std::cout << "\n--- Loading and Building Kernels ---" << std::endl;
        std::string kernelSource = loadKernelSource(kernel_file_path);
        program = createAndBuildProgram(context, device, kernelSource);
        if (!program) {
             throw std::runtime_error("Failed to create or build program.");
        }

        // *** Hardcode the Box Filter kernel name ***
        const char* resizeKernelName = "resizeGrayscaleBoxFilter";
        std::cout << "Using resize/grayscale kernel: " << resizeKernelName << std::endl;
        std::cout << "Using Box Filter Radius: " << boxFilterRadius << " (" << (2*boxFilterRadius+1) << "x" << (2*boxFilterRadius+1) << ")" << std::endl;


        // Create all kernel objects from the program
        if (!createKernels(program,
                           resizeGrayscaleBoxKernel, resizeKernelName, // Use the hardcoded name
                           precomputeStatsKernel, "precompute_window_stats",
                           computeDisparityKernel, "compute_disparity_zncc"))
        {
            throw std::runtime_error("Failed to create one or more kernels.");
        }


        // --- 4. Calculate Output Dimensions & Create Image Objects ---
        std::cout << "\n--- Creating OpenCL Images ---" << std::endl;
        if (resizeFactor <= 0.0f) {
            throw std::runtime_error("Resize factor must be positive.");
        }
        // Calculate output dimensions based on resize factor
        unsigned out_width = static_cast<unsigned>(std::round(static_cast<float>(width) / resizeFactor));
        unsigned out_height = static_cast<unsigned>(std::round(static_cast<float>(height) / resizeFactor));
        // Ensure minimum size of 1x1
        out_width = std::max(1u, out_width);
        out_height = std::max(1u, out_height);
        std::cout << "Input dimensions: " << width << "x" << height << std::endl;
        std::cout << "Calculated output dimensions: " << out_width << "x" << out_height
                  << " (Resize Factor: " << resizeFactor << ")" << std::endl;


        if (!createImageObjects(context, width, height, out_width, out_height, // Pass calculated dims
                                im0_data_rgba, im1_data_rgba,
                                im0_image_rgba, im1_image_rgba,
                                im0_formated_uchar, im1_formated_uchar)) {
            throw std::runtime_error("Failed to create OpenCL image objects.");
        }
        cl_images.push_back(im0_image_rgba);
        cl_images.push_back(im1_image_rgba);
        cl_images.push_back(im0_formated_uchar);
        cl_images.push_back(im1_formated_uchar);

        // Verify input image objects (optional but good)
        // verifyImageObjects(queue, im0_image_rgba, width, height);

        // --- 5. Execute Resize & Grayscale Kernel (Box Filter) ---
        std::cout << "\n--- Executing Resize/Grayscale Kernel (Box Filter) ---" << std::endl;
        auto stage_start_time = std::chrono::high_resolution_clock::now();

        // *** Global size is the size of the OUTPUT image ***
        size_t globalSizeResize[2] = {out_width, out_height};

        cl_int err_arg = 0; // Accumulate errors for clSetKernelArg

        // --- Run for im0 ---
        // Args: input_image, output_image, resizeFactor, windowRadius
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im0_image_rgba);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im0_formated_uchar);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 2, sizeof(cl_float), &resizeFactor);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 3, sizeof(cl_int), &boxFilterRadius); // Always set radius
        checkError(err_arg, "Setting resize (box filter) kernel arguments for im0");


        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL,
                                     globalSizeResize, NULL, // Use default local size
                                     0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im0");

        // --- Run for im1 ---
        // Reuse args 2 and 3 (factor, radius), just change images (args 0, 1)
        err_arg = 0;
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im1_image_rgba); // Change input
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im1_formated_uchar); // Change output
        checkError(err_arg, "Re-setting resize (box filter) image kernel arguments for im1");


        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL,
                                     globalSizeResize, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im1");

        // Wait for resize/grayscale to finish before proceeding
        err = clFinish(queue);
        checkError(err, "Waiting for resize kernels to finish");

        auto stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Resize/Grayscale kernels finished in " << timings_ms.back() << " ms." << std::endl;


        // --- 6. Read Back Formatted Images and Convert to Float Buffers ---
        std::cout << "\n--- Reading Formatted Images & Creating Float Buffers ---" << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        size_t formatted_image_size_pixels = static_cast<size_t>(out_width) * out_height;
        size_t formatted_image_size_bytes = formatted_image_size_pixels * sizeof(unsigned char); // 1 byte per pixel (CL_R)

        std::vector<unsigned char> im0_formatted_host(formatted_image_size_pixels);
        std::vector<unsigned char> im1_formatted_host(formatted_image_size_pixels);

        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {out_width, out_height, 1}; // Region matches output image dims

        // Read im0_formated_uchar
        err = clEnqueueReadImage(queue, im0_formated_uchar, CL_TRUE, // Blocking read
                                 origin, region, 0, 0, // row_pitch and slice_pitch are 0 for 2D
                                 im0_formatted_host.data(), 0, NULL, NULL);
        checkError(err, "Reading formatted image 0 back to host");

        // Read im1_formated_uchar
        err = clEnqueueReadImage(queue, im1_formated_uchar, CL_TRUE, // Blocking read
                                 origin, region, 0, 0,
                                 im1_formatted_host.data(), 0, NULL, NULL);
        checkError(err, "Reading formatted image 1 back to host");

        // *** Conditionally save intermediate formatted images ***
        if (saveFormattedDebugImages) {
            std::cout << "Saving debug formatted images..." << std::endl;
            saveGrayscalePNG(output_dir + "im0_formated_debug.png", im0_formatted_host, out_width, out_height);
            saveGrayscalePNG(output_dir + "im1_formated_debug.png", im1_formatted_host, out_width, out_height);
        } else {
             std::cout << "Skipping saving of debug formatted images." << std::endl;
        }


        // Convert uchar [0,255] to float [0.0, 255.0] on host for ZNCC kernel
        std::vector<float> left_float_host(formatted_image_size_pixels);
        std::vector<float> right_float_host(formatted_image_size_pixels);
        for (size_t i = 0; i < formatted_image_size_pixels; ++i) {
            left_float_host[i] = static_cast<float>(im0_formatted_host[i]);
            right_float_host[i] = static_cast<float>(im1_formatted_host[i]);
        }

        // Create float buffers on device and copy data
        size_t float_buffer_size_bytes = formatted_image_size_pixels * sizeof(float);
        left_float_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        float_buffer_size_bytes, left_float_host.data(), &err);
        checkError(err, "Creating left float buffer");
        cl_buffers.push_back(left_float_buf);

        right_float_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         float_buffer_size_bytes, right_float_host.data(), &err);
        checkError(err, "Creating right float buffer");
        cl_buffers.push_back(right_float_buf);
        std::cout << "Float buffers created and populated." << std::endl;

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Readback and Float Buffer creation finished in " << timings_ms.back() << " ms." << std::endl;

        // --- 7. Execute Precompute Stats Kernel ---
        std::cout << "\n--- Executing Precompute Stats Kernel ---" << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        // Create buffers for means and stddevs (size matches float buffers)
        left_means_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, float_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating left means buffer");
        cl_buffers.push_back(left_means_buf);

        left_stddevs_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, float_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating left stddevs buffer");
        cl_buffers.push_back(left_stddevs_buf);

        right_means_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, float_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating right means buffer");
        cl_buffers.push_back(right_means_buf);

        right_stddevs_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, float_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating right stddevs buffer");
        cl_buffers.push_back(right_stddevs_buf);

        // Global size for ZNCC steps (matches the formatted/float image dimensions)
        size_t globalSizeZNCC[2] = {out_width, out_height};

        // --- Run precompute for Left image ---
        // Args: input_float, means_out, stddevs_out, width, height, win_hsize
        err = clSetKernelArg(precomputeStatsKernel, 0, sizeof(cl_mem), &left_float_buf);
        err |= clSetKernelArg(precomputeStatsKernel, 1, sizeof(cl_mem), &left_means_buf);
        err |= clSetKernelArg(precomputeStatsKernel, 2, sizeof(cl_mem), &left_stddevs_buf);
        err |= clSetKernelArg(precomputeStatsKernel, 3, sizeof(int), &out_width); // Use output width
        err |= clSetKernelArg(precomputeStatsKernel, 4, sizeof(int), &out_height); // Use output height
        err |= clSetKernelArg(precomputeStatsKernel, 5, sizeof(int), &win_size);
        checkError(err, "Setting precompute kernel args for left");
        err = clEnqueueNDRangeKernel(queue, precomputeStatsKernel, 2, NULL,
                                     globalSizeZNCC, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing precompute kernel for left");

        // --- Run precompute for Right image ---
        err = clSetKernelArg(precomputeStatsKernel, 0, sizeof(cl_mem), &right_float_buf); // Change input
        err |= clSetKernelArg(precomputeStatsKernel, 1, sizeof(cl_mem), &right_means_buf); // Change mean output
        err |= clSetKernelArg(precomputeStatsKernel, 2, sizeof(cl_mem), &right_stddevs_buf); // Change stddev output
        // Args 3, 4, 5 (dims, win_size) are the same
        checkError(err, "Setting precompute kernel args for right");
        err = clEnqueueNDRangeKernel(queue, precomputeStatsKernel, 2, NULL,
                                     globalSizeZNCC, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing precompute kernel for right");

        // Wait for precomputation to finish
        err = clFinish(queue);
        checkError(err, "Waiting for precompute kernels to finish");

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Precompute Stats kernels finished in " << timings_ms.back() << " ms." << std::endl;


        // --- 8. Execute Compute Disparity (ZNCC) Kernel ---
        std::cout << "\n--- Executing Compute Disparity (ZNCC) Kernel ---" << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        // Create disparity output buffer (uchar, size matches formatted image)
        size_t disparity_buffer_size_bytes = formatted_image_size_pixels * sizeof(unsigned char);
        disparity_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating disparity buffer");
        cl_buffers.push_back(disparity_buf);

        // Set arguments for ZNCC kernel
        // Args: left_float, right_float, left_means, left_stddevs, right_means, right_stddevs, disparity_out, width, height, win_hsize, max_disp
        err = clSetKernelArg(computeDisparityKernel, 0, sizeof(cl_mem), &left_float_buf);
        err |= clSetKernelArg(computeDisparityKernel, 1, sizeof(cl_mem), &right_float_buf);
        err |= clSetKernelArg(computeDisparityKernel, 2, sizeof(cl_mem), &left_means_buf);
        err |= clSetKernelArg(computeDisparityKernel, 3, sizeof(cl_mem), &left_stddevs_buf);
        err |= clSetKernelArg(computeDisparityKernel, 4, sizeof(cl_mem), &right_means_buf);
        err |= clSetKernelArg(computeDisparityKernel, 5, sizeof(cl_mem), &right_stddevs_buf);
        err |= clSetKernelArg(computeDisparityKernel, 6, sizeof(cl_mem), &disparity_buf);
        err |= clSetKernelArg(computeDisparityKernel, 7, sizeof(int), &out_width); // Use output width
        err |= clSetKernelArg(computeDisparityKernel, 8, sizeof(int), &out_height); // Use output height
        err |= clSetKernelArg(computeDisparityKernel, 9, sizeof(int), &win_size);
        err |= clSetKernelArg(computeDisparityKernel, 10, sizeof(int), &max_disp);
        checkError(err, "Setting compute disparity kernel args");

        // Enqueue ZNCC kernel
        err = clEnqueueNDRangeKernel(queue, computeDisparityKernel, 2, NULL,
                                     globalSizeZNCC, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing compute disparity kernel");

        // Wait for ZNCC kernel to finish
        err = clFinish(queue);
        checkError(err, "Waiting for compute disparity kernel to finish");

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Compute Disparity (ZNCC) kernel finished in " << timings_ms.back() << " ms." << std::endl;


        // --- 9. Read Final Disparity Map ---
        std::cout << "\n--- Reading Final Disparity Map ---" << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        std::vector<unsigned char> disparity_map_host(formatted_image_size_pixels);
        err = clEnqueueReadBuffer(queue, disparity_buf, CL_TRUE, // Blocking read
                                  0, disparity_buffer_size_bytes,
                                  disparity_map_host.data(), 0, NULL, NULL);
        checkError(err, "Reading disparity map buffer back to host");

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Disparity Readback finished in " << timings_ms.back() << " ms." << std::endl;

        // --- 10. Save Disparity Map ---
        std::cout << "\n--- Saving Output ---" << std::endl;
        saveGrayscalePNG(output_dir + "disparity_map_ocl.png", disparity_map_host, out_width, out_height);

        // --- 11. Cleanup ---
        std::cout << "\n--- Cleaning Up ---" << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, precomputeStatsKernel, computeDisparityKernel, // Use specific kernel handle
                cl_buffers, cl_images);
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());


        // --- Calculate Total Time ---
         auto total_end_time = std::chrono::high_resolution_clock::now();
         double total_duration_ms = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
         // Sum kernel execution + readback times (indices 0, 2, 3, 4 in timings_ms)
         double gpu_plus_readback_ms = timings_ms[0] + timings_ms[2] + timings_ms[3] + timings_ms[4];

         std::cout << "\n-------------------- Summary --------------------" << std::endl;
         std::cout << "Resize/Grayscale Kernel : " << timings_ms[0] << " ms" << std::endl;
         std::cout << "Readback & Float Buf Prep: " << timings_ms[1] << " ms" << std::endl;
         std::cout << "Precompute Stats Kernel : " << timings_ms[2] << " ms" << std::endl;
         std::cout << "Compute Disparity Kernel: " << timings_ms[3] << " ms" << std::endl;
         std::cout << "Disparity Readback      : " << timings_ms[4] << " ms" << std::endl;
         std::cout << "Cleanup                 : " << timings_ms[5] << " ms" << std::endl;
         std::cout << "-------------------------------------------------" << std::endl;
         std::cout << "Total GPU execution + Readback Time: " << gpu_plus_readback_ms << " ms" << std::endl;
         std::cout << "Total Application Time (incl. I/O, setup): " << total_duration_ms << " ms" << std::endl;
         std::cout << "-------------------------------------------------" << std::endl;


        return 0;
    }
    catch (const OpenCLException& e) {
        std::cerr << "\n\n!--- OpenCL Error ---!\nMessage: " << e.what() << "\nError Code: " << e.getErrorCode() << std::endl;
        // Ensure cleanup is attempted even on error
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, precomputeStatsKernel, computeDisparityKernel, // Use specific kernel handle
                cl_buffers, cl_images);
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n\n!--- Standard Error ---!\nMessage: " << e.what() << std::endl;
        // Ensure cleanup is attempted even on error
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, precomputeStatsKernel, computeDisparityKernel, // Use specific kernel handle
                cl_buffers, cl_images);
        return 1;
    }
}