#include <CL/cl.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem> // Need C++17

// Custom OpenCLException
class OpenCLException : public std::runtime_error {
public:
    OpenCLException(const std::string& message, cl_int error)
        : std::runtime_error(message + ": Error code " + std::to_string(error)), error_code(error) {}

    cl_int getErrorCode() const { return error_code; }

private:
    cl_int error_code;
};

// Function to check OpenCL errors
void checkError(cl_int error, const std::string& message) {
    if (error != CL_SUCCESS) {
        throw OpenCLException(message, error);
    }
}

// Function to get and print device information
void printDeviceInfo(cl_device_id device) {
    cl_int error;
    cl_device_local_mem_type localMemType;
    cl_ulong localMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxClockFreq;
    cl_ulong maxConstantBufferSize;
    size_t maxWorkGroupSize;
    size_t maxWorkItemSizes[3];
    cl_bool imageSupport = CL_FALSE;

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

    error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport), &imageSupport, NULL);
    checkError(error, "Failed to get CL_DEVICE_IMAGE_SUPPORT");


    std::cout << "==== GPU Device Information ====" << std::endl;
    std::cout << "Local Mem Type: " << (localMemType == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL") << std::endl;
    std::cout << "Local Mem Size: " << localMemSize << " bytes" << std::endl;
    std::cout << "Max Compute Units: " << maxComputeUnits << std::endl;
    std::cout << "Max Clock Freq: " << maxClockFreq << " MHz" << std::endl;
    std::cout << "Max Constant Buffer Size: " << maxConstantBufferSize << " bytes" << std::endl;
    std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
    std::cout << "Max Work Item Sizes: [" << maxWorkItemSizes[0] << ", "
              << maxWorkItemSizes[1] << ", " << maxWorkItemSizes[2] << "]" << std::endl;
    std::cout << "Image Support: " << (imageSupport ? "Yes" : "No") << std::endl;
    std::cout << "================================" << std::endl;

    if (!imageSupport) {
        std::cerr << "Warning: Selected OpenCL device does not support images!" << std::endl;
    }
}

// Function to load PNG images
bool loadImages(const std::string& file1, const std::string& file2,
                std::vector<unsigned char>& im0_data, std::vector<unsigned char>& im1_data,
                unsigned& width0, unsigned& height0,
                unsigned& width1, unsigned& height1) {
    unsigned error;

    error = lodepng::decode(im0_data, width0, height0, file1);
    if (error) {
        std::cerr << "Error loading " << file1 << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    error = lodepng::decode(im1_data, width1, height1, file2);
    if (error) {
        std::cerr << "Error loading " << file2 << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    if (width0 != width1 || height0 != height1) {
        std::cerr << "Error: Image dimensions do not match" << std::endl;
        return false;
    }

    std::cout << "Loaded " << file1 << ": " << width0 << " x " << height0 << " (RGBA)" << std::endl;
    std::cout << "Loaded " << file2 << ": " << width1 << " x " << height1 << " (RGBA)" << std::endl;

    return true;
}

// Function to save grayscale PNG
void saveGrayscalePNG(const std::string& filename,
                      const std::vector<unsigned char>& imageData,
                      unsigned width, unsigned height) {
    // Create directory if it doesn't exist
    std::filesystem::path filePath(filename);
    std::filesystem::path dirPath = filePath.parent_path();
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Creating output directory: " << dirPath << std::endl;
        std::filesystem::create_directories(dirPath);
    }

    unsigned error = lodepng::encode(filename, imageData, width, height, LCT_GREY, 8);
    if (error) {
        throw std::runtime_error("PNG encoder error saving " + filename + ": " +
                                 std::string(lodepng_error_text(error)));
    }
    std::cout << "Saved grayscale image to " << filename << " (" << width << "x" << height << ")" << std::endl;
}

// Function to initialize OpenCL
bool initOpenCL(cl_platform_id& platform, cl_device_id& device,
                cl_context& context, cl_command_queue& queue) {
    try {
        cl_int error;
        cl_uint numPlatforms;

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
            if(error != CL_SUCCESS) continue;
            std::cout << i + 1 << ": " << platformName << "\n";
        }

        // Find a device
        platform = nullptr;
        device = nullptr;
        for (cl_uint i = 0; i < numPlatforms; ++i) {
            cl_uint numDevices = 0;
            error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
            if (error != CL_SUCCESS || numDevices == 0) {
                 error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &numDevices);
                 if (error != CL_SUCCESS || numDevices == 0) continue;
            }

            std::vector<cl_device_id> devices(numDevices);
            cl_device_type selected_type = CL_DEVICE_TYPE_GPU;
            error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
             if (error != CL_SUCCESS) {
                 selected_type = CL_DEVICE_TYPE_DEFAULT;
                 error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, numDevices, devices.data(), nullptr);
                 if (error != CL_SUCCESS) continue;
             }

            platform = platforms[i];
            device = devices[0]; // Select the first one
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

        printDeviceInfo(device);

        // Create context
        cl_context_properties contextProps[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0 // Terminate list
        };
        context = clCreateContext(contextProps, 1, &device, nullptr, nullptr, &error);
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

// Function to create input/output image objects
bool createImageObjects(cl_context context,
                        unsigned in_width, unsigned in_height,
                        unsigned out_width, unsigned out_height,
                        const std::vector<unsigned char>& im0_data_rgba,
                        const std::vector<unsigned char>& im1_data_rgba,
                        cl_mem& im0_rgba_in, cl_mem& im1_rgba_in,
                        cl_mem& im0_gray_out, cl_mem& im1_gray_out)
{
    cl_int error = CL_SUCCESS;
    im0_rgba_in = im1_rgba_in = im0_gray_out = im1_gray_out = nullptr;

    try {
        // Image formats
        const cl_image_format input_format = {CL_RGBA, CL_UNORM_INT8}; // Input RGBA
        const cl_image_format output_format = {CL_R, CL_UNSIGNED_INT8}; // Output Gray uchar

        // Input image descriptor
        cl_image_desc input_desc = {};
        input_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        input_desc.image_width = in_width;
        input_desc.image_height = in_height;

        // Create input images
        im0_rgba_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    &input_format, &input_desc,
                                    const_cast<unsigned char*>(im0_data_rgba.data()), &error);
        checkError(error, "Failed to create input im0_rgba_in");

        im1_rgba_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    &input_format, &input_desc,
                                    const_cast<unsigned char*>(im1_data_rgba.data()), &error);
        checkError(error, "Failed to create input im1_rgba_in");

        // Output image descriptor
        cl_image_desc output_desc = {};
        output_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        output_desc.image_width = out_width;
        output_desc.image_height = out_height;

        // Create output grayscale images
        // Need Read/Write because resize writes, zncc reads
        im0_gray_out = clCreateImage(context, CL_MEM_READ_WRITE,
                                     &output_format, &output_desc,
                                     nullptr, &error);
        checkError(error, "Failed to create output im0_gray_out");

        im1_gray_out = clCreateImage(context, CL_MEM_READ_WRITE,
                                     &output_format, &output_desc,
                                     nullptr, &error);
        checkError(error, "Failed to create output im1_gray_out");

        std::cout << "Created OpenCL image objects. Input: " << in_width << "x" << in_height
                  << ", Formatted Grayscale Output: " << out_width << "x" << out_height << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        // Cleanup on failure
        if (im0_rgba_in) clReleaseMemObject(im0_rgba_in);
        if (im1_rgba_in) clReleaseMemObject(im1_rgba_in);
        if (im0_gray_out) clReleaseMemObject(im0_gray_out);
        if (im1_gray_out) clReleaseMemObject(im1_gray_out);

        std::cerr << "Image object creation failed: " << e.what() << std::endl;
        return false;
    }
}

// Function to verify image objects by reading back a pixel
void verifyImageObjects(cl_command_queue queue, cl_mem image_to_verify, unsigned width, unsigned height, const std::string& name) {
    if (!image_to_verify) {
        std::cerr << "Verification skipped: "<< name << " is null." << std::endl;
        return;
    }
    if (width == 0 || height == 0) {
         std::cerr << "Verification skipped (" << name <<"): Invalid dimensions (0)." << std::endl;
         return;
    }
    try {
        cl_int error;
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {1, 1, 1};  // Read one pixel

        cl_image_format format;
        error = clGetImageInfo(image_to_verify, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
        checkError(error, "Failed to get image format for verification");

        // Figure out how to read based on format
        std::vector<unsigned char> pixel_data_char;
        std::vector<float> pixel_data_float;
        void* read_ptr = nullptr;
        std::string format_str = "";

        if (format.image_channel_order == CL_R && format.image_channel_data_type == CL_UNSIGNED_INT8) {
            pixel_data_char.resize(1);
            read_ptr = pixel_data_char.data();
            format_str = "CL_R, UNSIGNED_INT8";
        } else if (format.image_channel_order == CL_RGBA && format.image_channel_data_type == CL_UNORM_INT8) {
            pixel_data_float.resize(4);
            read_ptr = pixel_data_float.data();
            format_str = "CL_RGBA, UNORM_INT8 (read as float)";
        } else {
             std::cerr << "Verification skipped (" << name << "): Unsupported format for simple verification." << std::endl;
             return;
        }


        // Make sure previous commands finished
        error = clFinish(queue);
        checkError(error, "Failed to finish queue before verification read");

        error = clEnqueueReadImage(queue, image_to_verify, CL_TRUE, origin, region, 0, 0,
                                  read_ptr, 0, nullptr, nullptr);
        checkError(error, "Failed to read from " + name + " for verification");

        // Print pixel value
        std::cout << "Verification: First pixel from " << name << " (GPU, format: " << format_str << "): ";
        if (!pixel_data_char.empty()) { // Grayscale uchar
            std::cout << "Value=" << static_cast<int>(pixel_data_char[0]) << std::endl;
        } else if (!pixel_data_float.empty()) { // RGBA UNORM read as float
             std::cout << "R=" << static_cast<int>(pixel_data_float[0] * 255.0f)
                       << ", G=" << static_cast<int>(pixel_data_float[1] * 255.0f)
                       << ", B=" << static_cast<int>(pixel_data_float[2] * 255.0f)
                       << ", A=" << static_cast<int>(pixel_data_float[3] * 255.0f) << std::endl;
        }

    }
    catch (const OpenCLException& e) {
        std::cerr << "Verification failed (" << name << "): Could not read from image: " << e.what() << std::endl;
    }
     catch (const std::exception& e) {
         std::cerr << "Verification failed (" << name << ") (std::exception): " << e.what() << std::endl;
     }
}

// Cleanup function
void cleanup(cl_context context, cl_command_queue queue, cl_program program,
             cl_kernel kernel1, cl_kernel kernel2,
             cl_sampler sampler,
             std::vector<cl_mem> buffers,
             std::vector<cl_mem> images)
{
    std::cout << "Cleaning up OpenCL resources..." << std::endl;
    cl_int err;

    // Flush and finish before releasing
    if (queue) {
        err = clFlush(queue);
        if(err != CL_SUCCESS) std::cerr << "Warning: clFlush failed during cleanup: " << err << std::endl;
        err = clFinish(queue);
        if(err != CL_SUCCESS) std::cerr << "Warning: clFinish failed during cleanup: " << err << std::endl;
    }

    // Release kernels
    if (kernel1) clReleaseKernel(kernel1);
    if (kernel2) clReleaseKernel(kernel2);

    // Release sampler
    if (sampler) clReleaseSampler(sampler);

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

// Function to load OpenCL kernels from file
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// Function to build program from kernel source
cl_program createAndBuildProgram(cl_context context, cl_device_id device, const std::string& kernelSource) {
    cl_int error;
    const char* source = kernelSource.c_str();
    size_t sourceLength = kernelSource.length();

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceLength, &error);
    checkError(error, "Failed to create program");

    // Build program
    std::cout << "Building OpenCL program..." << std::endl;
    // Add some common build options
    error = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-single-precision-constant -cl-mad-enable", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // Get build log if build failed
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize + 1);
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

// Function to create a single kernel
bool createKernel(cl_program program, cl_kernel& kernel, const char* kernel_name)
{
    cl_int error;
    kernel = nullptr;
    try {
        kernel = clCreateKernel(program, kernel_name, &error);
        checkError(error, std::string("Failed to create kernel: ") + kernel_name);
        std::cout << "Kernel created: '" << kernel_name << "'" << std::endl;
        return true;
    } catch (const OpenCLException& e) {
        std::cerr << "Kernel creation failed for '" << kernel_name << "': " << e.what() << std::endl;
        // No need to release kernel here, it should be null or invalid
        return false;
    }
}


int main(int argc, char* argv[]) {

    // Config
    const std::string left_image_path = "../ressources/im0.png";
    const std::string right_image_path = "../ressources/im1.png";
    const std::string kernel_file_path = "kernels.cl";
    const std::string output_dir = "../output/";

    // Resize Config
    const float resizeFactor = 4.0f; // >1.0 = Downscale
    const int boxFilterRadius = 2;   // Radius 2 -> 5x5 window

    // ZNCC Config
    const int ZNCC_WIN_SIZE = 9;    // Must be odd
    const int MAX_DISP = 64;        // 0..MAX_DISP-1. Must be <= 256
    static_assert(ZNCC_WIN_SIZE % 2 != 0, "ZNCC_WIN_SIZE must be odd.");
    static_assert(MAX_DISP > 0 && MAX_DISP <= 256, "MAX_DISP must be between 1 and 256.");

    // Debug flags
    const bool saveFormattedDebugImages = true;
    const bool verifyInputImages = true;
    const bool verifyFormattedImages = true;

    // Image data
    std::vector<unsigned char> im0_data_rgba, im1_data_rgba;
    unsigned width = 0, height = 0;
    unsigned width1_tmp, height1_tmp;

    // OpenCL stuff
    cl_int err;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_sampler sampler = nullptr;

    // Kernels
    cl_kernel resizeGrayscaleBoxKernel = nullptr;
    cl_kernel znccKernel = nullptr;

    // OpenCL Images
    cl_mem im0_image_rgba = nullptr;
    cl_mem im1_image_rgba = nullptr;
    cl_mem im0_formated_uchar = nullptr;
    cl_mem im1_formated_uchar = nullptr;

    // OpenCL Buffers
    cl_mem disparity_buf = nullptr;

    // Track resources for cleanup
    std::vector<cl_mem> cl_buffers;
    std::vector<cl_mem> cl_images;

    // Timers
    std::vector<double> timings_ms;


    try {
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // 1. Load Images
        std::cout << "Loading Images..." << std::endl;
        if (!loadImages(left_image_path, right_image_path,
                         im0_data_rgba, im1_data_rgba, width, height, width1_tmp, height1_tmp)) {
            return 1;
        }

        // 2. Initialize OpenCL
        std::cout << "\nInitializing OpenCL..." << std::endl;
        if (!initOpenCL(platform, device, context, queue)) {
            return 1;
        }

        // 3. Load and Build Kernels
        std::cout << "\nLoading and Building Kernels..." << std::endl;
        std::string kernelSource = loadKernelSource(kernel_file_path);
        program = createAndBuildProgram(context, device, kernelSource);
        if (!program) {
             throw std::runtime_error("Failed to create or build program.");
        }

        const char* resizeKernelName = "resizeGrayscaleBoxFilter";
        const char* znccKernelName = "zncc_stereo_match";

        std::cout << "Creating kernels..." << std::endl;
        if (!createKernel(program, resizeGrayscaleBoxKernel, resizeKernelName)) {
             throw std::runtime_error("Failed to create resize kernel.");
        }
        if (!createKernel(program, znccKernel, znccKernelName)) {
             throw std::runtime_error("Failed to create ZNCC kernel.");
        }
        std::cout << "Using Resize/Grayscale Kernel: " << resizeKernelName << std::endl;
        std::cout << "Using ZNCC Kernel: " << znccKernelName << std::endl;
        std::cout << "Box Filter Radius: " << boxFilterRadius << " (" << (2 * boxFilterRadius + 1) << "x" << (2 * boxFilterRadius + 1) << ")" << std::endl;
        std::cout << "ZNCC Window Size: " << ZNCC_WIN_SIZE << "x" << ZNCC_WIN_SIZE << std::endl;
        std::cout << "Max Disparity: " << MAX_DISP << std::endl;

        // 4. Calculate Output Dimensions & Create Resources
        std::cout << "\nCreating OpenCL Resources..." << std::endl;
        if (resizeFactor <= 0.0f) {
            throw std::runtime_error("Resize factor must be positive.");
        }
        // Calculate output size
        unsigned out_width = static_cast<unsigned>(std::round(static_cast<float>(width) / resizeFactor));
        unsigned out_height = static_cast<unsigned>(std::round(static_cast<float>(height) / resizeFactor));
        out_width = std::max(1u, out_width); // Min size 1x1
        out_height = std::max(1u, out_height);
        std::cout << "Input dimensions: " << width << "x" << height << std::endl;
        std::cout << "Calculated output dimensions: " << out_width << "x" << out_height
                  << " (Resize Factor: " << resizeFactor << ")" << std::endl;

        // Create Image Objects
        if (!createImageObjects(context, width, height, out_width, out_height,
                                im0_data_rgba, im1_data_rgba,
                                im0_image_rgba, im1_image_rgba,
                                im0_formated_uchar, im1_formated_uchar))
        {
            throw std::runtime_error("Failed to create OpenCL image objects.");
        }
        cl_images.push_back(im0_image_rgba);
        cl_images.push_back(im1_image_rgba);
        cl_images.push_back(im0_formated_uchar);
        cl_images.push_back(im1_formated_uchar);

        if(verifyInputImages) {
            verifyImageObjects(queue, im0_image_rgba, width, height, "im0_image_rgba");
        }

        // Create Sampler for ZNCC kernel
        cl_sampler_properties samplerProps[] = {
            CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE, // Pixel coords
            CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP_TO_EDGE,
            CL_SAMPLER_FILTER_MODE, CL_FILTER_NEAREST,
            0 // End list
        };
        sampler = clCreateSamplerWithProperties(context, samplerProps, &err);
        checkError(err, "Failed to create sampler");
        std::cout << "Created sampler." << std::endl;

        // Create Disparity Output Buffer
        size_t disparity_image_size_pixels = static_cast<size_t>(out_width) * out_height;
        size_t disparity_buffer_size_bytes = disparity_image_size_pixels * sizeof(unsigned char);
        disparity_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating disparity buffer");
        cl_buffers.push_back(disparity_buf);
        std::cout << "Created disparity output buffer (" << out_width << "x" << out_height << ", uchar)." << std::endl;


        // 5. Execute Resize & Grayscale Kernel
        std::cout << "\nExecuting Resize/Grayscale Kernel..." << std::endl;
        auto stage_start_time = std::chrono::high_resolution_clock::now();

        size_t globalSizeResize[2] = {out_width, out_height};
        cl_int err_arg = 0;

        // Process im0
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im0_image_rgba);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im0_formated_uchar);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 2, sizeof(cl_float), &resizeFactor);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 3, sizeof(cl_int), &boxFilterRadius);
        checkError(err_arg, "Setting resize kernel arguments for im0");

        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL,
                                     globalSizeResize, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im0");

        // Process im1
        err_arg = 0;
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im1_image_rgba);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im1_formated_uchar);
        checkError(err_arg, "Re-setting resize kernel image arguments for im1");

        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL,
                                     globalSizeResize, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im1");

        // Wait for resize to finish
        err = clFinish(queue);
        checkError(err, "Waiting for resize kernels to finish");

        auto stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Resize/Grayscale kernels finished in " << timings_ms.back() << " ms." << std::endl;

        // Verify and Save Intermediate Grayscale Images (Optional)
        if (verifyFormattedImages || saveFormattedDebugImages) {
             std::cout << "\nProcessing Intermediate Grayscale Images..." << std::endl;
             if (verifyFormattedImages) {
                verifyImageObjects(queue, im0_formated_uchar, out_width, out_height, "im0_formated_uchar");
             }
             if (saveFormattedDebugImages) {
                 // Read back formatted images
                 std::vector<unsigned char> im0_formatted_host(disparity_image_size_pixels);
                 std::vector<unsigned char> im1_formatted_host(disparity_image_size_pixels);
                 size_t origin[3] = {0, 0, 0};
                 size_t region[3] = {out_width, out_height, 1};

                 err = clEnqueueReadImage(queue, im0_formated_uchar, CL_TRUE, origin, region, 0, 0,
                                          im0_formatted_host.data(), 0, NULL, NULL);
                 checkError(err, "Reading formatted image 0 for debug saving");
                 err = clEnqueueReadImage(queue, im1_formated_uchar, CL_TRUE, origin, region, 0, 0,
                                          im1_formatted_host.data(), 0, NULL, NULL);
                 checkError(err, "Reading formatted image 1 for debug saving");

                 saveGrayscalePNG(output_dir + "im0_formated_debug.png", im0_formatted_host, out_width, out_height);
                 saveGrayscalePNG(output_dir + "im1_formated_debug.png", im1_formatted_host, out_width, out_height);
             }
        }


        // 6. Execute Compute Disparity (ZNCC) Kernel
        std::cout << "\nExecuting Compute Disparity (ZNCC) Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        size_t globalSizeZNCC[2] = {out_width, out_height};

        // Set arguments for ZNCC kernel
        // Args: left_image, right_image, disparity_map, width, height, WIN_SIZE, MAX_DISP, sampler
        err_arg = 0;
        err_arg |= clSetKernelArg(znccKernel, 0, sizeof(cl_mem), &im0_formated_uchar);
        err_arg |= clSetKernelArg(znccKernel, 1, sizeof(cl_mem), &im1_formated_uchar);
        err_arg |= clSetKernelArg(znccKernel, 2, sizeof(cl_mem), &disparity_buf);
        err_arg |= clSetKernelArg(znccKernel, 3, sizeof(int), &out_width);
        err_arg |= clSetKernelArg(znccKernel, 4, sizeof(int), &out_height);
        err_arg |= clSetKernelArg(znccKernel, 5, sizeof(int), &ZNCC_WIN_SIZE);
        err_arg |= clSetKernelArg(znccKernel, 6, sizeof(int), &MAX_DISP);
        err_arg |= clSetKernelArg(znccKernel, 7, sizeof(cl_sampler), &sampler);
        checkError(err_arg, "Setting ZNCC kernel arguments");

        // Enqueue ZNCC kernel
        err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL,
                                     globalSizeZNCC, NULL,
                                     0, NULL, NULL);
        checkError(err, "Enqueueing ZNCC kernel");

        // Wait for ZNCC kernel to finish
        err = clFinish(queue);
        checkError(err, "Waiting for ZNCC kernel to finish");

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Compute Disparity (ZNCC) kernel finished in " << timings_ms.back() << " ms." << std::endl;


        // 7. Read Final Disparity Map
        std::cout << "\nReading Final Disparity Map..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();

        std::vector<unsigned char> disparity_map_host(disparity_image_size_pixels);
        err = clEnqueueReadBuffer(queue, disparity_buf, CL_TRUE,
                                  0, disparity_buffer_size_bytes,
                                  disparity_map_host.data(), 0, NULL, NULL);
        checkError(err, "Reading disparity map buffer back to host");

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << "Disparity Readback finished in " << timings_ms.back() << " ms." << std::endl;

        // 8. Save Disparity Map
        std::cout << "\nSaving Output..." << std::endl;
        saveGrayscalePNG(output_dir + "disparity_map_ocl.png", disparity_map_host, out_width, out_height);

        // 9. Cleanup
        std::cout << "\nCleaning Up..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel,
                sampler,
                cl_buffers, cl_images);
        // Nullify handles
        resizeGrayscaleBoxKernel = nullptr; znccKernel = nullptr; sampler = nullptr;
        context = nullptr; queue = nullptr; program = nullptr;
        cl_buffers.clear(); cl_images.clear();

        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());


        // Calculate Total Time & Summary
         auto total_end_time = std::chrono::high_resolution_clock::now();
         double total_duration_ms = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
         double gpu_plus_readback_ms = timings_ms[0] + timings_ms[1] + timings_ms[2]; // Resize + ZNCC + Readback

         std::cout << "\n-------------------- Summary --------------------" << std::endl;
         std::cout << "Resize/Grayscale      : " << timings_ms[0] << " ms" << std::endl;
         std::cout << "Compute Disparity     : " << timings_ms[1] << " ms" << std::endl;
         std::cout << "Disparity Readback    : " << timings_ms[2] << " ms" << std::endl;
         std::cout << "Cleanup               : " << timings_ms[3] << " ms" << std::endl;
         std::cout << "-------------------------------------------------" << std::endl;
         std::cout << "Total GPU + Readback Time: " << gpu_plus_readback_ms << " ms" << std::endl;
         std::cout << "Total Application Time : " << total_duration_ms << " ms" << std::endl;
         std::cout << "-------------------------------------------------" << std::endl;


        return 0;
    }
    catch (const OpenCLException& e) {
        std::cerr << "\n\n!--- OpenCL Error ---!\nMessage: " << e.what() << "\nError Code: " << e.getErrorCode() << std::endl;
        // Attempt cleanup
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel, sampler,
                cl_buffers, cl_images);
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n\n!--- Standard Error ---!\nMessage: " << e.what() << std::endl;
        // Attempt cleanup
         cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel, sampler,
                cl_buffers, cl_images);
        return 1;
    }
}