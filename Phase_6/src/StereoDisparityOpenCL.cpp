#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
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
// Removed filesystem dependency

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
    size_t lastSlash = filename.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        std::string dirPath = filename.substr(0, lastSlash);
        if (!dirPath.empty()) {
            // Use system command to create directory
            std::string mkdirCmd = "mkdir -p '" + dirPath + "'";
            system(mkdirCmd.c_str());
        }
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
        #ifdef __APPLE__
        // Sur macOS, utiliser la fonction standard clCreateCommandQueue pour une meilleure compatibilité
        queue = clCreateCommandQueue(context, device, 0, &error);
        #else
        queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
        #endif
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
    cl_kernel kernel1, cl_kernel kernel2, cl_kernel kernel3, cl_kernel kernel4, // MODIFIED
    cl_sampler sampler,
    std::vector<cl_mem>& buffers,
    std::vector<cl_mem>& images)

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
    if (kernel3) clReleaseKernel(kernel3);
    if (kernel4) clReleaseKernel(kernel4); 

    if (sampler) clReleaseSampler(sampler);
    if (program) clReleaseProgram(program);

    for (cl_mem buf : buffers) {
        if (buf) clReleaseMemObject(buf);
    }
    buffers.clear(); // Clear the vector after releasing

    for (cl_mem img : images) {
        if (img) clReleaseMemObject(img);
    }
    images.clear(); // Clear the vector after releasing

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
    const std::string kernel_file_path = "kernels.cl"; // Ensure this path is correct
    const std::string output_dir = "../output/";
    try { // filesystem operations can throw
        if (!output_dir.empty()) {
            // Create directory if it doesn't exist using system command
            std::string mkdirCmd = "mkdir -p '" + output_dir + "'";
            system(mkdirCmd.c_str());
        }
    } catch (const std::exception& fs_err) {
        std::cerr << "Filesystem error creating output directory: " << fs_err.what() << std::endl;
        // Decide if this is fatal or if you can proceed assuming current dir for output
        return 1; 
    }


    // Resize Config
    const float resizeFactor = 4.0f; // >1.0 = Downscale
    const int boxFilterRadius = 2;   // Radius 2 -> 5x5 window

    // ZNCC Config
    const int ZNCC_WIN_SIZE = 9;    // Must be odd
    const int MAX_DISP = 64;        // Disparities 0..MAX_DISP-1.
                                    // If ZNCC_INVALID_DISP_OUTPUT in kernel is 255, MAX_DISP must be < 255.
    
    static_assert(ZNCC_WIN_SIZE % 2 != 0, "ZNCC_WIN_SIZE must be odd.");
    static_assert(MAX_DISP > 0 && MAX_DISP < 255, "MAX_DISP must be 1-254 if ZNCC_INVALID_DISP_OUTPUT (in .cl) is 255 and raw_disparity_map is uchar.");

    // Post-Processing Config (RE-ADDED and VERIFIED)
    const int CROSS_CHECK_THRESHOLD = 1;
    // This value must match #define ZNCC_INVALID_DISP_OUTPUT in your kernels.cl
    // It's used by the host to understand what the ZNCC kernel means by "invalid".
    // While not directly passed to zncc_stereo_match if it uses a #define,
    // it's crucial for consistency when interpreting its output or setting up other kernels.
    // const unsigned char KERNEL_ZNCC_INVALID_OUTPUT_MARKER = 255; // Reflects the #define in kernels.cl

    // This is the marker that cross_check_and_fill_kernel will *write* for occluded/mismatched pixels.
    const unsigned char POST_PROCESS_INVALID_MARKER_HOST = 255; 
    // This is the marker that scale_disparity_for_visualization_kernel will use to paint invalid pixels.
    const unsigned char VISUAL_INVALID_MARKER_HOST = 0;   // Black for invalid pixels in the final image


    // Debug flags
    const bool saveFormattedDebugImages = true;
    const bool verifyInputImages = true;    // Set to false for less console output during typical runs
    const bool verifyFormattedImages = true;
    const bool saveRawLRDisparity = true; 
    const bool saveRawRLDisparity = true; 
    const bool saveCrossCheckedRawDisparity = true; // Output of cross_check_and_fill


    // Image data
    std::vector<unsigned char> im0_data_rgba, im1_data_rgba;
    unsigned width = 0, height = 0;
    unsigned width1_tmp, height1_tmp;
    unsigned out_width = 0, out_height = 0;

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
    cl_kernel crossCheckAndFillKernel = nullptr; 
    cl_kernel scaleDispKernel = nullptr;        
    
    // OpenCL Images
    cl_mem im0_image_rgba_cl = nullptr; 
    cl_mem im1_image_rgba_cl = nullptr; 
    cl_mem im0_formated_uchar_cl = nullptr; 
    cl_mem im1_formated_uchar_cl = nullptr; 

    // OpenCL Buffers
    cl_mem d_rawDispLR = nullptr;          // Raw disparity map from left-to-right matching
    cl_mem d_rawDispRL = nullptr;          // Raw disparity map from right-to-left matching
    cl_mem d_finalRawDisp = nullptr; // Output of cross_check_and_fill_kernel (raw disparities)
    cl_mem d_visualizableDispMap = nullptr; // Final scaled disparity map for saving
    
    // Track resources for cleanup
    std::vector<cl_mem> cl_buffers;
    std::vector<cl_mem> cl_images;

    // Timers
    std::vector<double> timings_ms;
    std::vector<std::string> timing_labels;


    try {
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // 1. Load Images
        timing_labels.push_back("Image Loading");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        auto stage_start_time = std::chrono::high_resolution_clock::now();
        if (!loadImages(left_image_path, right_image_path,
                         im0_data_rgba, im1_data_rgba, width, height, width1_tmp, height1_tmp)) {
            return 1;
        }
        auto stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        // 2. Initialize OpenCL
        timing_labels.push_back("OpenCL Init");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        if (!initOpenCL(platform, device, context, queue)) {
            return 1;
        }
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        // 3. Load and Build Kernels
        timing_labels.push_back("Kernel Build");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        std::string kernelSource = loadKernelSource(kernel_file_path);
        program = createAndBuildProgram(context, device, kernelSource);
        checkError(program == nullptr ? -1 : CL_SUCCESS, "Failed to create or build program.");

        const char* resizeKernelName = "resizeGrayscaleBoxFilter";
        const char* znccKernelName = "zncc_stereo_match";
        const char* crossCheckFillKernelName = "cross_check_and_fill_kernel"; 
        const char* scaleDispKernelName = "scale_disparity_for_visualization_kernel"; 

        std::cout << "Creating kernels..." << std::endl;
        createKernel(program, resizeGrayscaleBoxKernel, resizeKernelName);
        createKernel(program, znccKernel, znccKernelName);
        createKernel(program, crossCheckAndFillKernel, crossCheckFillKernelName); 
        createKernel(program, scaleDispKernel, scaleDispKernelName);             
        checkError(resizeGrayscaleBoxKernel == nullptr ? -1 : CL_SUCCESS, "Failed to create resize kernel.");
        checkError(znccKernel == nullptr ? -1 : CL_SUCCESS, "Failed to create ZNCC kernel.");
        checkError(crossCheckAndFillKernel == nullptr ? -1 : CL_SUCCESS, "Failed to create cross-check & fill kernel.");
        checkError(scaleDispKernel == nullptr ? -1 : CL_SUCCESS, "Failed to create scale disparity kernel.");     

        std::cout << "Using Resize/Grayscale Kernel: " << resizeKernelName << std::endl;
        std::cout << "Using ZNCC Kernel: " << znccKernelName << std::endl;
        std::cout << "Using Cross-Check & Fill Kernel: " << crossCheckFillKernelName << std::endl;
        std::cout << "Using Scale Disparity Kernel: " << scaleDispKernelName << std::endl;
        std::cout << "Box Filter Radius: " << boxFilterRadius << " (" << (2 * boxFilterRadius + 1) << "x" << (2 * boxFilterRadius + 1) << ")" << std::endl;
        std::cout << "ZNCC Window Size: " << ZNCC_WIN_SIZE << "x" << ZNCC_WIN_SIZE << std::endl;
        std::cout << "Max Disparity: " << MAX_DISP << std::endl;
        std::cout << "Cross Check Threshold: " << CROSS_CHECK_THRESHOLD << std::endl;
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;


        // 4. Calculate Output Dimensions & Create Resources
        timing_labels.push_back("Resource Creation");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        if (resizeFactor <= 0.0f) {
            throw std::runtime_error("Resize factor must be positive.");
        }
        out_width = static_cast<unsigned>(std::round(static_cast<float>(width) / resizeFactor));
        out_height = static_cast<unsigned>(std::round(static_cast<float>(height) / resizeFactor));
        out_width = std::max(1u, out_width); 
        out_height = std::max(1u, out_height);
        std::cout << "Input dimensions: " << width << "x" << height << std::endl;
        std::cout << "Calculated output dimensions: " << out_width << "x" << out_height
                  << " (Resize Factor: " << resizeFactor << ")" << std::endl;

        createImageObjects(context, width, height, out_width, out_height,
                                im0_data_rgba, im1_data_rgba,
                                im0_image_rgba_cl, im1_image_rgba_cl,
                                im0_formated_uchar_cl, im1_formated_uchar_cl);
        cl_images.push_back(im0_image_rgba_cl); cl_images.push_back(im1_image_rgba_cl);
        cl_images.push_back(im0_formated_uchar_cl); cl_images.push_back(im1_formated_uchar_cl);

        if(verifyInputImages) {
            verifyImageObjects(queue, im0_image_rgba_cl, width, height, "im0_image_rgba_cl (input)");
            verifyImageObjects(queue, im1_image_rgba_cl, width, height, "im1_image_rgba_cl (input)");
        }

        #ifdef __APPLE__
        // On macOS, use the older sampler creation API
        sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
        #else
        cl_sampler_properties samplerProps[] = {
            CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE,
            CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP_TO_EDGE,
            CL_SAMPLER_FILTER_MODE, CL_FILTER_NEAREST,
            0
        };
        sampler = clCreateSamplerWithProperties(context, samplerProps, &err);
        #endif
        checkError(err, "Failed to create sampler");
        
        size_t disparity_image_size_pixels = static_cast<size_t>(out_width) * out_height;
        size_t disparity_buffer_size_bytes = disparity_image_size_pixels * sizeof(unsigned char);

        d_rawDispLR = clCreateBuffer(context, CL_MEM_READ_WRITE, disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating d_rawDispLR buffer"); cl_buffers.push_back(d_rawDispLR);
        d_rawDispRL = clCreateBuffer(context, CL_MEM_READ_WRITE, disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating d_rawDispRL buffer"); cl_buffers.push_back(d_rawDispRL);
        d_finalRawDisp = clCreateBuffer(context, CL_MEM_READ_WRITE, disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating d_finalRawDisp buffer"); cl_buffers.push_back(d_finalRawDisp);
        d_visualizableDispMap = clCreateBuffer(context, CL_MEM_WRITE_ONLY, disparity_buffer_size_bytes, nullptr, &err);
        checkError(err, "Creating d_visualizableDispMap buffer"); cl_buffers.push_back(d_visualizableDispMap);
        
        std::cout << "Created disparity buffers (" << out_width << "x" << out_height << ", uchar)." << std::endl;
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;


        // 5. Execute Resize & Grayscale Kernel
        timing_labels.push_back("Resize/Grayscale");
        std::cout << "\nExecuting " << timing_labels.back() << " Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        size_t globalSizeResize[2] = {out_width, out_height};
        cl_int err_arg = 0;
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im0_image_rgba_cl);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im0_formated_uchar_cl);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 2, sizeof(cl_float), &resizeFactor);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 3, sizeof(cl_int), &boxFilterRadius);
        checkError(err_arg, "Setting resize kernel arguments for im0");
        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL, globalSizeResize, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im0");
        err_arg = 0;
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 0, sizeof(cl_mem), &im1_image_rgba_cl);
        err_arg |= clSetKernelArg(resizeGrayscaleBoxKernel, 1, sizeof(cl_mem), &im1_formated_uchar_cl);
        // Args 2 & 3 (resizeFactor, boxFilterRadius) are already set if they don't change
        checkError(err_arg, "Re-setting resize kernel image arguments for im1");
        err = clEnqueueNDRangeKernel(queue, resizeGrayscaleBoxKernel, 2, NULL, globalSizeResize, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing resize kernel for im1");
        err = clFinish(queue);
        checkError(err, "Waiting for resize kernels to finish");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        if (verifyFormattedImages || saveFormattedDebugImages) {
             std::cout << "\nProcessing Intermediate Grayscale Images..." << std::endl;
             if (verifyFormattedImages) {
                verifyImageObjects(queue, im0_formated_uchar_cl, out_width, out_height, "im0_formated_uchar_cl (after resize)");
                verifyImageObjects(queue, im1_formated_uchar_cl, out_width, out_height, "im1_formated_uchar_cl (after resize)");
             }
             if (saveFormattedDebugImages) {
                 std::vector<unsigned char> im0_formatted_host(disparity_image_size_pixels);
                 std::vector<unsigned char> im1_formatted_host(disparity_image_size_pixels);
                 size_t origin[3] = {0, 0, 0}; size_t region[3] = {out_width, out_height, 1};
                 err = clEnqueueReadImage(queue, im0_formated_uchar_cl, CL_TRUE, origin, region, 0, 0, im0_formatted_host.data(), 0, NULL, NULL);
                 checkError(err, "Reading formatted image 0 for debug saving");
                 err = clEnqueueReadImage(queue, im1_formated_uchar_cl, CL_TRUE, origin, region, 0, 0, im1_formatted_host.data(), 0, NULL, NULL);
                 checkError(err, "Reading formatted image 1 for debug saving");
                 saveGrayscalePNG(output_dir + "im0_formated_debug.png", im0_formatted_host, out_width, out_height);
                 saveGrayscalePNG(output_dir + "im1_formated_debug.png", im1_formatted_host, out_width, out_height);
             }
        }

        // 6. Execute ZNCC Kernel for LR Disparity
        timing_labels.push_back("ZNCC LR Disparity");
        std::cout << "\nExecuting " << timing_labels.back() << " Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        // Définir la taille des groupes de travail pour optimiser l'utilisation de la mémoire locale
        // Choisir une taille qui est un multiple de 16 pour de meilleures performances sur GPU
        const size_t localSizeZNCC[2] = {16, 16}; // 16x16 = 256 threads par groupe
        
        // Ajuster la taille globale pour qu'elle soit un multiple de la taille locale
        size_t globalSizeZNCC[2] = {
            ((out_width + localSizeZNCC[0] - 1) / localSizeZNCC[0]) * localSizeZNCC[0],
            ((out_height + localSizeZNCC[1] - 1) / localSizeZNCC[1]) * localSizeZNCC[1]
        };
        int direction_right_to_left = 0; // 0 = left-to-right matching, 1 = right-to-left matching
        err_arg = 0;
        err_arg |= clSetKernelArg(znccKernel, 0, sizeof(cl_mem), &im0_formated_uchar_cl);
        err_arg |= clSetKernelArg(znccKernel, 1, sizeof(cl_mem), &im1_formated_uchar_cl);
        err_arg |= clSetKernelArg(znccKernel, 2, sizeof(cl_mem), &d_rawDispLR);
        err_arg |= clSetKernelArg(znccKernel, 3, sizeof(int), &out_width);
        err_arg |= clSetKernelArg(znccKernel, 4, sizeof(int), &out_height);
        err_arg |= clSetKernelArg(znccKernel, 5, sizeof(int), &ZNCC_WIN_SIZE);
        err_arg |= clSetKernelArg(znccKernel, 6, sizeof(int), &MAX_DISP);
        err_arg |= clSetKernelArg(znccKernel, 7, sizeof(cl_sampler), &sampler);
        err_arg |= clSetKernelArg(znccKernel, 8, sizeof(int), &direction_right_to_left); 

        checkError(err_arg, "Setting ZNCC (LR) kernel arguments");
        err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, globalSizeZNCC, localSizeZNCC, 0, NULL, NULL);
        checkError(err, "Enqueueing ZNCC (LR) kernel");
        err = clFinish(queue); 
        checkError(err, "Waiting for ZNCC (LR) kernel to finish");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        if (saveRawLRDisparity) { 
            std::vector<unsigned char> raw_lr_host(disparity_image_size_pixels);
            err = clEnqueueReadBuffer(queue, d_rawDispLR, CL_TRUE, 0, disparity_buffer_size_bytes, raw_lr_host.data(), 0, NULL, NULL);
            checkError(err, "Reading raw LR disparity for debug");
            saveGrayscalePNG(output_dir + "debug_raw_disp_lr.png", raw_lr_host, out_width, out_height);
        }

        // 7. Execute ZNCC Kernel for RL Disparity
        timing_labels.push_back("ZNCC RL Disparity");
        std::cout << "\nExecuting " << timing_labels.back() << " Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        direction_right_to_left = 1;
        err_arg = 0;
        err_arg |= clSetKernelArg(znccKernel, 0, sizeof(cl_mem), &im1_formated_uchar_cl); 
        err_arg |= clSetKernelArg(znccKernel, 1, sizeof(cl_mem), &im0_formated_uchar_cl); 
        err_arg |= clSetKernelArg(znccKernel, 2, sizeof(cl_mem), &d_rawDispRL);       
        // Args 3-7 are already set if znccKernel is the same object and no other kernel used it
        // However, it's safer to reset them if unsure or if they could change.
        // For this example, assuming they don't need to be reset for the second ZNCC call.
        err_arg |= clSetKernelArg(znccKernel, 8, sizeof(int), &direction_right_to_left); 

        checkError(err_arg, "Setting ZNCC (RL) kernel arguments (image/buffer only)");
        err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, globalSizeZNCC, localSizeZNCC, 0, NULL, NULL);
        checkError(err, "Enqueueing ZNCC (RL) kernel");
        err = clFinish(queue); 
        checkError(err, "Waiting for ZNCC (RL) kernel to finish");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        if (saveRawRLDisparity) { 
            std::vector<unsigned char> raw_rl_host(disparity_image_size_pixels);
            err = clEnqueueReadBuffer(queue, d_rawDispRL, CL_TRUE, 0, disparity_buffer_size_bytes, raw_rl_host.data(), 0, NULL, NULL);
            checkError(err, "Reading raw RL disparity for debug");
            saveGrayscalePNG(output_dir + "debug_raw_disp_rl.png", raw_rl_host, out_width, out_height);
        }

        // 8. Execute Cross-Check and Occlusion Fill Kernel
        timing_labels.push_back("CrossCheck & Fill");
        std::cout << "\nExecuting " << timing_labels.back() << " Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        err_arg = 0;
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 0, sizeof(cl_mem), &d_rawDispLR);
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 1, sizeof(cl_mem), &d_rawDispRL);
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 2, sizeof(cl_mem), &d_finalRawDisp); 
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 3, sizeof(int), &out_width);
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 4, sizeof(int), &out_height);
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 5, sizeof(int), &CROSS_CHECK_THRESHOLD);
        err_arg |= clSetKernelArg(crossCheckAndFillKernel, 6, sizeof(unsigned char), &POST_PROCESS_INVALID_MARKER_HOST);
        checkError(err_arg, "Setting Cross-Check & Fill kernel arguments");
        err = clEnqueueNDRangeKernel(queue, crossCheckAndFillKernel, 2, NULL, globalSizeZNCC, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing Cross-Check & Fill kernel");
        err = clFinish(queue); 
        checkError(err, "Waiting for Cross-Check & Fill kernel to finish");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        if (saveCrossCheckedRawDisparity) { 
            std::vector<unsigned char> cross_checked_raw_host(disparity_image_size_pixels);
            err = clEnqueueReadBuffer(queue, d_finalRawDisp, CL_TRUE, 0, disparity_buffer_size_bytes, cross_checked_raw_host.data(), 0, NULL, NULL);
            checkError(err, "Reading cross-checked raw disparity for debug");
            saveGrayscalePNG(output_dir + "debug_cross_checked_filled_raw_disp.png", cross_checked_raw_host, out_width, out_height);
        }

        // 9. Execute Scale Disparity for Visualization Kernel
        timing_labels.push_back("Scale Disparity Map");
        std::cout << "\nExecuting " << timing_labels.back() << " Kernel..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        err_arg = 0;
        err_arg |= clSetKernelArg(scaleDispKernel, 0, sizeof(cl_mem), &d_finalRawDisp);
        err_arg |= clSetKernelArg(scaleDispKernel, 1, sizeof(cl_mem), &d_visualizableDispMap); 
        err_arg |= clSetKernelArg(scaleDispKernel, 2, sizeof(int), &out_width);
        err_arg |= clSetKernelArg(scaleDispKernel, 3, sizeof(int), &out_height);
        err_arg |= clSetKernelArg(scaleDispKernel, 4, sizeof(int), &MAX_DISP); // This is MAX_DISP_PARAM in kernel                  
        err_arg |= clSetKernelArg(scaleDispKernel, 5, sizeof(unsigned char), &POST_PROCESS_INVALID_MARKER_HOST); 
        err_arg |= clSetKernelArg(scaleDispKernel, 6, sizeof(unsigned char), &VISUAL_INVALID_MARKER_HOST);    
        checkError(err_arg, "Setting Scale Disparity kernel arguments");
        err = clEnqueueNDRangeKernel(queue, scaleDispKernel, 2, NULL, globalSizeZNCC, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing Scale Disparity kernel");
        err = clFinish(queue); 
        checkError(err, "Waiting for Scale Disparity kernel to finish");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;


        // 10. Read Final Visualizable Disparity Map
        timing_labels.push_back("Final Disparity Readback");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        std::vector<unsigned char> disparity_map_host(disparity_image_size_pixels);
        err = clEnqueueReadBuffer(queue, d_visualizableDispMap, CL_TRUE, 
                                  0, disparity_buffer_size_bytes,
                                  disparity_map_host.data(), 0, NULL, NULL);
        checkError(err, "Reading visualizable disparity map buffer back to host");
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;

        // 11. Save Disparity Map
        std::cout << "\nSaving Output..." << std::endl;
        saveGrayscalePNG(output_dir + "disparity_map_ocl_final_visual.png", disparity_map_host, out_width, out_height);

        // 12. Cleanup
        timing_labels.push_back("Cleanup");
        std::cout << "\n" << timing_labels.back() << "..." << std::endl;
        stage_start_time = std::chrono::high_resolution_clock::now();
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel, crossCheckAndFillKernel, scaleDispKernel,
                sampler,
                cl_buffers, cl_images);
        resizeGrayscaleBoxKernel = nullptr; znccKernel = nullptr; 
        crossCheckAndFillKernel = nullptr; scaleDispKernel = nullptr; 
        sampler = nullptr;
        context = nullptr; queue = nullptr; program = nullptr;
        stage_end_time = std::chrono::high_resolution_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(stage_end_time - stage_start_time).count());
        std::cout << timings_ms.back() << " ms." << std::endl;


        // Calculate Total Time & Summary
         auto total_end_time = std::chrono::high_resolution_clock::now();
         double total_duration_ms = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
         
         std::cout << "\n-------------------- Summary --------------------" << std::endl;
         double gpu_processing_plus_readback_ms = 0.0;
         for (size_t k_idx = 0; k_idx < timings_ms.size(); ++k_idx) {
             if (k_idx < timing_labels.size()) {
                std::string label_padded = timing_labels[k_idx];
                if (label_padded.length() < 28) { 
                    label_padded.append(28 - label_padded.length(), ' ');
                }
                std::cout << label_padded << ": " << timings_ms[k_idx] << " ms" << std::endl;
                 if (timing_labels[k_idx] != "Image Loading" && 
                     timing_labels[k_idx] != "OpenCL Init" &&
                     timing_labels[k_idx] != "Kernel Build" &&
                     timing_labels[k_idx] != "Resource Creation" &&
                     timing_labels[k_idx] != "Cleanup") {
                     gpu_processing_plus_readback_ms += timings_ms[k_idx];
                 }
             } else {
                 std::cout << "Timing " << k_idx << std::string(15, ' ') << ": " << timings_ms[k_idx] << " ms (No Label)" << std::endl;
             }
         }
         std::cout << "-------------------------------------------------" << std::endl;
         std::cout << "Total GPU Kernels + Readback Time  : " << gpu_processing_plus_readback_ms << " ms" << std::endl;
         std::cout << "Total Application Time             : " << total_duration_ms << " ms" << std::endl;
         std::cout << "-------------------------------------------------" << std::endl;

        return 0;
    }
    catch (const OpenCLException& e) {
        std::cerr << "\n\n!--- OpenCL Error ---!\nMessage: " << e.what() << "\nError Code: " << e.getErrorCode() << std::endl;
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel, crossCheckAndFillKernel, scaleDispKernel,
                sampler, cl_buffers, cl_images);
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n\n!--- Standard Error ---!\nMessage: " << e.what() << std::endl;
        cleanup(context, queue, program,
                resizeGrayscaleBoxKernel, znccKernel, crossCheckAndFillKernel, scaleDispKernel,
                sampler, cl_buffers, cl_images);
        return 1;
    }
}