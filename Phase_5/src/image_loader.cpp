#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <OpenCL/opencl.h>
#include "stb_image.h"
#include <iostream>
#include <chrono>
#include "stb_image_write.h"
#include "StereoDisparity.h"
#include "image_processing.h"

int main(int argc, char* argv[]) {
    const char* left_path = "ressources/image_0_bw.png";
    const char* right_path = "ressources/image_1_bw.png";
    const char* output_path = "disparity_map.png";
    int max_disparity = 50;
    int window_size = 9;
    bool use_opencl = true; // Default to use OpenCL
    
    // Process command line arguments
    if(argc > 2) {
        left_path = argv[1];
        right_path = argv[2];
    }
    if(argc > 3) {
        output_path = argv[3];
    }
    if(argc > 4) {
        max_disparity = atoi(argv[4]);
    }
    if(argc > 5) {
        window_size = atoi(argv[5]);
    }
    if(argc > 6) {
        use_opencl = (atoi(argv[6]) != 0); // 0 for CPU, other for OpenCL
    }
    
    std::cout << "Loading images..." << std::endl;
    
    if (use_opencl) {
        // ===== OpenCL Version =====
        cl_int err;

        cl_uint numPlatforms;
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        if (err != CL_SUCCESS || numPlatforms == 0) {
            std::cerr << "No OpenCL platform found. Using CPU version." << std::endl;
            use_opencl = false;
        } else {
            std::vector<cl_platform_id> platforms(numPlatforms);
            err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
            if (err != CL_SUCCESS) {
                std::cerr << "Error retrieving platforms. Using CPU version." << std::endl;
                use_opencl = false;
            } else {
                cl_platform_id platform = platforms[0];

                cl_uint numDevices;
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
                if (err != CL_SUCCESS || numDevices == 0) {
                    std::cerr << "No OpenCL device found. Using CPU version." << std::endl;
                    use_opencl = false;
                } else {
                    std::vector<cl_device_id> devices(numDevices);
                    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
                    if (err != CL_SUCCESS) {
                        std::cerr << "Error retrieving devices. Using CPU version." << std::endl;
                        use_opencl = false;
                    } else {
                        cl_device_id device = devices[0];
                        
                        // Get and display GPU information
                        cl_ulong local_mem_size;
                        cl_ulong global_mem_size;
                        cl_uint compute_units;
                        cl_uint max_clock_freq;
                        cl_uint max_const_buffer_size;
                        size_t max_work_group_size;
                        cl_uint max_work_item_dims;
                        size_t max_work_item_sizes[3];
                        
                        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_freq, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_uint), &max_const_buffer_size, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dims, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
                        
                        std::cout << "\n===== GPU Information =====" << std::endl;
                        std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << local_mem_size << " bytes" << std::endl;
                        std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: " << global_mem_size / (1024*1024) << " MB" << std::endl;
                        std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << compute_units << std::endl;
                        std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << max_clock_freq << " MHz" << std::endl;
                        std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << max_const_buffer_size / 1024 << " KB" << std::endl;
                        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << max_work_group_size << std::endl;
                        std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_work_item_dims << std::endl;
                        std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: [" << max_work_item_sizes[0] << ", " 
                                  << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]" << std::endl;
                        std::cout << "===================================\n" << std::endl;
                        
                        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
                        if (err != CL_SUCCESS) {
                            std::cerr << "Error creating context. Using CPU version." << std::endl;
                            use_opencl = false;
                        } else {
                            // Create a command queue with profiling support
                            cl_command_queue commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
                            if (err != CL_SUCCESS) {
                                std::cerr << "Error creating command queue. Using CPU version." << std::endl;
                                clReleaseContext(context);
                                use_opencl = false;
                            } else {
                                std::cout << "OpenCL context and command queue created successfully." << std::endl;

                                int width, height, channels;

                                unsigned char* image0 = stbi_load(left_path, &width, &height, &channels, 0);
                                unsigned char* image1 = stbi_load(right_path, &width, &height, &channels, 0);
                                if (!image0 || !image1) {
                                    std::cerr << "Error loading images." << std::endl;
                                    if (image0) stbi_image_free(image0);
                                    if (image1) stbi_image_free(image1);
                                    clReleaseCommandQueue(commandQueue);
                                    clReleaseContext(context);
                                    return 1;
                                }
                                
                                std::cout << "Left image: " << width << "x" << height << ", " << channels << " channels" << std::endl;
                                std::cout << "Right image: " << width << "x" << height << ", " << channels << " channels" << std::endl;
                                
                                // Convert images to grayscale with OpenCL
                                std::cout << "\nConverting images to grayscale with OpenCL..." << std::endl;
                                std::vector<unsigned char> grayImage0 = convertToGrayscaleGPU(context, commandQueue, image0, width, height, channels);
                                std::vector<unsigned char> grayImage1 = convertToGrayscaleGPU(context, commandQueue, image1, width, height, channels);
                                
                                // Resize images if necessary
                                int target_width = width;
                                int target_height = height;
                                bool resize_images = false;
                                
                                // If images are too large, resize them
                                if (width > 1024 || height > 1024) {
                                    resize_images = true;
                                    float scale = std::min(1024.0f / width, 1024.0f / height);
                                    target_width = static_cast<int>(width * scale);
                                    target_height = static_cast<int>(height * scale);
                                    std::cout << "\nResizing images to " << target_width << "x" << target_height << "..." << std::endl;
                                    
                                    grayImage0 = resizeGrayscaleGPU(context, commandQueue, grayImage0, width, height, target_width, target_height);
                                    grayImage1 = resizeGrayscaleGPU(context, commandQueue, grayImage1, width, height, target_width, target_height);
                                    
                                    // Save resized images for verification
                                    stbi_write_png("resized_left.png", target_width, target_height, 1, grayImage0.data(), target_width);
                                    stbi_write_png("resized_right.png", target_width, target_height, 1, grayImage1.data(), target_width);
                                    std::cout << "Resized images saved as 'resized_left.png' and 'resized_right.png'" << std::endl;
                                }

                                // Save original grayscale images for comparison
                                stbi_write_png("left_gray.png", width, height, 1, grayImage0.data(), width);
                                stbi_write_png("right_gray.png", width, height, 1, grayImage1.data(), width);
                                std::cout << "Original grayscale images saved as 'left_gray.png' and 'right_gray.png'" << std::endl;

                                size_t imageSize = target_width * target_height * sizeof(unsigned char);
                                // Create buffers for grayscale images
                                cl_mem imageBuffer0 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &err);
                                cl_mem imageBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &err);
                                
                                if(err != CL_SUCCESS) {
                                    std::cerr << "Error creating buffers. Using CPU version." << std::endl;
                                    stbi_image_free(image0);
                                    stbi_image_free(image1);
                                    clReleaseCommandQueue(commandQueue);
                                    clReleaseContext(context);
                                    use_opencl = false;
                                } else {
                                    // Copy grayscale image data into buffers
                                    err = clEnqueueWriteBuffer(commandQueue, imageBuffer0, CL_TRUE, 0, imageSize, grayImage0.data(), 0, NULL, NULL);
                                    err |= clEnqueueWriteBuffer(commandQueue, imageBuffer1, CL_TRUE, 0, imageSize, grayImage1.data(), 0, NULL, NULL);
                                    
                                    if(err != CL_SUCCESS) {
                                        std::cerr << "Error copying image data. Using CPU version." << std::endl;
                                        stbi_image_free(image0);
                                        stbi_image_free(image1);
                                        clReleaseMemObject(imageBuffer0);
                                        clReleaseMemObject(imageBuffer1);
                                        clReleaseCommandQueue(commandQueue);
                                        clReleaseContext(context);
                                        use_opencl = false;
                                    } else {
                                        // Execute disparity calculation with OpenCL
                                        std::cout << "\nCalculating disparity map with OpenCL..." << std::endl;
                                        auto start_time = std::chrono::high_resolution_clock::now();
                                        
                                        std::vector<unsigned char> disparity = computeStereoDisparity(context, commandQueue, imageBuffer0, imageBuffer1, target_width, target_height, 1);
                                        
                                        auto end_time = std::chrono::high_resolution_clock::now();
                                        std::chrono::duration<double> elapsed = end_time - start_time;
                                        std::cout << "OpenCL computation time: " << elapsed.count() << " seconds" << std::endl;
                                        
                                        // Save disparity map
                                        std::cout << "Saving disparity map to " << output_path << std::endl;
                                        stbi_write_png(output_path, target_width, target_height, 1, disparity.data(), target_width);
                                        
                                        // Note for the user about result files
                                        std::cout << "\nResult files generated:" << std::endl;
                                        std::cout << "1. left_gray.png, right_gray.png - Images originales en niveaux de gris" << std::endl;
                                        if (resize_images) {
                                            std::cout << "2. resized_left.png, resized_right.png - Images redimensionnées" << std::endl;
                                        }
                                        std::cout << "3. disparity_map.png - Carte de disparité finale après lissage" << std::endl;
                                        std::cout << "4. disparity_cross_checked.png - Carte de disparité après cross-check" << std::endl;
                                        std::cout << "5. disparity_occlusion_filled.png - Carte de disparité après remplissage des occlusions" << std::endl;
                                        
                                        // Release resources
                                        stbi_image_free(image0);
                                        stbi_image_free(image1);
                                        clReleaseMemObject(imageBuffer0);
                                        clReleaseMemObject(imageBuffer1);
                                        clReleaseCommandQueue(commandQueue);
                                        clReleaseContext(context);
                                        
                                        return 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ===== CPU Version =====
    if (!use_opencl) {
        std::cout << "Using CPU algorithm..." << std::endl;
        
        // Load images
        Image left = load_image(left_path);
        Image right = load_image(right_path);
        
        if(left.width != right.width || left.height != right.height) {
            std::cerr << "Error: Images must have the same dimensions!" << std::endl;
            return 1;
        }
        
        std::cout << "Calculating disparity map with CPU..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Image disparity = computeDisparityCPU(left, right, max_disparity, window_size);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "CPU computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // Save disparity map
        std::cout << "Saving disparity map to " << output_path << std::endl;
        save_disparity(output_path, disparity);
    }
    
    return 0;
}
