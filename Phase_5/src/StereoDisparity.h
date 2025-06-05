#ifndef STEREO_DISPARITY_H
#define STEREO_DISPARITY_H

#include <vector>
#include <OpenCL/opencl.h>

// Structure to represent an image
struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

// Image loading and saving functions
Image load_image(const char* filename);
void save_disparity(const char* filename, const Image& img);

// Precompute window values for both images
void precomputeWindowValues(const Image& img, std::vector<double>& means, std::vector<double>& stdDevs, int win_size);

// CPU version of the stereo disparity algorithm
Image computeDisparityCPU(const Image& left, const Image& right, int max_disp, int win_size);

// OpenCL version of the stereo disparity algorithm
std::vector<unsigned char> computeStereoDisparity(cl_context context, cl_command_queue commandQueue,
                                                   cl_mem imageBuffer0, cl_mem imageBuffer1,
                                                   int width, int height, int channels);

// Function to display profiling results
void printProfilingInfo(const std::string& kernelName, double executionTime, size_t globalWorkSize);

// CPU version of stereo disparity calculation
std::vector<unsigned char> computeStereoDisparityCPU(unsigned char* image0, unsigned char* image1,
                                                    int width, int height, int channels);

#endif // STEREO_DISPARITY_H