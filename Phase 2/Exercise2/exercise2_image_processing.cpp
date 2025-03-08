#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include "lodepng.h"

// Function to measure time in microseconds
double get_time_in_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

// Function to read an image using LodePNG
std::vector<unsigned char> ReadImage(const char* filename, unsigned int& width, unsigned int& height) {
    double start_time = get_time_in_microseconds();
    
    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, width, height, filename);
    
    if (error) {
        std::cout << "Error " << error << ": " << lodepng_error_text(error) << std::endl;
        return std::vector<unsigned char>();
    }
    
    double end_time = get_time_in_microseconds();
    std::cout << "ReadImage time: " << (end_time - start_time) << " microseconds" << std::endl;
    
    return image;
}

// Function to resize the image to 1/4 of the original size
std::vector<unsigned char> ResizeImage(const std::vector<unsigned char>& original, 
                                      unsigned int original_width, unsigned int original_height, 
                                      unsigned int& new_width, unsigned int& new_height) {
    double start_time = get_time_in_microseconds();
    
    new_width = original_width / 4;
    new_height = original_height / 4;
    
    std::vector<unsigned char> resized(new_width * new_height * 4);
    
    // Take every 4th pixel
    for (unsigned int y = 0; y < new_height; y++) {
        for (unsigned int x = 0; x < new_width; x++) {
            unsigned int original_idx = ((y * 4) * original_width + (x * 4)) * 4;
            unsigned int resized_idx = (y * new_width + x) * 4;
            
            resized[resized_idx + 0] = original[original_idx + 0]; // R
            resized[resized_idx + 1] = original[original_idx + 1]; // G
            resized[resized_idx + 2] = original[original_idx + 2]; // B
            resized[resized_idx + 3] = original[original_idx + 3]; // A
        }
    }
    
    double end_time = get_time_in_microseconds();
    std::cout << "ResizeImage time: " << (end_time - start_time) << " microseconds" << std::endl;
    
    return resized;
}

// Function to convert image to grayscale
std::vector<unsigned char> GrayScaleImage(const std::vector<unsigned char>& rgba, 
                                         unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    std::vector<unsigned char> grayscale(width * height);
    
    for (unsigned int i = 0; i < width * height; i++) {
        unsigned int idx = i * 4;
        // Y = 0.2126R + 0.7152G + 0.0722B
        grayscale[i] = (unsigned char)(0.2126 * rgba[idx] + 0.7152 * rgba[idx + 1] + 0.0722 * rgba[idx + 2]);
    }
    
    double end_time = get_time_in_microseconds();
    std::cout << "GrayScaleImage time: " << (end_time - start_time) << " microseconds" << std::endl;
    
    return grayscale;
}

// Function to apply a 5x5 filter (Gaussian blur)
std::vector<unsigned char> ApplyFilter(const std::vector<unsigned char>& grayscale, 
                                      unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    // Gaussian 5x5 filter kernel
    float kernel[5][5] = {
        {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f}
    };
    
    std::vector<unsigned char> filtered(width * height, 0);
    
    // Apply filter
    for (unsigned int y = 2; y < height - 2; y++) {
        for (unsigned int x = 2; x < width - 2; x++) {
            float sum = 0.0f;
            
            // Apply kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int pixel_x = x + kx;
                    int pixel_y = y + ky;
                    sum += grayscale[pixel_y * width + pixel_x] * kernel[ky + 2][kx + 2];
                }
            }
            
            // Store result
            filtered[y * width + x] = (unsigned char)sum;
        }
    }
    
    double end_time = get_time_in_microseconds();
    std::cout << "ApplyFilter time: " << (end_time - start_time) << " microseconds" << std::endl;
    
    return filtered;
}

// Function to write a grayscale image to file
void WriteImage(const char* filename, const std::vector<unsigned char>& grayscale, 
               unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    // Convert grayscale to RGBA for LodePNG
    std::vector<unsigned char> rgba(width * height * 4);
    
    for (unsigned int i = 0; i < width * height; i++) {
        unsigned int idx = i * 4;
        rgba[idx + 0] = grayscale[i]; // R
        rgba[idx + 1] = grayscale[i]; // G
        rgba[idx + 2] = grayscale[i]; // B
        rgba[idx + 3] = 255;          // A
    }
    
    // Save the image
    unsigned error = lodepng::encode(filename, rgba, width, height);
    if (error) {
        std::cout << "Error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
    
    double end_time = get_time_in_microseconds();
    std::cout << "WriteImage time: " << (end_time - start_time) << " microseconds" << std::endl;
}

// Function to display profiling information
void ProfilingInfo(double total_time, double read_time, double resize_time, 
                  double grayscale_time, double filter_time, double write_time) {
    std::cout << "\n===== Profiling Information =====" << std::endl;
    std::cout << "Total execution time: " << total_time << " microseconds (" 
              << (total_time / 1000.0) << " ms)" << std::endl;
    std::cout << "ReadImage: " << read_time << " microseconds (" 
              << (read_time / total_time) * 100.0 << "% of total)" << std::endl;
    std::cout << "ResizeImage: " << resize_time << " microseconds (" 
              << (resize_time / total_time) * 100.0 << "% of total)" << std::endl;
    std::cout << "GrayScaleImage: " << grayscale_time << " microseconds (" 
              << (grayscale_time / total_time) * 100.0 << "% of total)" << std::endl;
    std::cout << "ApplyFilter: " << filter_time << " microseconds (" 
              << (filter_time / total_time) * 100.0 << "% of total)" << std::endl;
    std::cout << "WriteImage: " << write_time << " microseconds (" 
              << (write_time / total_time) * 100.0 << "% of total)" << std::endl;
    std::cout << "=================================" << std::endl;
}

int main() {
    double total_start_time = get_time_in_microseconds();
    double read_time = 0, resize_time = 0, grayscale_time = 0, filter_time = 0, write_time = 0;
    
    // Read image
    unsigned int width, height;
    double read_start = get_time_in_microseconds();
    std::vector<unsigned char> image = ReadImage("image_0.png", width, height);
    double read_end = get_time_in_microseconds();
    read_time = read_end - read_start;
    
    if (image.empty()) {
        std::cout << "Failed to read image" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Original image dimensions: " << width << " x " << height << std::endl;
    
    // Resize image
    unsigned int resized_width, resized_height;
    double resize_start = get_time_in_microseconds();
    std::vector<unsigned char> resized = ResizeImage(image, width, height, resized_width, resized_height);
    double resize_end = get_time_in_microseconds();
    resize_time = resize_end - resize_start;
    
    if (resized.empty()) {
        std::cout << "Failed to resize image" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Resized image dimensions: " << resized_width << " x " << resized_height << std::endl;
    
    // Convert to grayscale
    double grayscale_start = get_time_in_microseconds();
    std::vector<unsigned char> grayscale = GrayScaleImage(resized, resized_width, resized_height);
    double grayscale_end = get_time_in_microseconds();
    grayscale_time = grayscale_end - grayscale_start;
    
    if (grayscale.empty()) {
        std::cout << "Failed to convert to grayscale" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Save grayscale image
    WriteImage("image_0_gray.png", grayscale, resized_width, resized_height);
    
    // Apply filter
    double filter_start = get_time_in_microseconds();
    std::vector<unsigned char> filtered = ApplyFilter(grayscale, resized_width, resized_height);
    double filter_end = get_time_in_microseconds();
    filter_time = filter_end - filter_start;
    
    if (filtered.empty()) {
        std::cout << "Failed to apply filter" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Save filtered image
    double write_start = get_time_in_microseconds();
    WriteImage("image_0_bw.png", filtered, resized_width, resized_height);
    double write_end = get_time_in_microseconds();
    write_time = write_end - write_start;
    
    double total_end_time = get_time_in_microseconds();
    double total_time = total_end_time - total_start_time;
    
    // Display profiling information
    ProfilingInfo(total_time, read_time, resize_time, grayscale_time, filter_time, write_time);
    
    return EXIT_SUCCESS;
}
