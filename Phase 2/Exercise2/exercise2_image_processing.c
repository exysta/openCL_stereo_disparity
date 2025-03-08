#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "lodepng.h"

// Function to measure time in microseconds
double get_time_in_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

// Function to read an image using LodePNG
unsigned char* ReadImage(const char* filename, unsigned int* width, unsigned int* height) {
    double start_time = get_time_in_microseconds();
    
    unsigned char* image = NULL;
    unsigned error = lodepng_decode32_file(&image, width, height, filename);
    
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return NULL;
    }
    
    double end_time = get_time_in_microseconds();
    printf("ReadImage time: %.2f microseconds\n", end_time - start_time);
    
    return image;
}

// Function to resize the image to 1/4 of the original size
unsigned char* ResizeImage(unsigned char* original, unsigned int original_width, unsigned int original_height, 
                          unsigned int* new_width, unsigned int* new_height) {
    double start_time = get_time_in_microseconds();
    
    *new_width = original_width / 4;
    *new_height = original_height / 4;
    
    unsigned char* resized = (unsigned char*)malloc((*new_width) * (*new_height) * 4);
    if (!resized) {
        printf("Memory allocation failed for resized image\n");
        return NULL;
    }
    
    // Take every 4th pixel
    for (unsigned int y = 0; y < *new_height; y++) {
        for (unsigned int x = 0; x < *new_width; x++) {
            unsigned int original_idx = ((y * 4) * original_width + (x * 4)) * 4;
            unsigned int resized_idx = (y * (*new_width) + x) * 4;
            
            resized[resized_idx + 0] = original[original_idx + 0]; // R
            resized[resized_idx + 1] = original[original_idx + 1]; // G
            resized[resized_idx + 2] = original[original_idx + 2]; // B
            resized[resized_idx + 3] = original[original_idx + 3]; // A
        }
    }
    
    double end_time = get_time_in_microseconds();
    printf("ResizeImage time: %.2f microseconds\n", end_time - start_time);
    
    return resized;
}

// Function to convert image to grayscale
unsigned char* GrayScaleImage(unsigned char* rgba, unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    unsigned char* grayscale = (unsigned char*)malloc(width * height);
    if (!grayscale) {
        printf("Memory allocation failed for grayscale image\n");
        return NULL;
    }
    
    for (unsigned int i = 0; i < width * height; i++) {
        unsigned int idx = i * 4;
        // Y = 0.2126R + 0.7152G + 0.0722B
        grayscale[i] = (unsigned char)(0.2126 * rgba[idx] + 0.7152 * rgba[idx + 1] + 0.0722 * rgba[idx + 2]);
    }
    
    double end_time = get_time_in_microseconds();
    printf("GrayScaleImage time: %.2f microseconds\n", end_time - start_time);
    
    return grayscale;
}

// Function to apply a 5x5 filter (Gaussian blur)
unsigned char* ApplyFilter(unsigned char* grayscale, unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    // Gaussian 5x5 filter kernel
    float kernel[5][5] = {
        {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {1.0f/256.0f, 4.0f/256.0f, 6.0f/256.0f, 4.0f/256.0f, 1.0f/256.0f}
    };
    
    unsigned char* filtered = (unsigned char*)malloc(width * height);
    if (!filtered) {
        printf("Memory allocation failed for filtered image\n");
        return NULL;
    }
    
    // Initialize with zeros
    for (unsigned int i = 0; i < width * height; i++) {
        filtered[i] = 0;
    }
    
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
    printf("ApplyFilter time: %.2f microseconds\n", end_time - start_time);
    
    return filtered;
}

// Function to write a grayscale image to file
void WriteImage(const char* filename, unsigned char* grayscale, unsigned int width, unsigned int height) {
    double start_time = get_time_in_microseconds();
    
    // Convert grayscale to RGBA for LodePNG
    unsigned char* rgba = (unsigned char*)malloc(width * height * 4);
    if (!rgba) {
        printf("Memory allocation failed for RGBA conversion\n");
        return;
    }
    
    for (unsigned int i = 0; i < width * height; i++) {
        unsigned int idx = i * 4;
        rgba[idx + 0] = grayscale[i]; // R
        rgba[idx + 1] = grayscale[i]; // G
        rgba[idx + 2] = grayscale[i]; // B
        rgba[idx + 3] = 255;          // A
    }
    
    // Save the image
    unsigned error = lodepng_encode32_file(filename, rgba, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }
    
    free(rgba);
    
    double end_time = get_time_in_microseconds();
    printf("WriteImage time: %.2f microseconds\n", end_time - start_time);
}

// Function to display profiling information
void ProfilingInfo(double total_time, double read_time, double resize_time, 
                  double grayscale_time, double filter_time, double write_time) {
    printf("\n===== Profiling Information =====\n");
    printf("Total execution time: %.2f microseconds (%.2f ms)\n", 
           total_time, total_time / 1000.0);
    printf("ReadImage: %.2f microseconds (%.2f%% of total)\n", 
           read_time, (read_time / total_time) * 100.0);
    printf("ResizeImage: %.2f microseconds (%.2f%% of total)\n", 
           resize_time, (resize_time / total_time) * 100.0);
    printf("GrayScaleImage: %.2f microseconds (%.2f%% of total)\n", 
           grayscale_time, (grayscale_time / total_time) * 100.0);
    printf("ApplyFilter: %.2f microseconds (%.2f%% of total)\n", 
           filter_time, (filter_time / total_time) * 100.0);
    printf("WriteImage: %.2f microseconds (%.2f%% of total)\n", 
           write_time, (write_time / total_time) * 100.0);
    printf("=================================\n");
}

int main() {
    double total_start_time = get_time_in_microseconds();
    double read_time = 0, resize_time = 0, grayscale_time = 0, filter_time = 0, write_time = 0;
    
    // Read image
    unsigned int width, height;
    double read_start = get_time_in_microseconds();
    unsigned char* image = ReadImage("../TestImages/image_0.png", &width, &height);
    double read_end = get_time_in_microseconds();
    read_time = read_end - read_start;
    
    if (!image) {
        printf("Failed to read image\n");
        return EXIT_FAILURE;
    }
    
    printf("Original image dimensions: %u x %u\n", width, height);
    
    // Resize image
    unsigned int resized_width, resized_height;
    double resize_start = get_time_in_microseconds();
    unsigned char* resized = ResizeImage(image, width, height, &resized_width, &resized_height);
    double resize_end = get_time_in_microseconds();
    resize_time = resize_end - resize_start;
    
    if (!resized) {
        printf("Failed to resize image\n");
        free(image);
        return EXIT_FAILURE;
    }
    
    printf("Resized image dimensions: %u x %u\n", resized_width, resized_height);
    
    // Convert to grayscale
    double grayscale_start = get_time_in_microseconds();
    unsigned char* grayscale = GrayScaleImage(resized, resized_width, resized_height);
    double grayscale_end = get_time_in_microseconds();
    grayscale_time = grayscale_end - grayscale_start;
    
    if (!grayscale) {
        printf("Failed to convert to grayscale\n");
        free(image);
        free(resized);
        return EXIT_FAILURE;
    }
    
    // Save grayscale image
    WriteImage("../TestImages/image_0_gray.png", grayscale, resized_width, resized_height);
    
    // Apply filter
    double filter_start = get_time_in_microseconds();
    unsigned char* filtered = ApplyFilter(grayscale, resized_width, resized_height);
    double filter_end = get_time_in_microseconds();
    filter_time = filter_end - filter_start;
    
    if (!filtered) {
        printf("Failed to apply filter\n");
        free(image);
        free(resized);
        free(grayscale);
        return EXIT_FAILURE;
    }
    
    // Save filtered image
    double write_start = get_time_in_microseconds();
    WriteImage("../TestImages/image_0_bw.png", filtered, resized_width, resized_height);
    double write_end = get_time_in_microseconds();
    write_time = write_end - write_start;
    
    double total_end_time = get_time_in_microseconds();
    double total_time = total_end_time - total_start_time;
    
    // Display profiling information
    ProfilingInfo(total_time, read_time, resize_time, grayscale_time, filter_time, write_time);
    
    // Clean up
    free(image);
    free(resized);
    free(grayscale);
    free(filtered);
    
    return EXIT_SUCCESS;
}
