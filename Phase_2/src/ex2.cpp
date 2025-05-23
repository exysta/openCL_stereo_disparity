#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include "lodepng.h"

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
};

// Function to read an image using LodePNG
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

// Function to resize an image to 1/16 of its original size
Image resizeImage(const Image& original) {
    Image resized;
    
    // Calculate new dimensions (every 4th pixel in each dimension)
    resized.width = original.width / 4;
    resized.height = original.height / 4;
    
    // Allocate memory for the resized image (4 bytes per pixel for RGBA)
    resized.data.resize(resized.width * resized.height * 4);
    
    // Sample every 4th pixel from the original image
    for (unsigned y = 0; y < resized.height; y++) {
        for (unsigned x = 0; x < resized.width; x++) {
            unsigned original_x = x * 4;
            unsigned original_y = y * 4;
            
            // Calculate positions in data arrays
            unsigned original_pos = (original_y * original.width + original_x) * 4;
            unsigned resized_pos = (y * resized.width + x) * 4;
            
            // Copy the pixel
            for (int c = 0; c < 4; c++) {
                resized.data[resized_pos + c] = original.data[original_pos + c];
            }
        }
    }
    
    std::cout << "Resized image to " << resized.width << "x" << resized.height << std::endl;
    return resized;
}

// Function to convert an RGBA image to grayscale
Image convertToGrayscale(const Image& color) {
    Image grayscale;
    grayscale.width = color.width;
    grayscale.height = color.height;
    
    // For grayscale, we only need 1 byte per pixel
    grayscale.data.resize(grayscale.width * grayscale.height);
    
    for (unsigned y = 0; y < grayscale.height; y++) {
        for (unsigned x = 0; x < grayscale.width; x++) {
            // Position in the color image (4 bytes per pixel)
            unsigned color_pos = (y * grayscale.width + x) * 4;
            
            // Position in the grayscale image (1 byte per pixel)
            unsigned gray_pos = y * grayscale.width + x;
            
            // Convert RGB to grayscale using the recommended formula
            // Y = 0.2126R + 0.7152G + 0.0722B
            unsigned char r = color.data[color_pos];
            unsigned char g = color.data[color_pos + 1];
            unsigned char b = color.data[color_pos + 2];
            
            grayscale.data[gray_pos] = static_cast<unsigned char>(
                0.2126 * r + 0.7152 * g + 0.0722 * b
            );
        }
    }
    
    std::cout << "Converted image to grayscale" << std::endl;
    return grayscale;
}

// Function to apply a 5x5 Gaussian blur filter to a grayscale image
Image applyFilter(const Image& grayscale) {
    Image filtered;
    filtered.width = grayscale.width;
    filtered.height = grayscale.height;
    filtered.data.resize(filtered.width * filtered.height);
    
    // Define a 5x5 Gaussian blur kernel
    // This is a typical Gaussian kernel with σ ≈ 1.0
    const float kernel[5][5] = {
        {1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f}
    };
    
    // Apply the filter
    for (unsigned y = 0; y < filtered.height; y++) {
        for (unsigned x = 0; x < filtered.width; x++) {
            float sum = 0.0f;
            
            // Apply the kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    // Calculate the position to sample from the input image
                    int sampleX = x + kx;
                    int sampleY = y + ky;
                    
                    // Handle boundary conditions (mirror padding)
                    if (sampleX < 0) sampleX = -sampleX;
                    if (sampleY < 0) sampleY = -sampleY;
                    if (sampleX >= static_cast<int>(grayscale.width)) sampleX = 2 * grayscale.width - sampleX - 1;
                    if (sampleY >= static_cast<int>(grayscale.height)) sampleY = 2 * grayscale.height - sampleY - 1;
                    
                    // Get the sample value
                    unsigned char sample = grayscale.data[sampleY * grayscale.width + sampleX];
                    
                    // Apply kernel weight
                    sum += sample * kernel[ky + 2][kx + 2];
                }
            }
            
            // Store the result
            filtered.data[y * filtered.width + x] = static_cast<unsigned char>(sum);
        }
    }
    
    std::cout << "Applied 5x5 Gaussian blur filter" << std::endl;
    return filtered;
}

// Function to save a grayscale image using LodePNG
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

// Function to save an RGBA image using LodePNG
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

// Function to display profiling information
void displayProfilingInfo(const ExecutionTiming& timing) {
    std::cout << "\n===== Profiling Information =====" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Read image:          " << timing.readTime << " seconds" << std::endl;
    std::cout << "Resize image:        " << timing.resizeTime << " seconds" << std::endl;
    std::cout << "Convert to grayscale: " << timing.grayscaleTime << " seconds" << std::endl;
    std::cout << "Apply filter:         " << timing.filterTime << " seconds" << std::endl;
    std::cout << "Write image:          " << timing.writeTime << " seconds" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total execution time: " << timing.totalTime << " seconds" << std::endl;
    std::cout << "=================================" << std::endl;
}

int main() {
    ExecutionTiming timing;
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Process first image (im0.png)
    // Read the image
    auto start = std::chrono::high_resolution_clock::now();
    Image originalImage0 = readImage("../ressources/im0.png");
    auto end = std::chrono::high_resolution_clock::now();
    timing.readTime = std::chrono::duration<double>(end - start).count();
    
    if (originalImage0.data.empty()) {
        std::cerr << "Failed to read the image im0.png." << std::endl;
        return 1;
    }
    
    // Resize the image
    start = std::chrono::high_resolution_clock::now();
    Image resizedImage0 = resizeImage(originalImage0);
    end = std::chrono::high_resolution_clock::now();
    timing.resizeTime = std::chrono::duration<double>(end - start).count();
    
    // Save the resized image for visualization
    writeRGBAImage("../output/image_0_resized.png", resizedImage0);
    
    // Convert to grayscale
    start = std::chrono::high_resolution_clock::now();
    Image grayscaleImage0 = convertToGrayscale(resizedImage0);
    end = std::chrono::high_resolution_clock::now();
    timing.grayscaleTime = std::chrono::duration<double>(end - start).count();
    
    // Save the grayscale image
    writeImage("../output/image_0_grayscale.png", grayscaleImage0);
    
    // Apply the filter
    start = std::chrono::high_resolution_clock::now();
    Image filteredImage0 = applyFilter(grayscaleImage0);
    end = std::chrono::high_resolution_clock::now();
    timing.filterTime = std::chrono::duration<double>(end - start).count();
    
    // Write the final image
    start = std::chrono::high_resolution_clock::now();
    writeImage("../output/image_0_bw.png", filteredImage0);
    end = std::chrono::high_resolution_clock::now();
    timing.writeTime = std::chrono::duration<double>(end - start).count();
    
    // Process second image (im1.png)
    std::cout << "\nProcessing second image..." << std::endl;
    
    // Read the image
    start = std::chrono::high_resolution_clock::now();
    Image originalImage1 = readImage("../ressources/im1.png");
    end = std::chrono::high_resolution_clock::now();
    
    if (originalImage1.data.empty()) {
        std::cerr << "Failed to read the image im1.png." << std::endl;
        return 1;
    }
    
    // Resize the image
    Image resizedImage1 = resizeImage(originalImage1);
    
    // Save the resized image for visualization
    writeRGBAImage("../output/image_1_resized.png", resizedImage1);
    
    // Convert to grayscale
    Image grayscaleImage1 = convertToGrayscale(resizedImage1);
    
    // Save the grayscale image
    writeImage("../output/image_1_grayscale.png", grayscaleImage1);
    
    // Apply the filter
    Image filteredImage1 = applyFilter(grayscaleImage1);
    
    // Write the final image
    writeImage("../output/image_1_bw.png", filteredImage1);
    
    // Calculate total execution time
    auto end_total = std::chrono::high_resolution_clock::now();
    timing.totalTime = std::chrono::duration<double>(end_total - start_total).count();
    
    // Display profiling information
    displayProfilingInfo(timing);
    
    return 0;
}