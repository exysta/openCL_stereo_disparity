#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "lodepng.h"

// Always include OpenMP header, usage will be conditional
#include <omp.h>

using namespace std;

struct Image {
    int width, height;
    vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

// Structure to track execution timing
struct ExecutionTiming {
    double readTime;
    double resizeTime;
    double grayscaleTime;
    double filterTime;
    double writeTime;
    double totalTime;
    double scaleAndGreyscaleTime;
    double disparity_elapsed;
};

// Add a global variable to track whether OpenMP should be used
bool useOpenMP = false;

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
    
    // Conditionally use OpenMP
    if (useOpenMP) {
        #pragma omp parallel for
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
    } else {
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
    
    if (useOpenMP) {
        #pragma omp parallel for
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
    } else {
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
    
    if (useOpenMP) {
        #pragma omp parallel for
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
    } else {
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
    }
    
    std::cout << "Applied 5x5 Gaussian blur filter" << std::endl;
    return filtered;
}

// Function to save a grayscale image using LodePNG
void writeImage(const char* filename, const Image& grayscale) {
    // Convert the grayscale image to RGBA for saving
    std::vector<unsigned char> output_data(grayscale.width * grayscale.height * 4);
    
    if (useOpenMP) {
        #pragma omp parallel for
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
    } else {
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



Image load_image(const char* filename) {
    Image img;
    int channels;
    unsigned char* data = stbi_load(filename, &img.width, &img.height, &channels, 1);
    
    if(!data) {
        cerr << "Error loading image: " << filename << endl;
        exit(1);
    }
    
    img.data.assign(data, data + img.width * img.height);
    stbi_image_free(data);
    return img;
}

void save_disparity(const char* filename, const Image& img) {
    stbi_write_png(filename, img.width, img.height, 1, img.data.data(), img.width);
}

// Pre-compute window values for both images
void precomputeWindowValues(const Image& img, vector<double>& means, vector<double>& stdDevs, int win_size) {
    int width = img.width;
    int height = img.height;
    
    means.resize(width * height);
    stdDevs.resize(width * height);

    if (useOpenMP) {
        #pragma omp parallel for
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                double sum = 0;
                double sumSq = 0;
                int count = 0;
                
                for(int wy = max(0, y - win_size); wy <= min(height - 1, y + win_size); wy++) {
                    for(int wx = max(0, x - win_size); wx <= min(width - 1, x + win_size); wx++) {
                        double val = img.at(wx, wy);
                        sum += val;
                        sumSq += val * val;
                        count++;
                    }
                }
                
                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);
                double stdDev = sqrt(max(0.0, variance));
                
                means[y * width + x] = mean;
                stdDevs[y * width + x] = stdDev;
            }
        }
    } else {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                double sum = 0;
                double sumSq = 0;
                int count = 0;
                
                for(int wy = max(0, y - win_size); wy <= min(height - 1, y + win_size); wy++) {
                    for(int wx = max(0, x - win_size); wx <= min(width - 1, x + win_size); wx++) {
                        double val = img.at(wx, wy);
                        sum += val;
                        sumSq += val * val;
                        count++;
                    }
                }
                
                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);
                double stdDev = sqrt(max(0.0, variance));
                
                means[y * width + x] = mean;
                stdDevs[y * width + x] = stdDev;
            }
        }
    }
}

Image computeDisparity(const Image& left, const Image& right, int max_disp, int win_size) {
    int width = left.width;
    int height = left.height;
    Image disparity{width, height, vector<unsigned char>(width * height, 0)};
    
    // Pre-compute window means and standard deviations
    vector<double> leftMeans, leftStdDevs, rightMeans, rightStdDevs;
    precomputeWindowValues(left, leftMeans, leftStdDevs, win_size);
    precomputeWindowValues(right, rightMeans, rightStdDevs, win_size);
    
    if (useOpenMP) {
        #pragma omp parallel for
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                float max_zncc = -1.0;  // ZNCC range is [-1,1]
                int best_d = 0;
                
                double meanL = leftMeans[y * width + x];
                double stdDevL = leftStdDevs[y * width + x];
                
                // Skip computation if standard deviation is too small (uniform area)
                if(stdDevL < 1.0) continue;

                for(int d = 0; d <= min(max_disp, x); d++) {
                    int xR = x - d;
                    
                    double meanR = rightMeans[y * width + xR];
                    double stdDevR = rightStdDevs[y * width + xR];
                    
                    // Skip if right window has uniform texture
                    if(stdDevR < 1.0) continue;
                    
                    // Calculate ZNCC directly
                    double numerator = 0;
                    int validPoints = 0;
                    
                    for(int wy = max(0, y - win_size); wy <= min(height - 1, y + win_size); wy++) {
                        // Process window rows in continuous memory blocks when possible
                        int wxStart = max(0, x - win_size);
                        int wxEnd = min(width - 1, x + win_size);
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
    } else {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                float max_zncc = -1.0;  // ZNCC range is [-1,1]
                int best_d = 0;
                
                double meanL = leftMeans[y * width + x];
                double stdDevL = leftStdDevs[y * width + x];
                
                // Skip computation if standard deviation is too small (uniform area)
                if(stdDevL < 1.0) continue;

                for(int d = 0; d <= min(max_disp, x); d++) {
                    int xR = x - d;
                    
                    double meanR = rightMeans[y * width + xR];
                    double stdDevR = rightStdDevs[y * width + xR];
                    
                    // Skip if right window has uniform texture
                    if(stdDevR < 1.0) continue;
                    
                    // Calculate ZNCC directly
                    double numerator = 0;
                    int validPoints = 0;
                    
                    for(int wy = max(0, y - win_size); wy <= min(height - 1, y + win_size); wy++) {
                        // Process window rows in continuous memory blocks when possible
                        int wxStart = max(0, x - win_size);
                        int wxEnd = min(width - 1, x + win_size);
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
    }
    
    return disparity;
}

int savePerfToFile(const char* filePath, const chrono::duration<double> elapsed)
{
    // Create an output file stream (ofstream) object
    ofstream outputFile(filePath, std::ios::app);

    // Check if the file was opened successfully
    if (!outputFile) {
        cerr << "Error opening file for writing: " << filePath << endl;
        return 1;
    }
    
    if (useOpenMP) {
        outputFile << "Computation time parallel : " << elapsed.count() << " seconds" << endl;
    } else {
        outputFile << "Computation time single threaded : " << elapsed.count() << " seconds" << endl;
    }

    outputFile.close();

    cout << "Data written to " << filePath << endl;
    return 0;
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
    std::cout << "Total Grayscale execution time: " << timing.scaleAndGreyscaleTime << " seconds" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Disparity computation time: " << timing.disparity_elapsed << " seconds" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total computation time: " << timing.totalTime << " seconds" << endl;
    std::cout << "=================================" << std::endl;
}

int ScaleAndGreyscale(ExecutionTiming& timing)
{
    auto start_scaleAndGreyscale = std::chrono::high_resolution_clock::now();

    // Read the image
    auto start = std::chrono::high_resolution_clock::now();
    Image originalImage0 = readImage("../ressources/im0.png");
    Image originalImage1 = readImage("../ressources/im1.png");

    auto end = std::chrono::high_resolution_clock::now();
    timing.readTime = std::chrono::duration<double>(end - start).count();
    
    if (originalImage0.data.empty() || originalImage1.data.empty() ) {
        std::cerr << "Failed to read the images." << std::endl;
        return 1;
    }
    
    // Resize the image
    start = std::chrono::high_resolution_clock::now();
    Image resizedImage0 = resizeImage(originalImage0);
    Image resizedImage1 = resizeImage(originalImage1);

    end = std::chrono::high_resolution_clock::now();
    timing.resizeTime = std::chrono::duration<double>(end - start).count();
    
    // Save the resized image for visualization
    writeRGBAImage("../output/image_0_resized.png", resizedImage0);
    writeRGBAImage("../output/image_1_resized.png", resizedImage1);

    // Convert to grayscale
    start = std::chrono::high_resolution_clock::now();
    Image grayscaleImage0 = convertToGrayscale(resizedImage0);
    Image grayscaleImage1 = convertToGrayscale(resizedImage1);

    end = std::chrono::high_resolution_clock::now();
    timing.grayscaleTime = std::chrono::duration<double>(end - start).count();
    
    // Save the grayscale image
    writeImage("../output/image_0_grayscale.png", grayscaleImage0);
    writeImage("../output/image_1_grayscale.png", grayscaleImage1);

    // Apply the filter
    start = std::chrono::high_resolution_clock::now();
    Image filteredImage0 = applyFilter(grayscaleImage0);
    Image filteredImage1 = applyFilter(grayscaleImage1);

    end = std::chrono::high_resolution_clock::now();
    timing.filterTime = std::chrono::duration<double>(end - start).count();
    
    // Write the final image
    start = std::chrono::high_resolution_clock::now();
    writeImage("../output/image_0_bw.png", filteredImage0);
    writeImage("../output/image_1_bw.png", filteredImage1);

    end = std::chrono::high_resolution_clock::now();
    timing.writeTime = std::chrono::duration<double>(end - start).count();

    // Calculate total execution time
    auto end_scaleAndGreyscale = std::chrono::high_resolution_clock::now();
    timing.scaleAndGreyscaleTime = std::chrono::duration<double>(end_scaleAndGreyscale - start_scaleAndGreyscale).count();
    return 0;
}

int main(int argc, char* argv[]) {
    ExecutionTiming timing;

    // Parse command line arguments to determine whether to use OpenMP
    if (argc > 1) {
        string arg = argv[1];
        if (arg == "-p" || arg == "--parallel") {
            useOpenMP = true;
            cout << "Running in parallel mode with OpenMP" << endl;
        } else if (arg == "-s" || arg == "--serial") {
            useOpenMP = false;
            cout << "Running in serial mode without OpenMP" << endl;
        } else {
            cout << "Unknown option: " << arg << endl;
            cout << "Usage: " << argv[0] << " [-p|--parallel|-s|--serial] [other arguments...]" << endl;
            cout << "Default: Running in serial mode" << endl;
        }
    }

    // Shift arguments if necessary
    if (argc > 1) {
        for (int i = 1; i < argc - 1; i++) {
            argv[i] = argv[i + 1];
        }
        argc--;
    }

    auto start_total = std::chrono::high_resolution_clock::now();

    ScaleAndGreyscale(timing);
    const char* left_path = "../ressources/image_0_bw.png";
    const char* right_path = "../ressources/image_1_bw.png";
    
    // Set output path based on OpenMP usage
    const char* output_path;
    if (useOpenMP) {
        output_path = "../output/disparityParallel.png";
    } else {
        output_path = "../output/disparitySingleThread.png";
    }
    
    const char* logfilePath = "../output/log.txt";

    int max_disparity = 50;
    int window_size = 9;
    
    // Parse command line arguments if provided
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
    
    cout << "Loading images..." << endl;
    Image left = load_image(left_path);
    Image right = load_image(right_path);
    
    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }



    cout << "Computing disparity map..." << endl;
    auto start_disparity_time = chrono::high_resolution_clock::now();
    Image disparity = computeDisparity(left, right, max_disparity, window_size);
    auto end_disparity_time = chrono::high_resolution_clock::now();
    chrono::duration<double> disparity_elapsed = end_disparity_time - start_disparity_time;
    timing.disparity_elapsed = disparity_elapsed.count();


    auto end_total = std::chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed =  end_total - start_total;
    timing.totalTime =  total_elapsed.count();

    // Display profiling information
    displayProfilingInfo(timing);
    //savePerfToFile(logfilePath,elapsed);

    cout << "Saving disparity map to " << output_path << endl;
    save_disparity(output_path, disparity);

    return 0;
}