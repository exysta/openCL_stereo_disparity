#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "lodepng.h"

using namespace std;

// Structure to hold image data
struct Image {
    unsigned width, height;
    vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

// Structure to track execution timing
struct ExecutionTiming {
    double loadImagesTime;
    double precomputeWindowsTime;
    double disparityComputationTime;
    double saveImageTime;
    double totalTime;
};

// Function to display profiling information
void displayProfilingInfo(const ExecutionTiming& timing) {
    std::cout << "\n===== Profiling Information =====" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Load images:           " << timing.loadImagesTime << " seconds" << std::endl;
    std::cout << "Precompute windows:    " << timing.precomputeWindowsTime << " seconds" << std::endl;
    std::cout << "Disparity computation: " << timing.disparityComputationTime << " seconds" << std::endl;
    std::cout << "Save image:            " << timing.saveImageTime << " seconds" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total computation time: " << timing.totalTime << " seconds" << std::endl;
    std::cout << "=================================" << std::endl;
}

// Function to read a grayscale image
Image loadImage(const char* filename) {
    vector<unsigned char> png;
    unsigned width, height;
    unsigned error = lodepng::decode(png, width, height, filename, LodePNGColorType::LCT_GREY, 8);
    
    if (error) {
        cerr << "Error loading image: " << lodepng_error_text(error) << endl;
        exit(1);
    }
    
    Image img;
    img.width = width;
    img.height = height;
    img.data = png;
    
    return img;
}

// Function to save a grayscale image
void saveImage(const char* filename, const Image& img) {
    unsigned error = lodepng::encode(filename, img.data, img.width, img.height, LodePNGColorType::LCT_GREY, 8);
    
    if (error) {
        cerr << "Error saving image: " << lodepng_error_text(error) << endl;
    }
}

// Pre-compute window values for both images
void precomputeWindowValues(const Image& img, vector<double>& means, vector<double>& stdDevs, int win_size) {
    int width = img.width;
    int height = img.height;
    
    means.resize(width * height);
    stdDevs.resize(width * height);
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            double sum = 0;
            double sumSq = 0;
            int count = 0;
            
            for(int wy = max(0, y - win_size); wy <= min((int)height - 1, y + win_size); wy++) {
                for(int wx = max(0, x - win_size); wx <= min((int)width - 1, x + win_size); wx++) {
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

// Compute disparity map
Image computeDisparity(const Image& left, const Image& right, int max_disp, int win_size, double& precomputeTime) {
    int width = left.width;
    int height = left.height;
    Image disparity;
    disparity.width = width;
    disparity.height = height;
    disparity.data.resize(width * height, 0);
    
    // Pre-compute window means and standard deviations
    vector<double> leftMeans, leftStdDevs, rightMeans, rightStdDevs;
    
    auto precompute_start = chrono::high_resolution_clock::now();
    precomputeWindowValues(left, leftMeans, leftStdDevs, win_size);
    precomputeWindowValues(right, rightMeans, rightStdDevs, win_size);
    auto precompute_end = chrono::high_resolution_clock::now();
    precomputeTime = chrono::duration<double>(precompute_end - precompute_start).count();
    
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
    
    return disparity;
}

int main(int argc, char* argv[]) {
    ExecutionTiming timing = {}; // Initialize all timing values to 0
    auto start_total = chrono::high_resolution_clock::now();
    
    const char* left_path = "../output/image_0_bw.png";
    const char* right_path = "../output/image_1_bw.png";
    const char* output_path = "../output/disparity.png";
    
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
    auto load_start = chrono::high_resolution_clock::now();
    Image left = loadImage(left_path);
    Image right = loadImage(right_path);
    auto load_end = chrono::high_resolution_clock::now();
    timing.loadImagesTime = chrono::duration<double>(load_end - load_start).count();
    
    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }

    cout << "Computing disparity map..." << endl;
    double precomputeTime = 0.0;
    auto disparity_start = chrono::high_resolution_clock::now();
    
    Image disparity = computeDisparity(left, right, max_disparity, window_size, precomputeTime);
    
    auto disparity_end = chrono::high_resolution_clock::now();
    timing.precomputeWindowsTime = precomputeTime;
    timing.disparityComputationTime = chrono::duration<double>(disparity_end - disparity_start).count();
    
    cout << "Saving disparity map to " << output_path << endl;
    auto save_start = chrono::high_resolution_clock::now();
    saveImage(output_path, disparity);
    auto save_end = chrono::high_resolution_clock::now();
    timing.saveImageTime = chrono::duration<double>(save_end - save_start).count();
    
    auto end_total = chrono::high_resolution_clock::now();
    timing.totalTime = chrono::duration<double>(end_total - start_total).count();
    
    // Display profiling information
    displayProfilingInfo(timing);
    
    return 0;
}
