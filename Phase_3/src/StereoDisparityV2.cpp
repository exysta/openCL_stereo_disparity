#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

struct Image {
    int width, height;
    vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

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

Image computeDisparity(const Image& left, const Image& right, int max_disp, int win_size) {
    int width = left.width;
    int height = left.height;
    Image disparity{width, height, vector<unsigned char>(width * height, 0)};
    
    // Pre-compute window means and standard deviations
    vector<double> leftMeans, leftStdDevs, rightMeans, rightStdDevs;
    precomputeWindowValues(left, leftMeans, leftStdDevs, win_size);
    precomputeWindowValues(right, rightMeans, rightStdDevs, win_size);
    
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
    const char* left_path = "../ressources/image_0_bw.png";
    const char* right_path = "../ressources/image_1_bw.png";
    const char* output_path = "../output/disparityV2.png";
    
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
    auto start_time = chrono::high_resolution_clock::now();
    
    Image disparity = computeDisparity(left, right, max_disparity, window_size);
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "Computation time: " << elapsed.count() << " seconds" << endl;
    
    cout << "Saving disparity map to " << output_path << endl;
    save_disparity(output_path, disparity);
    
    return 0;
}