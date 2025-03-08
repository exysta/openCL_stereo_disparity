#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

struct Image {
    int width, height;
    vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        return data[y * width + x];
    }
    
    void set(int x, int y, unsigned char value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = value;
        }
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

Image median_filter(const Image& input, int kernel_size) {
    kernel_size = max(3, kernel_size);
    Image output{input.width, input.height, vector<unsigned char>(input.width * input.height)};
    int half_kernel = kernel_size / 2;
    
    for(int y = 0; y < input.height; y++) {
        for(int x = 0; x < input.width; x++) {
            vector<unsigned char> window;
            window.reserve(kernel_size * kernel_size);
            
            for(int ky = -half_kernel; ky <= half_kernel; ky++) {
                for(int kx = -half_kernel; kx <= half_kernel; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if(nx >= 0 && nx < input.width && ny >= 0 && ny < input.height) {
                        window.push_back(input.at(nx, ny));
                    }
                }
            }
            
            sort(window.begin(), window.end());
            output.set(x, y, window[window.size() / 2]);
        }
    }
    
    return output;
}

Image enhance_contrast(const Image& input) {
    Image output{input.width, input.height, vector<unsigned char>(input.width * input.height)};
    vector<int> histogram(256, 0);
    for(int i = 0; i < input.width * input.height; i++) {
        histogram[input.data[i]]++;
    }
    

    vector<int> cumulative_hist(256, 0);
    cumulative_hist[0] = histogram[0];
    for(int i = 1; i < 256; i++) {
        cumulative_hist[i] = cumulative_hist[i-1] + histogram[i];
    }
    

    vector<unsigned char> lut(256, 0);
    int total_pixels = input.width * input.height;
    for(int i = 0; i < 256; i++) {
        lut[i] = static_cast<unsigned char>(255.0 * cumulative_hist[i] / total_pixels);
    }
    

    for(int i = 0; i < input.width * input.height; i++) {
        output.data[i] = lut[input.data[i]];
    }
    
    double gamma = 1.2;
    for(int i = 0; i < input.width * input.height; i++) {
        double normalized = output.data[i] / 255.0;
        double corrected = pow(normalized, 1.0/gamma);
        output.data[i] = static_cast<unsigned char>(corrected * 255.0);
    }
    
    return output;
}

Image computeDisparity(const Image& left, const Image& right, int max_disp, int win_size, bool use_optimization = true) {
    Image disparity{left.width, left.height, vector<unsigned char>(left.width * left.height, 0)};
    int win_radius = win_size / 2;
    

    cout << "Using sequential computation" << endl;
    

    for(int y = 0; y < left.height; y++) {

        if(y % 10 == 0) {
            cout << "Processing line: " << y << " / " << left.height 
                 << " (" << (y * 100 / left.height) << "%)\r" << flush;
        }
        
        for(int x = 0; x < left.width; x++) {
            float max_zncc = -INFINITY;
            int best_d = 0;
            

            if (x < win_radius || y < win_radius || 
                x >= left.width - win_radius || y >= left.height - win_radius) {
                continue;
            }

            for(int d = 0; d <= max_disp; d++) {
                if(x - d < win_radius) continue;


                if (use_optimization) {
                    double meanL = 0, meanR = 0;
                    int count = 0;
                    
                    // Calculer les moyennes
                    for(int wy = -win_radius; wy <= win_radius; wy++) {
                        for(int wx = -win_radius; wx <= win_radius; wx++) {
                            int xxL = x + wx;
                            int yyL = y + wy;
                            int xxR = x - d + wx;
                            int yyR = y + wy;
                            
                            meanL += left.at(xxL, yyL);
                            meanR += right.at(xxR, yyR);
                            count++;
                        }
                    }
                    
                    meanL /= count;
                    meanR /= count;
                    
                    // Calculer ZNCC
                    double numerator = 0, denomL = 0, denomR = 0;
                    for(int wy = -win_radius; wy <= win_radius; wy++) {
                        for(int wx = -win_radius; wx <= win_radius; wx++) {
                            int xxL = x + wx;
                            int yyL = y + wy;
                            int xxR = x - d + wx;
                            int yyR = y + wy;
                            
                            double diffL = left.at(xxL, yyL) - meanL;
                            double diffR = right.at(xxR, yyR) - meanR;
                            
                            numerator += diffL * diffR;
                            denomL += diffL * diffL;
                            denomR += diffR * diffR;
                        }
                    }
                    
                    double zncc = (denomL > 0 && denomR > 0) ? 
                                numerator / sqrt(denomL * denomR) : -1;
                    
                    if(zncc > max_zncc) {
                        max_zncc = zncc;
                        best_d = d;
                    }
                } 
                else {

                    double meanL = 0, meanR = 0;
                    int count = 0;
                    
                    // Calculer les moyennes
                    for(int wy = -win_radius; wy <= win_radius; wy++) {
                        for(int wx = -win_radius; wx <= win_radius; wx++) {
                            int xxL = x + wx;
                            int yyL = y + wy;
                            int xxR = x - d + wx;
                            int yyR = y + wy;
                            
                            if(xxL >= 0 && xxL < left.width && yyL >= 0 && yyL < left.height &&
                               xxR >= 0 && xxR < right.width && yyR >= 0 && yyR < right.height) {
                                meanL += left.at(xxL, yyL);
                                meanR += right.at(xxR, yyR);
                                count++;
                            }
                        }
                    }
                    
                    if(count == 0) continue;
                    meanL /= count;
                    meanR /= count;
                    
                    // Calculer ZNCC
                    double numerator = 0, denomL = 0, denomR = 0;
                    for(int wy = -win_radius; wy <= win_radius; wy++) {
                        for(int wx = -win_radius; wx <= win_radius; wx++) {
                            int xxL = x + wx;
                            int yyL = y + wy;
                            int xxR = x - d + wx;
                            int yyR = y + wy;
                            
                            if(xxL >= 0 && xxL < left.width && yyL >= 0 && yyL < left.height &&
                               xxR >= 0 && xxR < right.width && yyR >= 0 && yyR < right.height) {
                                double diffL = left.at(xxL, yyL) - meanL;
                                double diffR = right.at(xxR, yyR) - meanR;
                                
                                numerator += diffL * diffR;
                                denomL += diffL * diffL;
                                denomR += diffR * diffR;
                            }
                        }
                    }
                    
                    double zncc = (denomL != 0 && denomR != 0) ? 
                                numerator / sqrt(denomL * denomR) : 0;
                    
                    if(zncc > max_zncc) {
                        max_zncc = zncc;
                        best_d = d;
                    }
                }
            }
            

            float normalized_disp = (float)best_d / max_disp;
            

            if (normalized_disp > 0.3 && normalized_disp < 0.7) {
                normalized_disp = 0.3 + (normalized_disp - 0.3) * 1.3;
            }
            

            normalized_disp = pow(normalized_disp, 0.75);
            
            disparity.set(x, y, static_cast<unsigned char>(normalized_disp * 255));
        }
    }
    
    cout << "\nProcessing complete: 100%" << endl;
    return disparity;
}

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -left <file>     : Left image file (default: image_0_bw.png)" << endl;
    cout << "  -right <file>    : Right image file (default: image_1_bw.png)" << endl;
    cout << "  -out <file>      : Output disparity file (default: disparity.png)" << endl;
    cout << "  -disp <value>    : Maximum disparity value (default: 30)" << endl;
    cout << "  -win <value>     : Window size (default: 7)" << endl;
    cout << "  -filter          : Apply median filter to output (default: off)" << endl;
    cout << "  -enhance         : Enhance contrast of output (default: off)" << endl;
    cout << "  -no-opt          : Disable optimizations (default: optimizations on)" << endl;
    cout << "  -help            : Display this help message" << endl;
}

int main(int argc, char* argv[]) {
    string left_file = "image_0_bw.png";
    string right_file = "image_1_bw.png";
    string output_file = "disparity.png";
    int max_disparity = 80;
    int window_size = 11;
    bool apply_filter = true;
    bool enhance = true;
    bool use_optimization = true;
    

    for(int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if(arg == "-left" && i + 1 < argc) {
            left_file = argv[++i];
        } else if(arg == "-right" && i + 1 < argc) {
            right_file = argv[++i];
        } else if(arg == "-out" && i + 1 < argc) {
            output_file = argv[++i];
        } else if(arg == "-disp" && i + 1 < argc) {
            max_disparity = stoi(argv[++i]);
        } else if(arg == "-win" && i + 1 < argc) {
            window_size = stoi(argv[++i]);

            if(window_size % 2 == 0) {
                window_size++;
                cout << "Window size adjusted to " << window_size << " (must be odd)" << endl;
            }
        } else if(arg == "-filter") {
            apply_filter = true;
        } else if(arg == "-enhance") {
            enhance = true;
        } else if(arg == "-no-opt") {
            use_optimization = false;
        } else if(arg == "-help") {
            print_usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown option: " << arg << endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    

    cout << "Parameters:" << endl;
    cout << "  Left image: " << left_file << endl;
    cout << "  Right image: " << right_file << endl;
    cout << "  Output file: " << output_file << endl;
    cout << "  Max disparity: " << max_disparity << endl;
    cout << "  Window size: " << window_size << endl;
    cout << "  Apply filter: " << (apply_filter ? "yes" : "no") << endl;
    cout << "  Enhance contrast: " << (enhance ? "yes" : "no") << endl;
    cout << "  Use optimization: " << (use_optimization ? "yes" : "no") << endl;
    

    cout << "Loading images..." << endl;
    auto start_time = high_resolution_clock::now();
    
    Image left = load_image(left_file.c_str());
    Image right = load_image(right_file.c_str());
    
    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }
    
    cout << "Image dimensions: " << left.width << "x" << left.height << endl;
    

    cout << "Computing disparity map..." << endl;
    auto compute_start = high_resolution_clock::now();
    
    Image disparity = computeDisparity(left, right, max_disparity, window_size, use_optimization);
    
    auto compute_end = high_resolution_clock::now();
    auto compute_duration = duration_cast<milliseconds>(compute_end - compute_start);
    

    if(apply_filter) {
        cout << "Applying advanced filtering..." << endl;
        

        Image original_disparity = disparity;
        

        disparity = median_filter(disparity, 3);
        

        cout << "Applying bilateral filter..." << endl;
        Image temp_disparity = disparity;
        

        const int spatial_radius = 5;
        const double spatial_sigma = 3.0;
        const double range_sigma = 30.0;
        
        for(int y = spatial_radius; y < disparity.height - spatial_radius; y++) {
            for(int x = spatial_radius; x < disparity.width - spatial_radius; x++) {
                double sum_weights = 0.0;
                double sum_values = 0.0;
                int center_value = disparity.at(x, y);
                
                for(int wy = -spatial_radius; wy <= spatial_radius; wy++) {
                    for(int wx = -spatial_radius; wx <= spatial_radius; wx++) {
                        int nx = x + wx;
                        int ny = y + wy;
                        int neighbor_value = disparity.at(nx, ny);
                        

                        double spatial_dist = sqrt(wx*wx + wy*wy);
                        double spatial_weight = exp(-(spatial_dist*spatial_dist) / (2*spatial_sigma*spatial_sigma));
                        

                        double range_dist = abs(center_value - neighbor_value);
                        double range_weight = exp(-(range_dist*range_dist) / (2*range_sigma*range_sigma));
                        

                        double weight = spatial_weight * range_weight;
                        
                        sum_weights += weight;
                        sum_values += weight * neighbor_value;
                    }
                }
                

                if(sum_weights > 0) {
                    temp_disparity.set(x, y, static_cast<unsigned char>(sum_values / sum_weights));
                }
            }
        }
        disparity = temp_disparity;
        

        cout << "Fixing boundary artifacts..." << endl;
        for(int y = 0; y < disparity.height; y++) {
            for(int x = 0; x < disparity.width; x++) {

                if(x > 5 && x < disparity.width - 5 && y > 5 && y < disparity.height - 5) {
                    int center = disparity.at(x, y);
                    int count_similar = 0;
                    int sum_similar = 0;
                    

                    for(int wy = -1; wy <= 1; wy++) {
                        for(int wx = -1; wx <= 1; wx++) {
                            if(wx == 0 && wy == 0) continue;
                            
                            int neighbor = disparity.at(x+wx, y+wy);
                            if(abs(center - neighbor) < 40) {
                                count_similar++;
                                sum_similar += neighbor;
                            }
                        }
                    }
                    

                    if(count_similar < 3) {

                        disparity.set(x, y, original_disparity.at(x, y));
                    } else if(count_similar < 5) {

                        disparity.set(x, y, sum_similar / count_similar);
                    }
                }
            }
        }
    }
    
    if(enhance) {
        cout << "Enhancing contrast..." << endl;
        disparity = enhance_contrast(disparity);
        

        cout << "Applying final smoothing..." << endl;
        Image temp = disparity;
        for(int y = 2; y < disparity.height - 2; y++) {
            for(int x = 2; x < disparity.width - 2; x++) {

                int sum = 0;
                int count = 0;
                
                for(int wy = -2; wy <= 2; wy++) {
                    for(int wx = -2; wx <= 2; wx++) {

                        int weight = 5 - (abs(wx) + abs(wy));
                        if(weight <= 0) weight = 1;
                        
                        sum += disparity.at(x+wx, y+wy) * weight;
                        count += weight;
                    }
                }
                
                temp.set(x, y, sum / count);
            }
        }
        disparity = temp;
    }
    

    cout << "Saving disparity map to " << output_file << "..." << endl;
    save_disparity(output_file.c_str(), disparity);
    
    auto end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_time - start_time);
    

    cout << "Done!" << endl;
    cout << "Computation time: " << compute_duration.count() / 1000.0 << " seconds" << endl;
    cout << "Total processing time: " << total_duration.count() / 1000.0 << " seconds" << endl;
    
    return 0;
}
