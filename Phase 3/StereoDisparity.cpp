#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

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

Image computeDisparity(const Image& left, const Image& right, int max_disp, int win_size) {
    Image disparity{left.width, left.height, vector<unsigned char>(left.width * left.height)};
    
    for(int y = 0; y < left.height; y++) {
        // Afficher la progression toutes les 10 lignes
        if(y % 10 == 0) {
            cout << "Processing line: " << y << " / " << left.height << " (" << (y * 100 / left.height) << "%)\r" << flush;
        }
        for(int x = 0; x < left.width; x++) {
            float max_zncc = -INFINITY;
            int best_d = 0;

            for(int d = 0; d <= max_disp; d++) {
                if(x - d < 0) continue;

                double meanL = 0, meanR = 0;
                int count = 0;

                // Calculate means
                for(int wy = -win_size; wy <= win_size; wy++) {
                    for(int wx = -win_size; wx <= win_size; wx++) {
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

                // Calculate ZNCC
                double numerator = 0, denomL = 0, denomR = 0;
                for(int wy = -win_size; wy <= win_size; wy++) {
                    for(int wx = -win_size; wx <= win_size; wx++) {
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

            disparity.data[y * left.width + x] = static_cast<unsigned char>((best_d * 255) / max_disp);
        }
    }
    
    return disparity;
}

int main() {
    Image left = load_image("image_0_bw.png");
    Image right = load_image("image_1_bw.png");
    
    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }

    int max_disparity = 65;
    int window_size = 9;
    
    Image disparity = computeDisparity(left, right, max_disparity, window_size);
    save_disparity("disparity.png", disparity);
    
    cout << "Disparity map saved as disparity.png" << endl;
    return 0;
}