#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lodepng.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

struct Image {
    unsigned width, height;
    vector<unsigned char> data;
    
    unsigned char at(int x, int y) const {
        return data[y * width + x];
    }
};

//Example 1
//Decode from disk to raw pixels with a single function call
Image decodeOneStep(const char* filename) {
    vector<unsigned char> image_data; //the raw pixels
    Image image;
    unsigned width, height;
  
    //decode
    unsigned error = lodepng::decode(image_data, width, height, filename);
    image.data = image_data;
    image.height,image.width = height,width;

    //if there's an error, display it
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    else return image;
    //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
  }

  //Example 1
//Encode from raw pixels to disk with a single function call
//The image argument has width * height RGBA pixels or width * height * 4 bytes
void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height) {
    //Encode the image
    unsigned error = lodepng::encode(filename, image, width, height);
  
    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  }

int main() {
    Image left = decodeOneStep("view1.png");
    Image right = decodeOneStep("view5.png");
    
    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }

    int max_disparity = 64;
    int window_size = 5;
    
    Image disparity = computeDisparity(left, right, max_disparity, window_size);
    save_disparity("disparity.png", disparity);
    
    cout << "Disparity map saved as disparity.png" << endl;
    return 0;
}