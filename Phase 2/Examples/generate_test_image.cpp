#include <iostream>
#include "lodepng.h"

// Generate a test image with a simple pattern
int main() {
    const unsigned int width = 2940;
    const unsigned int height = 2016;
    
    // Allocate memory for the image (RGBA format)
    std::vector<unsigned char> image(width * height * 4);
    
    // Generate a gradient pattern
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned char r = (unsigned char)(x * 255 / width);
            unsigned char g = (unsigned char)(y * 255 / height);
            unsigned char b = (unsigned char)((x + y) * 255 / (width + height));
            
            unsigned int idx = (y * width + x) * 4;
            image[idx + 0] = r;     // R
            image[idx + 1] = g;     // G
            image[idx + 2] = b;     // B
            image[idx + 3] = 255;   // A (fully opaque)
        }
    }
    
    // Save the image
    unsigned error = lodepng::encode("image_0.png", image, width, height);
    if (error) {
        std::cout << "Error " << error << ": " << lodepng_error_text(error) << std::endl;
        return 1;
    }
    
    std::cout << "Test image generated successfully: image_0.png (" << width << "x" << height << ")" << std::endl;
    return 0;
}
