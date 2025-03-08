#include <stdio.h>
#include <stdlib.h>
#define LODEPNG_COMPILE_CPP
#include "lodepng.h"

// Generate a test image with a simple pattern
int main() {
    const unsigned int width = 2940;
    const unsigned int height = 2016;
    
    // Allocate memory for the image (RGBA format)
    unsigned char* image = (unsigned char*)malloc(width * height * 4);
    if (!image) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
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
    unsigned error = lodepng_encode32_file("image_0.png", image, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        free(image);
        return 1;
    }
    
    printf("Test image generated successfully: image_0.png (%ux%u)\n", width, height);
    free(image);
    return 0;
}
