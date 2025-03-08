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
    image.height = height;
    image.width = width;
    
    //if there's an error, display it
    if (error) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return Image();  // return an empty Image on error
    }
    return image;
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
// Function to compute disparity map using ZNCC (Zero-Normalized Cross-Correlation)
std::vector<unsigned char> computeDisparityMap(
    const std::vector<unsigned char>& leftImage,
    const std::vector<unsigned char>& rightImage,
    int width, int height,
    int maxDisparity,
    int windowSize)
{
    // Initialize disparity image with zeros
    std::vector<unsigned char> disparityImage(width * height, 0);
    
    // Half window size for calculations
    int halfWindowSize = windowSize / 2;
    
    // For each pixel in the image
    for (int j = halfWindowSize; j < height - halfWindowSize; j++) {
        for (int i = halfWindowSize; i < width - halfWindowSize; i++) {
            double bestZNCC = -1.0; // Initialize with lowest possible ZNCC value
            int bestDisparity = 0;
            
            // Try each possible disparity
            for (int d = 0; d <= maxDisparity; d++) {
                // Skip if we would go out of bounds
                if (i - d < halfWindowSize) continue;
                
                // Calculate mean values for each window
                double leftMean = 0.0;
                double rightMean = 0.0;
                
                // Sum values in windows to calculate means
                for (int winY = -halfWindowSize; winY <= halfWindowSize; winY++) {
                    for (int winX = -halfWindowSize; winX <= halfWindowSize; winX++) {
                        // Calculate flat indices for 2D coordinates
                        int leftIdx = (j + winY) * width + (i + winX);
                        int rightIdx = (j + winY) * width + (i - d + winX);
                        
                        leftMean += static_cast<double>(leftImage[leftIdx]);
                        rightMean += static_cast<double>(rightImage[rightIdx]);
                    }
                }
                
                // Divide by window area to get actual means
                int windowArea = windowSize * windowSize;
                leftMean /= windowArea;
                rightMean /= windowArea;
                
                // Calculate ZNCC numerator and denominator parts
                double numerator = 0.0;
                double leftDenominator = 0.0;
                double rightDenominator = 0.0;
                
                // Calculate ZNCC components for each pixel in window
                for (int winY = -halfWindowSize; winY <= halfWindowSize; winY++) {
                    for (int winX = -halfWindowSize; winX <= halfWindowSize; winX++) {
                        // Calculate flat indices for 2D coordinates
                        int leftIdx = (j + winY) * width + (i + winX);
                        int rightIdx = (j + winY) * width + (i - d + winX);
                        
                        // Get normalized pixel values
                        double leftNorm = static_cast<double>(leftImage[leftIdx]) - leftMean;
                        double rightNorm = static_cast<double>(rightImage[rightIdx]) - rightMean;
                        
                        // Calculate components for ZNCC formula
                        numerator += leftNorm * rightNorm;
                        leftDenominator += leftNorm * leftNorm;
                        rightDenominator += rightNorm * rightNorm;
                    }
                }
                
                // Calculate final ZNCC value, avoiding division by zero
                double zncc = 0.0;
                if (leftDenominator > 0 && rightDenominator > 0) {
                    zncc = numerator / (std::sqrt(leftDenominator) * std::sqrt(rightDenominator));
                }
                
                // Update best disparity if current ZNCC is better
                if (zncc > bestZNCC) {
                    bestZNCC = zncc;
                    bestDisparity = d;
                }
            }
            
            // Assign best disparity value to the disparity image
            disparityImage[j * width + i] = static_cast<unsigned char>(bestDisparity);
        }
    }
    
    return disparityImage;
}

int main() {
    Image left = decodeOneStep("../ressources/disp1.png");
    Image right = decodeOneStep("../ressources/disp5.png");
    
    int MAX_DISP = 260;    // Maximum disparity value, based on im0 and im1, depends on each images
    int WIN_SIZE = 9;     // Window size (should be odd)

    if(left.width != right.width || left.height != right.height) {
        cerr << "Error: Images must have the same dimensions!" << endl;
        return 1;
    }

    std::vector<unsigned char> disparityMap = computeDisparityMap(left.data , right.data,left.width,left.height, MAX_DISP, WIN_SIZE);

    // Save results
    encodeOneStep("../output/disparityV2.png", disparityMap, left.width, left.height);

    cout << "Disparity map saved as disparityV2.png" << endl;
    return 0;
}