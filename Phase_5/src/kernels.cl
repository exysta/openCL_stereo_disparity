__kernel void resizeGrayscaleBoxFilter(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    const float resizeFactor, // >1.0 for downscale, <1.0 for upscale
    const int windowRadius    // e.g., 0 for 1x1, 1 for 3x3, 2 for 5x5 window
) {
    // Sampler - LINEAR often still gives better results even with box filter,
    // as it smooths the individual samples before averaging.
    // Use CLK_FILTER_NEAREST if you want a strict nearest-neighbor box filter.
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_LINEAR; // or CLK_FILTER_NEAREST

    // Get the coordinates of the pixel this work-item is responsible for
    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);

    // Get output dimensions
    const int out_width = get_image_width(output_image);
    const int out_height = get_image_height(output_image);

    // Boundary check for output image
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    // --- Coordinate Mapping ---
    // Calculate the corresponding *center* floating-point coordinate in the input image.
    const float in_center_x = ( (float)out_x + 0.5f ) * resizeFactor;
    const float in_center_y = ( (float)out_y + 0.5f ) * resizeFactor;

    // --- Box Filter Averaging ---
    float accumulated_gray = 0.0f;
    // Calculate window dimensions based on radius
    const int windowDiameter = (2 * windowRadius) + 1;
    const float numPixelsInWindow = (float)(windowDiameter * windowDiameter);

    // Loop through the window centered around the mapped input coordinate
    for (int j = -windowRadius; j <= windowRadius; ++j) {
        for (int i = -windowRadius; i <= windowRadius; ++i) {

            // Calculate the specific coordinate in the input image to sample for this window pixel
            float sample_x = in_center_x + (float)i;
            float sample_y = in_center_y + (float)j;

            // Read the RGBA pixel value [0.0, 1.0] using the sampler
            float4 input_pixel = read_imagef(input_image, sampler, (float2)(sample_x, sample_y));

            // Convert the sampled pixel to grayscale (Luminance)
            float gray_sample = dot(input_pixel.xyz, (float3)(0.299f, 0.587f, 0.114f));

            // Accumulate the grayscale value
            accumulated_gray += gray_sample;
        }
    }

    // Calculate the average grayscale value for the window
    float average_gray_float = accumulated_gray / numPixelsInWindow;


    // --- Convert and Write Output ---
    // Scale to [0, 255] and convert to uchar with saturation
    uchar final_gray_value_uchar = convert_uchar_sat(average_gray_float * 255.0f);

    // Write the final averaged grayscale result to the output image (assuming CL_R format)
    write_imageui(output_image, (int2)(out_x, out_y), (uint4)(final_gray_value_uchar, 0, 0, 0));
    // If output format was CL_RGBA:
    // write_imageui(output_image, (int2)(out_x, out_y), (uint4)(final_gray_value_uchar, final_gray_value_uchar, final_gray_value_uchar, 255));
}

// Kernel to precompute mean and standard deviation for windows
__kernel void precompute_window_stats(
    __global const float* input_image, // Input grayscale image data (float)
    __global float* output_means,      // Output buffer for means
    __global float* output_stddevs,    // Output buffer for standard deviations
    int width,
    int height,
    int win_hsize                    // Half-size of the window (win_size from C++)
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) {
        return; // Out of bounds
    }

    float sum = 0.0f;
    float sumSq = 0.0f;
    int count = 0;

    // Calculate window boundaries, clamping to image edges
    int win_start_y = max(0, y - win_hsize);
    int win_end_y   = min(height - 1, y + win_hsize);
    int win_start_x = max(0, x - win_hsize);
    int win_end_x   = min(width - 1, x + win_hsize);

    // Iterate over the window
    for (int wy = win_start_y; wy <= win_end_y; wy++) {
        for (int wx = win_start_x; wx <= win_end_x; wx++) {
            float val = input_image[wy * width + wx];
            sum += val;
            sumSq += val * val;
            count++;
        }
    }

    // Calculate mean and standard deviation
    // Add epsilon to count to prevent division by zero if window somehow has 0 pixels (shouldn't happen with proper bounds)
    float mean = 0.0f;
    float stdDev = 0.0f;
    float fCount = (float)count; // Use float for division

    if (fCount > 0.0f) {
        mean = sum / fCount;
        // Prevent negative variance due to floating point errors
        float variance = max(0.0f, (sumSq / fCount) - (mean * mean));
        stdDev = sqrt(variance);
    }

    // Write results to output buffers
    int index = y * width + x;
    output_means[index] = mean;
    output_stddevs[index] = stdDev;
}

// Kernel to compute disparity using ZNCC
__kernel void compute_disparity_zncc(
    __global const float* left_image,    // Left grayscale image data (float)
    __global const float* right_image,   // Right grayscale image data (float)
    __global const float* left_means,    // Precomputed means for left image
    __global const float* left_stddevs,  // Precomputed stddevs for left image
    __global const float* right_means,   // Precomputed means for right image
    __global const float* right_stddevs, // Precomputed stddevs for right image
    __global uchar* output_disparity,   // Output disparity map (uchar 0-255)
    int width,
    int height,
    int win_hsize,                     // Half-size of the window
    int max_disp                       // Maximum disparity to check
) {
    int x = get_global_id(0); // x coordinate in the left image
    int y = get_global_id(1); // y coordinate in the left image

    if (x >= width || y >= height) {
        return; // Out of bounds
    }

    int indexL = y * width + x;
    float meanL = left_means[indexL];
    float stdDevL = left_stddevs[indexL];

    // Skip calculation if the left window standard deviation is too small (textureless)
    // Use a small epsilon to avoid floating point issues
    float stdDevThreshold = 1.0f;
    if (stdDevL < stdDevThreshold) {
        output_disparity[indexL] = 0; // Assign zero disparity for textureless areas
        return;
    }

    float max_zncc_val = -1.0f; // ZNCC is in [-1, 1]
    int best_d = 0;

    // Iterate through possible disparities
    // d = 0 means comparing (x, y) in left with (x, y) in right
    // d > 0 means comparing (x, y) in left with (x-d, y) in right
    for (int d = 0; d <= max_disp; d++) {
        int xR = x - d; // Corresponding x coordinate in the right image

        // Check if the right coordinate is within bounds
        if (xR < 0) {
            break; // No need to check further disparities for this pixel
        }

        int indexR = y * width + xR;
        float meanR = right_means[indexR];
        float stdDevR = right_stddevs[indexR];

        // Skip if the right window is textureless
        if (stdDevR < stdDevThreshold) {
            continue;
        }

        // Calculate ZNCC numerator
        float numerator = 0.0f;
        int validPoints = 0;

        // Define window boundaries for the *current* pixel (x,y) in left
        // and (xR, y) in right. Clamping ensures we stay within image bounds.
        int win_start_y = max(0, y - win_hsize);
        int win_end_y   = min(height - 1, y + win_hsize);
        int win_start_xL = max(0, x - win_hsize);    // Window x-start in Left
        int win_end_xL   = min(width - 1, x + win_hsize); // Window x-end in Left

        // Iterate through the window centered at (x,y) in the left image
        for (int wy = win_start_y; wy <= win_end_y; wy++) {
            for (int wxL = win_start_xL; wxL <= win_end_xL; wxL++) {
                int wxR = wxL - d; // Corresponding window x in the Right image

                // Ensure the corresponding pixel in the right image's *window* is valid
                // The center pixel xR is already checked, but window edges might go out
                if (wxR >= 0 && wxR < width) {
                     // Ensure this point wxR is also within the conceptual window boundaries of the right pixel centered at xR.
                     // This check is implicitly handled by iterating over the *left* window and calculating the corresponding right coord wxR,
                     // as long as we ensure wxR stays within the image (0 to width-1).
                    float valL = left_image[wy * width + wxL];
                    float valR = right_image[wy * width + wxR];

                    numerator += (valL - meanL) * (valR - meanR);
                    validPoints++;
                }
            }
        }

        // Calculate ZNCC value if denominator is valid
        if (validPoints > 0) {
            // Add a small epsilon to prevent division by zero/very small numbers
            float denominator = stdDevL * stdDevR * (float)validPoints;
            if (denominator > 1e-6f) { // Avoid division by zero or near-zero
                 float zncc = numerator / denominator;

                 // Clamp ZNCC to [-1, 1] range just in case of float inaccuracies
                 zncc = clamp(zncc, -1.0f, 1.0f);

                if (zncc > max_zncc_val) {
                    max_zncc_val = zncc;
                    best_d = d;
                }
            }
        }
    } // End disparity loop (d)

    // Normalize disparity value to [0, 255] and write to output
    // Use convert_uchar_sat for safe conversion and clamping to 0-255 range
    uchar normalized_d = 0;
    if (max_disp > 0) { // Avoid division by zero if max_disp is 0
       normalized_d = convert_uchar_sat(((float)best_d * 255.0f) / (float)max_disp);
    }
    output_disparity[indexL] = normalized_d;
}