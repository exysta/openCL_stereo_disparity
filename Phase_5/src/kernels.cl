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
            float gray_sample = dot(input_pixel.xyz, (float3)(0.2126f, 0.7152f, 0.0722f));

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

__kernel void zncc_stereo_match(
    __read_only image2d_t left_image,       // Format: CL_R, CL_UNSIGNED_INT8
    __read_only image2d_t right_image,      // Format: CL_R, CL_UNSIGNED_INT8
    __global uchar* disparity_map,
    const int width,
    const int height,
    const int WIN_SIZE,
    const int MAX_DISP,
    const sampler_t sampler)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    // Boundary check: Ensure the work-item is within the image bounds
    if (i >= width || j >= height) {
        return;
    }

    const int win_radius = WIN_SIZE / 2;
    const int num_pixels_in_window = WIN_SIZE * WIN_SIZE;
    // Use float for division to avoid potential integer truncation issues earlier
    const float inv_num_pixels = 1.0f / (float)num_pixels_in_window;

    float max_zncc_score = -1.0f; // Initialize with lowest possible valid ZNCC score
    int best_disparity = 0;

    // Pre-calculate the mean for the left window once
    float sum_L = 0.0f;
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
            // Use int2 for coordinates passed to read_image
            int2 coord_L = (int2)(i + win_x, j + win_y);
            // Read unsigned int, take the first component (R channel), cast to float
            sum_L += (float)read_imageui(left_image, sampler, coord_L).x;
        }
    }
    float mean_L = sum_L * inv_num_pixels;
    float sum_sq_diff_L = 0.0f; // Calculate sum_sq_diff_L here as well

    // Calculate sum of squared differences for the left window
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
            int2 coord_L = (int2)(i + win_x, j + win_y);
            float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
            float diff_L = val_L - mean_L;
            sum_sq_diff_L += diff_L * diff_L;
        }
    }


    // Iterate through possible disparities
    for (int d = 0; d < MAX_DISP; ++d) {
        // Ensure the right window's center pixel is within the image bounds
        // The window itself might still go out of bounds, handled by the sampler
        if (i < d) {
           continue; // Skip disparities that would require pixels outside the left image edge
        }

        float sum_R = 0.0f;

        // Calculate mean for the right window (shifted by disparity d)
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_R = (int2)(i - d + win_x, j + win_y);
                sum_R += (float)read_imageui(right_image, sampler, coord_R).x;
            }
        }
        float mean_R = sum_R * inv_num_pixels;

        float numerator = 0.0f;
        // float sum_sq_diff_L = 0.0f; // Moved calculation outside the loop
        float sum_sq_diff_R = 0.0f;

        // Calculate numerator and sum of squared differences for the right window
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_L = (int2)(i + win_x, j + win_y);
                int2 coord_R = (int2)(i - d + win_x, j + win_y);

                // Read pixel values
                float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
                float val_R = (float)read_imageui(right_image, sampler, coord_R).x;

                // Calculate differences from mean
                float diff_L = val_L - mean_L;
                float diff_R = val_R - mean_R;

                // Accumulate for ZNCC formula
                numerator += diff_L * diff_R;
                // sum_sq_diff_L += diff_L * diff_L; // Moved calculation outside the loop
                sum_sq_diff_R += diff_R * diff_R;
            }
        }

        // Calculate ZNCC score for the current disparity
        float current_zncc_score = -2.0f; // Default score for invalid/flat areas
        float denominator = sqrt(sum_sq_diff_L * sum_sq_diff_R);
        // Use a small epsilon to avoid division by zero or near-zero (instability)
        const float epsilon = 1e-6f; // Adjust if needed, but 1e-6 is common

        if (denominator > epsilon) {
            current_zncc_score = numerator / denominator;
            // Clamp score to valid range [-1, 1] just in case of float inaccuracies
            current_zncc_score = clamp(current_zncc_score, -1.0f, 1.0f);
        }
        // else: score remains -2.0, won't be chosen unless all scores are -2.0

        // Update best disparity if current score is higher
        if (current_zncc_score > max_zncc_score) {
            max_zncc_score = current_zncc_score;
            best_disparity = d;
        }
    }

    // Calculate output index
    int output_index = j * width + i;

    // --- FIX: Scale disparity to [0, 255] range before inversion ---
    uchar output_value;
    if (MAX_DISP <= 1) {
        // If MAX_DISP is 0 or 1, disparity can only be 0.
        // Map disparity 0 to 255 (brightest) based on the inversion scheme.
        output_value = 255;
    } else {
        // Scale disparity from [0, MAX_DISP-1] to [0, 255]
        // Use float division for accuracy.
        float scaled_disparity = ((float)best_disparity * 255.0f) / ((float)MAX_DISP - 1.0f);

        // Clamp scaled value to [0, 255] before casting (safety measure)
        scaled_disparity = clamp(scaled_disparity, 0.0f, 255.0f);

        // Invert: Higher disparity means darker pixel (lower uchar value)
        // This matches the visual appearance of the desired output image.
        output_value = 255 - (uchar)scaled_disparity;
    }
    // --- END FIX ---

    // Write the final scaled and inverted disparity value
    disparity_map[output_index] = output_value;


    /* --- OLD VERSION (commented out) ---
    // This version causes the bright output if MAX_DISP << 255
    int output_index = j * width + i;
    disparity_map[output_index] = 255 - (uchar)best_disparity;
    */
}