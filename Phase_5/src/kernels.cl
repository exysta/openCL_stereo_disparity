__kernel void resizeGrayscaleBoxFilter(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    const float resizeFactor, // >1.0 for downscale, <1.0 for upscale
    const int windowRadius    // e.g., 0 for 1x1, 1 for 3x3, 2 for 5x5
) {
    // Use LINEAR filtering for smoother samples
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_LINEAR; // or CLK_FILTER_NEAREST

    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);

    const int out_width = get_image_width(output_image);
    const int out_height = get_image_height(output_image);

    // Check if we are inside the output image bounds
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    // Find the corresponding center point in the input image
    const float in_center_x = ( (float)out_x + 0.5f ) * resizeFactor;
    const float in_center_y = ( (float)out_y + 0.5f ) * resizeFactor;

    // Average pixels in a window
    float accumulated_gray = 0.0f;
    // window size
    const int windowDiameter = (2 * windowRadius) + 1;
    const float numPixelsInWindow = (float)(windowDiameter * windowDiameter);

    // Loop over the filter window
    for (int j = -windowRadius; j <= windowRadius; ++j) {
        for (int i = -windowRadius; i <= windowRadius; ++i) {

            // Input coordinate for this sample
            float sample_x = in_center_x + (float)i;
            float sample_y = in_center_y + (float)j;

            // Read pixel
            float4 input_pixel = read_imagef(input_image, sampler, (float2)(sample_x, sample_y));

            // Convert to grayscale (Luminance formula)
            float gray_sample = dot(input_pixel.xyz, (float3)(0.2126f, 0.7152f, 0.0722f));

            // Add to total
            accumulated_gray += gray_sample;
        }
    }

    // Calculate average
    float average_gray_float = accumulated_gray / numPixelsInWindow;

    // Convert to uchar [0, 255]
    uchar final_gray_value_uchar = convert_uchar_sat(average_gray_float * 255.0f);

    // Write result (assumes CL_R output format)
    write_imageui(output_image, (int2)(out_x, out_y), (uint4)(final_gray_value_uchar, 0, 0, 0));
    // If output format was CL_RGBA:
    // write_imageui(output_image, (int2)(out_x, out_y), (uint4)(final_gray_value_uchar, final_gray_value_uchar, final_gray_value_uchar, 255));
}


__kernel void zncc_stereo_match(
    __read_only image2d_t left_image,
    __read_only image2d_t right_image,
    __global uchar* disparity_map,
    const int width,
    const int height,
    const int WIN_SIZE,
    const int MAX_DISP,
    const sampler_t sampler)
{
    const int i = get_global_id(0); // current pixel x
    const int j = get_global_id(1); // current pixel y

    // Check image bounds
    if (i >= width || j >= height) {
        return;
    }

    const int win_radius = WIN_SIZE / 2;
    const int num_pixels_in_window = WIN_SIZE * WIN_SIZE;
    const float inv_num_pixels = 1.0f / (float)num_pixels_in_window; // for calculating mean

    float max_zncc_score = -1.0f; // best score found so far
    int best_disparity = 0;

    // Calculate mean for the left window (only needs to be done once)
    float sum_L = 0.0f;
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
            int2 coord_L = (int2)(i + win_x, j + win_y);
            // read R channel as float
            sum_L += (float)read_imageui(left_image, sampler, coord_L).x;
        }
    }
    float mean_L = sum_L * inv_num_pixels;
    float sum_sq_diff_L = 0.0f; // Sum of (L_i - mean_L)^2

    // Calculate sum of squared differences for left window (needed for denominator)
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
            int2 coord_L = (int2)(i + win_x, j + win_y);
            float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
            float diff_L = val_L - mean_L;
            sum_sq_diff_L += diff_L * diff_L;
        }
    }


    // Try all possible disparities
    for (int d = 0; d < MAX_DISP; ++d) {
        // Check if the right window's center pixel is valid
        if (i < d) {
           continue; // Skip if right pixel coord (i-d) is negative
        }

        float sum_R = 0.0f;

        // Calculate mean for the shifted right window
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_R = (int2)(i - d + win_x, j + win_y);
                sum_R += (float)read_imageui(right_image, sampler, coord_R).x;
            }
        }
        float mean_R = sum_R * inv_num_pixels;

        float numerator = 0.0f;    // Sum of (L_i - mean_L) * (R_i - mean_R)
        // float sum_sq_diff_L = 0.0f; // Calculated above
        float sum_sq_diff_R = 0.0f; // Sum of (R_i - mean_R)^2

        // Calculate numerator and sum of squared differences for right window
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_L = (int2)(i + win_x, j + win_y);
                int2 coord_R = (int2)(i - d + win_x, j + win_y);

                float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
                float val_R = (float)read_imageui(right_image, sampler, coord_R).x;

                float diff_L = val_L - mean_L;
                float diff_R = val_R - mean_R;

                numerator += diff_L * diff_R;
                // sum_sq_diff_L += diff_L * diff_L; // Calculated above
                sum_sq_diff_R += diff_R * diff_R;
            }
        }

        // Calculate ZNCC
        float current_zncc_score = -2.0f; // Invalid score default
        float denominator = sqrt(sum_sq_diff_L * sum_sq_diff_R);
        const float epsilon = 1e-6f; // small value to avoid div by zero

        if (denominator > epsilon) {
            current_zncc_score = numerator / denominator;
            // Ensure score is in valid range [-1, 1]
            current_zncc_score = clamp(current_zncc_score, -1.0f, 1.0f);
        }
        // else: score stays invalid (-2.0)

        // Found a better match?
        if (current_zncc_score > max_zncc_score) {
            max_zncc_score = current_zncc_score;
            best_disparity = d;
        }
    }

    // Output index in the 1D disparity map array
    int output_index = j * width + i;

    uchar output_value;
    if (MAX_DISP <= 1) {
        // Handle edge case where max disparity is 0 or 1
        output_value = 255; // map disparity 0 to brightest value
    } else {
        // Scale disparity [0, MAX_DISP-1] to [0, 255]
        float scaled_disparity = ((float)best_disparity * 255.0f) / ((float)MAX_DISP - 1.0f);

        // Clamp just in case
        scaled_disparity = clamp(scaled_disparity, 0.0f, 255.0f);

        // Invert: higher disparity -> darker pixel
        output_value = 255 - (uchar)scaled_disparity;
    }

    // Write output value
    disparity_map[output_index] = output_value;

}