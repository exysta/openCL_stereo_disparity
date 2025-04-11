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

    if (i >= width || j >= height) {
        return;
    }

    const int win_radius = WIN_SIZE / 2;
    const int num_pixels_in_window = WIN_SIZE * WIN_SIZE;
    const float inv_num_pixels = 1.0f / (float)num_pixels_in_window;

    float max_zncc_score = -1.0f;
    int best_disparity = 0;

    for (int d = 0; d < MAX_DISP; ++d) {
        if (i < d) {
           continue;
        }

        float sum_L = 0.0f;
        float sum_R = 0.0f;

        // *** CHANGE 1: Read using read_imageui and convert ***
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_L = (int2)(i + win_x, j + win_y);
                int2 coord_R = (int2)(i - d + win_x, j + win_y);

                // Read as unsigned int (uint4 because image reads return 4 components)
                uint4 pixel_L_ui = read_imageui(left_image, sampler, coord_L);
                uint4 pixel_R_ui = read_imageui(right_image, sampler, coord_R);

                // Convert the relevant component (x) to float
                float val_L = (float)pixel_L_ui.x;
                float val_R = (float)pixel_R_ui.x;

                sum_L += val_L;
                sum_R += val_R;
            }
        }
        float mean_L = sum_L * inv_num_pixels;
        float mean_R = sum_R * inv_num_pixels;

        float numerator = 0.0f;
        float sum_sq_diff_L = 0.0f;
        float sum_sq_diff_R = 0.0f;

        // *** CHANGE 2: Read using read_imageui and convert ***
        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
            for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                int2 coord_L = (int2)(i + win_x, j + win_y);
                int2 coord_R = (int2)(i - d + win_x, j + win_y);

                uint4 pixel_L_ui = read_imageui(left_image, sampler, coord_L);
                uint4 pixel_R_ui = read_imageui(right_image, sampler, coord_R);

                float val_L = (float)pixel_L_ui.x;
                float val_R = (float)pixel_R_ui.x;

                float diff_L = val_L - mean_L;
                float diff_R = val_R - mean_R;

                numerator += diff_L * diff_R;
                sum_sq_diff_L += diff_L * diff_L;
                sum_sq_diff_R += diff_R * diff_R;
            }
        }

        float current_zncc_score = 0.0f;
        float denominator = sqrt(sum_sq_diff_L * sum_sq_diff_R);
        const float epsilon = 1e-6f; // Maybe increase epsilon slightly if needed

        if (denominator > epsilon) {
            current_zncc_score = numerator / denominator;
        }

        if (current_zncc_score > max_zncc_score) {
            max_zncc_score = current_zncc_score;
            best_disparity = d;
        }
    }

    int output_index = j * width + i;
    disparity_map[output_index] = 255 - (uchar)best_disparity;
}