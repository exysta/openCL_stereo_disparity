// Stereo disparity calculation kernel
__kernel void disparity(
    __global const uchar* left_img,    // left image
    __global const uchar* right_img,    // right image
    __global uchar* disparity_map,       // output disparity map
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Matching parameters
    const int window_size = 11;      // window size
    const int half_window = window_size / 2;
    const int max_disp = 50;     // maximum disparity
    const float min_zncc = 0.5f;  // minimum ZNCC threshold
    
    // Check if we have enough space for the window
    if (x < half_window || y < half_window || x >= width - half_window || y >= height - half_window) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    // Calculate mean and standard deviation for the left window
    float sum_left = 0.0f;
    float sum_sq_left = 0.0f;
    int count = 0;
    
    // Calculate mean for the left window
    for (int wy = -half_window; wy <= half_window; wy++) {
        for (int wx = -half_window; wx <= half_window; wx++) {
            int ny = y + wy;
            int nx = x + wx;
            
            // Check if we are within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int left_index = ny * width + nx;
                float val = (float)left_img[left_index];
                sum_left += val;
                sum_sq_left += val * val;
                count++;
            }
        }
    }
    
    float mean_left = sum_left / count;
    float var_left = (sum_sq_left / count) - (mean_left * mean_left);
    float std_dev_left = sqrt(var_left > 0.0001f ? var_left : 0.0001f);  // Avoid division by zero
    
    // If standard deviation is too low (uniform area), skip calculation
    if (std_dev_left < 1.0f) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    float best_zncc = -1.0f;  // ZNCC is in [-1,1], start at -1
    int best_disp = 0;
    
    // Iterate over possible disparities
    for (int d = 0; d <= min(max_disp, x); d++) {
        int xR = x - d;
        
        // Check if we are within bounds
        if (xR < half_window) continue;
        
        // Calculate mean and standard deviation for the right window
        float sum_right = 0.0f;
        float sum_sq_right = 0.0f;
        count = 0;
        
        // Calculate mean for the right window
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx = xR + wx;
                
                // Check if we are within bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int right_index = ny * width + nx;
                    float val = (float)right_img[right_index];
                    sum_right += val;
                    sum_sq_right += val * val;
                    count++;
                }
            }
        }
        
        float mean_right = sum_right / count;
        float var_right = (sum_sq_right / count) - (mean_right * mean_right);
        float std_dev_right = sqrt(var_right > 0.0001f ? var_right : 0.0001f);  // Avoid division by zero
        
        // If standard deviation is too low (uniform area), skip this disparity
        if (std_dev_right < 1.0f) continue;
        
        // Calculate ZNCC
        float numerator = 0.0f;
        count = 0;
        
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx_left = x + wx;
                int nx_right = xR + wx;
                
                // Check if we are within bounds
                if (nx_left >= 0 && nx_left < width && nx_right >= 0 && nx_right < width && ny >= 0 && ny < height) {
                    int left_index = ny * width + nx_left;
                    int right_index = ny * width + nx_right;
                    float diff_left = (float)left_img[left_index] - mean_left;
                    float diff_right = (float)right_img[right_index] - mean_right;
                    numerator += diff_left * diff_right;
                    count++;
                }
            }
        }
        
        float zncc = numerator / (count * std_dev_left * std_dev_right);
        
        // Keep the disparity with the best correlation
        if (zncc > best_zncc) {
            best_zncc = zncc;
            best_disp = d;
        }
    }
    
    // If the best correlation is too low, consider no match
    if (best_zncc < min_zncc) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    // Normalize disparity to [0,255]
    disparity_map[y * width + x] = (uchar)((best_disp * 255) / max_disp);
}

// Disparity calculation from right to left (for cross-checking)
__kernel void disparity_right_to_left(
    __global const uchar* left_img,    // left image (will be target)
    __global const uchar* right_img,   // right image (will be reference)
    __global uchar* disparity_map,    // output disparity map (right-to-left)
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Matching parameters
    const int window_size = 11;     // window size
    const int half_window = window_size / 2;
    const int max_disp = 50;       // maximum disparity
    const float min_zncc = 0.5f;   // minimum ZNCC threshold
    
    // Check if we have enough space for the window
    if (x < half_window || y < half_window || x >= width - half_window || y >= height - half_window) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    // Calculate mean and standard deviation for the right window (reference)
    float sum_right = 0.0f;
    float sum_sq_right = 0.0f;
    int count = 0;
    
    // Calculate mean for the right window
    for (int wy = -half_window; wy <= half_window; wy++) {
        for (int wx = -half_window; wx <= half_window; wx++) {
            int ny = y + wy;
            int nx = x + wx;
            
            // Check if we are within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int right_index = ny * width + nx;
                float val = (float)right_img[right_index];
                sum_right += val;
                sum_sq_right += val * val;
                count++;
            }
        }
    }
    
    float mean_right = sum_right / count;
    float var_right = (sum_sq_right / count) - (mean_right * mean_right);
    float std_dev_right = sqrt(var_right > 0.0001f ? var_right : 0.0001f);  // Avoid division by zero
    
    // If standard deviation is too low (uniform area), skip calculation
    if (std_dev_right < 1.0f) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    float best_zncc = -1.0f;  // ZNCC is in [-1,1], start at -1
    int best_disp = 0;
    
    // Iterate over possible disparities (in the opposite direction compared to left-to-right)
    for (int d = 0; d <= min(max_disp, width-1-x); d++) {
        int xL = x + d;  // Left image coordinates (x + disparity)
        
        // Check if we are within bounds
        if (xL >= width - half_window) continue;
        
        // Calculate mean and standard deviation for the left window
        float sum_left = 0.0f;
        float sum_sq_left = 0.0f;
        count = 0;
        
        // Calculate mean for the left window
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx = xL + wx;
                
                // Check if we are within bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int left_index = ny * width + nx;
                    float val = (float)left_img[left_index];
                    sum_left += val;
                    sum_sq_left += val * val;
                    count++;
                }
            }
        }
        
        float mean_left = sum_left / count;
        float var_left = (sum_sq_left / count) - (mean_left * mean_left);
        float std_dev_left = sqrt(var_left > 0.0001f ? var_left : 0.0001f);  // Avoid division by zero
        
        // If standard deviation is too low (uniform area), skip this disparity
        if (std_dev_left < 1.0f) continue;
        
        // Calculate ZNCC
        float numerator = 0.0f;
        count = 0;
        
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx_right = x + wx;
                int nx_left = xL + wx;
                
                // Check if we are within bounds
                if (nx_left >= 0 && nx_left < width && nx_right >= 0 && nx_right < width && ny >= 0 && ny < height) {
                    int right_index = ny * width + nx_right;
                    int left_index = ny * width + nx_left;
                    float diff_right = (float)right_img[right_index] - mean_right;
                    float diff_left = (float)left_img[left_index] - mean_left;
                    numerator += diff_right * diff_left;
                    count++;
                }
            }
        }
        
        float zncc = numerator / (count * std_dev_right * std_dev_left);
        
        // Keep the disparity with the best correlation
        if (zncc > best_zncc) {
            best_zncc = zncc;
            best_disp = d;
        }
    }
    
    // If the best correlation is too low, consider no match
    if (best_zncc < min_zncc) {
        disparity_map[y * width + x] = 0;
        return;
    }
    
    // Normalize disparity to [0,255]
    disparity_map[y * width + x] = (uchar)((best_disp * 255) / max_disp);
}

// Cross-check kernel to identify occlusions
__kernel void cross_check(
    __global const uchar* left_disp,   // disparity from left to right
    __global const uchar* right_disp,  // disparity from right to left
    __global uchar* checked_disp,      // output disparity map after cross-check
    const int width,
    const int height,
    const int max_disp,               // maximum disparity
    const int threshold               // cross-check threshold
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Get the left-to-right disparity
    const uchar disp_l = left_disp[y * width + x];
    
    // Invalid disparity value (already marked)
    if (disp_l == 0) {
        checked_disp[y * width + x] = 0; // Mark as occlusion
        return;
    }
    
    // Convert normalized disparity back to pixel units
    const float disp_factor = (float)max_disp / 255.0f;
    const int d = (int)(disp_l * disp_factor + 0.5f);
    
    // Compute corresponding point in right image
    const int xr = x - d;
    
    // If the corresponding point is outside the image, it's an occlusion
    if (xr < 0 || xr >= width) {
        checked_disp[y * width + x] = 0; // Mark as occlusion
        return;
    }
    
    // Get the right-to-left disparity at the corresponding point
    const uchar disp_r = right_disp[y * width + xr];
    
    // Convert right-to-left disparity to pixel units
    const int dr = (int)(disp_r * disp_factor + 0.5f);
    
    // Cross-check: if left and right disparities are consistent (within threshold)
    // abs(d - dr) <= threshold, then the match is good
    if (abs(d - dr) <= threshold) {
        checked_disp[y * width + x] = disp_l; // Keep the original disparity
    } else {
        checked_disp[y * width + x] = 0; // Mark as occlusion
    }
}

// Occlusion filling kernel
__kernel void fill_occlusions(
    __global const uchar* input_disp,  // disparity map with occlusions (marked as 0)
    __global uchar* output_disp,       // output filled disparity map
    const int width,
    const int height,
    const int iteration                // iteration number for multi-pass filling
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Current pixel value
    const uchar disp = input_disp[y * width + x];
    
    // If this is not an occluded pixel, keep its value
    if (disp > 0) {
        output_disp[y * width + x] = disp;
        return;
    }
    
    // This is an occluded pixel, try to fill it
    
    // Define the search neighborhood radius based on iteration
    // Start with smaller radius and increase with iterations
    const int radius = 2 + iteration * 2; // Increases with each iteration
    
    // Variables to accumulate weighted disparities
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Parameters for weighted filling
    const float sigma_space = 10.0f;   // Spatial distance weight
    const float sigma_color = 30.0f;   // Color similarity weight (if we had color info)
    
    // For background interpolation, favor points to the right (background)
    // Search area with a bias towards points to the right (lower disparity)
    for (int wy = -radius; wy <= radius; wy++) {
        for (int wx = -radius; wx <= radius; wx++) {
            // Bias the search more towards right side
            // This helps to fill occluded areas with background disparities
            if (iteration % 2 == 0 && wx < 0) continue; // Right-side bias on even iterations
            if (iteration % 2 == 1 && wx > 0) continue; // Left-side bias on odd iterations
            
            int ny = y + wy;
            int nx = x + wx;
            
            // Check if the neighboring pixel is within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uchar neighbor_disp = input_disp[ny * width + nx];
                
                // Skip occluded points
                if (neighbor_disp == 0) continue;
                
                // Calculate spatial distance weight
                float space_dist = (float)(wx*wx + wy*wy);
                float space_weight = exp(-space_dist / (2.0f * sigma_space * sigma_space));
                
                // Apply weight
                float weight = space_weight;
                weighted_sum += weight * (float)neighbor_disp;
                weight_sum += weight;
            }
        }
    }
    
    // If we found valid neighboring pixels, compute weighted average
    if (weight_sum > 0.0f) {
        output_disp[y * width + x] = (uchar)(weighted_sum / weight_sum);
    } else {
        // If no valid neighbors found, keep it as an occlusion
        // (Will be filled in later iterations or by a subsequent smoothing step)
        output_disp[y * width + x] = 0;
    }
}

// Disparity smoothing kernel
__kernel void smooth_disparity(
    __global const uchar* input,
    __global uchar* output,
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Use a larger window for better smoothing
    const int filter_size = 7;
    const int half_filter = filter_size / 2;
    
    // Ignore borders
    if (x < half_filter || y < half_filter || x >= width - half_filter || y >= height - half_filter) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    // 7x7 median filter
    uchar values[filter_size * filter_size];
    int idx = 0;
    int hist[256] = {0}; // Histogram for mode detection
    
    for (int wy = -half_filter; wy <= half_filter; wy++) {
        for (int wx = -half_filter; wx <= half_filter; wx++) {
            int ny = y + wy;
            int nx = x + wx;
            
            // Check if we are within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uchar val = input[ny * width + nx];
                values[idx++] = val;
                hist[val]++; // Update histogram
            }
        }
    }
    
    // Find the modal value (most frequent) in the window
    int max_count = 0;
    uchar mode_value = 0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] > max_count) {
            max_count = hist[i];
            mode_value = (uchar)i;
        }
    }
    
    // Simple sort to find the median
    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                uchar temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    // The median is the middle element
    uchar median_value = values[idx / 2];
    
    // Discontinuity detection and correction
    int center_value = input[y * width + x];
    int sum_diff = 0;
    int count = 0;
    int max_diff = 0;
    
    // Check differences with neighbors
    for (int wy = -2; wy <= 2; wy++) { 
        for (int wx = -2; wx <= 2; wx++) {
            if (wx == 0 && wy == 0) continue; // Ignore the center pixel
            
            int ny = y + wy;
            int nx = x + wx;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_value = input[ny * width + nx];
                int diff = abs(center_value - neighbor_value);
                sum_diff += diff;
                max_diff = max(max_diff, diff);
                count++;
            }
        }
    }
    
    // Decide the final value based on several criteria
    uchar final_value;
    
    // If the average difference is too large, it's likely an artifact
    if (count > 0 && ((float)sum_diff / count > 20.0f || max_diff > 50)) {
        // Use a combination of median and mode to reduce artifacts
        if (max_count > idx / 3) { // If the mode is significant (>33% of pixels)
            final_value = mode_value;
        } else {
            final_value = median_value;
        }
    } else {
        // No artifact detected, use a simplified bilateral filter
        // that preserves edges better
        float sum_weights = 0.0f;
        float sum_values = 0.0f;
        float sigma_space = 2.0f;
        float sigma_range = 10.0f;
        
        for (int wy = -2; wy <= 2; wy++) {
            for (int wx = -2; wx <= 2; wx++) {
                int ny = y + wy;
                int nx = x + wx;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uchar neighbor_value = input[ny * width + nx];
                    float space_dist = (float)(wx*wx + wy*wy);
                    float range_dist = (float)((int)neighbor_value - (int)center_value) * ((int)neighbor_value - (int)center_value);
                    
                    // Combined weight (spatial distance and value difference)
                    float weight = exp(-space_dist/(2.0f*sigma_space*sigma_space)) * 
                                 exp(-range_dist/(2.0f*sigma_range*sigma_range));
                    
                    sum_weights += weight;
                    sum_values += weight * (float)neighbor_value;
                }
            }
        }
        
        if (sum_weights > 0.0f) {
            final_value = (uchar)(sum_values / sum_weights);
        } else {
            final_value = median_value; // Fallback to median
        }
    }
    
    output[y * width + x] = final_value;
}