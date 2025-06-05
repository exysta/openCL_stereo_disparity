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

#define ZNCC_INVALID_DISP_OUTPUT  255
#define MAX_LOCAL_SIZE 4096  // Par exemple, 64x64 pixels

__kernel void zncc_stereo_match(
    __read_only image2d_t left_image,
    __read_only image2d_t right_image,
    __global uchar* raw_disparity_map,
    const int width,
    const int height,
    const int WIN_SIZE,
    const int MAX_DISP,
    const sampler_t sampler,
    const int direction  // 0 = left-to-right matching, 1 = right-to-left matching

    )
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

    float max_zncc_score = -2.0f; // best score found so far
    int best_disparity_candidate = -1;
 float sum_L = 0.0f;
    
    // Vectoriser la lecture des pixels en traitant 4 pixels à la fois lorsque possible
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        int win_x = -win_radius;
        
        // Traiter 4 pixels à la fois quand c'est possible
        for (; win_x + 3 <= win_radius; win_x += 4) {
            // Lire 4 pixels consécutifs
            uint4 val_L_vec;
            val_L_vec.x = read_imageui(left_image, sampler, (int2)(i + win_x, j + win_y)).x;
            val_L_vec.y = read_imageui(left_image, sampler, (int2)(i + win_x + 1, j + win_y)).x;
            val_L_vec.z = read_imageui(left_image, sampler, (int2)(i + win_x + 2, j + win_y)).x;
            val_L_vec.w = read_imageui(left_image, sampler, (int2)(i + win_x + 3, j + win_y)).x;
            
            // Additionner les 4 valeurs
            sum_L += (float)val_L_vec.x + (float)val_L_vec.y + (float)val_L_vec.z + (float)val_L_vec.w;
        }
        
        // Traiter les pixels restants un par un
        for (; win_x <= win_radius; ++win_x) {
            int2 coord_L = (int2)(i + win_x, j + win_y);
            sum_L += (float)read_imageui(left_image, sampler, coord_L).x;
        }
    }
    
    float mean_L = sum_L * inv_num_pixels;
    float sum_sq_diff_L = 0.0f; // Sum of (L_i - mean_L)^2

    // Calculate sum of squared differences for left window (needed for denominator)
    for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
        int win_x = -win_radius;
        
        // Traiter 4 pixels à la fois quand c'est possible
        for (; win_x + 3 <= win_radius; win_x += 4) {
            float4 val_L_vec;
            val_L_vec.x = (float)read_imageui(left_image, sampler, (int2)(i + win_x, j + win_y)).x;
            val_L_vec.y = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 1, j + win_y)).x;
            val_L_vec.z = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 2, j + win_y)).x;
            val_L_vec.w = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 3, j + win_y)).x;
            
            // Calculer les différences vectorisées
            float4 diff_L_vec = val_L_vec - mean_L;
            
            // Calculer les carrés et les additionner
            sum_sq_diff_L += dot(diff_L_vec, diff_L_vec);
        }
        
        // Traiter les pixels restants un par un
        for (; win_x <= win_radius; ++win_x) {
            int2 coord_L = (int2)(i + win_x, j + win_y);
            float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
            float diff_L = val_L - mean_L;
            sum_sq_diff_L += diff_L * diff_L;
        }
    }

    // Optimisation: traiter plusieurs disparités en parallèle lorsque possible
    int d = 0;
    while (d < MAX_DISP) {
        // Traiter 4 disparités à la fois si possible
        if (d + 3 < MAX_DISP) {
            float4 zncc_scores = (float4)(-2.0f);
            int4 disp_vec;
            disp_vec.x = direction == 0 ? d : -d;
            disp_vec.y = direction == 0 ? (d+1) : -(d+1);
            disp_vec.z = direction == 0 ? (d+2) : -(d+2);
            disp_vec.w = direction == 0 ? (d+3) : -(d+3);
            
            // Vérifier si les disparités sont valides
            bool4 valid_disp;
            valid_disp.x = !((direction == 0 && i < d) || (direction == 1 && i + d >= width));
            valid_disp.y = !((direction == 0 && i < (d+1)) || (direction == 1 && i + (d+1) >= width));
            valid_disp.z = !((direction == 0 && i < (d+2)) || (direction == 1 && i + (d+2) >= width));
            valid_disp.w = !((direction == 0 && i < (d+3)) || (direction == 1 && i + (d+3) >= width));
            
            if (valid_disp.x || valid_disp.y || valid_disp.z || valid_disp.w) {
                // Calculer les scores ZNCC pour les 4 disparités
                for (int disp_idx = 0; disp_idx < 4; disp_idx++) {
                    if ((disp_idx == 0 && valid_disp.x) || 
                        (disp_idx == 1 && valid_disp.y) || 
                        (disp_idx == 2 && valid_disp.z) || 
                        (disp_idx == 3 && valid_disp.w)) {
                        
                        int current_d = d + disp_idx;
                        int disp = direction == 0 ? current_d : -current_d;
                        
                        float sum_R = 0.0f;
                        
                        // Calculer la moyenne pour la fenêtre droite décalée
                        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
                            int win_x = -win_radius;
                            
                            // Traiter 4 pixels à la fois quand c'est possible
                            for (; win_x + 3 <= win_radius; win_x += 4) {
                                uint4 val_R_vec;
                                val_R_vec.x = read_imageui(right_image, sampler, (int2)(i - disp + win_x, j + win_y)).x;
                                val_R_vec.y = read_imageui(right_image, sampler, (int2)(i - disp + win_x + 1, j + win_y)).x;
                                val_R_vec.z = read_imageui(right_image, sampler, (int2)(i - disp + win_x + 2, j + win_y)).x;
                                val_R_vec.w = read_imageui(right_image, sampler, (int2)(i - disp + win_x + 3, j + win_y)).x;
                                
                                sum_R += (float)val_R_vec.x + (float)val_R_vec.y + (float)val_R_vec.z + (float)val_R_vec.w;
                            }
                            
                            // Traiter les pixels restants un par un
                            for (; win_x <= win_radius; ++win_x) {
                                int2 coord_R = (int2)(i - disp + win_x, j + win_y);
                                sum_R += (float)read_imageui(right_image, sampler, coord_R).x;
                            }
                        }
                        
                        float mean_R = sum_R * inv_num_pixels;
                        float numerator = 0.0f;
                        float sum_sq_diff_R = 0.0f;
                        
                        // Calculer le numérateur et la somme des carrés des différences pour la fenêtre droite
                        for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
                            int win_x = -win_radius;
                            
                            // Traiter 4 pixels à la fois quand c'est possible
                            for (; win_x + 3 <= win_radius; win_x += 4) {
                                // Lire 4 pixels consécutifs de l'image gauche
                                float4 val_L_vec;
                                val_L_vec.x = (float)read_imageui(left_image, sampler, (int2)(i + win_x, j + win_y)).x;
                                val_L_vec.y = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 1, j + win_y)).x;
                                val_L_vec.z = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 2, j + win_y)).x;
                                val_L_vec.w = (float)read_imageui(left_image, sampler, (int2)(i + win_x + 3, j + win_y)).x;
                                
                                // Lire 4 pixels consécutifs de l'image droite
                                float4 val_R_vec;
                                val_R_vec.x = (float)read_imageui(right_image, sampler, (int2)(i - disp + win_x, j + win_y)).x;
                                val_R_vec.y = (float)read_imageui(right_image, sampler, (int2)(i - disp + win_x + 1, j + win_y)).x;
                                val_R_vec.z = (float)read_imageui(right_image, sampler, (int2)(i - disp + win_x + 2, j + win_y)).x;
                                val_R_vec.w = (float)read_imageui(right_image, sampler, (int2)(i - disp + win_x + 3, j + win_y)).x;
                                
                                // Calculer les différences vectorisées
                                float4 diff_L_vec = val_L_vec - mean_L;
                                float4 diff_R_vec = val_R_vec - mean_R;
                                
                                // Calculer le numérateur et la somme des carrés
                                numerator += dot(diff_L_vec, diff_R_vec);
                                sum_sq_diff_R += dot(diff_R_vec, diff_R_vec);
                            }
                            
                            // Traiter les pixels restants un par un
                            for (; win_x <= win_radius; ++win_x) {
                                int2 coord_L = (int2)(i + win_x, j + win_y);
                                int2 coord_R = (int2)(i - disp + win_x, j + win_y);
                                
                                float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
                                float val_R = (float)read_imageui(right_image, sampler, coord_R).x;
                                
                                float diff_L = val_L - mean_L;
                                float diff_R = val_R - mean_R;
                                
                                numerator += diff_L * diff_R;
                                sum_sq_diff_R += diff_R * diff_R;
                            }
                        }
                        
                        // Calculer le score ZNCC
                        float current_zncc_score = -2.0f; // Score invalide par défaut
                        float denominator = sqrt(sum_sq_diff_L * sum_sq_diff_R);
                        const float epsilon = 1e-6f; // petite valeur pour éviter la division par zéro
                        
                        if (denominator > epsilon) {
                            current_zncc_score = numerator / denominator;
                            // S'assurer que le score est dans la plage valide [-1, 1]
                            current_zncc_score = clamp(current_zncc_score, -1.0f, 1.0f);
                        }
                        
                        // Stocker le score dans le vecteur
                        if (disp_idx == 0) zncc_scores.x = current_zncc_score;
                        else if (disp_idx == 1) zncc_scores.y = current_zncc_score;
                        else if (disp_idx == 2) zncc_scores.z = current_zncc_score;
                        else zncc_scores.w = current_zncc_score;
                    }
                }
                
                // Trouver le meilleur score parmi les 4 disparités
                if (valid_disp.x && zncc_scores.x > max_zncc_score) {
                    max_zncc_score = zncc_scores.x;
                    best_disparity_candidate = d;
                }
                if (valid_disp.y && zncc_scores.y > max_zncc_score) {
                    max_zncc_score = zncc_scores.y;
                    best_disparity_candidate = d + 1;
                }
                if (valid_disp.z && zncc_scores.z > max_zncc_score) {
                    max_zncc_score = zncc_scores.z;
                    best_disparity_candidate = d + 2;
                }
                if (valid_disp.w && zncc_scores.w > max_zncc_score) {
                    max_zncc_score = zncc_scores.w;
                    best_disparity_candidate = d + 3;
                }
            }
            
            d += 4;
        } else {
            // Traiter les disparités restantes une par une (code original)
            int disp = direction == 0 ? d : -d;
            
            // Check if the right window's center pixel is valid
            if ((direction == 0 && i < d) || (direction == 1 && i + d >= width)) {
                d++;
                continue; // Skip if right pixel coord is out of bounds
            }
            
            float sum_R = 0.0f;
            
            // Calculate mean for the shifted right window
            for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
                for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                    int2 coord_R = (int2)(i - disp + win_x, j + win_y);
                    sum_R += (float)read_imageui(right_image, sampler, coord_R).x;
                }
            }
            float mean_R = sum_R * inv_num_pixels;
            
            float numerator = 0.0f;    // Sum of (L_i - mean_L) * (R_i - mean_R)
            float sum_sq_diff_R = 0.0f; // Sum of (R_i - mean_R)^2
            
            // Calculate numerator and sum of squared differences for right window
            for (int win_y = -win_radius; win_y <= win_radius; ++win_y) {
                for (int win_x = -win_radius; win_x <= win_radius; ++win_x) {
                    int2 coord_L = (int2)(i + win_x, j + win_y);
                    int2 coord_R = (int2)(i - disp + win_x, j + win_y);
                    
                    float val_L = (float)read_imageui(left_image, sampler, coord_L).x;
                    float val_R = (float)read_imageui(right_image, sampler, coord_R).x;
                    
                    float diff_L = val_L - mean_L;
                    float diff_R = val_R - mean_R;
                    
                    numerator += diff_L * diff_R;
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
            
            // Found a better match?
            if (current_zncc_score > max_zncc_score) {
                max_zncc_score = current_zncc_score;
                best_disparity_candidate = d;
            }
            
            d++;
        }
    }
    
    const int output_index = j * width + i;
    const float ZNCC_ACCEPTANCE_THRESHOLD = 0.5f; // Example threshold
    
    // If max_zncc_score is good enough AND we actually found a candidate disparity
    if (max_zncc_score > ZNCC_ACCEPTANCE_THRESHOLD && best_disparity_candidate != -1) {
        raw_disparity_map[output_index] = (uchar)best_disparity_candidate; // Store raw disparity
    } else {

        // No acceptable match found, or MAX_DISP was 0
        raw_disparity_map[output_index] = ZNCC_INVALID_DISP_OUTPUT; // Use the #defined marker
    }
}

//define the invalid value to be the same as for the ZNCC
#define ZNCC_NO_MATCH_OR_BORDER_MARKER ZNCC_INVALID_DISP_OUTPUT

// --- Combined CrossCheck and OcclusionFill Kernel ---
__kernel void cross_check_and_fill_kernel(
    __global const uchar* disparity_map_lr,     // Input: Left-to-Right disparity map
    __global const uchar* disparity_map_rl,     // Input: Right-to-Left disparity map
    __global uchar* final_disparity_map,    // Output: Validated and filled disparity map
    const int width,
    const int height,
    const int cross_check_threshold,
    const uchar POST_INVALID_MARKER
    // ZNCC_NO_MATCH_OR_BORDER_MARKER should be a #define or passed if it can vary
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    const int current_idx = y * width + x;
    uchar disp_lr_val = disparity_map_lr[current_idx];
    uchar cross_checked_disp_val;

    // --- Part 1: Cross-Check ---
    if (disp_lr_val == ZNCC_NO_MATCH_OR_BORDER_MARKER) {
        cross_checked_disp_val = POST_INVALID_MARKER;
    } else {
        int x_in_rl_map_coord = x - disp_lr_val;
        if (x_in_rl_map_coord >= 0 && x_in_rl_map_coord < width) {
            uchar disp_rl_val = disparity_map_rl[y * width + x_in_rl_map_coord];
            if (disp_rl_val == ZNCC_NO_MATCH_OR_BORDER_MARKER || abs((int)disp_lr_val - (int)disp_rl_val) > cross_check_threshold) {
                cross_checked_disp_val = POST_INVALID_MARKER;
            } else {
                cross_checked_disp_val = disp_lr_val;
            }
        } else {
            cross_checked_disp_val = POST_INVALID_MARKER;
        }
    }

    // --- Part 2: Occlusion Fill (if needed) ---
    if (cross_checked_disp_val != POST_INVALID_MARKER) {
        final_disparity_map[current_idx] = cross_checked_disp_val;
    } else {
        // Pixel is occluded after cross-check, try to fill it by looking left.
        // IMPORTANT: This fill logic reads from disparity_map_lr.
        // It assumes that if disparity_map_lr[y * width + search_x] was valid,
        // it would also pass its own cross-check. This is a simplification.
        // A more robust fill would re-evaluate cross-check for pixels to the left.
        // However, for simplicity and to avoid excessive re-computation in a single kernel,
        // we'll use the simpler "propagate last valid *original LR* disparity".
        uchar fill_value = POST_INVALID_MARKER;

        for (int search_x = x - 1; search_x >= 0; --search_x) {
            uchar prev_lr_disp = disparity_map_lr[y * width + search_x]; // Read from original LR

            // We need to ensure this 'prev_lr_disp' itself would be valid.
            // This requires a "mini" cross-check for the 'prev_lr_disp'
            if (prev_lr_disp != ZNCC_NO_MATCH_OR_BORDER_MARKER) {
                int prev_x_in_rl_map_coord = search_x - prev_lr_disp;
                if (prev_x_in_rl_map_coord >= 0 && prev_x_in_rl_map_coord < width) {
                    uchar prev_rl_disp = disparity_map_rl[y * width + prev_x_in_rl_map_coord];
                    if (prev_rl_disp != ZNCC_NO_MATCH_OR_BORDER_MARKER && abs((int)prev_lr_disp - (int)prev_rl_disp) <= cross_check_threshold) {
                        fill_value = prev_lr_disp; // This previous LR disparity is valid
                        break;
                    }
                }
            }
            // If the loop continues, it means prev_lr_disp was not valid or ZNCC_NO_MATCH
        }
        final_disparity_map[current_idx] = fill_value;
    }
}

// --- Kernel to Scale Raw Disparity Map for Visualization ---
// Input: raw disparity values (e.g., 0 to MAX_DISP-1) and an invalid marker
// Output: uchar values (0-255) for visualization
__kernel void scale_disparity_for_visualization_kernel(
    __global const uchar* raw_disparity_map_input,    // Input: Raw disparity values
    __global uchar* visualizable_disparity_map_output, // Output: Scaled uchar values (0-255)
    const int width,
    const int height,
    const int MAX_DISP_PARAM,                          // The maximum disparity value (e.g., 64 if range is 0-63)
    const uchar POST_INVALID_MARKER,                 // Marker for invalid/occluded pixels in input
    const uchar VISUAL_INVALID_MARKER                // What to map invalid pixels to in output (e.g., 0 for black)
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    const int current_idx = y * width + x;
    uchar raw_disp_val = raw_disparity_map_input[current_idx];

    uchar output_value;

    if (raw_disp_val == POST_INVALID_MARKER) {
        output_value = VISUAL_INVALID_MARKER; // e.g., 0 for black
    } else {
        // Scale valid disparity [0, MAX_DISP_PARAM-1] to [0, 255]
        // If MAX_DISP_PARAM is 1 (meaning only disparity 0 is possible), map it to 255.
        // Otherwise, perform linear scaling.
        if (MAX_DISP_PARAM <= 1) { // Handles the case where MAX_DISP was 0 or 1
            if (raw_disp_val == 0) { // Assuming 0 is the only valid disparity
                output_value = 255; // Map disparity 0 to brightest
            } else {
                output_value = VISUAL_INVALID_MARKER; // Should not happen if raw_disp_val wasn't POST_INVALID_MARKER
            }
        } else {
            // Ensure raw_disp_val doesn't exceed MAX_DISP_PARAM-1 if it's not the invalid marker
            // This can happen if MAX_DISP_PARAM is less than the range of uchar and POST_INVALID_MARKER is e.g. 255
            uchar clamped_raw_disp = min(raw_disp_val, (uchar)(MAX_DISP_PARAM - 1));

            float scaled_disparity_float = ((float)clamped_raw_disp * 255.0f) / ((float)MAX_DISP_PARAM - 1.0f);
            
            // Typically, for visualization, higher disparity (further away object perceived as closer due to shift)
            // is mapped to brighter values, or sometimes inverted.
            // Current: 0 disparity -> 0, MAX_DISP-1 disparity -> 255
            output_value = convert_uchar_sat(scaled_disparity_float);

            // Optional Inversion (if you want higher disparities to be darker):
            // output_value = 255 - convert_uchar_sat(scaled_disparity_float);
        }
    }

    visualizable_disparity_map_output[current_idx] = output_value;
}