// image_processing_kernel.cl
// Kernel for image resizing and grayscale conversion

// Convert RGB to grayscale using luminance formula
__kernel void rgb_to_grayscale(
    __global const uchar* input,    // input image (RGB)
    __global uchar* output,         // output image (grayscale)
    const int width,
    const int height,
    const int channels              // number of input image channels (3 for RGB)
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int idx = y * width + x;
    
    // Luminance formula: Y = 0.299R + 0.587G + 0.114B
    float gray = 0.299f * input[idx*channels] 
                + 0.587f * input[idx*channels+1] 
                + 0.114f * input[idx*channels+2];
    
    output[idx] = convert_uchar_sat(gray);
}

// Bilinear interpolation for image resizing
__kernel void resize_grayscale(
    __global const uchar* input,    // input image (grayscale)
    __global uchar* output,         // output image (resized grayscale)
    const int src_w,
    const int src_h,
    const int dst_w,
    const int dst_h
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= dst_w || y >= dst_h) return;
    
    // Calculate normalized coordinates
    float x_ratio = (float)(src_w - 1) / dst_w;
    float y_ratio = (float)(src_h - 1) / dst_h;
    
    float src_x = x * x_ratio;
    float src_y = y * y_ratio;
    
    // Get neighboring pixel coordinates
    int x1 = clamp((int)floor(src_x), 0, src_w-1);
    int y1 = clamp((int)floor(src_y), 0, src_h-1);
    int x2 = clamp(x1 + 1, 0, src_w-1);
    int y2 = clamp(y1 + 1, 0, src_h-1);
    
    // Calculate interpolation weights
    float x_weight = src_x - x1;
    float y_weight = src_y - y1;
    
    // Perform bilinear interpolation
    float val = (1-x_weight)*(1-y_weight)*input[y1*src_w + x1]
              + x_weight*(1-y_weight)*input[y1*src_w + x2]
              + (1-x_weight)*y_weight*input[y2*src_w + x1]
              + x_weight*y_weight*input[y2*src_w + x2];
    
    output[y*dst_w + x] = convert_uchar_sat(val);
}
