__kernel void resizeImage(__global const uchar4* input, __global uchar4* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int newWidth = width / 4;
    int newHeight = height / 4;
    if (x < newWidth && y < newHeight) {
        int srcX = x * 4;
        int srcY = y * 4;
        output[y * newWidth + x] = input[srcY * width + srcX];
        printf("Resize: (%d,%d) -> (%d,%d)\n", srcX, srcY, x, y);
    }
}

__kernel void grayscaleImage(__global const uchar4* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int newWidth = width / 4;
    int newHeight = height / 4;
    if (x < newWidth && y < newHeight) {
        uchar4 pixel = input[y * newWidth + x];
        output[y * newWidth + x] = (uchar)(0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z);
        printf("Grayscale: processing pixel (%d,%d)\n", x, y);
    }
}

__kernel void applyFilter(__global const uchar* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int newWidth = width / 4;
    int newHeight = height / 4;
    if (x >= 2 && x < newWidth - 2 && y >= 2 && y < newHeight - 2) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                sum += input[(y + j) * newWidth + (x + i)];
            }
        }
        output[y * newWidth + x] = (uchar)(sum / 25.0f);
        printf("Filter: processing pixel (%d,%d)\n", x, y);
    }
}
