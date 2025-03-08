# Exercise 3: OpenCL Image Processing

This exercise implements the same image processing operations as Exercise 2, but using OpenCL for parallel processing:
- Image resizing (1/4 of original size)
- Grayscale conversion
- 5x5 filter application

## Files
- `exercise3_image_processing.c`: Main implementation file
- `exercise3_kernels.cl`: OpenCL kernel implementations
- `lodepng.h` and `lodepng.cpp`: PNG image handling library
- `Makefile`: Compilation instructions

## Compilation
```bash
make
```

## Execution
```bash
./exercise3
```

## Input/Output
- Input: `image_0.png`
- Output: `image_0_bw.png` (processed image)

## Performance
The program includes profiling information to compare execution times with Exercise 2.
