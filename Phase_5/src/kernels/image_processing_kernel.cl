// image_processing_kernel.cl
// Kernel pour le redimensionnement et la conversion en niveaux de gris des images

// Conversion RGB vers niveaux de gris
__kernel void rgb_to_grayscale(
    __global const uchar* input,    // image d'entrée (RGB)
    __global uchar* output,         // image de sortie (niveaux de gris)
    const int width,
    const int height,
    const int channels              // nombre de canaux de l'image d'entrée (3 pour RGB)
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int input_idx = (y * width + x) * channels;
    const int output_idx = y * width + x;
    
    // Conversion RGB vers niveaux de gris en utilisant la formule standard
    // Y = 0.299*R + 0.587*G + 0.114*B
    float r = (float)input[input_idx];
    float g = (float)input[input_idx + 1];
    float b = (float)input[input_idx + 2];
    
    output[output_idx] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
}

// Redimensionnement d'une image en niveaux de gris
__kernel void resize_grayscale(
    __global const uchar* input,    // image d'entrée (niveaux de gris)
    __global uchar* output,         // image de sortie (niveaux de gris redimensionnée)
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Calculer les coordonnées correspondantes dans l'image source
    // en utilisant l'interpolation bilinéaire
    float src_x = ((float)x + 0.5f) * ((float)src_width / (float)dst_width) - 0.5f;
    float src_y = ((float)y + 0.5f) * ((float)src_height / (float)dst_height) - 0.5f;
    
    int src_x_int = (int)src_x;
    int src_y_int = (int)src_y;
    
    // Assurer que les coordonnées sont dans les limites de l'image source
    src_x_int = max(0, min(src_width - 2, src_x_int));
    src_y_int = max(0, min(src_height - 2, src_y_int));
    
    // Calculer les poids pour l'interpolation
    float dx = src_x - src_x_int;
    float dy = src_y - src_y_int;
    
    // Récupérer les valeurs des pixels voisins
    int idx00 = src_y_int * src_width + src_x_int;
    int idx01 = src_y_int * src_width + (src_x_int + 1);
    int idx10 = (src_y_int + 1) * src_width + src_x_int;
    int idx11 = (src_y_int + 1) * src_width + (src_x_int + 1);
    
    float p00 = (float)input[idx00];
    float p01 = (float)input[idx01];
    float p10 = (float)input[idx10];
    float p11 = (float)input[idx11];
    
    // Interpolation bilinéaire
    float value = (1.0f - dx) * (1.0f - dy) * p00 +
                  dx * (1.0f - dy) * p01 +
                  (1.0f - dx) * dy * p10 +
                  dx * dy * p11;
    
    output[y * dst_width + x] = (uchar)value;
}
