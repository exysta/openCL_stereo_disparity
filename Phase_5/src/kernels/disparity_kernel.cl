// disparity_kernel.cl
__kernel void disparity(
    __global const uchar* image0,    // image de gauche (gauche)
    __global const uchar* image1,    // image de droite (droite)
    __global uchar* disparity,       // image de disparité en sortie
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Vérifier que nous sommes dans les limites de l'image
    if (x >= width || y >= height) {
        return;
    }
    
    // Paramètres du matching
    const int window_size = 11;      // taille de la fenêtre augmentée (doit être impair)
    const int half_window = window_size / 2;
    const int max_disparity = 50;     // plage maximale de disparité
    const float min_zncc_threshold = 0.5f;  // seuil minimum augmenté pour réduire les artefacts
    
    // Vérifier que nous avons assez d'espace pour la fenêtre
    if (x < half_window || y < half_window || x >= width - half_window || y >= height - half_window) {
        disparity[y * width + x] = 0;
        return;
    }
    
    // Calcul de la moyenne et de l'écart-type pour la fenêtre de gauche
    float sum_left = 0.0f;
    float sum_sq_left = 0.0f;
    int count = 0;
    
    // Calcul de la moyenne pour la fenêtre de gauche
    for (int wy = -half_window; wy <= half_window; wy++) {
        for (int wx = -half_window; wx <= half_window; wx++) {
            int ny = y + wy;
            int nx = x + wx;
            
            // Vérifier que nous sommes dans les limites
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int left_index = ny * width + nx;
                float val = (float)image0[left_index];
                sum_left += val;
                sum_sq_left += val * val;
                count++;
            }
        }
    }
    
    float mean_left = sum_left / count;
    float var_left = (sum_sq_left / count) - (mean_left * mean_left);
    float std_dev_left = sqrt(var_left > 0.0001f ? var_left : 0.0001f);  // Éviter division par zéro
    
    // Si l'écart-type est trop faible (zone uniforme), on saute le calcul
    if (std_dev_left < 1.0f) {
        disparity[y * width + x] = 0;
        return;
    }
    
    float best_zncc = -1.0f;  // ZNCC est dans [-1,1], on commence à -1
    int best_disparity = 0;
    
    // On itère sur les disparités possibles
    for (int d = 0; d <= min(max_disparity, x); d++) {
        int xR = x - d;
        
        // Vérifier que l'on reste dans l'image
        if (xR < half_window) continue;
        
        // Calcul de la moyenne et de l'écart-type pour la fenêtre de droite
        float sum_right = 0.0f;
        float sum_sq_right = 0.0f;
        count = 0;
        
        // Calcul de la moyenne pour la fenêtre de droite
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx = xR + wx;
                
                // Vérifier que nous sommes dans les limites
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int right_index = ny * width + nx;
                    float val = (float)image1[right_index];
                    sum_right += val;
                    sum_sq_right += val * val;
                    count++;
                }
            }
        }
        
        float mean_right = sum_right / count;
        float var_right = (sum_sq_right / count) - (mean_right * mean_right);
        float std_dev_right = sqrt(var_right > 0.0001f ? var_right : 0.0001f);  // Éviter division par zéro
        
        // Si l'écart-type est trop faible (zone uniforme), on saute cette disparité
        if (std_dev_right < 1.0f) continue;
        
        // Calcul du ZNCC
        float numerator = 0.0f;
        count = 0;
        
        for (int wy = -half_window; wy <= half_window; wy++) {
            for (int wx = -half_window; wx <= half_window; wx++) {
                int ny = y + wy;
                int nx_left = x + wx;
                int nx_right = xR + wx;
                
                // Vérifier que nous sommes dans les limites
                if (nx_left >= 0 && nx_left < width && nx_right >= 0 && nx_right < width && ny >= 0 && ny < height) {
                    int left_index = ny * width + nx_left;
                    int right_index = ny * width + nx_right;
                    float diff_left = (float)image0[left_index] - mean_left;
                    float diff_right = (float)image1[right_index] - mean_right;
                    numerator += diff_left * diff_right;
                    count++;
                }
            }
        }
        
        float zncc = numerator / (count * std_dev_left * std_dev_right);
        
        // On garde la disparité avec la meilleure corrélation
        if (zncc > best_zncc) {
            best_zncc = zncc;
            best_disparity = d;
        }
    }
    
    // Si la meilleure corrélation est trop faible, on considère qu'il n'y a pas de correspondance
    if (best_zncc < min_zncc_threshold) {
        disparity[y * width + x] = 0;
        return;
    }
    
    // Normaliser la disparité sur [0,255]
    disparity[y * width + x] = (uchar)((best_disparity * 255) / max_disparity);
}

// Kernel de lissage multi-étape pour réduire les artefacts
__kernel void smooth_disparity(
    __global const uchar* input_disparity,  // carte de disparité d'entrée
    __global uchar* output_disparity,       // carte de disparité lissée en sortie
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Vérifier que nous sommes dans les limites de l'image
    if (x >= width || y >= height) {
        return;
    }
    
    // Utiliser une fenêtre plus grande pour un meilleur lissage
    const int filter_size = 7; // Augmenté à 7x7
    const int half_filter = filter_size / 2;
    
    // Ignorer les bords
    if (x < half_filter || y < half_filter || x >= width - half_filter || y >= height - half_filter) {
        output_disparity[y * width + x] = input_disparity[y * width + x];
        return;
    }
    
    // Filtre médian 7x7
    uchar values[filter_size * filter_size];
    int idx = 0;
    int hist[256] = {0}; // Histogramme pour détection des modes
    
    for (int wy = -half_filter; wy <= half_filter; wy++) {
        for (int wx = -half_filter; wx <= half_filter; wx++) {
            int ny = y + wy;
            int nx = x + wx;
            
            // Vérifier que nous sommes dans les limites
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uchar val = input_disparity[ny * width + nx];
                values[idx++] = val;
                hist[val]++; // Mettre à jour l'histogramme
            }
        }
    }
    
    // Trouver la valeur modale (la plus fréquente) dans la fenêtre
    int max_count = 0;
    uchar mode_value = 0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] > max_count) {
            max_count = hist[i];
            mode_value = (uchar)i;
        }
    }
    
    // Tri simple pour trouver la médiane
    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                uchar temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    // La médiane est l'élément du milieu
    uchar median_value = values[idx / 2];
    
    // Détection et correction des discontinuités
    int center_value = input_disparity[y * width + x];
    int sum_diff = 0;
    int count = 0;
    int max_diff = 0;
    
    // Vérifier les différences avec les voisins
    for (int wy = -2; wy <= 2; wy++) { // Élargi à 5x5 pour la détection des artefacts
        for (int wx = -2; wx <= 2; wx++) {
            if (wx == 0 && wy == 0) continue; // Ignorer le pixel central
            
            int ny = y + wy;
            int nx = x + wx;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_value = input_disparity[ny * width + nx];
                int diff = abs(center_value - neighbor_value);
                sum_diff += diff;
                max_diff = max(max_diff, diff);
                count++;
            }
        }
    }
    
    // Décider de la valeur finale en fonction de plusieurs critères
    uchar final_value;
    
    // Si la différence moyenne est trop grande, c'est probablement un artefact
    if (count > 0 && ((float)sum_diff / count > 20.0f || max_diff > 50)) {
        // Utiliser une combinaison de médiane et de mode pour réduire les artefacts
        if (max_count > idx / 3) { // Si le mode est significatif (>33% des pixels)
            final_value = mode_value;
        } else {
            final_value = median_value;
        }
    } else {
        // Pas d'artefact détecté, utiliser un filtre bilatéral simplifié
        // qui préserve mieux les bords
        float sum_weights = 0.0f;
        float sum_values = 0.0f;
        float sigma_space = 2.0f;
        float sigma_range = 10.0f;
        
        for (int wy = -2; wy <= 2; wy++) {
            for (int wx = -2; wx <= 2; wx++) {
                int ny = y + wy;
                int nx = x + wx;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uchar neighbor_value = input_disparity[ny * width + nx];
                    float space_dist = (float)(wx*wx + wy*wy);
                    float range_dist = (float)((int)neighbor_value - (int)center_value) * ((int)neighbor_value - (int)center_value);
                    
                    // Poids combiné (distance spatiale et différence de valeur)
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
            final_value = median_value; // Fallback à la médiane
        }
    }
    
    output_disparity[y * width + x] = final_value;
}