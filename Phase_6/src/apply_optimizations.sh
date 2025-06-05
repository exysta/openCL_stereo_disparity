#!/bin/bash

# Définir la constante MAX_LOCAL_SIZE dans kernels.cl
if ! grep -q "MAX_LOCAL_SIZE" kernels.cl; then
    sed -i '' 's/#define ZNCC_INVALID_DISP_OUTPUT  255/#define ZNCC_INVALID_DISP_OUTPUT  255\n#define MAX_LOCAL_SIZE 4096  \/\/ Par exemple, 64x64 pixels/' kernels.cl
fi

# Modifier le code C++ pour utiliser la mémoire locale
# 1. Définir la taille des groupes de travail locaux
sed -i '' 's/size_t globalSizeZNCC\[2\] = {out_width, out_height};/\/\/ Définir la taille des groupes de travail pour optimiser l'\''utilisation de la mémoire locale\n        \/\/ Choisir une taille qui est un multiple de 16 pour de meilleures performances sur GPU\n        const size_t localSizeZNCC[2] = {16, 16}; \/\/ 16x16 = 256 threads par groupe\n        \n        \/\/ Ajuster la taille globale pour qu'\''elle soit un multiple de la taille locale\n        size_t globalSizeZNCC[2] = {\n            ((out_width + localSizeZNCC[0] - 1) \/ localSizeZNCC[0]) * localSizeZNCC[0],\n            ((out_height + localSizeZNCC[1] - 1) \/ localSizeZNCC[1]) * localSizeZNCC[1]\n        };/' StereoDisparityOpenCL.cpp

# 2. Modifier les appels à clEnqueueNDRangeKernel pour les kernels ZNCC
sed -i '' 's/err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, globalSizeZNCC, NULL, 0, NULL, NULL);/err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, globalSizeZNCC, localSizeZNCC, 0, NULL, NULL);/g' StereoDisparityOpenCL.cpp

echo "Optimisations appliquées avec succès !"
