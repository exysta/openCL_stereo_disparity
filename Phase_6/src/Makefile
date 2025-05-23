CXX = g++
CXXFLAGS = -Wall -O3 -march=native -ftree-vectorize -DCL_TARGET_OPENCL_VERSION=200

# Détection du système d'exploitation pour les flags OpenCL
ifeq ($(shell uname), Darwin)
    # macOS utilise le framework OpenCL
    LDFLAGS = -framework OpenCL
else
    # Linux et autres utilisent la bibliothèque OpenCL
    LDFLAGS = -lOpenCL
endif

TARGET = StereoDisparityOpenCL
SRCS = StereoDisparityOpenCL.cpp lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean