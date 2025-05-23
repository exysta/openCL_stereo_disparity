CXX = g++
CXXFLAGS = -Wall -O3 -march=native -ftree-vectorize
TARGET = StereoDisparityParallel
SRCS = StereoDisparityParallel.cpp lodepng.cpp

# Détection du système d'exploitation pour les flags OpenMP
ifeq ($(shell uname), Darwin)
    # macOS - utiliser clang avec les flags OpenMP appropriés
    CXX = clang++
    CXXFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
else
    # Linux et autres systèmes
    CXXFLAGS += -fopenmp
endif

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
