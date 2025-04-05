CXX = g++
CXXFLAGS = -Wall -O3 -march=native -ftree-vectorize 
TARGET = StereoDisparityParallel
SRCS = StereoDisparityParallel.cpp lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -fopenmp $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
