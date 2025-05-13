CXX = g++
CXXFLAGS = -Wall -O3 -march=native -ftree-vectorize -DCL_TARGET_OPENCL_VERSION=200
LDFLAGS = -lOpenCL
TARGET = StereoDisparityOpenCL
SRCS = StereoDisparityOpenCL.cpp lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean