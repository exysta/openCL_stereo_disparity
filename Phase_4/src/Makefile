CXX = g++
CXXFLAGS = -O2 -Wall
TARGET = StereoDisparityParallel
SRCS = StereoDisparityParallel.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -fopenmp $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
