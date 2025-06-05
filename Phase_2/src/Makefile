CXX = g++
CXXFLAGS = -lOpenCL
TARGET = ex3
SRCS = ex3.c lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -o $(TARGET) $(SRCS) $(CXXFLAGS)

clean:
	rm -f $(TARGET)