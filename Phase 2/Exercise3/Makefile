CXX = g++
CXXFLAGS = -framework OpenCL
TARGET = exercise3
SRCS = exercise3_image_processing.c lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -o $(TARGET) $(SRCS) $(CXXFLAGS)

clean:
	rm -f $(TARGET)
