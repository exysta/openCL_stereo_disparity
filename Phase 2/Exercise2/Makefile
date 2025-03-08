CXX = g++
CXXFLAGS = -Wall
TARGET = exercise2_image_processing
SRCS = exercise2_image_processing.c lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -o $(TARGET) $(SRCS) $(CXXFLAGS)

clean:
	rm -f $(TARGET)
