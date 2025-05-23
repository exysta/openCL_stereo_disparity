CXX = g++
TARGETS = ex2 ex3 StereoDisparity

ifeq ($(shell uname), Darwin)
    # macOS specific flags
    OPENCL_FLAGS = -framework OpenCL
else
    # Linux/other specific flags
    OPENCL_FLAGS = -lOpenCL
endif

all: $(TARGETS)

ex2: ex2.cpp lodepng.cpp
	$(CXX) -o $@ $^

ex3: ex3.c lodepng.cpp
	$(CXX) -o $@ $^ $(OPENCL_FLAGS)

StereoDisparity: StereoDisparity.cpp lodepng.cpp
	$(CXX) -o $@ $^

clean:
	rm -f $(TARGETS)