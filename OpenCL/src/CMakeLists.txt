cmake_minimum_required(VERSION 3.10) # 3.1 << C_STANDARD 11

project(HelloOpenCL LANGUAGES C)

# Now find OpenCL package
find_package(OpenCL REQUIRED)


# Create an executable
add_executable(${PROJECT_NAME} helloOpenCL.c)

# Link OpenCL library
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

# Set C standards
set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

# Set OpenCL target version
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=100)
