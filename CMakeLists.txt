cmake_minimum_required(VERSION 3.16)
project(light_bar_detect CXX)

set(CMAKE_CXX_STANDARD 14)

# 指定 OpenCV 的安装位置（就是我们源码编译安装的 4.5.4 版本）
set(OpenCV_DIR /usr/local/opencv4.5.4/lib/cmake/opencv4)

# 在 CMake 中搜索外部库 OpenCV，REQUIRED 表示必须找到才能继续
find_package(OpenCV 4.5.4 REQUIRED)

# 指定头文件搜索路径，这样代码里 #include <opencv2/opencv.hpp> 才能找到
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(light_bar src/main.cpp)

# 链接 OpenCV 库
target_link_libraries(light_bar ${OpenCV_LIBS})
