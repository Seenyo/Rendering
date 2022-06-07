cmake_minimum_required(VERSION 3.22)
project(rendering1)

set(CMAKE_CXX_STANDARD 14)
include_directories("C:/Users/narus/eigen-3.4.0")

add_executable(rendering1 main.cpp Camera.cpp Camera.h)
target_link_libraries(rendering1 libopengl32.a libglu32.a libfreeglut.a)
