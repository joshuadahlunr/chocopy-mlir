cmake_minimum_required(VERSION 3.21)
project(HeterogenousChocopy CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(cpp-peglib)

add_executable(chocopy driver.cpp)
target_link_libraries(chocopy PUBLIC peglib)