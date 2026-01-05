cmake_minimum_required(VERSION 3.21)
project(HeterogenousChocopy CXX)

add_subdirectory(cpp-peglib)

add_executable(chocopy parser.cpp)
target_link_libraries(chocopy PUBLIC peglib)