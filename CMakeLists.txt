cmake_minimum_required(VERSION 3.3)
project(CThreads)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -fopenmp -lrt")

set(SOURCE_FILES main.c)
add_executable(CThreads ${SOURCE_FILES})