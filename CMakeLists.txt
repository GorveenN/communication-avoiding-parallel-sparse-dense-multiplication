cmake_minimum_required(VERSION 3.10)
project(mpi_matmul)

include_directories(include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

set(CMAKE_CXX_COMPILER CC)
set(CMAKE_C_COMPILER cc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -g3 -Wall -O3")

add_executable(matrixmul
        include/argparse.h
        include/communicator.h
        include/densematgen.h
        include/matmul.h
        include/matrix.h
        include/parser.h
        include/utils.h
        include/zip.h
        src/argparse.cpp
        src/communicator.cpp
        src/densematgen.cpp
        src/matmul.cpp
        src/matrix.cpp
        src/parser.cpp
        src/utils.cpp
        main.cpp )
