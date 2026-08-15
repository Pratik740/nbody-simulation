cmake_minimum_required(VERSION 3.16)
project(NBodySim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(nbody_sim
    src/main.cpp
    src/Simulation.cpp
)

target_include_directories(nbody_sim PRIVATE include)
target_compile_options(nbody_sim PRIVATE -Wall -Wextra -O2)