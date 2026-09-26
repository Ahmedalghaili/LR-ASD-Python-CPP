#!/bin/bash
#Chih-Yuan Yang 2026/8/03
#Build the Robot Nurse Helper project with CMake
#I wrote this shell script file to call another shell script file.

# The workstation uses a user-local CUDA toolkit for the CUDA-enabled
# whisper.cpp build.  Export it here so future server rebuilds can resolve
# libcudart/cuBLAS without requiring a system-wide CUDA installation.
cuda_whisper_lib="$HOME/.local/cuda-12.8/usr/local/cuda-12.8/targets/x86_64-linux/lib"
if [ -d "$cuda_whisper_lib" ]; then
    export LIBRARY_PATH="$cuda_whisper_lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$cuda_whisper_lib:${LD_LIBRARY_PATH:-}"
fi

if [ $# == 1 ]; then
    if [[ "$1" == "clean" ]]; then
        rm -rf build
    elif [[ "$1" == "Zenbo" || "$1" == "ZenboJrII" ]]; then
        echo "Building for Zenbo or ZenboJrII"
        cmake -S . -B build -DROBOT_MODEL=Zenbo
        cmake --build build -j $(nproc)
    elif [[ "$1" == "Kebbi" ]]; then
        echo "Building for Kebbi"
        cmake -S . -B build -DROBOT_MODEL=Kebbi -DCMAKE_CXX_FLAGS="-Wno-psabi"
        cmake --build build -j $(nproc)
    else
        echo "Invalid argument: $1"
        echo "Usage: ./build_project.sh [clean|Zenbo|ZenboJrII|Kebbi]"
    fi
else
    echo "Usage: ./build_project.sh [clean|Zenbo|ZenboJrII|Kebbi]"
fi
