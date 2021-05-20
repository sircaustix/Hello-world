#!/bin/bash

g++ -g src/**.cpp -Wall -std=c++17  -fopenmp -Iinclude -I/usr/include/opencv4 -I/usr/local/include/eigen3 -Iinclude/spams -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lblas -llapack -o bin/main
