CPP=g++
CPPFLAGS= -O2 -Iinclude -Iinclude/spams -I/usr/include/boost `pkg-config --cflags opencv4` `pkg-config --cflags eigen3` -fopenmp -Wall
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lblas -llapack
SRCDIR=src/
OUTDIR=bin/
SRC_FILES=$(filter-out src/main.cpp, $(wildcard src/*.cpp))
main:$(SRCDIR)main.cpp
		$(CPP) $(CPPFLAGS) $(SRC_FILES) $^ -o $(OUTDIR)$@ $(LDFLAGS)
