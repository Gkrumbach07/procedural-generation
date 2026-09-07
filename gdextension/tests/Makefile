CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
all: test_cubesphere
test_cubesphere: test_cubesphere.cpp ../src/cubesphere_math.h
	$(CXX) $(CXXFLAGS) -I../src test_cubesphere.cpp -o test_cubesphere
test: test_cubesphere
	./test_cubesphere ../../bake/tests/data/cubesphere_vectors.bin
clean:
	rm -f test_cubesphere
.PHONY: all test clean
