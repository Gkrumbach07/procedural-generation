CXX ?= g++
# -ffp-contract=off: the default (bit-exact) mode of test_cubesphere needs
# the same operation order as the Python/numba reference (FMA contraction,
# GCC's default on aarch64 and with -march=native, changes ~2700 records).
CXXFLAGS ?= -std=c++17 -O2 -ffp-contract=off -Wall -Wextra
all: test_cubesphere
test_cubesphere: test_cubesphere.cpp ../src/cubesphere_math.h
	$(CXX) $(CXXFLAGS) -I../src test_cubesphere.cpp -o test_cubesphere
test: test_cubesphere
	./test_cubesphere ../../bake/tests/data/cubesphere_vectors.bin
# for a libm whose tan/atan differ from glibc: relax to 1e-12
test-tolerant: test_cubesphere
	./test_cubesphere ../../bake/tests/data/cubesphere_vectors.bin --tolerant
clean:
	rm -f test_cubesphere
.PHONY: all test clean
