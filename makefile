final.out: matrix.hpp neural_network.hpp test_xor.cpp
	g++ -std=c++20 matrix.hpp neural_network.hpp test_xor.cpp

clean:
	rm *.o *.gch