final.out: matrix.hpp neural_network.hpp tests.hpp main.cpp
	g++ -std=c++20 matrix.hpp neural_network.hpp tests.hpp main.cpp

clean:
	rm *.o *.gch