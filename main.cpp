#include <iostream>
#include "neural_network.hpp"
#include <vector>
#include <cstdio>
#include <string>
#include <cmath>

// TODO:
// 1) Инпут из файла и сразу проверить на зависимости y = x и y = x с шумом
// 2) класс Tester - который позволяет получить качество для переданной модели на переданных данных
// 3) Поменять лосс с true - pred на MSE для регрессии (сделать возможность выбирать лосс)

// Можно сделать регрессию - если поменять сигмоид на релу

// XOR - разделение y=x и внутренняя-внешняя область круга

template<typename T>
std::vector<T> vectorFromCsv(std::string path) {

}

// class which tests the quality of the network and outputs results to the file; For classification will return accuracy; MSE for regression.
template<typename T>
void getQuality(std::vector<std::vector<T>> X_test, std::vector<std::vector<T>> Y_true, sp::SimpleNeuralNetwork nn, bool is_classification=true) {
    /*
    is_classification: if true: will return accuracy
    */
    T running_sq_sum = 0.;
    float running_correct = 0.;

    for (size_t i = 0; i < X_test.size(); i++) {
        std::vector<T> input = X_test[i];
        nn.feedForword(input);
        std::vector<T> preds = nn.getPredictions();
        if (is_classification) {
            if (std::round(preds[0]) == Y_true[i][0]) running_correct += 1.;
        } else {
            running_sq_sum += (preds[0] - Y_true[i][0]) * (preds[0] - Y_true[i][0]);
        }
    }

    if (is_classification) {
        std::cout << "Accuracy:" << running_correct / X_test.size() << std::endl;
    } else {
        std::cout << "MSE:" << running_sq_sum / X_test.size() << std::endl;
    }
    std::cout << "---------------" << std::endl;
}

void xorTesting() {
    // creating neural network

    std::vector<uint32_t> topology = {2, 5, 1};
    sp::SimpleNeuralNetwork nn(topology, 1.);
    
    // XOR: показываю, что оно вообще способно обучаться
    std::vector<std::vector<float>> trainInput = {
        {1.0f, 1.0f},
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
    }; 
    std::vector<std::vector<float>> trainOutput = {
        {1.0f},
        {1.0f},
        {0.0f},
        {0.0f},
    };

    uint32_t epochs = 10000;
    
    //training the neural network with randomized data
    std::cout << "training start\n";

    for(uint32_t ep_num = 0; ep_num < epochs; ep_num++)
    {
        uint32_t index = rand() % 4;
        nn.feedForword(trainInput[index]);
        nn.backPropagate(trainOutput[index]);
    }

    std::cout << "training complete\n";

    for(std::vector<float> input: trainInput)
    {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        std::cout << input[0] << ", " << input[1] <<" => " << std::round(preds[0]) << std::endl;    // we can predict vector values, that's why preds[0] - in our topology it is vec of 1 elem
    }

    getQuality(trainInput, trainOutput, nn, true);
}

void squareTesting() {
    std::vector<uint32_t> topology = {1, 3, 5, 7, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.0004, "reg");

    std::vector<std::vector<float>> trainInput = {
        // {0.25},
        {1.0},
        {4.},
        {9.},
        {16.},
        {25.},
        {36.}
    };

    std::vector<std::vector<float>> trainOutput = {
        // {0.5},
        {1.},
        {2.},
        {3.},
        {4.},
        {5.},
        {6.}
    }; 

    uint32_t epochs = 15000;
    
    //training the neural network with randomized data
    std::cout << "training start\n";

    for(uint32_t ep_num = 0; ep_num < epochs; ep_num++)
    {
        uint32_t index = rand() % 4;
        nn.feedForword(trainInput[index]);
        nn.backPropagate(trainOutput[index]);
    }

    std::cout << "training complete\n";

    for(std::vector<float> input: trainInput)
    {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        std::cout << input[0] <<" => " << preds[0] << std::endl;    // we can predict vector values, that's why preds[0] - in our topology it is vec of 1 elem
    }

    getQuality(trainInput, trainOutput, nn, false);
}

// If something is not working - tell it is just simplest version possible with very stupid error
int main()
{
    // creating neural network

    xorTesting();

    squareTesting();

    // Теперь на других данных:

    // std::vector<uint32_t> topology = {2, 5, 5, 1};
    // sp::SimpleNeuralNetwork nn(topology, 1.);
    
    //sample dataset

    // std::vector<std::vector<float>> trainInput = {
    //     {5.4270585780980145, 9.144429648348328},
    //     {0.7306832812116131, -6.303622731929135},
    //     {5.185058349058192, 4.505132934036242},
    //     {1.9273194271715521, -3.162254415251489},
    //     {-7.167978859208053, -6.844930218005205}
    // }; 
    // std::vector<std::vector<float>> trainOutput = {
    //     {1.0},
    //     {-1.0},
    //     {1.0},
    //     {-1.0},
    //     {1.0}
    // };

    return 0;
}
