#pragma once

#include <iostream>
#include "neural_network.hpp"
#include <vector>
#include <cstdio>
#include <string>
#include <cmath>

void xorTesting() {
    // creating neural network

    std::cout << "XOR" << std::endl;
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

void rootTesting() {
    std::cout << "root" << std::endl;
    std::vector<uint32_t> topology = {1, 3, 5, 7, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.0004, "reg");

    std::vector<std::vector<float>> trainInput = {
        // {0.25},
        {1.0},
        {4.},
        {9.},
        {16.},
        {25.},
        // {36.}
    };

    std::vector<std::vector<float>> trainOutput = {
        // {0.5},
        {1.},
        {2.},
        {3.},
        {4.},
        {5.},
        // {6.}
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

void identicalTesting() {
    std::cout << "Identical" << std::endl;
    std::vector<uint32_t> topology = {1, 3, 5, 7, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.0004, "reg");

    std::vector<std::vector<float>> trainInput = {
        // {0.25},
        {1.0},
        {2.},
        {3.},
        {4.},
        {4.5},
        {5.},
        // {36.}
    };

    std::vector<std::vector<float>> trainOutput = {
        // {0.5},
        {1.},
        {2.},
        {3.},
        {4.},
        {4.5},
        {5.},
        // {6.}
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

void linearTesting() {
    std::cout << "y=2x+1" << std::endl;
    std::vector<uint32_t> topology = {1, 3, 5, 7, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.0004, "reg");

    std::vector<std::vector<float>> trainInput = {
        // {0.25},
        {1.0},
        {2.},
        {3.},
        {4.},
        {4.5},
        {5.},
        // {36.}
    };

    std::vector<std::vector<float>> trainOutput = {
        // {0.5},
        {1. * 2. + 1.},
        {2. * 2. + 1.},
        {3. * 2.+ 1.},
        {4. * 2. + 1.},
        {4.5 * 2. + 1.},
        {5. * 2. + 1.},
        // {6.}
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

void lineSepTesting() {
    std::cout << "Separation by y = x" << std::endl;

    std::vector<uint32_t> topology = {2, 3, 5, 7, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.001, "cl");

    std::vector<std::vector<float>> trainInput = {
        // {0.25},
        {1.0, 2.0},
        {1.0, 0.0},
        {1.0, 3.0},
        {2.0, 1.0}

        // {36.}
    };

    std::vector<std::vector<float>> trainOutput = {
        // {0.5},
        {1.0},
        {0.},
        {1.0},
        {0.}
        // {6.}
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
        std::cout << input[0] << ',' << input[1] <<" => " << preds[0] << std::endl;    // we can predict vector values, that's why preds[0] - in our topology it is vec of 1 elem
    }

    getQuality(trainInput, trainOutput, nn, true);
}