#pragma once

#include "matrix.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>

namespace sp
{
    inline float Sigmoid(float x)
    {
        return 1.0f / (1 + exp(-x));
    }

    //derivative of activation function
    inline float DSigmoid(float x)
    {
        return (x * (1 - x));
    }

    inline float ReLU(float x)
    {
        if (x >= 0) return x;
        return 0;
    }

    // Derivative of relu
    inline float DReLU(float x)
    {
        if (x >= 0) return 1;
        return 0;
    }

    inline float DId(float x)
    {
        return 1;
    }
    // calss representing a simple densely connected neural network
    class SimpleNeuralNetwork
    {
        public:
            std::string _type;       // classification (cl) or regression (reg)
            std::string _loss;      // simple / MSE / entropy
            std::vector<uint32_t> _topology;
            std::vector<Matrix2D<float>> _weightMatrices;
            std::vector<Matrix2D<float>> _valueMatrices;
            std::vector<Matrix2D<float>> _biasMatrices;
            float _learningRate;
        public:
            
            // topology defines the no.of neurons for each layer
            // learning rate defines how much modification should be done in each backwords propagation i.e. training
            SimpleNeuralNetwork(std::vector<uint32_t> topology, float learningRate=0.1f, std::string type="cl", std::string loss="simple"):
                _topology(topology),
                _weightMatrices({}),
                _valueMatrices({}),
                _biasMatrices({}),
                _learningRate(learningRate),
                _type(type)
            {
                // initializing weight and bias matrices with random weights
                for(uint32_t i = 0; i < topology.size() - 1; i++)
                {
                    Matrix2D<float> weightMatrix(topology[i + 1], topology[i]); 
                    weightMatrix = weightMatrix.applyFunction([](const float &val){
                        return (float)rand() / RAND_MAX;
                    });
                    _weightMatrices.push_back(weightMatrix);
                    
                    Matrix2D<float> biasMatrix(topology[i + 1], 1);
                    biasMatrix = biasMatrix.applyFunction([](const float &val){
                        return (float)rand() / RAND_MAX;
                    });
                    _biasMatrices.push_back(biasMatrix);
   
                }
                _valueMatrices.resize(topology.size());
            }

            // function to generate output from given input vector
            bool feedForword(std::vector<float> input)
            {
                if(input.size() != _topology[0])
                    return false;
                // creating input matrix 
                Matrix2D<float> values(input.size(), 1);
                for(uint32_t i = 0; i < input.size(); i++)
                    values._vals[i] = input[i];
                
                //forwarding inputs to next layers
                for(uint32_t i = 0; i < _weightMatrices.size(); i++)
                {
                    // y = activationFunc( x1 * w1 + x2 * w2 + ... + b)  
                    _valueMatrices[i] = values;
                    values = values.multiply(_weightMatrices[i]);
                    values = values.add(_biasMatrices[i]);
                    if (this->_type == "cl") {
                        values = values.applyFunction(Sigmoid);
                        // _valueMatrices[_weightMatrices.size()] = values;
                    }
                    else if (i < _weightMatrices.size() - 1 && this->_type != "cl") {      // Перед аутпутом при регрессии не надо использовать релу
                        values = values.applyFunction(ReLU);
                        // _valueMatrices[_weightMatrices.size()] = values;
                    }
                }
                _valueMatrices[_weightMatrices.size()] = values;
                return true;
            }

            // function to train with given output vector
            bool backPropagate(std::vector<float> targetOutput)
            {
                if(targetOutput.size() != _topology.back())
                    return false;

                // determine the simple error:
                // error = target - output;
                Matrix2D<float> errors(targetOutput.size(), 1);
                errors._vals = targetOutput;
                // so: d(error) / dw = -value; for output layer:
                errors = errors.add(_valueMatrices.back().negative());

                // back propagating the error from output layer to input layer: reverse cycle
                // and adjusting weights of weight matrices and bias matrics
                for (int32_t i = _weightMatrices.size() - 1; i >= 0; i--)
                {
                    //calculating errrors for previous layer
                    Matrix2D<float> prevErrors = errors.multiply(_weightMatrices[i].transpose());

                    //calculating gradient i.e. delta weight (dw)
                    //dw = lr * error * d/dx(activated value)
                    if (this->_type == "cl") {
                        Matrix2D<float> dOutputs = _valueMatrices[i + 1].applyFunction(DSigmoid);
                        Matrix2D<float> gradients = errors.multiplyElements(dOutputs);
                        gradients = gradients.multiplyScaler(_learningRate);
                        Matrix2D<float> weightGradients = _valueMatrices[i].transpose().multiply(gradients);

                        //adjusting bias and weight
                        _biasMatrices[i] = _biasMatrices[i].add(gradients);
                        _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                    }
                    else if (i != _weightMatrices.size() - 1) {
                        Matrix2D<float> dOutputs = _valueMatrices[i + 1].applyFunction(DReLU);
                        Matrix2D<float> gradients = errors.multiplyElements(dOutputs);
                        gradients = gradients.multiplyScaler(_learningRate);
                        Matrix2D<float> weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                    
                        //adjusting bias and weight
                        _biasMatrices[i] = _biasMatrices[i].add(gradients);
                        _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                    }
                    else {
                        Matrix2D<float> dOutputs = _valueMatrices[i + 1].applyFunction(DId);
                        Matrix2D<float> gradients = errors.multiplyElements(dOutputs);
                        gradients = gradients.multiplyScaler(_learningRate);
                        Matrix2D<float> weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                        
                        //adjusting bias and weight
                        _biasMatrices[i] = _biasMatrices[i].add(gradients);
                        _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                    }
                    
                    errors = prevErrors;
                }
                return true;
            }
            
            // function to retrive final output
            std::vector<float> getPredictions()
            {
                return _valueMatrices.back()._vals;
            }

    }; // class SimpleNeuralNetwork

}

// function which tests the quality of the network and outputs results to the file; For classification will return accuracy; MSE for regression.
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