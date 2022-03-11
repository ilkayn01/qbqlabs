# Perceptron Model - Neural Networks

# Introduction
Invented in 1957 by Frank Rosenblatt at the Cornell Aeronautical Laboratory, a perceptron is the simplest neural network possible: a computational model of a single neuron. A perceptron consists of one or more inputs, a processor, and a single output.

It has 3 units:

1.   Input (Sensory)
2.   Hidden (Associator Unit)
3.   Output (Response Unit)

Pointers:

* Weight updation between hidden and output layer
* Checks for error between hidden and output layer
* Error = target - calculated
* Incase of error, weights are adjusted

# Training Algorithm

Step 0: initialize weights, bias, learning rate (between 0 and 1)

Step 1: Check for stopping condition

Step 2: Perform steps 3-5 for binary training vector s:t

Step 3: Apply activation function to input layer

Step 4: Calculate output response Yin

Step 5: Make adjustments in weights and bias

> wi (new) = wi (old) + αtxi

> b (new) = b(old) + αt

where α = learning rate;

      t = target (1 or -1);
      
      b = bias

Step 6: Check for stopping condition. If there is no change in weights, stop the training process.