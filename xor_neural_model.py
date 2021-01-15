#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:31:27 2021

"""

# Implement a toy neural network model, following Chapter 5 in NMT book (Koehn)
# This neural network model computes XOR.

import math
import numpy as np


# Defining the weight matrices and bias units.
W = np.array([[3,4],[2,3]])
b = np.array([-2,-4])
W2 = np.array([5,-5])
b2 = np.array([-2])

# Defining the activation function sigmoid, which is simply
# 1/(1+e^(-x))

@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@np.vectorize
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x)) # Look up my notes for an explanation of this formula.

# Define input and output vectors
# XOR truth table:
# 1 0 1
# 0 0 0
# 0 1 1
# 1 1 0
    
x = np.array([1,0])
t = np.array([1])

# Forward computation (inference, predicting y by linear combination of input values, weights,
# and the activation function)

s = W.dot(x) + b   # This is s = Wx + b
h = sigmoid(s)      # hidden layer

z = W2.dot(h) + b2    # Output layer
y = sigmoid(z)    # Predicting y
#print(y) [0.7425526]

# Backward computation: training by back propagation, ie. computing the error term
# of the output layer and propagate it to the other layers.

error = 1/2 * (t - y)**2
mu = 1 # learning rate

# Computing the error terms for the output layer.
delta_2 = (t - y) * sigmoid_derivative(z)
delta_W2 = mu * delta_2 * h
delta_b2 = mu * delta_2

# Back-progagating the error term (delta_2) to inner layers
delta_1 = delta_2 * W2 * sigmoid_derivative(s)
delta_W = mu * np.array([delta_1]).T * x
delta_b = mu * delta_1

# Backward computation (same as above but using chain rule explicitly)
# Computing the derivative of error with respect to y.
d_error_d_y = t - y
d_y_d_z = sigmoid_derivative(z)
d_error_d_z = d_error_d_y * d_y_d_z # chain rule

d_z_d_W2 = h
d_error_d_W2 = d_error_d_z * d_z_d_W2 # chain rule # This is the same as delta_2
print(delta_W2) # [0.03597961 0.00586666]
print(d_error_d_W2) # [0.03597961 0.00586666]

# More analogous calculations
d_z_d_b2 = 1
d_error_d_b2 = d_error_d_z * d_z_d_b2

d_z_d_h = W2
d_error_d_h = d_error_d_z * d_z_d_h

d_s_d_h = sigmoid_derivative(s)
d_error_d_s = d_error_d_h * d_s_d_h

d_W_d_s = x
d_error_d_W = np.array([d_error_d_s]).T * d_W_d_s
print(delta_W)
print(d_error_d_W)

d_b_d_s = 1
d_error_d_b = d_error_d_s * d_b_d_s



