# Easy Problem: Concentric Circles Classification
Classification task on two concentric circles.

## Problem Description
The goal is to classify each point into one of two classes based on two input features:

- x
- y

The two output classes are:
- 0: outer circle
- 1: inner circle

## Dataset
The dataset is provided as `concentric_circles_train.csv` & `concentric_circles_test.csv`.

### Columns
- x
- y
- label

## Suggested Network Topology
- Input layer: 2 neurons
- Hidden layer: 8 neurons
- Output layer: 1 neuron

Suggested topology: **2-8-1**

## Task Type
Binary Classification

## Suggested Activation Functions
Choose from the defined benchmark set:
- ReLU
- GeLU
- Sigmoid
- TanH
- Swish
- Identity

## Training
The number of epochs: *100* 
