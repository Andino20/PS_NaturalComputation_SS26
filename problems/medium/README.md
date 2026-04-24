# Medium Problem: Iris Dataset Classification
Classification task on the Iris dataset.

## Problem Description
The goal is to classify iris flowers into one of three species based on four input features:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

The three output classes are:
- 0 = setosa
- 1 = versicolor
- 2 = virginica

## Dataset
The dataset is provided as `iris_train.csv` & `iris_test.csv`.

### Columns
- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)
- label

## Suggested Network Topology
- Input layer: 4 neurons
- Hidden layer: 8 neurons
- Output layer: 3 neurons

Suggested topology: **4-8-3**

## Task Type
Multiclass classification

## Suggested Activation Functions
Choose from the defined benchmark set:
- ReLU
- Leaky ReLU
- Sigmoid
- TanH
- Swish
- Identity

## Training
The number of epochs: 150
