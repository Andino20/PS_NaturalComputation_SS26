# Hard Problem - Breast Cancer Wisconsin Classification

Classification task on the Breast Cancer Wisconsin dataset.

## Problem Description
The goal is to classify tumor samples into one of two classes based on 30 input features.

The two output classes are:
- 0 = malignant
- 1 = benign

This dataset represents a more realistic and more complex classification task than the easy and medium benchmark problems.

## Dataset
The dataset is provided as `breast_cancer.csv`.

### Columns
The CSV file contains 30 input features and 1 label column.

Examples of input features include:
- mean radius
- mean texture
- mean perimeter
- mean area
- mean smoothness
- and other related measurements

The final column is:
- label

## Suggested Network Topology
- Input layer: 30 neurons
- Hidden layer 1: 16 neurons
- Hidden layer 2: 8 neurons
- Output layer: 1 neuron

Suggested topology: **30-16-8-1**

## Task Type
Binary classification

## Suggested Activation Functions
Choose from the defined benchmark set:
- ReLU
- Leaky ReLU
- Sigmoid
- TanH
- Swish
- Identity

## Training
The number of epochs: *TODO*

## Notes
- The label meanings should be documented clearly:
    - 0 = malignant
    - 1 = benign