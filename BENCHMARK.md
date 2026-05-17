# Benchmark Definition

This document defines the common baseline setup for comparing activation functions in artificial neural networks.

The goal is to keep the benchmark simple and comparable: each experiment should change only the activation function used in the hidden layers, while the dataset, network architecture, training algorithm, and evaluation metrics remain fixed.

## Benchmark Problems

| Difficulty | Problem | Task | Input Features | Output Neurons | Topology | Epochs |
|---|---|---|---:|---:|---|---:|
| Easy | Two Moons | Binary classification | 2 | 1 | `2-8-1` | 100 |
| Medium | Concentric Circles | Binary classification | 2 | 1 | `2-8-8-1` | 150 |
| Hard | Crossing Spirals | Binary classification | 6 | 1 | `6-16-16-1` | 250 |

All three benchmark problems are binary classification tasks. This keeps the output layer, loss function, and evaluation procedure identical across all difficulties.

## Activation Function Placement

Activation functions are placed as follows:

| Layer | Activation Function |
|---|---|
| Input layer | None |
| Hidden layer(s) | The activation function being tested |
| Output layer | Sigmoid |

Input neurons do not use an activation function. They only pass the input features into the network.

The output layer is not part of the activation function comparison. It always uses Sigmoid because all benchmark problems are binary classification tasks.

## Activation Functions

The following activation functions are part of the benchmark set:

| Name | Formula |
|---|---|
| ReLU | `f(x) = max(0, x)` |
| GeLU | `f(x) = x * Phi(x)` or the framework's default GeLU implementation |
| Sigmoid | `f(x) = 1 / (1 + exp(-x))` |
| TanH | `f(x) = tanh(x)` |
| Swish | `f(x) = x * sigmoid(x)` |
| Identity | `f(x) = x` |

The Step function is excluded from the benchmark because it is not suitable for backpropagation. Its derivative is zero almost everywhere, so gradient-based training cannot update the network weights in a useful way.

## Baseline Experiment Rule

For the first baseline experiment, each hidden neuron in the network uses the same activation function.

Example for the Medium problem:

```text
Input layer:      2 neurons, no activation function
Hidden layer 1:   8 neurons, ReLU
Hidden layer 2:   8 neurons, ReLU
Output layer:     1 neuron, Sigmoid
```

The same network is then trained again with TanH, again with Sigmoid, again with Swish, and so on.

Only the hidden activation function changes between runs. All other settings must stay fixed.

## Training Algorithm

The common baseline training algorithm is:

```text
Backpropagation with mini-batch stochastic gradient descent
```

The optimizer should be plain SGD without momentum for the basic baseline. More advanced optimizers such as Adam may be tested later, but they are not part of the basic comparison.

### Training Parameters

| Parameter | Value |
|---|---:|
| Optimizer | Mini-batch SGD |
| Learning rate | `0.01` |
| Batch size | `32` |
| Loss function | Binary Cross Entropy |
| Weight initialization | Glorot/Xavier uniform |
| Bias initialization | `0` |
| Random seed | `42` |

The same initialization method and random seed should be used for all activation functions. This keeps the comparison focused on the activation function instead of random starting conditions.

## Input Preprocessing

Input features should be standardized using the training set statistics:

```text
x_standardized = (x - train_mean) / train_std
```

The mean and standard deviation are computed only on the training set. The same values are then applied to the test set.

Labels are kept as `0` and `1`.

## Evaluation

Each trained model is evaluated on the test set after training.

The primary metrics are:

- Final training loss
- Final test loss
- Training accuracy
- Test accuracy

For converting the Sigmoid output to a class label, use the threshold:

```text
prediction = 1 if output >= 0.5 else 0
```

## Required Result Table

Baseline results should be reported in this format:

| Problem | Activation Function | Train Loss | Test Loss | Train Accuracy | Test Accuracy |
|---|---|---:|---:|---:|---:|
| Easy | ReLU | TODO | TODO | TODO | TODO |
| Easy | GeLU | TODO | TODO | TODO | TODO |
| Easy | Sigmoid | TODO | TODO | TODO | TODO |
| Easy | TanH | TODO | TODO | TODO | TODO |
| Easy | Swish | TODO | TODO | TODO | TODO |
| Easy | Identity | TODO | TODO | TODO | TODO |

The same table structure should be used for Medium and Hard.

## Reference Implementation

The repository contains a small dependency-free reference implementation:

```bash
python3 run_benchmark.py
```

It trains all benchmark problems with all selected activation functions and writes the final metrics to:

```text
results/baseline_results.csv
```

## Baseline Scope

The basic baseline does not include:

- Different activation functions per layer
- Different activation functions per neuron
- Optimizer tuning per activation function
- Architecture tuning per activation function
- Step function experiments

These variations may be useful later, but they should only be tested after the basic baseline results are available.

## Experiment Question

The baseline experiment answers the question:

> Under identical training conditions, which activation function performs best on each benchmark problem?
