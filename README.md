# Perceptron Model

This project implements a simple Perceptron Model in Python using Numpy and Matplotlib. The Perceptron is a type of linear classifier, and it makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

## Prerequisites

- Python 3
- Numpy Library
- Matplotlib Library

## Installation

To install the required libraries, run:

```bash
pip install numpy matplotlib
```

## Usage

To run the program, navigate to the directory containing perceptron.py and use the following command:

``` bash

python3 perceptron.py

```

## Code Overview

The code generates random x and y coordinates for points, classifies them based on their position relative to a linear decision boundary, and then implements and trains a Perceptron model to classify these points.

### Main Components
##### Data Generation and Classification:

- Generates random points and classifies them as either 1 or -1 based on their position relative to a linear boundary defined by the equation  𝑦 = 𝑎𝑥 + 𝑏.

#### Perceptron Class:

- Initializes the Perceptron model with default parameters.
- Implements the activation function (step function).
- Updates weights and bias based on the error.
#### Example
Below is a brief snippet showing the initialization of the Perceptron model:
```python

# Initialize the Perceptron model
model = Perceptron(X, classes, learning_rate=0.0001, epochs=1000)

# Train the model
model.train()
```
## Visualization
The program also includes a visualization of the decision boundary and the data points. This is done using Matplotlib's animation module to show how the decision boundary changes over the epochs.

# Author
This program is developed by Uma Maheshwari Banna.