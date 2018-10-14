# Perceptron and Winnow algorithm
Single-Layer Vanilla (basic) Perceptron, Averaged Perceptron, and Winnow algorithm.

The given scripts use MNIST dataset (handwritten digit in 28x28 pixelsr) to predict the digit (0 ~ 9) for given images.

Since there are 10 classes (0 ~ 9), 10 Perceptrons (neurons) are trained for each digit.


# Examples to run each script

## 1. Vanilla Perceptron
<br />python vanilla_perceptron.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />e.g., python vanilla_perceptron.py 9000 40 0.0001 data

## 2. Averaged Perceptron
<br />python averaged_perceptron.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />e.g., python averaged_perceptron.py 3000 100 0.0001 data

## 3. Winnow algorithm
<br />python winnow.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />python winnow.py 1000 20 1.000001 data
<br />*note: learning rate is defined as the number multiplied/divided to update weights
