## Perceptron and Winnow algorithm
The simplest kind of neural network is a single-layer perceptron network.
<br />This repository implements Single-Layer Vanilla (basic) Perceptron, Averaged Perceptron, and Winnow algorithm from the scratch without using scikit-learn (sklearn) or pandas.
<br />The given scripts classify the images of handwriten digits (0 ~ 9; 28x28 pixels).
<br />10 Perceptrons (neurons) are trained that will combinedly learn to classify the handwritten digits.
<br />Each Perceptron has 785 inputs (28x28 + 1 for bias) and 1 output, and each Perceptron's target is one of the ten digits.


## How to run the scripts

### 1. Vanilla Perceptron
python vanilla_perceptron.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />e.g., python vanilla_perceptron.py 9000 40 0.0001 data

### 2. Averaged Perceptron
python averaged_perceptron.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />e.g., python averaged_perceptron.py 3000 100 0.0001 data

### 3. Winnow algorithm
python winnow.py [number of training samples] [number of epoch] [learning rate] [path to data folder]
<br />python winnow.py 1000 20 1.000001 data
<br />*note: learning rate is defined as the number multiplied/divided to update weights


## Dataset
MNIST: 60k training examples and 10k test examples.
<br />Dataset available at http://yann.lecun.com/exdb/mnist/.
<br />Each digit is represented as a 28x28 dimensional vector, each cell having a value in range [0, 255].
<br />In the script, each value is scaled to be in range [0, 1].
