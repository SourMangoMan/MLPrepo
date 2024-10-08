# Multi-Layer-Perceptron-repo

This repository keeps track of variations on a multi-layer perceptron trained on handwritten digits (MNIST dataset). I am new to deep learning and wanted to play around with the code whose origin is in Tariq Rashid's book ["Make Your Own Neural Network"](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608). The code was originally written in MATLAB for the Mathematics for Deep Learning module at Brunel University. It has since been converted to python with additional functionalities.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a05c8dd1-ce6e-4872-a612-9673aa347822" />
  <br>
  <em>Figure 1: https://www.researchgate.net/publication/361444345_Handwritten_Multi-Digit_Recognition_With_Machine_Learning</em>
</p>

The `ann1HL.py` and `ann3HL.py` files are adapted from a piece of code written in `MATLAB` for an 2nd year undergrad project. They are MLPs with one hidden layer and three hidden layers respectively trained to determine if the handwritten digit in the data could be found in a given student ID. For example, if my ID was 1112223 and a handwritten two was present, the model should return a one. If the handwritten digit were a six instead, the model should output a zero. The dataset on which these models are trained are subsets of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

The code in `annNHL.py` is my own initiative based on the code from the two previously mentioned. This features as many layers as the user would like with any number of nodes in each layer.

`backwards.py` takes `ann3HL.py` and redesigns it to output a vector of size (10, 1). The position of the highest value in the vector would be digit the model thinks it is looking at. Then we take this output and try to reconstruct the input. This is a work in progress at the moment.
