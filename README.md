# 6.862 - Application of Flow-Based Sampling for Lattice Field Theories 

In lattice field theories, the different field configurations that can occur are distributed according to the exponential of the physical action. One of the main computational costs for the calculation of observables is the generation of these field configurations, since the probability distributions are very difficult to sample from. It is possible to use machine learning flow algorithms to train a simple prior to resemble the theoretical probability distribution, as shown by Albergo et al (arXiv:1904.12072) for a phi^2 scalar field theory in a 2-dimensional lattice.

This code package is based on arXiv:2101.08176 and uses convolutional neural networks to train such flow algorithms for 2D scalar field theories. Additionally, it gives the option to use CNNs with kernels forced to be rotationally invariant, as well as equivariant CNNs (ENNs) 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This code uses pytorch as well as e2cnn (quva-lab.github.io/e2cnn/) 

```
import torch 
import e2cnn
```


## Running the code 

There are three sets of scripts, one designed to work with a normal CNN, one with a rotationally invariant CNN kernel and one with ENNs.

For a search over hyperparameter space for ENN (and equivalently for the other two options), run "./hyperoptimization_for_ENN.py L", where L is an integer corresponding to the lattice size.

To train the ENN model (and equivalently for the other two options), run "./train_ENN.py L N", where if N=0 the hyperparameters that minimize the loss are used, whil if N=1 those that maximize the Monte Carlo Markov chain acceptance rate are used.


## Authors

* **Dimitra Pefkou, Saba Nejad, Raspberry Simpson**

## License

Open 

## Acknowledgments

* Thanks to Albergo et al. for providing the framework for this work 

