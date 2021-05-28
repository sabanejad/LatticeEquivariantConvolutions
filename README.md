# 6.862 - Application of Flow-Based Sampling for Lattice Field Theories 

Quantum Chromodynamics (QCD) is the central theoretical framework for understanding the strong (nuclear) force; however, it is very difficult to solve analytically. This code package is based of Albergo et al. (  arXiv:2101.08176) and incorporates hyperparameter tuning using raytune. In addition, this checks the acceptance rate of ML-generated field configurations. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This code utilizes pytorch as well as ray 

```
import torch 
import ray
```


## Running the code 

All code are jupyter notebooks. The notebook titled "Hyperparamter Tuning" using RayTune to adjust the learning rate, and hidden layer size. The notebook titled "n_epoch_analyis" tests the acceptance rate of generated field configurations as a function of epoch number 


## Authors

* **Dimitra Pefkou, Saba Nejad, Raspberry Simpson**

## License

Open 

## Acknowledgments

* Thanks to Albergo et al. for providing the framework for this work 

