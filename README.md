## Skin-cancer MNIST10000 Dataset: Training an CNN using pytorch and Baysian Optimization with optuna"

Author: Stefan Rietzinger, Date: 12.1.2025

---
### Summary
The objective of this project is to train a Convolutional Neural Network (CNN) to classify cancerous melanoma cases from the "Skin Cancer MNIST 10000" dataset without the use of pre-trained models. The **pytorch** framework was utilised to define the CNN architecture, while **optuna** was employed for hyperparameter tuning via Bayesian optimisation. The model was trained on an "NVIDIA GeForce RTX 4060" with 8GB DDR6 VRAM. 

--- 
### Files overview:


---
### Challenges
The project encountered numerous challenges, including the following:
1. **Imbalanced Dataset:** The dataset was found to be imbalanced, necessitating the implementation of techniques such as undersampling to address this issue. The *_undersampling()* method was implemented directly within the custom dataset class.

2. **Overfitting:** Furthermore, the model exhibited a tendency to overfit. To address these challenges, a range of techniques were employed, including:
    - L2-regularization
    - Dropout rate
    - varying the models complexity
    - Early Stopping
    - Data Augmentation: *random flip* and *random rotation*

3. **Limited GPU RAM:** 
Due to the restriction of GPU-RAM to 8GB, the complexity of the models was constrained, necessitating the maintenance of a lower complexity level than would have been preferable.

---
### Hyperparameter tuning: 
The library [**optuna**] (https://optuna.org/) was utilised for the training of hyperparameters. The Hyperparameter tuning process can be observed in the search.py file. 

In the context of single-objective optimisation, Optuna employs [*Bayesian optimisation*](https://en.wikipedia.org/wiki/Bayesian_optimization). This approach is expected to facilitate more expeditious and effective hyperparameter tuning in comparison to conventional methods such as "GridSearch" or "RandomSearch". The optimization process involves the simultaneous adjustment of multiple parameters, including those that govern the architecture of the CNN itself, in addition to those that regulate the regularization or learning rate.
- lr
- num_conv_layers
- optimizer
- weight_decay
- activation_func
- dropout_rate
- fc_size1
- conv_filters_2
- num_fc_layers
- use_batchnorm
- kernel_size
- conv_filters_1

The graph entitled "param_importances.png", located in the "search" folder, provides a visual representation of the Hyperparameter importances. The optimal solution was determined after 134 rounds, owing to the substantial number of tuning parameters.

It is also noteworthy that Optuna is capable of performing multi-objective optimisation through the utilisation of a genetic algorithm. [*NSGA-II algorithm*](https://pymoo.org/algorithms/moo/nsga2.html) (Non-dominated Sorting Genetic Algorithm II). It is planned that this will be used in the future for the development of a model that will maximise performance whilst minimising thel models complexity.

---
### Results:


---
### Possible Optimizations:

---
