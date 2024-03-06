
[![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)]()
[![PyPI version](https://img.shields.io/pypi/v/openfl)]()
[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

# Likelihood-based privacy attack.
In this repository, we open-source our privacy attack from our paper. This privacy attack allows you to measure the associated privacy risk of your (protected) data set. The privacy risk is estimated within the framework of differential privacy: 

$\hat{\varepsilon} =\max \left(\log \left(\frac{1-(1/N)-\text{FPR}}{\text{FNR}}\right), \log \left(\frac{1-(1/N)-\text{FNR}}{\text{FPR}}\right)\right),$

where _N_ is the sample size. _FPR_ is the false positive rate (or probability of membership given non-membership of the training data); _FNR_ is the false negative rate (or probability of membership given non-membership of the training data). 

## Installation
Use the following code to install this package.

```python
!pip install git+https://github.com/GilianPonte/likelihood_based_privacy_attack.git
from likelihood_based_privacy_attack import attacks
```

## Likelihood-based privacy attack code
We provide a function that replicates our privacy attack in likelihood_privacy_attack.py. This function requires the following parameters: 

1. `seed`: a random seed for replication purposes.
2. `simulations`: the number of simulations.
3. `train`: requires the real training set.
4. `adversary`: requires the adversary set.
5. `outside_training`: requires observations that were not seen during training or adversary.
6. `protected_training`: requires the protected training data (protected version of `train`).
7. `protected_adversary`: requires the protected adversary data (protected version of `adversary`).

## Tutorial using swapping in Python and R (see for R below).
We provide an example with public churn data here:
```python
# first we need to import some dependencies.
import pandas as pd
from sklearn.model_selection import train_test_split

# import privacy attack
!pip install git+https://github.com/GilianPonte/likelihood_based_privacy_attack.git
from likelihood_based_privacy_attack import attacks

# we read public churn data
url = 'https://raw.githubusercontent.com/albayraktaroglu/Datasets/master/churn.csv'
churn = pd.read_csv(url, index_col=0)

# select some variables
churn = churn.iloc[:,6:15]
samples = 300 # select the number of observations
churn = pd.DataFrame.drop_duplicates(churn) # drop duplicates

# here we create the train, adversary and outside_training set.
churn, evaluation_outside_training = train_test_split(churn, train_size = int(samples*2/3), test_size = int(samples*1/3)) 
train, adversary_training = train_test_split(churn, train_size = int(samples*1/3))

# define our data protected method. We use swapping 25% of the observations.
def swapping(percent, data):
  import random
  import numpy as np
  swap_data = data
  idx = random.randint(0,8) # pick a random variable
  variable = np.array(data.iloc[:,idx]) # select variable from data
  ix_size = int(percent * len(variable) * 0.5) # select proportion to shuffle
  ix_1 = np.random.choice(len(variable), size=ix_size, replace=False) # select rows to shuffle
  ix_2 = np.random.choice(len(variable), size=ix_size, replace=False) # select rows to shuffle
  b1 = variable[ix_1] # take rows from variable and create b
  b2 = variable[ix_2] # take rows from variable and create b

  variable[ix_2] = b1 # swap 1
  variable[ix_1] = b2 # swap 2

  swap_data.iloc[:,idx] = variable  # place variable back in original data
  return swap_data

# apply protection to train and adversary
swap25_train = swapping(percent = 0.25, data = train) # apply swapping 25% to train
swap25_adversary_training = swapping(percent = 0.25, data = adversary_training)  # apply swapping 25% to adv

# apply privacy attack
attacks.privacy_attack(seed = 1, simulations = 10, train = train, adversary = adversary_training, outside_training = evaluation_outside_training,
protected_training = swap25_train, protected_adversary = swap25_adversary_training)
```
## Now in R:
```R

# Load required libraries
library(tidyverse)
library(magrittr)
#install.packages("ks")
#install.packages("KernSmooth")
library(KernSmooth)
library(ks)

# Read public churn data
url <- 'https://raw.githubusercontent.com/albayraktaroglu/Datasets/master/churn.csv'
churn <- read.csv(url)

# Select some variables
churn <- churn[, 7:15]
samples <- 300 # Select the number of observations
churn <- churn[!duplicated(churn), ] # Drop duplicates

# Create the train, adversary, and outside_training set
set.seed(42)
train_indices <- sample(1:nrow(churn), size = samples*2/3)
train <- churn[train_indices, ]
adversary_training <- churn[-train_indices, ]

# Define the data protected method: Swapping 25% of the observations
swapping <- function(percent, data) {
  set.seed(42)
  idx <- sample(1:ncol(data), 1) # Pick a random variable
  variable <- data[, idx] # Select variable from data
  ix_size <- percent * length(variable) * 0.5 # Select proportion to shuffle
  ix_1 <- sample(seq_along(variable), size = ix_size, replace = FALSE) # Select rows to shuffle
  ix_2 <- sample(seq_along(variable), size = ix_size, replace = FALSE) # Select rows to shuffle
  b1 <- variable[ix_1] # Take rows from variable and create b
  b2 <- variable[ix_2] # Take rows from variable and create b
  variable[ix_2] <- b1 # Swap 1
  variable[ix_1] <- b2 # Swap 2
  data[, idx] <- variable  # Place variable back in original data
  return(data)
}

# Apply protection to train and adversary
swap25_train <- swapping(percent = 0.25, data = train) # Apply swapping 25% to train
swap25_adversary_training <- swapping(percent = 0.25, data = adversary_training)  # Apply swapping 25% to adv

# Define function for cross-validation to find optimal bandwidth
find_optimal_bandwidth <- function(data) {
  cv_result <- numeric()
  for (bw in seq(0.2, 1, by = 0.1)) {
    cv_result <- c(cv_result, sum(bkde(data, bandwidth = bw)$y))
  }
  optimal_bandwidth <- (which.max(cv_result) - 1) * 0.1 + 0.1
  return(optimal_bandwidth)
}

# Define privacy attack
privacy_attack <- function(seed, simulations, train, adversary, outside_training, protected_training, protected_adversary) {
  set.seed(seed)
  epsilons <- c() # To store results
  
  for (iter in 1:simulations) {
    set.seed(iter) # Again for reproducibility
    cat("iteration is ", iter, "\n")
    
    # To prevent a naive model
    N <- nrow(train) / 10
    
    # Step 1, 2, and 3 from paper
    bandwidths <- 10^seq(-1, 1, length.out = 20) # Vary the bandwidth
    
    density_train <- numeric(nrow(train)) # Initialize vector to store densities for train data
    density_adversary <- numeric(nrow(train)) # Initialize vector to store densities for adversary data
    
    # Loop over each column of the dataset
    for (i in 1:ncol(protected_training)) {
      # Perform cross-validation to find optimal bandwidth
      optimal_bandwidth <- find_optimal_bandwidth(protected_training[, i])
      
      # Estimate pdf from train data using optimal bandwidth
      kde_train <- bkde(protected_training[, i], bandwidth = optimal_bandwidth)
      kde_adversary <- bkde(protected_adversary[, i], bandwidth = optimal_bandwidth)
      
      density_train <- density_train + kde_train$y # Accumulate densities for train data
      density_adversary <- density_adversary + kde_adversary$y # Accumulate densities for adversary data
    }
    
    # Calculate average densities
    density_train <- density_train / ncol(protected_training)
    density_adversary <- density_adversary / ncol(protected_adversary)
    
    # Calculate TPR
    TPR <- sum(density_train > density_adversary) / length(density_train)
    
    # Step 5
    density_train_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside training data
    density_adversary_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside adversary data
    
    # Loop over each column of the dataset
    for (i in 1:ncol(protected_training)) {
      # Estimate pdf from outside training data using optimal bandwidth
      kde_train_new <- bkde(outside_training[, i], bandwidth = optimal_bandwidth, gridsize = 1000)
      kde_adversary_new <- bkde(outside_training[, i], bandwidth = optimal_bandwidth, gridsize = 1000)
      
      density_train_new <- density_train_new + kde_train_new$y # Accumulate densities for outside training data
      density_adversary_new <- density_adversary_new + kde_adversary_new$y # Accumulate densities for outside adversary data
    }
    
    # Calculate average densities
    density_train_new <- density_train_new / ncol(outside_training)
    density_adversary_new <- density_adversary_new / ncol(outside_training)
    
    # Calculate FPR
    FPR <- sum(density_train_new > density_adversary_new) / length(density_train_new)
    
    TNR <- 1 - FPR
    FNR <- 1 - TPR
    epsilons <- c(epsilons, max(log((1 - (1/N) - FPR) / FNR), log((1 - (1/N) - FNR) / FPR))) # Append resulting epsilon to epsilons
    cat("FPR is ", FPR, "\n")
    cat("FNR is ", FNR, "\n")
    cat("TPR is ", TPR, "\n")
    cat("TNR is ", TNR, "\n")
    cat("empirical epsilon = ", max(log((1 - (1/N) - FPR) / FNR), log((1 - (1/N) - FNR) / FPR)), "\n")
  }
  return(list(epsilons = epsilons, FPR = FPR, TNR = TNR, FNR = FNR, TPR = TPR))
}

# Apply privacy attack
privacy_attack(seed = 1, simulations = 10, train = train, adversary = adversary_training, 
               outside_training = adversary_training, protected_training = swap25_train, 
               protected_adversary = swap25_adversary_training)
```

In line with our paper, we find that swapping gives infinite privacy risk.

## Citing this work
Please cite this privacy attack using: _to be determined_
