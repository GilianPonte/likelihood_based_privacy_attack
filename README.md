
[![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)]()
[![PyPI version](https://img.shields.io/pypi/v/openfl)]()
[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

# Likelihood-based privacy attack.
In this repository, we open-source our privacy attack from our paper. This privacy attack allows you to measure the associated privacy risk of your (protected) data set. The privacy risk is estimated within the framework of differential privacy: 

$\hat{\varepsilon} =\max \left(\log \left(\frac{1-(1/N)-\text{FPR}}{\text{FNR}}\right), \log \left(\frac{1-(1/N)-\text{FNR}}{\text{FPR}}\right)\right),$

where _N_ is the sample size. _FPR_ is the false positive rate (or probability of membership given non-membership of the training data); _FNR_ is the false negative rate (or probability of membership given non-membership of the training data). 

## Likelihood-based privacy attack code
We provide a function that replicates our privacy attack in likelihood_privacy_attack.py. This function requires the following parameters: 

1. `seed`: a random seed for replication purposes.
2. `simulations`: the number of simulations.
3. `train`: requires the real training set.
4. `adversary`: requires the adversary set.
5. `outside_training`: requires observations that were not seen during training or adversary.
6. `protected_training`: requires the protected training data (protected version of `train`).
7. `protected_adversary`: requires the protected adversary data (protected version of `adversary`).

## Example using swapping.
We provide an example with public churn data here:
```python
# first we need to import some dependencies.
import pandas as pd
from sklearn.model_selection import train_test_split

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

# define privacy attack
def privacy_attack(seed, simulations, train, adversary, outside_training, protected_training, protected_adversary):
  import numpy as np
  from sklearn.model_selection import GridSearchCV
  from sklearn.neighbors import KernelDensity
  import math
  import random
  np.random.seed(seed)

  epsilons = np.array([]) # to store results
  
  for iter in range(0,simulations):
    random.seed(iter) # again for reprod.
    np.random.seed(iter)
    print("iteration is " + str(iter))

    # to prevent a naive model
    N = len(train)/10

    # step 1, 2 from paper
    params = {"bandwidth": np.logspace(-1, 1, 20)} # vary the bandwith
    grid_train = GridSearchCV(KernelDensity(), params, n_jobs = -1) # cross validate for bandwiths
    grid_train.fit(protected_training) # estimate pdf from train data.
    kde_train = grid_train.best_estimator_ # get best estimator

    grid = GridSearchCV(KernelDensity(), params, n_jobs = -1) # cross validate (CV)
    grid.fit(protected_adversary) # estimate pdf from adversary data
    kde_adversary = grid.best_estimator_ # get best estimator from CV

    # step 3
    density_train = kde_train.score_samples(train) # score train examples from train on pdf_train
    density_adversary = kde_adversary.score_samples(train) # score train examples from train on pdf_adversary
    TPR = sum(density_train > density_adversary)/len(density_train) # calculate TPR

    # step 4
    density_train_new = kde_train.score_samples(outside_training) # score eval_outside examples on train density
    density_adversary_new = kde_adversary.score_samples(outside_training) # score eval_outside examples on adversary density
    FPR = sum(density_train_new > density_adversary_new)/len(density_train_new) # calculate FPR
    TNR = 1 - FPR
    FNR = 1 - TPR
    epsilons = np.append(epsilons,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))) # append resulting epsilon to epsilons
    print("FPR is " + str(FPR))
    print("FNR is " + str(FNR))
    print("TPR is " + str(TPR))
    print("TNR is " + str(TNR))
    print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
  return epsilons, FPR, TNR, FNR, TPR

# apply privacy attack
privacy_attack(seed = 1, simulations = 10, train = train, adversary = adversary_training, outside_training = evaluation_outside_training,
protected_training = swap25_train, protected_adversary = swap25_adversary_training)
```

In line with our paper, we find that swapping gives infinite privacy risk.

## Citing this work
Please cite this privacy attack using: _to be determined_
