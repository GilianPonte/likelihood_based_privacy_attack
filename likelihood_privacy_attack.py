def privacy_attack(seed, simulations, train, adversary, outside_training, protected_training, protected_adversary):
  import numpy as np
  from sklearn.model_selection import GridSearchCV
  from sklearn.neighbors import KernelDensity
  import math
  import random
  np.random.seed(seed)
  # to store results
  epsilons = np.array([])
  
  
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
