#' Function to perform the privacy attack
#'
#' This function takes protected and privacy-sensitive data as input
#'
#' @param seed The seed for replication purposes.
#' @param simulations The number of simulations.
#' @param train The unprotected training set.
#' @param adversary The unprotected adversary set.
#' @param outside_training The new incoming samples.
#' @param protected_training The protected training set.
#' @param protected_adversary The protected adversary set.
#' @return list of epsilons, FPR, TNR, FNR, TPR
#' @examples See https://github.com/GilianPonte/likelihood_based_privacy_attack/blob/main/README.md

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
