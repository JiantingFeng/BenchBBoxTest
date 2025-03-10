import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
from .base import CITest


class ConditionalRandomizationTest(CITest):
    """
    Conditional Randomization Test (CRT) for conditional independence testing.
    
    Reference: Candes et al. (2018)
    """
    
    def __init__(self, n_permutations: int = 1000, random_state: int = None):
        """
        Initialize the Conditional Randomization Test.
        
        Args:
            n_permutations: Number of permutations to use
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
    
    def test(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
             alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Perform a conditional independence test using CRT.
        
        Args:
            X: The first variable
            Y: The second variable
            Z: The conditioning variable
            alpha: Significance level
            
        Returns:
            Tuple containing:
                - Boolean indicating rejection of the null hypothesis (True if X and Y are dependent given Z)
                - p-value
        """
        # Fit a model to predict X from Z
        if X.shape[1] == 1:  # Univariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z, X.ravel())
        else:  # Multivariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z, X)
        
        # Compute the test statistic
        if Y.shape[1] == 1:  # Univariate Y
            model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_y.fit(np.hstack((X, Z)), Y.ravel())
            y_pred = model_y.predict(np.hstack((X, Z)))
            test_statistic = mean_squared_error(Y.ravel(), y_pred)
        else:  # Multivariate Y
            test_statistic = 0
            for j in range(Y.shape[1]):
                model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                model_y.fit(np.hstack((X, Z)), Y[:, j])
                y_pred = model_y.predict(np.hstack((X, Z)))
                test_statistic += mean_squared_error(Y[:, j], y_pred)
        
        # Permutation test
        permutation_statistics = []
        for _ in range(self.n_permutations):
            # Generate X_perm ~ P(X|Z)
            X_perm = self._generate_conditional_samples(X, Z, model_x)
            
            # Compute the test statistic for the permuted data
            if Y.shape[1] == 1:  # Univariate Y
                model_y_perm = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                model_y_perm.fit(np.hstack((X_perm, Z)), Y.ravel())
                y_pred_perm = model_y_perm.predict(np.hstack((X_perm, Z)))
                perm_statistic = mean_squared_error(Y.ravel(), y_pred_perm)
            else:  # Multivariate Y
                perm_statistic = 0
                for j in range(Y.shape[1]):
                    model_y_perm = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                    model_y_perm.fit(np.hstack((X_perm, Z)), Y[:, j])
                    y_pred_perm = model_y_perm.predict(np.hstack((X_perm, Z)))
                    perm_statistic += mean_squared_error(Y[:, j], y_pred_perm)
            
            permutation_statistics.append(perm_statistic)
        
        # Compute p-value
        p_value = np.mean(np.array(permutation_statistics) <= test_statistic)
        
        # Reject the null hypothesis if p-value <= alpha
        reject = p_value <= alpha
        
        return reject, p_value
    
    def _generate_conditional_samples(self, X: np.ndarray, Z: np.ndarray, 
                                     model_x) -> np.ndarray:
        """
        Generate samples from the conditional distribution P(X|Z).
        
        Args:
            X: The first variable
            Z: The conditioning variable
            model_x: Model for predicting X from Z
            
        Returns:
            Samples from P(X|Z)
        """
        # Predict X from Z
        X_pred = model_x.predict(Z)
        
        # Compute residuals
        if X.shape[1] == 1:  # Univariate X
            residuals = X.ravel() - X_pred
        else:  # Multivariate X
            residuals = X - X_pred
        
        # Permute residuals
        perm_idx = np.random.permutation(len(residuals))
        permuted_residuals = residuals[perm_idx]
        
        # Generate permuted X
        if X.shape[1] == 1:  # Univariate X
            X_perm = X_pred + permuted_residuals
            X_perm = X_perm.reshape(-1, 1)
        else:  # Multivariate X
            X_perm = X_pred + permuted_residuals
        
        return X_perm


class HoldoutRandomizationTest(CITest):
    """
    Holdout Randomization Test (HRT) for conditional independence testing.
    
    Reference: Tansey et al. (2018)
    """
    
    def __init__(self, n_permutations: int = 1000, test_size: float = 0.3, random_state: int = None):
        """
        Initialize the Holdout Randomization Test.
        
        Args:
            n_permutations: Number of permutations to use
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.test_size = test_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def test(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
             alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Perform a conditional independence test using HRT.
        
        Args:
            X: The first variable
            Y: The second variable
            Z: The conditioning variable
            alpha: Significance level
            
        Returns:
            Tuple containing:
                - Boolean indicating rejection of the null hypothesis (True if X and Y are dependent given Z)
                - p-value
        """
        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
            X, Y, Z, test_size=self.test_size, random_state=self.random_state
        )
        
        # Fit a model to predict X from Z using training data
        if X.shape[1] == 1:  # Univariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z_train, X_train.ravel())
        else:  # Multivariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z_train, X_train)
        
        # Compute the test statistic using testing data
        if Y.shape[1] == 1:  # Univariate Y
            model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_y.fit(np.hstack((X_train, Z_train)), Y_train.ravel())
            y_pred = model_y.predict(np.hstack((X_test, Z_test)))
            test_statistic = mean_squared_error(Y_test.ravel(), y_pred)
        else:  # Multivariate Y
            test_statistic = 0
            for j in range(Y.shape[1]):
                model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                model_y.fit(np.hstack((X_train, Z_train)), Y_train[:, j])
                y_pred = model_y.predict(np.hstack((X_test, Z_test)))
                test_statistic += mean_squared_error(Y_test[:, j], y_pred)
        
        # Permutation test
        permutation_statistics = []
        for _ in range(self.n_permutations):
            # Generate X_perm ~ P(X|Z) for test data
            X_perm_test = self._generate_conditional_samples(X_test, Z_test, model_x)
            
            # Compute the test statistic for the permuted data
            if Y.shape[1] == 1:  # Univariate Y
                model_y_perm = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                model_y_perm.fit(np.hstack((X_train, Z_train)), Y_train.ravel())
                y_pred_perm = model_y_perm.predict(np.hstack((X_perm_test, Z_test)))
                perm_statistic = mean_squared_error(Y_test.ravel(), y_pred_perm)
            else:  # Multivariate Y
                perm_statistic = 0
                for j in range(Y.shape[1]):
                    model_y_perm = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                    model_y_perm.fit(np.hstack((X_train, Z_train)), Y_train[:, j])
                    y_pred_perm = model_y_perm.predict(np.hstack((X_perm_test, Z_test)))
                    perm_statistic += mean_squared_error(Y_test[:, j], y_pred_perm)
            
            permutation_statistics.append(perm_statistic)
        
        # Compute p-value
        p_value = np.mean(np.array(permutation_statistics) <= test_statistic)
        
        # Reject the null hypothesis if p-value <= alpha
        reject = p_value <= alpha
        
        return reject, p_value
    
    def _generate_conditional_samples(self, X: np.ndarray, Z: np.ndarray, 
                                     model_x) -> np.ndarray:
        """
        Generate samples from the conditional distribution P(X|Z).
        
        Args:
            X: The first variable
            Z: The conditioning variable
            model_x: Model for predicting X from Z
            
        Returns:
            Samples from P(X|Z)
        """
        # Predict X from Z
        X_pred = model_x.predict(Z)
        
        # Compute residuals
        if X.shape[1] == 1:  # Univariate X
            residuals = X.ravel() - X_pred
        else:  # Multivariate X
            residuals = X - X_pred
        
        # Permute residuals
        perm_idx = np.random.permutation(len(residuals))
        permuted_residuals = residuals[perm_idx]
        
        # Generate permuted X
        if X.shape[1] == 1:  # Univariate X
            X_perm = X_pred + permuted_residuals
            X_perm = X_perm.reshape(-1, 1)
        else:  # Multivariate X
            X_perm = X_pred + permuted_residuals
        
        return X_perm


class ProjectedCovarianceTest(CITest):
    """
    Projected Covariance Measure for conditional independence testing.
    
    Reference: Shah and Peters (2020)
    """
    
    def __init__(self, n_permutations: int = 1000, random_state: int = None):
        """
        Initialize the Projected Covariance Test.
        
        Args:
            n_permutations: Number of permutations to use
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
    
    def test(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
             alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Perform a conditional independence test using the Projected Covariance Measure.
        
        Args:
            X: The first variable
            Y: The second variable
            Z: The conditioning variable
            alpha: Significance level
            
        Returns:
            Tuple containing:
                - Boolean indicating rejection of the null hypothesis (True if X and Y are dependent given Z)
                - p-value
        """
        # Fit models to predict X and Y from Z
        if X.shape[1] == 1:  # Univariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z, X.ravel())
            X_pred = model_x.predict(Z).reshape(-1, 1)
        else:  # Multivariate X
            model_x = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_x.fit(Z, X)
            X_pred = model_x.predict(Z)
        
        if Y.shape[1] == 1:  # Univariate Y
            model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_y.fit(Z, Y.ravel())
            Y_pred = model_y.predict(Z).reshape(-1, 1)
        else:  # Multivariate Y
            model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model_y.fit(Z, Y)
            Y_pred = model_y.predict(Z)
        
        # Compute residuals
        X_res = X - X_pred
        Y_res = Y - Y_pred
        
        # Compute test statistic (projected covariance)
        test_statistic = np.sum(X_res * Y_res) / np.sqrt(np.sum(X_res**2) * np.sum(Y_res**2))
        
        # Permutation test
        permutation_statistics = []
        for _ in range(self.n_permutations):
            # Permute residuals
            perm_idx = np.random.permutation(len(X_res))
            X_res_perm = X_res[perm_idx]
            
            # Compute permuted test statistic
            perm_statistic = np.sum(X_res_perm * Y_res) / np.sqrt(np.sum(X_res_perm**2) * np.sum(Y_res**2))
            permutation_statistics.append(perm_statistic)
        
        # Compute p-value (two-sided test)
        p_value = np.mean(np.abs(permutation_statistics) >= np.abs(test_statistic))
        
        # Reject the null hypothesis if p-value <= alpha
        reject = p_value <= alpha
        
        return reject, p_value 