import numpy as np
import math

class BayesClassifier():
    def __init__(self, mode:str='QDA'):
        # the class labels (will be set upon fit)
        self.labels = []
        # model learnable parameters will be set upon fit
        self.means = []
        self.covs = []
        self.priors = []
        self.weighted_cov = None        # in case of LDA only
        
        # TODO 1: Assert that mode is one of ['QDA', 'LDA', 'Naive'] and set it to self.mode
        # assert goes here
        assert mode in ['QDA', 'LDA', 'Naive'], "Not supported"
        self.mode = mode

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.means = []
        self.covs = []
        self.priors = []
        # TODO 2: Extract the labels from y_train using Numpy
        self.labels = np.unique(y_train)
        # TODO 3: Compute the model parameters
        for label in self.labels:
            # TODO 3.1: Extract the data belonging to the current class
            x_train_given_class = x_train[y_train==label]
            # TODO 3.2: Compute the class mean and add it to self.means
            class_mean = np.mean(x_train_given_class, axis=0)
            self.means.append(class_mean)
            # TODO 3.3: Compute the class covariance using np.cov and add it to self.cov
            # Check the documentation of np.cov and what layout it assumes about the input
            # Set bias=False in case of QDA only and use a one-line if for that
            #!!! BIAS = TRUE, NOT FALSE
            # Sources:
                # LDA: Bias = true (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
                # QDA: Bias = true (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)       
            class_cov = np.cov(x_train_given_class, rowvar=False, bias=True)
            self.covs.append(class_cov)
            # TODO 3.4: Compute the class   prior and add it to self.priors
            class_prior = x_train_given_class.shape[0] / y_train.shape[0]
            self.priors.append(class_prior)
        
        # TODO 4: Convert the model parameters from lists to numpy arrays
        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.priors = np.array(self.priors)
        
        # TODO 5: In case of LDA, compute the weighted covariance matrix (self.weighted_cov)
        # In one line, sum the covariance matrices for each class weighted (multiplied) by their priors
        if self.mode == 'LDA':
            self.weighted_cov = np.sum(self.priors[:, None, None] * self.covs, axis=0)
            # reset self.covs to an empty list to save memory as we no longer need it
            self.covs = []
        # TODO 6: In case of Naive Bayes, diagonalize each of the covariance matrices to enforce independence
        elif self.mode == 'Naive':
            # diagonalize the covariance matrices in one line
            self.covs = self.covs * np.identity(self.covs.shape[1])
        # In case of QDA, we maintain the general setup and just return
        else: 
             return

    # TODO 7: Define the normal distribution N(x, μ, Σ) probability density function
    @staticmethod
    def N(x:np.ndarray, μ:np.ndarray, Σ:np.ndarray)->float:
        scale = 1 / ((2*np.pi)**(x.shape[0]/2) * np.sqrt(np.linalg.det(Σ)))                              # start by defining the denominator 
        diff = (x-μ)[:, np.newaxis] # -> make col vector
        prob = scale * np.exp(-0.5 * (diff.T @ np.linalg.inv(Σ) @ diff).item())
        return prob    
    
    # TODO 8: Given a single x, compute P(C|x) using Bayes rule for each class
    def predict_proba_x(self, x:np.ndarray)->np.ndarray:
        if self.mode != 'LDA':
            # TODO 8.1: Compute P(x|C)P(C) for each class in a numpy array in one line while using covariance matrices in self.covs
            prob_product = np.array([self.N(x, mu, cov)*prior for prior, mu, cov in zip(self.priors, self.means, self.covs)])
        else:
            # TODO 8.2: Compute P(x|C)P(C) for each class in a numpy array in one line while using the same covariance matrix self.weighted_cov
            prob_product = np.array([self.N(x, mu, self.weighted_cov)*prior for prior, mu in zip(self.priors, self.means)])
        return prob_product/np.sum(prob_product) 
    
    def predict_proba(self, x_val:np.ndarray)->np.ndarray:
        # given x_val of dimensions (m,n) apply predict_proba_x to each row (point x) to return array of probabilities (m, k)
        return np.apply_along_axis(self.predict_proba_x, axis=1, arr=x_val)
    
    
    def predict(self, x_val:np.ndarray)->np.ndarray:
        # TODO 9: Get the final predictions by computing argmax over the result from self.predict_proba
        y_pred_prob = self.predict_proba(x_val=x_val)
        y_pred_inds = np.argmax(y_pred_prob, axis=1)
        # replace each prediction with label in self.labels
        y_pred = self.labels[y_pred_inds]
        return y_pred
    
    def score(self, x_val:np.ndarray, y_val:np.ndarray)->float:
        y_pred = self.predict(x_val)
        # TODO 10: compute accuracy in one line by comparing y_val and y_pred 
        acc = float(np.mean(y_pred == y_val))
        return acc