import numpy as np
from numpy.linalg import det, inv

class KDEstimator():
    def __init__(self, bump:str='Gauss', bandwidth:str='Silverman') -> None:
        # TODO 0: Set the hyperparameters from input
        self.bump = bump
        self.bandwidth = bandwidth
        
        # set during fit
        self.M, self.N = None, None         # number of observations and features in each
        self.x_train = None                 # store training data to use in inference
        self.h = None                       # the actual value of the bandwidth

        # validation
        assert bump in ['Gauss', 'Rect'], f"Only Gauss and Rect bumps are supported but you passed {bump} to KDEstimator"
        
        # TODO 1: Assert that bandwidth is either 'Silverman' or 'Scott' or an instance of int or float
        bandwidth_is_valid = bandwidth in ['Silverman', 'Scott'] or type(bandwidth) in [int, float]
        assert bandwidth_is_valid, f"bandwidth must be either 'Silverman' or 'Scott' or a number but you passed {bandwidth} to KDEstimator"
        
        
    def fit(self, x_train):
        # TODO 2: Set M and N and x_train
        self.M, self.N = x_train.shape
        self.x_train = x_train
        
        M, N = self.M, self.N               # to avoid corrupting eqns with self
        
        # TODO 3: compute avg_σ as defined earlier
        avg_σ = np.mean(np.std(x_train, axis=0))
        
        # TODO 4: Set self.h in case of both Silverman and Scott
        if self.bandwidth == 'Silverman':
            self.h = avg_σ * (4/(M*(N+2))) ** (1/(N+4))
        elif self.bandwidth =='Scott':
            self.h = avg_σ * M ** (-1/(N+4))
        else:                           
            # it must be an int or float in this case so we use it directly
            self.h = self.bandwidth

        return self
    
    def g(self, x):
        if self.bump == 'Gauss':
            N = self.N
            π = np.pi
            # TODO 5: Implement the Gaussian bump while using einsum for vectorization
            scale = (2*π) ** (N/2)
            e = np.exp(-0.5 * np.einsum('ij,ij->i', x, x))
            return e/scale
        
        elif self.bump == 'Rect':
            # TODO 6: Implement the Rectangular bump
            return np.where(np.all(np.abs(x) <= 0.5, axis=1), 1, 0)
    
    def ϕ(self, x):
        h, N = self.h, self.N
        # TODO 7: Implement ϕ as defined earlier 
        return self.g(x/h) / (h ** N)
    
    def P(self, x):
        scale = 1/(self.M)
        xₘ = self.x_train
        # TODO 8: Implement P as defined earlier; remember no for loops allowed for this file
        # to subtract all combinations of test and training points
        # x.shape = (N) -> (1, N)
        # x_train.shape = (M, N)
        # (1, N) - (M, N) -broadcast-> (M, N) -> sum.axis=1
        return np.sum(self.ϕ(x[None, :]-self.x_train),axis=0) * scale 
    
    def transform(self, x_val):
        # TODO 9: Apply P to each row of (m,n) matrix x_val using np.apply_along_axis
        # if x_val is 1D apply P to x_val directly (same line of code)
        return self.P(x_val) if x_val.ndim == 1 else np.apply_along_axis(self.P, axis=1, arr=x_val)
    
    def fit_transform(self, x_data):
        # fit on x_data then transform
        return self.fit(x_data).transform(x_data)