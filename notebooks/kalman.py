import numpy as np 
from numpy.linalg import pinv
from numpy.linalg import det

class Kalman():
    '''Implements a 1D (observation dimension) Bayesian Kalman filter following the probabilistic approach of Murphy page ~641.  
       dim_z is the number of measurement inputs
    
    Attributes
    ----------
    mu : numpy.array(dim_mu, 1)
        State estimate vector
        
    sigma : numpy.array(dim_x, dim_x)
        Covariance matrix
    
    A : numpy.array(dim_mu, dim_mu)
        State Transition matrix
    
    B : numpy.array(dim_mu, dim_u)
        Control transition matrix
    
    C : numpy.array(dim_mu, dim_mu)
        Measurement function
    
    D : numpy.array(dim_mu, dim_u)
        Control observation matrix
    
    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix
        
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
        
    S : numpy.array(dim_z, dim_z)
        Observation Noise Estimate. For now set to R 
        
    '''
    
    def __init__(self, mu_0, sigma_0, A, B, C, D, Q, R, state_labels=None):
        '''
        dim_mu = state dimension
        dim_y  = observation dimension
        dim_u  = control dimension

        Parameters
        ----------
        mu_0 : numpy.array(dim_mu, 1)
            Initial state estimate vector

        sigma_0 : numpy.array(dim_mu, dim_mu)
            Initial covariance matrix

        A : numpy.array(dim_mu, dim_mu)
            State Transition matrix

        B : numpy.array(dim_mu, dim_u)
            Control transition matrix

        C : numpy.array(dim_y, dim_mu)
            Measurement function

        D : numpy.array(dim_mu, dim_u)
            Control observation matrix

        R : numpy.array(dim_y, dim_y)
            Measurement noise matrix

        Q : numpy.array(dim_mu, dim_mu)
            Process noise matrix

        state_labels : list(dim_mu)
            Labels the state vector by name.  Unused other than conveinience. 
        '''
        self.A = A   # Parameter matrix A 
        self.B = B   # Parameter matrix B 
        self.C = C   # Parameter matrix C 
        self.D = D   # Parameter matrix D
        self.Q = Q   # State noise covaraiance matrix 
        self.R = R   # Observation noise covariance matrix
        self.S = self.R # Observation Noise Estimate. For now set to R 
        self.mu = mu_0 # Initial state estimate 
        self.sigma = sigma_0 # Initial state covariance 
        self.state_labels = state_labels
        
    def predict(self, u=None): 
        ''' Murphy Sec. 18.3.1.1'''
        
        # Predicted state covariance 
        self.sigma = np.dot(np.dot(self.A, self.sigma), self.A.T) + self.Q
        
        # if there is no control input do not include it 
        if u is None:
            self.mu = np.dot(self.A, self.mu)  # Predict state mean 
        else:
            self.mu = np.dot(self.A, self.mu) + np.dot(self.B, self.u)

        
    def update(self, Y):
        '''
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.  Murphy Sec. 18.3.1.2
        
        Parameters
        ----------
        Y : np.array(dim_z)
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.

        '''
        
        self.y = np.dot(self.C,self.mu) # Posterior predictive mean 
        
        r = Y - self.y # residual 
        # If the residual is an NaN (observation invalid), set it to zero so there is no update
        # I.e. r=0 -> K=0 for that sensor 
        # TODO: Revisit this since it will impact log-likelihood
        r[np.isnan(r)] = 0
        nan_idx = np.where(np.isnan(r))[0]

        S = np.dot(np.dot(self.C, self.sigma), self.C.T) + self.R #         
        
        try:
            S_inverse = pinv(S)
        except: 
            return 'nan'
        K = np.dot(np.dot(self.sigma, self.C.T), S_inverse) # Kalman Gain 
        
        # Correct the state covariance and mean 
        self.mu = self.mu + np.dot(K, r)
        I_KC = np.identity(len(self.mu)) - np.dot(K,self.C)
        self.sigma = np.dot(np.dot(I_KC, self.sigma), I_KC.T) + np.dot(np.dot(K,self.R), K.T)
        
        # Update the class attribute values 
        self.K = K 
        self.S = S 

        # Gaussian log-likeliehood 
        loglikelihood = -len(r)/2.*np.log(2*np.pi)-np.log(det(S))/2.-np.dot(np.dot(r, S_inverse), r.T)/2.
        #return np.sum(r**2)

        return loglikelihood
