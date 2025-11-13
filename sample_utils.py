import numpy as np
from scipy.optimize import fsolve

def geometric_dist(v, N=8):
    """
    Generate an N-element geometric series summing to 1,
    starting with initial value v (before reversing), and
    reversed so that v is at the end.
    
    Returns:
        np.array: series of length N summing to 1
    """
    # Solve for common ratio r
    def equation(r):
        return v * (1 - r**N) / (1 - r) - 1
    
    r_initial_guess = 0.5
    r_solution = fsolve(equation, r_initial_guess)[0]
    
    # Construct series
    series = np.array([v * r_solution**n for n in range(N)])
    
    # Normalize to sum to 1 (safety)
    series /= series.sum()
    
    return series
