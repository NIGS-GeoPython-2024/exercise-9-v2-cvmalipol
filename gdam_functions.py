import numpy as np

def gaussian(mean, stddev, x_array):
    """
    Function used to calculate the Gaussian (normal) distribution for a given mean, standard deviation, and an array of x values.

    Parameters:
    -----
    mean: <float>
        The mean value of the distribution
    std: <float>
        The standard deviation of the distribution
    x_array: <list> or <numpy array>
        The values at which to calculate the Gaussian probabilities

    Returns:
    -----
    <numpy array>
        The calculated Gaussian probabilities for each value in the x_array
    """
    # Create an empty array to store the probability values
    probabilities = np.zeroes(len(x_array))

    # Calculate the Gaussian for each value in x_array
    for i in range(len(x_array)):
        x = x_array[i]
        probabilities[i] = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

    return probabilities

def linregress(x, y):
    # Delta function
    delta = (len(x) * np.sum(x**2)) - (np.sum(x))**2

    # Intercept formula
    A = ((np.sum(x**2) * np.sum(y)) - (np.sum(x) * np.sum(x*y))) / delta
    
    # Slope formula
    B = ((len(x) * np.sum(x * y)) - (np.sum(x) * np.sum(y))) / delta
    
    # Return intercept and slope
    return A, B

def pearson(x, y):
    """
    This is for calculating the correlation coefficient r between two variables
    ---
    Parameters:
    x: <list> or <numpy array> 
    y: <list> or <numpy arrray> 
    """
    x_val = x - np.mean(x)
    y_val = y - np.mean(y)

    # Correlation coefficient calculation
    numerator = np.sum(x_val * y_val) 
    denom = np.sqrt(np.sum(x_val**2) * np.sum(y_val**2)) 
    r = numerator / denom
    
    return r

def chi_squared(obs, exp, std):
    """
    This is for calculating the chi squared.
    ---
    Parameters: 
    obs: observed value
    exp: expected value
    std: standard deviation
    ---
    """
    summ = (obs - exp)**2 / std**2
    N = len(obs)
    chi_sq = 1/N * np.sum(summ)

    return chi_sq

