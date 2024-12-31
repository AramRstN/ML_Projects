import numpy as np 
#import matplotlib.pyplot as plt

def fake_data (n=1000, noise=0.5):
    np.random.seed(42)
    X = np.linspace(-10, 10, n)
    # Ground truth line: y = 3x + 8
    slope = 3
    intercept = 8
    noise = np.random.randn(n) * noise
    y = slope * X + intercept + noise
    return X, y

# Hypothesis
def h_w(x, w):
    return w[0] + w[1] * x  

def linear_regression_closed_form(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]  
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

X, y = fake_data(n=50, noise=5.0)
w = linear_regression_closed_form(X, y)
print(f"Parameters (w): ")
print(f"w_1 = {w[1]:.2f}, w_0 = {w[0]:.2f}")

y_pred = h_w(X, w)

print(y_pred)
