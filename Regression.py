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

def poly_features(X, degree):
    X_poly = np.c_[np.ones(len(X))]
    for i in range(1, degree + 1):
        X_poly = np.c_[X_poly, X**i]
    return X_poly

def polynomial_regression(X,y, degree):
    X_poly = poly_features(X, degree)
    w = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return w

# Linear Regression
X, y = fake_data(n=50, noise=5.0)
w = linear_regression_closed_form(X, y)
print(f"Parameters (w): ")
print(f"w_1 = {w[1]:.2f}, w_0 = {w[0]:.2f}")

y_pred = h_w(X, w)

print(y_pred)


#Polynimial Regression

degree = 3
weights = polynomial_regression(X, y, degree)

X_fit = np.linspace(X.min(), X.max(), 200)
X_fit_poly = poly_features(X_fit, degree)
y_poly_pred = X_fit_poly.dot(weights)
