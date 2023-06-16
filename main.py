import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f"device: {device}" )

import numpy as np

def adam_optimizer(initial_point, learning_rate=1, num_iterations=1000):
    rho1 = 0.9
    rho2 = 0.999
    delta = 1e-8

    # Initialize variables
    theta = np.array(initial_point, dtype=np.float64)
    s = np.zeros_like(theta)
    r = np.zeros_like(theta)
    t = 0

    for _ in range(num_iterations):
        t += 1
        gradient = compute_gradient(theta)

        s = rho1 * s + (1 - rho1) * gradient
        r = rho2 * r + (1 - rho2) * (gradient ** 2)

        m_hat = s / (1 - rho1 ** t)
        v_hat = r / (1 - rho2 ** t)

        update = -learning_rate * m_hat / (np.sqrt(v_hat) + delta)
        theta += update

    return theta

def compute_gradient(point):
    x1, x2 = point[0], point[1]
    gradient = np.array([0.4 * x1, 20 * x2])

    return gradient

# Test the Adam optimizer on the "Dachrinne" function
initial_point = [10, 1]
solution = adam_optimizer(initial_point)

print("Optimized solution:", solution)

