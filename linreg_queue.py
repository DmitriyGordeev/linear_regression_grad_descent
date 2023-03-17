from collections import deque
import numpy as np
import pandas as pd

""" This class performs linear regression while accepting new values on the go.
Suitable for real-time application, when we need to find k and b params in a formula y = kx + b
while accepting new incoming values.
"""

class LinRegQueue:
    def __init__(self, learning_rate, subsample_size):
        self.lr = learning_rate
        self.subsample_size = subsample_size
        self.k = 0.0                # y = kx + b
        self.b = 0.0                # y = kx + b
        self.min_error = 0.0        # min error, if reached we stop param search
        self.deque_x = deque(maxlen=100)
        self.deque_y = deque(maxlen=100)
        self.indices = np.arange(0, subsample_size)


    def grad_component(self, x, y):
        grad_k = -2 * x * (y - self.k * x - self.b)
        grad_b = -2 * (y - self.k * x - self.b)
        return grad_k, grad_b


    def update(self, new_x, new_y):
        """ Performs linear regression in a real-time while processing incoming values """
        self.deque_x.append(new_x)
        self.deque_y.append(new_y)

        # this just indicates that we found 'k' and 'b' earlier and
        # need to ignore further calculation
        if self.min_error < 0.0:
            return

        x_sub = self.deque_x
        y_sub = self.deque_y

        if len(self.deque_x) > self.subsample_size:
            # select n=subsample_size random points from x and y arrays to calculate k and b
            random_indices = np.random.choice(self.indices, self.subsample_size)
            x_sub = np.take(self.deque_x, random_indices)
            y_sub = np.take(self.deque_y, random_indices)

        error, grad_k, grad_b = self.forward(x_sub, y_sub)

        if error < self.min_error:
            print (f"error {error} reached min_error {self.min_error}")
            self.min_error = -1.0

        # recalculate k and b
        self.k = self.k - self.lr * grad_k
        self.b = self.b - self.lr * grad_b
        print (f"error = {error}, new k = {self.k}, new b = {self.b}")



    def forward(self, x_values, y_values):
        """ Calculates gradients for k and b in y = kx + b formula """
        assert len(x_values) == len(y_values)
        error = 0
        gk = 0
        gb = 0
        for j in range(len(x_values)):
            y_est = self.k * x_values[j] + self.b
            error += (y_est - y_values[j]) * (y_est - y_values[j])
            grad_k, grad_b = self.grad_component(x_values[j], y_values[j])
            gk += grad_k
            gb += grad_b
        return error, gk, gb



if __name__ == "__main__":

    df = pd.read_csv("xy_points/data.csv", header=None)

    reg_queue = LinRegQueue(learning_rate=0.001, subsample_size=10)

    # example of accepting incoming values to do regression at the same time,
    # instead of this loop we can accept values by network or from database to find 'k' and 'b' immediately
    for t in df.itertuples():
        x = t[1]
        y = t[2]
        reg_queue.update(x, y)
