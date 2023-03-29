# Linear regression
Simple example of linear regression with gradient descent algorithm
Samples some amount of data on each iteration (idea is similar to stochastic grad descent) and calculates gradients for k and b (y = kx + b)
to find new estimations for k and b.

Also contains a class (LinRegQueue) to perform search of k and b on the event-basis (when we have new data incoming in runtime)
Useful for event-driven architecture
