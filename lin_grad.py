import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import math


def grad_component(x, y, k, b):
    grad_k = -2 * x * (y - k * x - b)
    grad_b = -2 * (y - k * x - b)
    return grad_k, grad_b


def main():
    df = pd.read_csv("data/data.csv", header=None)

    itr = 3
    sample_size = int(df.shape[0] * 0.05)
    print("sample_size =", sample_size)

    k = 0
    b = 0
    lr = 0.00005
    min_error = 0.01
    x_min = -10
    x_max = 10

    x = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    idx = np.arange(0, len(x))

    for i in range(itr):
        r_indices = np.random.choice(idx, sample_size)
        x_sub = np.take(x, r_indices)
        y_sub = np.take(y, r_indices)

        error = 0
        gK = 0
        gB = 0
        for j in range(len(x_sub)):
            y_est = k * x_sub[j] + b
            error += (y_est - y_sub[j]) * (y_est - y_sub[j])
            grad_k, grad_b = grad_component(x_sub[j], y_sub[j], k, b)
            gK += grad_k
            gB += grad_b

        # if error is low enough stop
        error = math.sqrt(error)
        print(error)
        if error <= min_error and i > 5:
            break

        # recalculate k and b
        k = k - lr * gK
        b = b - lr * gB

        # plot current step
        line_y1 = k * x_min + b
        line_y2 = k * x_max + b

        fig = plot.figure(figsize=(16, 9))
        plot.plot(x_sub, y_sub, 'b.')
        plot.plot(x_sub, y_sub, 'b.')
        plot.plot([x_min, x_max], [line_y1, line_y2], 'r')
        plot.plot([x_min, x_max], [line_y1, line_y2], 'r.')
        plot.grid()
        fig.savefig(f"charts/plot_{i}.png", dpi=fig.dpi)

    # print and plot results
    print("final k", k)
    print("final b", b)


if __name__ == "__main__":
    main()
