import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x, y):
    return 2 * x * (x * w - y)


w = 1.0
learning_rate = 0.01
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_list = []
mse_list = []

print("predict (before training)", 4, round(forward(4), 2))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - learning_rate * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", w, "loss=", l)

print("predict (after training)", "4 hours", round(forward(4)))
