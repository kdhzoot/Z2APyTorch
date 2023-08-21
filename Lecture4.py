import torch
from torch.autograd import Variable

# automatically rebuild computational graph
# compute gradient
# just use Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# I will use gradient
w = Variable(torch.Tensor([1.0]), requires_grad=True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)


# x and y are fixed data.
# because we used w in forward func, loss and forward is automatically computational graph
print("predict (before training)", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        # l is compu~ graph. so we can use backward function
        # compute gradient of all variable (in this case w)
        l.backward()

        # diff between data and data[0]
        # data : [1, 2, 3]
        # data[0] : [1]
        # in this code, we have one w value
        # however, if w is not one, such as x1 * w1 + x2 * w2 = y
        # we need to use array to hold both w1 & w2 in one variable
        # that's why we use data or data[0]

        print("\tgrad: ", x_val, y_val, w.grad.data[0])

        # update gradient
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()

    print("progress: ", epoch, l.data[0])

print("predict (after training", 4, forward(4).data[0])


"""
# automatically builds computational graph
x = Variable(torch.randn(1, 10))
prev_h = Variable(torch.randn(1, 20))
W_h = Variable(torch.randn(20, 20))
W_x = Variable(torch.randn(20, 10))

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h
"""
