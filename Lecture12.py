import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import sys

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [0, 1, 0, 2, 3, 3]
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]
y_data = [1, 0, 2, 3, 3, 4]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5
hidden_size = 5
batch_size = 1
sequence_length = 6
num_layers = 1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, hidden):
        x = x.view(batch_size, sequence_length, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, num_classes)
        return hidden, out

    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    hidden = model.init_hidden()
    hidden, outputs = model(inputs, hidden)
    optimizer.zero_grad()

    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
