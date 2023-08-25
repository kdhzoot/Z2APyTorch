import torch
import torch.nn.functional as F
from torch.autograd import Variable

x_data = Variable(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.tensor([[0.], [0.], [1.], [1.]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1000):
    # forward pass : compute y
    y_pred = model(x_data)

    # loss 계산 및 출력
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    # grad 0으로 초기화하고, 다시 계산, weight 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 결과
hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model.forward(hour_var).data[0][0] > 0.5)

hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hour ", 7.0, model.forward(hour_var).data[0][0] > 0.5)