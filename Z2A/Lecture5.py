import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class Model(torch.nn.Module):
    def __init__(self):
        """
        두개의 nn.Linear module 초기화
        """
        super(Model, self).__init__()
        # 입력 하나 출력 하나
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        입력을 받아 결과값 출력
        생성자에서 정의한 module을 사용
        :param x: 입력값 x
        :return:결과값 y
        """
        y_pred = self.linear(x)
        return y_pred


# 모델 생성
model = Model()
# loss function과 optimizer 지정
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습
for epoch in range(500):
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
hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])
