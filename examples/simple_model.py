import sys
sys.path.append(".")
import my_torch.nn as nn
import my_torch.optim as optim
import my_torch as torch
torch.Tensor.auto_name = True

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,3)
        self.fc2 = nn.Linear(3,4)
        self.fc3 = nn.Linear(4,5)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x) ; x.name = "z1"
        x = self.relu(x) ; x.name = "a1"
        x = self.fc2(x) ; x.name = "z2"
        x = self.relu(x) ; x.name = "a2"
        x = self.fc3(x) ; x.name = "z3"
        return x
    
x = torch.Tensor([[1, 2],
                  [3, 4],
                  [5, 6]]); x.name = 'x'

# y = [x1, x2, x1+x2, x1*x2, x1^2]
y = torch.Tensor([[1, 2, 3, 2, 1],      # [1, 2, 1+2, 1*2, 1^2]
                  [3, 4, 7, 12, 9],     # [3, 4, 3+4, 3*4, 3^2]
                  [5, 6, 11, 30, 25]])  # [5, 6, 5+6, 5*6, 5^2]
y.name = 'y'

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

iters = 1
for i in range(iters):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat,y)
    leaves = loss.backward()
    torch.print_partial_d(leaves)
    optimizer.step()

    if i % 1000 == 0:
        print(f"Iter {i}: Loss = {loss.item():.4f}")
print(y_hat)