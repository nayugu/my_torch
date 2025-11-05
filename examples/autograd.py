import sys
sys.path.append("../my_torch")
import numpy as np
import my_torch as torch
from my_torch import print_partial_d
torch.Tensor.auto_name = True

a = torch.Tensor([-2*np.pi], name='a', requires_grad=True)
b = torch.Tensor([3], name='b', requires_grad=True)
c = torch.Tensor([4], name='c', requires_grad=True)
d = torch.Tensor([5], name='d', requires_grad=True)

# Feel free to change these equations to test autograd!
x = a * b * c * d
y = a + b + c + d
z = a.sin() + b.cos() + c ** d

# Comment this part to test auto-naming
x.name = 'x'
y.name = 'y'
z.name = 'z'

print(x)
print(y)
print(z)

x_leaves = x.backward()
y_leaves = y.backward()
z_leaves = z.backward()

print_partial_d(x_leaves)
print_partial_d(y_leaves)
print_partial_d(z_leaves)

print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
