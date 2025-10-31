import sys
sys.path.append("../my_torch")
from my_torch.nn import *

a = Tensor([2],requires_grad=True)
b = Tensor([3],requires_grad=True)
c = Tensor([4],requires_grad=True)
d = Tensor([5],requires_grad=True)

x = a * b * c * d
y = 2 * a + b + c + d

x_leaves = x.backward()
y_leaves = y.backward()

print_partial_d(x_leaves)
print_partial_d(y_leaves)

print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
