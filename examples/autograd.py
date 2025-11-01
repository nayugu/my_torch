import sys
sys.path.append("../my_torch")
from my_torch.nn import *
Tensor.auto_name = True

a = Tensor([-2*np.pi], name='a', requires_grad=True)
b = Tensor([3], name='b', requires_grad=True)
c = Tensor([4], name='c', requires_grad=True)
d = Tensor([5], name='d', requires_grad=True)

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
