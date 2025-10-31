import sys
sys.path.append("../my_torch")
import my_torch as torch

w = torch.Tensor([[-1,1]], name='w', requires_grad=True)
x = torch.Tensor([[2,3],
                  [4,5]], name='x', requires_grad=True)
b = torch.Tensor([[-2,-2]], name='b', requires_grad=True)

z = (w @ x) + b ; #z.name = 'z'
a = z.sigmoid() ; #a.name = 'a'

print(z)
print(a)
a_leaves = a.backward()

torch.print_partial_d(a_leaves)
