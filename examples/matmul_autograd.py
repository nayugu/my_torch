import sys
sys.path.append("../my_torch")
import my_torch as torch
torch.Tensor.auto_name = True

w = torch.Tensor([[-1,1]], name='w', requires_grad=True)
x = torch.Tensor([[2,3],
                  [4,5]], name='x', requires_grad=True)
b = torch.Tensor([[-2,-2]], name='b', requires_grad=True)

z = (w @ x) + b ; z.name = 'z' # Comment this to test auto-naming
a = z.sigmoid() ; a.name = 'a' # Comment this to test auto-naming

print(z)
print(a)
a_leaves = a.backward()

torch.print_partial_d(a_leaves)
