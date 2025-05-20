import torch
from torch import nn

print('PyTorch version:', torch.__version__)

x = torch.randn(4, 8, requires_grad=True)
ln = nn.LayerNorm(8, elementwise_affine=True, bias=False)
y = ln(x)
print('Input requires_grad:', x.requires_grad)
print('Output requires_grad:', y.requires_grad)
y.sum().backward()
print('Input grad is None:', x.grad is None)
print('Input grad:', x.grad)
