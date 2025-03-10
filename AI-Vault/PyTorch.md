One can refer to [this link](https://www.runoob.com/pytorch/pytorch-basic.html).

Main concepts:
- 张量（Tensor）
- 自动求导（Autograd）
- 神经网络（nn.Module）
- 优化器（Optimizers）
- 设备（Device）

# Autograd 

```python
import numpy as np
import torch

x = torch.randn(2,2,requires_grad=True)
print(x)
y = x + 2
z = y * y * 3
out = z.mean()
print(out)
out.backward()
print(x.grad)
# The following is the derivative of z concerning x by hand
print(3*(x+2)/2)
# The result agrees with the x.grad 
```