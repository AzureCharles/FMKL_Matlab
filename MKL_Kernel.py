# 导入pytorch库
import torch

# 初始化两个随机的张量，形状分别为(5, 3)和(4, 3)
x = torch.randn(5, 3) # x.shape = (5, 3)
y = torch.randn(4, 3) # y.shape = (4, 3)

# 打印x和y
print("x:")
print(x)
print("y:")
print(y)

# 对x和y分别使用unsqueeze()，参数分别为1和0
x_unsqueezed = x.unsqueeze(1) # x_unsqueezed.shape = (5, 1, 3)
y_unsqueezed = y.unsqueeze(0) # y_unsqueezed.shape = (1, 4, 3)

# 打印x_unsqueezed和y_unsqueezed
print("x_unsqueezed:")
print(x_unsqueezed)
print("y_unsqueezed:")
print(y_unsqueezed)

# 对x_unsqueezed和y_unsqueezed进行减法运算，利用广播机制
diff = x_unsqueezed - y_unsqueezed # diff.shape = (5, 4, 3)

# 打印diff
print("diff:")
print(diff)

