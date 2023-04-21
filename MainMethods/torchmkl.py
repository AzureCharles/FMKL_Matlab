# A module converted from the matlab source:
# https://github.com/ericstrobl/deepMKL
import numpy as np
import torch

def normalizeKernel(K):
    if not torch.is_tensor(K):
    # convert np matrix to tensor
        K = torch.from_numpy(K)
    
    # get the diagonal elements of K
    D = torch.diag(K)
    # create a diagonal matrix from D
    D = torch.diag_embed(D)
    # normalize K by D*D
    
    #D = np.diag(K)
    #return K/np.sqrt(D*D[:, None])
    return K/torch.sqrt(D*D.permute(0, 2, 1))
    

# K = np.array([[1,2],[2,2]]) # 用你的矩阵数据替换
# D = np.diag(K) # 提取K的对角线元素
# M = D * D[:, None] # 将D与它的转置相乘，得到一个对角矩阵
# K / M # 用M的每个元素去除K的每一列

def rbf_kernel(tensor,sig=1):
    n=tensor.shape[1]
    K=tensor/sig^2
    d=torch.diag(K)
    K=K-torch.ones((n, 1))*d.t()/2-d*torch.ones((1, n))/2
    
    return torch.exp(K)
    
def rbf_kernel2(coord,sig):
    #function K = rbf2()
    return torch.exp(-2/2 * sig**2 * (1-coord))


    
def DetermineSig(dot):
    # 假设PairWiseDistance是一个已经定义好的函数，它接受一个pytorch张量dot，返回一个pytorch张量Dis
    Dis = PairWiseDistance(dot)
    # 取Dis的上三角部分，得到一个张量xDis
    xDis = torch.triu(Dis)
    # 把xDis转换成一个一维张量，并赋值给自己
    xDis = xDis.flatten()
    # 把xDis中等于0的元素删除，得到一个只包含非零距离的一维张量
    xDis = xDis[xDis != 0]
    # 计算xDis的中位数，并赋值给sig
    sig = torch.median(xDis)
    # 如果sig是一个NaN值，就把它替换成1
    if torch.isnan(sig):
        sig = 1
    # 返回sig作为函数的输出
    return sig
