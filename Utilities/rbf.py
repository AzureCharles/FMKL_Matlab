import torch

def rbf(dot, sig):
    n = dot.size(0)
    K = dot / sig**2
    d = K.diag()
    K = K - torch.outer(d / 2, torch.ones(n))
    K = K - torch.outer(torch.ones(n), d / 2)
    K = torch.exp(K)
    return K

def rbf2(coord, sig):
    K = torch.exp(-2 / (2 * sig**2) * (1 - coord))
    return K

def normalizeKernel(K):
    diag_K = torch.diag(K)
    kNorm = K / torch.sqrt(torch.outer(diag_K, diag_K))
    return kNorm
    


