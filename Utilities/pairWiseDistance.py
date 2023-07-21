import torch

def PairWiseDistance(dot):
    n = dot.size(0)
    Md = torch.diag(dot)
    Dis = Md.unsqueeze(0) + Md.unsqueeze(1) - 2 * dot
    Dis = torch.sqrt(Dis)
    return Dis

def DetermineSig(dot):
    Dis = PairWiseDistance(dot)
    xDis = torch.triu(Dis)
    xDis = xDis.view(-1)
    xDis = xDis[xDis != 0]
    sig = torch.median(xDis)
    if torch.isnan(sig):
        sig = torch.tensor(1.0)
    return sig
