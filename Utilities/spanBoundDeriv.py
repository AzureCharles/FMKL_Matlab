import torch
import torch.optim as optim

def span_bound_deriv(model, Kn, Kd, Ytrain):
    ri, _ = Kn.size()
    ri = int(ri ** 0.5)
    Kn = Kn.view(ri, ri)
    Kd = Kd.view(ri, ri)

    sv_indicesS, IX = torch.sort(model.sv_indices)
    sv_coefS = torch.abs(model.sv_coef[IX])

    Ksv = Kn[sv_indicesS][:, sv_indicesS]
    r = Ksv.size(0)
    Ksv = Ksv + 0.001 * torch.eye(r)  # Add a small regularization to the kernel matrix
    KTsv = torch.ones(r + 1, r + 1)
    KTsv[:r, :r] = Ksv
    KTsv[-1, -1] = 0
    KTsv[-1, -1] = 0.001  # Add a small regularization to KTsv

    Kksv = Kd[sv_indicesS][:, sv_indicesS]
    Kk0sv = torch.zeros(r + 1, r + 1)
    Kk0sv[:r, :r] = Kksv

    Q = torch.zeros(r + 1, r + 1)
    Q[::r + 1] = 1. / torch.abs(sv_coefS)
    
    G = torch.zeros(r + 1, r + 1)
    G[::r + 1] = -1. / torch.pow(torch.abs(sv_coefS), 2)

    B = KTsv + Q
    Binv = torch.inverse(B)

    A = torch.inverse(KTsv)
    A = A[:-1, :-1]

    Ysv = torch.zeros(r, r)
    Ysv[::r + 1] = Ytrain[sv_indicesS]

    F = torch.diag(torch.cat((Ysv.mm(A).mm(Kksv).mm(Ysv).mm(sv_coefS), torch.zeros(1))))

    S = 1. / torch.diag(B) - torch.diag(Q)
    dSdT = torch.pow(1. / Binv.diag(), 2) * Binv.diag().mm(Kk0sv + G.mm(F)).mm(Binv.diag()) - G.mm(F).diag()

    c = 5
    d = 0
    span = torch.sum(1. / (1 + torch.exp(-c * S + d)))
    dTdT = torch.sum(torch.pow(1. / (1 + torch.exp(-c * S + d)), 2) * torch.exp(-c * S + d) * (-c * dSdT))

    return dTdT, span
