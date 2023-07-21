import torch

def two_layer_deriv(Kf, K, betas2, sig):
    r, _ = Kf.size()
    r = int(torch.sqrt(r))
    Kf = Kf.view(r, r)
    K = K.view(r, r)

    poly_deriv = 2 * (Kf + 1) * K
    poly_deriv2 = 3 * ((Kf + 1) ** 2) * K
    rbf_deriv = torch.exp(-2 / (2 * sig**2) * (1 - Kf)) * (2 / (2 * sig**2) * K)
    lin_deriv = Kf

    answ = (
        betas2[0] * normalize_kernel_grad(Kf, poly_deriv)
        + betas2[1] * normalize_kernel_grad(Kf, poly_deriv2)
        + betas2[2] * normalize_kernel_grad(Kf, rbf_deriv)
        + betas2[3] * normalize_kernel_grad(Kf, lin_deriv)
    )

    return answ


def poly_deriv(Kf, K):
    return 2 * (Kf + 1) * K


def poly_deriv2(Kf, K):
    return 3 * ((Kf + 1) ** 2) * K


def rbf_deriv(Kf, K, sig):
    return torch.exp(-2 / (2 * sig**2) * (1 - Kf)) * (2 / (2 * sig**2) * K)


def lin_deriv(Kf):
    return Kf


def normalize_kernel_grad(Kf, Kf_deriv):
    dKf = Kf.diag()
    dKf_deriv = Kf_deriv.diag()

    answ = (
        Kf_deriv * (torch.sqrt(torch.ger(dKf, dKf)))
        + Kf * (-0.5 * torch.pow(torch.ger(dKf, dKf), 1.5)) * torch.ger(dKf_deriv, dKf)
        + torch.ger(dKf, dKf_deriv)
    )

    return answ
