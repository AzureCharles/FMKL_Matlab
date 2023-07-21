import torch
import twoLayerDeriv

def three_layer_deriv(Kf2, K2, Kf1, K1, betas2, betas3, sig):
    r, _ = K2.size()
    r = int(torch.sqrt(r))
    Kf2 = Kf2.view(r, r)
    K2 = K2.view(r, r, 4)

    rbf_deriv = torch.exp(-2 / (2 * sig**2) * (1 - K2[:, :, 0])) * (
        2 / (2 * sig**2) * two_layer_deriv(Kf1, K1, betas2, sig)
    )
    poly_deriv = 2 * (K2[:, :, 1] + 1) * two_layer_deriv(Kf1, K1, betas2, sig)
    poly2_deriv = 3 * ((K2[:, :, 2] + 1) ** 2) * two_layer_deriv(Kf1, K1, betas2, sig)
    lin_deriv = two_layer_deriv(Kf1, K1, betas2, sig)

    answ = (
        betas3[0] * normalize_kernel_grad(Kf2, rbf_deriv)
        + betas3[1] * normalize_kernel_grad(Kf2, poly_deriv)
        + betas3[2] * normalize_kernel_grad(Kf2, poly2_deriv)
        + betas3[3] * normalize_kernel_grad(Kf2, lin_deriv)
    )

    return answ


def rbf_deriv(K2, Kf1, K1, betas2, sig):
    return torch.exp(-2 / (2 * sig**2) * (1 - K2)) * (
        2 / (2 * sig**2) * two_layer_deriv(Kf1, K1, betas2, sig)
    )


def poly_deriv(K2, Kf1, K1, betas2, sig):
    return 2 * (K2 + 1) * two_layer_deriv(Kf1, K1, betas2, sig)


def poly2_deriv(K2, Kf1, K1, betas2, sig):
    return 3 * ((K2 + 1) ** 2) * two_layer_deriv(Kf1, K1, betas2, sig)


def lin_deriv(Kf1, K1, betas2, sig):
    return two_layer_deriv(Kf1, K1, betas2, sig)


def normalize_kernel_grad(Kf, Kf_deriv):
    dKf = Kf.diag()
    dKf_deriv = Kf_deriv.diag()

    answ = (
        Kf_deriv * (torch.sqrt(torch.ger(dKf, dKf)))
        + Kf * (-0.5 * torch.pow(torch.ger(dKf, dKf), 1.5)) * torch.ger(dKf_deriv, dKf)
        + torch.ger(dKf, dKf_deriv)
    )

    return answ


