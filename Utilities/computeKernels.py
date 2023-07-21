import torch

def computeKernels(dotx, sig, betas, nLayers):
    r, _ = dotx.size()
    K = torch.zeros(r * r, 4, nLayers)
    Kf = torch.zeros(r * r, nLayers)
    
    for t in range(1, nLayers + 1):
        if t == 1:
            Krbf = rbf(dotx, sig)
        else:
            Krbf = normalize_kernel(rbf2(dotx, sig))

        Kpoly2 = normalize_kernel((dotx + 1) ** 2)
        Kpoly3 = normalize_kernel((dotx + 1) ** 3)
        Klin = normalize_kernel(dotx)

        K[:, 0, t - 1] = Krbf.view(-1)
        K[:, 1, t - 1] = Kpoly2.view(-1)
        K[:, 2, t - 1] = Kpoly3.view(-1)
        K[:, 3, t - 1] = Klin.view(-1)

        dotx = K[:, :, t - 1] * betas[t - 1, :]
        Kf[:, t - 1] = torch.sum(dotx, dim=1)
        dotx = Kf[:, t - 1].view(r, r)

    return K, Kf

# Implement the rbf and normalize_kernel functions as required.
