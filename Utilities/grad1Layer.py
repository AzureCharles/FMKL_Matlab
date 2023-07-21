import torch

def grad1Layer(model, betas, LR, Kf, K, y):
    Kf_1 = Kf[:, 0]
    K_1_1 = K[:, 0, 0]
    K_1_2 = K[:, 0, 1]
    K_1_3 = K[:, 0, 2]
    K_1_4 = K[:, 0, 3]

    # first layer
    dTdT1, _ = span_bound_deriv(model, Kf_1, K_1_1, y)
    dTdT2, _ = span_bound_deriv(model, Kf_1, K_1_2, y)
    dTdT3, _ = span_bound_deriv(model, Kf_1, K_1_3, y)
    dTdT4, span = span_bound_deriv(model, Kf_1, K_1_4, y)

    # display
    print('Span:', span.item())

    # gradient step
    Dbetas = torch.tensor([dTdT1, dTdT2, dTdT3, dTdT4])
    betas = betas - LR * Dbetas

    return betas, span.item()
