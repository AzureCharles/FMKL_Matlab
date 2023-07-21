import torch
from threeLayerDeriv import *
from twoLayerDeriv import *
from spanBoundDeriv import *
def grad3Layer(model, betas, LR, Kf, K, sig, y):
    Kf_3 = Kf[:, 2]
    K_1_3 = K[:, 0, 2]
    K_2_3 = K[:, 1, 2]
    K_3_3 = K[:, 2, 2]
    K_4_3 = K[:, 3, 2]

    # Third layer
    dTdT9, _ = span_bound_deriv(model, Kf_3, K_1_3, y)
    dTdT10, _ = span_bound_deriv(model, Kf_3, K_2_3, y)
    dTdT11, _ = span_bound_deriv(model, Kf_3, K_3_3, y)
    dTdT12, span = span_bound_deriv(model, Kf_3, K_4_3, y)

    # Second layer
    Kd5 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 0, 0], betas[1], betas[2], sig)
    Kd6 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 1, 0], betas[1], betas[2], sig)
    Kd7 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 2, 0], betas[1], betas[2], sig)
    Kd8 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 3, 0], betas[1], betas[2], sig)
    dTdT5, _ = span_bound_deriv(model, Kf_3, Kd5, y)
    dTdT6, _ = span_bound_deriv(model, Kf_3, Kd6, y)
    dTdT7, _ = span_bound_deriv(model, Kf_3, Kd7, y)
    dTdT8, _ = span_bound_deriv(model, Kf_3, Kd8, y)

    # First layer
    Kd1 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 0, 0], betas[1], betas[2], sig)
    Kd2 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 1, 0], betas[1], betas[2], sig)
    Kd3 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 2, 0], betas[1], betas[2], sig)
    Kd4 = three_layer_deriv(Kf[:, 1], K[:, :, 1], Kf[:, 0], K[:, 3, 0], betas[1], betas[2], sig)
    dTdT1, _ = span_bound_deriv(model, Kf_3, Kd1, y)
    dTdT2, _ = span_bound_deriv(model, Kf_3, Kd2, y)
    dTdT3, _ = span_bound_deriv(model, Kf_3, Kd3, y)
    dTdT4, _ = span_bound_deriv(model, Kf_3, Kd4, y)

    # Display
    print('Span:', span.item())

    # Gradient step
    Dbetas = torch.tensor([[dTdT1, dTdT2, dTdT3, dTdT4], [dTdT5, dTdT6, dTdT7, dTdT8], [dTdT9, dTdT10, dTdT11, dTdT12]])
    betas = betas - LR * Dbetas

    return betas, span.item()
