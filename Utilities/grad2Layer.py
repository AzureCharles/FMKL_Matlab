import torch
from twoLayerDeriv import *
from spanBoundDeriv import * 
def grad2Layer(model, betas, LR, Kf, K, sig, y):
    Kf_2 = Kf[:, 1]
    K_1_2 = K[:, 0, 1]
    K_2_2 = K[:, 1, 1]
    K_3_2 = K[:, 2, 1]
    K_4_2 = K[:, 3, 1]

    # Second layer
    dTdT5, _ = span_bound_deriv(model, Kf_2, K_1_2, y)
    dTdT6, _ = span_bound_deriv(model, Kf_2, K_2_2, y)
    dTdT7, _ = span_bound_deriv(model, Kf_2, K_3_2, y)
    dTdT8, span = span_bound_deriv(model, Kf_2, K_4_2, y)

    # First layer
    Kd1 = two_layer_deriv(Kf[:, 0], K[:, 0, 0], betas[1], sig)
    Kd2 = two_layer_deriv(Kf[:, 0], K[:, 1, 0], betas[1], sig)
    Kd3 = two_layer_deriv(Kf[:, 0], K[:, 2, 0], betas[1], sig)
    Kd4 = two_layer_deriv(Kf[:, 0], K[:, 3, 0], betas[1], sig)
    dTdT1, _ = span_bound_deriv(model, Kf_2, Kd1, y)
    dTdT2, _ = span_bound_deriv(model, Kf_2, Kd2, y)
    dTdT3, _ = span_bound_deriv(model, Kf_2, Kd3, y)
    dTdT4, _ = span_bound_deriv(model, Kf_2, Kd4, y)

    # Display
    print('Span:', span.item())

    # Gradient step
    Dbetas = torch.tensor([[dTdT1, dTdT2, dTdT3, dTdT4], [dTdT5, dTdT6, dTdT7, dTdT8]])
    betas = betas - LR * Dbetas

    return betas, span.item()
