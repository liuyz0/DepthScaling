import numpy as np
import torch
import pandas as pd

# ── Data loading (mirrors notebook cell 2) ───────────────────────────────────
fitting_data = pd.read_csv('get_fitting_data_chinchilla.csv')
m_raw    = fitting_data['d_model'].values
ell_raw  = fitting_data['n_layers'].values
D_raw    = fitting_data['num_tokens'].values
loss_raw = fitting_data['loss'].values

# ── Fitting function ─────────────────────────────────────────────────────────
def fit(m, ell, D, losses, num_epochs=50000):
    coeffs    = torch.nn.Parameter(torch.tensor([np.log(200.), np.log(200.), np.log(10.)]))
    exponents = torch.nn.Parameter(torch.tensor([1.0, 1.0, 0.33]))
    E         = torch.nn.Parameter(torch.tensor(1.69))

    optimizer = torch.optim.Adam([
        {'params': coeffs,    'lr': 0.005},
        {'params': exponents, 'lr': 0.0005},
        {'params': E,         'lr': 0.005},
    ])

    for _ in range(num_epochs):
        optimizer.zero_grad()
        lnc_m, lnc_ell, lnc_D = coeffs
        a_m,   a_ell,   a_D   = exponents
        pred = ((lnc_D - a_D * D.log()).exp()
              + (lnc_m - a_m * m.log()).exp()
              + (lnc_ell - a_ell * ell.log()).exp()
              + E).log()
        loss = (losses.log() - pred).pow(2).mean() * 100
        loss.backward()
        optimizer.step()

    return {
        'c_m':   coeffs[0].exp().item(),
        'c_ell': coeffs[1].exp().item(),
        'c_D':   coeffs[2].exp().item(),
        'a_m':   exponents[0].item(),
        'a_ell': exponents[1].item(),
        'a_D':   exponents[2].item(),
        'E':     E.item(),
    }

# standard error 

def standard_error(params, m, ell, D, losses):
    # Calculate standard errors using the Jacobian
    # Enable gradient computation for all parameters
    coeffs = torch.tensor([params['c_m'], params['c_ell'], params['c_D']]).log().requires_grad_(True)
    exponents = torch.tensor([params['a_m'], params['a_ell'], params['a_D']], requires_grad=True)
    E = torch.tensor(params['E'], requires_grad=True)

    # Calculate residuals
    lnc_m, lnc_ell, lnc_D = coeffs
    a_m, a_ell, a_D = exponents
    expected_log_loss = ((lnc_D - a_D * D.log()).exp() + (lnc_m - a_m * m.log()).exp() + (lnc_ell - a_ell * ell.log()).exp() + E).log()
    residuals = losses - expected_log_loss.exp()

    # Number of observations and parameters
    n = len(losses)
    p = 7  # total number of parameters (3 coeffs + 3 exponents + 1 E)

    # Calculate Jacobian matrix using autograd
    jacobian = []
    for i in range(n):
        grads = torch.autograd.grad(residuals[i], [coeffs, exponents, E], 
                                    retain_graph=True, create_graph=False)
        # Flatten all gradients into a single vector
        grad_vec = torch.cat([grads[0], grads[1], grads[2].unsqueeze(0)])
        jacobian.append(grad_vec.detach())

    J = torch.stack(jacobian)  # Shape: (n, p)

    # Calculate residual variance (Mean Squared Error)
    residual_variance = (residuals.detach() ** 2).sum() / (n - p)
    #print(f"Residual variance: {residual_variance.item()}")

    # Calculate covariance matrix: Cov = residual_variance * (J^T J)^(-1)
    JTJ = J.T @ J
    try:
        covariance_matrix = residual_variance * torch.linalg.inv(JTJ)
        
        # Standard errors are the square root of the diagonal elements
        standard_errors = torch.sqrt(torch.diag(covariance_matrix))
        
        return {
            'c_m': standard_errors[0].item(),
            'c_ell': standard_errors[1].item(),
            'c_D': standard_errors[2].item(),
            'a_m': standard_errors[3].item(),
            'a_ell': standard_errors[4].item(),
            'a_D': standard_errors[5].item(),
            'E': standard_errors[6].item()
        }

    except RuntimeError as e:
        print(f"Could not invert matrix: {e}")
        print("Using pseudo-inverse instead...")
        covariance_matrix = residual_variance * torch.linalg.pinv(JTJ)
        standard_errors = torch.sqrt(torch.diag(covariance_matrix))
        
        return {
            'c_m': standard_errors[0].item(),
            'c_ell': standard_errors[1].item(),
            'c_D': standard_errors[2].item(),
            'a_m': standard_errors[3].item(),
            'a_ell': standard_errors[4].item(),
            'a_D': standard_errors[5].item(),
            'E': standard_errors[6].item()
        }

# ── Data filtering ───────────────────────────────────────────────────────────
nr_of_models_excludeds = [1,10,20,30,40,50,60,70,80,90,100]
sorted_losses = sorted(loss_raw)
records = []
param_names = ['c_m', 'c_ell', 'c_D', 'a_m', 'a_ell', 'a_D', 'E', 'std_c_m', 'std_c_ell', 'std_c_D', 'std_a_m', 'std_a_ell', 'std_a_D', 'std_E']

for nr_of_models_excluded in nr_of_models_excludeds:
    indices = [i for i in range(len(m_raw))
            if loss_raw[i] < sorted_losses[-nr_of_models_excluded]]

    m_all    = torch.tensor(m_raw[indices])
    ell_all  = torch.tensor(ell_raw[indices]) * 100
    D_all    = torch.tensor(D_raw[indices]) / 1e6
    loss_all = torch.tensor(loss_raw[indices])

    N = len(loss_all)

    params = fit(m_all, ell_all, D_all, loss_all)
    params_std = standard_error(params, m_all, ell_all, D_all, loss_all)
    params.update({f'std_{k}': v for k, v in params_std.items()})
    records.append(params)

    print(f"Excluded {nr_of_models_excluded} models, alpha_ell: {params['a_ell']:.4f}, std_alpha_ell: {params['std_a_ell']:.4f}")

df = pd.DataFrame(records, columns=param_names)
df.to_csv('scanignore_results.csv', index=False)
