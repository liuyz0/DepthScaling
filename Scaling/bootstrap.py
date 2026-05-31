import numpy as np
import torch
import pandas as pd

# ── Data loading (mirrors notebook cell 2) ───────────────────────────────────
fitting_data = pd.read_csv('get_fitting_data_chinchilla.csv')
m_raw    = fitting_data['d_model'].values
ell_raw  = fitting_data['n_layers'].values
D_raw    = fitting_data['num_tokens'].values
loss_raw = fitting_data['loss'].values

nr_of_models_excluded = 40
sorted_losses = sorted(loss_raw)
indices = [i for i in range(len(m_raw))
           if loss_raw[i] < sorted_losses[-nr_of_models_excluded]]

m_all    = torch.tensor(m_raw[indices])
ell_all  = torch.tensor(ell_raw[indices]) * 100
D_all    = torch.tensor(D_raw[indices]) / 1e6
loss_all = torch.tensor(loss_raw[indices])

N = len(loss_all)

# ── Fitting function ─────────────────────────────────────────────────────────
def fit(m, ell, D, losses, num_epochs=10000):
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

# ── Bootstrap ────────────────────────────────────────────────────────────────
N_BOOTSTRAP = 200
param_names = ['c_m', 'c_ell', 'c_D', 'a_m', 'a_ell', 'a_D', 'E']
records = []

print(f"Running {N_BOOTSTRAP} bootstrap fits on {N} data points...")
for b in range(N_BOOTSTRAP):
    idx = np.random.choice(N, N, replace=True)
    idx = torch.tensor(idx)
    params = fit(m_all[idx], ell_all[idx], D_all[idx], loss_all[idx])
    records.append(params)
    if (b + 1) % 20 == 0:
        print(f"  {b+1}/{N_BOOTSTRAP} done")

df = pd.DataFrame(records, columns=param_names)

print("\n── Bootstrap results ──────────────────────────────────────────────────")
print(f"{'Parameter':<10} {'Mean':>12} {'Std':>12} {'2.5%':>12} {'97.5%':>12}")
print("-" * 60)
for col in param_names:
    vals = df[col].values
    print(f"{col:<10} {vals.mean():>12.4f} {vals.std():>12.4f} "
          f"{np.percentile(vals, 2.5):>12.4f} {np.percentile(vals, 97.5):>12.4f}")

df.to_csv('bootstrap_results.csv', index=False)
print("\nFull bootstrap samples saved to bootstrap_results.csv")
