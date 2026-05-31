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
def fit(m, ell, D, losses, num_epochs=1000):
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

    return coeffs, exponents, E

# ── Evaluation helper ─────────────────────────────────────────────────────────
def eval_mse(m, ell, D, losses, coeffs, exponents, E):
    lnc_m, lnc_ell, lnc_D = coeffs
    a_m,   a_ell,   a_D   = exponents
    with torch.no_grad():
        pred = ((lnc_D - a_D * D.log()).exp()
              + (lnc_m - a_m * m.log()).exp()
              + (lnc_ell - a_ell * ell.log()).exp()
              + E).log()
        mse = (losses.log() - pred).pow(2).mean().item()
        mae_relative = ((losses - pred.exp()).abs() / losses).mean().item()
    return mse, mae_relative

# ── Held-out validation with multiple random splits ──────────────────────────
TEST_FRACTION = 0.2
N_SPLITS = 10

print(f"Running {N_SPLITS} held-out splits ({int(TEST_FRACTION*100)}% test each) on {N} data points...")
print(f"Train size: ~{int(N*(1-TEST_FRACTION))}, Test size: ~{int(N*TEST_FRACTION)}\n")

results = []
for split in range(N_SPLITS):
    perm       = np.random.permutation(N)
    n_test     = int(N * TEST_FRACTION)
    test_idx   = torch.tensor(perm[:n_test])
    train_idx  = torch.tensor(perm[n_test:])

    coeffs, exponents, E = fit(
        m_all[train_idx], ell_all[train_idx], D_all[train_idx], loss_all[train_idx]
    )

    train_mse, train_rel = eval_mse(
        m_all[train_idx], ell_all[train_idx], D_all[train_idx], loss_all[train_idx],
        coeffs, exponents, E
    )
    test_mse, test_rel = eval_mse(
        m_all[test_idx], ell_all[test_idx], D_all[test_idx], loss_all[test_idx],
        coeffs, exponents, E
    )

    results.append({
        'split':     split + 1,
        'train_mse': train_mse,
        'test_mse':  test_mse,
        'train_rel_mae': train_rel,
        'test_rel_mae':  test_rel,
    })
    print(f"  Split {split+1:2d}: train MSE={train_mse:.6f}  test MSE={test_mse:.6f}  "
          f"| train relMAE={train_rel:.4f}  test relMAE={test_rel:.4f}")

df = pd.DataFrame(results)
print("\n── Summary ────────────────────────────────────────────────────────────")
print(f"{'Metric':<20} {'Train mean':>12} {'Train std':>12} {'Test mean':>12} {'Test std':>12}")
print("-" * 72)
for col_train, col_test, label in [
    ('train_mse',     'test_mse',     'MSE (log-loss²)'),
    ('train_rel_mae', 'test_rel_mae', 'Relative MAE'),
]:
    print(f"{label:<20} {df[col_train].mean():>12.6f} {df[col_train].std():>12.6f} "
          f"{df[col_test].mean():>12.6f} {df[col_test].std():>12.6f}")

df.to_csv('held_out_validation_results.csv', index=False)
print("\nPer-split results saved to held_out_validation_results.csv")
