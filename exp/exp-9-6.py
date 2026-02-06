# MSE loss, try to distinguish ODE and ensemble only first
# 4 hours run
# just exp-9-2 but 80000 steps

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import sys
import time

# Hyperparameters
task_id = int(sys.argv[1]) # from 0 to 5
num_tasks = int(sys.argv[2]) # should be 2 * 3 = 6
assert num_tasks == 6

teacher_id = (task_id) % 3 # index for teacher replicates
rho_id = (task_id) // 3 # index for temperatures

rhos = [0.0, 1.0]
rho = rhos[rho_id]
s_n_layers = [6, 12, 16, 24, 32, 48]  # student RN
t_n_layers = 128
batch_size = 1024
num_steps = 80_000
log_interval = 500

# ---- Config ----
@dataclass
class ToyConfig:
    vocab_size: int = 128     # vocab size
    n_layer: int = 48         # blocks
    n_embd: int = 32         # model width (d_model)

# ---- Utils ----
def rmsnorm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

# ---- Core module ----
class MLP(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        hidden = config.n_embd * 4
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=True)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x) ** 2 # relu^2 activation
        x = self.c_proj(x)
        return x

class Layer(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(rmsnorm(x))
        return x

# ---- The toy model ----
class ToyModel(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        #self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def scale_projections_(self, teacher = False):
        factor = 1 / (self.config.n_layer ** 0.5) if teacher else 0.0
        for layer in self.layers:
            layer.mlp.c_proj.weight.data.mul_(factor)
        #if not teacher:
            #self.head.weight.data.zero_()

    def forward(self, x: torch.Tensor, temperature: float = 1.0, output_hidden: bool = False):
        # x: (B, n_embd)
        hidden_states = []
        x = rmsnorm(x)  # (B, n_embd)
        if output_hidden:
            hidden_states.append(x)
        for layer in self.layers:
            x = layer(x)
            if output_hidden:
                hidden_states.append(x)
        # hidden_states: List of (B, n_embd), len = n_layer + 1
        x = rmsnorm(x)  # (B, n_embd)
        #logits = self.head(x) / temperature  # (B, vocab_size)
        if output_hidden:
            return {'logits': x, 'hidden_states': hidden_states}
        else:
            return x
        
# ---- lr scheduler ----
# linear decay once reach 0.8 of total steps, no warmup
def get_lr(step: int, total_steps: int, base_lr: float):
    if step < total_steps * 0.8:
        return base_lr
    else:
        return base_lr * (total_steps - step) / (total_steps * 0.2) * 0.9 + base_lr * 0.1
        
# ---- Main script ----
train_losses = torch.zeros(len(s_n_layers), num_steps)
test_losses = torch.zeros(len(s_n_layers), num_steps // log_interval)

t_cfg = ToyConfig(n_layer=t_n_layers)
t_model = ToyModel(t_cfg)
t_model.scale_projections_(teacher=True)
trained_students = []

if rho == 1.0:
    with torch.no_grad():
        for t_layer_id in range(1, t_cfg.n_layer):
            t_model.layers[t_layer_id].mlp.c_proj.weight.data = t_model.layers[0].mlp.c_proj.weight.data
            t_model.layers[t_layer_id].mlp.c_fc.weight.data = t_model.layers[0].mlp.c_fc.weight.data
            t_model.layers[t_layer_id].mlp.c_fc.bias.data = t_model.layers[0].mlp.c_fc.bias.data

start_time = time.time()
for s_idx, s_n_layer in enumerate(s_n_layers):
    s_cfg = ToyConfig(n_layer=s_n_layer)
    s_model = ToyModel(s_cfg)
    s_model.scale_projections_(teacher=False)
    optimizer = torch.optim.Adam(s_model.parameters(), lr=6e-4)

    for step in range(num_steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step, num_steps, 6e-4 * 2 / s_n_layer ** 0.5) 
            # important for MLP layers to have lr scaling with depth
            # later adding Head, no need to scale its lr with depth
        inputs = torch.randn(batch_size, s_cfg.n_embd)
        with torch.no_grad():
            t_logits = t_model(inputs)  # (B, vocab_size)
            #t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
        s_logits = s_model(inputs)
        #s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
        #loss = F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True)
        loss = F.mse_loss(s_logits, t_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses[s_idx, step] = loss.item()
        
        if (step + 1) % log_interval == 0:
            # test loss 10 batches
            with torch.no_grad():
                test_loss = 0.0
                for _ in range(10):
                    inputs = torch.randn(batch_size, s_cfg.n_embd)
                    t_logits = t_model(inputs)  # (B, vocab_size)
                    #t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
                    s_logits = s_model(inputs)
                    #s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
                    #loss = F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True)
                    loss = F.mse_loss(s_logits, t_logits)
                    test_loss += loss.item()
                test_loss /= 10
                test_losses[s_idx, (step+1) // log_interval - 1] = test_loss
    
    print(f"Teacher {teacher_id}, Rho {rho:.1f}, Student {s_n_layer}, Time {time.time() - start_time:.2f} sec, Test Loss {test_loss:.4f}")
    trained_students.append(s_model.state_dict())

# save results
torch.save({'train_losses': train_losses, 
            'test_losses': test_losses, 
            'trained_students': trained_students,
            'teacher': t_model.state_dict()},
           f'../outputs/exp-9-6-{task_id}.pt')