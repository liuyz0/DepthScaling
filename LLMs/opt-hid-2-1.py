# study the hidden states
# layers 0 to L-1, first go through last transfomer, then final norm, then lm head

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time

custom_cache_dir = '../cache' # cloud
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 48 # 48 documents as a batch
max_length = 1024 # max length to generate (should be <= model max length, 2048 for OPT)
num_docs = batch_size * 100 # total number of documents to generate
# chunk_size_time = None

dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
data_iter = iter(dataset)

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
).to(device).eval()
num_layers = model.config.num_hidden_layers

# ---------- Model helpers ----------
def get_final_norm_and_head(model):
    # OPT family
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "final_layer_norm") and hasattr(model, "lm_head"):
        return model.model.decoder.final_layer_norm, model.lm_head
    # Pythia / GPT-NeoX
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm") and hasattr(model, "embed_out"):
        return model.gpt_neox.final_layer_norm, model.embed_out

final_norm, lm_head = get_final_norm_and_head(model)
last_layer = model.model.decoder.layers[-1].eval()

def prepare_attn_mask_opt(attention_mask: torch.Tensor, seq_len: int, dtype, device):
    # OPT uses combined causal + padding mask
    # attention_mask: (B, T) with 1=keep, 0=pad
    if attention_mask is None:
        return None
    B = attention_mask.shape[0]
    # Create causal mask (lower triangular)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # Expand padding mask to (B, 1, 1, T)
    expanded_attn_mask = attention_mask[:, None, None, :].expand(B, 1, seq_len, seq_len).to(device)
    # Combine: 0 where we should attend, -inf otherwise
    combined_mask = torch.zeros(B, 1, seq_len, seq_len, device=device, dtype=dtype)
    combined_mask = combined_mask.masked_fill(causal_mask, torch.finfo(dtype).min)
    combined_mask = combined_mask.masked_fill(expanded_attn_mask == 0, torch.finfo(dtype).min)
    return combined_mask

def build_position_ids(attention_mask: torch.Tensor):
    # OPT: positions start at padding_idx + 1 (typically 2) for real tokens
    # attention_mask: (B, T) with 1 for real tokens, 0 for pad
    pos_ids = (attention_mask.cumsum(dim=-1) - 1 + 2) * attention_mask  # +2 for OPT offset, mask out padding
    return pos_ids.long()

# initialize results
CE = {layer: None for layer in range(num_layers)} #torch.zeros(max_length-1, num_layers, device=device, dtype=torch.float32) # cross-entropy at each position
#H = {layer: None for layer in range(num_layers)} #torch.zeros(max_length-1, num_layers, device=device, dtype=torch.float32)  # entropy at each position
Theta = {layer: None for layer in range(num_layers)} #torch.zeros(max_length-1, num_layers, device=device, dtype=torch.float32)  # rotation of hidden state
Theta_dh = {layer: None for layer in range(num_layers-1)} # angle between dh
norms = {layer: None for layer in range(num_layers+1)}  # norm of hidden states
angle_to_end = {layer: None for layer in range(num_layers)}  # angle between h_l and h_L
#proj_s = {layer: None for layer in range(num_layers+1)}  # projection onto h_L - h_0
#Count = torch.zeros(max_length-1, dtype=torch.int32, device=device) # count of how many times each position was seen

start_time = time.time()

for step in range(num_docs // batch_size):
    batch = [next(data_iter)['text'] for _ in range(batch_size)]
    inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
        truncation=True,
        max_length=max_length) # ids and attention masks (batch_size, seq_len)
    
    valid_mask = (inputs['attention_mask'][:, 1:] > 0) & (inputs['attention_mask'][:, :-1] > 0) # (batch_size, seq_len-1)
    valid_mask = valid_mask.to(device)
    targets = inputs['input_ids'][:, 1:].to(device) # (batch_size, seq_len-1)
    # NeoX expects a prepared (broadcasted) attention mask; reuse the model helper
    # Shape after prep: (B, 1, T, T), with -inf on masked positions
    prepared_mask = prepare_attn_mask_opt(inputs['attention_mask'], max_length, dtype=next(model.parameters()).dtype, device=device)
    # position_ids = build_position_ids(inputs['attention_mask'].to(device))  # (B, T)
    
    with torch.no_grad():
        out = model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        hidden_states = out.hidden_states # tuple of (batch_size, seq_len, hidden_size), length num_layers+1 (including input embeddings)
        dhL0 = hidden_states[num_layers][:,:-1,:] - hidden_states[0][:,:-1,:]  # (batch_size, seq_len-1, hidden_size)
        dh = [hidden_states[layer+1] - hidden_states[layer] for layer in range(num_layers)] # length num_layers, each (batch_size, seq_len, hidden_size)

        for layer in range(num_layers+1):
            """
            if num_layers > layer > 0:
                dhl0 = hidden_states[layer][:,:-1,:] - hidden_states[0][:,:-1,:]  # (batch_size, seq_len-1, hidden_size)
                proj_s_layer = ((dhl0 * dhL0).sum(dim = -1) / (dhL0.norm(dim = -1).pow(2) + 1e-10))  # (batch_size, seq_len-1)
            elif layer == 0:
                proj_s_layer = torch.zeros_like(dhL0[:,:,0])  # (batch_size, seq_len-1)
            else: # layer == num_layers
                proj_s_layer = torch.ones_like(dhL0[:,:,0])  # (batch_size, seq_len-1)
                
            if proj_s[layer] is None:
                proj_s[layer] = proj_s_layer[valid_mask].cpu()  # (num_valid_positions)
            else:
                proj_s[layer] = torch.cat([proj_s[layer], proj_s_layer[valid_mask].cpu()])  # (num_tokens_so_far)
            """
            if norms[layer] is None:
                norms[layer] = (hidden_states[layer][:,:-1,:].norm(dim=-1).to(torch.float32))[valid_mask].cpu()  # (num_valid_positions)
            else:
                norms[layer] = torch.cat([norms[layer], (hidden_states[layer][:,:-1,:].norm(dim=-1).to(torch.float32))[valid_mask].cpu()])  # (num_tokens_so_far)
        
        for layer in range(num_layers-1): # 0 to L - 2, we use logits for the last hidden state directly
            hidden_state_ = last_layer(hidden_states[layer],
                                        attention_mask=prepared_mask)[0]  # (batch_size, seq_len, hidden_size)
            logits = lm_head(final_norm(hidden_state_))[:, :-1, :].to(torch.float32) # (batch_size, seq_len-1, vocab_size)
            
            log_probs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len-1, vocab_size)

            CE_layer = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len-1)
            # H_layer = -(log_probs * log_probs.exp()).sum(dim=-1) # (batch_size, seq_len-1)

            #CE[:, layer] += (CE_layer * valid_mask).sum(dim=0) # (seq_len-1)
            if CE[layer] is None:
                # find the elements where valid_mask is True, and put into one vector, not summing
                CE[layer] = CE_layer[valid_mask].cpu()  # (num_valid_positions)
            else:
                CE[layer] = torch.cat([CE[layer], CE_layer[valid_mask].cpu()])  # (num_tokens_so_far)
            #H[:, layer] += (H_layer * valid_mask).sum(dim=0) # (seq_len-1)
            '''
            if H[layer] is None:
                H[layer] = H_layer[valid_mask].cpu()  # (num_valid_positions)
            else:
                H[layer] = torch.cat([H[layer], H_layer[valid_mask].cpu()])  # (num_tokens_so_far)
            '''
            #Theta[:, layer] += (F.cosine_similarity(hidden_states[layer][:,:-1,:], 
                                #hidden_states[layer+1][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos() * valid_mask).sum(dim = 0)
            if Theta[layer] is None:
                Theta[layer] = (F.cosine_similarity(hidden_states[layer][:,:-1,:], 
                                    hidden_states[layer+1][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()
            else:
                Theta[layer] = torch.cat([Theta[layer], (F.cosine_similarity(hidden_states[layer][:,:-1,:], 
                                    hidden_states[layer+1][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()])
            if Theta_dh[layer] is None:
                Theta_dh[layer] = (F.cosine_similarity(dh[layer][:,:-1,:], 
                                    dh[layer+1][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()
            else:
                Theta_dh[layer] = torch.cat([Theta_dh[layer], (F.cosine_similarity(dh[layer][:,:-1,:], 
                                    dh[layer+1][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()])
            if angle_to_end[layer] is None:
                angle_to_end[layer] = (F.cosine_similarity(hidden_states[layer][:,:-1,:], 
                                    hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()
            else:
                angle_to_end[layer] = torch.cat([angle_to_end[layer], (F.cosine_similarity(hidden_states[layer][:,:-1,:], 
                                    hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()])

        # final layer (L-1)
        logits = out.logits[:, :-1, :].to(torch.float32) # (batch_size, seq_len-1, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len-1, vocab_size)
        CE_layer = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len-1)
        # H_layer = -(log_probs * log_probs.exp()).sum(dim=-1) # (batch_size, seq_len-1)

        if CE[num_layers-1] is None:
            CE[num_layers-1] = CE_layer[valid_mask].cpu()  # (num_valid_positions)
        else:
            CE[num_layers-1] = torch.cat([CE[num_layers-1], CE_layer[valid_mask].cpu()])  # (num_tokens_so_far)
        '''
        if H[num_layers-1] is None:
            H[num_layers-1] = H_layer[valid_mask].cpu()  # (num_valid_positions)
        else:
            H[num_layers-1] = torch.cat([H[num_layers-1], H_layer[valid_mask].cpu()])  # (num_tokens_so_far)
        '''
        #Theta[num_layers-1] = (F.cosine_similarity(hidden_states[num_layers-1][:,:-1,:], 
                            #hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos() * valid_mask).sum(dim = 0)
        if Theta[num_layers-1] is None:
            Theta[num_layers-1] = (F.cosine_similarity(hidden_states[num_layers-1][:,:-1,:], 
                                hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()
        else:
            Theta[num_layers-1] = torch.cat([Theta[num_layers-1], (F.cosine_similarity(hidden_states[num_layers-1][:,:-1,:], 
                                hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()])
        if angle_to_end[num_layers-1] is None:
            angle_to_end[num_layers-1] = (F.cosine_similarity(hidden_states[num_layers-1][:,:-1,:], 
                                hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()
        else:
            angle_to_end[num_layers-1] = torch.cat([angle_to_end[num_layers-1], (F.cosine_similarity(hidden_states[num_layers-1][:,:-1,:], 
                                hidden_states[num_layers][:,:-1,:], dim = -1).clamp(min = -1.0, max = 1.0).to(torch.float32).acos())[valid_mask].cpu()])

print(f"Time for {num_docs} docs: {time.time() - start_time:.2f} seconds")

# finalize
CE = torch.stack([CE[layer] for layer in range(num_layers)], dim=1)  # (total_num_tokens, num_layers)
#H = torch.stack([H[layer] for layer in range(num_layers)], dim=1)  # (total_num_tokens, num_layers)
Theta = torch.stack([Theta[layer] for layer in range(num_layers)], dim=1)  # (total_num_tokens, num_layers)
Theta_dh = torch.stack([Theta_dh[layer] for layer in range(num_layers-1)], dim=1)  # (total_num_tokens, num_layers-1)
norms = torch.stack([norms[layer] for layer in range(num_layers+1)], dim=1)  # (total_num_tokens, num_layers+1)
angle_to_end = torch.stack([angle_to_end[layer] for layer in range(num_layers)], dim=1)  # (total_num_tokens, num_layers)
#proj_s = torch.stack([proj_s[layer] for layer in range(num_layers+1)], dim=1)  # (total_num_tokens, num_layers+1)

# save
torch.save({'cross_entropy': CE, 'theta': Theta, 'theta_dh': Theta_dh, 
            'norms': norms, 'angle_to_end': angle_to_end}, 
           f'../outputs/opt-hid-2-1.pt') # cloud
