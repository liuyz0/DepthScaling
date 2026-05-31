"""
Causal tracing across Pythia models of different depths.

For each model, computes p(first-Seattle-token) when patching clean hidden
states into a corrupted run at every (layer, token) position.

Corruption: add Gaussian noise (3 * embed_std) to subject token embeddings.
Patching:   restore the hidden state at one (layer, token) to its clean value.
Metric:     p(Seattle) = softmax(logits)[target_id] read at pred_pos (last
            token of "downtown", which predicts the next token "Seattle").

Results saved as:  results/<model-tag>.pt
  {
    "results":       FloatTensor (n_layers, n_tokens)  -- p(Seattle) per patch
    "p_clean":       float
    "p_corrupt":     float
    "tokens":        list[str]
    "subject_range": (int, int)   -- [start, end) token indices of subject
    "pred_pos":      int          -- token position whose output predicts Seattle
    "target_id":     int          -- token id of first "Seattle" token
    "model":         str
    "sentence":      str
    "subject":       str
  }
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-12b",
]

SENTENCE = "The space needle is in downtown Seattle"
SUBJECT = "space needle"
custom_cache_dir = '../../../orcd/scratch/DepthScaling/cache' # cloud

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_token_range(tokenizer, token_ids, text):
    """Return [start, end) token indices of `text` within the sequence.

    Uses character-level prefix matching so it is robust to any byte-level
    BPE tokenisation.
    """
    ids = token_ids.tolist()
    full = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    char_s = full.lower().find(text.lower())
    assert char_s >= 0, f"'{text}' not found in '{full}'"
    char_e = char_s + len(text)

    tok_s = tok_e = None
    for i in range(1, len(ids) + 1):
        prefix_len = len(
            tokenizer.decode(ids[:i], clean_up_tokenization_spaces=False)
        )
        if tok_s is None and prefix_len > char_s:
            tok_s = i - 1
        if prefix_len >= char_e:
            tok_e = i
            break

    assert tok_s is not None and tok_e is not None, f"Range not found for '{text}'"
    return tok_s, tok_e


# ---------------------------------------------------------------------------
# Main tracing function
# ---------------------------------------------------------------------------

def run_causal_trace(model_name, sentence, subject, device, results_dir):
    print(f"\n{'='*60}\nModel: {model_name}\nCUDA: {torch.cuda.is_available()}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)

    # bfloat16 for models >= 1b (A100/H100 support it natively and it is more
    # numerically stable than float16)
    #large = any(x in model_name for x in ["1b", "1.4b", "2.8b", "6.9b", "12b"])
    dtype = torch.float16 #if large else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=custom_cache_dir, torch_dtype=dtype
    ).to(device)
    model.eval()

    # ---- Tokenise --------------------------------------------------------
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)  # (1, T)
    T = input_ids.shape[1]
    n_layers = model.config.num_hidden_layers
    layers = model.gpt_neox.layers
    ids_1d = input_ids[0]  # (T,)

    # ---- Token positions -------------------------------------------------
    subj_s, subj_e = find_token_range(tokenizer, ids_1d, subject)

    # pred_pos = last token of "downtown" (its output predicts " Seattle")
    # target_id = first token of " Seattle" in the actual sequence
    downtown_s, downtown_e = find_token_range(tokenizer, ids_1d, "downtown")
    pred_pos = downtown_e - 1
    target_id = ids_1d[downtown_e].item()  # first token after "downtown" = Seattle

    token_strs = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False)
        for t in ids_1d.tolist()
    ]
    print(f"Tokens:       {token_strs}")
    print(f"Subject:      [{subj_s}, {subj_e}) = {token_strs[subj_s:subj_e]}")
    print(f"pred_pos:     {pred_pos}  ('{token_strs[pred_pos]}')")
    print(f"target token: '{tokenizer.decode([target_id])}'  (id={target_id})")
    print(f"n_layers={n_layers},  n_tokens={T}")

    # ---- Noise (fixed for reproducibility) --------------------------------
    embed_std = model.gpt_neox.embed_in.weight.float().std().item()
    noise_level = 3.0 * embed_std

    d = model.config.hidden_size
    noise = torch.zeros(1, T, d, device=device, dtype=dtype)
    noise[0, subj_s:subj_e] = (
        torch.randn(subj_e - subj_s, d, device=device).to(dtype) * noise_level
    )

    # ------------------------------------------------------------------
    # Step 1: Clean run — save hidden states at every layer
    # ------------------------------------------------------------------
    clean_states = {}  # layer_idx -> (1, T, d)

    def make_save_hook(l):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            clean_states[l] = hs.detach().clone()
        return hook

    hooks = [
        layers[i].register_forward_hook(make_save_hook(i)) for i in range(n_layers)
    ]
    with torch.no_grad():
        logits_clean = model(input_ids=input_ids, use_cache=False).logits
    p_clean = logits_clean[0, pred_pos].float().softmax(-1)[target_id].item()
    for h in hooks:
        h.remove()
    print(f"p_clean   = {p_clean:.4f}")

    # ------------------------------------------------------------------
    # Step 2: Corrupted run — add fixed noise to subject embeddings
    # ------------------------------------------------------------------
    def corrupt_hook(module, inp, out):
        return out + noise

    h = model.gpt_neox.embed_in.register_forward_hook(corrupt_hook)
    with torch.no_grad():
        logits_corrupt = model(input_ids=input_ids, use_cache=False).logits
    p_corrupt = logits_corrupt[0, pred_pos].float().softmax(-1)[target_id].item()
    h.remove()
    print(f"p_corrupt = {p_corrupt:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Patching — for each layer, batch over all token positions
    #
    # Batch of T examples, all with corrupted embeddings.
    # Example j: also patch the hidden state at (layer l, token j) back to
    # its clean value.
    # ------------------------------------------------------------------
    results = torch.zeros(n_layers, T)

    batch_ids = input_ids.expand(T, -1)        # (T, T)
    batch_noise = noise.expand(T, -1, -1)      # (T, T, d)
    tok_idx = torch.arange(T, device=device)   # [0, 1, ..., T-1]

    def corrupt_hook_batch(module, inp, out):
        return out + batch_noise

    for l in range(n_layers):
        # clean_states[l]: (1, T, d)  →  expand to (T, T, d) for batch indexing
        cs = clean_states[l].expand(T, -1, -1)

        def patch_hook(module, inp, out, _cs=cs):
            hs = out[0] if isinstance(out, tuple) else out
            patched = hs.clone()
            # Example j: restore hidden state at token position j
            patched[tok_idx, tok_idx] = _cs[tok_idx, tok_idx]
            return (patched,) + out[1:] if isinstance(out, tuple) else patched

        h1 = model.gpt_neox.embed_in.register_forward_hook(corrupt_hook_batch)
        h2 = layers[l].register_forward_hook(patch_hook)

        with torch.no_grad():
            logits_patch = model(input_ids=batch_ids, use_cache=False).logits
        # logits_patch: (T, T, vocab)  →  read pred_pos for each example
        results[l] = (
            logits_patch[:, pred_pos].float().softmax(-1)[:, target_id].cpu()
        )

        h1.remove()
        h2.remove()

        if (l + 1) % 8 == 0 or l == n_layers - 1:
            print(f"  layer {l + 1}/{n_layers}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(results_dir, "causal-"+f"{tag}.pt")
    torch.save(
        {
            "results": results,             # (n_layers, n_tokens)
            "p_clean": p_clean,
            "p_corrupt": p_corrupt,
            "tokens": token_strs,
            "subject_range": (subj_s, subj_e),
            "pred_pos": pred_pos,
            "target_id": target_id,
            "model": model_name,
            "sentence": sentence,
            "subject": subject,
        },
        path,
    )
    print(f"Saved → {path}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal tracing across Pythia models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="HuggingFace model names to trace (default: all six Pythia sizes)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results_dir", default="../../../orcd/pool/DepthScaling/outputs")
    args = parser.parse_args()

    for model_name in args.models:
        run_causal_trace(
            model_name, SENTENCE, SUBJECT, args.device, args.results_dir
        )


if __name__ == "__main__":
    main()
