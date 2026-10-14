"""Lab 27: the three pre-training objectives side by side - CLM, MLM, Prefix-LM.

Same 8-token sentence, three ways to turn it into a training signal. For each objective we print:
  - the ATTENTION MASK (which positions each position may look at: 1 = may see)
  - the LABELS (what is predicted where; -100 = no loss at that position)
Then we run all three on one tiny random transformer and show they are just different masks + labels
over the SAME network. That is the whole difference between GPT-style, BERT-style and T5/UL2-prefix models.
"""
import torch

torch.manual_seed(0)
tokens = ["The", "pump", "must", "stay", "below", "6", "bar", "."]
ids = torch.arange(len(tokens)) + 10          # pretend token ids
T = len(tokens)
MASK_ID = 3


def show(name, attn, labels, inputs):
    print(f"\n=== {name} ===")
    print("input :", " ".join(f"{t:>6}" for t in inputs))
    print("labels:", " ".join(f"{(tokens[l] if l != -100 else '-'):>6}" for l in labels.tolist()))
    print("attention mask (row = position asking, col = position it may read):")
    for i in range(T):
        print(f"  {inputs[i]:>6} " + " ".join("1" if attn[i, j] else "." for j in range(T)))


# 1) CLM (causal language modelling, GPT/Llama/Qwen/DeepSeek): see only the past, predict the NEXT token.
clm_attn = torch.tril(torch.ones(T, T, dtype=torch.bool))
# labels hold the POSITION of the target token (-100 = no loss); position i is trained to output tokens[i+1]
clm_lbl = torch.full((T,), -100)
clm_lbl[:-1] = torch.arange(1, T)
show("CLM  (decoder-only: every position predicts the next token, sees only the past)", clm_attn,
     clm_lbl, tokens)

# 2) MLM (masked language modelling, BERT/RoBERTa): see EVERYTHING, predict only the masked tokens.
mlm_attn = torch.ones(T, T, dtype=torch.bool)
masked = torch.tensor([1, 5])                                 # 15% in BERT; 2 of 8 here
mlm_inputs = [("[MASK]" if i in masked.tolist() else t) for i, t in enumerate(tokens)]
mlm_lbl = torch.full((T,), -100)
mlm_lbl[masked] = masked
show("MLM  (encoder-only: bidirectional, loss ONLY on masked positions)", mlm_attn, mlm_lbl, mlm_inputs)

# 3) Prefix-LM (T5 / UL2 / GLM style): the prefix is bidirectional, the rest is causal; loss on the suffix.
P = 4
pre_attn = torch.tril(torch.ones(T, T, dtype=torch.bool))
pre_attn[:, :P] = True                                        # everyone may read the whole prefix
pre_attn[:P, P:] = False                                      # prefix may not read the future suffix
pre_lbl = torch.full((T,), -100)
pre_lbl[P - 1:-1] = torch.arange(P, T)                        # predict suffix tokens only
show(f"Prefix-LM (first {P} tokens bidirectional 'context', suffix generated causally)", pre_attn, pre_lbl,
     tokens)

# Same network, three objectives: only the mask and the labels change.
emb = torch.nn.Embedding(64, 32)
layer = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
head = torch.nn.Linear(32, 64)
x_ids = ids.clone()


def loss_for(attn, labels, inp_ids):
    h = layer(emb(inp_ids)[None], src_mask=~attn)             # PyTorch: True = BLOCKED, so invert
    logits = head(h)[0]
    target = torch.where(labels >= 0, ids[labels.clamp(min=0)], torch.full_like(labels, -100))
    return torch.nn.functional.cross_entropy(logits, target, ignore_index=-100)


mlm_ids = x_ids.clone(); mlm_ids[masked] = MASK_ID
print("\nloss on the same random network:")
print(f"  CLM       {loss_for(clm_attn, clm_lbl, x_ids):.4f}  trained positions: {(clm_lbl >= 0).sum().item()}/{T}")
print(f"  MLM       {loss_for(mlm_attn, mlm_lbl, mlm_ids):.4f}  trained positions: {(mlm_lbl >= 0).sum().item()}/{T}")
print(f"  Prefix-LM {loss_for(pre_attn, pre_lbl, x_ids):.4f}  trained positions: {(pre_lbl >= 0).sum().item()}/{T}")
print("""
READING
- CLM trains on ~every token (dense signal) and can GENERATE: this is why every modern chat LLM is CLM.
- MLM trains on ~15% of tokens and sees both sides: excellent for understanding/embeddings/classification,
  but it cannot generate text left-to-right. Fine-tune BERT-style models for classification/NER/retrieval.
- Prefix-LM: bidirectional over the instruction/context, causal over the answer. SFT with assistant-only loss
  on a CLM model is the SAME label pattern as Prefix-LM (loss on the suffix) but keeps the causal mask.
- Fine-tuning must use the objective the model was PRE-trained with: CLM data/collator for Llama/Qwen,
  MLM/classification heads for BERT, seq2seq for T5. Mixing them is the classic 'loss goes down, model is junk'.""")
