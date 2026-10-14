"""Lab 26: classic RLHF with PPO, from scratch - reward model, critic, KL penalty (InstructGPT recipe, 2022).

TRL 1.13 no longer ships a PPOTrainer (production PPO lives in OpenRLHF / verl / NeMo-RL). Writing the loop
once by hand is the fastest way to understand what DPO and GRPO were designed to remove.

The four models of PPO-RLHF:
  policy  pi        the model being trained (starts from the SFT model)
  ref     pi_ref    frozen copy of the SFT model: the KL anchor
  reward  r_phi     trained on human preference pairs with the Bradley-Terry loss  -log sigmoid(r(c) - r(r))
  critic  V         value head: predicts the future reward at every token (the 'baseline' for advantages)

Per-token reward:  r_t = -beta * (log pi(a_t) - log pi_ref(a_t))   (+ r_phi(prompt, answer) on the LAST token)
Advantages:        GAE(gamma=1, lambda=0.95) from r_t and V
Policy loss:       -min(rho * A, clip(rho, 1-eps, 1+eps) * A),  rho = pi(a_t) / pi_old(a_t)
Value loss:        (V - returns)^2
"""
import copy, sys, tempfile, warnings
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
warnings.filterwarnings("ignore")
from tiny import build_tiny  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)
QS = ["What is the maximum pressure?", "Maximum pressure?", "How high may the pressure go?", "Tell me the pressure limit."]
GOOD = "The maximum pressure is 6 bar."
BAD = ["The maximum pressure is 9 bar.", "Hold the red button for five seconds.", "I am not sure.",
       "The pressure is pressure pressure.", "Warranty claims require the serial number."]
GEN = 12


def prompt_ids(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"][0]


def full_ids(q, a):
    return tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                   return_tensors="pt", return_dict=True)["input_ids"][0]


# ---------------- 0) SFT model (short warm start, full-parameter on the toy) ----------------
policy = AutoModelForCausalLM.from_pretrained(model_dir)
opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
for step in range(40):
    q = QS[step % len(QS)]
    ids = full_ids(q, GOOD)[None]
    n_prompt = len(prompt_ids(q))
    labels = ids.clone(); labels[:, :n_prompt] = -100               # assistant-only loss
    loss = policy(input_ids=ids, labels=labels).loss
    opt.zero_grad(); loss.backward(); opt.step()
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def success(model, n=64):
    torch.manual_seed(123)
    ids = prompt_ids(QS[0])[None].repeat(n, 1)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=GEN, do_sample=True,
                         temperature=1.0, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return sum("6 bar" in t for t in texts) / n, texts[0]


print("0) SFT warm start: sampled success %.2f  sample %r" % success(policy))

# ---------------- 1) reward model: Bradley-Terry on preference pairs ----------------
class Scalar(torch.nn.Module):
    """Backbone + linear head on the LAST token's hidden state -> one scalar per sequence."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, ids, per_token=False):
        h = self.backbone(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        v = self.head(h).squeeze(-1)
        return v if per_token else v[:, -1]


rm = Scalar(copy.deepcopy(ref).requires_grad_(True))
opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
pairs = [(q, GOOD, b) for q in QS for b in BAD]
for epoch in range(30):
    for q, c, r in pairs:
        loss = -F.logsigmoid(rm(full_ids(q, c)[None]) - rm(full_ids(q, r)[None])).mean()   # Bradley-Terry
        opt.zero_grad(); loss.backward(); opt.step()
rm.eval()
with torch.no_grad():
    acc = sum(float(rm(full_ids(q, c)[None]) > rm(full_ids(q, r)[None])) for q, c, r in pairs) / len(pairs)
print(f"1) reward model trained on {len(pairs)} pairs: pairwise accuracy {acc:.2f}")

# ---------------- 2) PPO with critic and KL penalty ----------------
def run_ppo(beta, iters=25, batch=16, eps=0.2, lam=0.95, lr=5e-4):
    pol = copy.deepcopy(ref).requires_grad_(True).train()
    critic = Scalar(copy.deepcopy(ref).requires_grad_(True))        # value model, init from the SFT model
    opt_p = torch.optim.AdamW(pol.parameters(), lr=lr)
    opt_v = torch.optim.AdamW(critic.parameters(), lr=lr)
    log = []
    for it in range(iters):
        q = QS[it % len(QS)]
        p_ids = prompt_ids(q)[None].repeat(batch, 1)
        P = p_ids.shape[1]
        with torch.no_grad():                                          # ROLLOUT with the current policy
            seq = pol.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=GEN,
                               do_sample=True, temperature=1.0, pad_token_id=tok.pad_token_id)
            act = seq[:, P:]
            alive = torch.ones_like(act, dtype=torch.float)            # mask tokens after the first EOS
            eos_pos = (act == tok.eos_token_id).float().cumsum(1)
            alive[:, 1:] = (eos_pos[:, :-1] == 0).float()
            def logp(model):
                lg = model(input_ids=seq).logits[:, P - 1:-1]
                return torch.log_softmax(lg.float(), -1).gather(-1, act[..., None]).squeeze(-1)
            old_lp, ref_lp = logp(pol), logp(ref)
            values = critic(seq, per_token=True)[:, P - 1:-1]
            score = rm(seq)                                            # one scalar per answer
            kl = (old_lp - ref_lp) * alive
            rewards = -beta * kl
            last = (alive.sum(1).long() - 1).clamp(min=0)
            rewards[torch.arange(batch), last] += score                # RM reward on the final token
            adv = torch.zeros_like(rewards); gae = torch.zeros(batch)
            for t in reversed(range(act.shape[1])):                    # GAE, gamma = 1
                nxt = values[:, t + 1] if t + 1 < act.shape[1] else torch.zeros(batch)
                delta = rewards[:, t] + nxt * alive[:, t] - values[:, t]
                gae = (delta + lam * gae) * alive[:, t]
                adv[:, t] = gae
            returns = adv + values
            a_flat = adv[alive.bool()]
            adv = (adv - a_flat.mean()) / (a_flat.std() + 1e-8) * alive   # whiten
        for _ in range(2):                                             # PPO epochs over this batch
            new_lp = torch.log_softmax(pol(input_ids=seq).logits[:, P - 1:-1].float(), -1).gather(-1, act[..., None]).squeeze(-1)
            ratio = torch.exp(new_lp - old_lp)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
            pg = (pg * alive).sum() / alive.sum()
            opt_p.zero_grad(); pg.backward(); opt_p.step()
            v = critic(seq, per_token=True)[:, P - 1:-1]
            vl = (((v - returns) ** 2) * alive).sum() / alive.sum()
            opt_v.zero_grad(); vl.backward(); opt_v.step()
        log.append((float(score.mean()), float(kl.sum(1).mean())))
    return pol.eval(), log


for beta in (0.0, 0.2):
    pol, log = run_ppo(beta)
    rate, sample = success(pol)
    print(f"\n2) PPO beta={beta}:  iter  RM-score  KL(pi||ref)")
    for i in (0, len(log) // 2, len(log) - 1):
        print(f"                 {i:>4}  {log[i][0]:>8.3f}  {log[i][1]:>10.3f}")
    print(f"   sampled success {rate:.2f}; sample {sample!r}")

print("""
READING
- The reward model is a LEARNED proxy (20 toy pairs here). PPO maximises the proxy, not the truth. Without a
  KL anchor (beta=0) the policy drifts far from the SFT model to please the proxy (watch the KL column grow)
  and the text degenerates while the RM score rises: reward over-optimisation (Gao et al. 2022). beta > 0
  keeps the KL bounded. On this toy BOTH runs exploit the weak RM somewhat: a 20-pair reward model is
  easy to fool. Real RMs are trained on 10k-1M human comparisons and still get exploited: always keep KL.
- PPO-RLHF keeps FOUR models in memory (policy, ref, reward, critic): ~4x the memory of SFT. That cost is
  the reason for:  DPO (no reward model, no critic, offline pairs)  and  GRPO (no critic: group baseline).
- Production PPO for LLMs: OpenRLHF, verl, NeMo-RL. For most teams: SFT -> DPO, or SFT -> GRPO when the
  task has a verifier. Use PPO when you have a strong reward model and the infrastructure for it.""")
