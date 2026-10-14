"""Lab 25: GRPO - Group Relative Policy Optimization (DeepSeekMath 2024, used for DeepSeek-R1).

PPO needs a CRITIC (value network as big as the policy) to estimate how good each state is. GRPO removes it:
for each prompt, sample a GROUP of G answers, score each with a reward FUNCTION (a verifier: unit tests,
exact-match maths, JSON schema, regex...), and use the group-normalised reward as the advantage:

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)        ("scale_rewards='group'")
    loss = - mean_i [ min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i) ]  + beta * KL(pi || pi_ref)
           rho_i = pi(token)/pi_old(token)     (the PPO clipped surrogate, applied per token)

Answers better than their siblings are pushed up, worse ones down. No reward model, no critic.
In TRL 1.13 the DEFAULTS are beta=0.0 (no KL term, no reference model kept) and loss_type='dapo'
(token-level normalisation from the DAPO paper). Older tutorials assume beta=0.04: check your version.

Toy task: make the tiny model answer "What is the maximum pressure?" with a completion containing "6 bar".
Stage 0 shows the classic failure (reward hacking from a random start); stages 1-2 show the real recipe.
"""
import logging, sys, tempfile, warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)

prompts = [{"prompt": [{"role": "user", "content": q}]} for q in
           ["What is the maximum pressure?", "Maximum pressure?", "How high may the pressure go?",
            "Tell me the pressure limit."]] * 8
ds = Dataset.from_list(prompts)


def text_of(c):
    return c[0]["content"] if isinstance(c, list) else str(c)


def reward_hackable(completions, **kwargs):
    """A BAD reward: partial credit for pieces. The policy can farm it without answering."""
    return [1.0 if "6 bar" in text_of(c) else 0.3 * ("6" in text_of(c)) + 0.3 * ("bar" in text_of(c))
            for c in completions]


def reward_strict(completions, **kwargs):
    """A VERIFIER: 1 for the fact, 0 otherwise. (An earlier version also paid -0.5 for repetition: the policy
    learned that an EMPTY answer (0) beats a wrong one (-0.5) and collapsed to silence. Penalties need care.)"""
    return [1.0 if "6 bar" in text_of(c) else 0.0 for c in completions]


def reward_short_ungated(completions, **kwargs):
    """A shaping reward that is NOT tied to correctness: short junk scores. (Measured: it gets farmed.)"""
    return [max(0.0, 1.0 - len(text_of(c)) / 60) * 0.2 for c in completions]


def reward_short_gated(completions, **kwargs):
    """The fix: the length bonus is paid ONLY on correct answers, so it can only refine, never replace."""
    return [max(0.0, 1.0 - len(text_of(c)) / 60) * 0.2 if "6 bar" in text_of(c) else 0.0 for c in completions]


@torch.no_grad()
def sampled_success(model, n=64):
    """Fraction of SAMPLED answers (temperature 1) that contain the fact: what RL actually moves."""
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the maximum pressure?"}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**{k: v.repeat(n, 1) for k, v in ids.items()}, max_new_tokens=12, do_sample=True,
                         temperature=1.0, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return sum("6 bar" in t for t in texts) / n, texts[0]


def grpo(model_path, rewards, epochs, lr=2e-3):
    """Full-parameter GRPO on the toy model, loaded FROM DISK (see the gotcha in READING)."""
    args = GRPOConfig(output_dir=str(tmp / "grpo"), per_device_train_batch_size=8, num_generations=8,
                      max_completion_length=12, num_train_epochs=epochs, learning_rate=lr, logging_steps=4,
                      temperature=1.0, beta=0.0, save_strategy="no", report_to=[], use_cpu=True, bf16=False,
                      disable_tqdm=True, seed=0)
    tr = GRPOTrainer(model=AutoModelForCausalLM.from_pretrained(model_path), reward_funcs=rewards, args=args,
                     train_dataset=ds, processing_class=tok)
    tr.train()
    hist = [h for h in tr.state.log_history if "reward" in h]
    print(f"{'step':>6}{'reward':>10}{'reward_std':>12}{'frac_zero_std':>15}{'grad_norm':>11}")
    for h in hist[:: max(1, len(hist) // 6)]:
        print(f"{h['step']:>6}{h['reward']:>10.3f}{h.get('reward_std', 0):>12.3f}"
              f"{h.get('frac_reward_zero_std', 0):>15.3f}{float(h.get('grad_norm', 0)):>11.3f}")
    return tr.model.eval()


print("STAGE 0 - GRPO from a RANDOM model with a HACKABLE reward")
m0 = grpo(model_dir, [reward_hackable], 1)
rate, sample = sampled_success(m0)
print(f"  sampled success {rate:.2f}; a sample: {sample!r}   <- reward hacking: it farms the partial credit\n")

print("STAGE 1 - short SFT warm start (the R1 'cold start'): teach the format, not perfection")
from trl import SFTConfig, SFTTrainer  # noqa: E402
sft_rows = [{"messages": [{"role": "user", "content": p["prompt"][0]["content"]},
                          {"role": "assistant", "content": "The maximum pressure is 6 bar."}]} for p in prompts]
sft = SFTTrainer(model=AutoModelForCausalLM.from_pretrained(model_dir), train_dataset=Dataset.from_list(sft_rows),
                 processing_class=tok,
                 args=SFTConfig(output_dir=str(tmp / "sft"), max_steps=40, per_device_train_batch_size=8,
                                learning_rate=3e-3, save_strategy="no", report_to=[], use_cpu=True, bf16=False,
                                disable_tqdm=True, logging_steps=1000))   # full FT: the toy model is random
sft.train()
sft.save_model(str(tmp / "sft_model"))          # save -> reload: NEVER pass the trainer's model object on
m1 = AutoModelForCausalLM.from_pretrained(tmp / "sft_model").eval()
before, s1 = sampled_success(m1)
print(f"  after SFT warm start: sampled success {before:.2f}; a sample: {s1!r}\n")

print("STAGE 2a - GRPO on the SFT model: strict verifier + an UNGATED length reward")
m2a = grpo(tmp / "sft_model", [reward_strict, reward_short_ungated], 1)
after_a, s2a = sampled_success(m2a)
print(f"  sampled success {before:.2f} -> {after_a:.2f}; a sample: {s2a!r}"
      + ("   <- hacked the LENGTH reward: short junk beats long correct" if after_a < before else "") + "\n")

print("STAGE 2b - same, but the length reward is GATED on correctness")
m2b = grpo(tmp / "sft_model", [reward_strict, reward_short_gated], 1)
after_b, s2b = sampled_success(m2b)
print(f"  sampled success {before:.2f} -> {after_b:.2f}; a sample: {s2b!r}")
if s2b.count("6 bar") > 1:
    print("  ^ the verifier is satisfied, but the answer SPAMS the fact: the verifier only checked '6 bar' is present."
          "\n    Your reward function IS your task definition. Add an exact-format check (e.g. a regex for one clean"
          "\n    sentence) and re-run; never ship on reward alone - read samples and run the held-out eval.")
print("""
READING
- 'reward' (sum of all reward functions) should rise; reward_std inside a group must be > 0 or there is no
  signal: if all G answers get the same reward, the advantage is 0 ('frac_reward_zero_std' -> 1.0 = stuck).
- GRPO needs a model that SOMETIMES already gets it right: SFT first, then GRPO. From a random model it
  can only work on tasks this tiny. That is why R1 used a cold-start SFT stage before RL.
- Reward hacking is the main failure: the model finds the cheapest way to score. Stage 0 farmed partial
  credit; stage 2a farmed the length bonus. Gate shaping rewards on correctness (2b). Negative penalties
  can teach 'say nothing' (an empty answer scores 0, a wrong one scores less): watch mean length.
  Use several rewards (correctness + format + length), hold out a test set, and READ samples.
- GOTCHA (measured while writing this lab): passing the model OBJECT returned by SFTTrainer straight into
  GRPOTrainer in the same process gave grad_norm = 0.0 and no learning. Saving it and reloading it with
  from_pretrained fixed it. Real pipelines save between stages anyway: SFT -> save -> GRPO from the path.
- On real models use LoRA (peft_config=LoraConfig(...)) and lr ~1e-6 (full) or ~1e-5..5e-5 (LoRA); the
  toy uses full-parameter updates and a huge lr only because it is a random 2-layer model.
- Use GRPO when the answer is VERIFIABLE (maths, code with tests, extraction to a schema, tool calls).
  Use DPO/ORPO when quality is a human preference with no checker.""")
