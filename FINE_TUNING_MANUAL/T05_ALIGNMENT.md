# T05 — Preference Alignment: RLHF-PPO, DPO, GRPO, ORPO (and KTO)

**WHERE:** read anywhere; labs on your PC (each runs in under a minute on CPU).
**GOAL:** know why SFT alone is not enough, exactly what each alignment method optimises, what it costs in memory, and which one to use. Every lab below was run while writing this manual; the outputs are real, **including the failures**, because the failures are the lesson.

---

## 1. Why SFT alone is not enough

SFT shows the model **one** good answer per prompt and says "imitate this". It never shows what **worse** looks like. Consequences:

- **No notion of "better":** two acceptable answers (concise vs rambling, hedged vs confident, safe vs unsafe) are equally "right" to SFT.
- **Exposure to only good behaviour:** the model never learns to avoid specific bad behaviours (sycophancy, unsafe compliance, made-up citations).
- **Imitation ceiling:** SFT can only be as good as the demonstrations; preference/RL methods can push beyond them when the **comparison** is easier to judge than writing the ideal answer (it usually is).
- **Verifiable tasks** (maths, code, schema extraction) have a checker; SFT ignores it, RL can optimise against it directly.

**The order is always:** SFT first → then preference/RL. Alignment on a model that cannot follow instructions amplifies noise.

## 2. RLHF with PPO: reward model, critic, KL penalty (InstructGPT, 2022)

```text
STEP 1  SFT model π_SFT
STEP 2  REWARD MODEL r_φ(x, y): a copy of the LM with a scalar head, trained on human comparisons
        Bradley-Terry loss:  L_RM = − log σ( r_φ(x, y_chosen) − r_φ(x, y_rejected) )
STEP 3  PPO: maximise reward while staying close to π_SFT
        per-token reward  r_t = − β · ( log π(a_t) − log π_ref(a_t) )      ← the KL penalty
                          + r_φ(x, y) on the final token
        critic V(s_t) predicts future reward → advantages A_t by GAE (γ = 1, λ = 0.95)
        policy loss  − min( ρ_t·A_t , clip(ρ_t, 1−ε, 1+ε)·A_t ),   ρ_t = π(a_t)/π_old(a_t),  ε = 0.2
        value loss   ( V(s_t) − return_t )²
FOUR models in memory: policy, reference, reward model, critic  →  ~4x the memory of SFT
```

TRL 1.13 no longer ships a `PPOTrainer` (production PPO for LLMs lives in OpenRLHF, verl, NeMo-RL), so the lab implements the full loop in ~150 lines of PyTorch:

```powershell
python labs/lab26_rlhf_ppo.py
```

**YOU SHOULD SEE (real output):**

```text
0) SFT warm start: sampled success 0.86  sample 'The maximum pressure is 6 bar bar.'
1) reward model trained on 20 pairs: pairwise accuracy 1.00

2) PPO beta=0.0:  iter  RM-score  KL(pi||ref)
                    0    -0.235       0.000
                   12     1.644       0.079
                   24     2.052       5.337
   sampled success 0.86; sample 'The maximum maximum maximum maximum maximum pressure is 6 bar.'

2) PPO beta=0.2:  iter  RM-score  KL(pi||ref)
                    0     0.017       0.000
                   12     0.148       2.093
                   24     2.140       1.407
   sampled success 0.83; sample 'The maximum pressure is 6 is 6 is 6 bar.'
```

**Read it:** without the KL anchor (β = 0) the policy drifted to KL 5.3 and the text degenerated while the reward model's score kept rising. That is **reward over-optimisation**: the RM is a learned proxy, PPO maximises the proxy, not the truth (Gao et al. 2022). β = 0.2 kept KL bounded. Both runs exploit this weak 20-pair RM somewhat. Real RMs are trained on 10k-1M comparisons and **still** get exploited, so the KL term is never optional.

## 3. DPO: direct optimisation, no reward model (Rafailov et al. 2023)

DPO shows that the PPO objective's optimal policy can be written in closed form, so the reward model **cancels out**, and you can optimise the policy directly on preference pairs:

```text
L_DPO = − log σ( β · [ log π_θ(y_w|x)/π_ref(y_w|x)  −  log π_θ(y_l|x)/π_ref(y_l|x) ] )
         y_w = chosen, y_l = rejected, π_ref = frozen SFT model, β ≈ 0.1
implicit reward  r̂(x, y) = β · log π_θ(y|x)/π_ref(y|x)
TWO models in memory (policy + frozen reference); with LoRA the reference is "adapter disabled" → ~1 model
```

```powershell
python labs/lab23_dpo.py
```

**YOU SHOULD SEE (real output):**

```text
  step      loss  reward acc    margin    chosen  rejected
     4    0.6858       0.688    0.0149    0.0004   -0.0145
    16    0.6223       1.000    0.1484    0.0127   -0.1358
    32    0.5776       1.000    0.2502    0.0212   -0.2290
```

| Metric | Meaning | Healthy |
|---|---|---|
| loss | starts at ln 2 = 0.693 (chance) | falls slowly; alone says little |
| `rewards/accuracies` | share of pairs where chosen beats rejected | rises toward 1.0 |
| `rewards/margins` | how far apart | grows steadily |
| `rewards/chosen` / `rejected` | log-ratio vs reference | chosen ≥ 0, rejected negative |

**DPO pitfalls:** length bias (if your judge preferred longer answers, DPO makes the model verbose: compare mean length before/after); pairs with no real quality gap are noise (the toolkit keeps only pairs with a judge margin ≥ 2 and a consistent position-swapped verdict); it can quietly break format compliance, so re-run the SFT eval suites after DPO.

## 4. GRPO: Group Relative Policy Optimization (DeepSeekMath 2024 → DeepSeek-R1)

GRPO removes PPO's **critic**: for each prompt, sample a **group** of G answers, score each with a reward **function** (a verifier), and use the group-normalised reward as the advantage:

```text
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)               (TRL: scale_rewards="group")
loss = − mean_tokens[ min( ρ·A_i , clip(ρ, 1−ε, 1+ε)·A_i ) ]  + β · KL(π ‖ π_ref)
TRL 1.13 DEFAULTS: beta = 0.0 (no KL term, no reference model kept), loss_type = "dapo" (token-level
normalisation from the DAPO paper), num_generations = 8. Older tutorials assume beta = 0.04: check yours.
```

```powershell
python labs/lab25_grpo.py
```

**YOU SHOULD SEE (real output, abridged):**

```text
STAGE 0 - GRPO from a RANDOM model with a HACKABLE reward
  sampled success 0.03; a sample: ' 6f.. claim.. belf. 6 6'   <- reward hacking: it farms the partial credit
STAGE 1 - short SFT warm start (the R1 'cold start'): teach the format, not perfection
  after SFT warm start: sampled success 0.12; a sample: 'an the maximumis. The maximum pressure is 6 bar'
STAGE 2a - GRPO on the SFT model: strict verifier + an UNGATED length reward
  sampled success 0.12 -> 0.00; a sample: '...'   <- hacked the LENGTH reward: short junk beats long correct
STAGE 2b - same, but the length reward is GATED on correctness
  step    reward  reward_std  frac_zero_std  grad_norm
     4     0.071       0.199          0.500      0.000
    12     0.385       0.493          0.000      3.421
    24     1.041       0.208          0.000      1.226
  sampled success 0.12 -> 0.97; a sample: ' 6 bar 6 bar the 6 bar 6\n. 6 bar'
  ^ the verifier is satisfied, but the answer SPAMS the fact: the verifier only checked '6 bar' is present.
```

**Four lessons, all measured here:**

1. **GRPO needs a model that is sometimes right.** From random weights there is no signal (`frac_reward_zero_std` near 1 = every answer in a group scored the same = zero advantage). That is why R1 had a cold-start SFT stage.
2. **Every shaping reward gets farmed.** Partial credit was farmed (stage 0); an ungated length bonus was farmed (2a). Gate shaping rewards on correctness.
3. **The verifier IS the task definition.** 2b reached 97% "success" by spamming the fact. Add a format check (a regex for one clean sentence, a JSON schema, unit tests), hold out a test set, and **read samples**.
4. **A gotcha found while writing this lab:** passing the model object returned by `SFTTrainer` straight into `GRPOTrainer` in the same process gave `grad_norm = 0.0` and no learning. Saving with `save_model()` and reloading with `from_pretrained()` fixed it. Real pipelines save between stages anyway: **SFT → save → GRPO from the path.**

**Use GRPO when the answer is verifiable:** maths with a final number, code with unit tests, extraction validated by a schema, tool calls checked by execution, classification with a label.

## 5. ORPO: single stage, no reference model, no reward model (Hong et al. 2024)

ORPO folds preference learning **into SFT**:

```text
L_ORPO = L_SFT(y_w)  +  λ · L_OR
L_OR   = − log σ( log odds(y_w|x) − log odds(y_l|x) ),     odds(y|x) = P(y|x) / (1 − P(y|x))
P(y|x) = length-normalised sequence likelihood;  λ (TRL: beta) ≈ 0.1 - 0.5
ONE model in memory: no frozen reference (DPO needs one), no reward model, no critic
```

In TRL 1.13 ORPO moved to `trl.experimental.orpo` (`from trl.experimental.orpo import ORPOConfig, ORPOTrainer`).

```powershell
python labs/lab24_orpo.py
```

**YOU SHOULD SEE (real output):**

```text
  step                loss            nll_loss      log_odds_ratio  rewards/accuracies     rewards/margins
    10              5.8420              5.7751             -0.6683              0.7750              0.0052
    40              5.1837              5.1399             -0.4378              1.0000              0.0600
    80              4.9740              4.9383             -0.3570              1.0000              0.0845
```

`nll_loss` is the SFT part (falls as the chosen answers are learned); `log_odds_ratio` rising toward 0 and `rewards/margins` growing are the preference part.

## 6. KTO (bonus): only "good" / "bad" labels, no pairs

KTO (Ethayarajh et al. 2024) uses a prospect-theory loss on **unpaired** examples labelled desirable/undesirable: the thumbs-up/thumbs-down data most products already collect. `trl.KTOTrainer` (present in 1.13). Use it when you have ratings but not side-by-side comparisons.

## 7. Choosing (the table you will actually use)

| | PPO-RLHF | DPO | GRPO | ORPO | KTO |
|---|---|---|---|---|---|
| needs | SFT model + reward model + pairs to train it | SFT model + pairs | SFT model + **reward function** | pairs (SFT not required first) | good/bad labels |
| models in memory | 4 | 2 (≈1 with LoRA) | 1-2 (1 when beta = 0) | 1 | 2 |
| online sampling | yes (slow) | no (offline) | yes | no | no |
| best for | strong RM + infrastructure | human preferences on style/safety/helpfulness | maths, code, extraction, tool use | small budget, pairs from day one | product feedback data |
| main risk | reward hacking, instability, cost | length bias, overfitting pairs | reward hacking of the verifier | weaker separate control of SFT vs preference | label noise |
| TRL 1.13 | not shipped (OpenRLHF/verl) | `DPOTrainer` | `GRPOTrainer` | `trl.experimental.orpo` | `KTOTrainer` |

**Default for a small team:** SFT → DPO for preference/style; SFT → GRPO when a checker exists; ORPO when memory is tight and you have pairs from the start. Practical commands on a real model: P13.

---

## CHECKPOINT

- [ ] I can write the DPO loss and say what β does.
- [ ] I can explain why PPO needs a KL term, using lab 26's numbers.
- [ ] I can explain GRPO's group advantage and why GRPO needs SFT first, using lab 25's stages.
- [ ] I know which alignment method fits my data (pairs? labels? a verifier?).
