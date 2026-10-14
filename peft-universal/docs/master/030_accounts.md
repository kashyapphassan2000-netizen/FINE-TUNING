# 30. Accounts, licences and keys

## 30.1 Hugging Face

```browser
1. https://huggingface.co/join → create account → verify e-mail.
2. Settings → Access Tokens → Create new token.
   - Name: peft-train. Type: Fine-grained (or Write).
   - Permissions: read access to public gated repos you accepted; write access to your own repos.
3. Copy the token once (starts with hf_). Store it in a password manager.
4. Gated models (e.g. meta-llama/Llama-3.1-8B-Instruct): open the model page → read the licence →
   fill the form → wait for "You have been granted access".
```

## 30.2 Hyperstack

```browser
1. https://console.hyperstack.cloud → sign up → verify.
2. Billing → activate billing and add credit (VMs cannot be deployed without it).
3. API Keys → generate a key (only needed for the autoscaler / scripted deploys).
4. Environments → Create environment (it belongs to one region; pick one where your GPU flavor is in stock).
5. SSH keys → Import SSH key → paste the PUBLIC key from chapter 55.1 → choose the environment.
```

## 30.3 LLM API provider (Euron or any OpenAI-compatible)

```browser
1. Create an API key in your provider's dashboard.
2. Find its OpenAI-compatible base URL in its docs (it usually ends with /v1).
3. Put both in .env as OPENAI_BASE_URL and OPENAI_API_KEY. Put the model id in SYNTH_MODEL.
```

:::warn
This guide does not hard-code any provider URL. Confirm the exact base URL and model ids in your provider's documentation.
:::

## 30.4 GitHub, LangSmith, (optional) W&B, Google Drive

```browser
GitHub:    Settings → SSH and GPG keys → add your laptop's public key (or use HTTPS + a token).
LangSmith: Settings → API Keys → create → LANGSMITH_API_KEY in .env
           (useful for tracing the synthetic-data API calls if you wrap them; the trainer does not report to LangSmith).
W&B:       optional training dashboards → WANDB_API_KEY, then train.report_to: wandb
Drive:     nothing to create; rclone does OAuth in chapter 65.
```

:::check
- HF token with write access saved; licence accepted for gated models.
- Hyperstack billing active, environment created, SSH key imported.
- LLM provider base URL, key and model id written down.
:::
