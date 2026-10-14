# 66. Serving with vLLM (a new GPU session)

:::why
vLLM keeps the KV cache in fixed-size pages (PagedAttention) and batches requests continuously, so one GPU serves many users. It exposes an OpenAI-compatible API, so any OpenAI client works.
:::

## 66.1 New session

Deploy a new VM (chapter 55; size it with the SERVING line of the estimator), run preflight, and install only the serving env:

```gpu
git clone git@github.com:<you>/peft-universal.git && cd peft-universal
bash scripts/gpu_preflight.sh
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-serve --python 3.11 && uv pip install --python .venv-serve/bin/python -r requirements-serve.txt
cp .env.example .env && nano .env      # HF_TOKEN, VLLM_API_KEY (long random string)
```

## 66.2 Start the server

@file scripts/serve_vllm.sh

```gpu Mode A: merged model (any method)
MODEL=<you>/llama31-8b-pump SERVED_NAME=pump MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
```

```gpu Mode B: one base + several LoRA adapters (lora/qlora/loraplus only)
MODEL=meta-llama/Llama-3.1-8B-Instruct SERVED_NAME=base \
ADAPTERS="pump=<you>/llama31-8b-pump-lora legal=<you>/llama31-8b-legal-lora" MAX_LORA_RANK=16 \
bash scripts/serve_vllm.sh
```

Every flag in the script was checked in the vLLM v0.29.0 source: `--served-model-name`, `--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, `--api-key`, `--enable-lora`, `--max-loras`, `--max-lora-rank` (allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512), `--lora-modules name=path`. Use `--chat-template file.jinja` if you trained with a custom template.

:::warn
- vLLM's own help text warns that `--api-key` alone does not secure every endpoint. Do not expose port 8000 to the internet; put the gateway (chapter 70) in front and firewall 8000 to the gateway's IP.
- `TP` defaults to the number of GPUs. Tensor parallel size must divide the model's attention heads.
- If you see out-of-memory at start-up, lower `MAX_MODEL_LEN` or `GPU_UTIL`.
:::

## 66.3 Call it

```gpu Smoke test on the VM
curl -s localhost:8000/health && echo healthy
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"pump","messages":[{"role":"user","content":"Max pressure?"}],"max_tokens":64}'
```

```py Any OpenAI client (through the gateway in production)
from openai import OpenAI
client = OpenAI(base_url="http://<gateway-host>:8080/v1", api_key="<client key>")
r = client.chat.completions.create(model="pump", messages=[{"role": "user", "content": "Max pressure?"}])
print(r.choices[0].message.content)
```

In multi-LoRA mode the `model` field selects the adapter (`pump`, `legal`) or the base (`base`).

:::check
- `/health` returns 200.
- A chat completion returns an answer from my fine-tuned model.
:::
