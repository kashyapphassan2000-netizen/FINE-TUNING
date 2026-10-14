# 96. An OpenAI-style public API: keys, limits, access, metering

:::why In one line
Users call ONE URL with THEIR key, exactly like the OpenAI API. The gateway authenticates the key, applies that key's rate limit and model permissions, forwards to vLLM, and meters tokens per key.
:::

## 96.1 What is implemented and tested

|Capability|How|
|---|---|
|OpenAI-compatible routes|`/v1/chat/completions`, `/v1/completions`, `/v1/models` (any OpenAI SDK works by changing `base_url`)|
|self-serve keys|`POST /admin/keys` issues `sk-...` once; only its SHA-256 hash is stored (a test checks the key file contains no plaintext)|
|revocation|`DELETE /admin/keys/{key_id}`, persisted across restarts (tested)|
|per-key rate limit|`rpm` per key, token bucket (tested: 429 when exceeded)|
|per-key model access|`models` per key → 403 for others (tested)|
|metering|requests, prompt and completion tokens per key at `/admin/usage` (tested) — the basis of billing|
|browser access|`ALLOWED_ORIGINS` enables CORS for a web front end|
|failover|unhealthy upstreams skipped (chapter 56)|

## 96.2 Commands

```gpu issue a key for a customer, on the running gateway
python -m peftkit.keys remote-create --gateway https://api.yourdomain.com --admin-key $ADMIN --name acme --rpm 60 --models mymodel
```

```any or offline, before deploying (the gateway reads GATEWAY_KEYS_FILE)
python -m peftkit.keys create --file serving/keys.json --name acme --rpm 60 --models mymodel
python -m peftkit.keys list --file serving/keys.json
python -m peftkit.keys revoke --file serving/keys.json --key-id <id>
```

```py what your customer runs (the official OpenAI Python SDK, only base_url changes)
from openai import OpenAI
client = OpenAI(base_url="https://api.yourdomain.com/v1", api_key="sk-...")
r = client.chat.completions.create(model="mymodel", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

## 96.3 Proven end to end

The rehearsal (chapter 2, step 12) issues a key to a new customer through the admin API, calls the model with it, and reads the metered usage: the customer got **"The maximum pressure is 6 bar."** and usage showed 1 request, 21 prompt tokens, 8 completion tokens.

:::warn What a real public launch still needs (not built here)
TLS certificates and a domain; a sign-up page with abuse controls (email verification, CAPTCHA); billing (Stripe or similar) reading `/admin/usage`; terms of service and an acceptable-use policy; logging with retention rules; Redis-backed rate limits when you run more than one gateway replica. The gateway gives you the mechanics; these are product and legal work.
:::
