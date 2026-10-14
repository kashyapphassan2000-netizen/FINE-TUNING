# 98. Do you need a frontend? And what can be hosted for free?

## 98.1 Frontend: optional

The **API is the product**. Developers use it with the OpenAI SDK (chapter 96). A frontend is only for people who want to chat in a browser.

@file serving/ui/index.html

A single static HTML page: the user types the gateway URL, their own key and the model name; it calls `/v1/chat/completions`. Host it free on any static host (for example GitHub Pages) and set `ALLOWED_ORIGINS` on the gateway to that page's origin.

:::warn Never put a secret key in a public web page
Anything in a public page is readable by anyone. The page above asks each user for their own key. If you want a public demo without keys, put a server-side proxy in front with its own strict rate limit.
:::

## 98.2 Free and nearly-free options, honestly (reported, checked 21 September 2026)

|Option|What it gives|Good for|Not for|
|---|---|---|---|
|your laptop + `serve_local`|a real OpenAI-compatible API, CPU|you alone, small models, demos|anyone else, speed|
|Hugging Face Spaces, free CPU (reported 2 vCPU, 16 GB RAM, sleeps when idle)|a public demo URL|tiny models, UI demos|real traffic|
|Hugging Face ZeroGPU (free accounts: a few minutes of GPU per day, Gradio only; exact quota and GPU model differ between sources)|short GPU bursts inside a Gradio app|a showcase demo|an API others build on|
|Kaggle (~30 GPU-hours/week reported) and Colab (varies)|free notebooks|experiments, small fine-tunes|serving the public (sessions end; terms prohibit it as a service)|
|serverless GPU sign-up credits|real GPUs for days|prototypes|production|

Bottom line: **free is for you and for demos.** A model that many users call through an API needs a paid GPU: the cheapest honest path is one small on-demand GPU running vLLM + the gateway, stopped when idle, then autoscaling once there is steady traffic.

## 98.3 The single-user path, for free

```laptop
python -m peftkit.serve_local --model outputs/<run>/merged --name mymodel --api-key local-secret --port 8000
# then any OpenAI client on the same machine:
#   OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-secret")
```
