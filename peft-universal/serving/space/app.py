"""Hugging Face Space: chat with your fine-tuned model.

Configured ONLY through Space settings (Settings -> Variables and secrets), never by editing this file:
  MODEL_ID          Hub id of your merged model (local) or the served model name (endpoint)
  BACKEND           local | endpoint
  SYSTEM_PROMPT     optional: the system prompt you TRAINED with (parity: same prompt -> same behaviour)
  ENDPOINT_URL      endpoint backend: https://<your-api>/v1  (vLLM or the peft-universal gateway)
  ENDPOINT_API_KEY  endpoint backend: SECRET (a key created with `python -m peftkit.keys`)
  HF_TOKEN          SECRET, only if MODEL_ID is a private repo
"""
import os

import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")
BACKEND = os.environ.get("BACKEND", "local")
SYSTEM = os.environ.get("SYSTEM_PROMPT") or None
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def _messages(message, history):
    msgs = [{"role": "system", "content": SYSTEM}] if SYSTEM else []
    for turn in history or []:                       # gradio passes OpenAI-style {"role", "content"} dicts
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            content = turn["content"]
            msgs.append({"role": turn["role"], "content": content if isinstance(content, str) else str(content)})
    msgs.append({"role": "user", "content": message})
    return msgs


if BACKEND == "endpoint":
    from openai import OpenAI

    client = OpenAI(base_url=os.environ["ENDPOINT_URL"], api_key=os.environ["ENDPOINT_API_KEY"])

    def respond(message, history):
        stream = client.chat.completions.create(model=MODEL_ID, messages=_messages(message, history),
                                                max_tokens=MAX_NEW, temperature=0.2, stream=True)
        text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                yield text
else:
    import threading

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=os.environ.get("HF_TOKEN"),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None).eval()

    def respond(message, history):
        enc = tok.apply_chat_template(_messages(message, history), add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        threading.Thread(target=model.generate, kwargs=dict(**enc, streamer=streamer, max_new_tokens=MAX_NEW,
                                                            do_sample=False)).start()
        text = ""
        for piece in streamer:
            text += piece
            yield text


demo = gr.ChatInterface(
    respond,
    title=os.environ.get("SPACE_TITLE", f"Chat with {MODEL_ID}"),
    description=("Fine-tuned with peft-universal. Answers can be wrong; do not enter personal or confidential data."),
    examples=[e for e in os.environ.get("EXAMPLES", "").split("|") if e] or None,
)

if __name__ == "__main__":
    demo.launch()
