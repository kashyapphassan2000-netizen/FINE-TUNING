"""Deterministic OpenAI-compatible fake that routes on the system prompt (for synthetic-data tests)."""
import itertools
import json

from fastapi import FastAPI, Request

app = FastAPI()
counter = itertools.count()


def reply(obj):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(obj) if not isinstance(obj, str) else obj}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    sys_msg = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
    user = body["messages"][-1]["content"]
    t = body.get("temperature", 0.7)
    if "design diverse task instructions" in sys_msg:
        n = next(counter)
        verbs = ["Explain", "Compare", "Summarize", "Troubleshoot", "Draft", "Classify"]
        topics = ["bearing wear", "impeller cavitation", "motor overheating", "seal leakage", "valve chatter",
                  "flow sensor drift", "vibration alarms", "warranty claims"]
        first = f"{verbs[n % 6]} {topics[n % 8]} for a plant operator"
        return reply({"instructions": [first, first + " today",  # second one is a near-duplicate
                                       "Draw an image of a pump",
                                       f"{verbs[(n + 3) % 6]} how {topics[(n + 5) % 8]} affects uptime"]})
    if "Respond to the instruction" in sys_msg:
        return reply({"input": "", "output": f"Answer: {user[:30]}"})
    if "Score the ASSISTANT answer" in sys_msg:
        return reply({"rationale": "r", "score": 5 if "GOOD" in user else 1 if "BAD" in user else 3})
    if "compare two answers" in sys_msg:
        a_part = user.split("[ANSWER A]")[1].split("[ANSWER B]")[0]
        return reply({"rationale": "r", "winner": "A" if "GOOD" in a_part else "B"})
    text = "GOOD precise answer" if t <= 0.5 else ("BAD rambling answer" if t >= 1.0 else "meh answer")
    return reply(f"{text} (t={t})")
