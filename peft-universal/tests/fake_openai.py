"""Minimal OpenAI-compatible chat server for tests. First request returns 429 to exercise retries."""
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
calls = {"n": 0}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    calls["n"] += 1
    if calls["n"] == 1:
        return JSONResponse({"error": "slow down"}, status_code=429)
    if req.headers.get("authorization") != "Bearer sk-test":
        return JSONResponse({"error": "auth"}, status_code=401)
    body = await req.json()
    passage = body["messages"][-1]["content"]
    pairs = {"pairs": [{"question": f"What does this say ({i})? {passage[-12:]}", "answer": f"It says {i}."}
                       for i in range(2)]}
    content = "```json\n" + json.dumps(pairs) + "\n```"  # models often wrap JSON in fences
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}
