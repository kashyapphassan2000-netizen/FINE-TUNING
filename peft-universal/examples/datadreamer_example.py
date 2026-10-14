"""DataDreamer 0.46.0 reference (class names checked in the wheel; NOT executed here).
DataDreamer pins datasets<4, huggingface-hub<1.0, numpy<2 -> it CANNOT share the training env.
    uv venv .venv-dd --python 3.11 && uv pip install --python .venv-dd/bin/python datadreamer.dev==0.46.0
Check the DataDreamer docs for exact step arguments before running."""
import os

from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import DataFromPrompt

with DataDreamer("./dd_output"):
    llm = OpenAI(model_name=os.environ["SYNTH_MODEL"], api_key=os.environ["OPENAI_API_KEY"],
                 base_url=os.environ["OPENAI_BASE_URL"])
    questions = DataFromPrompt(
        "Generate pump-support questions",
        args={"llm": llm, "n": 100, "temperature": 1.0,
              "instruction": "Write one realistic customer question about industrial pump maintenance."},
        outputs={"generations": "questions"},
    )
