"""distilabel 1.5.3 reference pipeline: Self-Instruct -> answers -> UltraFeedback rating -> DPO format.
Class/argument names were checked against the distilabel 1.5.3 wheel source. NOT executed in this guide
(needs a live LLM endpoint). Install into its own env:  uv pip install "distilabel[openai]==1.5.3"
"""
import os

from distilabel.models.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import SelfInstruct, TextGeneration, UltraFeedback
from distilabel.steps.formatting.dpo import FormatTextGenerationDPO


def llm(model: str) -> OpenAILLM:
    return OpenAILLM(model=model, base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])


with Pipeline(name="domain-sft-dpo") as pipeline:
    seeds = LoadDataFromDicts(data=[{"input": "Pump maintenance for industrial water systems"},
                                    {"input": "Warranty and returns policy for pumps"}])
    instruct = SelfInstruct(llm=llm(os.environ["SYNTH_MODEL"]), num_instructions=5,
                            application_description="A support assistant for industrial pump customers")
    # SelfInstruct outputs a list column `instructions`; ExpandColumns turns it into rows named `instruction`
    from distilabel.steps import ExpandColumns

    expand = ExpandColumns(columns={"instructions": "instruction"})
    answer_a = TextGeneration(name="gen_strong", llm=llm(os.environ["SYNTH_MODEL"]))
    answer_b = TextGeneration(name="gen_weak", llm=llm(os.environ.get("WEAK_MODEL", os.environ["SYNTH_MODEL"])))
    group = GroupColumns(columns=["generation", "model_name"], output_columns=["generations", "model_names"])
    rate = UltraFeedback(llm=llm(os.environ.get("JUDGE_MODEL", os.environ["SYNTH_MODEL"])), aspect="overall-rating")
    to_dpo = FormatTextGenerationDPO()
    seeds >> instruct >> expand >> [answer_a, answer_b] >> group >> rate >> to_dpo

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.save_to_disk("data/interim/distilabel_out")
